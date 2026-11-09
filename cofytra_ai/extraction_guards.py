"""
Deterministic guards for F3 knowledge extraction.

Every model-proposed candidate either becomes a strict ``KnowledgeCandidate`` or is dropped and
counted under one of the three frozen ``ExtractionDropCounts`` reasons:

  unknown_source   no source IDs, or an ID that is not a founder-authored message in this request
                   (assistant messages and unknown IDs included)
  quote_not_found  quote missing, too short, or not verbatim in a cited message; external reference
                   not present in a cited message; statement contains numbers or payment claims that
                   the cited text does not contain
  invalid_status   SENSITIVE CONTENT in statement/quote/reference (credentials, email, phone, payment
                   card, IBAN/account data, confidentiality markers; checked first; temporary mapping
                   until a dedicated counter exists, D13);
                   category/status combination not allowed; external_source without a specific,
                   named reference; a reference on a non-external status; any other field-contract
                   failure (empty/oversized statement, more than 5 sources); over the 20-candidate limit

Corrections that never discard information (counted in ``GuardOutcome.corrections``, not dropped):

  status_downgraded       quote contains first-person hedging ("I think", "we believe", "probably")
                          but was labelled an observation -> founder_hypothesis (evidence -> assumption)
  duplicate_cleared       model said "duplicate" but the evidence is insufficient (different category,
                          status, numbers or negation, or low wording overlap) -> shared as a new item
  duplicate_detected      exact restatement of an existing item that the model did not flag
  batch_duplicate_merged  the same claim proposed twice in one response -> one candidate

Conflicting claims are never merged: a claim is a duplicate only if it says the same thing with the
same category, status and numbers.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

from pydantic import ValidationError

from cofytra_ai.models import (
    ALLOWED_STATUSES,
    MAX_CANDIDATES,
    MAX_SOURCES_PER_CANDIDATE,
    EpistemicStatus,
    ExtractionDropCounts,
    ExtractRequest,
    KnowledgeCandidate,
    KnowledgeCategory,
    MessageRole,
    SharedKnowledgeItem,
    normalize_text,
)

MIN_QUOTE_WORDS = 3
DUPLICATE_CONTAINMENT = 0.8

_STOPWORDS = frozenset(
    "the and for are but you your our with this that have has had was were will would should could "
    "what when where which who how why can just about into from they them their there then than also "
    "very some any all its a an of to in on at by is it be as or so if we i".split()
)
_NEGATIONS = frozenset({"not", "no", "never", "don't", "doesn't", "didn't", "won't", "isn't", "aren't", "cannot", "can't", "none"})
_HEDGE = re.compile(
    r"\b(i|we)\s+(really\s+)?(think|believe|guess|assume|suspect|bet|expect|hope|feel|reckon)\b"
    r"|\bi'?m\s+(pretty\s+|fairly\s+)?sure\b|\bprobably\b|\bmaybe\b|\bmy\s+(guess|hunch|gut)\b",
    re.IGNORECASE,
)
_PAYMENT = re.compile(
    r"\b(pay|pays|paid|paying|payment|payments|price|prices|pricing|priced|purchas\w*|buy|buys|buying|bought|"
    r"subscri\w*|revenue|pre-?orders?|loi|lois|eur|euros?|usd|dollars?|gbp)\b|[$€£]",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_GENERIC_REFERENCE_WORDS = frozenset(
    "a an the some this that our my report reports article articles study studies survey surveys research data "
    "source sources industry market online internet website site web paper papers according to blog post news "
    "statistic statistics stats analysis analyst analysts expert experts google".split()
)


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def content_tokens(text: str) -> set[str]:
    out = set()
    for tok in re.findall(r"[0-9a-zÀ-ɏ']+", text.casefold()):
        tok = tok.strip("'")
        if not tok or tok in _STOPWORDS:
            continue
        out.add(tok[:-1] if len(tok) > 4 and tok.endswith("s") and not tok.endswith("ss") else tok)
    return out


def numbers_in(text: str) -> set[str]:
    return {n.replace(",", "") for n in _NUMBER.findall(text)}


def has_negation(text: str) -> bool:
    words = set(re.findall(r"[a-z']+", text.casefold()))
    return bool(words & _NEGATIONS) or "n't" in text.casefold()


def is_hedged(text: str) -> bool:
    return bool(_HEDGE.search(text))


def mentions_payment(text: str) -> bool:
    return bool(_PAYMENT.search(text))


def is_specific_reference(reference: str) -> bool:
    """A named source ('EU Diagnostics Industry Report 2025'), not a generic one ('an industry report')."""
    words = re.findall(r"[0-9a-zÀ-ɏ]+", reference.casefold())
    return any(w not in _GENERIC_REFERENCE_WORDS for w in words) or "http" in reference.casefold()


# ---------------------------------------------------------------------------
# Sensitive content (credentials, contact details, payment/bank data, confidentiality markers)
# Conservative patterns: each needs a strong shape (prefix, separator, checksum or explicit keyword).
# ---------------------------------------------------------------------------

_SECRET_PATTERNS = [re.compile(p) for p in (
    r"\bsk-(?:proj-|live-|test-|ant-)?[A-Za-z0-9_\-]{16,}",          # OpenAI / Anthropic-style keys
    r"\b(?:sk|pk|rk)_(?:live|test)_[A-Za-z0-9]{10,}",                # Stripe
    r"\bAKIA[0-9A-Z]{16}\b",                                          # AWS access key id
    r"\bgh[pousr]_[A-Za-z0-9]{30,}\b|\bgithub_pat_[A-Za-z0-9_]{20,}",  # GitHub
    r"\bxox[abprs]-[A-Za-z0-9-]{10,}",                                # Slack
    r"\bAIza[0-9A-Za-z_\-]{30,}",                                     # Google API key
    r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}",  # JWT
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
)] + [re.compile(p, re.IGNORECASE) for p in (
    r"\bbearer\s+[A-Za-z0-9._\-]{16,}",
    r"\b(?:password|passwd|pwd|passcode|pin\s*code)\s*(?:is|was|=|:)\s*['\"]?[^\s'\"]{4,}",
    r"\b(?:api[_\s-]?key|secret[_\s-]?key|client[_\s-]?secret|access[_\s-]?token|auth[_\s-]?token|refresh[_\s-]?token|token|secret)"
    r"\s*(?:=|:)\s*['\"]?[A-Za-z0-9_\-./+]{8,}",
    r"://[^/\s:@]+:[^/\s@]+@",                                        # credentials inside a URL
)]
_EMAIL = re.compile(r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b")
_PHONE_CANDIDATES = [re.compile(p) for p in (
    r"(?<![\w+])\+\d{1,3}(?:[\s.\-/]?\(?\d{1,4}\)?){2,5}",         # international: +49 151 2345678
    r"(?<!\w)\(\d{3}\)\s?\d{3}[\s.\-]\d{4}\b",                      # (555) 123-4567
    r"(?<![\w.\-])\d{3}[.\-]\d{3}[.\-]\d{4}\b",                     # 555-123-4567 (dash/dot only)
    r"(?<![\w.,])0\d{2,4}[\s\-/]\d{3,4}[\s\-]?\d{3,5}\b",           # national: 0151 2345678, 030-1234567
)]
_CARD_CANDIDATE = re.compile(r"(?<![\d.,])\d(?:[ \-]?\d){12,18}(?![\d.,])")
_IBAN_CANDIDATE = re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){2,7}(?:[ ]?[A-Z0-9]{1,3})?\b")
_ACCOUNT = re.compile(
    r"\b(?:(?:bank\s+)?account|acct|routing)\s*(?:number|no\.?|num|#)\s*(?:is|=|:)?\s*\d[\d\s\-]{5,}\d"
    r"|\bsort\s*code\s*(?:is|=|:)?\s*\d{2}[\s\-]?\d{2}[\s\-]?\d{2}\b"
    r"|\b(?:swift|bic)\s*(?:code)?\s*(?:is|=|:)\s*[A-Z]{6}[A-Z0-9]{2}(?:[A-Z0-9]{3})?\b",
    re.IGNORECASE,
)
_CONFIDENTIAL = re.compile(
    r"\b(?:off[\s-]the[\s-]record|do\s+not\s+share|don'?t\s+share|not\s+for\s+sharing|do\s+not\s+tell|don'?t\s+tell"
    r"|keep\s+(?:this|it)\s+(?:between\s+us|private|secret|confidential)|between\s+(?:you\s+and\s+me|us\s+only)"
    r"|(?:strictly|highly|completely|totally)\s+confidential|(?:this|that|it)\s+is\s+confidential|it'?s\s+confidential"
    r"|under\s+(?:an\s+)?nda|private\s+and\s+confidential)\b"
    r"|^\s*confidential\b|\bconfidential\s*:",
    re.IGNORECASE,
)


def _digits(text: str) -> str:
    return re.sub(r"\D", "", text)


def _luhn_ok(number: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(number)):
        d = int(ch)
        if i % 2 == 1:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def _iban_ok(candidate: str) -> bool:
    iban = candidate.replace(" ", "")
    if not 15 <= len(iban) <= 34:
        return False
    rearranged = iban[4:] + iban[:4]
    return int("".join(str(int(c, 36)) for c in rearranged)) % 97 == 1


def sensitive_kinds(text: str) -> set[str]:
    """Kinds of sensitive content found in ``text`` (e.g. {"email"}); never the values themselves."""
    if not text:
        return set()
    kinds: set[str] = set()
    if any(p.search(text) for p in _SECRET_PATTERNS):
        kinds.add("credential")
    if _EMAIL.search(text):
        kinds.add("email")
    for pattern in _PHONE_CANDIDATES:
        if any(8 <= len(_digits(m.group())) <= 15 for m in pattern.finditer(text)):
            kinds.add("phone")
            break
    if any(_luhn_ok(_digits(m.group())) for m in _CARD_CANDIDATE.finditer(text) if 13 <= len(_digits(m.group())) <= 19):
        kinds.add("payment_card")
    if any(_iban_ok(m.group()) for m in _IBAN_CANDIDATE.finditer(text.upper())) or _ACCOUNT.search(text):
        kinds.add("bank_account")
    if _CONFIDENTIAL.search(text):
        kinds.add("confidential")
    return kinds


def same_claim(candidate_statement: str, candidate_category: KnowledgeCategory, candidate_status: EpistemicStatus,
               existing: SharedKnowledgeItem) -> bool:
    """Sufficient evidence that a candidate restates an existing item (never true for conflicting claims)."""
    if existing.category != candidate_category or existing.epistemic_status != candidate_status:
        return False
    if numbers_in(candidate_statement) != numbers_in(existing.statement):
        return False
    if has_negation(candidate_statement) != has_negation(existing.statement):
        return False
    a, b = content_tokens(candidate_statement), content_tokens(existing.statement)
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    if not shorter:
        return False
    return len(shorter & longer) / len(shorter) >= DUPLICATE_CONTAINMENT


# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------

@dataclass
class GuardOutcome:
    candidates: list[KnowledgeCandidate]
    dropped: ExtractionDropCounts
    corrections: Counter = field(default_factory=Counter)


def _get(raw: Any, name: str, default: Any = None) -> Any:
    return raw.get(name, default) if isinstance(raw, dict) else getattr(raw, name, default)


def guard_candidates(request: ExtractRequest, raw_candidates: list[Any]) -> GuardOutcome:
    founder_msgs = {
        m.id: m for m in request.messages
        if m.role == MessageRole.FOUNDER and m.author_id == request.founder.id
    }
    existing = {k.id: k for k in request.existing_knowledge}
    drops: Counter = Counter()
    fixes: Counter = Counter()
    kept: list[KnowledgeCandidate] = []
    seen: dict[tuple[str, str], int] = {}

    for raw in raw_candidates:
        statement = str(_get(raw, "statement") or "").strip()
        quote = str(_get(raw, "source_quote") or "").strip()
        reference = (_get(raw, "external_reference") or None)
        reference = reference.strip() if isinstance(reference, str) and reference.strip() else None

        # 0. Sensitive content: never share it. Checked first; counted under invalid_status (temporary
        #    mapping, see D13). Only the kinds are recorded, never the values. Quotes are never edited.
        kinds = sensitive_kinds(statement) | sensitive_kinds(quote) | sensitive_kinds(reference or "")
        if kinds:
            drops["invalid_status"] += 1
            fixes["sensitive_content"] += 1
            for kind in kinds:
                fixes[f"sensitive:{kind}"] += 1
            continue

        # 1. Sources: founder-authored messages of this request only.
        source_ids = list(dict.fromkeys(_get(raw, "source_message_ids") or []))
        if not source_ids or any(i not in founder_msgs for i in source_ids):
            drops["unknown_source"] += 1
            continue
        cited_texts = [normalize_text(founder_msgs[i].text) for i in source_ids]
        cited_raw = " ".join(founder_msgs[i].text for i in source_ids)

        # 2. Quote and support: verbatim, non-trivial, and the statement adds no unsupported facts.
        if len(quote.split()) < MIN_QUOTE_WORDS or not any(normalize_text(quote) in t for t in cited_texts):
            drops["quote_not_found"] += 1
            continue
        if reference is not None and not any(normalize_text(reference) in t for t in cited_texts):
            drops["quote_not_found"] += 1
            continue
        if not numbers_in(statement) <= numbers_in(cited_raw):
            drops["quote_not_found"] += 1
            continue
        if mentions_payment(statement) and not mentions_payment(quote):
            drops["quote_not_found"] += 1
            continue

        # 3. Category / epistemic status.
        try:
            category = KnowledgeCategory(_get(raw, "category"))
            status = EpistemicStatus(_get(raw, "epistemic_status"))
        except ValueError:
            drops["invalid_status"] += 1
            continue
        if status == EpistemicStatus.AI_SUGGESTION:
            drops["invalid_status"] += 1
            continue
        if status == EpistemicStatus.EXTERNAL_SOURCE and (reference is None or not is_specific_reference(reference)):
            drops["invalid_status"] += 1
            continue
        if status != EpistemicStatus.EXTERNAL_SOURCE and reference is not None:
            drops["invalid_status"] += 1
            continue
        if status == EpistemicStatus.FOUNDER_OBSERVATION and is_hedged(quote):
            status = EpistemicStatus.FOUNDER_HYPOTHESIS
            if category == KnowledgeCategory.EVIDENCE:
                category = KnowledgeCategory.ASSUMPTION
            fixes["status_downgraded"] += 1
        if status not in ALLOWED_STATUSES[category]:
            drops["invalid_status"] += 1
            continue
        if len(source_ids) > MAX_SOURCES_PER_CANDIDATE:
            drops["invalid_status"] += 1
            continue

        # 4. Duplicates of existing shared knowledge (never for conflicting claims).
        dup_id = _get(raw, "duplicate_of_knowledge_id") or None
        if dup_id is not None and not (dup_id in existing and same_claim(statement, category, status, existing[dup_id])):
            dup_id = None
            fixes["duplicate_cleared"] += 1
        if dup_id is None:
            exact = next((k.id for k in request.existing_knowledge
                          if k.category == category and k.epistemic_status == status
                          and normalize_text(k.statement) == normalize_text(statement)), None)
            if exact is not None:
                dup_id = exact
                fixes["duplicate_detected"] += 1

        # 5. Strict contract model.
        try:
            candidate = KnowledgeCandidate(
                category=category, statement=statement, epistemic_status=status,
                confidence=_get(raw, "confidence"), source_message_ids=source_ids, source_quote=quote,
                external_reference=reference, duplicate_of_knowledge_id=dup_id,
            )
        except ValidationError:
            drops["invalid_status"] += 1
            continue

        # 6. Same claim twice in one response -> one candidate.
        key = (category.value, normalize_text(statement))
        if key in seen:
            fixes["batch_duplicate_merged"] += 1
            continue
        seen[key] = len(kept)
        kept.append(candidate)

    if len(kept) > MAX_CANDIDATES:
        drops["invalid_status"] += len(kept) - MAX_CANDIDATES
        kept = kept[:MAX_CANDIDATES]

    return GuardOutcome(
        candidates=kept,
        dropped=ExtractionDropCounts(
            unknown_source=drops["unknown_source"],
            quote_not_found=drops["quote_not_found"],
            invalid_status=drops["invalid_status"],
        ),
        corrections=fixes,
    )


def founder_message_ids(request: ExtractRequest) -> set[str]:
    return {m.id for m in request.messages if m.role == MessageRole.FOUNDER and m.author_id == request.founder.id}


def describe(outcome: GuardOutcome) -> dict[str, Optional[int]]:
    """Counts only (safe to log): no statements, quotes or IDs."""
    return {**outcome.dropped.model_dump(), **dict(outcome.corrections), "kept": len(outcome.candidates)}
