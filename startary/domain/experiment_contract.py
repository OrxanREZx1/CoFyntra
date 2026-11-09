"""
Deterministic experiment contract and critic-decision rules (Milestone 6.2.5).

Single source of truth for the deterministic checks applied to a generated experiment draft, and
for how a critic decision is interpreted and routed, so the critic node, the validator, the router,
assess_can_execute, and telemetry can never disagree.

Findings have two severities:
  * VIOLATION — a reliable, objective defect. It produces a deterministic REVISE without spending a
    reasoning-model critique, and its message becomes the generator's revision instruction.
  * ADVISORY  — a heuristic or context note. It never blocks; it is shown to the LLM critic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Optional


class ContractSeverity(StrEnum):
    VIOLATION = "VIOLATION"
    ADVISORY = "ADVISORY"


class ContractCode(StrEnum):
    # Violations
    REQUIRED_FIELD_EMPTY = "REQUIRED_FIELD_EMPTY"
    ASSUMPTION_MISMATCH = "ASSUMPTION_MISMATCH"
    INVALID_SAMPLE_SIZE = "INVALID_SAMPLE_SIZE"
    COST_EXCEEDS_BUDGET = "COST_EXCEEDS_BUDGET"
    PROTOTYPE_NOT_AVAILABLE = "PROTOTYPE_NOT_AVAILABLE"
    BUILD_SCOPE_UNDEFINED = "BUILD_SCOPE_UNDEFINED"
    CONSENT_STEP_MISSING = "CONSENT_STEP_MISSING"
    GUARDIAN_CONSENT_MISSING = "GUARDIAN_CONSENT_MISSING"
    # Advisories
    BUDGET_UNKNOWN = "BUDGET_UNKNOWN"
    COST_NOT_COMPARABLE = "COST_NOT_COMPARABLE"
    BUILD_AUTHORIZED_WITHIN_SCOPE = "BUILD_AUTHORIZED_WITHIN_SCOPE"
    POSSIBLE_MINORS = "POSSIBLE_MINORS"
    PERSONAL_DATA_PRIVACY = "PERSONAL_DATA_PRIVACY"


# Which critic quality dimension a deterministic violation fails.
VIOLATION_DIMENSION: dict[ContractCode, str] = {
    ContractCode.REQUIRED_FIELD_EMPTY: "falsifiability",
    ContractCode.ASSUMPTION_MISMATCH: "assumption_alignment",
    ContractCode.INVALID_SAMPLE_SIZE: "evidence_validity",
    ContractCode.COST_EXCEEDS_BUDGET: "feasibility",
    ContractCode.PROTOTYPE_NOT_AVAILABLE: "feasibility",
    ContractCode.BUILD_SCOPE_UNDEFINED: "scope",
    ContractCode.CONSENT_STEP_MISSING: "feasibility",
    ContractCode.GUARDIAN_CONSENT_MISSING: "feasibility",
}


@dataclass(frozen=True)
class ContractFinding:
    code: ContractCode
    severity: ContractSeverity
    message: str
    # What established the finding (e.g. EXPERIMENT_PARTICIPANTS, PARTICIPANT_ACCESS, FOUNDER_CONTEXT).
    basis: str = ""


@dataclass
class ExperimentContractResult:
    violations: list[ContractFinding] = field(default_factory=list)
    advisories: list[ContractFinding] = field(default_factory=list)

    @property
    def violation_codes(self) -> list[str]:
        return [f.code.value for f in self.violations]

    @property
    def advisory_codes(self) -> list[str]:
        return [f.code.value for f in self.advisories]

    @property
    def failed_dimensions(self) -> list[str]:
        return sorted({VIOLATION_DIMENSION[f.code] for f in self.violations})

    @property
    def finding_bases(self) -> list[str]:
        """Non-sensitive provenance of each finding, e.g. "GUARDIAN_CONSENT_MISSING:PARTICIPANT_ACCESS"."""
        return [f"{f.code.value}:{f.basis}" for f in self.violations + self.advisories if f.basis]

    @property
    def revision_instructions(self) -> str:
        return " ".join(f.message for f in self.violations)


def _violation(code: ContractCode, message: str, basis: str = "") -> ContractFinding:
    return ContractFinding(code, ContractSeverity.VIOLATION, message, basis)


def _advisory(code: ContractCode, message: str, basis: str = "") -> ContractFinding:
    return ContractFinding(code, ContractSeverity.ADVISORY, message, basis)


def _field_value(field_dict: Any) -> Any:
    return field_dict.get("value") if isinstance(field_dict, dict) else field_dict


def _is_known_fact(field_dict: Any) -> bool:
    return (
        isinstance(field_dict, dict)
        and field_dict.get("value") is not None
        and str(field_dict.get("provenance") or "UNKNOWN").upper() not in ("UNKNOWN", "CONFLICTING")
    )


def _is_true(value: Any) -> bool:
    return value is True or str(value).strip().lower() in ("true", "yes", "1")


# ---------------------------------------------------------------------------
# Required fields
# ---------------------------------------------------------------------------

REQUIRED_TEXT_FIELDS = ("objective", "method", "evidence_to_collect", "support_signal", "contradiction_signal", "inconclusive_signal")
REQUIRED_LIST_FIELDS = ("founder_actions", "valid_inferences", "invalid_inferences")


def check_required_fields(exp: dict[str, Any]) -> list[ContractFinding]:
    """Whitespace-only text or lists of blank strings pass the schema but carry no content."""
    missing = [f for f in REQUIRED_TEXT_FIELDS if len(str(exp.get(f) or "").strip()) < 5]
    missing += [f for f in REQUIRED_LIST_FIELDS if not any(str(x).strip() for x in (exp.get(f) or []))]
    if not missing:
        return []
    return [_violation(
        ContractCode.REQUIRED_FIELD_EMPTY,
        f"Fill in the empty required fields with concrete content: {', '.join(missing)}.",
    )]


# ---------------------------------------------------------------------------
# Assumption alignment
# ---------------------------------------------------------------------------

def check_assumption_alignment(exp: dict[str, Any], active_assumption: dict[str, Any] | None) -> list[ContractFinding]:
    """
    generate_experiment stamps every draft with the id of the assumption it was generated for
    (code-owned, never model-generated). A stamp that differs from the active assumption means the
    draft is stale and tests a different assumption. Semantic alignment remains the critic's job.
    """
    stamp = exp.get("assumption_id")
    active_id = (active_assumption or {}).get("id")
    if stamp and active_id and str(stamp) != str(active_id):
        return [_violation(
            ContractCode.ASSUMPTION_MISMATCH,
            "This draft was generated for a different critical assumption; design the experiment for the active critical assumption.",
        )]
    return []


# ---------------------------------------------------------------------------
# Sample size (no arbitrary upper limits)
# ---------------------------------------------------------------------------

def check_sample_size(exp: dict[str, Any]) -> list[ContractFinding]:
    size = exp.get("target_sample_size")
    if isinstance(size, (int, float)) and not isinstance(size, bool) and size <= 0:
        return [_violation(ContractCode.INVALID_SAMPLE_SIZE, "Target sample size must be explicitly specified and at least 1.")]
    return []


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------

_CURRENCY_ALIASES = {
    "$": "USD", "usd": "USD", "dollar": "USD", "dollars": "USD",
    "€": "EUR", "eur": "EUR", "euro": "EUR", "euros": "EUR",
    "£": "GBP", "gbp": "GBP",
    "₼": "AZN", "azn": "AZN", "manat": "AZN",
}
_NUM = r"(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*(k)?"
_PREFIX_MONEY = re.compile(r"([$€£₼])\s*" + _NUM, re.IGNORECASE)
_SUFFIX_MONEY = re.compile(_NUM + r"\s*(usd|eur|gbp|azn|dollars?|euros?|manat)\b", re.IGNORECASE)
_BARE_NUMBER = re.compile(r"^\s*" + _NUM + r"\s*$", re.IGNORECASE)
_ZERO_COST = re.compile(r"\b(free|no cost|zero|near[- ]zero|nothing)\b", re.IGNORECASE)


@dataclass(frozen=True)
class MoneyAmount:
    low: float
    high: float
    currency: Optional[str]


def _to_number(num: str, k: Optional[str]) -> float:
    value = float(num.replace(",", ""))
    return value * 1000 if k else value


def parse_money(value: Any, allow_bare_number: bool = False) -> Optional[MoneyAmount]:
    """
    Parse a money amount or range. Only numbers anchored to a currency marker count, so
    "$50 over 2 weeks" is 50 (not 2). Bare numbers are accepted only for fields known to hold
    money (the founder's budget). Returns None when nothing comparable is present.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return MoneyAmount(float(value), float(value), None)
    text = str(value)
    amounts: list[float] = []
    currencies: set[str] = set()
    for symbol, num, k in _PREFIX_MONEY.findall(text):
        amounts.append(_to_number(num, k))
        currencies.add(_CURRENCY_ALIASES[symbol])
    for num, k, word in _SUFFIX_MONEY.findall(text):
        amounts.append(_to_number(num, k))
        currencies.add(_CURRENCY_ALIASES[word.lower()])
    if not amounts and allow_bare_number:
        bare = _BARE_NUMBER.match(text)
        if bare:
            amounts.append(_to_number(bare.group(1), bare.group(2)))
    if not amounts:
        if _ZERO_COST.search(text):
            return MoneyAmount(0.0, 0.0, None)
        return None
    currency = currencies.pop() if len(currencies) == 1 else None
    return MoneyAmount(min(amounts), max(amounts), currency)


class BudgetStatus(StrEnum):
    NO_COST = "NO_COST"
    WITHIN_BUDGET = "WITHIN_BUDGET"
    EXCEEDS_BUDGET = "EXCEEDS_BUDGET"
    BUDGET_UNKNOWN = "BUDGET_UNKNOWN"
    NOT_COMPARABLE = "NOT_COMPARABLE"


@dataclass(frozen=True)
class BudgetComparison:
    status: BudgetStatus
    cost: Optional[MoneyAmount] = None
    budget: Optional[MoneyAmount] = None


def compare_cost_to_budget(estimated_cost: Any, execution_context: dict[str, Any] | None) -> BudgetComparison:
    """
    Compare an experiment's estimated cost with the founder's available_budget_amount.
    An unknown budget stays unknown (never treated as zero). A cost range only exceeds the budget
    when even its lower bound does. Different currencies are reported as not comparable.
    """
    cost = parse_money(estimated_cost)
    if cost is None or cost.low <= 0:
        return BudgetComparison(BudgetStatus.NO_COST, cost)
    exec_ctx = execution_context or {}
    budget_field = exec_ctx.get("available_budget_amount")
    if not _is_known_fact(budget_field):
        return BudgetComparison(BudgetStatus.BUDGET_UNKNOWN, cost)
    budget = parse_money(_field_value(budget_field), allow_bare_number=True)
    if budget is None:
        return BudgetComparison(BudgetStatus.NOT_COMPARABLE, cost)
    budget_currency = budget.currency
    currency_field = exec_ctx.get("budget_currency")
    if budget_currency is None and _is_known_fact(currency_field):
        budget_currency = _CURRENCY_ALIASES.get(str(_field_value(currency_field)).strip().lower(), str(_field_value(currency_field)).strip().upper())
    if cost.currency and budget_currency and cost.currency != budget_currency:
        return BudgetComparison(BudgetStatus.NOT_COMPARABLE, cost, budget)
    if cost.low > budget.high:
        return BudgetComparison(BudgetStatus.EXCEEDS_BUDGET, cost, budget)
    return BudgetComparison(BudgetStatus.WITHIN_BUDGET, cost, budget)


def _fmt(amount: float) -> str:
    return f"{amount:,.0f}"


def check_budget(exp: dict[str, Any], execution_context: dict[str, Any] | None) -> list[ContractFinding]:
    comparison = compare_cost_to_budget(exp.get("estimated_cost"), execution_context)
    if comparison.status == BudgetStatus.EXCEEDS_BUDGET:
        return [_violation(
            ContractCode.COST_EXCEEDS_BUDGET,
            f"Estimated cost ({_fmt(comparison.cost.low)}) exceeds the founder's available budget "
            f"({_fmt(comparison.budget.high)}). Redesign the experiment to fit within the stated budget.",
        )]
    if comparison.status == BudgetStatus.BUDGET_UNKNOWN:
        return [_advisory(
            ContractCode.BUDGET_UNKNOWN,
            "The founder's budget is unknown. Do not assume it is zero; judge whether the estimated cost is realistic for this stage.",
        )]
    if comparison.status == BudgetStatus.NOT_COMPARABLE:
        return [_advisory(
            ContractCode.COST_NOT_COMPARABLE,
            "The estimated cost and the founder's budget could not be compared deterministically (different currency or format); check affordability.",
        )]
    return []


# ---------------------------------------------------------------------------
# Prototype availability vs. build-readiness decision
# ---------------------------------------------------------------------------

# Experiment types that need a working product artifact. MVP_TEST / USABILITY_TEST are legacy
# labels kept for robustness; ExperimentType itself only defines PROTOTYPE_TEST.
PRODUCT_DEPENDENT_TYPES = frozenset({"PROTOTYPE_TEST", "MVP_TEST", "USABILITY_TEST"})
BUILD_MINIMAL_PROTOTYPE = "BUILD_MINIMAL_PROTOTYPE"


class PrototypeStatus(StrEnum):
    NOT_APPLICABLE = "NOT_APPLICABLE"              # experiment does not depend on a product artifact
    EXISTING_ARTIFACT = "EXISTING_ARTIFACT"        # prototype / MVP / launched product exists: test it
    BUILD_AUTHORIZED = "BUILD_AUTHORIZED"          # build-readiness authorized a bounded build-and-test probe
    BUILD_SCOPE_UNDEFINED = "BUILD_SCOPE_UNDEFINED"  # BUILD decision without a bounded scope: not authorized
    UNAVAILABLE = "UNAVAILABLE"                    # needs an artifact that neither exists nor is authorized


@dataclass(frozen=True)
class PrototypeAssessment:
    status: PrototypeStatus
    minimum_build_scope: tuple[str, ...] = ()
    must_not_build: tuple[str, ...] = ()


def has_product_artifact(progress: dict[str, Any] | None) -> bool:
    prog = progress or {}
    return any(_is_true(_field_value(prog.get(k))) for k in ("prototype_exists", "mvp_exists", "launched"))


def assess_prototype_requirement(
    experiment_type: Any,
    progress: dict[str, Any] | None,
    build_readiness: dict[str, Any] | None,
) -> PrototypeAssessment:
    """
    Distinguishes testing an EXISTING prototype from BUILDING a minimal prototype to test an
    assumption. Building is allowed only when the build-readiness arbiter decided
    BUILD_MINIMAL_PROTOTYPE and defined a bounded minimum_build_scope.
    """
    if str(experiment_type or "").upper() not in PRODUCT_DEPENDENT_TYPES:
        return PrototypeAssessment(PrototypeStatus.NOT_APPLICABLE)
    if has_product_artifact(progress):
        return PrototypeAssessment(PrototypeStatus.EXISTING_ARTIFACT)
    br = build_readiness if isinstance(build_readiness, dict) else {}
    if str(br.get("decision") or "").upper() == BUILD_MINIMAL_PROTOTYPE:
        scope = tuple(str(s) for s in (br.get("minimum_build_scope") or []) if str(s).strip())
        must_not = tuple(str(s) for s in (br.get("must_not_build") or []) if str(s).strip())
        if scope:
            return PrototypeAssessment(PrototypeStatus.BUILD_AUTHORIZED, scope, must_not)
        return PrototypeAssessment(PrototypeStatus.BUILD_SCOPE_UNDEFINED)
    return PrototypeAssessment(PrototypeStatus.UNAVAILABLE)


def check_prototype(exp: dict[str, Any], progress: dict[str, Any] | None, build_readiness: dict[str, Any] | None) -> list[ContractFinding]:
    assessment = assess_prototype_requirement(exp.get("experiment_type"), progress, build_readiness)
    if assessment.status == PrototypeStatus.UNAVAILABLE:
        return [_violation(
            ContractCode.PROTOTYPE_NOT_AVAILABLE,
            "This experiment tests a working prototype, but none exists and building one has not been authorized. "
            "Test the assumption manually (interviews, observation, concierge) instead.",
        )]
    if assessment.status == PrototypeStatus.BUILD_SCOPE_UNDEFINED:
        return [_violation(
            ContractCode.BUILD_SCOPE_UNDEFINED,
            "Building a prototype was proposed without a bounded minimum build scope; keep the experiment manual.",
        )]
    if assessment.status == PrototypeStatus.BUILD_AUTHORIZED:
        must_not = f" Must NOT build: {'; '.join(assessment.must_not_build)}." if assessment.must_not_build else ""
        return [_advisory(
            ContractCode.BUILD_AUTHORIZED_WITHIN_SCOPE,
            "Build readiness decided BUILD_MINIMAL_PROTOTYPE. Building is authorized ONLY within this minimum scope: "
            f"{'; '.join(assessment.minimum_build_scope)}.{must_not} A bounded build-and-test probe within this scope is "
            "not build-first bias; anything beyond it is.",
        )]
    return []


# ---------------------------------------------------------------------------
# Privacy and consent
# ---------------------------------------------------------------------------

_RECORDING = re.compile(
    r"\b(?:video|audio|screen)[- ]?record(?:ing|ings|ed)?\b"
    r"|\brecord(?:ing|ed|s)?\s+(?:the\s+|their\s+|each\s+|all\s+|a\s+few\s+)?(?:matches|match|games?|interviews?|calls?|sessions?|conversations?|trainings?|practices?|players?)\b"
    r"|\brecordings?\s+of\b"
    r"|\b(?:film|films|filming|filmed|footage|camera|cameras|camcorder|gopro|photograph|photographs|photographing)\b"
    r"|\b(?:screenshot|screenshots|screen captures?)\b"
    r"|\bvideos?\s+of\s+(?:the\s+|their\s+)?(?:matches|match|games?|players?|trainings?|sessions?)\b"
    r"|\b(?:match|game|training)\s+(?:videos?|footage)\b"
    r"|\bupload(?:s|ed|ing)?\s+(?:their\s+|the\s+|existing\s+|recorded\s+)?(?:match\s+)?videos?\b",
    re.IGNORECASE,
)
# Recording a conversation captures the interviewee (e.g. an adult coach); recording matches, training
# or players can capture every player on the pitch. Matches of _RECORDING that are not conversation
# recordings are treated as capturing athletes (or an unspecified group).
_CONVERSATION_RECORDING = re.compile(
    r"^(?:audio[- ]?record\w*|screen[- ]?record\w*|screenshots?|screen captures?)$"
    r"|^record\w*\s+(?:the\s+|their\s+|each\s+|all\s+|a\s+few\s+)?(?:interviews?|calls?|conversations?)$"
    r"|^recordings?\s+of$",
    re.IGNORECASE,
)
_MINOR_MODIFIER = r"(?:youth|junior|academy|school|teenage|boys'?|girls'?|u-?(?:[89]|1[0-8])|under[- ]?1[0-8])"
# Minors as PEOPLE or GROUPS OF PEOPLE (players, teams, matches) — not markets or adult roles.
# "youth football academies" (a market) and "coaches at youth academies" (adults) do not match.
_MINOR_GROUP = re.compile(
    r"\b(?:minors?|children|child|kids|teenagers?|teens|under[- ]?1[0-8]s|u-?(?:[89]|1[0-8])s)\b"
    r"|\b" + _MINOR_MODIFIER + r"(?:\s+(?:youth|junior|academy|football|soccer|club|age[- ]group))*\s+"
    r"(?:players?|athletes?|footballers?|squads?|teams?|matches|match|games?|training|sessions?|league|kids)\b"
    r"|\b(?:players?|athletes?|footballers?|children|kids)\s+aged\s+(?:[5-9]|1[0-7])\b",
    re.IGNORECASE,
)
# A minor group named only to qualify an adult role ("coaches of youth teams") is not a participant.
_ADULT_ROLE_QUALIFIER = re.compile(
    r"\b(?:coach(?:es)?|managers?|staff|parents?|organi[sz]ers?|directors?|owners?|trainers?)\s+"
    r"(?:of|for|at|from|running|leading)\s+(?:the\s+|their\s+|our\s+|two\s+|three\s+|\d+\s+)?$",
    re.IGNORECASE,
)
_EXPLICIT_ADULTS = re.compile(
    r"\b(?:adults?|over[- ]18s?|18\+|aged\s+18\s+or\s+(?:over|older)|senior\s+(?:teams?|players?|squads?)|"
    r"men's\s+teams?|women's\s+teams?)\b",
    re.IGNORECASE,
)
# Broad youth vocabulary: only ever produces an advisory (age unknown), never a violation.
_YOUTH_MENTION = re.compile(
    r"\b(?:minors?|children|child|kids|youth|juniors?|teenagers?|teens|academy|academies|under[- ]?1[0-8]s?|u-?(?:[89]|1[0-8])s?)\b",
    re.IGNORECASE,
)
_CONSENT = re.compile(
    r"\b(?:consent|permission|permissions|authori[sz]ation|authori[sz]e|authori[sz]ed|opt[- ]?in|"
    r"agree(?:s|d)?\s+to\s+be\s+(?:recorded|filmed)|privacy|anonymi[sz](?:e|ed|ing|ation)|blur(?:red|ring)?)\b",
    re.IGNORECASE,
)
_GUARDIAN = re.compile(r"\b(?:parents?|parental|guardians?)\b", re.IGNORECASE)
_PERSONAL_DATA = re.compile(
    r"\b(?:collect|gather|store|save|export|share|log)\w*\b[^.]{0,40}\b(?:email addresses|phone numbers|personal data|"
    r"personal information|contact details|home addresses|dates? of birth|location data|identifiable|names|ages|photos)\b",
    re.IGNORECASE,
)

_PRIVACY_FIELDS = ("objective", "method", "target_participants_or_scope", "evidence_to_collect")
_PRIVACY_LIST_FIELDS = ("founder_actions", "required_access")


def _experiment_text(exp: dict[str, Any]) -> str:
    parts = [str(exp.get(f) or "") for f in _PRIVACY_FIELDS]
    for f in _PRIVACY_LIST_FIELDS:
        parts.extend(str(x) for x in (exp.get(f) or []))
    return "\n".join(parts)


def _names_minor_participants(text: str) -> bool:
    """True when the text names minors as people or groups, except as a qualifier of an adult role."""
    for match in _MINOR_GROUP.finditer(text or ""):
        if _ADULT_ROLE_QUALIFIER.search(text[max(0, match.start() - 40):match.start()]):
            continue
        return True
    return False


@dataclass(frozen=True)
class PrivacySignals:
    """Deterministic privacy facts about a draft (shared by the contract check and M6.2.6 authorization tagging)."""
    records_people: bool
    records_athletes: bool
    collects_personal_data: bool
    has_consent: bool
    has_guardian: bool
    adults_stated: bool
    minors_basis: Optional[str]


def privacy_signals(exp: dict[str, Any], participant_context: str = "") -> PrivacySignals:
    text = _experiment_text(exp)
    recordings = [m.group(0).strip() for m in _RECORDING.finditer(text)]
    records_athletes = any(not _CONVERSATION_RECORDING.match(r) for r in recordings)
    collects_personal_data = bool(_PERSONAL_DATA.search(text))
    adults_stated = bool(_EXPLICIT_ADULTS.search(text))
    participants = " ".join([str(exp.get("target_participants_or_scope") or "")] + [str(x) for x in (exp.get("required_access") or [])])

    if _names_minor_participants(participants):
        minors_basis = "EXPERIMENT_PARTICIPANTS"
    elif (records_athletes or collects_personal_data) and _names_minor_participants(text):
        minors_basis = "EXPERIMENT_TEXT"
    elif records_athletes and not adults_stated and _names_minor_participants(participant_context):
        minors_basis = "PARTICIPANT_ACCESS"
    else:
        minors_basis = None
    return PrivacySignals(
        records_people=bool(recordings),
        records_athletes=records_athletes,
        collects_personal_data=collects_personal_data,
        has_consent=bool(_CONSENT.search(text)),
        has_guardian=bool(_GUARDIAN.search(text)),
        adults_stated=adults_stated,
        minors_basis=minors_basis,
    )


def check_privacy_consent(exp: dict[str, Any], founder_context: str = "", participant_context: str = "") -> list[ContractFinding]:
    """
    Experiments that record audio/video, film matches, or capture screenshots/photos of people must
    include an explicit permission/consent step. A parent/guardian (or club) authorization step is
    required only when the EXPERIMENT ITSELF involves minors (Milestone 6.2.5.1):
      * minors are the experiment's declared participants (target_participants_or_scope / required_access);
      * the experiment films athletes/matches, or collects identifiable data, about named minor groups;
      * the experiment films athletes/matches and the founder's participant-access fact
        (``participant_context``) names minor teams/players, unless the experiment states adults.
    General founder conversation (``founder_context``, e.g. "our market is youth football academies")
    never creates a violation by itself; with unconfirmed ages it produces a POSSIBLE_MINORS advisory.
    Interviews with adults (e.g. coaches) are untouched beyond recording consent.
    This is a practical safeguard, not a statement of legal compliance.
    """
    text = _experiment_text(exp)
    sig = privacy_signals(exp, participant_context)
    records_people, records_athletes = sig.records_people, sig.records_athletes
    collects_personal_data, has_consent = sig.collects_personal_data, sig.has_consent
    has_guardian, adults_stated, minors_basis = sig.has_guardian, sig.adults_stated, sig.minors_basis

    findings: list[ContractFinding] = []
    if records_people and not has_consent:
        findings.append(_violation(
            ContractCode.CONSENT_STEP_MISSING,
            "The experiment records people (video/audio/footage/screenshots). Add an explicit founder action to obtain "
            "permission from everyone recorded (and the team or venue where relevant) before recording, and state how "
            "the recordings will be stored, shared, and deleted or anonymized.",
            basis="EXPERIMENT_TEXT",
        ))
    if minors_basis and not has_guardian:
        findings.append(_violation(
            ContractCode.GUARDIAN_CONSENT_MISSING,
            "This experiment involves minors (recording, identifiable data, or direct participation). Add an explicit "
            "step to obtain parent/guardian (and club) authorization before involving any player under 18.",
            basis=minors_basis,
        ))
    elif not minors_basis and records_people and not adults_stated:
        youth_in_context = bool(_YOUTH_MENTION.search(founder_context or "") or _YOUTH_MENTION.search(text))
        if records_athletes or youth_in_context:
            findings.append(_advisory(
                ContractCode.POSSIBLE_MINORS,
                ("Youth football is mentioned (founder context or experiment), but the ages of the people this experiment records are "
                 "not stated. Confirm everyone recorded is 18 or older; otherwise a parent/guardian and club "
                 "authorization step is required.")
                if youth_in_context else
                ("The experiment records people. If any participants may be under 18, a parent/guardian or club "
                 "authorization step is also required."),
                basis="YOUTH_MENTION" if youth_in_context else "UNSTATED_AGES",
            ))
    if not records_people and collects_personal_data and not has_consent and not minors_basis:
        findings.append(_advisory(
            ContractCode.PERSONAL_DATA_PRIVACY,
            "The experiment collects identifiable personal information; check that participants are told what is "
            "collected and why, and agree to it.",
            basis="EXPERIMENT_TEXT",
        ))
    return findings


# ---------------------------------------------------------------------------
# Aggregate contract
# ---------------------------------------------------------------------------

def founder_context_text(state: dict[str, Any]) -> str:
    msgs = state.get("founder_messages") or []
    return "\n".join([str(state.get("raw_context") or "")] + [str(m.get("content", "")) for m in msgs if isinstance(m, dict) and m.get("role") == "founder"])


def participant_access_text(state: dict[str, Any]) -> str:
    """The founder's stated participant access (customer_access fact): who the experiment can actually involve."""
    access = (state.get("execution_context") or {}).get("customer_access")
    value = _field_value(access) if _is_known_fact(access) else None
    return str(value or "")


class Authorization(StrEnum):
    """Permissions an experiment needs BEFORE it may be executed (Milestone 6.2.6). Never auto-granted."""
    RECORDING_PERMISSION = "RECORDING_PERMISSION"
    GUARDIAN_AUTHORIZATION = "GUARDIAN_AUTHORIZATION"
    PERSONAL_DATA_NOTICE = "PERSONAL_DATA_NOTICE"


def required_authorizations(exp: dict[str, Any] | None, state: dict[str, Any]) -> list[str]:
    """
    Authorizations the founder must actually obtain before running the experiment. Derived from the
    same privacy signals as the contract check; a permission STEP in the plan does not mean it is granted.
    """
    sig = privacy_signals(exp or {}, participant_access_text(state))
    codes: list[str] = []
    if sig.records_people:
        codes.append(Authorization.RECORDING_PERMISSION.value)
    if sig.minors_basis:
        codes.append(Authorization.GUARDIAN_AUTHORIZATION.value)
    if sig.collects_personal_data:
        codes.append(Authorization.PERSONAL_DATA_NOTICE.value)
    return codes


AUTHORIZATION_LABELS = {
    Authorization.RECORDING_PERMISSION.value: "permission from everyone who will be recorded",
    Authorization.GUARDIAN_AUTHORIZATION.value: "parent/guardian and club authorization for every participant under 18",
    Authorization.PERSONAL_DATA_NOTICE.value: "participants' agreement to the personal information being collected",
}


def pending_authorizations(experiment: dict[str, Any] | None) -> list[str]:
    """Authorizations still outstanding. Nothing in Startary marks them OBTAINED; only the founder can."""
    exp = experiment or {}
    if exp.get("authorization_status") == "OBTAINED":
        return []
    return [str(a) for a in (exp.get("required_authorizations") or [])]


def pending_authorization_notice(experiment: dict[str, Any] | None) -> str:
    """Founder-facing notice shown before the handoff when permissions are required (empty if none)."""
    pending = pending_authorizations(experiment)
    if not pending:
        return ""
    lines = ["BEFORE YOU START: these permissions are required and have NOT been obtained yet:"]
    lines += [f"- {AUTHORIZATION_LABELS.get(a, a)}" for a in pending]
    lines.append("Do not record anyone, involve minors, or collect personal information until you have them.")
    return "\n".join(lines)


def evaluate_experiment_contract(exp: dict[str, Any] | None, state: dict[str, Any]) -> ExperimentContractResult:
    """Run every deterministic check on a draft. Pure: never mutates the draft or the state."""
    exp = exp or {}
    findings = (
        check_required_fields(exp)
        + check_assumption_alignment(exp, state.get("active_assumption") or state.get("active_critical_assumption"))
        + check_sample_size(exp)
        + check_budget(exp, state.get("execution_context"))
        + check_prototype(exp, state.get("startup_progress"), state.get("build_readiness"))
        + check_privacy_consent(exp, founder_context_text(state), participant_access_text(state))
    )
    return ExperimentContractResult(
        violations=[f for f in findings if f.severity == ContractSeverity.VIOLATION],
        advisories=[f for f in findings if f.severity == ContractSeverity.ADVISORY],
    )


# ---------------------------------------------------------------------------
# Critic decision interpretation (strict) and routing
# ---------------------------------------------------------------------------

CRITIC_DIMENSIONS = ("assumption_alignment", "evidence_validity", "feasibility", "cost_efficiency", "falsifiability", "scope")
CRITIC_VERDICTS = ("ACCEPT", "REVISE")
DIMENSION_RESULTS = ("PASS", "FAIL")
MAX_EXPERIMENT_REVISIONS = 1


def strict_token(value: Any, allowed: tuple[str, ...]) -> Optional[str]:
    """Exact token match after trimming/upper-casing. "ACCEPTED" or "accept with changes" -> None."""
    if value is None:
        return None
    token = str(getattr(value, "value", value)).strip().upper()
    return token if token in allowed else None


@dataclass(frozen=True)
class CriticDecision:
    verdict: str                              # effective verdict used for routing
    model_verdict: Optional[str]              # verdict as emitted (None if absent/invalid or deterministic)
    dimensions: dict[str, Optional[str]]      # PASS / FAIL / NOT_EVALUATED / None (invalid)
    failed_dimensions: list[str]
    verdict_flag_conflict: bool
    decision_source: str                      # LLM | DETERMINISTIC


def interpret_critic_output(raw: dict[str, Any]) -> CriticDecision:
    """
    Fail-closed interpretation of an LLM critique. ACCEPT requires an exact ACCEPT verdict AND all
    six dimensions PASS (the critic prompt's own contract). Any FAIL, missing/invalid flag, or
    non-ACCEPT verdict yields REVISE. Disagreements between verdict and flags are flagged.
    """
    model_verdict = strict_token(raw.get("verdict"), CRITIC_VERDICTS)
    dims = {d: strict_token(raw.get(d), DIMENSION_RESULTS) for d in CRITIC_DIMENSIONS}
    failed = [d for d, v in dims.items() if v == "FAIL"]
    all_pass = all(v == "PASS" for v in dims.values())
    verdict = "ACCEPT" if (model_verdict == "ACCEPT" and all_pass) else "REVISE"
    conflict = (model_verdict == "ACCEPT" and not all_pass) or (model_verdict == "REVISE" and all_pass) or model_verdict is None
    return CriticDecision(verdict, model_verdict, dims, failed, conflict, "LLM")


def deterministic_decision(contract: ExperimentContractResult) -> CriticDecision:
    failed = contract.failed_dimensions
    dims = {d: ("FAIL" if d in failed else "NOT_EVALUATED") for d in CRITIC_DIMENSIONS}
    return CriticDecision("REVISE", None, dims, failed, False, "DETERMINISTIC")


def experiment_critique_route(verdict: Any, revision_count: int, revision_strategy: Optional[str] = None) -> str:
    """
    Routing rule shared by route_on_experiment_critique and critic telemetry.
    revision_strategy is only set by the M6.2.6 V2 loop; TARGETED_EDIT goes to revise_experiment.
    """
    if strict_token(verdict, CRITIC_VERDICTS) == "ACCEPT":
        return "save_final_experiment"
    if int(revision_count or 0) <= MAX_EXPERIMENT_REVISIONS:
        return "revise_experiment" if revision_strategy == "TARGETED_EDIT" else "generate_experiment"
    return "handle_experiment_failure_stop"


# Issue categories (codes only, so telemetry never carries critic free text).
_ISSUE_CATEGORY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("BUILD_FIRST", re.compile(r"\b(build|building|develop|app|platform|software|code|coding|feature)\b", re.I)),
    # "pitch" alone is ambiguous (a football pitch), so only pitching the solution/product counts.
    ("CONTAMINATION", re.compile(
        r"pitch(?:es|ing|ed)?\s+(?:the|your|a|our)\s+(?:solution|product|prototype|idea|app|tool)"
        r"|\bpitching\b|\bdemo(?:s|ing)?\b|\b(?:sell|sells|selling)\b|leading question|contaminat|\bbias(?:ed)?\b", re.I)),
    ("FALSIFIABILITY", re.compile(r"(falsif|threshold|decision rule|contradiction signal)", re.I)),
    ("EVIDENCE_VALIDITY", re.compile(r"(willingness to pay|stated intent|opinion|hypothetical|would you|validity)", re.I)),
    ("SAMPLE", re.compile(r"\b(sample|participants?|recruit\w*)\b", re.I)),
    ("BUDGET_COST", re.compile(r"(budget|cost|\$|expensive|paid ads)", re.I)),
    ("TIMELINE", re.compile(r"\b(weeks?|days?|timeline|too long|duration)\b", re.I)),
    ("ACCESS", re.compile(r"\b(access|reach|channel|network)\b", re.I)),
    ("PROTOTYPE", re.compile(r"\b(prototype|mvp)\b", re.I)),
    ("OVERPOWERED_CRITERIA", re.compile(r"(stronger than|top three|overstat)", re.I)),
    ("PRIVACY_CONSENT", re.compile(r"(consent|privacy|record|film|footage|minor|guardian|personal data)", re.I)),
    ("SCOPE", re.compile(r"(scope|multi-phase|multi-stage|complex|roadmap)", re.I)),
    ("ASSUMPTION_ALIGNMENT", re.compile(r"(assumption|hypothesis|misalign|secondary)", re.I)),
)


def classify_issue_codes(issues: list[Any]) -> list[str]:
    codes: set[str] = set()
    for issue in issues or []:
        text = str(issue)
        matched = [code for code, pattern in _ISSUE_CATEGORY_PATTERNS if pattern.search(text)]
        codes.update(matched or ["OTHER"])
    return sorted(codes)


def critic_decision_metadata(
    decision: CriticDecision,
    *,
    revision_round: int,
    revision_count_after: int,
    issues: list[Any],
    contract: ExperimentContractResult,
    attempts: int,
    failure_kinds: list[str],
    revision_strategy: Optional[str] = None,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Structured, non-sensitive critic decision record for Startary telemetry and LangSmith metadata.
    ``extra`` carries the M6.2.6 V2 revision fields (only present when the V2 flag is on).
    Contains only enums, codes, counts and flags: no founder text, experiment text, or critic prose.
    """
    meta: dict[str, Any] = {
        "critic_verdict": decision.verdict,
        "critic_model_verdict": decision.model_verdict,
        "verdict_flag_conflict": decision.verdict_flag_conflict,
        "failed_dimensions": list(decision.failed_dimensions),
        "revision_trigger": None if decision.verdict == "ACCEPT" else decision.decision_source,
        "revision_round": revision_round,
        "issue_count": len(issues or []),
        "issue_codes": classify_issue_codes(issues) if decision.decision_source == "LLM" else [],
        "contract_violation_codes": contract.violation_codes,
        "contract_advisory_codes": contract.advisory_codes,
        "contract_finding_bases": contract.finding_bases,
        "critic_attempts": attempts,
        "llm_failure_kinds": list(failure_kinds),
        "next_route": experiment_critique_route(decision.verdict, revision_count_after, revision_strategy),
    }
    for dim, result in decision.dimensions.items():
        meta[f"dim_{dim}"] = result
    if extra:
        meta.update(extra)
    return meta
