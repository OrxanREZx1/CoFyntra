"""
Cofytra AI contract v1: Pydantic request/response models for the five AI capabilities.

The Django backend builds a request model from trusted, already-authorized data, calls one of
the functions in ``cofytra_ai.service``, and turns the response model into persistence
operations itself. This package never touches Django models or the database.

Data layers (never mixed):
  PrivateMessage       one founder's own private chat with the assistant (chat/extract only)
  KnowledgeCandidate   AI-extracted claim proposed for sharing; NOT persisted, NOT shared yet
  SharedKnowledgeItem  knowledge Django has persisted and shared with the team
  CommittedState       team-approved startup state and committed decisions (Django-owned)

Contract rules enforced here:
  - Unknown fields are rejected everywhere (``extra="forbid"``).
  - No epistemic status implies independent verification.
  - Every ID in a response must come from its request (see ``contract_violations``).
  - Catch-up requests cannot carry private messages.

This module imports only pydantic and the standard library.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal, Optional

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StringConstraints, model_validator

CONTRACT_VERSION = "v1"

# ---------------------------------------------------------------------------
# Context-size limits. Django trims context (most recent first) before calling;
# a request over any limit fails validation instead of being silently truncated.
# ---------------------------------------------------------------------------
MAX_ID_CHARS = 128
MAX_DISPLAY_NAME_CHARS = 80
MAX_PRIVATE_MESSAGES = 20
MAX_MESSAGE_CHARS = 4000
MAX_KNOWLEDGE_ITEMS = 60
MAX_STATEMENT_CHARS = 500
MAX_QUOTE_CHARS = 1000
MAX_EXTERNAL_REFERENCE_CHARS = 300
MAX_DECISIONS = 30
MAX_PENDING_PROPOSALS = 20
MAX_TITLE_CHARS = 120
MAX_TOPIC_CHARS = 300
MAX_DECISION_SUMMARY_CHARS = 1000

MAX_REPLY_CHARS = 6000
MAX_SUMMARY_CHARS = 2000
MAX_CANDIDATES = 20
MAX_SOURCES_PER_CANDIDATE = 5
MAX_POINTS_PER_BUCKET = 10
MAX_IDS_PER_POINT = 10
MAX_CONFLICTS = 10
MAX_TRADE_OFFS = 5
MAX_WARNINGS = 10

_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.:\-]*$"

Id = Annotated[str, StringConstraints(min_length=1, max_length=MAX_ID_CHARS, pattern=_ID_PATTERN)]


def _text(max_chars: int) -> Any:
    """Non-empty text after stripping; empty strings are rejected (use None for 'absent')."""
    return Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=max_chars)]


DisplayName = _text(MAX_DISPLAY_NAME_CHARS)
MessageText = _text(MAX_MESSAGE_CHARS)
Statement = _text(MAX_STATEMENT_CHARS)
Quote = _text(MAX_QUOTE_CHARS)
ExternalReference = _text(MAX_EXTERNAL_REFERENCE_CHARS)
Title = _text(MAX_TITLE_CHARS)
Topic = _text(MAX_TOPIC_CHARS)
DecisionSummary = _text(MAX_DECISION_SUMMARY_CHARS)
ShortText = _text(300)
MediumText = _text(1000)
LongText = _text(MAX_SUMMARY_CHARS)
ReplyText = _text(MAX_REPLY_CHARS)
VersionTag = _text(64)


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _require_unique(ids: list[str], what: str) -> None:
    seen: set[str] = set()
    dupes = sorted({i for i in ids if i in seen or seen.add(i)})  # type: ignore[func-returns-value]
    if dupes:
        raise ValueError(f"duplicate {what}: {', '.join(dupes)}")


def normalize_text(text: str) -> str:
    """Whitespace- and case-insensitive form used for verbatim quote checks."""
    return " ".join(text.split()).casefold()


# ---------------------------------------------------------------------------
# Vocabularies
# ---------------------------------------------------------------------------

class KnowledgeCategory(StrEnum):
    PROBLEM = "problem"
    TARGET_CUSTOMER = "target_customer"
    MARKET = "market"
    ASSUMPTION = "assumption"
    EVIDENCE = "evidence"


class EpistemicStatus(StrEnum):
    """
    Where a claim comes from. None of these values means "independently verified":
    Cofytra never verifies claims. There is deliberately no VERIFIED status.
    """

    FOUNDER_HYPOTHESIS = "founder_hypothesis"    # founder belief, guess, or assertion with no reported observation
    FOUNDER_OBSERVATION = "founder_observation"  # founder reports something they observed first-hand (interviews, usage, sales)
    AI_SUGGESTION = "ai_suggestion"              # originated from the AI, adopted by a founder; never founder-stated
    EXTERNAL_SOURCE = "external_source"          # founder cites an external source (report, article, dataset); not checked


class Confidence(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class MessageRole(StrEnum):
    FOUNDER = "founder"
    ASSISTANT = "assistant"


# Which statuses each category may carry. Assumptions are never observations; evidence is never a guess.
ALLOWED_STATUSES: dict[KnowledgeCategory, frozenset[EpistemicStatus]] = {
    KnowledgeCategory.PROBLEM: frozenset(EpistemicStatus),
    KnowledgeCategory.TARGET_CUSTOMER: frozenset(EpistemicStatus),
    KnowledgeCategory.MARKET: frozenset(EpistemicStatus),
    KnowledgeCategory.ASSUMPTION: frozenset({EpistemicStatus.FOUNDER_HYPOTHESIS, EpistemicStatus.AI_SUGGESTION}),
    KnowledgeCategory.EVIDENCE: frozenset({EpistemicStatus.FOUNDER_OBSERVATION, EpistemicStatus.EXTERNAL_SOURCE}),
}

# Extraction reads founder-authored text only, so it can never produce an AI suggestion.
CANDIDATE_STATUSES = frozenset(EpistemicStatus) - {EpistemicStatus.AI_SUGGESTION}


def _check_status_rules(category: KnowledgeCategory, status: EpistemicStatus, external_reference: Optional[str]) -> None:
    if status not in ALLOWED_STATUSES[category]:
        raise ValueError(f"epistemic_status {status.value!r} is not allowed for category {category.value!r}")
    if status == EpistemicStatus.EXTERNAL_SOURCE and not external_reference:
        raise ValueError("external_source items require external_reference")
    if status != EpistemicStatus.EXTERNAL_SOURCE and external_reference:
        raise ValueError("external_reference is only allowed with epistemic_status 'external_source'")


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------

class FounderRef(_Contract):
    id: Id
    display_name: Optional[DisplayName] = None


class PrivateMessage(_Contract):
    """One message from the requesting founder's own private assistant chat."""

    id: Id
    role: MessageRole
    author_id: Optional[Id] = Field(default=None, description="Founder ID for founder messages; null for assistant messages.")
    text: MessageText
    created_at: AwareDatetime

    @model_validator(mode="after")
    def _author_matches_role(self) -> "PrivateMessage":
        if self.role == MessageRole.FOUNDER and self.author_id is None:
            raise ValueError("founder messages require author_id")
        if self.role == MessageRole.ASSISTANT and self.author_id is not None:
            raise ValueError("assistant messages must have author_id null")
        return self


class SharedKnowledgeItem(_Contract):
    """Knowledge Django has persisted and shared with the project team."""

    id: Id
    author_id: Id
    category: KnowledgeCategory
    statement: Statement
    epistemic_status: EpistemicStatus
    source_quote: Optional[Quote] = None
    external_reference: Optional[ExternalReference] = None
    shared_at: AwareDatetime

    @model_validator(mode="after")
    def _status_rules(self) -> "SharedKnowledgeItem":
        _check_status_rules(self.category, self.epistemic_status, self.external_reference)
        return self


class CommittedDecision(_Contract):
    id: Id
    title: Title
    summary: DecisionSummary
    committed_at: AwareDatetime
    referenced_knowledge_ids: list[Id] = Field(default_factory=list, max_length=MAX_KNOWLEDGE_ITEMS)


class CommittedState(_Contract):
    """Team-approved startup state. Only Django changes it, through its approval flow."""

    version: int = Field(ge=1)
    problem: Optional[Statement] = None
    target_customer: Optional[Statement] = None
    market: Optional[Statement] = None
    decisions: list[CommittedDecision] = Field(default_factory=list, max_length=MAX_DECISIONS)

    @model_validator(mode="after")
    def _unique(self) -> "CommittedState":
        _require_unique([d.id for d in self.decisions], "decision ids")
        return self

    def decision_ids(self) -> set[str]:
        return {d.id for d in self.decisions}


class PendingProposal(_Contract):
    """A proposal awaiting team approval (Django-owned)."""

    id: Id
    title: Title
    author_id: Id
    created_at: AwareDatetime
    referenced_knowledge_ids: list[Id] = Field(default_factory=list, max_length=MAX_KNOWLEDGE_ITEMS)


class ResponseMeta(_Contract):
    request_id: Id
    contract_version: Literal["v1"] = CONTRACT_VERSION
    prompt_version: VersionTag
    model: Optional[VersionTag] = Field(default=None, description="Resolved model; null when no LLM call was made.")
    tier: Literal["fast", "reasoning", "none"]
    latency_ms: int = Field(ge=0)
    attempts: int = Field(ge=0, le=3, description="LLM attempts; 0 when answered deterministically.")

    @model_validator(mode="after")
    def _consistent(self) -> "ResponseMeta":
        if self.tier == "none" and (self.model is not None or self.attempts != 0):
            raise ValueError("tier 'none' requires model null and attempts 0")
        if self.tier != "none" and (self.model is None or self.attempts < 1):
            raise ValueError("an LLM tier requires a model and at least one attempt")
        return self


class _Request(_Contract):
    request_id: Id
    project_id: Id


def _knowledge_ids(items: list[SharedKnowledgeItem]) -> list[str]:
    return [k.id for k in items]


def _check_private_messages(messages: list[PrivateMessage], founder_id: str) -> None:
    _require_unique([m.id for m in messages], "message ids")
    for m in messages:
        if m.role == MessageRole.FOUNDER and m.author_id != founder_id:
            raise ValueError(f"message {m.id} belongs to another founder; only the requesting founder's own messages are allowed")
    for prev, cur in zip(messages, messages[1:]):
        if cur.created_at < prev.created_at:
            raise ValueError("messages must be ordered oldest to newest")


# ---------------------------------------------------------------------------
# 1. chat
# ---------------------------------------------------------------------------

class ChatRequest(_Request):
    founder: FounderRef
    message: MessageText
    recent_messages: list[PrivateMessage] = Field(default_factory=list, max_length=MAX_PRIVATE_MESSAGES)
    shared_knowledge: list[SharedKnowledgeItem] = Field(default_factory=list, max_length=MAX_KNOWLEDGE_ITEMS)
    committed_state: Optional[CommittedState] = None

    @model_validator(mode="after")
    def _identity(self) -> "ChatRequest":
        _check_private_messages(self.recent_messages, self.founder.id)
        _require_unique(_knowledge_ids(self.shared_knowledge), "knowledge ids")
        return self


class ChatResponse(_Contract):
    reply: ReplyText
    referenced_knowledge_ids: list[Id] = Field(default_factory=list, max_length=MAX_KNOWLEDGE_ITEMS)
    referenced_decision_ids: list[Id] = Field(default_factory=list, max_length=MAX_DECISIONS)
    meta: ResponseMeta

    @model_validator(mode="after")
    def _unique(self) -> "ChatResponse":
        _require_unique(self.referenced_knowledge_ids, "referenced_knowledge_ids")
        _require_unique(self.referenced_decision_ids, "referenced_decision_ids")
        return self


# ---------------------------------------------------------------------------
# 2. extract_knowledge
# ---------------------------------------------------------------------------

class ExtractRequest(_Request):
    founder: FounderRef
    messages: list[PrivateMessage] = Field(min_length=1, max_length=MAX_PRIVATE_MESSAGES)
    existing_knowledge: list[SharedKnowledgeItem] = Field(default_factory=list, max_length=MAX_KNOWLEDGE_ITEMS)

    @model_validator(mode="after")
    def _identity(self) -> "ExtractRequest":
        _check_private_messages(self.messages, self.founder.id)
        if not any(m.role == MessageRole.FOUNDER for m in self.messages):
            raise ValueError("messages must contain at least one founder-authored message")
        _require_unique(_knowledge_ids(self.existing_knowledge), "knowledge ids")
        return self


class KnowledgeCandidate(_Contract):
    """An extracted claim proposed for sharing. It has no ID: Django assigns one if it is accepted."""

    category: KnowledgeCategory
    statement: Statement
    epistemic_status: EpistemicStatus
    confidence: Confidence
    source_message_ids: list[Id] = Field(min_length=1, max_length=MAX_SOURCES_PER_CANDIDATE)
    source_quote: Quote
    external_reference: Optional[ExternalReference] = None
    duplicate_of_knowledge_id: Optional[Id] = None

    @model_validator(mode="after")
    def _status_rules(self) -> "KnowledgeCandidate":
        if self.epistemic_status not in CANDIDATE_STATUSES:
            raise ValueError("extracted candidates cannot have epistemic_status 'ai_suggestion'")
        _check_status_rules(self.category, self.epistemic_status, self.external_reference)
        _require_unique(self.source_message_ids, "source_message_ids")
        return self


class ExtractionDropCounts(_Contract):
    """How many model-proposed candidates the service discarded, and why."""

    unknown_source: int = Field(default=0, ge=0)
    quote_not_found: int = Field(default=0, ge=0)
    invalid_status: int = Field(default=0, ge=0)


class ExtractResponse(_Contract):
    candidates: list[KnowledgeCandidate] = Field(default_factory=list, max_length=MAX_CANDIDATES)
    dropped: ExtractionDropCounts = Field(default_factory=ExtractionDropCounts)
    meta: ResponseMeta


# ---------------------------------------------------------------------------
# 3. catch_up
# ---------------------------------------------------------------------------

class CatchupRequest(_Request):
    """Summarizes the other founder's SHARED knowledge. Has no field for private messages."""

    viewer: FounderRef
    other_founder: FounderRef
    since: Optional[AwareDatetime] = None
    knowledge: list[SharedKnowledgeItem] = Field(default_factory=list, max_length=MAX_KNOWLEDGE_ITEMS)
    pending_proposals: list[PendingProposal] = Field(default_factory=list, max_length=MAX_PENDING_PROPOSALS)
    committed_state: Optional[CommittedState] = None

    @model_validator(mode="after")
    def _identity(self) -> "CatchupRequest":
        if self.viewer.id == self.other_founder.id:
            raise ValueError("viewer and other_founder must be different founders")
        _require_unique(_knowledge_ids(self.knowledge), "knowledge ids")
        _require_unique([p.id for p in self.pending_proposals], "proposal ids")
        for k in self.knowledge:
            if k.author_id != self.other_founder.id:
                raise ValueError(f"knowledge {k.id} is not authored by other_founder")
            if self.since is not None and k.shared_at < self.since:
                raise ValueError(f"knowledge {k.id} was shared before 'since'")
        return self


class CatchupPoint(_Contract):
    text: Statement
    knowledge_ids: list[Id] = Field(min_length=1, max_length=MAX_IDS_PER_POINT)

    @model_validator(mode="after")
    def _unique(self) -> "CatchupPoint":
        _require_unique(self.knowledge_ids, "knowledge_ids")
        return self


class PendingDecisionPoint(_Contract):
    proposal_id: Id
    text: Statement


class CatchupResponse(_Contract):
    summary: LongText
    discoveries: list[CatchupPoint] = Field(default_factory=list, max_length=MAX_POINTS_PER_BUCKET)
    assumptions: list[CatchupPoint] = Field(default_factory=list, max_length=MAX_POINTS_PER_BUCKET)
    evidence: list[CatchupPoint] = Field(default_factory=list, max_length=MAX_POINTS_PER_BUCKET)
    pending_decisions: list[PendingDecisionPoint] = Field(default_factory=list, max_length=MAX_PENDING_PROPOSALS)
    meta: ResponseMeta


# Bucket -> categories whose items may be cited in it.
CATCHUP_BUCKETS: dict[str, frozenset[KnowledgeCategory]] = {
    "discoveries": frozenset({KnowledgeCategory.PROBLEM, KnowledgeCategory.TARGET_CUSTOMER, KnowledgeCategory.MARKET}),
    "assumptions": frozenset({KnowledgeCategory.ASSUMPTION}),
    "evidence": frozenset({KnowledgeCategory.EVIDENCE}),
}


# ---------------------------------------------------------------------------
# 4. detect_conflicts
# ---------------------------------------------------------------------------

class ConflictKind(StrEnum):
    CONTRADICTION = "contradiction"  # both cannot be true
    TENSION = "tension"              # could both be true, but pull the strategy in different directions


class ConflictRequest(_Request):
    knowledge: list[SharedKnowledgeItem] = Field(default_factory=list, max_length=MAX_KNOWLEDGE_ITEMS)

    @model_validator(mode="after")
    def _unique(self) -> "ConflictRequest":
        _require_unique(_knowledge_ids(self.knowledge), "knowledge ids")
        return self


class ClaimRef(_Contract):
    knowledge_id: Id
    author_id: Id
    statement: Statement


class Conflict(_Contract):
    claim_a: ClaimRef
    claim_b: ClaimRef
    kind: ConflictKind
    explanation: MediumText
    question_to_resolve: ShortText

    @model_validator(mode="after")
    def _cross_founder(self) -> "Conflict":
        if self.claim_a.knowledge_id == self.claim_b.knowledge_id:
            raise ValueError("a conflict needs two different knowledge items")
        if self.claim_a.author_id == self.claim_b.author_id:
            raise ValueError("a conflict must be between claims of different founders")
        return self


class ConflictResponse(_Contract):
    conflicts: list[Conflict] = Field(default_factory=list, max_length=MAX_CONFLICTS)
    meta: ResponseMeta

    @model_validator(mode="after")
    def _unique_pairs(self) -> "ConflictResponse":
        pairs = [frozenset((c.claim_a.knowledge_id, c.claim_b.knowledge_id)) for c in self.conflicts]
        if len(pairs) != len(set(pairs)):
            raise ValueError("the same pair of claims appears in more than one conflict")
        return self


# ---------------------------------------------------------------------------
# 5. draft_proposal
# ---------------------------------------------------------------------------

class ConflictPair(_Contract):
    claim_a_id: Id
    claim_b_id: Id


class ProposalRequest(_Request):
    requested_by: FounderRef
    topic: Topic
    knowledge: list[SharedKnowledgeItem] = Field(default_factory=list, max_length=MAX_KNOWLEDGE_ITEMS)
    conflict: Optional[ConflictPair] = None
    committed_state: Optional[CommittedState] = None

    @model_validator(mode="after")
    def _refs(self) -> "ProposalRequest":
        ids = _knowledge_ids(self.knowledge)
        _require_unique(ids, "knowledge ids")
        if self.conflict is not None:
            if self.conflict.claim_a_id == self.conflict.claim_b_id:
                raise ValueError("conflict claims must be different items")
            missing = {self.conflict.claim_a_id, self.conflict.claim_b_id} - set(ids)
            if missing:
                raise ValueError(f"conflict references knowledge not in the request: {', '.join(sorted(missing))}")
        return self


class ProposalDraft(_Contract):
    """A draft only. The AI never approves or commits; Django's approval flow does."""

    status: Literal["draft"] = "draft"
    title: Title
    rationale: LongText
    trade_offs: list[Statement] = Field(min_length=1, max_length=MAX_TRADE_OFFS)
    next_action: Statement
    referenced_knowledge_ids: list[Id] = Field(default_factory=list, max_length=MAX_KNOWLEDGE_ITEMS)
    referenced_decision_ids: list[Id] = Field(default_factory=list, max_length=MAX_DECISIONS)

    @model_validator(mode="after")
    def _unique(self) -> "ProposalDraft":
        _require_unique(self.referenced_knowledge_ids, "referenced_knowledge_ids")
        _require_unique(self.referenced_decision_ids, "referenced_decision_ids")
        return self


class ProposalWarningCode(StrEnum):
    RELIES_ON_HYPOTHESIS = "relies_on_hypothesis"
    RELIES_ON_AI_SUGGESTION = "relies_on_ai_suggestion"
    NO_SUPPORTING_KNOWLEDGE = "no_supporting_knowledge"
    UNRESOLVED_CONFLICT = "unresolved_conflict"


class ProposalWarning(_Contract):
    code: ProposalWarningCode
    message: ShortText
    knowledge_ids: list[Id] = Field(default_factory=list, max_length=MAX_KNOWLEDGE_ITEMS)


class ProposalResponse(_Contract):
    proposal: ProposalDraft
    warnings: list[ProposalWarning] = Field(default_factory=list, max_length=MAX_WARNINGS)
    meta: ResponseMeta


# ---------------------------------------------------------------------------
# Errors (raised as cofytra_ai.errors.CofytraAIError; Django maps them to HTTP)
# ---------------------------------------------------------------------------

class ErrorCode(StrEnum):
    INVALID_REQUEST = "invalid_request"            # request failed contract validation (a Django bug, not the user's)
    PROVIDER_CONFIG = "provider_config"            # missing/invalid API key, model not accessible
    PROVIDER_UNAVAILABLE = "provider_unavailable"  # timeout, rate limit, 5xx after the internal retry
    INVALID_MODEL_OUTPUT = "invalid_model_output"  # model output still malformed after the corrective retry
    TOKEN_BUDGET = "token_budget"                  # output budget exhausted even after the one expansion
    NOT_IMPLEMENTED = "not_implemented"            # capability not built yet (contract stub)
    INTERNAL = "internal"


ERROR_RETRYABLE: dict[ErrorCode, bool] = {
    ErrorCode.INVALID_REQUEST: False,
    ErrorCode.PROVIDER_CONFIG: False,
    ErrorCode.PROVIDER_UNAVAILABLE: True,
    ErrorCode.INVALID_MODEL_OUTPUT: True,
    ErrorCode.TOKEN_BUDGET: False,
    ErrorCode.NOT_IMPLEMENTED: False,
    ErrorCode.INTERNAL: False,
}

# Suggested DRF status codes. Django owns the actual HTTP response.
SUGGESTED_HTTP_STATUS: dict[ErrorCode, int] = {
    ErrorCode.INVALID_REQUEST: 500,
    ErrorCode.PROVIDER_CONFIG: 503,
    ErrorCode.PROVIDER_UNAVAILABLE: 503,
    ErrorCode.INVALID_MODEL_OUTPUT: 502,
    ErrorCode.TOKEN_BUDGET: 502,
    ErrorCode.NOT_IMPLEMENTED: 501,
    ErrorCode.INTERNAL: 500,
}


class FieldIssue(_Contract):
    field: ShortText
    issue: ShortText


class ErrorInfo(_Contract):
    """Serializable error payload. Never contains founder text or model output."""

    code: ErrorCode
    message: ShortText
    retryable: bool
    request_id: Optional[Id] = None
    details: list[FieldIssue] = Field(default_factory=list, max_length=50)

    @model_validator(mode="after")
    def _retryable_matches_code(self) -> "ErrorInfo":
        if self.retryable != ERROR_RETRYABLE[self.code]:
            raise ValueError(f"retryable must be {ERROR_RETRYABLE[self.code]} for code {self.code.value!r}")
        return self


# ---------------------------------------------------------------------------
# Capability registry and reference-integrity check
# ---------------------------------------------------------------------------

CAPABILITIES: dict[str, tuple[type[BaseModel], type[BaseModel]]] = {
    "chat": (ChatRequest, ChatResponse),
    "extract_knowledge": (ExtractRequest, ExtractResponse),
    "catch_up": (CatchupRequest, CatchupResponse),
    "detect_conflicts": (ConflictRequest, ConflictResponse),
    "draft_proposal": (ProposalRequest, ProposalResponse),
}


def _unknown(ids: list[str], allowed: set[str], where: str) -> list[str]:
    return [f"{where}: unknown id {i!r}" for i in ids if i not in allowed]


def contract_violations(request: BaseModel, response: BaseModel) -> list[str]:
    """
    Cross-check a response against its request. Returns human-readable violations (empty = OK).

    Guarantees that every ID in the response came from the request, that extraction quotes are
    verbatim, that catch-up buckets cite the right categories, and that conflicts preserve both
    claims exactly. Service functions run this before returning; Django may run it too.
    """
    v: list[str] = []
    meta = getattr(response, "meta", None)
    if meta is not None and meta.request_id != getattr(request, "request_id", None):
        v.append("meta.request_id does not match the request")

    decision_ids = request.committed_state.decision_ids() if getattr(request, "committed_state", None) else set()

    if isinstance(request, ChatRequest) and isinstance(response, ChatResponse):
        v += _unknown(response.referenced_knowledge_ids, set(_knowledge_ids(request.shared_knowledge)), "referenced_knowledge_ids")
        v += _unknown(response.referenced_decision_ids, decision_ids, "referenced_decision_ids")

    elif isinstance(request, ExtractRequest) and isinstance(response, ExtractResponse):
        founder_msgs = {m.id: m for m in request.messages if m.role == MessageRole.FOUNDER}
        existing = set(_knowledge_ids(request.existing_knowledge))
        for n, c in enumerate(response.candidates):
            where = f"candidates[{n}]"
            v += _unknown(c.source_message_ids, set(founder_msgs), f"{where}.source_message_ids (founder messages only)")
            cited = [normalize_text(founder_msgs[i].text) for i in c.source_message_ids if i in founder_msgs]
            if cited and not any(normalize_text(c.source_quote) in t for t in cited):
                v.append(f"{where}.source_quote is not verbatim in any cited message")
            if c.external_reference and cited and not any(normalize_text(c.external_reference) in t for t in cited):
                v.append(f"{where}.external_reference does not appear in any cited message")
            if c.duplicate_of_knowledge_id is not None:
                v += _unknown([c.duplicate_of_knowledge_id], existing, f"{where}.duplicate_of_knowledge_id")

    elif isinstance(request, CatchupRequest) and isinstance(response, CatchupResponse):
        by_id = {k.id: k for k in request.knowledge}
        for bucket, categories in CATCHUP_BUCKETS.items():
            for n, point in enumerate(getattr(response, bucket)):
                where = f"{bucket}[{n}].knowledge_ids"
                v += _unknown(point.knowledge_ids, set(by_id), where)
                for i in point.knowledge_ids:
                    if i in by_id and by_id[i].category not in categories:
                        v.append(f"{where}: {i!r} has category {by_id[i].category.value!r}, not allowed in {bucket}")
        v += _unknown([p.proposal_id for p in response.pending_decisions], {p.id for p in request.pending_proposals}, "pending_decisions.proposal_id")

    elif isinstance(request, ConflictRequest) and isinstance(response, ConflictResponse):
        by_id = {k.id: k for k in request.knowledge}
        for n, c in enumerate(response.conflicts):
            for side in ("claim_a", "claim_b"):
                ref: ClaimRef = getattr(c, side)
                where = f"conflicts[{n}].{side}"
                item = by_id.get(ref.knowledge_id)
                if item is None:
                    v.append(f"{where}: unknown id {ref.knowledge_id!r}")
                    continue
                if ref.author_id != item.author_id:
                    v.append(f"{where}.author_id does not match the knowledge item")
                if ref.statement != item.statement:
                    v.append(f"{where}.statement is not the verbatim knowledge statement")

    elif isinstance(request, ProposalRequest) and isinstance(response, ProposalResponse):
        known = set(_knowledge_ids(request.knowledge))
        v += _unknown(response.proposal.referenced_knowledge_ids, known, "proposal.referenced_knowledge_ids")
        v += _unknown(response.proposal.referenced_decision_ids, decision_ids, "proposal.referenced_decision_ids")
        for n, w in enumerate(response.warnings):
            v += _unknown(w.knowledge_ids, known, f"warnings[{n}].knowledge_ids")

    else:
        v.append(f"no contract pairs {type(request).__name__} with {type(response).__name__}")
    return v
