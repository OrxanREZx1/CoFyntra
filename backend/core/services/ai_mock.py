"""
Deterministic, contract-valid mock for all five capabilities. Demo/testing only.

Enabled only by ``COFYTRA_AI_MODE=mock`` (which requires DEBUG or an explicit opt-in, see ai_adapter.get_backend)
or by injecting ``MockBackend()`` in tests. Responses:
  - are real cofytra_ai response models, so they go through the same validation as production output;
  - echo the request_id, cite only IDs present in the request, and are a pure function of the request;
  - report ``tier="none"``, ``model=None``, ``attempts=0``: no LLM is involved, and the meta says so;
  - never echo whole founder messages; extraction quotes a short verbatim prefix, as the contract requires.
"""

from __future__ import annotations

from pydantic import BaseModel

from cofytra_ai.models import (
    CATCHUP_BUCKETS,
    MAX_CANDIDATES,
    MAX_CONFLICTS,
    MAX_POINTS_PER_BUCKET,
    MAX_STATEMENT_CHARS,
    MAX_TITLE_CHARS,
    CatchupRequest,
    CatchupResponse,
    ChatRequest,
    ChatResponse,
    ConflictRequest,
    ConflictResponse,
    EpistemicStatus,
    ExtractRequest,
    ExtractResponse,
    KnowledgeCategory,
    MessageRole,
    ProposalRequest,
    ProposalResponse,
    ResponseMeta,
    normalize_text,
)

MOCK_PREFIX = "[mock]"
_QUOTE_CHARS = 200


def _meta(request, capability: str) -> ResponseMeta:
    return ResponseMeta(request_id=request.request_id, prompt_version=f"cofytra.mock.{capability}.v1",
                        model=None, tier="none", latency_ms=0, attempts=0)


def _fit(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _name(founder) -> str:
    return founder.display_name or "Your co-founder"


def mock_chat(request: ChatRequest) -> ChatResponse:
    state = request.committed_state
    reply = (f"{MOCK_PREFIX} Reply generated without an LLM. Context used: {len(request.recent_messages)} earlier "
             f"private messages, {len(request.shared_knowledge)} shared knowledge items, committed state "
             f"{'revision ' + str(state.version) if state else 'not set yet'}.")
    return ChatResponse(
        reply=reply,
        referenced_knowledge_ids=[k.id for k in request.shared_knowledge[-3:]],
        referenced_decision_ids=[d.id for d in state.decisions[-1:]] if state else [],
        meta=_meta(request, "chat"),
    )


def mock_extract_knowledge(request: ExtractRequest) -> ExtractResponse:
    """One candidate from the latest founder message: a verbatim prefix, labelled as an untested hypothesis."""
    founder_messages = [m for m in request.messages if m.role == MessageRole.FOUNDER]
    candidates = []
    for message in founder_messages[-1:][:MAX_CANDIDATES]:
        quote = message.text[:_QUOTE_CHARS].strip()
        duplicate = next((k.id for k in request.existing_knowledge
                          if normalize_text(k.statement) == normalize_text(quote)), None)
        candidates.append({
            "category": KnowledgeCategory.PROBLEM, "statement": _fit(quote, MAX_STATEMENT_CHARS),
            "epistemic_status": EpistemicStatus.FOUNDER_HYPOTHESIS, "confidence": "low",
            "source_message_ids": [message.id], "source_quote": quote, "external_reference": None,
            "duplicate_of_knowledge_id": duplicate,
        })
    return ExtractResponse(candidates=candidates, meta=_meta(request, "extract"))


def mock_catch_up(request: CatchupRequest) -> CatchupResponse:
    name = _name(request.other_founder)
    if not request.knowledge and not request.pending_proposals:
        return CatchupResponse(summary=f"{name} has not shared new knowledge or proposals since your last catch-up.",
                               meta=_meta(request, "catchup"))
    buckets: dict[str, list[dict]] = {bucket: [] for bucket in CATCHUP_BUCKETS}
    for item in request.knowledge:
        bucket = next(b for b, categories in CATCHUP_BUCKETS.items() if item.category in categories)
        if len(buckets[bucket]) < MAX_POINTS_PER_BUCKET:
            text = f"{MOCK_PREFIX} {name} shared ({item.epistemic_status.value}): {item.statement}"
            buckets[bucket].append({"text": _fit(text, MAX_STATEMENT_CHARS), "knowledge_ids": [item.id]})
    return CatchupResponse(
        summary=(f"{MOCK_PREFIX} {name} shared {len(request.knowledge)} knowledge items; "
                 f"{len(request.pending_proposals)} proposals await team approval."),
        pending_decisions=[{"proposal_id": p.id, "text": _fit(f"{MOCK_PREFIX} Awaiting approval: {p.title}",
                                                              MAX_STATEMENT_CHARS)}
                           for p in request.pending_proposals],
        meta=_meta(request, "catchup"),
        **buckets,
    )


def mock_detect_conflicts(request: ConflictRequest) -> ConflictResponse:
    """Flags a 'tension' for each pair of different founders' items in the same strategic category."""
    strategic = {KnowledgeCategory.PROBLEM, KnowledgeCategory.TARGET_CUSTOMER, KnowledgeCategory.MARKET}
    conflicts = []
    items = request.knowledge
    for i, a in enumerate(items):
        for b in items[i + 1:]:
            if len(conflicts) >= MAX_CONFLICTS:
                break
            if a.author_id != b.author_id and a.category == b.category and a.category in strategic:
                conflicts.append({
                    "claim_a": {"knowledge_id": a.id, "author_id": a.author_id, "statement": a.statement},
                    "claim_b": {"knowledge_id": b.id, "author_id": b.author_id, "statement": b.statement},
                    "kind": "tension",
                    "explanation": f"{MOCK_PREFIX} Both founders made different claims about {a.category.value}.",
                    "question_to_resolve": f"{MOCK_PREFIX} Which claim has stronger evidence?",
                })
    return ConflictResponse(conflicts=conflicts, meta=_meta(request, "conflicts"))


def mock_draft_proposal(request: ProposalRequest) -> ProposalResponse:
    by_id = {k.id: k for k in request.knowledge}
    if request.conflict:
        referenced = [request.conflict.claim_a_id, request.conflict.claim_b_id]
    else:
        referenced = [k.id for k in request.knowledge[-5:]]
    warnings = []
    if not request.knowledge:
        warnings.append({"code": "no_supporting_knowledge", "message": f"{MOCK_PREFIX} No shared knowledge supports this draft."})
    hypotheses = [i for i in referenced if by_id[i].epistemic_status == EpistemicStatus.FOUNDER_HYPOTHESIS]
    if hypotheses:
        warnings.append({"code": "relies_on_hypothesis", "message": f"{MOCK_PREFIX} Relies on untested hypotheses.",
                         "knowledge_ids": hypotheses})
    if request.conflict:
        warnings.append({"code": "unresolved_conflict", "message": f"{MOCK_PREFIX} Addresses an open conflict.",
                         "knowledge_ids": referenced})
    state = request.committed_state
    return ProposalResponse(
        proposal={
            "title": _fit(f"{MOCK_PREFIX} {request.topic}", MAX_TITLE_CHARS),
            "rationale": f"{MOCK_PREFIX} Draft generated without an LLM from {len(request.knowledge)} knowledge items.",
            "trade_offs": [f"{MOCK_PREFIX} Placeholder trade-off; replace before approval."],
            "next_action": f"{MOCK_PREFIX} Review this draft together and edit it before approval.",
            "referenced_knowledge_ids": referenced,
            "referenced_decision_ids": [d.id for d in state.decisions[-1:]] if state else [],
        },
        warnings=warnings,
        meta=_meta(request, "proposal"),
    )


_HANDLERS = {
    "chat": mock_chat,
    "extract_knowledge": mock_extract_knowledge,
    "catch_up": mock_catch_up,
    "detect_conflicts": mock_detect_conflicts,
    "draft_proposal": mock_draft_proposal,
}


class MockBackend:
    name = "mock"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def call(self, capability: str, request: BaseModel) -> BaseModel:
        self.calls.append(capability)
        return _HANDLERS[capability](request)
