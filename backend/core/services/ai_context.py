"""
Builds trusted ``cofytra_ai`` request models from authorized Django ORM records.

Rules applied by every builder:
  - The caller's identity is the authenticated ``User`` passed in, never a client-supplied ID.
    Active project membership (and message ownership) is checked BEFORE any context is read.
  - IDs are ``str(pk)`` (UUIDs, and integer user PKs, both match the contract ID pattern).
  - Datetimes are Django's timezone-aware values.
  - Nothing stored is modified. Trimming affects only the request and is reported in ``ContextReport``.

Context trimming policy (deterministic; limits come from cofytra_ai.models):
  private messages  the founder's own chat only; most recent MAX_PRIVATE_MESSAGES, sent oldest -> newest
  knowledge         most recent MAX_KNOWLEDGE_ITEMS (chat, extract, proposal)
                    conflicts: an equal per-author quota first, then the most recent of the rest
                    catch-up: OLDEST first after the cursor, so a long backlog is paged, never skipped
  decisions         most recent MAX_DECISIONS up to the latest committed revision
  pending proposals most recent MAX_PENDING_PROPOSALS
  founder text      never shortened; over-limit founder input is rejected
  assistant text    stored in full (up to MAX_REPLY_CHARS). Only its *context representation* is shortened
                    when longer than MAX_MESSAGE_CHARS: the first characters are kept and
                    ``ASSISTANT_SHORTENED_MARKER`` is appended, for a total of exactly MAX_MESSAGE_CHARS or less.

Contract note: Django's ``CommittedState.next_action`` is not sent; cofytra_ai's CommittedState has no
such field yet (it rejects unknown fields). Add it when the AI team ships it.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Generic, Iterable, Optional, TypeVar

from django.core.exceptions import PermissionDenied, ValidationError
from django.db.models import Q
from django.utils import timezone
from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from cofytra_ai.models import (
    MAX_DECISIONS,
    MAX_DISPLAY_NAME_CHARS,
    MAX_KNOWLEDGE_ITEMS,
    MAX_MESSAGE_CHARS,
    MAX_PENDING_PROPOSALS,
    MAX_PRIVATE_MESSAGES,
    MAX_TOPIC_CHARS,
    CatchupRequest,
    ChatRequest,
    ConflictRequest,
    ErrorCode,
    ExtractRequest,
    MessageRole,
    ProposalRequest,
)
from core.models import (
    CatchupCursor,
    CommittedDecision,
    DecisionProposal,
    PrivateMessage,
    Project,
    ProjectMembership,
    ProposalVersion,
    SharedKnowledge,
)
from core.services.ai_adapter import AIAdapterError, logger

RequestT = TypeVar("RequestT", bound=BaseModel)

ASSISTANT_SHORTENED_MARKER = "\n[Earlier assistant reply shortened for context; the full reply is stored.]"


@dataclass(frozen=True)
class ContextReport:
    """What the builder left out or shortened, so trimming is never silent."""

    omitted: dict[str, int] = field(default_factory=dict)
    shortened_message_ids: tuple[str, ...] = ()
    catch_up_complete: Optional[bool] = None
    catch_up_covered_until: Optional[datetime] = None
    catch_up_covered_knowledge_id: Optional[str] = None


@dataclass(frozen=True)
class BuiltContext(Generic[RequestT]):
    request: RequestT
    report: ContextReport


# ---------------------------------------------------------------------------
# Authorization and identity
# ---------------------------------------------------------------------------

def require_active_founder(project: Project, user) -> ProjectMembership:
    if user is None or not getattr(user, "is_authenticated", False):
        raise PermissionDenied("Authentication required.")
    membership = ProjectMembership.objects.filter(project=project, user=user, is_active=True).first()
    if membership is None:
        raise PermissionDenied("Not an active founder of this project.")
    return membership


def other_active_founder(project: Project, viewer):
    others = list(ProjectMembership.objects.filter(project=project, is_active=True).exclude(user=viewer)
                  .select_related("user")[:2])
    if len(others) != 1:
        raise ValidationError("Catch-up needs exactly one other active founder in the project.")
    return others[0].user


def contract_id(value) -> str:
    return str(value)


def new_request_id(capability: str) -> str:
    return f"{capability}-{uuid.uuid4().hex}"


def founder_ref(user) -> dict:
    name = (user.get_full_name() or user.get_username() or "").strip()[:MAX_DISPLAY_NAME_CHARS].strip()
    return {"id": contract_id(user.pk), "display_name": name or None}


# ---------------------------------------------------------------------------
# Record -> contract dict conversions
# ---------------------------------------------------------------------------

def message_context_text(message: PrivateMessage) -> tuple[str, bool]:
    """Context representation of a stored message. Returns (text, shortened). Never mutates the record."""
    text = message.content
    if len(text) <= MAX_MESSAGE_CHARS:
        return text, False
    if message.role != MessageRole.ASSISTANT.value:
        raise ValidationError("Founder messages are never shortened for AI context.")
    head = text[: MAX_MESSAGE_CHARS - len(ASSISTANT_SHORTENED_MARKER)].rstrip()
    return head + ASSISTANT_SHORTENED_MARKER, True


def _message_dicts(messages: Iterable[PrivateMessage]) -> tuple[list[dict], tuple[str, ...]]:
    out, shortened = [], []
    for m in messages:
        text, was_shortened = message_context_text(m)
        if was_shortened:
            shortened.append(contract_id(m.pk))
        out.append({"id": contract_id(m.pk), "role": m.role, "author_id": m.contract_author_id,
                    "text": text, "created_at": m.created_at})
    return out, tuple(shortened)


def knowledge_dict(k: SharedKnowledge) -> dict:
    return {"id": contract_id(k.pk), "author_id": contract_id(k.author_id), "category": k.category,
            "statement": k.statement, "epistemic_status": k.epistemic_status, "source_quote": k.source_quote,
            "external_reference": k.external_reference, "shared_at": k.shared_at}


def _sorted_chronologically(items: Iterable[SharedKnowledge]) -> list[SharedKnowledge]:
    return sorted(items, key=lambda k: (k.shared_at, str(k.pk)))


def _recent(queryset, limit: int, *order) -> tuple[list, int]:
    """Most recent ``limit`` rows (by ``order`` descending), returned oldest -> newest, plus how many were left out."""
    total = queryset.count()
    rows = list(queryset.order_by(*(f"-{o}" for o in order))[:limit])
    rows.reverse()
    return rows, total - len(rows)


def recent_knowledge(project: Project, limit: int = MAX_KNOWLEDGE_ITEMS) -> tuple[list[SharedKnowledge], int]:
    return _recent(project.knowledge.all(), limit, "shared_at", "id")


def balanced_knowledge(project: Project, limit: int = MAX_KNOWLEDGE_ITEMS) -> tuple[list[SharedKnowledge], int]:
    """Equal per-author quota (most recent first), remainder filled by overall recency. Deterministic."""
    total = project.knowledge.count()
    authors = list(project.knowledge.order_by("author_id").values_list("author_id", flat=True).distinct())
    if not authors:
        return [], 0
    chosen: dict = {}
    quota = limit // len(authors)
    for author_id in authors:
        for k in project.knowledge.filter(author_id=author_id).order_by("-shared_at", "-id")[:quota]:
            chosen[k.pk] = k
    remaining = limit - len(chosen)
    if remaining > 0:
        for k in project.knowledge.exclude(pk__in=list(chosen)).order_by("-shared_at", "-id")[:remaining]:
            chosen[k.pk] = k
    return _sorted_chronologically(chosen.values()), total - len(chosen)


def committed_state_dict(project: Project) -> tuple[Optional[dict], int]:
    """Latest committed revision (contract ``version``) with its most recent decisions; None if nothing committed."""
    state = project.latest_committed_state()
    if state is None:
        return None, 0
    decisions_qs = (CommittedDecision.objects.filter(project=project, committed_state__revision__lte=state.revision)
                    .prefetch_related("knowledge_links"))
    decisions, omitted = _recent(decisions_qs, MAX_DECISIONS, "committed_at", "id")
    return {
        "version": state.revision,
        "problem": state.problem,
        "target_customer": state.target_customer,
        "market": state.market,
        "decisions": [
            {"id": contract_id(d.pk), "title": d.title, "summary": d.summary, "committed_at": d.committed_at,
             "referenced_knowledge_ids": [contract_id(link.knowledge_id)
                                          for link in sorted(d.knowledge_links.all(), key=lambda link: link.pk)]}
            for d in decisions
        ],
    }, omitted


def _build(model_cls: type[RequestT], data: dict, capability: str) -> RequestT:
    """Validate the assembled context. Failure is a Django-side bug, reported as invalid_request."""
    try:
        return model_cls.model_validate(data)
    except PydanticValidationError as exc:
        logger.warning("ai_context_invalid capability=%s request_id=%s issues=%s",
                       capability, data.get("request_id"), exc.error_count())
        raise AIAdapterError(capability, ErrorCode.INVALID_REQUEST, data.get("request_id")) from None


def _founder_history(project: Project, founder, *, before: PrivateMessage, inclusive: bool):
    position = Q(created_at__lt=before.created_at) | Q(created_at=before.created_at, id__lt=before.pk)
    if inclusive:
        position |= Q(pk=before.pk)
    return PrivateMessage.objects.filter(project=project, owner=founder).filter(position)


def _require_own_founder_message(project: Project, founder, message: PrivateMessage) -> None:
    if (message.project_id != project.pk or message.owner_id != founder.pk
            or message.role != MessageRole.FOUNDER.value):
        raise PermissionDenied("The message is not a founder message in this founder's own chat.")


# ---------------------------------------------------------------------------
# Builders, one per capability
# ---------------------------------------------------------------------------

def build_chat_request(*, project: Project, founder, message: PrivateMessage) -> BuiltContext[ChatRequest]:
    """``message`` is the founder's stored message being answered; history is that founder's chat only."""
    require_active_founder(project, founder)
    _require_own_founder_message(project, founder, message)
    history, omitted_messages = _recent(_founder_history(project, founder, before=message, inclusive=False),
                                        MAX_PRIVATE_MESSAGES, "created_at", "id")
    recent_messages, shortened = _message_dicts(history)
    knowledge, omitted_knowledge = recent_knowledge(project)
    committed, omitted_decisions = committed_state_dict(project)
    data = {
        "request_id": new_request_id("chat"),
        "project_id": contract_id(project.pk),
        "founder": founder_ref(founder),
        "message": message.content,
        "recent_messages": recent_messages,
        "shared_knowledge": [knowledge_dict(k) for k in knowledge],
        "committed_state": committed,
    }
    report = ContextReport(omitted={"messages": omitted_messages, "knowledge": omitted_knowledge,
                                    "decisions": omitted_decisions}, shortened_message_ids=shortened)
    return BuiltContext(_build(ChatRequest, data, "chat"), report)


def build_extract_request(*, project: Project, founder, source_message: PrivateMessage) -> BuiltContext[ExtractRequest]:
    """Extraction window: the founder's own chat up to and including ``source_message``."""
    require_active_founder(project, founder)
    _require_own_founder_message(project, founder, source_message)
    window, omitted_messages = _recent(_founder_history(project, founder, before=source_message, inclusive=True),
                                       MAX_PRIVATE_MESSAGES, "created_at", "id")
    messages, shortened = _message_dicts(window)
    knowledge, omitted_knowledge = recent_knowledge(project)
    data = {
        "request_id": new_request_id("extract"),
        "project_id": contract_id(project.pk),
        "founder": founder_ref(founder),
        "messages": messages,
        "existing_knowledge": [knowledge_dict(k) for k in knowledge],
    }
    report = ContextReport(omitted={"messages": omitted_messages, "knowledge": omitted_knowledge},
                           shortened_message_ids=shortened)
    return BuiltContext(_build(ExtractRequest, data, "extract_knowledge"), report)


def _pending_proposals(project: Project) -> tuple[list[dict], int]:
    proposals, omitted = _recent(project.proposals.filter(status=DecisionProposal.Status.PENDING,
                                                          current_version_number__gte=1),
                                 MAX_PENDING_PROPOSALS, "created_at", "id")
    out = []
    for p in proposals:
        version = ProposalVersion.objects.get(proposal=p, version_number=p.current_version_number)
        out.append({"id": contract_id(p.pk), "title": version.title, "author_id": contract_id(p.created_by_id),
                    "created_at": p.created_at,
                    "referenced_knowledge_ids": [contract_id(i) for i in version.knowledge_links.order_by("id")
                                                 .values_list("knowledge_id", flat=True)]})
    return out, omitted


def build_catchup_request(*, project: Project, viewer) -> BuiltContext[CatchupRequest]:
    """
    The other founder's SHARED knowledge since the viewer's cursor. No private message is ever read.

    Paging: items after the composite (shared_at, knowledge_id) cursor are taken oldest-first. The contract
    still receives the timestamp as ``since``; the ID is Django-only and disambiguates timestamp ties.
    The cursor itself is not updated here.
    """
    require_active_founder(project, viewer)
    other = other_active_founder(project, viewer)
    cursor = CatchupCursor.objects.filter(project=project, viewer=viewer, other_founder=other).first()
    since = cursor.last_caught_up_at if cursor else None

    pending = project.knowledge.filter(author=other)
    if since is not None:
        if cursor.last_knowledge_id is None:
            pending = pending.filter(Q(shared_at__gt=since) | Q(shared_at=since, id__gte=uuid.UUID(int=0)))
        else:
            pending = pending.filter(
                Q(shared_at__gt=since) | Q(shared_at=since, id__gt=cursor.last_knowledge_id)
            )
    total = pending.count()
    page = list(pending.order_by("shared_at", "id")[: MAX_KNOWLEDGE_ITEMS + 1])
    complete = len(page) <= MAX_KNOWLEDGE_ITEMS
    page = page[:MAX_KNOWLEDGE_ITEMS]
    covered_until = page[-1].shared_at if page else since
    covered_knowledge_id = str(page[-1].pk) if page else None

    proposals, omitted_proposals = _pending_proposals(project)
    committed, omitted_decisions = committed_state_dict(project)
    data = {
        "request_id": new_request_id("catchup"),
        "project_id": contract_id(project.pk),
        "viewer": founder_ref(viewer),
        "other_founder": founder_ref(other),
        "since": since,
        "knowledge": [knowledge_dict(k) for k in page],
        "pending_proposals": proposals,
        "committed_state": committed,
    }
    report = ContextReport(omitted={"knowledge": total - len(page), "pending_proposals": omitted_proposals,
                                    "decisions": omitted_decisions},
                           catch_up_complete=complete, catch_up_covered_until=covered_until,
                           catch_up_covered_knowledge_id=covered_knowledge_id)
    return BuiltContext(_build(CatchupRequest, data, "catch_up"), report)


def build_conflict_request(*, project: Project, requested_by) -> BuiltContext[ConflictRequest]:
    require_active_founder(project, requested_by)
    knowledge, omitted = balanced_knowledge(project)
    data = {
        "request_id": new_request_id("conflicts"),
        "project_id": contract_id(project.pk),
        "knowledge": [knowledge_dict(k) for k in knowledge],
    }
    return BuiltContext(_build(ConflictRequest, data, "detect_conflicts"), ContextReport(omitted={"knowledge": omitted}))


def build_proposal_request(*, project: Project, requested_by, topic: str,
                           knowledge_ids: Optional[Iterable[str]] = None,
                           conflict_ids: Optional[tuple[str, str]] = None) -> BuiltContext[ProposalRequest]:
    """
    ``knowledge_ids`` / ``conflict_ids`` are client input: they are resolved only within this project,
    and unknown IDs are rejected rather than ignored. Without ``knowledge_ids`` the most recent items are used;
    conflict items are always included.
    """
    require_active_founder(project, requested_by)
    topic = (topic or "").strip()
    if not topic or len(topic) > MAX_TOPIC_CHARS:
        raise ValidationError({"topic": f"Topic must be 1-{MAX_TOPIC_CHARS} characters."})

    def resolve(ids: Iterable[str], what: str) -> list[SharedKnowledge]:
        ids = list(dict.fromkeys(str(i) for i in ids))
        try:
            found = {str(k.pk): k for k in project.knowledge.filter(pk__in=ids)}
        except ValidationError:
            found = {}
        if len(found) != len(ids):
            raise ValidationError({what: "Contains knowledge that does not exist in this project."})
        return [found[i] for i in ids]

    conflict_items: list[SharedKnowledge] = []
    if conflict_ids is not None:
        if len(conflict_ids) != 2 or str(conflict_ids[0]) == str(conflict_ids[1]):
            raise ValidationError({"conflict_ids": "Must be two different knowledge items."})
        conflict_items = resolve(conflict_ids, "conflict_ids")

    omitted = 0
    if knowledge_ids is not None:
        selected = resolve(knowledge_ids, "knowledge_ids")
        if len(selected) > MAX_KNOWLEDGE_ITEMS:
            raise ValidationError({"knowledge_ids": f"At most {MAX_KNOWLEDGE_ITEMS} items can be referenced."})
    else:
        selected, omitted = recent_knowledge(project)
    chosen = {k.pk: k for k in selected}
    for item in conflict_items:
        if item.pk not in chosen:
            if len(chosen) >= MAX_KNOWLEDGE_ITEMS:
                oldest = next(k for k in _sorted_chronologically(chosen.values())
                              if k.pk not in {c.pk for c in conflict_items})
                del chosen[oldest.pk]
                omitted += 1
            chosen[item.pk] = item

    committed, omitted_decisions = committed_state_dict(project)
    data = {
        "request_id": new_request_id("proposal"),
        "project_id": contract_id(project.pk),
        "requested_by": founder_ref(requested_by),
        "topic": topic,
        "knowledge": [knowledge_dict(k) for k in _sorted_chronologically(chosen.values())],
        "conflict": ({"claim_a_id": contract_id(conflict_items[0].pk), "claim_b_id": contract_id(conflict_items[1].pk)}
                     if conflict_items else None),
        "committed_state": committed,
    }
    report = ContextReport(omitted={"knowledge": omitted, "decisions": omitted_decisions})
    return BuiltContext(_build(ProposalRequest, data, "draft_proposal"), report)
