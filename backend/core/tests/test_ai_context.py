"""Context builders: authorization, founder isolation, trimming policy and the reply-length policy."""

from datetime import timedelta

import pytest
from django.core.exceptions import PermissionDenied, ValidationError
from django.utils import timezone

from cofytra_ai.models import MAX_MESSAGE_CHARS, CatchupRequest, ChatRequest, ErrorCode
from core.models import CatchupCursor, DecisionProposal, PrivateMessage, Project, ProposalVersion
from core.services import ai_context as ctx
from core.services.ai_adapter import AIAdapterError

pytestmark = pytest.mark.django_db

T0 = timezone.now() - timedelta(days=2)


def at(minutes: float):
    return T0 + timedelta(minutes=minutes)


# Valid request construction and founder chat isolation ---------------------------------------

def test_chat_request_holds_only_the_founders_own_history(project, ana, ben, make_message):
    a1 = make_message(project, ana, "Ana first message", created_at=at(1))
    make_message(project, ben, "BEN-PRIVATE-TEXT", created_at=at(2))
    r1 = make_message(project, ana, "Assistant answer to Ana", role="assistant", reply_to=a1, created_at=at(3))
    current = make_message(project, ana, "Ana current message", created_at=at(4))
    make_message(project, ana, "Ana later message", created_at=at(5))

    request = ctx.build_chat_request(project=project, founder=ana, message=current).request

    assert isinstance(request, ChatRequest)
    assert request.request_id.startswith("chat-") and request.project_id == str(project.pk)
    assert request.founder.id == str(ana.pk) and request.founder.display_name == "ana"
    assert request.message == "Ana current message"
    assert [m.id for m in request.recent_messages] == [str(a1.pk), str(r1.pk)]
    assert [(m.role.value, m.author_id) for m in request.recent_messages] == [("founder", str(ana.pk)),
                                                                              ("assistant", None)]
    assert all(timezone.is_aware(m.created_at) for m in request.recent_messages)
    assert "BEN-PRIVATE-TEXT" not in request.model_dump_json()


def test_chat_builder_checks_membership_and_ownership_first(project, other_project, ana, ben, cara, make_message):
    bens = make_message(project, ben)
    with pytest.raises(PermissionDenied):
        ctx.build_chat_request(project=project, founder=ana, message=bens)
    with pytest.raises(PermissionDenied):
        ctx.build_chat_request(project=project, founder=cara, message=bens)
    reply = make_message(project, ana, "reply", role="assistant", reply_to=make_message(project, ana))
    with pytest.raises(PermissionDenied):
        ctx.build_chat_request(project=project, founder=ana, message=reply)
    caras = make_message(other_project, cara)
    with pytest.raises(PermissionDenied):
        ctx.build_chat_request(project=project, founder=ana, message=caras)


def test_extract_window_is_own_chat_up_to_the_source_message(project, ana, ben, make_message):
    first = make_message(project, ana, "I talked to 8 clinic managers.", created_at=at(1))
    make_message(project, ben, "Ben private", created_at=at(2))
    source = make_message(project, ana, "Most use paper calendars.", created_at=at(3))
    make_message(project, ana, "After the source", created_at=at(4))
    request = ctx.build_extract_request(project=project, founder=ana, source_message=source).request
    assert [m.id for m in request.messages] == [str(first.pk), str(source.pk)]
    assert request.request_id.startswith("extract-")


# Shared knowledge context ----------------------------------------------------------------------

def test_shared_knowledge_is_project_scoped_and_attributed(project, other_project, ana, ben, cara,
                                                           make_message, make_knowledge):
    mine = make_knowledge(project, ana, shared_at=at(1))
    theirs = make_knowledge(project, ben, category="market", epistemic_status="external_source",
                            statement="About 12,000 labs in the EU.",
                            external_reference="EU Diagnostics Industry Report 2025", shared_at=at(2))
    make_knowledge(other_project, cara, statement="Other startup's claim")
    current = make_message(project, ana)
    items = ctx.build_chat_request(project=project, founder=ana, message=current).request.shared_knowledge
    assert [(k.id, k.author_id) for k in items] == [(str(mine.pk), str(ana.pk)), (str(theirs.pk), str(ben.pk))]
    assert items[1].external_reference == "EU Diagnostics Industry Report 2025"
    assert items[1].epistemic_status.value == "external_source"


# Latest committed-state selection ----------------------------------------------------------------

def test_no_committed_state_is_null(project, ana, make_message):
    request = ctx.build_chat_request(project=project, founder=ana, message=make_message(project, ana)).request
    assert request.committed_state is None


def test_latest_committed_revision_is_selected(project, ana, ben, make_message, commit_revision):
    commit_revision(project, ana, 1, title="Start B2B", problem="Old problem statement.")
    second = commit_revision(project, ben, 2, title="Clinics first", problem="Clinics lose hours to scheduling.",
                             target_customer="Independent clinics", next_action="Interview five clinics.")
    state = ctx.build_chat_request(project=project, founder=ana, message=make_message(project, ana)).request.committed_state
    assert state.version == 2
    assert (state.problem, state.target_customer, state.market) == ("Clinics lose hours to scheduling.",
                                                                     "Independent clinics", None)
    assert [d.title for d in state.decisions] == ["Start B2B", "Clinics first"]
    assert state.decisions[-1].id == str(second.pk)
    assert "next_action" not in state.model_dump()  # not in the cofytra_ai contract yet


# Context-size trimming ---------------------------------------------------------------------------

def test_private_messages_keep_most_recent_twenty_oldest_first(project, ana, make_message):
    history = [make_message(project, ana, f"message {i}", created_at=at(i)) for i in range(25)]
    current = make_message(project, ana, "current", created_at=at(30))
    built = ctx.build_chat_request(project=project, founder=ana, message=current)
    assert [m.id for m in built.request.recent_messages] == [str(m.pk) for m in history[5:]]
    assert built.report.omitted["messages"] == 5


def test_knowledge_keeps_most_recent_sixty(project, ana, make_message, make_knowledge):
    items = [make_knowledge(project, ana, statement=f"claim {i}", shared_at=at(i)) for i in range(65)]
    built = ctx.build_chat_request(project=project, founder=ana, message=make_message(project, ana))
    assert [k.id for k in built.request.shared_knowledge] == [str(k.pk) for k in items[5:]]
    assert built.report.omitted["knowledge"] == 5


def test_decisions_keep_most_recent_thirty(project, ana, make_message, commit_revision):
    decisions = [commit_revision(project, ana, r, title=f"Decision {r}") for r in range(1, 36)]
    built = ctx.build_chat_request(project=project, founder=ana, message=make_message(project, ana))
    assert [d.id for d in built.request.committed_state.decisions] == [str(d.pk) for d in decisions[5:]]
    assert built.request.committed_state.version == 35 and built.report.omitted["decisions"] == 5


def test_pending_proposals_keep_most_recent_twenty(project, ana, ben):
    proposals = []
    for i in range(25):
        p = DecisionProposal.objects.create(project=project, created_by=ana, base_state_revision=0, created_at=at(i))
        ProposalVersion.create_next(proposal=p, created_by=ana, title=f"Proposal {i}", rationale="r",
                                    trade_offs=["t"], next_action="n")
        proposals.append(p)
    built = ctx.build_catchup_request(project=project, viewer=ben)
    assert [p.id for p in built.request.pending_proposals] == [str(p.pk) for p in proposals[5:]]
    assert built.request.pending_proposals[-1].title == "Proposal 24"
    assert built.report.omitted["pending_proposals"] == 5


def test_conflict_context_gives_each_founder_a_fair_share(project, ana, ben, make_knowledge):
    for i in range(70):
        make_knowledge(project, ana, statement=f"ana {i}", shared_at=at(i))
    for i in range(10):
        make_knowledge(project, ben, statement=f"ben {i}", shared_at=at(i))
    built = ctx.build_conflict_request(project=project, requested_by=ana)
    authors = [k.author_id for k in built.request.knowledge]
    assert len(authors) == 60 and authors.count(str(ben.pk)) == 10 and authors.count(str(ana.pk)) == 50
    assert built.report.omitted["knowledge"] == 20


# Reply-length mismatch policy --------------------------------------------------------------------

def test_long_assistant_reply_is_stored_in_full_and_shortened_only_in_context(project, ana, make_message):
    question = make_message(project, ana, "Tell me everything.", created_at=at(1))
    long_reply = " ".join(f"w{i}" for i in range(1500))[:6000].strip()
    reply = make_message(project, ana, long_reply, role="assistant", reply_to=question, created_at=at(2))
    current = make_message(project, ana, "Follow-up", created_at=at(3))
    assert len(long_reply) > MAX_MESSAGE_CHARS

    first = ctx.build_chat_request(project=project, founder=ana, message=current)
    second = ctx.build_chat_request(project=project, founder=ana, message=current)
    text = first.request.recent_messages[1].text
    assert len(text) <= MAX_MESSAGE_CHARS
    assert text.endswith(ctx.ASSISTANT_SHORTENED_MARKER.strip())
    assert long_reply.startswith(text[: -len(ctx.ASSISTANT_SHORTENED_MARKER)].rstrip())
    assert text == second.request.recent_messages[1].text  # deterministic
    assert first.report.shortened_message_ids == (str(reply.pk),)
    reply.refresh_from_db()
    assert reply.content == long_reply  # the stored message is untouched


def test_founder_messages_are_never_shortened(project, ana, make_message):
    oversized = PrivateMessage(project=project, owner=ana, role="founder", content="x" * 4500, created_at=at(1))
    PrivateMessage.objects.bulk_create([oversized])  # bypasses model validation on purpose
    current = make_message(project, ana, "current", created_at=at(2))
    with pytest.raises(ValidationError):
        ctx.build_chat_request(project=project, founder=ana, message=current)
    with pytest.raises(AIAdapterError) as exc:
        ctx.build_chat_request(project=project, founder=ana, message=PrivateMessage.objects.get(pk=oversized.pk))
    assert exc.value.code is ErrorCode.INVALID_REQUEST


# Catch-up: no private messages, cursor and paging ----------------------------------------------

def test_catchup_has_only_the_other_founders_shared_knowledge(project, ana, ben, make_message, make_knowledge):
    make_message(project, ana, "ANA-SECRET-MESSAGE")
    make_message(project, ben, "BEN-SECRET-MESSAGE")
    make_knowledge(project, ana, statement="Ana's own claim")
    bens = make_knowledge(project, ben, statement="Ben's shared claim")
    request = ctx.build_catchup_request(project=project, viewer=ana).request
    assert isinstance(request, CatchupRequest)
    assert request.viewer.id == str(ana.pk) and request.other_founder.id == str(ben.pk)
    assert [k.id for k in request.knowledge] == [str(bens.pk)]
    dumped = request.model_dump_json()
    assert "SECRET-MESSAGE" not in dumped and "Ana's own claim" not in dumped
    assert not any("message" in name for name in CatchupRequest.model_fields)


def test_catchup_pages_oldest_first_from_the_cursor(project, ana, ben, make_knowledge):
    items = [make_knowledge(project, ben, statement=f"ben {i}", shared_at=at(i)) for i in range(65)]
    first = ctx.build_catchup_request(project=project, viewer=ana)
    assert first.request.since is None
    assert [k.id for k in first.request.knowledge] == [str(k.pk) for k in items[:60]]
    assert first.report.catch_up_complete is False and first.report.catch_up_covered_until == items[59].shared_at

    CatchupCursor.objects.create(project=project, viewer=ana, other_founder=ben,
                                 last_caught_up_at=first.report.catch_up_covered_until,
                                 last_knowledge_id=first.report.catch_up_covered_knowledge_id)
    second = ctx.build_catchup_request(project=project, viewer=ana)
    assert second.request.since == items[59].shared_at
    assert [k.id for k in second.request.knowledge] == [str(k.pk) for k in items[60:]]
    assert second.report.catch_up_complete is True


def test_catchup_composite_cursor_pages_items_tied_at_the_page_boundary(project, ana, ben, make_knowledge):
    for i in range(59):
        make_knowledge(project, ben, statement=f"ben {i}", shared_at=at(i))
    tied = [make_knowledge(project, ben, statement=f"tied {i}", shared_at=at(100)) for i in range(2)]
    first = ctx.build_catchup_request(project=project, viewer=ana)
    assert len(first.request.knowledge) == 60 and first.report.catch_up_covered_until == at(100)
    first_tied_id = first.report.catch_up_covered_knowledge_id
    assert first_tied_id in {str(k.pk) for k in tied}
    CatchupCursor.objects.create(project=project, viewer=ana, other_founder=ben,
                                 last_caught_up_at=first.report.catch_up_covered_until,
                                 last_knowledge_id=first.report.catch_up_covered_knowledge_id)
    second = ctx.build_catchup_request(project=project, viewer=ana)
    assert len(second.request.knowledge) == 1
    assert second.request.knowledge[0].id in {str(k.pk) for k in tied}
    assert second.request.knowledge[0].id != first_tied_id


def test_legacy_timestamp_only_cursor_replays_ties_instead_of_skipping(project, ana, ben, make_knowledge):
    timestamp = at(100)
    tied = [
        make_knowledge(project, ben, statement=f"legacy tied {i}", shared_at=timestamp)
        for i in range(2)
    ]
    CatchupCursor.objects.create(
        project=project,
        viewer=ana,
        other_founder=ben,
        last_caught_up_at=timestamp,
    )

    built = ctx.build_catchup_request(project=project, viewer=ana)

    assert {item.id for item in built.request.knowledge} == {str(item.pk) for item in tied}


def test_catchup_needs_exactly_one_other_founder(ana):
    solo = Project.objects.create(title="Solo")
    solo.add_founder(ana)
    with pytest.raises(ValidationError):
        ctx.build_catchup_request(project=solo, viewer=ana)


# Proposal context: client-supplied IDs are resolved, never trusted ------------------------------

def test_proposal_rejects_foreign_or_malformed_knowledge_ids(project, other_project, ana, cara, make_knowledge):
    foreign = make_knowledge(other_project, cara)
    for ids in ([str(foreign.pk)], ["not-a-uuid"]):
        with pytest.raises(ValidationError):
            ctx.build_proposal_request(project=project, requested_by=ana, topic="Pick a segment", knowledge_ids=ids)


def test_proposal_always_includes_conflict_items(project, ana, ben, make_knowledge):
    old_conflict = make_knowledge(project, ben, category="target_customer", epistemic_status="founder_hypothesis",
                                  statement="Labs first.", shared_at=at(0))
    for i in range(65):
        make_knowledge(project, ana, statement=f"claim {i}", shared_at=at(i + 1))
    newest = project.knowledge.order_by("-shared_at").first()
    built = ctx.build_proposal_request(project=project, requested_by=ana, topic="Pick a segment",
                                       conflict_ids=(str(old_conflict.pk), str(newest.pk)))
    ids = [k.id for k in built.request.knowledge]
    assert len(ids) == 60 and str(old_conflict.pk) in ids
    assert built.request.conflict.claim_a_id == str(old_conflict.pk)


def test_proposal_topic_is_validated_not_truncated(project, ana):
    with pytest.raises(ValidationError):
        ctx.build_proposal_request(project=project, requested_by=ana, topic="t" * 301)
    with pytest.raises(PermissionDenied):
        ctx.build_proposal_request(project=project, requested_by=None, topic="ok")
