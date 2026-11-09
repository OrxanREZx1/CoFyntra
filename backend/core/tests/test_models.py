"""
Model and constraint tests for STEP 2.

Two layers are tested separately:
  - ORM validation (save() -> full_clean() -> clean()) raises ValidationError;
  - database constraints, reached by bulk_create()/update() which bypass save(), raise IntegrityError.
Every IntegrityError check runs inside transaction.atomic() so PostgreSQL can roll back to a savepoint.
"""

from datetime import timedelta

import pytest
from django.core.exceptions import ValidationError
from django.db import IntegrityError, transaction
from django.db.models.deletion import RestrictedError
from django.utils import timezone

from cofytra_ai.models import ALLOWED_STATUSES, CANDIDATE_STATUSES, EpistemicStatus, KnowledgeCategory
from core.models import (
    CatchupCursor,
    CommittedDecision,
    CommittedState,
    DecisionProposal,
    KnowledgeSourceMessage,
    PrivateMessage,
    Project,
    ProjectMembership,
    ProposalApproval,
    ProposalVersion,
    ProposalVersionKnowledge,
    SharedKnowledge,
)

pytestmark = pytest.mark.django_db


def db_rejects(callable_):
    with pytest.raises(IntegrityError), transaction.atomic():
        callable_()


def new_proposal(project, creator, base=0):
    return DecisionProposal.objects.create(project=project, created_by=creator, base_state_revision=base)


def new_version(proposal, creator, **overrides):
    fields = {
        "title": "Run a one-week clinic vs. lab comparison",
        "rationale": "Both segment choices are still hypotheses.",
        "trade_offs": ["Delays the decision by a week."],
        "next_action": "Each founder runs five interviews.",
        "proposed_changes": {"target_customer": "Independent clinics"},
    }
    fields.update(overrides)
    return ProposalVersion.create_next(proposal=proposal, created_by=creator, **fields)


# 1. Project and membership creation -------------------------------------------------------------

def test_project_with_exactly_two_founders(project, ana, ben):
    memberships = list(project.memberships.order_by("seat"))
    assert [(m.user, m.seat, m.role, m.is_active) for m in memberships] == [
        (ana, 1, "founder", True), (ben, 2, "founder", True)]
    assert project.created_at is not None and timezone.is_aware(project.created_at)


def test_third_active_founder_is_rejected(project, cara):
    with pytest.raises(ValidationError):
        project.add_founder(cara)
    with pytest.raises(ValidationError):
        ProjectMembership.objects.create(project=project, user=cara, seat=1)


def test_database_enforces_two_active_seats(project, cara):
    db_rejects(lambda: ProjectMembership.objects.bulk_create([ProjectMembership(project=project, user=cara, seat=1)]))
    db_rejects(lambda: ProjectMembership.objects.bulk_create([ProjectMembership(project=project, user=cara, seat=3)]))


def test_deactivated_founder_frees_a_seat(project, ben, cara):
    membership = project.memberships.get(user=ben)
    membership.is_active = False
    membership.save()
    assert project.add_founder(cara).seat == 2


def test_blank_project_title_is_rejected():
    with pytest.raises(ValidationError):
        Project.objects.create(title="   ")
    db_rejects(lambda: Project.objects.bulk_create([Project(title="")]))


# 2. Duplicate membership rejection --------------------------------------------------------------

def test_duplicate_membership_is_rejected(ana):
    solo = Project.objects.create(title="Solo")
    solo.add_founder(ana)
    with pytest.raises(ValidationError):
        solo.add_founder(ana)
    db_rejects(lambda: ProjectMembership.objects.create(project=solo, user=ana, seat=2))


# 3. Private message persistence -----------------------------------------------------------------

def test_private_messages_persist_in_order_with_contract_identity(project, ana, make_message):
    question = make_message(project, ana, "Should we focus on clinics?", client_message_id="c-1")
    reply = make_message(project, ana, "Compare both segments first.", role="assistant", reply_to=question)
    thread = list(PrivateMessage.objects.filter(project=project, owner=ana))
    assert thread == [question, reply]
    assert question.contract_author_id == str(ana.pk) and reply.contract_author_id is None


def test_client_message_id_is_idempotent_per_founder(project, ana, ben, make_message):
    make_message(project, ana, client_message_id="c-1")
    make_message(project, ben, client_message_id="c-1")  # another founder's key space
    db_rejects(lambda: make_message(project, ana, "retry", client_message_id="c-1"))


def test_one_assistant_reply_per_founder_message(project, ana, make_message):
    question = make_message(project, ana)
    make_message(project, ana, "first reply", role="assistant", reply_to=question)
    db_rejects(lambda: make_message(project, ana, "second reply", role="assistant", reply_to=question))


def test_whitespace_only_text_is_rejected_everywhere(project, ana, make_knowledge):
    with pytest.raises(ValidationError):
        make_knowledge(project, ana, statement="  \n ")
    with pytest.raises(ValidationError):
        new_version(new_proposal(project, ana), ana, rationale="   ")


def test_message_content_rules(project, ana, make_message):
    with pytest.raises(ValidationError):
        make_message(project, ana, "   ")
    with pytest.raises(ValidationError):
        make_message(project, ana, "x" * 4001)
    assert make_message(project, ana, "y" * 6000, role="assistant")  # replies may be longer
    with pytest.raises(ValidationError):
        make_message(project, ana, "reply", role="assistant", client_message_id="c-9")


def test_private_messages_are_immutable(project, ana, make_message):
    message = make_message(project, ana)
    message.content = "edited"
    with pytest.raises(ValidationError):
        message.save()


# 4. Knowledge category / status constraints ---------------------------------------------------

@pytest.mark.parametrize("category", list(KnowledgeCategory))
@pytest.mark.parametrize("status", list(EpistemicStatus))
def test_knowledge_category_status_matrix(project, ana, make_knowledge, category, status):
    reference = "EU Diagnostics Industry Report 2025" if status == EpistemicStatus.EXTERNAL_SOURCE else None
    allowed = status in ALLOWED_STATUSES[category] and status in CANDIDATE_STATUSES
    create = lambda: make_knowledge(project, ana, category=category.value, epistemic_status=status.value,
                                    external_reference=reference)
    if allowed:
        assert create().pk is not None
    else:
        with pytest.raises(ValidationError):
            create()


def test_database_rejects_invalid_category_status(project, ana):
    item = SharedKnowledge(project=project, author=ana, category="assumption", statement="Labs integrate fast.",
                           epistemic_status="founder_observation", source_quote="q", origin="extracted")
    db_rejects(lambda: SharedKnowledge.objects.bulk_create([item]))


def test_external_reference_rules(project, ana, make_knowledge):
    with pytest.raises(ValidationError):
        make_knowledge(project, ana, category="market", epistemic_status="external_source")
    with pytest.raises(ValidationError):
        make_knowledge(project, ana, external_reference="Some report")
    no_ref = SharedKnowledge(project=project, author=ana, category="market", statement="12,000 labs",
                             epistemic_status="external_source", source_quote="q", origin="extracted")
    db_rejects(lambda: SharedKnowledge.objects.bulk_create([no_ref]))


def test_extracted_knowledge_needs_quote_and_is_never_ai_suggestion(project, ana, make_knowledge):
    with pytest.raises(ValidationError):
        make_knowledge(project, ana, source_quote=None)
    with pytest.raises(ValidationError):
        make_knowledge(project, ana, category="assumption", epistemic_status="ai_suggestion")
    ai = SharedKnowledge(project=project, author=ana, category="assumption", statement="s",
                         epistemic_status="ai_suggestion", source_quote="q", origin="extracted")
    db_rejects(lambda: SharedKnowledge.objects.bulk_create([ai]))


def test_conflicting_knowledge_is_preserved_not_merged(project, ana, ben, make_knowledge):
    labs = make_knowledge(project, ben, category="target_customer", epistemic_status="founder_hypothesis",
                          statement="Diagnostic labs are the best first customer.")
    clinics = make_knowledge(project, ana, category="target_customer", epistemic_status="founder_hypothesis",
                             statement="Independent clinics are the best first customer.")
    dup = make_knowledge(project, ana, category="target_customer", epistemic_status="founder_hypothesis",
                         statement="Clinics should come first.", duplicate_of=clinics)
    assert set(project.knowledge.all()) == {labs, clinics, dup}
    clinics.statement = "Overwritten"
    with pytest.raises(ValidationError):
        clinics.save()


# 5. Internal provenance relations -----------------------------------------------------------

def test_provenance_links_own_founder_messages_only(project, ana, ben, make_message, make_knowledge):
    own = make_message(project, ana)
    assistant = make_message(project, ana, "reply", role="assistant", reply_to=own)
    bens = make_message(project, ben)
    item = make_knowledge(project, ana)
    link = KnowledgeSourceMessage.objects.create(knowledge=item, message=own)
    assert list(item.internal_source_links.all()) == [link]
    for forbidden in (assistant, bens):
        with pytest.raises(ValidationError):
            KnowledgeSourceMessage.objects.create(knowledge=item, message=forbidden)
    db_rejects(lambda: KnowledgeSourceMessage.objects.bulk_create([KnowledgeSourceMessage(knowledge=item, message=own)]))


def test_shared_knowledge_has_no_direct_message_reference():
    related = {f.related_model for f in SharedKnowledge._meta.concrete_fields if f.is_relation}
    assert PrivateMessage not in related


def test_provenance_limit_and_restrict_on_message_delete(project, ana, make_message, make_knowledge):
    item = make_knowledge(project, ana)
    messages = [make_message(project, ana, f"message {i}") for i in range(6)]
    for message in messages[:5]:
        KnowledgeSourceMessage.objects.create(knowledge=item, message=message)
    with pytest.raises(ValidationError):
        KnowledgeSourceMessage.objects.create(knowledge=item, message=messages[5])
    with pytest.raises(RestrictedError):
        messages[0].delete()


def test_project_deletion_cascades_through_restricted_links(project, ana, make_message, make_knowledge):
    item = make_knowledge(project, ana)
    KnowledgeSourceMessage.objects.create(knowledge=item, message=make_message(project, ana))
    project.delete()
    assert not SharedKnowledge.objects.exists() and not PrivateMessage.objects.exists()


# 6. Proposal version uniqueness ---------------------------------------------------------------

def test_versions_are_numbered_hashed_and_immutable(project, ana, ben, make_knowledge):
    item = make_knowledge(project, ana)
    proposal = new_proposal(project, ana)
    v1 = new_version(proposal, ana, referenced_knowledge=[item])
    v2 = new_version(proposal, ben, title="Target clinics first")
    proposal.refresh_from_db()
    assert (v1.version_number, v2.version_number, proposal.current_version_number) == (1, 2, 2)
    assert v1.content_hash_matches() and v2.content_hash_matches() and v1.content_hash != v2.content_hash
    assert list(v1.referenced_knowledge.all()) == [item]
    v1.title = "changed"
    with pytest.raises(ValidationError):
        v1.save()


def test_duplicate_version_number_is_rejected(project, ana):
    proposal = new_proposal(project, ana)
    v1 = new_version(proposal, ana)
    with pytest.raises(ValidationError):
        ProposalVersion.objects.create(proposal=proposal, version_number=1, created_by=ana, title="t", rationale="r",
                                       trade_offs=["x"], next_action="n", content_hash=v1.content_hash)
    clone = ProposalVersion(proposal=proposal, version_number=1, created_by=ana, title="t", rationale="r",
                            trade_offs=["x"], next_action="n", content_hash=v1.content_hash)
    db_rejects(lambda: ProposalVersion.objects.bulk_create([clone]))


@pytest.mark.parametrize("changes", [{"stage": "seed"}, {"problem": ""}, {"market": 42}, ["problem"]])
def test_proposed_changes_are_limited_to_v1_fields(project, ana, changes):
    with pytest.raises(ValidationError):
        new_version(new_proposal(project, ana), ana, proposed_changes=changes)


@pytest.mark.parametrize("trade_offs", [[], ["a"] * 6, [""], "not a list"])
def test_trade_offs_shape(project, ana, trade_offs):
    with pytest.raises(ValidationError):
        new_version(new_proposal(project, ana), ana, trade_offs=trade_offs)


def test_tampered_reference_is_detected_by_hash(project, ana, make_knowledge):
    proposal = new_proposal(project, ana)
    version = new_version(proposal, ana)
    ProposalVersionKnowledge.objects.create(version=version, knowledge=make_knowledge(project, ana))
    assert not version.content_hash_matches()


# 7. Approval uniqueness -------------------------------------------------------------------------

def test_one_decision_per_founder_per_version(project, ana, ben):
    proposal = new_proposal(project, ana)
    v1 = new_version(proposal, ana)
    ProposalApproval.objects.create(proposal_version=v1, founder=ana, decision="approve", content_hash=v1.content_hash)
    ProposalApproval.objects.create(proposal_version=v1, founder=ben, decision="request_changes",
                                    content_hash=v1.content_hash)
    db_rejects(lambda: ProposalApproval.objects.create(proposal_version=v1, founder=ana, decision="request_changes",
                                                       content_hash=v1.content_hash))
    v2 = new_version(proposal, ben)
    assert ProposalApproval.objects.create(proposal_version=v2, founder=ana, decision="approve",
                                           content_hash=v2.content_hash)


def test_approval_must_match_exact_version_content(project, ana):
    proposal = new_proposal(project, ana)
    v1 = new_version(proposal, ana)
    v2 = new_version(proposal, ana, title="Different content")
    with pytest.raises(ValidationError):
        ProposalApproval.objects.create(proposal_version=v2, founder=ana, decision="approve",
                                        content_hash=v1.content_hash)


def test_references_freeze_after_a_decision(project, ana, make_knowledge):
    version = new_version(new_proposal(project, ana), ana)
    ProposalApproval.objects.create(proposal_version=version, founder=ana, decision="approve",
                                    content_hash=version.content_hash)
    with pytest.raises(ValidationError):
        ProposalVersionKnowledge.objects.create(version=version, knowledge=make_knowledge(project, ana))


# 8. Committed-state version structure ----------------------------------------------------------

def test_committed_state_revisions_are_contiguous_and_immutable(project):
    assert project.current_state_revision() == 0 and project.latest_committed_state() is None
    r1 = CommittedState.objects.create(project=project, revision=1, problem="Manual scheduling wastes hours.")
    with pytest.raises(ValidationError):
        CommittedState.objects.create(project=project, revision=3)
    r2 = CommittedState.objects.create(project=project, revision=2, problem=r1.problem, target_customer="Clinics",
                                       next_action="Interview five clinics.")
    assert project.current_state_revision() == 2 and project.latest_committed_state() == r2
    r2.market = "EU"
    with pytest.raises(ValidationError):
        r2.save()


def test_database_backstops_for_committed_state(project):
    CommittedState.objects.create(project=project, revision=1)
    db_rejects(lambda: CommittedState.objects.bulk_create([CommittedState(project=project, revision=1)]))
    db_rejects(lambda: CommittedState.objects.bulk_create([CommittedState(project=project, revision=0)]))


def test_proposal_version_produces_at_most_one_decision(project, ana):
    proposal = new_proposal(project, ana)
    version = new_version(proposal, ana)
    r1 = CommittedState.objects.create(project=project, revision=1)
    CommittedDecision.objects.create(project=project, source_proposal=proposal, source_proposal_version=version,
                                     committed_state=r1, title="Clinics first", summary="We start with clinics.")
    r2 = CommittedState.objects.create(project=project, revision=2)
    db_rejects(lambda: CommittedDecision.objects.create(
        project=project, source_proposal=proposal, source_proposal_version=version, committed_state=r2,
        title="Again", summary="Duplicate commit."))


def test_proposal_base_revision_cannot_be_ahead(project, ana):
    with pytest.raises(ValidationError):
        new_proposal(project, ana, base=1)


# 9. Catch-up cursor uniqueness -----------------------------------------------------------------

def test_catchup_cursor_unique_per_pair(project, ana, ben):
    now = timezone.now()
    cursor = CatchupCursor.objects.create(project=project, viewer=ana, other_founder=ben, last_caught_up_at=now)
    CatchupCursor.objects.create(project=project, viewer=ben, other_founder=ana, last_caught_up_at=now)
    db_rejects(lambda: CatchupCursor.objects.create(project=project, viewer=ana, other_founder=ben,
                                                    last_caught_up_at=now))
    cursor.last_caught_up_at = now + timedelta(minutes=5)
    cursor.save()
    cursor.last_caught_up_at = now
    with pytest.raises(ValidationError):
        cursor.save()


def test_catchup_cursor_viewer_must_differ(project, ana):
    with pytest.raises(ValidationError):
        CatchupCursor.objects.create(project=project, viewer=ana, other_founder=ana, last_caught_up_at=timezone.now())
    self_cursor = CatchupCursor(project=project, viewer=ana, other_founder=ana, last_caught_up_at=timezone.now())
    db_rejects(lambda: CatchupCursor.objects.bulk_create([self_cursor]))


# 10. Cross-project relationship validation -------------------------------------------------------

def test_non_members_cannot_own_project_records(project, cara, make_message, make_knowledge):
    with pytest.raises(ValidationError):
        make_message(project, cara)
    with pytest.raises(ValidationError):
        make_knowledge(project, cara)
    with pytest.raises(ValidationError):
        new_proposal(project, cara)
    with pytest.raises(ValidationError):
        CatchupCursor.objects.create(project=project, viewer=cara, other_founder=project.memberships.first().user,
                                     last_caught_up_at=timezone.now())


def test_inactive_founder_cannot_create_new_records(project, ben, make_message):
    membership = project.memberships.get(user=ben)
    membership.is_active = False
    membership.save()
    with pytest.raises(ValidationError):
        make_message(project, ben)


def test_cross_project_links_are_rejected(project, other_project, ana, cara, make_message, make_knowledge):
    foreign_message = make_message(other_project, cara)
    foreign_item = make_knowledge(other_project, cara)
    local_item = make_knowledge(project, ana)
    with pytest.raises(ValidationError):
        make_message(project, ana, "reply", role="assistant", reply_to=foreign_message)
    with pytest.raises(ValidationError):
        KnowledgeSourceMessage.objects.create(knowledge=local_item, message=foreign_message)
    with pytest.raises(ValidationError):
        make_knowledge(project, ana, duplicate_of=foreign_item)
    with pytest.raises(ValidationError):
        new_version(new_proposal(project, ana), ana, referenced_knowledge=[foreign_item])


def test_cross_project_approvals_and_decisions_are_rejected(project, other_project, ana, cara):
    version = new_version(new_proposal(project, ana), ana)
    with pytest.raises(ValidationError):
        ProposalApproval.objects.create(proposal_version=version, founder=cara, decision="approve",
                                        content_hash=version.content_hash)
    foreign_state = CommittedState.objects.create(project=other_project, revision=1)
    with pytest.raises(ValidationError):
        CommittedDecision.objects.create(project=other_project, source_proposal=version.proposal,
                                         source_proposal_version=version, committed_state=foreign_state,
                                         title="t", summary="s")
    local_state = CommittedState.objects.create(project=project, revision=1)
    with pytest.raises(ValidationError):
        CommittedDecision.objects.create(project=project, source_proposal=version.proposal,
                                         source_proposal_version=version, committed_state=foreign_state,
                                         title="t", summary="s")
    assert local_state.pk


def test_failed_version_creation_leaves_no_partial_rows(project, other_project, ana, cara, make_knowledge):
    proposal = new_proposal(project, ana)
    with pytest.raises(ValidationError):
        new_version(proposal, ana, referenced_knowledge=[make_knowledge(other_project, cara)])
    proposal.refresh_from_db()
    assert proposal.current_version_number == 0 and not proposal.versions.exists()
