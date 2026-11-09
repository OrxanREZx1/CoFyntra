"""
Cofytra persistence model (Django-owned; see docs/COFYTRA_AI_BACKEND_CONTRACT_V1.md).

Layers, mirroring the AI contract:
  PrivateMessage      one founder's private assistant chat; never visible to the other founder
  SharedKnowledge     knowledge shared with the whole project (D3: auto-shared from extraction)
  DecisionProposal    a proposal and its immutable ProposalVersions / ProposalApprovals
  CommittedState      append-only, team-approved startup state; one row per revision
  CommittedDecision   the decision a committed ProposalVersion produced

Vocabularies and size limits are imported from the AI-owned ``cofytra_ai.models`` so there is a
single source of truth; ``makemigrations --check`` fails if the contract changes them.

Integrity strategy:
  - Every save() runs field validation and clean() (``ValidatedModel``). clean() enforces the
    cross-table rules a database cannot express without composite foreign keys: every founder
    reference must be a member of the record's project, and every related record must belong
    to the same project.
  - Database constraints (unique, check, partial unique) are the final guard, including for
    bulk operations that bypass save().
  - History tables are append-only (``AppendOnlyModel``): updating a saved row raises.

Not implemented here (later steps): proposal approval/commit business logic and catch-up/conflict APIs.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from typing import Iterable, Optional

from django.conf import settings
from django.contrib.auth.models import AbstractUser
from django.core.exceptions import ValidationError
from django.db import models, transaction
from django.db.models import F, Max, Q
from django.utils import timezone

from cofytra_ai.models import (
    ALLOWED_STATUSES,
    CANDIDATE_STATUSES,
    MAX_DECISION_SUMMARY_CHARS,
    MAX_EXTERNAL_REFERENCE_CHARS,
    MAX_ID_CHARS,
    MAX_MESSAGE_CHARS,
    MAX_QUOTE_CHARS,
    MAX_REPLY_CHARS,
    MAX_SOURCES_PER_CANDIDATE,
    MAX_STATEMENT_CHARS,
    MAX_SUMMARY_CHARS,
    MAX_TITLE_CHARS,
    MAX_TRADE_OFFS,
    Confidence,
    EpistemicStatus,
    KnowledgeCategory,
    MessageRole,
)

MAX_ACTIVE_FOUNDERS = 2
FOUNDER_SEATS = (1, 2)
SHA256_HEX_CHARS = 64

# v1 committed-state fields a proposal may change (D9). Never derived from AI prose.
PROPOSABLE_STATE_FIELDS = ("problem", "target_customer", "market")


def _values(enum_cls) -> list[str]:
    return [member.value for member in enum_cls]


def _choices(enum_cls) -> list[tuple[str, str]]:
    return [(member.value, member.value.replace("_", " ").capitalize()) for member in enum_cls]


def _strip_or_none(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = value.strip()
    return value or None


def _strip_required(instance: models.Model, *fields: str) -> None:
    """Strip required text fields; whitespace-only values are rejected (field validation runs before clean())."""
    errors = {}
    for field in fields:
        value = (getattr(instance, field) or "").strip()
        setattr(instance, field, value)
        if not value:
            errors[field] = "This field cannot be blank."
    if errors:
        raise ValidationError(errors)


def _not_blank(field: str) -> Q:
    return ~Q(**{field: ""})


def _null_or_not_blank(field: str) -> Q:
    return Q(**{f"{field}__isnull": True}) | ~Q(**{field: ""})


def _require_membership(project_id, user_id, field: str, *, active: bool = True) -> None:
    if project_id is None or user_id is None:
        return
    memberships = ProjectMembership.objects.filter(project_id=project_id, user_id=user_id)
    if active:
        memberships = memberships.filter(is_active=True)
    if not memberships.exists():
        qualifier = "an active founder" if active else "a founder"
        raise ValidationError({field: f"must be {qualifier} of this project."})


class ValidatedModel(models.Model):
    """Runs field validation and clean() on every save(); DB constraints stay the final guard."""

    class Meta:
        abstract = True

    def save(self, *args, **kwargs):
        self.full_clean(validate_unique=False, validate_constraints=False)
        super().save(*args, **kwargs)


class AppendOnlyModel(ValidatedModel):
    """History rows: created once, never updated through the ORM."""

    class Meta:
        abstract = True

    def save(self, *args, **kwargs):
        if not self._state.adding:
            raise ValidationError(f"{type(self).__name__} records are immutable once created.")
        super().save(*args, **kwargs)


# ---------------------------------------------------------------------------
# Identity and projects
# ---------------------------------------------------------------------------

class User(AbstractUser):
    """Django's built-in user (no custom auth platform). Contract IDs are ``str(user.pk)``."""


class Project(ValidatedModel):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    title = models.CharField(max_length=MAX_TITLE_CHARS)
    created_at = models.DateTimeField(default=timezone.now, editable=False)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [models.CheckConstraint(condition=_not_blank("title"), name="project_title_not_blank")]

    def __str__(self) -> str:
        return self.title

    def clean(self) -> None:
        _strip_required(self, "title")

    def add_founder(self, user) -> "ProjectMembership":
        """Add an active founder in a free seat. The project row is locked to serialize concurrent joins."""
        with transaction.atomic():
            Project.objects.select_for_update().get(pk=self.pk)
            if self.memberships.filter(user=user).exists():
                raise ValidationError("This user already has a membership in this project.")
            used = set(self.memberships.filter(is_active=True).values_list("seat", flat=True))
            free = [seat for seat in FOUNDER_SEATS if seat not in used]
            if not free:
                raise ValidationError(f"A project can have at most {MAX_ACTIVE_FOUNDERS} active founders.")
            return ProjectMembership.objects.create(project=self, user=user, seat=free[0])

    def latest_committed_state(self) -> Optional["CommittedState"]:
        return self.committed_states.order_by("-revision").first()

    def current_state_revision(self) -> int:
        """0 when nothing has been committed yet (contract ``committed_state: null``)."""
        return self.committed_states.aggregate(top=Max("revision"))["top"] or 0


# Choice enums live at module level so model Meta classes can reference them in constraints.
class MembershipRole(models.TextChoices):
    FOUNDER = "founder", "Founder"


class PrivateMessageRole(models.TextChoices):
    FOUNDER = MessageRole.FOUNDER.value, "Founder"
    ASSISTANT = MessageRole.ASSISTANT.value, "Assistant"


class KnowledgeOrigin(models.TextChoices):
    EXTRACTED = "extracted", "Auto-shared from extract_knowledge (D3)"


class ProposalStatus(models.TextChoices):
    PENDING = "pending", "Awaiting approval"
    CHANGES_REQUESTED = "changes_requested", "Changes requested"
    COMMITTED = "committed", "Committed"
    WITHDRAWN = "withdrawn", "Withdrawn"
    STALE = "stale", "Stale (committed state moved on)"


class ApprovalDecision(models.TextChoices):
    APPROVE = "approve", "Approve"
    REQUEST_CHANGES = "request_changes", "Request changes"


class ProjectMembership(ValidatedModel):
    """
    A founder's membership. At most two active founders per project: enforced in clean() and
    ``Project.add_founder`` and, at the database level, by two seats with a partial unique index.
    """

    Role = MembershipRole

    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name="memberships")
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.PROTECT, related_name="project_memberships")
    role = models.CharField(max_length=16, choices=Role.choices, default=Role.FOUNDER)
    seat = models.PositiveSmallIntegerField(help_text="Founder seat 1 or 2; unique among active memberships.")
    is_active = models.BooleanField(default=True)
    joined_at = models.DateTimeField(default=timezone.now, editable=False)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["project", "user"], name="membership_unique_project_user"),
            models.UniqueConstraint(fields=["project", "seat"], condition=Q(is_active=True),
                                    name="membership_unique_active_seat"),
            models.CheckConstraint(condition=Q(seat__in=FOUNDER_SEATS), name="membership_seat_valid"),
            models.CheckConstraint(condition=Q(role__in=_values(MembershipRole)), name="membership_role_valid"),
        ]
        indexes = [models.Index(fields=["user", "is_active"], name="membership_user_active_idx")]

    def __str__(self) -> str:
        return f"{self.user_id} in {self.project_id} (seat {self.seat})"

    def clean(self) -> None:
        if self.is_active and self.project_id is not None:
            others = ProjectMembership.objects.filter(project_id=self.project_id, is_active=True)
            if self.pk is not None:
                others = others.exclude(pk=self.pk)
            if others.count() >= MAX_ACTIVE_FOUNDERS:
                raise ValidationError(f"A project can have at most {MAX_ACTIVE_FOUNDERS} active founders.")


# ---------------------------------------------------------------------------
# Private conversation (never shared)
# ---------------------------------------------------------------------------

class PrivateMessage(AppendOnlyModel):
    """
    One message in a founder's private assistant chat. Only ``owner`` may ever see it; it is never
    sent to catch_up, detect_conflicts or draft_proposal, and never exposed to the other founder.
    Contract mapping: ``author_id = str(owner_id)`` for founder messages, ``null`` for assistant ones.
    """

    Role = PrivateMessageRole

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name="private_messages")
    owner = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.PROTECT, related_name="private_messages",
                              help_text="The founder whose private chat this message belongs to.")
    role = models.CharField(max_length=16, choices=Role.choices)
    # Assistant replies may be up to MAX_REPLY_CHARS; founder messages are limited to MAX_MESSAGE_CHARS in clean().
    content = models.CharField(max_length=MAX_REPLY_CHARS)
    ai_response = models.JSONField(null=True, blank=True,
                                   help_text="Complete validated AI response for assistant messages; never shared.")
    client_message_id = models.CharField(max_length=MAX_ID_CHARS, null=True, blank=True,
                                         help_text="Client-supplied idempotency key for founder messages.")
    reply_to = models.ForeignKey("self", on_delete=models.RESTRICT, null=True, blank=True,
                                 related_name="assistant_replies",
                                 help_text="For assistant messages: the founder message this answers.")
    created_at = models.DateTimeField(default=timezone.now, editable=False)

    class Meta:
        ordering = ["created_at", "id"]
        indexes = [models.Index(fields=["project", "owner", "created_at"], name="pm_thread_created_idx")]
        constraints = [
            models.UniqueConstraint(fields=["project", "owner", "client_message_id"],
                                    condition=Q(client_message_id__isnull=False), name="pm_unique_client_message_id"),
            models.UniqueConstraint(fields=["reply_to"], condition=Q(reply_to__isnull=False),
                                    name="pm_one_reply_per_message"),
            models.CheckConstraint(condition=Q(role__in=_values(PrivateMessageRole)), name="pm_role_valid"),
            models.CheckConstraint(condition=_not_blank("content"), name="pm_content_not_blank"),
            models.CheckConstraint(condition=_null_or_not_blank("client_message_id"), name="pm_client_id_not_blank"),
            models.CheckConstraint(
                condition=(Q(role=MessageRole.FOUNDER.value, reply_to__isnull=True)
                           | Q(role=MessageRole.ASSISTANT.value, client_message_id__isnull=True)),
                name="pm_role_specific_fields",
            ),
            models.CheckConstraint(
                condition=(Q(role=MessageRole.FOUNDER.value, ai_response__isnull=True)
                           | Q(role=MessageRole.ASSISTANT.value)),
                name="pm_ai_response_role",
            ),
        ]

    @property
    def contract_author_id(self) -> Optional[str]:
        return str(self.owner_id) if self.role == MessageRole.FOUNDER.value else None

    def clean(self) -> None:
        _strip_required(self, "content")
        self.client_message_id = _strip_or_none(self.client_message_id)
        if self.role == MessageRole.FOUNDER.value:
            if len(self.content) > MAX_MESSAGE_CHARS:
                raise ValidationError({"content": f"Founder messages are limited to {MAX_MESSAGE_CHARS} characters."})
            if self.reply_to_id is not None:
                raise ValidationError({"reply_to": "Only assistant messages can reply to a message."})
            if self.ai_response is not None:
                raise ValidationError({"ai_response": "Only assistant messages can store an AI response."})
        elif self.role == MessageRole.ASSISTANT.value and self.client_message_id is not None:
            raise ValidationError({"client_message_id": "Only founder messages carry a client message ID."})
        _require_membership(self.project_id, self.owner_id, "owner")
        if self.reply_to_id is not None:
            parent = self.reply_to
            if (parent.project_id != self.project_id or parent.owner_id != self.owner_id
                    or parent.role != MessageRole.FOUNDER.value):
                raise ValidationError({"reply_to": "Must be a founder message in the same founder's chat and project."})


# ---------------------------------------------------------------------------
# Shared knowledge (D3: auto-shared, attributed, epistemically labelled)
# ---------------------------------------------------------------------------

def _category_status_condition() -> Q:
    condition = Q()
    for category, statuses in ALLOWED_STATUSES.items():
        condition |= Q(category=category.value, epistemic_status__in=sorted(s.value for s in statuses))
    return condition


_EXTERNAL = EpistemicStatus.EXTERNAL_SOURCE.value


class SharedKnowledge(AppendOnlyModel):
    """
    A claim shared with every member of the project.

    D3 (final): Django shares every candidate that passes contract validation, with no founder
    confirmation. ``source_quote`` is the short extracted excerpt the contract allows to be shared;
    it is NOT permission to expose the transcript. The private messages it came from are linked only
    through ``KnowledgeSourceMessage`` (internal provenance), which must never be serialized to the
    other founder. Items are append-only: conflicting claims are kept side by side, never merged.
    """

    Origin = KnowledgeOrigin

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name="knowledge")
    author = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.PROTECT, related_name="authored_knowledge")
    category = models.CharField(max_length=32, choices=_choices(KnowledgeCategory))
    statement = models.CharField(max_length=MAX_STATEMENT_CHARS)
    epistemic_status = models.CharField(max_length=32, choices=_choices(EpistemicStatus))
    source_quote = models.CharField(max_length=MAX_QUOTE_CHARS, null=True, blank=True)
    external_reference = models.CharField(max_length=MAX_EXTERNAL_REFERENCE_CHARS, null=True, blank=True)
    extraction_confidence = models.CharField(max_length=16, choices=_choices(Confidence), null=True, blank=True,
                                             help_text="AI confidence that the extraction is faithful, not that it is true.")
    extraction_key = models.CharField(max_length=64, null=True, blank=True, editable=False,
                                      help_text="Internal idempotency key for one extracted candidate.")
    origin = models.CharField(max_length=16, choices=Origin.choices)
    duplicate_of = models.ForeignKey("self", on_delete=models.SET_NULL, null=True, blank=True, related_name="+",
                                     help_text="AI hint that this restates an existing item. Both are kept.")
    shared_at = models.DateTimeField(default=timezone.now, editable=False)

    class Meta:
        verbose_name_plural = "shared knowledge"
        ordering = ["shared_at", "id"]
        indexes = [
            models.Index(fields=["project", "shared_at"], name="kn_project_shared_idx"),
            models.Index(fields=["project", "author", "shared_at"], name="kn_project_author_shared_idx"),
        ]
        constraints = [
            models.UniqueConstraint(fields=["extraction_key"], condition=Q(extraction_key__isnull=False),
                                    name="kn_unique_extraction_key"),
            models.CheckConstraint(condition=_category_status_condition(), name="kn_category_status_allowed"),
            models.CheckConstraint(
                condition=(Q(epistemic_status=_EXTERNAL, external_reference__isnull=False)
                           | (~Q(epistemic_status=_EXTERNAL) & Q(external_reference__isnull=True))),
                name="kn_external_reference_rule",
            ),
            models.CheckConstraint(
                condition=~Q(origin="extracted")
                | Q(source_quote__isnull=False, epistemic_status__in=sorted(s.value for s in CANDIDATE_STATUSES)),
                name="kn_extracted_rules",
            ),
            models.CheckConstraint(condition=Q(origin__in=_values(KnowledgeOrigin)), name="kn_origin_valid"),
            models.CheckConstraint(
                condition=Q(extraction_confidence__isnull=True) | Q(extraction_confidence__in=_values(Confidence)),
                name="kn_confidence_valid",
            ),
            models.CheckConstraint(condition=_not_blank("statement"), name="kn_statement_not_blank"),
            models.CheckConstraint(condition=_null_or_not_blank("source_quote"), name="kn_quote_not_blank"),
            models.CheckConstraint(condition=_null_or_not_blank("external_reference"), name="kn_reference_not_blank"),
        ]

    def __str__(self) -> str:
        return f"[{self.category}/{self.epistemic_status}] {self.statement[:60]}"

    def clean(self) -> None:
        _strip_required(self, "statement")
        self.source_quote = _strip_or_none(self.source_quote)
        self.external_reference = _strip_or_none(self.external_reference)
        errors: dict[str, str] = {}
        try:
            category, status = KnowledgeCategory(self.category), EpistemicStatus(self.epistemic_status)
        except ValueError:
            category = status = None  # field validation reports invalid choices
        if category is not None and status not in ALLOWED_STATUSES[category]:
            errors["epistemic_status"] = f"{status.value!r} is not allowed for category {category.value!r}."
        if status == EpistemicStatus.EXTERNAL_SOURCE and not self.external_reference:
            errors["external_reference"] = "Required when epistemic_status is 'external_source'."
        if status is not None and status != EpistemicStatus.EXTERNAL_SOURCE and self.external_reference:
            errors["external_reference"] = "Only allowed when epistemic_status is 'external_source'."
        if self.origin == self.Origin.EXTRACTED:
            if status is not None and status not in CANDIDATE_STATUSES:
                errors["epistemic_status"] = "Extracted knowledge can never be an 'ai_suggestion'."
            if not self.source_quote:
                errors["source_quote"] = "Extracted knowledge must keep its source quote."
        if errors:
            raise ValidationError(errors)
        _require_membership(self.project_id, self.author_id, "author")
        if self.duplicate_of_id is not None and self.duplicate_of.project_id != self.project_id:
            raise ValidationError({"duplicate_of": "Must be knowledge from the same project."})


class KnowledgeSourceMessage(AppendOnlyModel):
    """
    INTERNAL provenance (D4): which of the author's own private messages a shared item came from.
    Never serialized to the other founder; never sent to catch_up / detect_conflicts / draft_proposal.
    """

    knowledge = models.ForeignKey(SharedKnowledge, on_delete=models.CASCADE, related_name="internal_source_links")
    message = models.ForeignKey(PrivateMessage, on_delete=models.RESTRICT, related_name="knowledge_source_links")
    created_at = models.DateTimeField(default=timezone.now, editable=False)

    class Meta:
        constraints = [models.UniqueConstraint(fields=["knowledge", "message"], name="ksm_unique_knowledge_message")]
        indexes = [models.Index(fields=["message"], name="ksm_message_idx")]

    def clean(self) -> None:
        if self.knowledge_id is None or self.message_id is None:
            return
        knowledge, message = self.knowledge, self.message
        if message.project_id != knowledge.project_id:
            raise ValidationError({"message": "Must belong to the same project as the knowledge item."})
        if message.role != MessageRole.FOUNDER.value or message.owner_id != knowledge.author_id:
            raise ValidationError({"message": "Must be a founder-authored message from the knowledge author's own chat."})
        if KnowledgeSourceMessage.objects.filter(knowledge_id=self.knowledge_id).count() >= MAX_SOURCES_PER_CANDIDATE:
            raise ValidationError(f"A knowledge item can cite at most {MAX_SOURCES_PER_CANDIDATE} source messages.")


class ExtractionJobStatus(models.TextChoices):
    PENDING = "pending", "Pending"
    PROCESSING = "processing", "Processing"
    AWAITING_SAFETY_VALIDATION = "awaiting_safety_validation", "Awaiting safety validation"
    COMPLETED = "completed", "Completed"
    FAILED = "failed", "Failed"


class ExtractionJob(ValidatedModel):
    """Durable extraction state for one founder message; candidate output is internal until safe publication."""

    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name="extraction_jobs")
    founder = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.PROTECT, related_name="extraction_jobs")
    source_message = models.OneToOneField(PrivateMessage, on_delete=models.CASCADE,
                                          related_name="extraction_job")
    status = models.CharField(max_length=32, choices=ExtractionJobStatus.choices,
                              default=ExtractionJobStatus.PENDING)
    attempt_count = models.PositiveSmallIntegerField(default=0)
    created_at = models.DateTimeField(default=timezone.now, editable=False)
    updated_at = models.DateTimeField(auto_now=True)
    claimed_at = models.DateTimeField(null=True, blank=True)
    claim_token = models.UUIDField(null=True, blank=True)
    retry_after = models.DateTimeField(null=True, blank=True)
    last_error_code = models.CharField(max_length=64, blank=True, default="")
    extraction_result = models.JSONField(null=True, blank=True,
                                         help_text="Validated candidate payloads retained for safe resumable publication.")
    extracted_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["created_at", "id"]
        indexes = [
            models.Index(fields=["status", "created_at"], name="extract_status_created_idx"),
            models.Index(fields=["status", "claimed_at"], name="extract_status_claimed_idx"),
        ]
        constraints = [
            models.CheckConstraint(condition=Q(status__in=[choice for choice, _ in ExtractionJobStatus.choices]),
                                   name="extract_status_valid"),
            models.CheckConstraint(condition=Q(attempt_count__gte=0), name="extract_attempt_nonnegative"),
            models.CheckConstraint(condition=~Q(status=ExtractionJobStatus.PROCESSING)
                                   | Q(claimed_at__isnull=False), name="extract_processing_has_lease"),
            models.CheckConstraint(condition=~Q(status=ExtractionJobStatus.PROCESSING)
                                   | Q(claim_token__isnull=False), name="extract_processing_has_token"),
        ]

    def clean(self) -> None:
        if self.source_message_id is None:
            return
        message = self.source_message
        if (message.project_id != self.project_id or message.owner_id != self.founder_id
                or message.role != PrivateMessageRole.FOUNDER):
            raise ValidationError({"source_message": "Must be a founder-authored message owned by this founder and project."})


# ---------------------------------------------------------------------------
# Proposals, immutable versions and approvals (D9: Django-owned)
# ---------------------------------------------------------------------------

class DecisionProposal(ValidatedModel):
    """A proposal to change committed state. Status transitions belong to the approval step (STEP 6)."""

    Status = ProposalStatus

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name="proposals")
    created_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.PROTECT, related_name="created_proposals")
    status = models.CharField(max_length=24, choices=Status.choices, default=Status.PENDING)
    current_version_number = models.PositiveIntegerField(default=0, help_text="0 until the first version exists.")
    base_state_revision = models.PositiveIntegerField(
        help_text="CommittedState revision this proposal was drafted against; 0 = nothing committed yet.")
    created_at = models.DateTimeField(default=timezone.now, editable=False)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        indexes = [models.Index(fields=["project", "status"], name="proposal_project_status_idx")]
        constraints = [models.CheckConstraint(condition=Q(status__in=_values(ProposalStatus)), name="proposal_status_valid")]

    def clean(self) -> None:
        if self._state.adding:
            _require_membership(self.project_id, self.created_by_id, "created_by")
            if self.project_id is not None and self.base_state_revision is not None:
                if self.base_state_revision > self.project.current_state_revision():
                    raise ValidationError({"base_state_revision": "Cannot be ahead of the project's committed state."})


class ProposalVersion(AppendOnlyModel):
    """
    An immutable, founder-confirmed version of a proposal. Approvals bind to its ``content_hash``.
    ``proposed_changes`` is the explicit, founder-confirmed state change; it is never parsed from AI prose.
    Create versions only through ``ProposalVersion.create_next``.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    proposal = models.ForeignKey(DecisionProposal, on_delete=models.CASCADE, related_name="versions")
    version_number = models.PositiveIntegerField()
    created_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.PROTECT, related_name="proposal_versions")
    title = models.CharField(max_length=MAX_TITLE_CHARS)
    rationale = models.CharField(max_length=MAX_SUMMARY_CHARS)
    trade_offs = models.JSONField(help_text=f"1-{MAX_TRADE_OFFS} non-empty strings.")
    next_action = models.CharField(max_length=MAX_STATEMENT_CHARS)
    proposed_changes = models.JSONField(default=dict, blank=True,
                                        help_text=f"Subset of {list(PROPOSABLE_STATE_FIELDS)} -> new text.")
    content_hash = models.CharField(max_length=SHA256_HEX_CHARS)
    referenced_knowledge = models.ManyToManyField(SharedKnowledge, through="ProposalVersionKnowledge",
                                                  related_name="proposal_versions")
    created_at = models.DateTimeField(default=timezone.now, editable=False)

    class Meta:
        ordering = ["proposal", "version_number"]
        constraints = [
            models.UniqueConstraint(fields=["proposal", "version_number"], name="pv_unique_proposal_version"),
            models.CheckConstraint(condition=Q(version_number__gte=1), name="pv_version_number_positive"),
            models.CheckConstraint(condition=_not_blank("title"), name="pv_title_not_blank"),
            models.CheckConstraint(condition=_not_blank("rationale"), name="pv_rationale_not_blank"),
            models.CheckConstraint(condition=_not_blank("next_action"), name="pv_next_action_not_blank"),
            models.CheckConstraint(condition=_not_blank("content_hash"), name="pv_content_hash_not_blank"),
        ]

    @staticmethod
    def compute_content_hash(*, proposal_id, version_number: int, base_state_revision: int, title: str,
                             rationale: str, trade_offs: list[str], next_action: str,
                             proposed_changes: dict[str, str], referenced_knowledge_ids: Iterable) -> str:
        payload = {
            "schema": "cofytra.proposal_version.v1",
            "proposal_id": str(proposal_id),
            "version_number": version_number,
            "base_state_revision": base_state_revision,
            "title": title,
            "rationale": rationale,
            "trade_offs": list(trade_offs),
            "next_action": next_action,
            "proposed_changes": dict(sorted(proposed_changes.items())),
            "referenced_knowledge_ids": sorted(str(i) for i in referenced_knowledge_ids),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _hash_from(self, referenced_knowledge_ids: Iterable) -> str:
        return self.compute_content_hash(
            proposal_id=self.proposal_id, version_number=self.version_number,
            base_state_revision=self.proposal.base_state_revision, title=self.title, rationale=self.rationale,
            trade_offs=self.trade_offs, next_action=self.next_action, proposed_changes=self.proposed_changes,
            referenced_knowledge_ids=referenced_knowledge_ids)

    def content_hash_matches(self) -> bool:
        """Recompute the hash from the stored rows (including references) and compare."""
        return self.content_hash == self._hash_from(self.knowledge_links.values_list("knowledge_id", flat=True))

    @classmethod
    def create_next(cls, *, proposal: DecisionProposal, created_by, title: str, rationale: str,
                    trade_offs: list[str], next_action: str, proposed_changes: Optional[dict[str, str]] = None,
                    referenced_knowledge: Iterable[SharedKnowledge] = ()) -> "ProposalVersion":
        """Create the next immutable version atomically, with its references and content hash."""
        knowledge = list(referenced_knowledge)
        if len({k.pk for k in knowledge}) != len(knowledge):
            raise ValidationError({"referenced_knowledge": "Duplicate knowledge references."})
        with transaction.atomic():
            locked = DecisionProposal.objects.select_for_update().get(pk=proposal.pk)
            version = cls(proposal=locked, version_number=locked.current_version_number + 1, created_by=created_by,
                          title=title, rationale=rationale, trade_offs=trade_offs, next_action=next_action,
                          proposed_changes=proposed_changes or {})
            version.full_clean(exclude=["content_hash"], validate_unique=False, validate_constraints=False)
            version.content_hash = version._hash_from(k.pk for k in knowledge)
            version.save()
            for item in knowledge:
                ProposalVersionKnowledge.objects.create(version=version, knowledge=item)
            locked.current_version_number = version.version_number
            locked.save(update_fields=["current_version_number", "updated_at"])
        proposal.current_version_number = version.version_number
        return version

    def clean(self) -> None:
        _strip_required(self, "title", "rationale", "next_action")
        self.trade_offs = self._clean_trade_offs(self.trade_offs)
        self.proposed_changes = self._clean_proposed_changes(self.proposed_changes)
        if self.content_hash and (len(self.content_hash) != SHA256_HEX_CHARS
                                  or any(c not in "0123456789abcdef" for c in self.content_hash)):
            raise ValidationError({"content_hash": "Must be a lowercase SHA-256 hex digest."})
        if self.proposal_id is None:
            return
        _require_membership(self.proposal.project_id, self.created_by_id, "created_by")
        current = DecisionProposal.objects.filter(pk=self.proposal_id).values_list(
            "current_version_number", flat=True).get()
        if self.version_number != current + 1:
            raise ValidationError({"version_number": f"The next version of this proposal must be {current + 1}."})

    @staticmethod
    def _clean_trade_offs(value) -> list[str]:
        if not isinstance(value, list) or not 1 <= len(value) <= MAX_TRADE_OFFS:
            raise ValidationError({"trade_offs": f"Must be a list of 1-{MAX_TRADE_OFFS} strings."})
        cleaned = []
        for item in value:
            if not isinstance(item, str) or not item.strip() or len(item.strip()) > MAX_STATEMENT_CHARS:
                raise ValidationError({"trade_offs": f"Each trade-off must be non-empty text of at most {MAX_STATEMENT_CHARS} characters."})
            cleaned.append(item.strip())
        return cleaned

    @staticmethod
    def _clean_proposed_changes(value) -> dict[str, str]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValidationError({"proposed_changes": "Must be an object."})
        unknown = sorted(set(value) - set(PROPOSABLE_STATE_FIELDS))
        if unknown:
            raise ValidationError({"proposed_changes": f"Unsupported fields: {', '.join(map(str, unknown))}. "
                                                       f"Allowed: {', '.join(PROPOSABLE_STATE_FIELDS)}."})
        cleaned = {}
        for key, text in value.items():
            if not isinstance(text, str) or not text.strip() or len(text.strip()) > MAX_STATEMENT_CHARS:
                raise ValidationError({"proposed_changes": f"{key!r} must be non-empty text of at most {MAX_STATEMENT_CHARS} characters."})
            cleaned[key] = text.strip()
        return cleaned


class ProposalVersionKnowledge(AppendOnlyModel):
    version = models.ForeignKey(ProposalVersion, on_delete=models.CASCADE, related_name="knowledge_links")
    knowledge = models.ForeignKey(SharedKnowledge, on_delete=models.RESTRICT, related_name="+")

    class Meta:
        constraints = [models.UniqueConstraint(fields=["version", "knowledge"], name="pvk_unique_version_knowledge")]

    def clean(self) -> None:
        if self.version_id is None or self.knowledge_id is None:
            return
        if self.knowledge.project_id != self.version.proposal.project_id:
            raise ValidationError({"knowledge": "Must be knowledge from the proposal's project."})
        if self.version.approvals.exists():
            raise ValidationError("References cannot change after a version has received a decision.")


class ProposalApproval(AppendOnlyModel):
    """One founder's decision on one exact proposal version (bound by content hash)."""

    Decision = ApprovalDecision

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    proposal_version = models.ForeignKey(ProposalVersion, on_delete=models.CASCADE, related_name="approvals")
    founder = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.PROTECT, related_name="proposal_decisions")
    decision = models.CharField(max_length=24, choices=Decision.choices)
    content_hash = models.CharField(max_length=SHA256_HEX_CHARS,
                                    help_text="Hash of the exact version content the founder decided on.")
    decided_at = models.DateTimeField(default=timezone.now, editable=False)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["proposal_version", "founder"], name="approval_unique_version_founder"),
            models.CheckConstraint(condition=Q(decision__in=_values(ApprovalDecision)), name="approval_decision_valid"),
        ]

    def clean(self) -> None:
        if self.proposal_version_id is None:
            return
        version = self.proposal_version
        _require_membership(version.proposal.project_id, self.founder_id, "founder")
        if self.content_hash != version.content_hash:
            raise ValidationError({"content_hash": "Does not match the proposal version's content."})


# ---------------------------------------------------------------------------
# Committed state (append-only revisions) and committed decisions
# ---------------------------------------------------------------------------

class CommittedState(AppendOnlyModel):
    """
    One immutable revision of the team-approved startup state. The current state is the highest
    revision; ``committed_state: null`` in the contract means no revision exists yet.

    STEP 6 commit protocol (not implemented here): inside one transaction, lock the Project row with
    ``select_for_update()``, re-check ``proposal.base_state_revision == current_state_revision()``,
    insert revision N+1 and its CommittedDecision. ``unique(project, revision)`` is the backstop
    against concurrent commits. Contract mapping: ``version = revision``; ``next_action`` is sent
    only once the AI team adds ``CommittedState.next_action`` to cofytra_ai.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name="committed_states")
    revision = models.PositiveIntegerField()
    problem = models.CharField(max_length=MAX_STATEMENT_CHARS, null=True, blank=True)
    target_customer = models.CharField(max_length=MAX_STATEMENT_CHARS, null=True, blank=True)
    market = models.CharField(max_length=MAX_STATEMENT_CHARS, null=True, blank=True)
    next_action = models.CharField(max_length=MAX_STATEMENT_CHARS, null=True, blank=True)
    created_at = models.DateTimeField(default=timezone.now, editable=False)

    class Meta:
        ordering = ["project", "revision"]
        constraints = [
            models.UniqueConstraint(fields=["project", "revision"], name="cs_unique_project_revision"),
            models.CheckConstraint(condition=Q(revision__gte=1), name="cs_revision_positive"),
            *[models.CheckConstraint(condition=_null_or_not_blank(f), name=f"cs_{f}_not_blank")
              for f in ("problem", "target_customer", "market", "next_action")],
        ]

    def clean(self) -> None:
        for field in ("problem", "target_customer", "market", "next_action"):
            setattr(self, field, _strip_or_none(getattr(self, field)))
        if self.project_id is not None and self.revision is not None:
            expected = self.project.current_state_revision() + 1
            if self.revision != expected:
                raise ValidationError({"revision": f"The next revision for this project must be {expected}."})


class CommittedDecision(AppendOnlyModel):
    """The decision produced by committing exactly one proposal version (and so one proposal)."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name="committed_decisions")
    source_proposal = models.OneToOneField(DecisionProposal, on_delete=models.RESTRICT, related_name="committed_decision")
    source_proposal_version = models.OneToOneField(ProposalVersion, on_delete=models.RESTRICT,
                                                   related_name="committed_decision")
    committed_state = models.OneToOneField(CommittedState, on_delete=models.RESTRICT, related_name="decision",
                                           help_text="The state revision this decision produced.")
    title = models.CharField(max_length=MAX_TITLE_CHARS)
    summary = models.CharField(max_length=MAX_DECISION_SUMMARY_CHARS)
    referenced_knowledge = models.ManyToManyField(SharedKnowledge, through="CommittedDecisionKnowledge",
                                                  related_name="committed_decisions")
    committed_at = models.DateTimeField(default=timezone.now, editable=False)

    class Meta:
        ordering = ["committed_at", "id"]
        indexes = [models.Index(fields=["project", "committed_at"], name="decision_project_committed_idx")]
        constraints = [
            models.CheckConstraint(condition=_not_blank("title"), name="decision_title_not_blank"),
            models.CheckConstraint(condition=_not_blank("summary"), name="decision_summary_not_blank"),
        ]

    def clean(self) -> None:
        _strip_required(self, "title", "summary")
        if self.source_proposal_version_id is not None:
            version = self.source_proposal_version
            if version.proposal_id != self.source_proposal_id:
                raise ValidationError({"source_proposal_version": "Must be a version of source_proposal."})
            if version.proposal.project_id != self.project_id:
                raise ValidationError({"source_proposal_version": "Must belong to the same project."})
        if self.committed_state_id is not None and self.committed_state.project_id != self.project_id:
            raise ValidationError({"committed_state": "Must belong to the same project."})


class CommittedDecisionKnowledge(AppendOnlyModel):
    decision = models.ForeignKey(CommittedDecision, on_delete=models.CASCADE, related_name="knowledge_links")
    knowledge = models.ForeignKey(SharedKnowledge, on_delete=models.RESTRICT, related_name="+")

    class Meta:
        constraints = [models.UniqueConstraint(fields=["decision", "knowledge"], name="cdk_unique_decision_knowledge")]

    def clean(self) -> None:
        if self.decision_id is not None and self.knowledge_id is not None:
            if self.knowledge.project_id != self.decision.project_id:
                raise ValidationError({"knowledge": "Must be knowledge from the decision's project."})


# ---------------------------------------------------------------------------
# Catch-up cursor (D11: per viewer / other-founder pair)
# ---------------------------------------------------------------------------

class CatchupCursor(ValidatedModel):
    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name="catchup_cursors")
    viewer = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.PROTECT, related_name="catchup_cursors")
    other_founder = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.PROTECT, related_name="+")
    last_caught_up_at = models.DateTimeField(help_text="Time of the last successful catch-up; sent as 'since'.")
    created_at = models.DateTimeField(default=timezone.now, editable=False)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["project", "viewer", "other_founder"], name="cursor_unique_pair"),
            models.CheckConstraint(condition=~Q(viewer=F("other_founder")), name="cursor_viewer_not_other"),
        ]

    def clean(self) -> None:
        if self.viewer_id is not None and self.viewer_id == self.other_founder_id:
            raise ValidationError({"other_founder": "Must be a different founder from the viewer."})
        _require_membership(self.project_id, self.viewer_id, "viewer", active=self._state.adding)
        _require_membership(self.project_id, self.other_founder_id, "other_founder", active=self._state.adding)
        if not self._state.adding and self.last_caught_up_at is not None:
            previous = CatchupCursor.objects.filter(pk=self.pk).values_list("last_caught_up_at", flat=True).first()
            if previous is not None and self.last_caught_up_at < previous:
                raise ValidationError({"last_caught_up_at": "Cannot move backwards."})
