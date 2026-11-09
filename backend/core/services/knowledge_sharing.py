"""Validate extraction provenance and publish candidates only behind the safety gate."""

from __future__ import annotations

import hashlib

from django.core.exceptions import ValidationError
from django.db import transaction
from django.utils import timezone
from cofytra_ai.models import KnowledgeCandidate, MessageRole, normalize_text
from pydantic import ValidationError as PydanticValidationError

from core.models import (
    ExtractionJob,
    KnowledgeOrigin,
    KnowledgeSourceMessage,
    PrivateMessage,
    ProjectMembership,
    SharedKnowledge,
)


class AutoShareBlocked(Exception):
    """Raised when automatic sharing is attempted before every safety prerequisite is met."""


# Set by code only when the AI team's real sensitive-content guard is integrated and verified.
SENSITIVE_CONTENT_GUARD_INTEGRATED = False


def auto_share_status() -> tuple[bool, str]:
    if not SENSITIVE_CONTENT_GUARD_INTEGRATED:
        return False, "sensitive_content_guard_not_integrated"
    return True, "ok"


def ensure_auto_share_allowed() -> None:
    allowed, reason = auto_share_status()
    if not allowed:
        raise AutoShareBlocked(reason)


def _validated_candidate(raw) -> KnowledgeCandidate:
    try:
        data = raw.model_dump(mode="json") if isinstance(raw, KnowledgeCandidate) else raw
        return KnowledgeCandidate.model_validate(data)
    except PydanticValidationError as exc:
        raise ValidationError("The extracted candidate does not match the knowledge contract.") from exc


def validate_extracted_candidates(*, job: ExtractionJob, candidates: list[dict]):
    """Revalidate cached output, membership, source ownership, verbatim quotes, and category/status rules."""
    if (job.source_message.project_id != job.project_id or job.source_message.owner_id != job.founder_id
            or job.source_message.role != MessageRole.FOUNDER.value):
        raise ValidationError("The extraction job source is not owned by its founder in this project.")
    if not ProjectMembership.objects.filter(project=job.project, user=job.founder, is_active=True).exists():
        raise ValidationError("The source founder is no longer an active project member.")

    validated = [_validated_candidate(candidate) for candidate in candidates]
    source_ids = {source_id for candidate in validated for source_id in candidate.source_message_ids}
    sources = {
        str(message.pk): message
        for message in PrivateMessage.objects.filter(
            pk__in=source_ids,
            project=job.project,
            owner=job.founder,
            role=MessageRole.FOUNDER.value,
        )
    }

    for candidate in validated:
        if any(source_id not in sources for source_id in candidate.source_message_ids):
            raise ValidationError("A candidate cites a message outside its founder's private project chat.")
        if not any(
            normalize_text(candidate.source_quote) in normalize_text(sources[source_id].content)
            for source_id in candidate.source_message_ids
        ):
            raise ValidationError("A candidate's source quote is not verbatim in its cited founder messages.")

    duplicates = []
    for candidate in validated:
        duplicate = None
        if candidate.duplicate_of_knowledge_id is not None:
            duplicate = SharedKnowledge.objects.filter(
                project=job.project, pk=candidate.duplicate_of_knowledge_id
            ).first()
            if duplicate is None:
                raise ValidationError("A candidate refers to knowledge outside the current project context.")
        duplicates.append(duplicate)
    return validated, sources, duplicates


def publish_extracted_candidates(*, job: ExtractionJob, candidates: list[dict]) -> list[SharedKnowledge]:
    """Atomically persist validated claims and private provenance after the safety gate permits publication."""
    with transaction.atomic():
        membership = ProjectMembership.objects.select_for_update().filter(
            project=job.project,
            user=job.founder,
            is_active=True,
        ).first()
        if membership is None:
            raise ValidationError("The source founder is no longer an active project member.")
        validated, sources, duplicates = validate_extracted_candidates(job=job, candidates=candidates)
        ensure_auto_share_allowed()

        published = []
        for index, candidate in enumerate(validated):
            key = hashlib.sha256(f"{job.pk}:{index}".encode("utf-8")).hexdigest()
            knowledge, created = SharedKnowledge.objects.get_or_create(
                extraction_key=key,
                defaults={
                    "project": job.project,
                    "author": job.founder,
                    "category": candidate.category.value,
                    "statement": candidate.statement,
                    "epistemic_status": candidate.epistemic_status.value,
                    "source_quote": candidate.source_quote,
                    "external_reference": candidate.external_reference,
                    "extraction_confidence": candidate.confidence.value,
                    "origin": KnowledgeOrigin.EXTRACTED,
                    "duplicate_of": duplicates[index],
                    "shared_at": timezone.now(),
                },
            )
            if not created and (
                knowledge.project_id != job.project_id
                or knowledge.author_id != job.founder_id
                or knowledge.category != candidate.category.value
                or knowledge.statement != candidate.statement
                or knowledge.epistemic_status != candidate.epistemic_status.value
                or knowledge.source_quote != candidate.source_quote
                or knowledge.external_reference != candidate.external_reference
                or knowledge.extraction_confidence != candidate.confidence.value
                or knowledge.duplicate_of_id != (duplicates[index].pk if duplicates[index] else None)
            ):
                raise ValidationError("A prior retry produced inconsistent data for this extraction candidate.")
            if not created:
                existing_sources = set(knowledge.internal_source_links.values_list("message_id", flat=True))
                expected_sources = {sources[source_id].pk for source_id in candidate.source_message_ids}
                if existing_sources != expected_sources:
                    raise ValidationError("A prior retry produced inconsistent provenance for this extraction candidate.")
            for source_id in candidate.source_message_ids:
                KnowledgeSourceMessage.objects.get_or_create(
                    knowledge=knowledge,
                    message=sources[source_id],
                )
            published.append(knowledge)
        return published
