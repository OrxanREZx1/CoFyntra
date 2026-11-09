"""PostgreSQL-backed, lease-based extraction job claiming and processing."""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from datetime import timedelta

from django.core.exceptions import PermissionDenied as DjangoPermissionDenied, ValidationError
from django.db import DatabaseError, transaction
from django.db.models import Q
from django.utils import timezone
from pydantic import ValidationError as PydanticValidationError

from core.models import ExtractionJob, ExtractionJobStatus
from core.services import ai_adapter, ai_context
from core.services.knowledge_sharing import (
    AutoShareBlocked,
    auto_share_status,
    publish_extracted_candidates,
    validate_extracted_candidates,
)

logger = logging.getLogger("cofytra.extraction")

MAX_ATTEMPTS = 5
DEFAULT_LEASE_TIMEOUT = timedelta(minutes=10)


@dataclass(frozen=True)
class JobResult:
    job_id: int
    status: str
    error_code: str = ""


def claim_next_job(*, lease_timeout: timedelta = DEFAULT_LEASE_TIMEOUT,
                   now=None) -> ExtractionJob | None:
    """Claim one job with SKIP LOCKED; the claim transaction ends before any AI call."""
    now = now or timezone.now()
    stale_before = now - lease_timeout

    with transaction.atomic():
        ExtractionJob.objects.filter(
            status=ExtractionJobStatus.PROCESSING,
            claimed_at__lt=stale_before,
            attempt_count__gte=MAX_ATTEMPTS,
        ).update(
            status=ExtractionJobStatus.FAILED,
            claimed_at=None,
            claim_token=None,
            last_error_code="lease_expired_max_attempts",
            retry_after=None,
            updated_at=now,
        )

        eligible = (
            Q(status=ExtractionJobStatus.PENDING)
            | Q(
                status=ExtractionJobStatus.FAILED,
                attempt_count__lt=MAX_ATTEMPTS,
            )
            & Q(retry_after__lte=now)
            | Q(
                status=ExtractionJobStatus.PROCESSING,
                claimed_at__lt=stale_before,
                attempt_count__lt=MAX_ATTEMPTS,
            )
        )
        if auto_share_status()[0]:
            eligible |= Q(status=ExtractionJobStatus.AWAITING_SAFETY_VALIDATION)

        job = (ExtractionJob.objects.select_for_update(skip_locked=True)
               .filter(eligible).order_by("created_at", "pk").first())
        if job is None:
            return None

        job.status = ExtractionJobStatus.PROCESSING
        job.attempt_count += 1
        job.claimed_at = now
        job.claim_token = uuid.uuid4()
        job.retry_after = None
        job.save(update_fields=[
            "status", "attempt_count", "claimed_at", "claim_token", "retry_after", "updated_at",
        ])
        return job


def _update_claimed(job: ExtractionJob, **fields) -> bool:
    fields["updated_at"] = timezone.now()
    return ExtractionJob.objects.filter(
        pk=job.pk,
        status=ExtractionJobStatus.PROCESSING,
        claim_token=job.claim_token,
    ).update(**fields) == 1


def _save_extraction_result(job: ExtractionJob, candidates: list[dict]) -> bool:
    return _update_claimed(
        job,
        extraction_result=candidates,
        extracted_at=timezone.now(),
    )


def _complete(job: ExtractionJob) -> bool:
    return _update_claimed(
        job,
        status=ExtractionJobStatus.COMPLETED,
        claimed_at=None,
        claim_token=None,
        last_error_code="",
        retry_after=None,
    )


def _await_safety(job: ExtractionJob, reason: str) -> bool:
    return _update_claimed(
        job,
        status=ExtractionJobStatus.AWAITING_SAFETY_VALIDATION,
        claimed_at=None,
        claim_token=None,
        last_error_code=reason,
        retry_after=None,
    )


def _fail(job: ExtractionJob, code: str, *, retryable: bool = True) -> bool:
    delay_seconds = min(5 * (2 ** max(job.attempt_count - 1, 0)), 300)
    return _update_claimed(
        job,
        status=ExtractionJobStatus.FAILED,
        claimed_at=None,
        claim_token=None,
        last_error_code=code[:64],
        retry_after=timezone.now() + timedelta(seconds=delay_seconds)
        if retryable and job.attempt_count < MAX_ATTEMPTS else None,
    )


def process_claimed_job(job: ExtractionJob) -> JobResult:
    """Run inference outside transactions and persist results before attempting publication."""
    lease_lost = JobResult(job.pk, "lease_lost", "claim_superseded")
    try:
        candidates = job.extraction_result
        if candidates is None:
            source = job.source_message
            built = ai_context.build_extract_request(
                project=job.project,
                founder=job.founder,
                source_message=source,
            )
            response = ai_adapter.extract_knowledge(built.request)
            candidates = [candidate.model_dump(mode="json") for candidate in response.candidates]
            if not _save_extraction_result(job, candidates):
                return JobResult(job.pk, "lease_lost", "claim_superseded")

        if not candidates:
            if not _complete(job):
                return lease_lost
            return JobResult(job.pk, ExtractionJobStatus.COMPLETED)

        validate_extracted_candidates(job=job, candidates=candidates)
        allowed, reason = auto_share_status()
        if not allowed:
            if not _await_safety(job, reason):
                return lease_lost
            return JobResult(job.pk, ExtractionJobStatus.AWAITING_SAFETY_VALIDATION, reason)

        publish_extracted_candidates(job=job, candidates=candidates)
        if not _complete(job):
            return lease_lost
        return JobResult(job.pk, ExtractionJobStatus.COMPLETED)
    except AutoShareBlocked as exc:
        if not _await_safety(job, str(exc)):
            return lease_lost
        return JobResult(job.pk, ExtractionJobStatus.AWAITING_SAFETY_VALIDATION, str(exc))
    except ai_adapter.AIAdapterError as exc:
        if not _fail(job, exc.code.value, retryable=exc.retryable):
            return lease_lost
        logger.warning("extraction_job_failed job_id=%s error_code=%s", job.pk, exc.code.value)
        return JobResult(job.pk, ExtractionJobStatus.FAILED, exc.code.value)
    except PydanticValidationError:
        if not _fail(job, "invalid_extraction_output", retryable=False):
            return lease_lost
        logger.warning("extraction_job_failed job_id=%s error_code=invalid_extraction_output", job.pk)
        return JobResult(job.pk, ExtractionJobStatus.FAILED, "invalid_extraction_output")
    except DjangoPermissionDenied:
        if not _fail(job, "source_access_denied", retryable=False):
            return lease_lost
        logger.warning("extraction_job_failed job_id=%s error_code=source_access_denied", job.pk)
        return JobResult(job.pk, ExtractionJobStatus.FAILED, "source_access_denied")
    except ValidationError:
        if not _fail(job, "invalid_candidate_or_provenance", retryable=False):
            return lease_lost
        logger.warning("extraction_job_failed job_id=%s error_code=invalid_candidate_or_provenance", job.pk)
        return JobResult(job.pk, ExtractionJobStatus.FAILED, "invalid_candidate_or_provenance")
    except DatabaseError:
        if not _fail(job, "knowledge_persistence_failed"):
            return lease_lost
        logger.error("extraction_job_failed job_id=%s error_code=knowledge_persistence_failed", job.pk)
        return JobResult(job.pk, ExtractionJobStatus.FAILED, "knowledge_persistence_failed")
    except Exception as exc:
        code = "worker_internal_error"
        if not _fail(job, code):
            return lease_lost
        logger.error("extraction_job_failed job_id=%s error_code=%s exception_type=%s",
                     job.pk, code, type(exc).__name__)
        return JobResult(job.pk, ExtractionJobStatus.FAILED, code)


def run_one_job(*, lease_timeout: timedelta = DEFAULT_LEASE_TIMEOUT) -> JobResult | None:
    started = time.monotonic()
    job = claim_next_job(lease_timeout=lease_timeout)
    if job is None:
        return None
    result = process_claimed_job(job)
    logger.info("extraction_job_finished job_id=%s status=%s duration_ms=%d",
                result.job_id, result.status, int((time.monotonic() - started) * 1000))
    return result
