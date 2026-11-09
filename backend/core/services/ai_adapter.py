"""
Django's single boundary to the in-process ``cofytra_ai`` package.

Every capability goes through the same pipeline (``_invoke``):

  1. the request must be the capability's request model and is re-validated by Pydantic;
  2. the backend is called (production: ``cofytra_ai.service.<fn>(request)``, nothing else passed);
  3. the response must be the declared response model and is re-validated by Pydantic;
  4. ``contract_violations(request, response)`` must be empty;
  5. only then is the typed response returned to Django application code.

Any failure becomes an ``AIAdapterError`` carrying the contract ``ErrorCode``, the suggested HTTP
status and a fixed, safe message. Logs record capability, request ID, error code, counts and timing
only: never founder text, prompts, model output, exception messages or API keys.

Backends are replaceable: ``ServiceBackend`` (real cofytra_ai) or ``MockBackend`` (deterministic,
see ai_mock.py), selected by ``settings.COFYTRA_AI_MODE`` or injected with ``backend=`` in tests.
This module never calls extraction implicitly; ``extract_knowledge`` is its own operation.
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Protocol

from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from pydantic import BaseModel, ValidationError

from cofytra_ai import service as ai_service
from cofytra_ai.errors import CofytraAIError
from cofytra_ai.models import (
    CAPABILITIES,
    ERROR_RETRYABLE,
    SUGGESTED_HTTP_STATUS,
    CatchupRequest,
    CatchupResponse,
    ChatRequest,
    ChatResponse,
    ConflictRequest,
    ConflictResponse,
    ErrorCode,
    ExtractRequest,
    ExtractResponse,
    ProposalRequest,
    ProposalResponse,
    contract_violations,
)

logger = logging.getLogger("cofytra.ai_adapter")

AI_MODES = ("service", "mock")

# Fixed client-facing messages. AI-side messages are never forwarded.
SAFE_ERROR_MESSAGES: dict[ErrorCode, str] = {
    ErrorCode.INVALID_REQUEST: "The AI request could not be built from the project context.",
    ErrorCode.PROVIDER_CONFIG: "The AI service is not configured.",
    ErrorCode.PROVIDER_UNAVAILABLE: "The AI service is temporarily unavailable. Please try again.",
    ErrorCode.INVALID_MODEL_OUTPUT: "The AI service returned an invalid response. Please try again.",
    ErrorCode.TOKEN_BUDGET: "The AI service could not complete this request.",
    ErrorCode.NOT_IMPLEMENTED: "This AI capability is not available yet.",
    ErrorCode.INTERNAL: "An unexpected AI service error occurred.",
}


class AIAdapterError(Exception):
    """Safe, typed application error for any AI failure. ``str(exc)`` contains no founder content."""

    def __init__(self, capability: str, code: ErrorCode, request_id: Optional[str] = None):
        self.capability = capability
        self.code = code
        self.request_id = request_id
        self.retryable = ERROR_RETRYABLE[code]
        self.http_status = SUGGESTED_HTTP_STATUS[code]
        self.message = SAFE_ERROR_MESSAGES[code]
        super().__init__(f"{capability}: {code.value}")

    def public_payload(self) -> dict:
        return {"error": {"code": self.code.value, "message": self.message, "retryable": self.retryable,
                          "request_id": self.request_id}}


class AIBackend(Protocol):
    name: str

    def call(self, capability: str, request: BaseModel) -> BaseModel: ...


class ServiceBackend:
    """Calls the AI team's functions exactly as the contract specifies: ``fn(request)``, nothing else."""

    name = "service"

    def call(self, capability: str, request: BaseModel) -> BaseModel:
        return getattr(ai_service, capability)(request)


def get_backend() -> AIBackend:
    mode = getattr(settings, "COFYTRA_AI_MODE", "service")
    if mode == "service":
        return ServiceBackend()
    if mode == "mock":
        if not (settings.DEBUG or getattr(settings, "COFYTRA_AI_ALLOW_MOCK_WITHOUT_DEBUG", False)):
            raise ImproperlyConfigured("COFYTRA_AI_MODE=mock requires DJANGO_DEBUG or an explicit "
                                       "COFYTRA_AI_ALLOW_MOCK_WITHOUT_DEBUG opt-in.")
        from core.services.ai_mock import MockBackend

        return MockBackend()
    raise ImproperlyConfigured(f"COFYTRA_AI_MODE must be one of {AI_MODES}.")


def _fail(capability: str, code: ErrorCode, request_id: Optional[str], stage: str, **counts) -> AIAdapterError:
    extra = " ".join(f"{k}={v}" for k, v in counts.items())
    logger.warning("ai_call_failed capability=%s request_id=%s code=%s stage=%s %s",
                   capability, request_id, code.value, stage, extra)
    return AIAdapterError(capability, code, request_id)


def _invoke(capability: str, request: BaseModel, backend: Optional[AIBackend]) -> BaseModel:
    request_cls, response_cls = CAPABILITIES[capability]
    request_id = getattr(request, "request_id", None)
    if not isinstance(request, request_cls):
        raise _fail(capability, ErrorCode.INVALID_REQUEST, request_id, "request_type")
    try:
        validated_request = request_cls.model_validate(request.model_dump())
    except ValidationError as exc:
        raise _fail(capability, ErrorCode.INVALID_REQUEST, request_id, "request_validation",
                    issues=exc.error_count()) from None

    backend = backend or get_backend()
    started = time.monotonic()
    try:
        response = backend.call(capability, validated_request)
    except CofytraAIError as exc:
        raise _fail(capability, exc.code, request_id, "ai_service") from None
    except Exception as exc:  # never log the message: it may contain founder content
        raise _fail(capability, ErrorCode.INTERNAL, request_id, "ai_service", exception=type(exc).__name__) from None

    if not isinstance(response, response_cls):
        raise _fail(capability, ErrorCode.INVALID_MODEL_OUTPUT, request_id, "response_type")
    try:
        validated_response = response_cls.model_validate(response.model_dump())
    except ValidationError as exc:
        raise _fail(capability, ErrorCode.INVALID_MODEL_OUTPUT, request_id, "response_validation",
                    issues=exc.error_count()) from None
    violations = contract_violations(validated_request, validated_response)
    if violations:
        raise _fail(capability, ErrorCode.INVALID_MODEL_OUTPUT, request_id, "reference_integrity",
                    violations=len(violations))

    meta = validated_response.meta
    logger.info("ai_call_ok capability=%s request_id=%s backend=%s tier=%s attempts=%s duration_ms=%d",
                capability, request_id, backend.name, meta.tier, meta.attempts,
                int((time.monotonic() - started) * 1000))
    return validated_response


def chat(request: ChatRequest, *, backend: Optional[AIBackend] = None) -> ChatResponse:
    return _invoke("chat", request, backend)


def extract_knowledge(request: ExtractRequest, *, backend: Optional[AIBackend] = None) -> ExtractResponse:
    """Separate operation; never triggered by chat(). Its candidates are not shared here (see knowledge_sharing)."""
    return _invoke("extract_knowledge", request, backend)


def catch_up(request: CatchupRequest, *, backend: Optional[AIBackend] = None) -> CatchupResponse:
    return _invoke("catch_up", request, backend)


def detect_conflicts(request: ConflictRequest, *, backend: Optional[AIBackend] = None) -> ConflictResponse:
    return _invoke("detect_conflicts", request, backend)


def draft_proposal(request: ProposalRequest, *, backend: Optional[AIBackend] = None) -> ProposalResponse:
    return _invoke("draft_proposal", request, backend)
