"""
Cofytra AI LLM invocation layer.

One function, ``call_structured``, makes a structured-output LLM call through the existing Startary
provider (``startary.llm.provider.get_llm_client``) with Cofytra's bounded recovery policy:

  * at most TWO provider attempts in total per call;
  * malformed structured output   -> one retry with a schema-correction message;
  * transient provider error       -> one retry, unchanged;
  * token budget exhausted         -> one retry with an expanded completion budget;
  * permanent auth/config error    -> no retry;
  * anything else                  -> no retry.

Every failure is raised as ``CofytraAIError``. Output is returned only after full validation against
the schema, plus an optional caller-supplied check; partial output is never returned.

Privacy: logs and errors carry only capability, attempt, failure kind and exception type name. They
never carry prompts, founder text, model output, or API keys. LangSmith metadata goes through
``startary.observability.langsmith.sanitize_metadata``.

Imports: ``startary.config``, ``startary.llm.provider`` and ``startary.observability.langsmith`` only.
None of them loads LangGraph or the Startary workflow.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Generic, Literal, Optional, TypeVar

from pydantic import BaseModel, ValidationError

from cofytra_ai.errors import CofytraAIError
from cofytra_ai.models import CONTRACT_VERSION, ErrorCode, ResponseMeta
from startary.config import Settings
from startary.llm.provider import (
    LLMFailureKind,
    classify_llm_failure,
    get_llm_client,
    get_llm_client_resolution_info,
    is_valid_api_key,
    schema_correction_message,
)
from startary.observability.langsmith import get_langsmith_tracer, sanitize_metadata

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)
Tier = Literal["fast", "reasoning"]
LLMFactory = Callable[..., Any]  # call shape of startary.llm.provider.get_llm_client
OutputCheck = Callable[[Any], list[str]]  # returns rule violations; empty list = OK

MAX_ATTEMPTS = 2
DEFAULT_MAX_COMPLETION_TOKENS = 1500

_INVALID_OUTPUT_TYPE_NAMES = frozenset({"ValidationError", "OutputParserException", "JSONDecodeError", "MalformedOutputError"})


class MalformedOutputError(ValueError):
    """The model returned something that is not a valid instance of the requested schema."""

    def __init__(self, reason: str, violation_count: int = 0):
        super().__init__(reason)
        self.violation_count = violation_count


@dataclass(frozen=True)
class StructuredResult(Generic[T]):
    """A fully validated structured output plus the call metadata required by ``ResponseMeta``."""

    output: T
    model: str
    tier: Tier
    latency_ms: int
    attempts: int
    prompt_version: str

    def meta(self, request_id: str) -> ResponseMeta:
        return ResponseMeta(
            request_id=request_id,
            contract_version=CONTRACT_VERSION,
            prompt_version=self.prompt_version,
            model=self.model,
            tier=self.tier,
            latency_ms=self.latency_ms,
            attempts=self.attempts,
        )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_settings() -> Settings:
    """
    Load configuration from process environment variables only (D7).

    Any ``.env`` file is ignored, so the result does not depend on the working directory Django
    was started from. Variable names: see the contract doc, decision D7.
    """
    return Settings(_env_file=None)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Tracing
# ---------------------------------------------------------------------------

def build_run_config(capability: str, prompt_version: str, tier: str, attempt: int,
                     request_id: Optional[str] = None, retry_reason: Optional[str] = None,
                     callbacks: Optional[list[Any]] = None) -> dict[str, Any]:
    """
    LangChain run config for one attempt. Metadata contains operational values only and is passed
    through ``sanitize_metadata``. Note its forbidden-substring filter: no key may contain
    'prompt', 'message', 'input', 'context' or 'key', so the prompt version is sent as 'template_version'.
    """
    metadata = sanitize_metadata({
        "app": "cofytra",
        "capability": capability,
        "template_version": prompt_version,
        "contract_version": CONTRACT_VERSION,
        "model_tier": tier,
        "attempt": attempt,
        "request_id": request_id,
        "retry_reason": retry_reason,
    })
    config: dict[str, Any] = {
        "run_name": f"cofytra.{capability}:attempt_{attempt}",
        "metadata": metadata,
        "tags": ["cofytra", f"capability:{capability}", f"tier:{tier}", f"attempt:{attempt}"],
    }
    if callbacks:
        config["callbacks"] = list(callbacks)
    return config


def _tracing_callbacks(settings: Settings) -> list[Any]:
    """LangSmith tracer only when explicitly enabled; otherwise nothing (and no environment writes)."""
    if not getattr(settings, "langsmith_tracing", False):
        return []
    tracer = get_langsmith_tracer(settings)
    return [tracer] if tracer is not None else []


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------

def _failure_kind(exc: Exception) -> str:
    # Checked first: a validation error's text can contain model output (e.g. "403"), which the
    # provider's string-based permanent-error check would otherwise misread as an auth failure.
    if isinstance(exc, (ValidationError, MalformedOutputError)) or type(exc).__name__ in _INVALID_OUTPUT_TYPE_NAMES:
        return LLMFailureKind.INVALID_STRUCTURED_OUTPUT
    return classify_llm_failure(exc)


_RETRYABLE_KINDS = frozenset({
    LLMFailureKind.INVALID_STRUCTURED_OUTPUT,
    LLMFailureKind.TRANSIENT_API_ERROR,
    LLMFailureKind.TOKEN_BUDGET_EXHAUSTED,
})

_ERROR_FOR_KIND: dict[str, tuple[ErrorCode, str]] = {
    LLMFailureKind.PERMANENT_API_ERROR: (ErrorCode.PROVIDER_CONFIG, "The language model provider rejected the credentials or model configuration."),
    LLMFailureKind.TRANSIENT_API_ERROR: (ErrorCode.PROVIDER_UNAVAILABLE, "The language model provider is temporarily unavailable."),
    LLMFailureKind.INVALID_STRUCTURED_OUTPUT: (ErrorCode.INVALID_MODEL_OUTPUT, "The language model returned output that does not match the contract."),
    LLMFailureKind.TOKEN_BUDGET_EXHAUSTED: (ErrorCode.TOKEN_BUDGET, "The language model ran out of output budget."),
    LLMFailureKind.UNKNOWN_ERROR: (ErrorCode.INTERNAL, "The language model call failed unexpectedly."),
}


def _to_error(kind: str, exc: Exception, attempts: int, request_id: Optional[str]) -> CofytraAIError:
    code, message = _ERROR_FOR_KIND.get(kind, _ERROR_FOR_KIND[LLMFailureKind.UNKNOWN_ERROR])
    return CofytraAIError(code, f"{message} ({type(exc).__name__}, {attempts} attempt(s))", request_id)


def _correction_message(exc: Exception) -> dict[str, str]:
    if isinstance(exc, MalformedOutputError) and exc.violation_count:
        return {
            "role": "user",
            "content": (
                f"Your previous response broke {exc.violation_count} output rule(s). Return ONE complete object that "
                "uses only IDs and quotes present in the input and satisfies every field constraint."
            ),
        }
    return schema_correction_message(exc)  # names failing fields only, never values


def _validated(schema: type[T], raw: Any, check: Optional[OutputCheck]) -> T:
    if raw is None:
        raise MalformedOutputError("model returned no structured output")
    if isinstance(raw, BaseModel):
        raw = raw.model_dump()
    elif not isinstance(raw, dict):
        raise MalformedOutputError(f"model returned {type(raw).__name__}, expected an object")
    output = schema.model_validate(raw)  # full re-validation, including extra="forbid"
    if check is not None:
        violations = check(output)
        if violations:
            raise MalformedOutputError("output violates contract rules", violation_count=len(violations))
    return output


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def call_structured(
    schema: type[T],
    *,
    system_prompt: str,
    user_prompt: str,
    capability: str,
    prompt_version: str,
    request_id: Optional[str] = None,
    tier: Tier = "fast",
    max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
    output_check: Optional[OutputCheck] = None,
    settings: Optional[Settings] = None,
    llm_factory: Optional[LLMFactory] = None,
) -> StructuredResult[T]:
    """
    Invoke the LLM once (twice at most) and return a fully validated ``schema`` instance.

    ``output_check`` runs after schema validation (for example a ``contract_violations`` wrapper).
    A non-empty result counts as malformed output and is retried once.
    ``llm_factory`` defaults to ``startary.llm.provider.get_llm_client``; tests inject a ScriptedLLM.
    Raises ``CofytraAIError`` on every failure.
    """
    if tier not in ("fast", "reasoning"):
        raise ValueError(f"tier must be 'fast' or 'reasoning', got {tier!r}")
    settings = settings if settings is not None else load_settings()
    factory = llm_factory if llm_factory is not None else get_llm_client

    if llm_factory is None and not is_valid_api_key(getattr(settings, "openai_api_key", None)):
        raise CofytraAIError(ErrorCode.PROVIDER_CONFIG, "OPENAI_API_KEY is not configured.", request_id)

    base_messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    messages = base_messages
    tokens = max_completion_tokens
    reasoning_effort: Optional[str] = None
    retry_reason: Optional[str] = None
    callbacks = _tracing_callbacks(settings)
    started = time.monotonic()

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            llm = factory(settings, tier=tier, max_completion_tokens=tokens, reasoning_effort=reasoning_effort)
            structured = llm.with_structured_output(schema)
            config = build_run_config(capability, prompt_version, tier, attempt, request_id, retry_reason, callbacks)
            raw = structured.invoke(messages, config=config)
            output = _validated(schema, raw, output_check)
        except CofytraAIError:
            raise
        except Exception as exc:
            kind = _failure_kind(exc)
            logger.warning("cofytra_ai.%s attempt %d/%d failed: kind=%s type=%s",
                           capability, attempt, MAX_ATTEMPTS, kind, type(exc).__name__)
            if kind not in _RETRYABLE_KINDS or attempt == MAX_ATTEMPTS:
                raise _to_error(kind, exc, attempt, request_id) from None
            retry_reason = kind.lower()
            if kind == LLMFailureKind.INVALID_STRUCTURED_OUTPUT:
                messages = base_messages + [_correction_message(exc)]
            elif kind == LLMFailureKind.TOKEN_BUDGET_EXHAUSTED:
                tokens = max(int(max_completion_tokens * 1.6), max_completion_tokens + 1500)
                reasoning_effort = "low"
                messages = base_messages
            else:  # transient: same request again
                messages = base_messages
            continue

        latency_ms = int((time.monotonic() - started) * 1000)
        resolved = get_llm_client_resolution_info(llm).get("resolved_model") or settings.get_model_for_tier(tier)
        logger.info("cofytra_ai.%s ok: tier=%s model=%s attempts=%d latency_ms=%d",
                    capability, tier, resolved, attempt, latency_ms)
        return StructuredResult(output=output, model=str(resolved)[:64], tier=tier, latency_ms=latency_ms,
                                attempts=attempt, prompt_version=prompt_version)

    raise AssertionError("unreachable")  # the loop always returns or raises
