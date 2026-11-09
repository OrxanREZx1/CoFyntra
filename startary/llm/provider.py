"""
Startary LLM Provider Boundary.

This is the SINGLE place in the codebase that imports from langchain_openai.
All other modules receive an LLM client object — they never import provider
libraries directly.

Configuration:
  OPENAI_API_KEY          — required for live AI execution (set in .env)
  STARTARY_MODEL_NAME     — default model to use (default: gpt-4o)
  FAST_MODEL_NAME         — model for fast/triage tier (default: gpt-4o-mini)
  REASONING_MODEL_NAME    — model for strategic reasoning tier (default: o3-mini)
  FAST_MODEL_FALLBACK     — optional fallback if fast model is inaccessible
  REASONING_MODEL_FALLBACK— optional fallback if reasoning model is inaccessible
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import TYPE_CHECKING, Any, Optional

from startary.config import Settings

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None  # type: ignore[assignment, misc]

logger = logging.getLogger(__name__)

PLACEHOLDER_KEYS = frozenset({
    "PASTE_YOUR_OPENAI_API_KEY_HERE",
    "sk-...",
    "",
})

# Runtime Model Access Cache (verified once per process)
_MODEL_ACCESS_CACHE: dict[str, dict[str, Any]] = {}


def register_model_access(model_name: str, available: bool, reason: str = "AVAILABLE") -> None:
    """Manually register model accessibility in the runtime cache (used in preflight and unit tests)."""
    _MODEL_ACCESS_CACHE[model_name] = {"available": available, "reason": reason}


def clear_model_access_cache() -> None:
    """Clear runtime model access cache (used in tests)."""
    _MODEL_ACCESS_CACHE.clear()


def get_cached_model_access(model_name: str) -> Optional[dict[str, Any]]:
    """Retrieve cached access status for a model."""
    return _MODEL_ACCESS_CACHE.get(model_name)


def is_valid_api_key(api_key: str | None) -> bool:
    """
    Return True if an API key is present and not a known template placeholder.
    Does not log or expose the key value.
    """
    if not api_key:
        return False
    stripped = api_key.strip()
    return stripped not in PLACEHOLDER_KEYS


class LLMFailureKind:
    """Milestone 6.2.5: retry-relevant classification of a failed structured LLM call."""
    TOKEN_BUDGET_EXHAUSTED = "TOKEN_BUDGET_EXHAUSTED"
    INVALID_STRUCTURED_OUTPUT = "INVALID_STRUCTURED_OUTPUT"
    TRANSIENT_API_ERROR = "TRANSIENT_API_ERROR"
    PERMANENT_API_ERROR = "PERMANENT_API_ERROR"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"


_INVALID_OUTPUT_TYPES = ("ValidationError", "OutputParserException", "JSONDecodeError")
_TRANSIENT_TYPES = (
    "APITimeoutError", "APIConnectionError", "RateLimitError", "InternalServerError",
    "ServiceUnavailableError", "Timeout", "TimeoutError", "ConnectionError", "ReadTimeout",
)


def classify_llm_failure(exc: Exception) -> str:
    """
    Classify a failed structured LLM call so nodes can apply the right bounded recovery:
    token exhaustion and permanent errors stop immediately (the token wrapper already spent its one
    expansion); invalid structured output is retried once with schema feedback; transient API errors
    are retried once unchanged.
    """
    if is_token_budget_exhausted_error(exc):
        return LLMFailureKind.TOKEN_BUDGET_EXHAUSTED
    if is_permanent_model_error(exc):
        return LLMFailureKind.PERMANENT_API_ERROR
    if type(exc).__name__ in _INVALID_OUTPUT_TYPES:
        return LLMFailureKind.INVALID_STRUCTURED_OUTPUT
    status_code = getattr(exc, "status_code", None)
    if type(exc).__name__ in _TRANSIENT_TYPES or status_code in (408, 409, 429) or (isinstance(status_code, int) and status_code >= 500):
        return LLMFailureKind.TRANSIENT_API_ERROR
    if any(t in str(exc).lower() for t in ("timed out", "timeout", "connection error", "temporarily unavailable")):
        return LLMFailureKind.TRANSIENT_API_ERROR
    return LLMFailureKind.UNKNOWN_ERROR


def schema_correction_message(exc: Exception) -> dict[str, str]:
    """
    Corrective follow-up for a retry after invalid structured output. Names only the failing schema
    fields (never echoes model output or founder text).
    """
    fields: list[str] = []
    errors = getattr(exc, "errors", None)
    if callable(errors):
        try:
            fields = sorted({".".join(str(p) for p in e.get("loc", ())) for e in errors() if e.get("loc")})
        except Exception:
            fields = []
    field_note = f" (failing fields: {', '.join(fields)})" if fields else ""
    return {
        "role": "user",
        "content": (
            f"Your previous response did not satisfy the required output schema{field_note}. "
            "Return ONE complete object in which every required field is present and meets its constraints."
        ),
    }


def is_permanent_model_error(exc: Exception) -> bool:
    """
    Determine if an exception represents a permanent configuration or access failure
    where repeating the exact same request cannot succeed (e.g. 403 model_not_found,
    project access denied, invalid model ID, 401 authentication).
    """
    if exc is None:
        return False

    status_code = getattr(exc, "status_code", None)
    if status_code in (401, 403, 404):
        return True

    exc_str = str(exc).lower()
    code = getattr(exc, "code", None) or ""
    if str(code).lower() == "model_not_found":
        return True

    permanent_indicators = [
        "model_not_found",
        "does not have access to model",
        "project does not have access",
        "project `",
        "project_has_no_access",
        "invalid model",
        "model not found",
        "model does not exist",
        "incorrect api key",
        "invalid api key",
        "account_deactivated",
        "401",
        "403",
        "404",
        "unauthorized",
        "forbidden",
    ]
    return any(indicator in exc_str for indicator in permanent_indicators)


def is_transient_error(exc: Exception) -> bool:
    """
    Determine if an exception represents a transient failure where a retry may succeed
    (e.g. request timeout, temporary rate limit 429, connection reset, 5xx server error).
    """
    if exc is None:
        return False

    exc_type_name = type(exc).__name__
    if exc_type_name in (
        "TimeoutError",
        "APITimeoutError",
        "TimeoutException",
        "ReadTimeout",
        "ConnectTimeout",
        "RateLimitError",
        "APIConnectionError",
        "ConnectionError",
        "ConnectError",
    ):
        return True

    status_code = getattr(exc, "status_code", None)
    if status_code in (429, 500, 502, 503, 504):
        return True

    exc_str = str(exc).lower()
    transient_indicators = [
        "timed out",
        "timeout",
        "rate limit",
        "429",
        "connection reset",
        "connection error",
        "500",
        "502",
        "503",
        "504",
        "service unavailable",
    ]
    return any(indicator in exc_str for indicator in transient_indicators)


def check_model_access(api_key: str, model_name: str) -> tuple[bool, str]:
    """
    Check if the current OpenAI project has verified access to model_name.
    Uses cached result if available.
    """
    if model_name in _MODEL_ACCESS_CACHE:
        cached = _MODEL_ACCESS_CACHE[model_name]
        return bool(cached.get("available", False)), str(cached.get("reason", "AVAILABLE"))

    if not is_valid_api_key(api_key):
        return False, "api_key_not_configured"

    from langchain_openai import ChatOpenAI  # noqa: PLC0415

    try:
        test_client = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            temperature=0,
            max_retries=0,
            timeout=10,
        )
        test_client.invoke([{"role": "user", "content": "ping"}])
        register_model_access(model_name, True, "AVAILABLE")
        return True, "AVAILABLE"
    except Exception as exc:
        if is_permanent_model_error(exc):
            reason = "model_not_found / project_access_denied"
            register_model_access(model_name, False, reason)
            return False, reason
        # Transient error during probe: do not permanently ban model
        logger.warning("Transient probe failure checking model %s: %s", model_name, exc)
        return False, f"transient_probe_error: {exc}"


def resolve_model_for_tier(settings: Settings, tier: str = "reasoning") -> tuple[str, str, bool, Optional[str]]:
    """
    Resolve model according to verified availability and safe fallback hierarchy:
    Returns: (configured_model, resolved_model, fallback_used, fallback_reason)

    FAST Tier Precedence:
      1. FAST_MODEL_NAME if access verified
      2. FAST_MODEL_FALLBACK if configured and verified
      3. STARTARY_MODEL_NAME if verified
      4. Fail clearly

    REASONING Tier Precedence:
      1. REASONING_MODEL_NAME if access verified
      2. REASONING_MODEL_FALLBACK if configured and verified
      3. STARTARY_MODEL_NAME if verified
      4. Fail clearly
    """
    tier_lower = tier.lower().strip()
    is_fast = tier_lower in ("fast", "lightweight")

    if is_fast:
        configured_model = settings.fast_model_name or "gpt-4o-mini"
        fallback_candidates = [
            (settings.fast_model_fallback, "FAST_MODEL_UNAVAILABLE_FALLBACK_SELECTED"),
            (settings.model_name, "FAST_MODEL_UNAVAILABLE_DEFAULT_SELECTED"),
        ]
    else:
        configured_model = settings.reasoning_model_name or settings.model_name
        fallback_candidates = [
            (settings.reasoning_model_fallback, "REASONING_MODEL_UNAVAILABLE_FALLBACK_SELECTED"),
            (settings.model_name, "REASONING_MODEL_UNAVAILABLE_DEFAULT_SELECTED"),
        ]

    # Check configured model access in cache
    cached_status = get_cached_model_access(configured_model)
    if cached_status is None or cached_status.get("available") is True:
        # Not known to be unavailable -> use configured
        return configured_model, configured_model, False, None

    # Configured model is verified UNAVAILABLE -> search fallbacks
    primary_reason = cached_status.get("reason", "model_access_denied")
    for candidate, reason_code in fallback_candidates:
        if not candidate:
            continue
        c_status = get_cached_model_access(candidate)
        if c_status is None or c_status.get("available") is True:
            logger.warning(
                "Model '%s' is unavailable (%s). Falling back to '%s' (%s).",
                configured_model,
                primary_reason,
                candidate,
                reason_code,
            )
            return configured_model, candidate, True, reason_code

    # None available
    raise RuntimeError(
        f"Model access failure: Neither configured {tier_lower} model '{configured_model}' "
        f"nor any fallback model is available to this project."
    )


@dataclass
class ModelPreflightResult:
    """Startup preflight check result across configured tiers."""
    fast_configured: str
    fast_status: str
    fast_reason: Optional[str]
    fast_resolved: Optional[str]

    reasoning_configured: str
    reasoning_status: str
    reasoning_reason: Optional[str]
    reasoning_resolved: Optional[str]

    default_configured: str
    default_status: str
    default_reason: Optional[str]

    is_ready: bool


def run_model_preflight(settings: Settings, probe: bool = True) -> ModelPreflightResult:
    """
    Validate configured models at application startup before creating founder projects.
    Caches availability and logs diagnostic table.
    """
    api_key = settings.openai_api_key or ""
    fast_configured = settings.fast_model_name or "gpt-4o-mini"
    reasoning_configured = settings.reasoning_model_name or settings.model_name
    default_configured = settings.model_name

    logger.info("Running MODEL PREFLIGHT verification...")

    def evaluate_model(name: str) -> tuple[str, Optional[str]]:
        if not is_valid_api_key(api_key):
            return "UNCONFIGURED", "OPENAI_API_KEY missing or placeholder"
        if not probe:
            cached = get_cached_model_access(name)
            if cached:
                return ("AVAILABLE" if cached["available"] else "UNAVAILABLE"), cached.get("reason")
            return "UNKNOWN", None
        avail, reason = check_model_access(api_key, name)
        register_model_access(name, avail, reason or ("AVAILABLE" if avail else "UNAVAILABLE"))
        return ("AVAILABLE" if avail else "UNAVAILABLE"), reason

    fast_status, fast_reason = evaluate_model(fast_configured)
    reasoning_status, reasoning_reason = evaluate_model(reasoning_configured)
    default_status, default_reason = evaluate_model(default_configured)

    # If primary fast model is unavailable, check fallback
    fast_resolved = fast_configured
    if fast_status != "AVAILABLE":
        if settings.fast_model_fallback:
            fb_stat, _ = evaluate_model(settings.fast_model_fallback)
            if fb_stat == "AVAILABLE":
                fast_resolved = settings.fast_model_fallback
            elif default_status == "AVAILABLE":
                fast_resolved = default_configured
            else:
                fast_resolved = None
        elif default_status == "AVAILABLE":
            fast_resolved = default_configured
        else:
            fast_resolved = None

    # If primary reasoning model is unavailable, check fallback
    reasoning_resolved = reasoning_configured
    if reasoning_status != "AVAILABLE":
        if settings.reasoning_model_fallback:
            fb_stat, _ = evaluate_model(settings.reasoning_model_fallback)
            if fb_stat == "AVAILABLE":
                reasoning_resolved = settings.reasoning_model_fallback
            elif default_status == "AVAILABLE":
                reasoning_resolved = default_configured
            else:
                reasoning_resolved = None
        elif default_status == "AVAILABLE":
            reasoning_resolved = default_configured
        else:
            reasoning_resolved = None

    is_ready = bool(fast_resolved and reasoning_resolved)

    preflight_log = (
        "\n==================================================\n"
        "MODEL PREFLIGHT\n"
        "==================================================\n\n"
        f"FAST:\nconfigured={fast_configured}\nstatus={fast_status}\n"
        + (f"reason={fast_reason}\n" if fast_status != "AVAILABLE" and fast_reason else "")
        + f"resolved={fast_resolved}\n\n"
        f"REASONING:\nconfigured={reasoning_configured}\nstatus={reasoning_status}\n"
        + (f"reason={reasoning_reason}\n" if reasoning_status != "AVAILABLE" and reasoning_reason else "")
        + f"resolved={reasoning_resolved}\n\n"
        f"DEFAULT:\nconfigured={default_configured}\nstatus={default_status}\n"
        + (f"reason={default_reason}\n" if default_status != "AVAILABLE" and default_reason else "")
        + "=================================================="
    )
    logger.info(preflight_log)

    return ModelPreflightResult(
        fast_configured=fast_configured,
        fast_status=fast_status,
        fast_reason=fast_reason,
        fast_resolved=fast_resolved,
        reasoning_configured=reasoning_configured,
        reasoning_status=reasoning_status,
        reasoning_reason=reasoning_reason,
        reasoning_resolved=reasoning_resolved,
        default_configured=default_configured,
        default_status=default_status,
        default_reason=default_reason,
        is_ready=is_ready,
    )


class TokenBudgetExhaustedError(RuntimeError):
    """Raised when an LLM completion budget (reasoning + output) was exhausted before completion."""

    def __init__(
        self,
        message: str,
        node_name: str | None = None,
        token_usage: dict[str, Any] | None = None,
        original_error: Exception | None = None,
    ):
        super().__init__(message)
        self.node_name = node_name
        self.token_usage = token_usage
        self.original_error = original_error


def is_token_budget_exhausted_error(exc: Exception) -> bool:
    """Return True if exception indicates completion token budget was exhausted."""
    if exc is None:
        return False
    if isinstance(exc, TokenBudgetExhaustedError):
        return True
    exc_type_name = type(exc).__name__
    if exc_type_name in ("LengthFinishReasonError", "TokenBudgetExhaustedError"):
        return True
    exc_str = str(exc).lower()
    return any(
        indicator in exc_str
        for indicator in (
            "length limit was reached",
            "lengthfinishreasonerror",
            "finish_reason: length",
            "finish_reason='length'",
            "finish_reason was length",
            "token budget exhausted",
            "token budget was entirely consumed",
            "exceeded token budget",
            "hit max_tokens",
            "hit max_completion_tokens",
        )
    )


class ModelResolutionInfo(dict):
    """Resolution info supporting both dict key access and 4-tuple unpacking."""

    def __iter__(self):
        return iter((
            self.get("configured_model"),
            self.get("resolved_model"),
            self.get("fallback_used", False),
            self.get("fallback_reason"),
        ))


def get_llm_client_resolution_info(client: Any) -> ModelResolutionInfo:
    """Retrieve model resolution metadata attached to a ChatOpenAI client."""
    conf = getattr(client, "_startary_configured_model", getattr(client, "model_name", "unknown"))
    res = getattr(client, "_startary_resolved_model", getattr(client, "model_name", "unknown"))
    fb = getattr(client, "_startary_fallback_used", False)
    reason = getattr(client, "_startary_fallback_reason", None)
    effort = getattr(client, "_startary_reasoning_effort", None)
    tier = getattr(client, "_startary_model_tier", None)
    return ModelResolutionInfo({
        "configured_model": str(conf) if conf is not None else None,
        "resolved_model": str(res) if res is not None else None,
        "fallback_used": bool(fb),
        "fallback_reason": str(reason) if reason else None,
        # None means the provider/API default effort applies (reasoning tier today).
        "reasoning_effort": effort if isinstance(effort, str) else None,
        "model_tier": tier if isinstance(tier, str) else None,
    })


def get_llm_client(
    settings: Settings,
    tier: str = "reasoning",
    model: str | None = None,
    max_tokens: int | None = None,
    max_completion_tokens: int | None = None,
    reasoning_effort: str | None = None,
) -> "ChatOpenAI":
    """
    Construct and return the LangChain OpenAI chat client for the given task tier.
    Applies runtime fallback resolution if configured tier model is verified unavailable.
    """
    # Import here to keep the provider boundary clean and enable mocking
    from langchain_openai import ChatOpenAI  # noqa: PLC0415

    if tier not in ("fast", "reasoning"):
        raise ValueError(f"get_llm_client: invalid tier {tier!r}; must be 'fast' or 'reasoning'.")

    api_key = settings.openai_api_key
    if not is_valid_api_key(api_key):
        raise ValueError(
            "OpenAI API key is not configured. "
            "Open the project .env file and replace PASTE_YOUR_OPENAI_API_KEY_HERE with your API key."
        )

    if model:
        configured_model = model
        resolved_model = model
        fallback_used = False
        fallback_reason = None
    else:
        try:
            configured_model, resolved_model, fallback_used, fallback_reason = resolve_model_for_tier(settings, tier)
        except Exception:
            # Fallback to direct config if resolution fails unexpectedly
            configured_model = settings.get_model_for_tier(tier)
            resolved_model = configured_model
            fallback_used = False
            fallback_reason = None

    effective_max_tokens = max_completion_tokens or max_tokens
    if effective_max_tokens is None:
        effective_max_tokens = 2500 if tier == "fast" else 3000

    is_reasoning_model = (tier == "reasoning") or any(m in resolved_model.lower() for m in ("o1", "o3", "sol", "luna"))
    effective_reasoning_effort = reasoning_effort
    if effective_reasoning_effort is None and tier == "fast" and is_reasoning_model:
        effective_reasoning_effort = "low"

    logger.debug(
        "LLM client | configured=%s resolved=%s fallback_used=%s tier=%s max_completion_tokens=%s reasoning_effort=%s",
        configured_model,
        resolved_model,
        fallback_used,
        tier,
        effective_max_tokens,
        effective_reasoning_effort,
    )

    client_kwargs: dict[str, Any] = {
        "model": resolved_model,
        "api_key": api_key,
        "temperature": 0,          # Deterministic; diagnosis requires consistency
        "max_retries": 0,          # We manage retries ourselves in the node layer
        "timeout": 30.0 if tier == "fast" else 60.0,
        "max_completion_tokens": effective_max_tokens,
        "metadata": {
            "model_tier": tier,
            "configured_model": configured_model,
            "resolved_model": resolved_model,
            "fallback_used": fallback_used,
            "max_completion_tokens": effective_max_tokens,
            "reasoning_effort": effective_reasoning_effort,
        },
        "tags": [
            f"tier:{tier}",
            f"model:{resolved_model}",
        ],
    }
    if effective_reasoning_effort and is_reasoning_model:
        client_kwargs["reasoning_effort"] = effective_reasoning_effort

    client = ChatOpenAI(**client_kwargs)

    # Attach resolution metadata for telemetry
    client._startary_configured_model = configured_model
    client._startary_resolved_model = resolved_model
    client._startary_fallback_used = fallback_used
    client._startary_fallback_reason = fallback_reason
    client._startary_max_completion_tokens = effective_max_tokens
    client._startary_reasoning_effort = effective_reasoning_effort
    client._startary_model_tier = tier

    return client

