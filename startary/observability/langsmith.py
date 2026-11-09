"""
LangSmith Latency Observability Integration (Milestone 6.1).

Provides non-intrusive, opt-in tracing for Startary's AI reasoning loop.
Tracing is disabled by default (LANGSMITH_TRACING=false).
When enabled by developer for local profiling, captures:
- Top-level founder turn execution spans (duration, status, phase, iteration)
- Individual LangGraph node durations (native Pregel execution)
- Individual LLM calls (model, tier, latency, token counts, reasoning tokens)
- Token-exhaustion retry attempts and recovery budgets
- Privacy masking by default (LANGSMITH_HIDE_INPUTS=true, LANGSMITH_HIDE_OUTPUTS=true)
"""

from __future__ import annotations

import atexit
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Optional

from startary.config import Settings, get_settings

logger = logging.getLogger(__name__)

# Constants & safe defaults
DEFAULT_LANGSMITH_PROJECT = "startary-ai-latency"
DEFAULT_ENVIRONMENT = "local-ai-latency"
CODE_REVISION = "v0.4.1.4-alpha-core"

# Disallowed substrings in metadata/tags to guarantee zero leakage of founder PII or raw text
FORBIDDEN_METADATA_KEYS = (
    "prompt",
    "message",
    "raw_context",
    "context",
    "founder_name",
    "founder_messages",
    "startup_summary",
    "user_input",
    "input",
    "secret",
    "key",
    "password",
    "token_val",
)


@dataclass
class LangSmithConfig:
    """Validated configuration for LangSmith tracing."""
    enabled: bool
    api_key: Optional[str]
    project: str
    hide_inputs: bool
    hide_outputs: bool
    workspace_id: Optional[str] = None
    endpoint: Optional[str] = None


_cached_client: Any = None
_atexit_registered: bool = False


def is_tracing_enabled(settings: Optional[Settings] = None) -> bool:
    """Return True only if tracing has been explicitly enabled via settings or env."""
    if settings is not None:
        return bool(getattr(settings, "langsmith_tracing", False))
    return os.environ.get("LANGSMITH_TRACING", "").strip().lower() in ("true", "1", "yes")


def setup_langsmith_environment(settings: Optional[Settings] = None) -> LangSmithConfig:
    """
    Synchronize settings to standard LangSmith / LangChain environment variables.
    Tracing is DISABLED by default.
    """
    if settings is None:
        settings = get_settings()

    tracing_requested = is_tracing_enabled(settings)
    api_key = settings.langsmith_api_key or os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY")
    project = settings.langsmith_project or os.environ.get("LANGSMITH_PROJECT") or DEFAULT_LANGSMITH_PROJECT
    hide_inputs = settings.langsmith_hide_inputs
    hide_outputs = settings.langsmith_hide_outputs
    workspace_id = settings.langsmith_workspace_id or os.environ.get("LANGSMITH_WORKSPACE_ID")
    endpoint = settings.langsmith_endpoint or os.environ.get("LANGSMITH_ENDPOINT")

    # If tracing was requested but no API key is available, disable active sending to prevent crashes
    active_tracing = bool(tracing_requested and api_key and str(api_key).strip())

    if active_tracing:
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGSMITH_API_KEY"] = str(api_key).strip()
        os.environ["LANGCHAIN_API_KEY"] = str(api_key).strip()
        os.environ["LANGSMITH_PROJECT"] = project
        os.environ["LANGCHAIN_PROJECT"] = project
        os.environ["LANGSMITH_HIDE_INPUTS"] = "true" if hide_inputs else "false"
        os.environ["LANGCHAIN_HIDE_INPUTS"] = "true" if hide_inputs else "false"
        os.environ["LANGSMITH_HIDE_OUTPUTS"] = "true" if hide_outputs else "false"
        os.environ["LANGCHAIN_HIDE_OUTPUTS"] = "true" if hide_outputs else "false"

        if workspace_id:
            os.environ["LANGSMITH_WORKSPACE_ID"] = workspace_id
        if endpoint:
            os.environ["LANGSMITH_ENDPOINT"] = endpoint

        logger.info(
            "LangSmith observability enabled | project=%s hide_inputs=%s hide_outputs=%s",
            project, hide_inputs, hide_outputs,
        )
        global _atexit_registered
        if not _atexit_registered:
            atexit.register(flush_langsmith_traces)
            _atexit_registered = True
    else:
        # Explicitly ensure tracing is disabled in environment
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        if tracing_requested and not api_key:
            logger.warning(
                "LangSmith tracing was requested (LANGSMITH_TRACING=true), but LANGSMITH_API_KEY "
                "is not configured. Tracing is disabled to prevent execution errors."
            )

    return LangSmithConfig(
        enabled=active_tracing,
        api_key=str(api_key).strip() if api_key else None,
        project=project,
        hide_inputs=hide_inputs,
        hide_outputs=hide_outputs,
        workspace_id=workspace_id,
        endpoint=endpoint,
    )


def sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """
    Filter metadata to guarantee no confidential founder details, messages, or keys leak.
    Only allows structured, synthetic, or operational telemetry values.
    """
    sanitized: dict[str, Any] = {}
    for k, v in metadata.items():
        k_lower = str(k).lower()
        if any(bad in k_lower for bad in FORBIDDEN_METADATA_KEYS):
            continue
        if isinstance(v, (str, int, float, bool)) or v is None:
            # If string contains raw multi-line text or potential founder pitch, omit it
            if isinstance(v, str) and (len(v) > 120 or "\n" in v):
                continue
            sanitized[k] = v
        elif isinstance(v, (list, tuple)) and all(isinstance(x, (str, int, float, bool)) for x in v):
            sanitized[k] = list(v)
    return sanitized


def get_langsmith_tracer(settings: Optional[Settings] = None) -> Any | None:
    """
    Construct a LangChainTracer connected to the configured LangSmith project.
    Returns None if tracing is disabled or dependencies/credentials are missing.
    """
    cfg = setup_langsmith_environment(settings)
    if not cfg.enabled:
        return None

    try:
        from langchain_core.tracers.langchain import LangChainTracer
        from langsmith.client import Client

        global _cached_client
        if _cached_client is None:
            _cached_client = Client(
                api_key=cfg.api_key,
                api_url=cfg.endpoint,
                hide_inputs=cfg.hide_inputs,
                hide_outputs=cfg.hide_outputs,
            )

        tracer = LangChainTracer(
            project_name=cfg.project,
            client=_cached_client,
        )
        return tracer
    except Exception as exc:
        logger.warning("Could not initialize LangChainTracer: %s. Tracing skipped.", exc)
        return None


def build_trace_config(
    project_id: str,
    turn_number: int = 0,
    phase: Any = "UNDERSTAND",
    iteration: int = 1,
    run_type: str = "founder_turn",
    scenario_id: Optional[str] = None,
    settings: Optional[Settings] = None,
    extra_metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Build the standard LangGraph RunnableConfig for a turn invocation.
    Includes thread_id, run_name, privacy-safe metadata, and tags.
    """
    synth_scenario_id = scenario_id or f"scen-{project_id[:8]}"
    # Milestone 6.2.7: understanding mode of this run (same setting build_graph reads), for A/B filtering.
    parallel_understanding = getattr(settings or get_settings(), "parallel_understanding", False) is True

    base_metadata = {
        "scenario_id": synth_scenario_id,
        "founder_turn_number": turn_number,
        "workflow_phase": getattr(phase, "value", str(phase)),
        "iteration_number": iteration,
        "run_type": run_type,
        "code_revision": CODE_REVISION,
        "environment": DEFAULT_ENVIRONMENT,
        "parallel_understanding": parallel_understanding,
    }
    if extra_metadata:
        base_metadata.update(extra_metadata)

    safe_metadata = sanitize_metadata(base_metadata)

    phase_tag = getattr(phase, "value", str(phase)).lower()
    tags = [
        "startary",
        "ai-latency",
        f"phase:{phase_tag}",
        f"turn:{turn_number}",
        f"type:{run_type}",
        f"understanding:{'parallel' if parallel_understanding else 'sequential'}",
    ]

    config: dict[str, Any] = {
        "configurable": {"thread_id": project_id},
        "run_name": f"startary_turn_{turn_number}",
        "metadata": safe_metadata,
        "tags": tags,
    }

    tracer = get_langsmith_tracer(settings)
    if tracer is not None:
        config["callbacks"] = [tracer]

    return config


def emit_decision_span(run_name: str, metadata: dict[str, Any], tags: Optional[list[str]] = None) -> dict[str, Any]:
    """
    Milestone 6.2.5: make a deterministic decision visible in LangSmith.

    Emits a zero-cost child run (a no-op runnable with empty input) carrying only sanitized
    metadata; inside a LangGraph node it nests under the node's span through the active callback
    context. With tracing disabled it is a local no-op. Returns the sanitized metadata.
    """
    safe = sanitize_metadata(metadata)
    try:
        from langchain_core.runnables import RunnableLambda

        RunnableLambda(lambda _: None, name=run_name).invoke(
            {}, config={"run_name": run_name, "metadata": safe, "tags": list(tags or [])}
        )
    except Exception as exc:  # observability must never break the workflow
        logger.debug("emit_decision_span(%s) failed: %s", run_name, exc)
    return safe


def flush_langsmith_traces(timeout: float = 5.0) -> None:
    """
    Ensure all pending asynchronous trace spans are flushed before CLI process exit.
    Does nothing if tracing is disabled.
    """
    global _cached_client
    if _cached_client is not None:
        try:
            logger.debug("Flushing pending LangSmith traces (timeout=%.1fs)...", timeout)
            if hasattr(_cached_client, "flush"):
                _cached_client.flush(timeout=timeout)
            else:
                import time
                time.sleep(min(timeout, 0.5))
        except Exception as exc:
            logger.debug("Error while flushing LangSmith traces: %s", exc)
