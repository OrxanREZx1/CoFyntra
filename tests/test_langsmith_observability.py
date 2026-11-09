"""
Tests for LangSmith Latency Observability Integration (Milestone 6.1).

Validates:
1. Tracing is disabled by default (opt-in only, zero overhead, no API key needed).
2. Privacy masking by default (hide_inputs=True, hide_outputs=True).
3. Metadata sanitization (guarantee zero PII, raw context, prompts, or secrets leak into traces).
4. Standard trace configuration (thread_id, turn_number, run_name, tags).
5. LLM client metadata and tier tagging in get_llm_client.
6. Token-exhaustion retry tracking in _invoke_structured_with_token_retry.
7. Safe flushing and graceful degradation when unconfigured.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from startary.config import Settings
from startary.domain.enums import WorkflowPhase
from startary.llm.provider import get_llm_client
from startary.observability.langsmith import (
    CODE_REVISION,
    DEFAULT_ENVIRONMENT,
    DEFAULT_LANGSMITH_PROJECT,
    FORBIDDEN_METADATA_KEYS,
    build_trace_config,
    flush_langsmith_traces,
    get_langsmith_tracer,
    is_tracing_enabled,
    sanitize_metadata,
    setup_langsmith_environment,
)


class DummyOutputSchema(BaseModel):
    summary: str
    status: str


# ── 1. Default Configuration & Opt-in Semantics ──────────────────────────────


def test_tracing_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that tracing is disabled by default without explicit opt-in."""
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)

    # Defaults only: ignore the developer's .env, which may legitimately enable tracing for live runs.
    settings = Settings(_env_file=None, openai_api_key="sk-test-fake")
    assert settings.langsmith_tracing is False
    assert is_tracing_enabled(settings) is False

    cfg = setup_langsmith_environment(settings)
    assert cfg.enabled is False
    assert os.environ.get("LANGSMITH_TRACING") == "false"
    assert os.environ.get("LANGCHAIN_TRACING_V2") == "false"


def test_tracing_opt_in_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """If user opts in via flag/env but provides no API key, tracing remains safely inactive."""
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)

    settings = Settings(openai_api_key="sk-test-fake", langsmith_tracing=True, langsmith_api_key=None)
    assert is_tracing_enabled(settings) is True

    cfg = setup_langsmith_environment(settings)
    # enabled must be False because API key is absent
    assert cfg.enabled is False
    assert get_langsmith_tracer(settings) is None


def test_tracing_active_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    """When flag and API key are provided, tracing becomes active with proper env vars."""
    settings = Settings(
        openai_api_key="sk-test-fake",
        langsmith_tracing=True,
        langsmith_api_key="lsv2_pt_test_key_12345",
        langsmith_project="custom-latency-project",
        langsmith_hide_inputs=True,
        langsmith_hide_outputs=True,
    )
    cfg = setup_langsmith_environment(settings)
    assert cfg.enabled is True
    assert cfg.project == "custom-latency-project"
    assert os.environ.get("LANGSMITH_TRACING") == "true"
    assert os.environ.get("LANGCHAIN_TRACING_V2") == "true"
    assert os.environ.get("LANGSMITH_API_KEY") == "lsv2_pt_test_key_12345"
    assert os.environ.get("LANGSMITH_PROJECT") == "custom-latency-project"
    assert os.environ.get("LANGSMITH_HIDE_INPUTS") == "true"
    assert os.environ.get("LANGSMITH_HIDE_OUTPUTS") == "true"

    # Reset environment so subsequent tests remain strictly isolated
    setup_langsmith_environment(Settings(openai_api_key="sk-test-fake", langsmith_tracing=False))


# ── 2. Privacy & Metadata Sanitization ───────────────────────────────────────


def test_metadata_sanitization_removes_forbidden_keys() -> None:
    """Sanitizer must strip all confidential, founder PII, and prompt keys."""
    raw_metadata = {
        "scenario_id": "scen-test-01",
        "founder_turn_number": 1,
        "workflow_phase": "UNDERSTAND",
        # Forbidden keys:
        "raw_context": "Confidential pitch about revolutionary AI football scout",
        "founder_messages": [{"role": "founder", "content": "secret"}],
        "prompt": "You are Startary AI...",
        "openai_api_key": "sk-secret-key",
        "founder_name": "Alice and Bob",
        "user_input": "Our secret IP is...",
    }

    sanitized = sanitize_metadata(raw_metadata)

    # Allowed operational keys
    assert sanitized["scenario_id"] == "scen-test-01"
    assert sanitized["founder_turn_number"] == 1
    assert sanitized["workflow_phase"] == "UNDERSTAND"

    # Forbidden keys MUST NOT be present
    assert "raw_context" not in sanitized
    assert "founder_messages" not in sanitized
    assert "prompt" not in sanitized
    assert "openai_api_key" not in sanitized
    assert "founder_name" not in sanitized
    assert "user_input" not in sanitized


def test_metadata_sanitization_removes_multiline_and_long_strings() -> None:
    """Sanitizer must drop long unstructured text blocks that could contain founder info."""
    raw_metadata = {
        "iteration_number": 1,
        "short_status": "in_progress",
        "long_text": "x" * 150,
        "multiline_note": "line 1\nline 2",
    }
    sanitized = sanitize_metadata(raw_metadata)
    assert sanitized["iteration_number"] == 1
    assert sanitized["short_status"] == "in_progress"
    assert "long_text" not in sanitized
    assert "multiline_note" not in sanitized


# ── 3. Trace Config Construction ─────────────────────────────────────────────


def test_build_trace_config_disabled() -> None:
    """When tracing is disabled, config contains thread_id and run_name but no callbacks."""
    settings = Settings(openai_api_key="sk-test-fake", langsmith_tracing=False)
    config = build_trace_config(
        project_id="proj-12345678",
        turn_number=0,
        phase=WorkflowPhase.UNDERSTAND,
        iteration=1,
        run_type="initial_context",
        settings=settings,
    )

    assert config["configurable"] == {"thread_id": "proj-12345678"}
    assert config["run_name"] == "startary_turn_0"
    assert "callbacks" not in config
    assert config["metadata"]["founder_turn_number"] == 0
    assert config["metadata"]["code_revision"] == CODE_REVISION
    assert config["metadata"]["environment"] == DEFAULT_ENVIRONMENT
    assert "phase:understand" in config["tags"]
    assert "turn:0" in config["tags"]
    assert "type:initial_context" in config["tags"]


def test_build_trace_config_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """When tracing is enabled, config includes a LangChainTracer callback."""
    settings = Settings(
        openai_api_key="sk-test-fake",
        langsmith_tracing=True,
        langsmith_api_key="lsv2_pt_test_key_12345",
        langsmith_project="test-project",
    )

    with patch("startary.observability.langsmith.get_langsmith_tracer") as mock_tracer_factory:
        mock_tracer = MagicMock()
        mock_tracer_factory.return_value = mock_tracer

        config = build_trace_config(
            project_id="proj-abcdef123456",
            turn_number=2,
            phase=WorkflowPhase.UNDERSTAND,
            iteration=1,
            run_type="founder_reply",
            settings=settings,
        )

        assert config["configurable"] == {"thread_id": "proj-abcdef123456"}
        assert config["run_name"] == "startary_turn_2"
        assert config["callbacks"] == [mock_tracer]
        assert "turn:2" in config["tags"]
        assert "type:founder_reply" in config["tags"]


# ── 4. LLM Client Instrumentation ────────────────────────────────────────────


def test_get_llm_client_has_tier_and_model_metadata() -> None:
    """LLM client must carry model_tier, resolved_model, and tag instrumentation."""
    settings = Settings(
        openai_api_key="sk-test-fake",
        fast_model_name="gpt-5.6-luna",
        reasoning_model_name="gpt-5.6-sol",
    )

    fast_client = get_llm_client(settings, tier="fast", max_completion_tokens=1000)
    assert hasattr(fast_client, "metadata")
    assert fast_client.metadata["model_tier"] == "fast"
    assert fast_client.metadata["configured_model"] == "gpt-5.6-luna"
    assert fast_client.metadata["max_completion_tokens"] == 1000
    assert "tier:fast" in fast_client.tags
    assert "model:gpt-5.6-luna" in fast_client.tags

    reasoning_client = get_llm_client(settings, tier="reasoning", max_completion_tokens=2500)
    assert reasoning_client.metadata["model_tier"] == "reasoning"
    assert reasoning_client.metadata["configured_model"] == "gpt-5.6-sol"
    assert reasoning_client.metadata["max_completion_tokens"] == 2500
    assert "tier:reasoning" in reasoning_client.tags


# ── 5. Token Exhaustion Retry Observability ───────────────────────────────────


def test_retry_observability_tracks_attempt_metadata_and_tags() -> None:
    """_invoke_structured_with_token_retry must tag attempt 1 and attempt 2 correctly."""
    from openai import LengthFinishReasonError
    from startary.graph.nodes import _invoke_structured_with_token_retry

    settings = Settings(
        openai_api_key="sk-test-fake",
        fast_model_name="gpt-5.6-luna",
    )

    mock_runnable_attempt1 = MagicMock()
    mock_runnable_attempt2 = MagicMock()

    # Attempt 1 fails with LengthFinishReasonError
    mock_runnable_attempt1.invoke.side_effect = LengthFinishReasonError(completion=MagicMock())
    # Attempt 2 succeeds
    expected_output = DummyOutputSchema(summary="Success after retry", status="ok")
    mock_runnable_attempt2.invoke.return_value = expected_output

    expanded_client = MagicMock()
    expanded_client.with_structured_output.return_value = mock_runnable_attempt2

    with patch("startary.graph.nodes._safe_get_llm_client", return_value=expanded_client):
        result, latency = _invoke_structured_with_token_retry(
            structured_llm=mock_runnable_attempt1,
            messages=[{"role": "user", "content": "Analyze startup"}],
            settings=settings,
            tier="fast",
            base_max_tokens=1000,
            node_name="extract_startup_state",
            schema_cls=DummyOutputSchema,
        )

        assert result.summary == "Success after retry"
        assert result.status == "ok"
        assert latency >= 0.0

        # Verify attempt 1 was invoked with attempt:1 metadata
        assert mock_runnable_attempt1.invoke.call_count == 1
        call_1_kwargs = mock_runnable_attempt1.invoke.call_args[1]
        assert call_1_kwargs["config"]["metadata"]["attempt"] == 1
        assert "attempt:1" in call_1_kwargs["config"]["tags"]

        # Verify attempt 2 was invoked with attempt:2 metadata and retry reason
        assert mock_runnable_attempt2.invoke.call_count == 1
        call_2_kwargs = mock_runnable_attempt2.invoke.call_args[1]
        assert call_2_kwargs["config"]["metadata"]["attempt"] == 2
        assert call_2_kwargs["config"]["metadata"]["is_retry"] is True
        assert call_2_kwargs["config"]["metadata"]["retry_reason"] == "token_budget_exhausted"
        assert call_2_kwargs["config"]["metadata"]["expanded_max_tokens"] == 2500
        assert "attempt:2" in call_2_kwargs["config"]["tags"]
        assert "retry:token_expansion" in call_2_kwargs["config"]["tags"]


# ── 6. Flush and Graceful Cleanup ────────────────────────────────────────────


def test_flush_langsmith_traces_safe_when_unconfigured() -> None:
    """Calling flush_langsmith_traces when client is None does not raise."""
    flush_langsmith_traces()


def test_flush_langsmith_traces_invokes_client_flush() -> None:
    """When client is present, flush_langsmith_traces delegates to client.flush."""
    mock_client = MagicMock()
    with patch("startary.observability.langsmith._cached_client", mock_client):
        flush_langsmith_traces(timeout=2.0)
        mock_client.flush.assert_called_once_with(timeout=2.0)
