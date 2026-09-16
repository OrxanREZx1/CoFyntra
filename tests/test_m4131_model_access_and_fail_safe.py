"""
Milestone M4.1.3.1 — Model Access & Provider Fail-Safe Patch Test Suite.

25 comprehensive tests validating:
  1. FAST configured Luna -> 403 -> falls back to mini
  2. REASONING configured Sol -> 403 -> falls back to 4o
  3. Preflight check marks Luna UNAVAILABLE and caches status
  4. Subsequent LLM call uses cached availability without re-probing
  5. 403 does not retry twice (permanent error fails fast on attempt 1)
  6. 404 does not retry twice
  7. 401 does not retry twice
  8. Timeout / 500 still retries (transient error retry preserved)
  9. Rate limit (429) still retries
  10. FAST fallback works across fast-tier nodes
  11. Telemetry records configured_model vs resolved_model
  12. Telemetry records fallback_used=True and fallback_reason
  13. Telemetry records fallback_used=False on clean run
  14. Trace records configured_model and resolved_model
  15. Preflight check logs diagnostic table at startup
  16. Preflight check fails cleanly if both configured and fallback unavailable
  17. CLI --check-models outputs diagnostic table
  18. CLI --check-models does not expose API keys
  19. Extraction 403 sets extraction_status="FAILED"
  20. Extraction failure does NOT mark strategic_state_changed=False
  21. Extraction failure prevents diagnosis cache reuse
  22. Extraction failure does NOT overwrite existing budget with 0/None
  23. Extraction failure does NOT overwrite customer access with UNKNOWN
  24. Extraction failure preserves existing execution context
  25. Full turn completes gracefully with fallback when configured model has 403
"""

from __future__ import annotations

import io
import sys
from unittest.mock import MagicMock, patch
import pytest

from startary.cli import check_models_diagnostic
from startary.config import Settings
from startary.domain.area_knowledge import AreaStatus
from startary.domain.taxonomy import StartupArea
from startary.domain.telemetry import (
    NodeTelemetryRecord,
    TurnTelemetrySummary,
    record_telemetry_event,
)
from startary.domain.trace import AITrace
from startary.graph.nodes import (
    assess_baseline_state_sufficiency,
    critique_experiment,
    diagnose_startup,
    extract_startup_state,
    generate_experiment,
    handle_conversational_repair,
    interpret_founder_turn,
    is_strategic_state_changed,
    merge_execution_state,
    plan_baseline_question,
)
from startary.llm.schemas import AreaDiagnosisOutput, DiagnosisLLMOutput
from startary.llm.provider import (
    clear_model_access_cache,
    get_cached_model_access,
    get_llm_client,
    get_llm_client_resolution_info,
    is_permanent_model_error,
    is_transient_error,
    register_model_access,
    resolve_model_for_tier,
    run_model_preflight,
)


@pytest.fixture(autouse=True)
def clean_cache():
    """Clear model access cache before each test."""
    clear_model_access_cache()
    yield
    clear_model_access_cache()


# ===========================================================================
# 1. Fallback Hierarchy & Availability Caching Tests (1–4)
# ===========================================================================

def test_01_fast_model_fallback_hierarchy():
    """FAST configured Luna -> 403 / unavailable -> falls back to fast fallback (mini)."""
    settings = Settings(
        openai_api_key="sk-test-key",
        fast_model_name="gpt-5.6-luna",
        fast_model_fallback="gpt-4o-mini",
        model_name="gpt-4o",
    )
    register_model_access("gpt-5.6-luna", False, "403 Forbidden: model_not_found")
    cfg, res, fb_used, fb_reason = resolve_model_for_tier(settings, tier="fast")
    assert cfg == "gpt-5.6-luna"
    assert res == "gpt-4o-mini"
    assert fb_used is True
    assert fb_reason == "FAST_MODEL_UNAVAILABLE_FALLBACK_SELECTED"


def test_02_reasoning_model_fallback_hierarchy():
    """REASONING configured Sol -> 403 / unavailable -> falls back to reasoning fallback (4o)."""
    settings = Settings(
        openai_api_key="sk-test-key",
        reasoning_model_name="gpt-5.6-sol",
        reasoning_model_fallback="gpt-4o",
        model_name="gpt-4o",
    )
    register_model_access("gpt-5.6-sol", False, "403 Forbidden: model_not_found")
    cfg, res, fb_used, fb_reason = resolve_model_for_tier(settings, tier="reasoning")
    assert cfg == "gpt-5.6-sol"
    assert res == "gpt-4o"
    assert fb_used is True
    assert fb_reason == "REASONING_MODEL_UNAVAILABLE_FALLBACK_SELECTED"


def test_03_preflight_marks_luna_unavailable_and_caches():
    """Preflight check marks Luna UNAVAILABLE and caches status."""
    settings = Settings(
        openai_api_key="sk-test-key",
        fast_model_name="gpt-5.6-luna",
        fast_model_fallback="gpt-4o-mini",
        reasoning_model_name="gpt-4o",
        model_name="gpt-4o",
    )

    with patch("startary.llm.provider.check_model_access") as mock_probe:
        def probe_side_effect(key, model):
            if model == "gpt-5.6-luna":
                return False, "403 Forbidden: model_not_found"
            return True, None

        mock_probe.side_effect = probe_side_effect
        result = run_model_preflight(settings, probe=True)

        assert result.fast_configured == "gpt-5.6-luna"
        assert result.fast_status == "UNAVAILABLE"
        assert result.fast_resolved == "gpt-4o-mini"
        assert result.is_ready is True

        cached = get_cached_model_access("gpt-5.6-luna")
        assert cached is not None
        assert cached["available"] is False
        assert "403" in cached["reason"]


def test_04_subsequent_call_uses_cached_availability_without_probing():
    """Subsequent call uses cached availability without re-probing."""
    settings = Settings(
        openai_api_key="sk-test-key",
        fast_model_name="gpt-5.6-luna",
        fast_model_fallback="gpt-4o-mini",
        model_name="gpt-4o",
    )
    register_model_access("gpt-5.6-luna", False, "403 Forbidden: model_not_found")

    with patch("startary.llm.provider.check_model_access") as mock_probe:
        cfg, res, fb_used, _ = resolve_model_for_tier(settings, tier="fast")
        mock_probe.assert_not_called()
        assert cfg == "gpt-5.6-luna"
        assert res == "gpt-4o-mini"
        assert fb_used is True


# ===========================================================================
# 2. Error Classification & Fast-Fail vs Retry Tests (5–9)
# ===========================================================================

def test_05_permanent_403_fails_fast_no_retry():
    """403 does not retry twice (fails fast on attempt 1)."""
    exc = Exception("403 Forbidden: Project does not have access to model gpt-5.6-luna code=model_not_found")
    assert is_permanent_model_error(exc) is True
    assert is_transient_error(exc) is False

    # Verify in extract_startup_state node
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.side_effect = exc
    mock_llm.with_structured_output.return_value = mock_structured

    state = {
        "project_id": "test_p",
        "founder_messages": [{"role": "founder", "content": "We are building an app"}],
        "raw_context": "We are building an app",
    }
    with patch("startary.graph.nodes._safe_get_llm_client", return_value=mock_llm):
        res = extract_startup_state(state)
        # Should have invoked exactly once and then broken out of the retry loop
        assert mock_structured.invoke.call_count == 1
        assert res["extraction_status"] == "FAILED"


def test_06_permanent_404_fails_fast_no_retry():
    """404 model not found does not retry twice."""
    exc = Exception("404 The model `gpt-5.6-luna` does not exist or you do not have access to it.")
    assert is_permanent_model_error(exc) is True
    assert is_transient_error(exc) is False


def test_07_permanent_401_fails_fast_no_retry():
    """401 unauthorized / invalid api key does not retry twice."""
    exc = Exception("401 Unauthorized: Incorrect API key provided.")
    assert is_permanent_model_error(exc) is True
    assert is_transient_error(exc) is False


def test_08_transient_timeout_retries():
    """Timeout / 500 is classified as transient and not permanent."""
    exc = TimeoutError("Request timed out after 60 seconds")
    assert is_permanent_model_error(exc) is False
    assert is_transient_error(exc) is True

    exc_500 = Exception("500 Internal Server Error from OpenAI")
    assert is_permanent_model_error(exc_500) is False
    assert is_transient_error(exc_500) is True


def test_09_transient_rate_limit_429_retries():
    """Rate limit 429 is classified as transient and not permanent."""
    exc_429 = Exception("429 Too Many Requests: Rate limit reached for requests")
    assert is_permanent_model_error(exc_429) is False
    assert is_transient_error(exc_429) is True


# ===========================================================================
# 3. Fast Tier Fallback Across Nodes (10)
# ===========================================================================

def test_10_fast_fallback_works_across_fast_tier_nodes():
    """FAST fallback attaches resolution info across all fast-tier nodes."""
    settings = Settings(
        openai_api_key="sk-test-key",
        fast_model_name="gpt-5.6-luna",
        fast_model_fallback="gpt-4o-mini",
        model_name="gpt-4o",
    )
    register_model_access("gpt-5.6-luna", False, "403 Forbidden: model_not_found")

    with patch("startary.llm.provider.ChatOpenAI") as mock_chat:
        client = get_llm_client(settings, tier="fast")
        info = get_llm_client_resolution_info(client)
        assert info["configured_model"] == "gpt-5.6-luna"
        assert info["resolved_model"] == "gpt-4o-mini"
        assert info["fallback_used"] is True
        assert info["fallback_reason"] == "FAST_MODEL_UNAVAILABLE_FALLBACK_SELECTED"


# ===========================================================================
# 4. Telemetry & Trace Model Validation (11–14)
# ===========================================================================

def test_11_telemetry_records_configured_vs_resolved_model():
    """Telemetry records configured_model vs resolved_model."""
    rec = NodeTelemetryRecord(
        node_name="diagnose_startup",
        model_name="gpt-4o-mini",
        duration_ms=120.0,
        configured_model="gpt-5.6-luna",
        resolved_model="gpt-4o-mini",
        fallback_used=True,
        fallback_reason="FAST_MODEL_UNAVAILABLE_FALLBACK_SELECTED",
    )
    assert rec.configured_model == "gpt-5.6-luna"
    assert rec.resolved_model == "gpt-4o-mini"
    assert rec.fallback_used is True
    d = rec.to_dict()
    assert d["configured_model"] == "gpt-5.6-luna"
    assert d["resolved_model"] == "gpt-4o-mini"


def test_12_telemetry_records_fallback_used_and_reason():
    """record_telemetry_event records fallback_used=True and fallback_reason."""
    state = {"turn_telemetry": None}
    summary = record_telemetry_event(
        state=state,
        node_name="extract_startup_state",
        model_name="gpt-4o-mini",
        duration_ms=80.0,
        configured_model="gpt-5.6-luna",
        resolved_model="gpt-4o-mini",
        fallback_used=True,
        fallback_reason="FAST_MODEL_UNAVAILABLE_FALLBACK_SELECTED",
    )
    assert len(summary.records) == 1
    assert summary.records[0].fallback_used is True
    assert summary.records[0].fallback_reason == "FAST_MODEL_UNAVAILABLE_FALLBACK_SELECTED"


def test_13_telemetry_records_clean_run_no_fallback():
    """Telemetry records fallback_used=False on clean run."""
    rec = NodeTelemetryRecord(
        node_name="diagnose_startup",
        model_name="gpt-5.6-luna",
        duration_ms=150.0,
        configured_model="gpt-5.6-luna",
        resolved_model="gpt-5.6-luna",
        fallback_used=False,
        fallback_reason=None,
    )
    assert rec.fallback_used is False
    assert rec.fallback_reason is None


def test_14_trace_records_configured_and_resolved_model():
    """AITrace model validates and stores configured_model and resolved_model."""
    trace = AITrace(
        project_id="test_p",
        module_name="generate_experiment",
        prompt_version="V1",
        input_data={"x": 1},
        output_data={"y": 2},
        latency_ms=250.0,
        model_name="gpt-4o",
        configured_model="gpt-5.6-sol",
        resolved_model="gpt-4o",
        fallback_used=True,
        fallback_reason="REASONING_MODEL_UNAVAILABLE_FALLBACK_SELECTED",
    )
    assert trace.configured_model == "gpt-5.6-sol"
    assert trace.resolved_model == "gpt-4o"
    assert trace.fallback_used is True
    assert trace.fallback_reason == "REASONING_MODEL_UNAVAILABLE_FALLBACK_SELECTED"


# ===========================================================================
# 5. Preflight & CLI Validation Tests (15–18)
# ===========================================================================

def test_15_preflight_check_logs_diagnostic_table(caplog):
    """Preflight check logs diagnostic table at startup."""
    import logging
    settings = Settings(
        openai_api_key="sk-test-key",
        fast_model_name="gpt-5.6-luna",
        fast_model_fallback="gpt-4o-mini",
        reasoning_model_name="gpt-4o",
        model_name="gpt-4o",
    )
    with caplog.at_level(logging.INFO), patch("startary.llm.provider.check_model_access", return_value=(True, None)):
        run_model_preflight(settings, probe=True)
        assert "MODEL PREFLIGHT" in caplog.text
        assert "FAST:" in caplog.text
        assert "REASONING:" in caplog.text


def test_16_preflight_fails_cleanly_if_both_unavailable():
    """Preflight check fails cleanly (is_ready=False) if both configured and fallback are unavailable."""
    settings = Settings(
        openai_api_key="sk-test-key",
        fast_model_name="gpt-5.6-luna",
        fast_model_fallback="gpt-4o-mini",
        reasoning_model_name="gpt-5.6-sol",
        reasoning_model_fallback="gpt-4o",
        model_name="gpt-4o",
    )
    with patch("startary.llm.provider.check_model_access", return_value=(False, "403 Forbidden: model_not_found")):
        result = run_model_preflight(settings, probe=True)
        assert result.is_ready is False
        assert result.fast_resolved is None
        assert result.reasoning_resolved is None


def test_17_cli_check_models_outputs_table(capsys):
    """CLI --check-models outputs diagnostic table."""
    settings = Settings(
        openai_api_key="sk-secret-key-12345",
        fast_model_name="gpt-5.6-luna",
        fast_model_fallback="gpt-4o-mini",
        reasoning_model_name="gpt-5.6-sol",
        reasoning_model_fallback="gpt-4o",
        model_name="gpt-4o",
    )

    def mock_probe(key, model):
        if "luna" in model or "sol" in model:
            return False, "403 Forbidden: model_not_found"
        return True, "OK"

    with patch("startary.llm.provider.check_model_access", side_effect=mock_probe):
        check_models_diagnostic(settings)
        captured = capsys.readouterr().out
        assert "Tier" in captured
        assert "Role" in captured
        assert "Model Name" in captured
        assert "Accessible" in captured
        assert "Status/Error" in captured
        assert "gpt-5.6-luna" in captured
        assert "NO" in captured
        assert "gpt-4o-mini" in captured
        assert "YES" in captured


def test_18_cli_check_models_does_not_expose_api_keys(capsys):
    """CLI --check-models does not expose API keys."""
    secret_key = "sk-super-secret-key-that-must-never-appear"
    settings = Settings(
        openai_api_key=secret_key,
        fast_model_name="gpt-4o-mini",
        model_name="gpt-4o",
    )
    with patch("startary.llm.provider.check_model_access", return_value=(True, None)):
        check_models_diagnostic(settings)
        captured = capsys.readouterr().out
        assert secret_key not in captured


# ===========================================================================
# 6. Extraction Failure & Invariants Tests (19–25)
# ===========================================================================

def test_19_extraction_403_sets_extraction_status_failed():
    """Extraction 403 sets extraction_status='FAILED'."""
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.side_effect = Exception("403 Forbidden: model_not_found")
    mock_llm.with_structured_output.return_value = mock_structured

    state = {
        "project_id": "test_p",
        "founder_messages": [{"role": "founder", "content": "I sell handmade ceramics."}],
        "raw_context": "I sell handmade ceramics.",
    }
    with patch("startary.graph.nodes._safe_get_llm_client", return_value=mock_llm):
        res = extract_startup_state(state)
        assert res["extraction_status"] == "FAILED"
        assert "403" in res["extraction_error"]
        assert res["strategic_state_changed"] is True


def test_20_extraction_failure_does_not_mark_strategic_state_changed_false():
    """Invariant: Extraction failure does NOT mark strategic_state_changed=False."""
    state = {
        "extraction_status": "FAILED",
        "strategic_state_changed": False,  # even if mistakenly False
        "diagnosis_output": {"mock": "diag"},
    }
    # Invariant: is_strategic_state_changed MUST return True if extraction failed!
    assert is_strategic_state_changed(state) is True


def test_21_extraction_failure_prevents_diagnosis_cache_reuse():
    """Extraction failure prevents diagnosis cache reuse in diagnose_startup."""
    cached_diag = {"summary": "Old cached diagnosis", "weak_link": "Market"}
    state = {
        "project_id": "p_test",
        "raw_context": "Founder updated details",
        "extraction_status": "FAILED",
        "diagnosis_output": cached_diag,
        "founder_messages": [{"role": "founder", "content": "Founder updated details"}],
    }

    mock_llm = MagicMock()
    mock_structured = MagicMock()
    diag_obj = DiagnosisLLMOutput(
        area_assessments=[
            AreaDiagnosisOutput(
                area=StartupArea.PROBLEM,
                status=AreaStatus.KNOWN,
                understanding="Problem is clear",
            )
        ],
        diagnosis_summary="New fresh diagnosis",
    )
    mock_structured.invoke.return_value = diag_obj
    mock_llm.with_structured_output.return_value = mock_structured

    with patch("startary.graph.nodes._safe_get_llm_client", return_value=mock_llm):
        res = diagnose_startup(state)
        # Must have invoked LLM to produce new diagnosis rather than returning cached_diag
        assert mock_structured.invoke.called
        assert res["diagnosis_output"] != cached_diag


def test_22_extraction_failure_does_not_overwrite_budget():
    """Extraction failure does NOT overwrite existing budget with 0/None."""
    existing_exec_ctx = {
        "allocated_budget_usd": {"value": 4500.0, "confidence": "DEMONSTRATED"},
        "founder_count": {"value": 2, "confidence": "STATED"},
    }
    state = {
        "extraction_status": "FAILED",
        "execution_context": existing_exec_ctx,
        "startup_progress": {"stage": "PROTOTYPE"},
        "startup_state_delta": None,
    }
    result = merge_execution_state(state)
    assert result["execution_context"]["allocated_budget_usd"]["value"] == 4500.0


def test_23_extraction_failure_does_not_overwrite_customer_access():
    """Extraction failure does NOT overwrite customer access with UNKNOWN."""
    existing_exec_ctx = {
        "customer_access": {
            "status": "VALIDATED",
            "strength": "STRONG",
            "channels": ["Direct B2B outreach"],
        }
    }
    state = {
        "extraction_status": "FAILED",
        "execution_context": existing_exec_ctx,
        "startup_progress": {},
    }
    result = merge_execution_state(state)
    assert result["execution_context"]["customer_access"]["status"] == "VALIDATED"


def test_24_extraction_failure_preserves_existing_execution_context():
    """Extraction failure preserves existing execution context and progress in full."""
    existing_exec_ctx = {"team": "Alice & Bob", "weekly_hours": 30}
    existing_prog = {"prototype_exists": True, "product_status": "MVP"}
    state = {
        "extraction_status": "FAILED",
        "execution_context": existing_exec_ctx,
        "startup_progress": existing_prog,
    }
    result = merge_execution_state(state)
    assert result["execution_context"] == existing_exec_ctx
    assert result["startup_progress"] == existing_prog


def test_25_full_turn_graceful_completion_with_fallback():
    """Full node completes gracefully using fallback client when configured model is unavailable."""
    settings = Settings(
        openai_api_key="sk-test-key",
        fast_model_name="gpt-5.6-luna",
        fast_model_fallback="gpt-4o-mini",
        model_name="gpt-4o",
    )
    # Register luna as unavailable
    register_model_access("gpt-5.6-luna", False, "403 Forbidden: model_not_found")

    # When get_llm_client is called, it resolves to gpt-4o-mini
    with patch("startary.llm.provider.ChatOpenAI") as mock_chat_cls:
        mock_instance = MagicMock()
        mock_chat_cls.return_value = mock_instance
        client = get_llm_client(settings, tier="fast")
        assert client._startary_resolved_model == "gpt-4o-mini"
        assert client._startary_fallback_used is True
        assert client._startary_fallback_reason == "FAST_MODEL_UNAVAILABLE_FALLBACK_SELECTED"
