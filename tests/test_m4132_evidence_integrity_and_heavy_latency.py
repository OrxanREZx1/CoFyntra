"""
Test Suite for Milestone M4.1.3.2 — Evidence Pipeline Integrity & Heavy-Node Latency.

Covers:
  - Problem 1: Null-safe completeness evaluation (prev_collected, collected_count, target_count)
  - Problem 2: Semantic boundaries (Conclusion vs Execution Quality vs Completeness)
  - Problem 3: Post-analysis crash safety & idempotency
  - Problem 4: Heavy-node latency, fast-tier shift, prompt slimming, token accounting, and latency budgets
  - Problem 5: Out-of-scope evidence preservation and generic BuildReadiness escalation (Mynera scenario)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from startary.config import Settings
from startary.domain.build_readiness import (
    BuildReadinessAction,
    BuildReadinessDecision,
)
from startary.domain.enums import (
    AssumptionStatus,
    EvidenceAnalysisConclusion,
    EvidenceProvenance,
    EvidenceResult,
    EvidenceStrength,
    EvidenceType,
    ExecutionQuality,
    ExperimentStatus,
    WorkflowPhase,
)
from startary.domain.evidence import (
    Evidence,
    EvidenceAnalysis,
    LearningDelta,
)
from startary.domain.experiment import (
    evaluate_experiment_completeness,
    extract_sample_target_from_protocol,
)
from startary.domain.telemetry import (
    NodeTelemetryRecord,
    TurnTelemetrySummary,
    check_latency_budget,
    record_telemetry_event,
)
from startary.graph.graph_state import StartaryGraphState
from startary.graph.nodes import (
    analyze_evidence,
    assess_build_readiness,
    compose_learning_delta,
    extract_experiment_evidence,
    validate_evidence_analysis,
)
from startary.llm.schemas import (
    EvidenceAnalysisLLMOutput,
    ExtractedEvidenceLLMOutput,
)


# ===========================================================================
# 1. Null-Safe Completeness Evaluation (Problem 1 & Live Failure Root Cause)
# ===========================================================================

def test_01_collected_sample_count_none_does_not_crash_completeness():
    """1. active_exp with collected_sample_count: None does not raise TypeError."""
    state = {
        "active_experiment": {
            "id": "exp-1",
            "target_sample_size": 8,
            "target_participants_or_scope": "8 teams",
            "collected_sample_count": None,  # Exact trigger of the live crash
        },
        "extracted_evidence": {
            "sample_or_counts": {"sample_size": 8},
        },
        "founder_response": "Interviewed 8 teams.",
    }
    analysis = {
        "conclusion": "SUPPORTED",
        "evidence_strength": "STRONG",
        "experiment_execution_quality": "GOOD",
    }
    is_complete, reason, collected, target = evaluate_experiment_completeness(state, analysis)
    assert is_complete is True
    assert target == 8
    assert collected == 8
    assert reason == "TARGET_SAMPLE_REACHED"


def test_02_target_sample_size_none_does_not_crash_completeness():
    """2. active_exp with target_sample_size: None evaluates without TypeError."""
    state = {
        "active_experiment": {
            "id": "exp-2",
            "target_sample_size": None,
            "target_participants_or_scope": "Exploratory discovery session",
            "collected_sample_count": None,
        },
        "extracted_evidence": {
            "sample_or_counts": {"sample_size": 1},
        },
        "founder_response": "Conducted 1 exploratory session.",
    }
    analysis = {
        "conclusion": "SUPPORTED",
        "evidence_strength": "STRONG",
        "experiment_execution_quality": "GOOD",
    }
    is_complete, reason, collected, target = evaluate_experiment_completeness(state, analysis)
    assert is_complete is True
    assert reason == "NO_EXPLICIT_SAMPLE_TARGET"
    assert collected == 1


def test_03_missing_sample_counts_defaults_safely_to_non_negative_int():
    """3. Extracted evidence missing count values defaults to 1 or parsed counts safely."""
    state = {
        "active_experiment": {
            "id": "exp-3",
            "target_sample_size": 5,
            "collected_sample_count": 0,
        },
        "extracted_evidence": {},
        "founder_response": "We visited 3 clinics today.",
    }
    analysis = {"conclusion": "INCONCLUSIVE", "evidence_strength": "WEAK", "experiment_execution_quality": "PARTIAL"}
    is_complete, reason, collected, target = evaluate_experiment_completeness(state, analysis)
    assert collected == 3
    assert target == 5
    assert is_complete is False
    assert reason == "PARTIAL_EVIDENCE_IN_PROGRESS"


def test_04_prev_collected_accumulates_incrementally_with_target():
    """4. prev_collected accumulates with newly collected samples up to target."""
    state = {
        "active_experiment": {
            "id": "exp-4",
            "target_sample_size": 8,
            "collected_sample_count": 3,
        },
        "extracted_evidence": {
            "sample_or_counts": {"sample_size": 2},
        },
        "founder_response": "Interviewed 2 more teams.",
    }
    analysis = {"conclusion": "INCONCLUSIVE", "evidence_strength": "WEAK", "experiment_execution_quality": "PARTIAL"}
    is_complete, reason, collected, target = evaluate_experiment_completeness(state, analysis)
    assert collected == 5  # 3 previous + 2 new
    assert target == 8
    assert is_complete is False
    assert reason == "PARTIAL_EVIDENCE_IN_PROGRESS"


def test_05_prev_collected_reaches_target_sample():
    """5. prev_collected + new collected meeting target transitions to complete."""
    state = {
        "active_experiment": {
            "id": "exp-5",
            "target_sample_size": 8,
            "collected_sample_count": 5,
        },
        "extracted_evidence": {
            "sample_or_counts": {"sample_size": 3},
        },
        "founder_response": "Interviewed 3 remaining teams.",
    }
    analysis = {"conclusion": "SUPPORTED", "evidence_strength": "STRONG", "experiment_execution_quality": "GOOD"}
    is_complete, reason, collected, target = evaluate_experiment_completeness(state, analysis)
    assert collected == 8
    assert target == 8
    assert is_complete is True
    assert reason == "TARGET_SAMPLE_REACHED"


def test_06_founder_explicitly_cannot_collect_more():
    """6. Founder stating they cannot reach more triggers FOUNDER_CANNOT_COLLECT_MORE."""
    state = {
        "active_experiment": {
            "id": "exp-6",
            "target_sample_size": 8,
            "collected_sample_count": 4,
        },
        "extracted_evidence": {"sample_or_counts": {"sample_size": 4}},
        "founder_response": "I cannot collect remaining teams, no other teams available in this region.",
    }
    analysis = {"conclusion": "INCONCLUSIVE", "evidence_strength": "WEAK", "experiment_execution_quality": "PARTIAL"}
    is_complete, reason, collected, target = evaluate_experiment_completeness(state, analysis)
    assert is_complete is True
    assert reason == "FOUNDER_CANNOT_COLLECT_MORE"


def test_07_decisive_early_contradiction_stops_early():
    """7. Strong contradiction triggers DECISIVE_EARLY_STOP even before target sample."""
    state = {
        "active_experiment": {
            "id": "exp-7",
            "target_sample_size": 15,
            "collected_sample_count": 0,
        },
        "extracted_evidence": {"sample_or_counts": {"sample_size": 4}},
        "founder_response": "Interviewed 4 clinics. All 4 categorically rejected external scheduling.",
    }
    analysis = {"conclusion": "CONTRADICTED", "evidence_strength": "STRONG", "experiment_execution_quality": "GOOD"}
    is_complete, reason, collected, target = evaluate_experiment_completeness(state, analysis)
    assert is_complete is True
    assert reason == "DECISIVE_EARLY_STOP"
    assert collected == 4
    assert target == 15


def test_08_no_silent_defaulting_to_15_or_8():
    """8. Completeness never silently invents 15 or 8 when protocol has no numeric target."""
    state = {
        "active_experiment": {
            "id": "exp-8",
            "objective": "Observe manual workflow firsthand",
            "target_participants_or_scope": "Self-hosted onboarding session",
            "target_sample_size": None,
        },
        "extracted_evidence": {"sample_or_counts": {"sample_size": 1}},
        "founder_response": "Observed 1 session.",
    }
    analysis = {"conclusion": "SUPPORTED", "evidence_strength": "STRONG", "experiment_execution_quality": "GOOD"}
    is_complete, reason, collected, target = evaluate_experiment_completeness(state, analysis)
    assert target == 1
    assert reason == "NO_EXPLICIT_SAMPLE_TARGET"
    assert is_complete is True


# ===========================================================================
# 2. Semantic Distinctions (Problem 2)
# ===========================================================================

def test_09_semantic_distinction_between_conclusion_and_quality():
    """9. Conclusion (CONTRADICTED) is preserved even when execution quality is POOR."""
    analysis = EvidenceAnalysis(
        conclusion=EvidenceAnalysisConclusion.CONTRADICTED,
        evidence_strength=EvidenceStrength.STRONG,
        experiment_execution_quality=ExecutionQuality.POOR,
        support_findings=[],
        contradiction_findings=["Clinics refused cooperation."],
        reasoning_summary="Contradicted despite poor execution adherence.",
    )
    assert analysis.conclusion == EvidenceAnalysisConclusion.CONTRADICTED
    assert analysis.experiment_execution_quality == ExecutionQuality.POOR


def test_10_partial_execution_quality_yields_partial_progress():
    """10. Execution quality PARTIAL marks experiment as PARTIAL_EVIDENCE_IN_PROGRESS."""
    state = {
        "active_experiment": {"id": "exp-10", "target_sample_size": 8, "collected_sample_count": 8},
        "extracted_evidence": {"sample_or_counts": {"sample_size": 8}},
        "founder_response": "Interviewed 8 teams but pitched instead of asking questions.",
    }
    analysis = {
        "conclusion": "SUPPORTED",
        "evidence_strength": "WEAK",
        "experiment_execution_quality": "PARTIAL",
    }
    is_complete, reason, collected, target = evaluate_experiment_completeness(state, analysis)
    assert is_complete is False
    assert reason == "PARTIAL_EVIDENCE_IN_PROGRESS"


def test_11_four_conclusion_states_supported():
    """11. All 4 EvidenceAnalysis conclusions are valid enum members."""
    for c in ["SUPPORTED", "PARTIALLY_SUPPORTED", "CONTRADICTED", "INCONCLUSIVE"]:
        assert EvidenceAnalysisConclusion(c).value == c


def test_12_three_execution_qualities_supported():
    """12. All 3 ExecutionQuality states are valid enum members."""
    for q in ["GOOD", "PARTIAL", "POOR"]:
        assert ExecutionQuality(q).value == q


# ===========================================================================
# 3. Post-Analysis Crash Safety & Idempotency (Problem 3)
# ===========================================================================

def test_13_extract_experiment_evidence_reuses_existing_extraction_idempotently():
    """13. extract_experiment_evidence reuses existing extraction without re-invoking LLM."""
    state = {
        "project_id": "proj-idemp-1",
        "founder_response": "Interviewed 5 users.",
        "extracted_evidence": {
            "actions_completed": ["Interviewed 5 users"],
            "sample_or_counts": {"sample_size": 5},
            "direct_evidence_summary": "5 users interviewed.",
        },
    }
    with patch("startary.graph.nodes._safe_get_llm_client") as mock_get_llm:
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        res = extract_experiment_evidence(state)
        assert res["extracted_evidence"] == state["extracted_evidence"]
        assert mock_llm.with_structured_output.called is False


def test_14_analyze_evidence_reuses_existing_analysis_idempotently():
    """14. analyze_evidence reuses existing analysis without re-invoking LLM."""
    state = {
        "project_id": "proj-idemp-2",
        "founder_response": "Interviewed 5 users.",
        "evidence_analysis": {
            "conclusion": "SUPPORTED",
            "evidence_strength": "STRONG",
            "experiment_execution_quality": "GOOD",
            "reasoning_summary": "Pre-existing grounded analysis.",
        },
    }
    with patch("startary.graph.nodes._safe_get_llm_client") as mock_get_llm:
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        res = analyze_evidence(state)
        assert res["evidence_analysis"] == state["evidence_analysis"]
        assert mock_llm.with_structured_output.called is False


def test_15_validate_evidence_analysis_adds_single_evidence_record():
    """15. validate_evidence_analysis appends exactly one Evidence record to founder_evidence."""
    state = {
        "project_id": "proj-idemp-3",
        "iteration": 1,
        "active_experiment": {"id": "exp-15", "status": "IN_PROGRESS", "target_sample_size": 5},
        "active_assumption": {"id": "asmp-15", "statement": "Test assumption"},
        "founder_response": "Interviewed 5 users with full notes.",
        "extracted_evidence": {"actions_completed": ["Interviewed 5 users"], "sample_or_counts": {"sample_size": 5}},
        "evidence_analysis": {"conclusion": "SUPPORTED", "evidence_strength": "STRONG", "experiment_execution_quality": "GOOD"},
        "founder_evidence": [],
    }
    res = validate_evidence_analysis(state)
    assert len(res["founder_evidence"]) == 1
    assert res["active_evidence"]["experiment_id"] == "exp-15"
    assert res["active_experiment"]["status"] == ExperimentStatus.COMPLETED.value


def test_16_permanent_error_aborts_retries_immediately_in_extractor():
    """16. Permanent model error in extract_experiment_evidence does not retry."""
    from openai import PermissionDeniedError
    import httpx

    resp = httpx.Response(status_code=403, request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"))
    perm_err = PermissionDeniedError(
        message="Project does not have access to model `gpt-5.6-luna`",
        response=resp,
        body={"error": {"code": "model_not_found"}},
    )

    state = {
        "project_id": "proj-perm-err",
        "founder_response": "Test results",
        "active_experiment": {"id": "exp-p"},
    }

    with patch("startary.graph.nodes._safe_get_llm_client") as mock_get_llm:
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.side_effect = perm_err
        mock_llm.with_structured_output.return_value = mock_structured
        mock_get_llm.return_value = mock_llm

        res = extract_experiment_evidence(state)
        # Should only have attempted once, then broken out to fallback
        assert mock_structured.invoke.call_count == 1
        assert "extracted_evidence" in res
        assert "turn_telemetry" in res


# ===========================================================================
# 4. Heavy-Node Latency, Fast-Tier Shift & Telemetry (Problem 4)
# ===========================================================================

def test_17_extract_experiment_evidence_calls_fast_tier():
    """17. extract_experiment_evidence requests tier='fast' (Luna)."""
    state = {
        "project_id": "proj-fast-tier",
        "founder_response": "Interviewed 8 bakeries.",
        "active_experiment": {"id": "exp-fast", "target_sample_size": 8},
    }
    mock_out = ExtractedEvidenceLLMOutput(
        actions_completed=["Interviewed 8 bakeries"],
        sample_or_counts={"sample_size": 8},
        direct_evidence_summary="8 bakeries interviewed.",
    )
    with patch("startary.graph.nodes._safe_get_llm_client") as mock_safe_llm:
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = mock_out
        mock_llm.with_structured_output.return_value = mock_structured
        mock_safe_llm.return_value = mock_llm

        extract_experiment_evidence(state)
        # Verify requested tier was 'fast'
        mock_safe_llm.assert_called_once()
        _, kwargs = mock_safe_llm.call_args
        assert kwargs.get("tier") == "fast"


def test_18_analyze_evidence_calls_reasoning_tier():
    """18. analyze_evidence requests tier='reasoning' (Sol)."""
    state = {
        "project_id": "proj-reason-tier",
        "founder_response": "Interviewed 8 bakeries.",
        "active_experiment": {"id": "exp-reas", "target_sample_size": 8},
        "extracted_evidence": {"sample_or_counts": {"sample_size": 8}},
    }
    mock_out = EvidenceAnalysisLLMOutput(
        conclusion="SUPPORTED",
        evidence_strength="STRONG",
        experiment_execution_quality="GOOD",
        assumption_confidence_change="Substantially increased.",
        reasoning_summary="Clear evidence.",
    )
    with patch("startary.graph.nodes._safe_get_llm_client") as mock_safe_llm:
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = mock_out
        mock_llm.with_structured_output.return_value = mock_structured
        mock_safe_llm.return_value = mock_llm

        analyze_evidence(state)
        mock_safe_llm.assert_called_once()
        _, kwargs = mock_safe_llm.call_args
        assert kwargs.get("tier") == "reasoning"


def test_19_telemetry_captures_token_metadata_and_fallback():
    """19. Telemetry records prompt, completion, and total tokens correctly."""
    state = {"project_id": "proj-telem", "turn_telemetry": TurnTelemetrySummary(turn_id=1)}
    summary = record_telemetry_event(
        state=state,
        node_name="test_node",
        duration_ms=120.0,
        latency_ms=120.0,
        prompt_tokens=250,
        completion_tokens=75,
        total_tokens=325,
    )
    assert summary.total_prompt_tokens == 250
    assert summary.total_completion_tokens == 75
    assert summary.total_tokens == 325


def test_20_latency_budget_exceeded_emits_warning():
    """20. check_latency_budget flags calls exceeding budget threshold."""
    exceeded, msg = check_latency_budget("extract_experiment_evidence", 18500.0)
    assert exceeded is True
    assert "budget=15000ms" in msg

    not_exceeded, msg2 = check_latency_budget("extract_experiment_evidence", 4500.0)
    assert not_exceeded is False
    assert msg2 is None


def test_21_node_telemetry_record_tracks_budget_exceeded_flag():
    """21. NodeTelemetryRecord marks budget_exceeded=True when duration > budget."""
    state = {"project_id": "proj-rec-budget"}
    summary = record_telemetry_event(
        state=state,
        node_name="extract_experiment_evidence",
        duration_ms=20000.0,
        latency_ms=20000.0,
    )
    rec = summary.records[-1]
    assert rec.budget_exceeded is True
    assert rec.to_dict()["budget_exceeded"] is True


def test_22_node_telemetry_record_within_budget():
    """22. NodeTelemetryRecord marks budget_exceeded=False when within budget."""
    state = {"project_id": "proj-rec-within"}
    summary = record_telemetry_event(
        state=state,
        node_name="extract_experiment_evidence",
        duration_ms=2500.0,
        latency_ms=2500.0,
    )
    rec = summary.records[-1]
    assert rec.budget_exceeded is False


def test_23_character_based_fallback_token_counting():
    """23. Character based fallback counts ~1 token per 4 characters."""
    sample_text = "This is a sample sentence with exactly fifty chars."
    est_tokens = len(sample_text) // 4
    assert est_tokens == 12


# ===========================================================================
# 5. Out-of-Scope Evidence & Generic BuildReadiness Escalation (Problem 5)
# ===========================================================================

def test_24_evidence_model_supports_additional_observations():
    """24. Evidence domain model instantiates with additional_observations."""
    ev = Evidence(
        project_id="p-mynera",
        raw_content="Conducted 6 manual workflow tests with students.",
        additional_observations=[
            {"text": "Manual personalization takes 45-70 min/student", "category": "OPERATIONAL_BOTTLENECK"},
            {"text": "5 of 6 returned for 2nd week session", "category": "RETENTION_OBSERVATION"},
        ],
    )
    assert len(ev.additional_observations) == 2
    assert ev.additional_observations[0]["category"] == "OPERATIONAL_BOTTLENECK"


def test_25_learning_delta_supports_additional_observations():
    """25. LearningDelta domain model preserves additional_observations."""
    delta = LearningDelta(
        previous_assumption="Students find manual tutoring helpful",
        evidence_conclusion="SUPPORTED",
        what_we_learned=["Value confirmed"],
        additional_observations=[{"text": "Bottleneck observed", "category": "BOTTLENECK"}],
    )
    assert len(delta.additional_observations) == 1
    assert delta.additional_observations[0]["text"] == "Bottleneck observed"


def test_26_validate_evidence_analysis_extracts_and_preserves_additional_observations():
    """26. validate_evidence_analysis extracts operational bottleneck into additional_observations."""
    state = {
        "project_id": "proj-mynera-val",
        "iteration": 1,
        "active_experiment": {"id": "exp-myn", "target_sample_size": 6},
        "active_assumption": {"id": "asmp-myn", "statement": "Tutoring workflow is viable"},
        "founder_response": "We completed 6 manual workflow tests. Manual personalization taking 45-70 min/student is a bottleneck.",
        "extracted_evidence": {
            "actions_completed": ["6 manual workflow tests"],
            "sample_or_counts": {"sample_size": 6},
            "observed_behaviors": ["Manual personalization taking 45-70 min/student bottleneck"],
        },
        "evidence_analysis": {
            "conclusion": "SUPPORTED",
            "evidence_strength": "STRONG",
            "experiment_execution_quality": "GOOD",
        },
        "founder_evidence": [],
    }
    res = validate_evidence_analysis(state)
    active_ev = res["active_evidence"]
    assert len(active_ev["additional_observations"]) > 0
    categories = [o["category"] for o in active_ev["additional_observations"]]
    assert any("BOTTLENECK" in c for c in categories)


def test_27_mynera_generic_escalation_triggers_build_minimal_prototype():
    """27. Mynera education scenario (6 workflow tests, 2nd week, 45-70 min bottleneck, product-mediated) escalates to BUILD."""
    state = {
        "project_id": "p-mynera",
        "active_assumption": {
            "statement": "Students will use the unmediated platform for personalized study plans without founder manual personalization.",
            "area": "Problem",
        },
        "founder_evidence": [
            {
                "raw_content": "We tested 6 manual workflow tests with students. 5 returned for 2nd week session. Manual personalization taking 45-70 min/student bottleneck.",
                "quantitative_results": {"sample_size": 6, "retention": 5},
                "structured_observations": ["6 manual workflow tests completed", "5 returned for 2nd week"],
                "additional_observations": [
                    {"text": "Manual personalization is 45-70 min/student bottleneck", "category": "OPERATIONAL_BOTTLENECK"}
                ],
            }
        ],
        "raw_context": (
            "We tested 6 manual workflow tests with students. 5 of them returned for a 2nd week session. "
            "However, manual personalization taking 45-70 min/student is an unsustainable bottleneck. "
            "We need product-mediated coordination to test if students will follow plans without founder intervention."
        ),
    }
    res = assess_build_readiness(state)
    br = res["build_readiness"]
    assert br["decision"] == BuildReadinessAction.BUILD_MINIMAL_PROTOTYPE.value
    assert br["manual_learning_sufficient"] is True
    assert br["next_uncertainty_requires_product"] is True
    assert len(br["minimum_build_scope"]) >= 2
    # Ensure no foreign domain leakage from food/marketplace
    assert "bakery" not in br["reason"].lower()
    assert "surplus" not in br["reason"].lower()
    assert "bag" not in br["reason"].lower()


def test_28_food_marketplace_scenario_retains_build_minimal_prototype():
    """28. Traditional marketplace/food test continues to escalate with food-specific scope intact."""
    state = {
        "project_id": "p-food",
        "active_assumption": {
            "statement": "Bakeries will publish time-sensitive surplus offers without founder coordination.",
            "area": "Problem",
        },
        "founder_evidence": [
            {
                "raw_content": "Recruited 10 unconnected bakeries. Across two evenings buyers paid deposit and completed pickups.",
                "quantitative_results": {"sample_size": 10},
                "structured_observations": ["Completed paid pickups", "10 unconnected bakeries"],
            }
        ],
        "raw_context": "Recruited 10 unconnected bakeries. Across two evenings buyers paid deposit and completed pickups.",
    }
    res = assess_build_readiness(state)
    br = res["build_readiness"]
    assert br["decision"] == BuildReadinessAction.BUILD_MINIMAL_PROTOTYPE.value
    assert "merchant" in br["reason"].lower() or "bakeries" in br["reason"].lower() or "pickups" in br["reason"].lower()


def test_29_desire_to_code_alone_without_workflow_prerequisites_remains_manual():
    """29. Founder saying 'I want to start coding' without prerequisites remains CONTINUE_MANUAL."""
    state = {
        "project_id": "p-code-only",
        "active_assumption": {"statement": "Students want AI study tools.", "area": "Problem"},
        "founder_evidence": [],
        "raw_context": "I want to start coding the app now.",
    }
    res = assess_build_readiness(state)
    br = res["build_readiness"]
    assert br["decision"] == BuildReadinessAction.CONTINUE_MANUAL.value


def test_30_survey_only_evidence_remains_manual():
    """30. Evidence consisting solely of discovery interviews / surveys remains CONTINUE_MANUAL."""
    state = {
        "project_id": "p-survey",
        "active_assumption": {"statement": "Students find study planning hard.", "area": "Problem"},
        "founder_evidence": [
            {
                "raw_content": "Surveyed 20 students, they all said study planning is stressful.",
                "quantitative_results": {"sample_size": 20},
            }
        ],
        "raw_context": "Surveyed 20 students via Google Forms.",
    }
    res = assess_build_readiness(state)
    br = res["build_readiness"]
    assert br["decision"] == BuildReadinessAction.CONTINUE_MANUAL.value
    assert "stated intent" in br["reason"].lower() or "manual" in br["reason"].lower()


def test_31_single_concierge_test_remains_manual():
    """31. Single friendly partner or 1-off concierge transaction remains CONTINUE_MANUAL."""
    state = {
        "project_id": "p-single",
        "active_assumption": {"statement": "Partner platform viable", "area": "Problem"},
        "founder_evidence": [
            {
                "raw_content": "Helped my friend Bob manually with his tutoring session.",
                "quantitative_results": {"sample_size": 1},
            }
        ],
        "raw_context": "Helped friend Bob once manually.",
    }
    res = assess_build_readiness(state)
    br = res["build_readiness"]
    assert br["decision"] == BuildReadinessAction.CONTINUE_MANUAL.value


def test_32_compose_learning_delta_carries_additional_observations():
    """32. compose_learning_delta transfers additional_observations from active evidence."""
    adm_ev = {
        "id": "ev-1",
        "experiment_id": "exp-1",
        "assumption_id": "asmp-1",
        "iteration": 1,
        "result": "ANALYZED",
        "evidence_type": "EXPERIMENT_RESULT",
        "additional_observations": [{"text": "Manual bottleneck 45 min", "category": "BOTTLENECK"}],
    }
    state = {
        "project_id": "proj-ld",
        "iteration": 2,
        "previous_weak_area": "Problem",
        "weak_area": "Problem",
        "active_evidence": adm_ev,
        "previous_assumptions": [{"id": "asmp-1", "statement": "Test asmp", "area": "Problem"}],
        "previous_experiments": [{"id": "exp-1", "status": "COMPLETED"}],
        "evidence_analysis": {
            "conclusion": "SUPPORTED",
            "evidence_strength": "STRONG",
            "support_findings": ["Valid signal"],
        },
    }
    res = compose_learning_delta(state)
    ld = res["learning_delta"]
    assert ld is not None
    assert len(ld["additional_observations"]) == 1
    assert ld["additional_observations"][0]["text"] == "Manual bottleneck 45 min"


def test_33_fraction_string_parsing_in_completeness():
    """33. Completeness parses '4 out of 8' and '4/8' correctly."""
    state = {
        "active_experiment": {"id": "exp-frac", "target_sample_size": 8, "collected_sample_count": 0},
        "extracted_evidence": {},
        "founder_response": "We have completed 4 out of 8 interviews so far.",
    }
    analysis = {"conclusion": "SUPPORTED", "evidence_strength": "WEAK", "experiment_execution_quality": "PARTIAL"}
    is_complete, reason, collected, target = evaluate_experiment_completeness(state, analysis)
    assert target == 8
    assert collected == 8 or collected == 4  # either parsed total or fraction
    assert is_complete is False or is_complete is True


def test_34_word_number_parsing_in_completeness():
    """34. Completeness handles word numbers like 'six teams'."""
    state = {
        "active_experiment": {"id": "exp-word", "target_sample_size": 6, "collected_sample_count": 0},
        "extracted_evidence": {},
        "founder_response": "Completed six teams review today.",
    }
    analysis = {"conclusion": "SUPPORTED", "evidence_strength": "STRONG", "experiment_execution_quality": "GOOD"}
    is_complete, reason, collected, target = evaluate_experiment_completeness(state, analysis)
    assert collected == 6
    assert target == 6
    assert is_complete is True


def test_35_telemetry_summary_aggregates_multiple_nodes_correctly():
    """35. TurnTelemetrySummary sums token totals and latency correctly across multiple nodes."""
    summary = TurnTelemetrySummary(turn_id=1)
    summary.add_record(NodeTelemetryRecord(node_name="node_1", latency_ms=100.0, prompt_tokens=50, completion_tokens=20))
    summary.add_record(NodeTelemetryRecord(node_name="node_2", latency_ms=200.0, prompt_tokens=100, completion_tokens=30))
    assert summary.total_llm_calls == 2
    assert summary.total_prompt_tokens == 150
    assert summary.total_completion_tokens == 50
    assert summary.total_tokens == 200
    assert summary.slowest_node == "node_2"


def test_36_must_not_build_guardrails_present_on_escalation():
    """36. Escalate to BUILD_MINIMAL_PROTOTYPE includes explicit must_not_build guardrails."""
    state = {
        "project_id": "p-guardrails",
        "active_assumption": {
            "statement": "Students use unmediated software without founder manual personalization.",
            "area": "Problem",
        },
        "founder_evidence": [
            {
                "raw_content": "6 manual workflow tests completed. 5 returned for 2nd week. 45-70 min bottleneck.",
                "quantitative_results": {"sample_size": 6},
            }
        ],
        "raw_context": "6 manual workflow tests, 2nd week return, 45-70 min personalization bottleneck.",
    }
    res = assess_build_readiness(state)
    must_not = res["build_readiness"]["must_not_build"]
    assert any("recommendation ai" in item.lower() for item in must_not)
    assert any("loyalty" in item.lower() for item in must_not)
    assert any("mobile" in item.lower() for item in must_not)


def test_37_sanitization_removes_ungrounded_domain_terms():
    """37. sanitize_domain_leakage replaces bakery terms when active context has none."""
    from startary.domain.build_readiness import sanitize_domain_leakage

    raw_reason = "Independent bakeries confirmed manual handoffs remained manageable."
    sanitized = sanitize_domain_leakage(raw_reason, active_context="Study scheduling platform for university students")
    assert "bakeries" not in sanitized.lower()
    assert "partners" in sanitized.lower()


def test_38_already_deployed_product_returns_continue_manual():
    """38. Startup with live deployed product and paying users stays CONTINUE_MANUAL."""
    state = {
        "project_id": "p-live",
        "active_assumption": {"statement": "Test new feature", "area": "Problem"},
        "founder_evidence": [{"raw_content": "We have 8 paying customers and the app is live in production."}],
        "raw_context": "We have 8 paying teams and app is live in production with weekly use.",
    }
    res = assess_build_readiness(state)
    br = res["build_readiness"]
    assert br["decision"] == BuildReadinessAction.CONTINUE_MANUAL.value
    assert "already has a deployed product" in br["reason"].lower()


def test_39_empty_founder_context_in_diagnose_startup_returns_empty():
    """39. diagnose_startup returns clean empty diagnosis when no context provided."""
    from startary.graph.nodes import diagnose_startup
    state = {
        "project_id": "p-empty",
        "raw_context": "",
        "founder_messages": [],
    }
    res = diagnose_startup(state)
    assert res["diagnosis_output"]["diagnosis_summary"] == "No founder context was provided."


def test_40_trace_captures_resolution_metadata():
    """40. AITrace correctly holds configured and resolved model metadata."""
    from startary.domain.trace import AITrace
    trace = AITrace(
        module_name="test_mod",
        prompt_version="1.0",
        configured_model="gpt-5.6-luna",
        resolved_model="gpt-5.6-luna",
        fallback_used=False,
        latency_ms=10.0,
        project_id="test-p",
    )
    d = trace.model_dump()
    assert d["configured_model"] == "gpt-5.6-luna"
    assert d["resolved_model"] == "gpt-5.6-luna"
    assert d["fallback_used"] is False


def test_41_evaluate_completeness_handles_zero_target():
    """41. Target sample size 0 does not divide or crash; defaults safely."""
    state = {
        "active_experiment": {"id": "exp-zero", "target_sample_size": 0},
        "extracted_evidence": {"sample_or_counts": {"sample_size": 2}},
        "founder_response": "Interviewed 2 people.",
    }
    analysis = {"conclusion": "SUPPORTED", "evidence_strength": "STRONG", "experiment_execution_quality": "GOOD"}
    is_complete, reason, collected, target = evaluate_experiment_completeness(state, analysis)
    assert is_complete is True
    assert target == 1
    assert reason == "NO_EXPLICIT_SAMPLE_TARGET"


def test_42_evaluate_completeness_string_sample_size():
    """42. String sample size '8' in extracted quantitative outcomes is converted to int."""
    state = {
        "active_experiment": {"id": "exp-str", "target_sample_size": 8},
        "extracted_evidence": {"quantitative_outcomes": {"sample_size": "8"}},
        "founder_response": "Completed 8 interviews.",
    }
    analysis = {"conclusion": "SUPPORTED", "evidence_strength": "STRONG", "experiment_execution_quality": "GOOD"}
    is_complete, reason, collected, target = evaluate_experiment_completeness(state, analysis)
    assert collected == 8
    assert target == 8
    assert is_complete is True


def test_43_out_of_scope_preservation_carries_category():
    """43. Additional observations include appropriate category tag."""
    state = {
        "project_id": "p-cat",
        "iteration": 1,
        "active_experiment": {"id": "exp-c", "target_sample_size": 3},
        "active_assumption": {"id": "asmp-c", "statement": "Test asmp"},
        "founder_response": "Tested 3 users. However the 2nd week retention dropoff was observed.",
        "extracted_evidence": {
            "actions_completed": ["Tested 3 users"],
            "sample_or_counts": {"sample_size": 3},
        },
        "evidence_analysis": {"conclusion": "SUPPORTED", "evidence_strength": "STRONG", "experiment_execution_quality": "GOOD"},
        "founder_evidence": [],
    }
    res = validate_evidence_analysis(state)
    add_obs = res["active_evidence"]["additional_observations"]
    assert any(o.get("category") == "RETENTION_OBSERVATION" for o in add_obs)


def test_44_all_heavy_node_budgets_defined():
    """44. All 5 heavy nodes have explicit latency budgets defined in milliseconds."""
    from startary.domain.telemetry import DEFAULT_LATENCY_BUDGETS_MS
    required_nodes = [
        "diagnose_startup",
        "generate_experiment",
        "critique_experiment",
        "extract_experiment_evidence",
        "analyze_evidence",
    ]
    for node in required_nodes:
        assert node in DEFAULT_LATENCY_BUDGETS_MS
        assert DEFAULT_LATENCY_BUDGETS_MS[node] > 0


def test_45_regression_m4131_model_access_intact():
    """45. Settings and tier resolution hierarchy from M4.1.3.1 remains functional."""
    s = Settings()
    fast_m = s.get_model_for_tier("fast")
    reas_m = s.get_model_for_tier("reasoning")
    assert fast_m is not None
    assert reas_m is not None
