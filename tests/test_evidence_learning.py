"""
Unit and integration tests for Milestone 4: Evidence Intake, Evidence Analysis, and Re-Diagnosis.

Tests cover:
1. Domain model contracts (Evidence, EvidenceAnalysis, LearningDelta)
2. Evidence Extractor (concrete counts, quotes, future-intention detection)
3. Evidence Analyzer (SUPPORT, CONTRADICTION, INCONCLUSIVE, invalid inferences)
4. Deterministic State Updates (bounded updates, assumption status, iteration increment)
5. Re-Diagnosis & Weak-Link Shift
6. Graph routing (CAN_EXECUTE -> WAITING_FOR_EVIDENCE, interrupt, re-pause safety)
7. CV-001 UNDERSTAND / TEST boundary protection
"""

import json
from unittest.mock import MagicMock
import pytest

from startary.domain.enums import (
    AssumptionStatus,
    CriticDimensionResult,
    CriticVerdict,
    EvidenceAnalysisConclusion,
    EvidenceProvenance,
    EvidenceResult,
    EvidenceSourceType,
    EvidenceStrength,
    EvidenceSubmissionSufficiency,
    EvidenceType,
    ExecutionQuality,
    ExecutionReadiness,
    ExperimentStatus,
    ExperimentType,
    WorkflowPhase,
)
from startary.domain.evidence import (
    Evidence,
    EvidenceAnalysis,
    LearningDelta,
)
from startary.domain.experiment import CanExecuteAssessment
from startary.domain.taxonomy import StartupArea
from startary.graph.builder import (
    build_graph,
    route_after_can_execute,
    route_on_evidence_sufficiency,
    run_evidence_learning_pipeline,
)
from startary.graph.graph_state import StartaryGraphState
from startary.graph.nodes import (
    analyze_evidence,
    apply_validated_state_updates,
    assess_can_execute,
    assess_evidence_submission_sufficiency,
    compose_learning_delta,
    extract_experiment_evidence,
    increment_iteration,
    validate_evidence_analysis,
)
from startary.llm.schemas import (
    EvidenceAnalysisLLMOutput,
    ExtractedEvidenceItemLLMOutput,
    ExtractedEvidenceLLMOutput,
)


@pytest.fixture
def m4_base_state() -> StartaryGraphState:
    return {
        "project_id": "proj-shiftly-m4",
        "phase": WorkflowPhase.TEST,
        "iteration": 1,
        "raw_context": "Shiftly helps small café managers manage shift changes easily.",
        "weak_area": "Problem",
        "selection_confidence": "HIGH",
        "active_assumption": {
            "id": "asmp-shiftly-001",
            "statement": "Small café managers experience recurring operational disruption from cancellations or shift swaps that current workflows handle poorly.",
            "area": "Problem",
            "status": "UNTESTED",
            "is_critical": True,
            "iteration_created": 1,
        },
        "active_critical_assumption": {
            "id": "asmp-shiftly-001",
            "statement": "Small café managers experience recurring operational disruption from cancellations or shift swaps that current workflows handle poorly.",
            "area": "Problem",
            "status": "UNTESTED",
            "is_critical": True,
            "iteration_created": 1,
        },
        "active_experiment": {
            "id": "exp-shiftly-001",
            "assumption_id": "asmp-shiftly-001",
            "project_id": "proj-shiftly-m4",
            "experiment_type": "PROBLEM_INTERVIEW",
            "objective": "Test whether small café managers experience recurring operational disruption from staff cancellations.",
            "method": "Conduct 5-8 structured customer problem interviews with independent café managers.",
            "support_signal": "At least 5 of 8 managers report shift cancellations cause operational disruption at least twice a month.",
            "contradiction_signal": "At least 5 of 8 managers report cancellations are rare, easily handled via WhatsApp, or not a meaningful problem.",
            "inconclusive_signal": "Fewer than 4 managers interviewed, or managers report ambiguous workarounds.",
            "valid_inferences": ["Frequency of cancellation disruption", "Current manual workarounds used"],
            "invalid_inferences": ["Willingness to pay $29/mo", "Customer acquisition CAC", "Retention rate"],
            "status": "READY",
            "created_iteration": 1,
        },
        "startup_state": {
            "project_id": "proj-shiftly-m4",
            "raw_context": "Shiftly helps small café managers manage shift changes easily.",
            "area_knowledge": {
                "Problem": {
                    "area": "Problem",
                    "status": "PARTIALLY_KNOWN",
                    "understanding": "Founder claims small café managers suffer from shift swaps.",
                    "founder_claims": ["Café managers lose hours coordinating shift swaps."],
                    "evidence_provided": [],
                    "missing_information": "Real frequency and severity of cancellation disruption.",
                },
                "Target User": {
                    "area": "Target User",
                    "status": "PARTIALLY_KNOWN",
                    "understanding": "Small independent café managers.",
                    "founder_claims": ["Focusing on cafés with 5-15 staff."],
                    "evidence_provided": [],
                    "missing_information": None,
                },
                "Demand Signal": {
                    "area": "Demand Signal",
                    "status": "UNKNOWN",
                    "understanding": "No demand evidence collected yet.",
                    "founder_claims": [],
                    "evidence_provided": [],
                    "missing_information": "Whether managers want a software solution.",
                },
            },
        },
        "execution_context": {
            "founder_count": {"value": "2", "provenance": "EXPLICIT", "confidence": "HIGH"},
            "available_budget_amount": {"value": "500", "provenance": "EXPLICIT", "confidence": "HIGH"},
        },
        "startup_progress": {
            "mvp_exists": {"value": "false", "provenance": "EXPLICIT", "confidence": "HIGH"},
        },
        "founder_messages": [
            {"role": "founder", "content": "Shiftly helps café managers manage shift swaps."}
        ],
        "founder_evidence": [],
        "external_evidence": [],
        "active_evidence": None,
        "extracted_evidence": None,
        "evidence_clarification_turns": 0,
        "evidence_analysis": None,
        "learning_delta": None,
        "previous_diagnosis_snapshot": None,
        "previous_assumptions": [],
    }


# ---------------------------------------------------------------------------
# 1. Domain Models Tests
# ---------------------------------------------------------------------------

def test_evidence_schema_validates():
    """Evidence model validates with mandatory linkage and preserves raw content."""
    e = Evidence(
        project_id="proj-123",
        experiment_id="exp-001",
        assumption_id="asmp-001",
        iteration=1,
        evidence_type=EvidenceType.INTERVIEW_NOTES,
        raw_content="Interviewed 7 managers. 5 confirmed recurring scheduling disruptions.",
        provenance=EvidenceProvenance.FOUNDER_REPORTED,
    )
    assert e.experiment_id == "exp-001"
    assert e.assumption_id == "asmp-001"
    assert e.iteration == 1
    assert "7 managers" in e.raw_content
    assert e.content == e.raw_content
    assert e.result == EvidenceResult.PENDING


def test_evidence_analysis_model_validates():
    """EvidenceAnalysis validates conclusion, strength, and inference boundaries."""
    analysis = EvidenceAnalysis(
        conclusion=EvidenceAnalysisConclusion.SUPPORTED,
        evidence_strength=EvidenceStrength.MODERATE,
        support_findings=["5 of 7 managers reported frequent cancellations."],
        contradiction_findings=["2 managers said current chat groups are fine."],
        inconclusive_findings=[],
        experiment_execution_quality=ExecutionQuality.GOOD,
        valid_learnings=["Problem is frequent in small cafés with 5-15 staff."],
        invalid_learnings=["Does not prove willingness to pay."],
        assumption_confidence_change="Substantially strengthened belief in problem severity.",
        reasoning_summary="5/7 matches the predefined support threshold (>=5 of 8).",
    )
    assert analysis.conclusion == EvidenceAnalysisConclusion.SUPPORTED
    assert analysis.evidence_strength == EvidenceStrength.MODERATE
    assert len(analysis.support_findings) == 1
    assert len(analysis.invalid_learnings) == 1


def test_learning_delta_model_validates():
    """LearningDelta synthesizes difference between iterations."""
    delta = LearningDelta(
        previous_assumption="Small café managers experience disruption from cancellations.",
        evidence_conclusion="SUPPORTED",
        what_we_learned=["Disruption occurs 2-3 times per month in 5 of 7 venues."],
        what_became_more_certain=["Problem frequency is high."],
        what_became_less_certain=[],
        what_remains_unknown=["Will managers pay for dedicated software?"],
        weak_link_changed=True,
        previous_weak_link="Problem",
        new_weak_link="Demand Signal",
        founder_facing_summary="Problem confirmed. Bottleneck shifts to Demand Signal.",
    )
    assert delta.weak_link_changed is True
    assert delta.previous_weak_link == "Problem"
    assert delta.new_weak_link == "Demand Signal"


# ---------------------------------------------------------------------------
# 2. Evidence Extractor Tests
# ---------------------------------------------------------------------------

def test_extract_concrete_counts_and_behaviors(mocker, m4_base_state):
    """Evidence Extractor captures sample numbers, behaviors, and quotes."""
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = ExtractedEvidenceLLMOutput(
        actions_completed=["Conducted 7 interviews with café managers"],
        sample_or_counts={"interviewed": 7, "positive": 5, "negative": 2},
        observed_behaviors=["Managers covered shifts themselves", "Used WhatsApp group"],
        quotes=["I lost 3 hours last Friday chasing replacements"],
        quantitative_outcomes={"frequency_per_month": 2.5},
        incomplete_actions=["Did not interview 8th planned manager"],
        deviations_from_plan=[],
        missing_result_information=[],
        is_future_intention_only=False,
        evidence_items=[
            ExtractedEvidenceItemLLMOutput(observation="5/7 experienced cancellations", evidence_type="OBSERVATION")
        ],
        direct_evidence_summary="7 managers interviewed; 5 confirmed recurring operational disruption.",
    )
    mock_llm.with_structured_output.return_value = mock_structured
    mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
    mocker.patch("startary.graph.nodes.get_settings", return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"))

    m4_base_state["founder_response"] = (
        "I interviewed 7 café managers. 5 of them said cancellations disrupt operations at least twice a month. "
        "Two said WhatsApp works fine for them. One manager said 'I lost 3 hours last Friday chasing replacements'."
    )

    res = extract_experiment_evidence(m4_base_state)
    extracted = res["extracted_evidence"]

    assert extracted["is_future_intention_only"] is False
    assert extracted["sample_or_counts"]["interviewed"] == 7
    assert len(extracted["observed_behaviors"]) == 2
    assert "lost 3 hours" in extracted["quotes"][0]


def test_future_intention_not_treated_as_evidence(mocker, m4_base_state):
    """Promise or plan ('I will do it') is flagged as is_future_intention_only."""
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = ExtractedEvidenceLLMOutput(
        actions_completed=[],
        sample_or_counts={},
        observed_behaviors=[],
        quotes=[],
        quantitative_outcomes={},
        incomplete_actions=["All interviews still in future"],
        deviations_from_plan=[],
        missing_result_information=["No completed results"],
        is_future_intention_only=True,  # FUTURE INTENTION DETECTED
        evidence_items=[],
        direct_evidence_summary="Founder states plan to interview café owners tomorrow.",
    )
    mock_llm.with_structured_output.return_value = mock_structured
    mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
    mocker.patch("startary.graph.nodes.get_settings", return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"))

    m4_base_state["founder_response"] = "I will go visit 5 cafés tomorrow morning and ask them."
    res = extract_experiment_evidence(m4_base_state)
    assert res["extracted_evidence"]["is_future_intention_only"] is True

    # Check sufficiency routing
    m4_base_state["extracted_evidence"] = res["extracted_evidence"]
    suff = assess_evidence_submission_sufficiency(m4_base_state)
    assert suff["evidence_submission_sufficiency"] == EvidenceSubmissionSufficiency.NEEDS_CLARIFICATION.value
    assert "plan" in suff["evidence_clarification_question"].lower()


# ---------------------------------------------------------------------------
# 3. Evidence Analyzer Tests
# ---------------------------------------------------------------------------

def test_evidence_analysis_support_case(mocker, m4_base_state):
    """Evidence matching support signal produces SUPPORTED / PARTIALLY_SUPPORTED."""
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = EvidenceAnalysisLLMOutput(
        conclusion="PARTIALLY_SUPPORTED",
        evidence_strength="MODERATE",
        experiment_execution_quality="GOOD",
        support_findings=["5 of 7 managers confirmed repeated disruption."],
        contradiction_findings=["2 managers reported WhatsApp is completely adequate."],
        inconclusive_findings=[],
        deviations_from_plan=["Interviewed 7 instead of 8 planned."],
        valid_learnings=["Problem exists frequently for small independent venues."],
        invalid_learnings=["Does not indicate willingness to pay for a software tool."],
        assumption_confidence_change="Moderately strengthened problem validity.",
        reasoning_summary="5/7 positive responses satisfies the threshold for directional support.",
        recommended_state_updates={"Problem": "Verified frequent cancellation pain in 5/7 venues."},
    )
    mock_llm.with_structured_output.return_value = mock_structured
    mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
    mocker.patch("startary.graph.nodes.get_settings", return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"))

    m4_base_state["extracted_evidence"] = {
        "sample_or_counts": {"interviewed": 7, "disrupted": 5},
        "observed_behaviors": ["Chasing staff replacements"],
    }
    m4_base_state["founder_response"] = "7 managers interviewed, 5 disrupted, 2 said WhatsApp is fine."

    res = analyze_evidence(m4_base_state)
    analysis = res["evidence_analysis"]

    assert analysis["conclusion"] == "PARTIALLY_SUPPORTED"
    assert analysis["evidence_strength"] == "MODERATE"
    assert "willingness to pay" in analysis["invalid_learnings"][0].lower()


def test_evidence_analysis_contradiction_case(mocker, m4_base_state):
    """Evidence matching contradiction signal produces CONTRADICTED."""
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = EvidenceAnalysisLLMOutput(
        conclusion="CONTRADICTED",
        evidence_strength="STRONG",
        experiment_execution_quality="GOOD",
        support_findings=[],
        contradiction_findings=["6 of 8 managers said cancellations are rare and WhatsApp is fully sufficient."],
        inconclusive_findings=[],
        deviations_from_plan=[],
        valid_learnings=["The problem is not a major operational pain in this café profile."],
        invalid_learnings=[],
        assumption_confidence_change="Significantly weakened problem assumption.",
        reasoning_summary="6/8 managers contradicted the assumption, matching the contradiction signal.",
        recommended_state_updates={},
    )
    mock_llm.with_structured_output.return_value = mock_structured
    mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
    mocker.patch("startary.graph.nodes.get_settings", return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"))

    res = analyze_evidence(m4_base_state)
    analysis = res["evidence_analysis"]

    assert analysis["conclusion"] == "CONTRADICTED"
    assert len(analysis["contradiction_findings"]) == 1


def test_evidence_analysis_inconclusive_case(mocker, m4_base_state):
    """Vague or insufficient evidence produces INCONCLUSIVE."""
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = EvidenceAnalysisLLMOutput(
        conclusion="INCONCLUSIVE",
        evidence_strength="WEAK",
        experiment_execution_quality="POOR",
        support_findings=[],
        contradiction_findings=[],
        inconclusive_findings=["Only 2 friends interviewed; no behavioral observations."],
        deviations_from_plan=["Did not interview target café managers."],
        valid_learnings=[],
        invalid_learnings=["Casual polite praise cannot validate problem."],
        assumption_confidence_change="Unchanged due to insufficient sample and lack of rigor.",
        reasoning_summary="Sample size of 2 non-qualifying contacts is inconclusive.",
        recommended_state_updates={},
    )
    mock_llm.with_structured_output.return_value = mock_structured
    mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
    mocker.patch("startary.graph.nodes.get_settings", return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"))

    res = analyze_evidence(m4_base_state)
    analysis = res["evidence_analysis"]

    assert analysis["conclusion"] == "INCONCLUSIVE"
    assert analysis["experiment_execution_quality"] == "POOR"


# ---------------------------------------------------------------------------
# 4. Deterministic State Updates & Validation Tests
# ---------------------------------------------------------------------------

def test_assumption_status_updates_deterministically(m4_base_state):
    """validate_evidence_analysis maps conclusion to AssumptionStatus and updates experiment."""
    m4_base_state["evidence_analysis"] = {
        "conclusion": "SUPPORTED",
        "evidence_strength": "MODERATE",
        "reasoning_summary": "5/7 managers confirmed.",
    }
    m4_base_state["founder_response"] = "5/7 managers confirmed cancellations are disruptive."

    res = validate_evidence_analysis(m4_base_state)

    assert res["active_assumption"]["status"] == AssumptionStatus.SUPPORTED.value
    assert res["active_experiment"]["status"] == ExperimentStatus.COMPLETED.value
    assert len(res["founder_evidence"]) == 1
    assert res["founder_evidence"][0]["raw_content"] == m4_base_state["founder_response"]


def test_unrelated_startup_fields_remain_untouched(m4_base_state):
    """apply_validated_state_updates only modifies the tested area, preserving Team/Budget."""
    m4_base_state["evidence_analysis"] = {
        "conclusion": "SUPPORTED",
        "support_findings": ["Disruption happens 2x a month."],
        "reasoning_summary": "Supported by 5 interviews.",
    }

    initial_founder_count = m4_base_state["execution_context"]["founder_count"]["value"]
    initial_budget = m4_base_state["execution_context"]["available_budget_amount"]["value"]

    res = apply_validated_state_updates(m4_base_state)
    updated_st = res["startup_state"]

    # Problem area updated
    assert updated_st["area_knowledge"]["Problem"]["status"] == "KNOWN"
    assert len(updated_st["area_knowledge"]["Problem"]["evidence_provided"]) == 1

    # Unrelated execution context remains untouched
    assert m4_base_state["execution_context"]["founder_count"]["value"] == initial_founder_count
    assert m4_base_state["execution_context"]["available_budget_amount"]["value"] == initial_budget


def test_iteration_increments_and_preserves_history(m4_base_state):
    """increment_iteration advances iteration counter and archives previous assumption."""
    m4_base_state["active_assumption"]["status"] = "SUPPORTED"
    m4_base_state["evidence_analysis"] = {"conclusion": "SUPPORTED"}
    m4_base_state["founder_response"] = "5/7 managers confirmed cancellations are disruptive."
    res = increment_iteration(m4_base_state)

    assert res["iteration"] == 2
    assert res["phase"] == WorkflowPhase.LEARN
    assert len(res["previous_assumptions"]) == 1
    assert res["previous_assumptions"][0]["id"] == "asmp-shiftly-001"
    assert res["active_assumption"] is None
    assert res["active_experiment"] is None


def test_learning_delta_synthesized(m4_base_state):
    """compose_learning_delta synthesizes accurate before-and-after comparison."""
    m4_base_state["iteration"] = 2
    m4_base_state["founder_evidence"] = [{"id": "ev-test", "iteration": 1, "result": "ANALYZED"}]
    m4_base_state["startup_state"]["area_knowledge"]["Problem"]["evidence_provided"] = ["Interviewed 7 managers"]
    m4_base_state["previous_assumptions"] = [m4_base_state["active_assumption"]]
    m4_base_state["previous_weak_area"] = "Problem"
    m4_base_state["weak_area"] = "Demand Signal"
    m4_base_state["evidence_analysis"] = {
        "conclusion": "SUPPORTED",
        "evidence_strength": "MODERATE",
        "support_findings": ["5/7 managers confirmed pain."],
        "valid_learnings": ["Small cafés face real cancellation disruption."],
        "inconclusive_findings": ["Still unknown whether they will pay."],
    }

    res = compose_learning_delta(m4_base_state)
    delta = res["learning_delta"]

    assert delta["weak_link_changed"] is True
    assert delta["previous_weak_link"] == "Problem"
    assert delta["new_weak_link"] == "Demand Signal"
    assert "Demand Signal" in delta["founder_facing_summary"]


# ---------------------------------------------------------------------------
# 5. Graph Routing & Boundary Tests
# ---------------------------------------------------------------------------

def test_can_execute_routes_to_waiting_for_evidence(m4_base_state):
    """assess_can_execute with CAN_EXECUTE routes to compose_execution_handoff (Milestone 4.0.3)."""
    m4_base_state["can_execute_result"] = {
        "readiness": ExecutionReadiness.CAN_EXECUTE.value,
        "reason": "Founder can run test immediately.",
    }
    next_step = route_after_can_execute(m4_base_state)
    assert next_step == "compose_execution_handoff"


def test_needs_help_routes_to_pause(m4_base_state):
    """assess_can_execute with NEEDS_HELP routes to pause_needs_help."""
    m4_base_state["can_execute_result"] = {
        "readiness": ExecutionReadiness.NEEDS_HELP.value,
        "missing_requirements": ["Customer access"],
    }
    next_step = route_after_can_execute(m4_base_state)
    assert next_step == "pause_needs_help"


def test_evidence_sufficiency_routes():
    """route_on_evidence_sufficiency routes to clarification if needed and turns < 2."""
    state_clarify: StartaryGraphState = {
        "evidence_submission_sufficiency": EvidenceSubmissionSufficiency.NEEDS_CLARIFICATION.value,
        "evidence_clarification_turns": 0,
    }
    assert route_on_evidence_sufficiency(state_clarify) == "ask_evidence_clarification"

    state_done: StartaryGraphState = {
        "evidence_submission_sufficiency": EvidenceSubmissionSufficiency.SUFFICIENT_FOR_ANALYSIS.value,
        "evidence_clarification_turns": 0,
    }
    assert route_on_evidence_sufficiency(state_done) == "analyze_evidence"


def test_cv001_boundary_rule_in_prompts():
    """Verify CV-001 boundary rule is present in UNDERSTAND and weak-link planner prompts."""
    from startary.llm.prompts import (
        INFORMATION_GAP_PLANNER_SYSTEM_PROMPT,
        WEAK_LINK_CLARIFYING_QUESTION_SYSTEM_PROMPT,
    )
    # Check UNDERSTAND planner
    assert "UNDERSTAND BOUNDARY RULE (CV-001)" in INFORMATION_GAP_PLANNER_SYSTEM_PROMPT
    assert "NEVER ask the founder to go interview customers" in INFORMATION_GAP_PLANNER_SYSTEM_PROMPT
    assert "information the founder ALREADY KNOWS" in INFORMATION_GAP_PLANNER_SYSTEM_PROMPT

    # Check Weak-Link question planner
    assert "UNDERSTAND BOUNDARY RULE (CV-001)" in WEAK_LINK_CLARIFYING_QUESTION_SYSTEM_PROMPT
    assert "Do NOT ask the founder to:" in WEAK_LINK_CLARIFYING_QUESTION_SYSTEM_PROMPT
    assert "run interviews" in WEAK_LINK_CLARIFYING_QUESTION_SYSTEM_PROMPT


def test_execution_quality_separated_from_conclusion():
    """Execution quality is tracked separately from hypothesis validity."""
    analysis = EvidenceAnalysis(
        conclusion=EvidenceAnalysisConclusion.PARTIALLY_SUPPORTED,
        evidence_strength=EvidenceStrength.WEAK,
        support_findings=["3 managers confirmed cancellation pain."],
        contradiction_findings=[],
        inconclusive_findings=["Only 3 interviews completed out of 8 planned."],
        experiment_execution_quality=ExecutionQuality.PARTIAL,
        deviations_from_plan=["Stopped after 3 interviews due to founder time constraints."],
        valid_learnings=["Directional pain reported in small venues."],
        invalid_learnings=[],
        assumption_confidence_change="Slight directional increase.",
        reasoning_summary="Execution was partial (3/8), but observed instances were real.",
    )
    assert analysis.experiment_execution_quality == ExecutionQuality.PARTIAL
    assert analysis.conclusion == EvidenceAnalysisConclusion.PARTIALLY_SUPPORTED


def test_invalid_inferences_separated_from_valid_learnings():
    """Invalid inferences are tracked and prevented from bleeding into valid learnings."""
    analysis = EvidenceAnalysis(
        conclusion=EvidenceAnalysisConclusion.SUPPORTED,
        evidence_strength=EvidenceStrength.MODERATE,
        support_findings=["5/7 managers confirmed problem."],
        contradiction_findings=[],
        inconclusive_findings=[],
        experiment_execution_quality=ExecutionQuality.GOOD,
        valid_learnings=["Shift cancellation pain is frequent."],
        invalid_learnings=["Claiming users will pay $29/month is invalid for this interview method."],
        assumption_confidence_change="Strengthened problem validity.",
        reasoning_summary="Valid problem signals confirmed; pricing claim rejected.",
    )
    assert len(analysis.invalid_learnings) == 1
    assert "pricing claim rejected" in analysis.reasoning_summary.lower()


def test_learning_delta_contract_on_inconclusive(m4_base_state):
    """LearningDelta handles INCONCLUSIVE evidence cleanly without breaking state."""
    m4_base_state["iteration"] = 2
    m4_base_state["founder_evidence"] = [{"id": "ev-test", "iteration": 1, "result": "ANALYZED"}]
    m4_base_state["startup_state"]["area_knowledge"]["Problem"]["evidence_provided"] = ["Interviewed 2 friends"]
    m4_base_state["evidence_analysis"] = {
        "conclusion": "INCONCLUSIVE",
        "evidence_strength": "WEAK",
        "support_findings": [],
        "contradiction_findings": [],
        "inconclusive_findings": ["Insufficient sample; responses were ambiguous."],
        "valid_learnings": [],
    }
    m4_base_state["previous_assumptions"] = [m4_base_state["active_assumption"]]

    res = compose_learning_delta(m4_base_state)
    delta = res["learning_delta"]

    assert delta["evidence_conclusion"] == "INCONCLUSIVE"
    assert "inconclusive" in delta["founder_facing_summary"].lower()

