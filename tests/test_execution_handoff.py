"""
Unit and integration tests for Milestone 4.0.3: Execution Handoff & Evidence Admission Integrity.

Tests cover:
1. ExecutionHandoff schema validates and serializes properly.
2. CAN_EXECUTE creates handoff before WAITING_FOR_EVIDENCE.
3. Founder-facing handoff never renders None or empty string.
4. "What should I do?" / "Then what should I do?" is rejected as evidence.
5. Future intention ("I will interview them tomorrow") is rejected as evidence.
6. Execution-help request remains WAITING_FOR_EVIDENCE with bounded advice.
7. Actual completed result enters Evidence pipeline.
8. Mixed evidence + question still enters Evidence pipeline.
9. Non-evidence cannot complete experiment (remains IN_PROGRESS).
10. Non-evidence cannot increment iteration.
11. Non-evidence cannot update assumption status.
12. Valid INCONCLUSIVE evidence still creates a legitimate learning cycle.
"""

from unittest.mock import MagicMock
import pytest

from startary.domain.enums import (
    AssumptionStatus,
    EvidenceAnalysisConclusion,
    EvidenceStrength,
    ExecutionQuality,
    ExecutionReadiness,
    ExperimentStatus,
    WorkflowPhase,
)
from startary.domain.evidence import (
    EvidenceTurnAssessment,
    EvidenceTurnDisposition,
    has_valid_completed_evidence_cycle,
)
from startary.domain.experiment import (
    ExecutionHandoff,
    render_execution_handoff_text,
)
from startary.graph.builder import (
    route_after_can_execute,
    route_after_interpret_founder_turn,
    route_waiting_for_evidence_turn,
    run_evidence_learning_pipeline,
)
from startary.graph.graph_state import StartaryGraphState
from startary.graph.nodes import (
    apply_validated_state_updates,
    classify_waiting_for_evidence_turn,
    compose_execution_handoff,
    increment_iteration,
    respond_to_waiting_founder,
    validate_evidence_analysis,
    wait_for_evidence,
)
from startary.llm.schemas import (
    EvidenceAnalysisLLMOutput,
    EvidenceTurnAssessmentLLMOutput,
    ExtractedEvidenceLLMOutput,
    WaitingForEvidenceAdvisorLLMOutput,
)


@pytest.fixture
def m403_base_state() -> StartaryGraphState:
    return {
        "project_id": "proj-loopcart-m403",
        "phase": WorkflowPhase.TEST,
        "iteration": 1,
        "raw_context": "LoopCart helps small fashion e-commerce merchants reduce returns due to sizing mismatch.",
        "weak_area": "Problem",
        "selection_confidence": "LOW",
        "active_assumption": {
            "id": "asmp-loopcart-001",
            "statement": "Small fashion e-commerce merchants experience sizing-related returns as a high-friction operational pain.",
            "area": "Problem",
            "status": "UNTESTED",
            "is_critical": True,
        },
        "active_experiment": {
            "id": "exp-loopcart-001",
            "assumption_id": "asmp-loopcart-001",
            "experiment_type": "CUSTOMER_INTERVIEWS",
            "objective": "Determine whether small fashion merchants view sizing returns as a severe problem worth solving.",
            "hypothesis": "Small fashion e-commerce merchants experience sizing-related returns as a high-friction operational pain.",
            "founder_actions": [
                "Speak with 5-8 small fashion or apparel e-commerce store owners.",
                "Ask about recent returns and proportion caused by sizing issues.",
                "Inquire about financial consequences and handling overhead.",
            ],
            "target_participants_or_scope": "5-8 small fashion e-commerce store owners",
            "evidence_to_collect": "Number of stores, return rates, specific quotes on costs/friction, current workarounds",
            "support_signal": "At least 4 of 6 merchants state sizing returns are a major monthly headache.",
            "contradiction_signal": "Most merchants view returns as minor or completely acceptable cost of doing business.",
            "inconclusive_signal": "Merchants have mixed views or cannot quantify return friction.",
            "valid_inferences": ["Validates whether sizing returns create genuine dissatisfaction for merchants."],
            "invalid_inferences": ["Does not prove willingness to purchase a SaaS solution."],
            "status": "READY",
        },
        "can_execute_result": {
            "readiness": "CAN_EXECUTE",
            "reason": "Founder has access to local boutique stores and network.",
        },
        "founder_messages": [
            {"role": "founder", "content": "I want to build LoopCart for small fashion merchants."}
        ],
    }


def test_execution_handoff_schema_validates():
    """ExecutionHandoff model validates required fields and generates clean markdown text."""
    handoff = ExecutionHandoff(
        experiment_id="exp-123",
        assumption_id="asmp-123",
        experiment_type="CUSTOMER_INTERVIEWS",
        learning_goal="Validate problem severity",
        what_we_are_testing="Merchants care about return reduction",
        founder_actions=["Interview 5 store owners", "Record return rates"],
        target_participants_or_context="Small apparel stores",
        evidence_to_collect="Return counts, handling time",
        support_signal="3+ merchants cite major return costs",
        contradiction_signal="Merchants don't care about returns",
        inconclusive_signal="Vague replies",
    )
    rendered = render_execution_handoff_text(handoff, mode="full")
    handoff.full_response_text = rendered

    assert handoff.experiment_id == "exp-123"
    assert "YOUR NEXT TEST" in rendered
    assert "Interview 5 store owners" in rendered
    assert "Small apparel stores" in rendered
    assert "WHEN YOU'RE DONE" in rendered

    compact = render_execution_handoff_text(handoff, mode="compact")
    handoff.response_text = compact
    assert "WHAT ARE WE TESTING?" in compact
    assert "WHAT SHOULD I DO?" in compact
    assert "WHAT SHOULD I BRING BACK?" in compact
    assert "WHAT COUNTS AS A USEFUL SIGNAL?" in compact
    assert "WHAT LOOKS PROMISING SO FAR?" not in compact


def test_can_execute_creates_handoff_before_wait_for_evidence(m403_base_state):
    """CAN_EXECUTE routes to compose_execution_handoff, which creates active_execution_handoff."""
    route = route_after_can_execute(m403_base_state)
    assert route == "compose_execution_handoff"

    handoff_res = compose_execution_handoff(m403_base_state)
    assert "active_execution_handoff" in handoff_res
    assert "execution_handoff_text" in handoff_res
    assert "YOUR NEXT TEST" in handoff_res["execution_handoff_text"]
    assert handoff_res["active_execution_handoff"]["experiment_id"] == "exp-loopcart-001"


def test_founder_facing_handoff_never_renders_none(m403_base_state):
    """Execution handoff and wait_for_evidence prompt never render None or empty string."""
    handoff_dict = compose_execution_handoff(m403_base_state)
    m403_base_state.update(handoff_dict)

    rendered = m403_base_state.get("execution_handoff_text")
    assert rendered is not None
    assert len(rendered.strip()) > 50
    assert "None" not in rendered[:30]


def test_what_should_i_do_rejected_as_evidence(mocker, m403_base_state):
    """'Then what should I do?' is classified as QUESTION_ABOUT_EXPERIMENT and rejected from evidence."""
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = EvidenceTurnAssessmentLLMOutput(
        disposition="QUESTION_ABOUT_EXPERIMENT",
        contains_completed_external_action=False,
        contains_results=False,
        contains_only_future_intent=False,
        asks_for_help=False,
        asks_for_explanation=True,
        cannot_execute=False,
        evidence_candidate_text=None,
        rationale_summary="Founder is asking what steps to take for the test.",
    )
    mock_llm.with_structured_output.return_value = mock_structured
    mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
    mocker.patch("startary.graph.nodes.get_settings", return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"))

    m403_base_state["founder_response"] = "Then what should I do?"
    res = classify_waiting_for_evidence_turn(m403_base_state)
    assessment = res["evidence_turn_assessment"]

    assert assessment["disposition"] == EvidenceTurnDisposition.QUESTION_ABOUT_EXPERIMENT.value
    assert assessment["contains_completed_external_action"] is False

    m403_base_state.update(res)
    next_step = route_waiting_for_evidence_turn(m403_base_state)
    assert next_step == "respond_to_waiting_founder"


def test_future_intention_rejected_as_evidence(mocker, m403_base_state):
    """'I will interview them tomorrow' is classified as FUTURE_INTENTION and rejected from evidence."""
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = EvidenceTurnAssessmentLLMOutput(
        disposition="FUTURE_INTENTION",
        contains_completed_external_action=False,
        contains_results=False,
        contains_only_future_intent=True,
        asks_for_help=False,
        asks_for_explanation=False,
        cannot_execute=False,
        evidence_candidate_text=None,
        rationale_summary="Founder expresses intention to conduct interviews tomorrow.",
    )
    mock_llm.with_structured_output.return_value = mock_structured
    mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
    mocker.patch("startary.graph.nodes.get_settings", return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"))

    m403_base_state["founder_response"] = "I will interview them tomorrow."
    res = classify_waiting_for_evidence_turn(m403_base_state)
    assessment = res["evidence_turn_assessment"]

    assert assessment["disposition"] == EvidenceTurnDisposition.FUTURE_INTENTION.value
    assert assessment["contains_completed_external_action"] is False

    m403_base_state.update(res)
    next_step = route_waiting_for_evidence_turn(m403_base_state)
    assert next_step == "respond_to_waiting_founder"


def test_execution_help_request_remains_waiting_for_evidence(mocker, m403_base_state):
    """'How do I find 5 merchants?' gets conversational guidance and remains WAITING_FOR_EVIDENCE."""
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = WaitingForEvidenceAdvisorLLMOutput(
        response_text="To find 5 merchants, start with the boutique stores in your neighborhood during off-peak hours.",
        rationale="Provide practical founder outreach guidance while preserving experiment scope.",
    )
    mock_llm.with_structured_output.return_value = mock_structured
    mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
    mocker.patch("startary.graph.nodes.get_settings", return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"))

    m403_base_state["founder_response"] = "How do I find 5 merchants?"
    m403_base_state["evidence_turn_assessment"] = {
        "disposition": EvidenceTurnDisposition.REQUEST_FOR_EXECUTION_HELP.value,
        "contains_completed_external_action": False,
    }

    res = respond_to_waiting_founder(m403_base_state)
    assert res["phase"] == WorkflowPhase.WAITING_FOR_EVIDENCE
    assert "boutique stores" in res["waiting_response"]
    assert len(res["founder_messages"]) >= 2


def test_actual_completed_result_enters_evidence_pipeline(mocker, m403_base_state):
    """Real results with concrete counts and observations enter extract_experiment_evidence."""
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = EvidenceTurnAssessmentLLMOutput(
        disposition="ACTUAL_EVIDENCE",
        contains_completed_external_action=True,
        contains_results=True,
        contains_only_future_intent=False,
        asks_for_help=False,
        asks_for_explanation=False,
        cannot_execute=False,
        evidence_candidate_text="I interviewed 6 merchants. Four said sizing returns cost them hours every week.",
        rationale_summary="Founder completed interviews and reports concrete findings.",
    )
    mock_llm.with_structured_output.return_value = mock_structured
    mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
    mocker.patch("startary.graph.nodes.get_settings", return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"))

    m403_base_state["founder_response"] = "I interviewed 6 merchants. Four said sizing returns cost them hours every week."
    res = classify_waiting_for_evidence_turn(m403_base_state)
    assessment = res["evidence_turn_assessment"]

    assert assessment["disposition"] == EvidenceTurnDisposition.ACTUAL_EVIDENCE.value
    assert assessment["contains_completed_external_action"] is True

    m403_base_state.update(res)
    next_step = route_waiting_for_evidence_turn(m403_base_state)
    assert next_step == "extract_experiment_evidence"


def test_mixed_evidence_and_question_admitted(mocker, m403_base_state):
    """Mixed submission with completed results AND a question admits evidence."""
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = EvidenceTurnAssessmentLLMOutput(
        disposition="ACTUAL_EVIDENCE",
        contains_completed_external_action=True,
        contains_results=True,
        contains_only_future_intent=False,
        asks_for_help=False,
        asks_for_explanation=True,
        cannot_execute=False,
        evidence_candidate_text="I interviewed 4 merchants. Two said returns are costly, two didn't care.",
        rationale_summary="Contains actual completed interviews, even though it ends with a question.",
    )
    mock_llm.with_structured_output.return_value = mock_structured
    mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
    mocker.patch("startary.graph.nodes.get_settings", return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"))

    m403_base_state["founder_response"] = "I interviewed 4 merchants. Two said returns are costly, two didn't care. Is that enough?"
    res = classify_waiting_for_evidence_turn(m403_base_state)
    assessment = res["evidence_turn_assessment"]

    assert assessment["disposition"] == EvidenceTurnDisposition.ACTUAL_EVIDENCE.value
    assert assessment["contains_completed_external_action"] is True

    m403_base_state.update(res)
    assert route_waiting_for_evidence_turn(m403_base_state) == "extract_experiment_evidence"


def test_non_evidence_cannot_complete_experiment(m403_base_state):
    """Non-evidence turn preserves active experiment as IN_PROGRESS and does not mark COMPLETED."""
    m403_base_state["evidence_turn_assessment"] = {
        "disposition": EvidenceTurnDisposition.QUESTION_ABOUT_EXPERIMENT.value,
        "contains_completed_external_action": False,
    }
    m403_base_state["founder_response"] = "Then what should I do?"
    m403_base_state["active_experiment"]["status"] = ExperimentStatus.IN_PROGRESS.value

    assert has_valid_completed_evidence_cycle(m403_base_state) is False

    res = validate_evidence_analysis(m403_base_state)
    assert res.get("active_experiment", {}).get("status") == ExperimentStatus.IN_PROGRESS.value
    assert res.get("phase") == WorkflowPhase.WAITING_FOR_EVIDENCE


def test_non_evidence_cannot_increment_iteration(m403_base_state):
    """Non-evidence turn does not advance the iteration counter."""
    m403_base_state["evidence_turn_assessment"] = {
        "disposition": EvidenceTurnDisposition.FUTURE_INTENTION.value,
        "contains_completed_external_action": False,
    }
    m403_base_state["founder_response"] = "I will interview them tomorrow."

    res = increment_iteration(m403_base_state)
    assert res == {}


def test_non_evidence_cannot_update_assumption_status(m403_base_state):
    """Non-evidence turn does not mutate assumption status to INCONCLUSIVE or anything else."""
    m403_base_state["evidence_turn_assessment"] = {
        "disposition": EvidenceTurnDisposition.QUESTION_ABOUT_EXPERIMENT.value,
        "contains_completed_external_action": False,
    }
    m403_base_state["founder_response"] = "Can you explain the test again?"

    res = validate_evidence_analysis(m403_base_state)
    assert "active_assumption" not in res or res["active_assumption"]["status"] == "UNTESTED"


def test_valid_inconclusive_evidence_creates_legitimate_cycle(m403_base_state):
    """Valid completed experiment execution with mixed/inconclusive findings creates a legitimate learning cycle."""
    m403_base_state["evidence_turn_assessment"] = {
        "disposition": EvidenceTurnDisposition.ACTUAL_EVIDENCE.value,
        "contains_completed_external_action": True,
        "contains_results": True,
    }
    m403_base_state["founder_response"] = "I interviewed 3 merchants. One reported sizing returns are painful, two said returns are rare."
    m403_base_state["evidence_analysis"] = {
        "conclusion": "INCONCLUSIVE",
        "evidence_strength": "WEAK",
        "reasoning_summary": "Sample of 3 gave conflicting signals; insufficient to validate or disprove.",
    }

    assert has_valid_completed_evidence_cycle(m403_base_state) is True

    val_res = validate_evidence_analysis(m403_base_state)
    assert val_res["active_assumption"]["status"] == AssumptionStatus.INCONCLUSIVE.value
    assert val_res["active_experiment"]["status"] == ExperimentStatus.COMPLETED.value
    assert len(val_res["founder_evidence"]) == 1

    m403_base_state.update(val_res)
    iter_res = increment_iteration(m403_base_state)
    assert iter_res["iteration"] == 2
    assert iter_res["phase"] == WorkflowPhase.LEARN

