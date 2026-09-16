"""
Tests for Milestone 4.0.7.2.1: Evidence Admission Precedence Integrity.

Verifies:
1. completed evidence only -> ACTUAL_EVIDENCE
2. future intention only -> FUTURE_INTENTION
3. completed evidence + future intention -> ACTUAL_EVIDENCE (not overridden to FUTURE_INTENTION)
4. completed evidence + question -> ACTUAL_EVIDENCE (question preserved)
5. completed evidence + question + future plan -> ACTUAL_EVIDENCE (question preserved, evidence admitted)
6. future plan is not extracted as completed evidence
7. LastBite 8-interview regression case -> ACTUAL_EVIDENCE
8. existing FUTURE_INTENTION and mixed-turn tests remain intact
"""

from unittest.mock import MagicMock
import pytest

from startary.domain.enums import WorkflowPhase
from startary.domain.evidence import (
    EvidenceTurnAssessment,
    EvidenceTurnDisposition,
    get_evidence_turn_disposition,
    has_substantive_completed_evidence,
)
from startary.graph.builder import route_waiting_for_evidence_turn
from startary.graph.nodes import (
    assess_evidence_submission_sufficiency,
    classify_waiting_for_evidence_turn,
    extract_experiment_evidence,
)
from startary.llm.schemas import (
    EvidenceTurnAssessmentLLMOutput,
    ExtractedEvidenceLLMOutput,
)


@pytest.fixture
def base_evidence_state() -> dict:
    return {
        "project_id": "test-precedence-project",
        "phase": WorkflowPhase.WAITING_FOR_EVIDENCE,
        "iteration": 1,
        "active_experiment": {
            "id": "exp-test-001",
            "experiment_type": "INTERVIEW",
            "objective": "Test bakery willingness to sell surplus food at a discount",
            "founder_actions": ["Interview 8 bakeries about daily surplus"],
            "target_participants_or_scope": "8 local bakeries",
            "support_signal": "At least 5 bakeries report daily surplus and express interest in discounting",
            "contradiction_signal": "Fewer than 3 report surplus or interest",
            "status": "IN_PROGRESS",
        },
        "active_critical_assumption": {
            "id": "ca-test-001",
            "statement": "Local bakeries have recurring surplus food they are willing to discount.",
            "area": "TARGET_USER",
        },
        "founder_messages": [],
        "evidence_clarification_turns": 0,
    }


# ---------------------------------------------------------------------------
# Test 1: Completed evidence only -> ACTUAL_EVIDENCE
# ---------------------------------------------------------------------------
def test_completed_evidence_only_admitted(mocker, base_evidence_state):
    """Substantive completed evidence with no future talk is admitted as ACTUAL_EVIDENCE."""
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.side_effect = RuntimeError("Offline unit test")
    mock_llm.with_structured_output.return_value = mock_structured
    mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
    mocker.patch("startary.graph.nodes.get_settings", return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"))

    msg = "I interviewed 8 bakeries. 6 reported recurring surplus."
    base_evidence_state["founder_response"] = msg
    res = classify_waiting_for_evidence_turn(base_evidence_state)
    base_evidence_state.update(res)

    assessment = base_evidence_state["evidence_turn_assessment"]
    assert assessment["disposition"] == EvidenceTurnDisposition.ACTUAL_EVIDENCE.value
    assert assessment["contains_completed_external_action"] is True
    assert base_evidence_state["pending_post_evidence_question"] is None

    route = route_waiting_for_evidence_turn(base_evidence_state)
    assert route == "extract_experiment_evidence"


# ---------------------------------------------------------------------------
# Test 2: Future intention only -> FUTURE_INTENTION
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "future_msg",
    [
        "I am going to interview 8 bakeries next week.",
        "I will interview them tomorrow.",
        "Haven't started yet, but tomorrow I'll visit 5 shops.",
    ],
)
def test_future_intention_only_rejected(mocker, base_evidence_state, future_msg):
    """Pure future intentions without completed execution are rejected from evidence."""
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.side_effect = RuntimeError("Offline unit test")
    mock_llm.with_structured_output.return_value = mock_structured
    mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
    mocker.patch("startary.graph.nodes.get_settings", return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"))

    base_evidence_state["founder_response"] = future_msg
    res = classify_waiting_for_evidence_turn(base_evidence_state)
    base_evidence_state.update(res)

    assessment = base_evidence_state["evidence_turn_assessment"]
    assert assessment["disposition"] == EvidenceTurnDisposition.FUTURE_INTENTION.value
    assert assessment["contains_completed_external_action"] is False

    route = route_waiting_for_evidence_turn(base_evidence_state)
    assert route == "respond_to_waiting_founder"


# ---------------------------------------------------------------------------
# Test 3: Completed evidence + future intention -> ACTUAL_EVIDENCE
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "mixed_msg",
    [
        "I interviewed 8 bakeries. 6 reported recurring surplus. I will probably interview more next week.",
        "We ran the test for three days and got 5 paid pickups. Next week I want to test another bakery.",
        "Completed 5 store visits today. 4 managers said they have daily bread surplus. Tomorrow I will rest.",
    ],
)
def test_completed_evidence_plus_future_intention_admitted(mocker, base_evidence_state, mixed_msg):
    """Substantive completed evidence must NOT be overridden by incidental future-oriented language."""
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
        evidence_candidate_text=mixed_msg,
        rationale_summary="Founder reports completed tests meeting minimum evidence requirements.",
    )
    mock_llm.with_structured_output.return_value = mock_structured
    mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
    mocker.patch("startary.graph.nodes.get_settings", return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"))

    base_evidence_state["founder_response"] = mixed_msg
    res = classify_waiting_for_evidence_turn(base_evidence_state)
    base_evidence_state.update(res)

    assessment = base_evidence_state["evidence_turn_assessment"]
    assert assessment["disposition"] == EvidenceTurnDisposition.ACTUAL_EVIDENCE.value
    assert assessment["contains_completed_external_action"] is True
    assert assessment["contains_only_future_intent"] is False
    assert "[Overridden: future intention expressed]" not in assessment["rationale_summary"]

    route = route_waiting_for_evidence_turn(base_evidence_state)
    assert route == "extract_experiment_evidence"


def test_completed_evidence_plus_future_intention_deterministic_fallback(mocker, base_evidence_state):
    """Deterministic fallback also admits completed evidence over future intention."""
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.side_effect = RuntimeError("Offline unit test")
    mock_llm.with_structured_output.return_value = mock_structured
    mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
    mocker.patch("startary.graph.nodes.get_settings", return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"))

    base_evidence_state["founder_response"] = (
        "I interviewed 8 bakeries. 6 reported recurring surplus. I will probably interview more next week."
    )
    res = classify_waiting_for_evidence_turn(base_evidence_state)
    base_evidence_state.update(res)

    assessment = base_evidence_state["evidence_turn_assessment"]
    assert assessment["disposition"] == EvidenceTurnDisposition.ACTUAL_EVIDENCE.value
    assert assessment["contains_completed_external_action"] is True
    assert assessment["contains_only_future_intent"] is False

    route = route_waiting_for_evidence_turn(base_evidence_state)
    assert route == "extract_experiment_evidence"


# ---------------------------------------------------------------------------
# Test 4: Completed evidence + question -> ACTUAL_EVIDENCE
# ---------------------------------------------------------------------------
def test_completed_evidence_plus_question_admitted(mocker, base_evidence_state):
    """Evidence is admitted and founder question is preserved for post-analysis answering."""
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.side_effect = RuntimeError("Offline unit test")
    mock_llm.with_structured_output.return_value = mock_structured
    mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
    mocker.patch("startary.graph.nodes.get_settings", return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"))

    base_evidence_state["founder_response"] = (
        "I completed the interviews. 4 agreed, 2 declined. What should I do next?"
    )
    res = classify_waiting_for_evidence_turn(base_evidence_state)
    base_evidence_state.update(res)

    assessment = base_evidence_state["evidence_turn_assessment"]
    assert assessment["disposition"] == EvidenceTurnDisposition.ACTUAL_EVIDENCE.value
    assert assessment["contains_completed_external_action"] is True
    assert base_evidence_state["pending_post_evidence_question"] is not None
    assert "what should i do next" in base_evidence_state["pending_post_evidence_question"].lower()

    route = route_waiting_for_evidence_turn(base_evidence_state)
    assert route == "extract_experiment_evidence"


# ---------------------------------------------------------------------------
# Test 5: Completed evidence + question + future plan -> ACTUAL_EVIDENCE
# ---------------------------------------------------------------------------
def test_completed_evidence_plus_question_plus_future_plan(mocker, base_evidence_state):
    """Turn with completed evidence, future plan, and question is admitted with question preserved."""
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.side_effect = RuntimeError("Offline unit test")
    mock_llm.with_structured_output.return_value = mock_structured
    mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
    mocker.patch("startary.graph.nodes.get_settings", return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"))

    base_evidence_state["founder_response"] = (
        "I interviewed 8 businesses and 6 had surplus. Next week I will interview 10 more. What do you think?"
    )
    res = classify_waiting_for_evidence_turn(base_evidence_state)
    base_evidence_state.update(res)

    assessment = base_evidence_state["evidence_turn_assessment"]
    assert assessment["disposition"] == EvidenceTurnDisposition.ACTUAL_EVIDENCE.value
    assert assessment["contains_completed_external_action"] is True
    assert base_evidence_state["pending_post_evidence_question"] is not None
    assert "what do you think" in base_evidence_state["pending_post_evidence_question"].lower()

    route = route_waiting_for_evidence_turn(base_evidence_state)
    assert route == "extract_experiment_evidence"


# ---------------------------------------------------------------------------
# Test 6: Future plan is NOT extracted as completed evidence
# ---------------------------------------------------------------------------
def test_future_plan_not_extracted_as_completed_evidence(mocker, base_evidence_state):
    """Evidence extractor post-processing filters out future promises from actions_completed."""
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = ExtractedEvidenceLLMOutput(
        actions_completed=[
            "interviewed 8 businesses about surplus",
            "will interview 10 more businesses next week",
            "plan to test another bakery tomorrow",
        ],
        sample_or_counts={"approached": 8, "positive": 6},
        observed_behaviors=["bakeries throwing away 10-15 loaves nightly"],
        quotes=["Disposal fees are frustrating"],
        quantitative_outcomes={"positive_rate": 0.75},
        incomplete_actions=[],
        deviations_from_plan=[],
        missing_result_information=[],
        is_future_intention_only=True,
        evidence_items=[],
        direct_evidence_summary="8 businesses interviewed, 6 had surplus.",
    )
    mock_llm.with_structured_output.return_value = mock_structured
    mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
    mocker.patch("startary.graph.nodes.get_settings", return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"))

    base_evidence_state["founder_response"] = (
        "I interviewed 8 businesses and 6 had surplus. Next week I will interview 10 more."
    )
    res = extract_experiment_evidence(base_evidence_state)
    extracted = res["extracted_evidence"]

    assert extracted["is_future_intention_only"] is False

    actions = extracted["actions_completed"]
    assert "interviewed 8 businesses about surplus" in actions
    assert not any("will interview" in a for a in actions)
    assert not any("plan to" in a for a in actions)

    base_evidence_state.update(res)
    suff_res = assess_evidence_submission_sufficiency(base_evidence_state)
    q = suff_res.get("evidence_clarification_question")
    assert q is None or "It sounds like this is a plan for what you will do" not in q


# ---------------------------------------------------------------------------
# Test 7: LastBite 8-interview regression case -> ACTUAL_EVIDENCE
# ---------------------------------------------------------------------------
def test_lastbite_8_interview_regression_case_admitted(mocker, base_evidence_state):
    """
    Live run regression: Founder submitted 8 completed interviews with quantified surplus
    and direct quotes, but mentioned 'tomorrow I will follow up'.
    Must be admitted as ACTUAL_EVIDENCE, not overridden to FUTURE_INTENTION.
    """
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
        evidence_candidate_text=(
            "Completed 8 business interviews with local bakeries and cafes. "
            "6 reported recurring daily surplus of 10-15 items. "
            "They said disposal costs are frustrating and would test a discount pickup model."
        ),
        rationale_summary="8 business interviews completed meeting threshold; observations and quotes provided.",
    )
    mock_llm.with_structured_output.return_value = mock_structured
    mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
    mocker.patch("startary.graph.nodes.get_settings", return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"))

    submission = (
        "Completed 8 business interviews with local bakeries and cafes. "
        "6 reported recurring daily surplus of 10-15 items. "
        "They said disposal costs are frustrating and would test a discount pickup model. "
        "I will follow up with them tomorrow and test 2 more."
    )
    base_evidence_state["founder_response"] = submission
    res = classify_waiting_for_evidence_turn(base_evidence_state)
    base_evidence_state.update(res)

    assessment = base_evidence_state["evidence_turn_assessment"]
    assert assessment["disposition"] == EvidenceTurnDisposition.ACTUAL_EVIDENCE.value
    assert assessment["contains_completed_external_action"] is True
    assert assessment["contains_only_future_intent"] is False
    assert "[Overridden: future intention expressed]" not in assessment["rationale_summary"]

    route = route_waiting_for_evidence_turn(base_evidence_state)
    assert route == "extract_experiment_evidence"


# ---------------------------------------------------------------------------
# Test 8: Existing FUTURE_INTENTION and mixed-turn tests still pass
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "founder_text,expected_disp,expected_route",
    [
        (
            "I will interview them tomorrow.",
            EvidenceTurnDisposition.FUTURE_INTENTION,
            "respond_to_waiting_founder",
        ),
        (
            "I will interview them tomorrow. What should I ask?",
            EvidenceTurnDisposition.FUTURE_INTENTION,
            "respond_to_waiting_founder",
        ),
        (
            "I haven't started. Can you explain again?",
            EvidenceTurnDisposition.QUESTION_ABOUT_EXPERIMENT,
            "respond_to_waiting_founder",
        ),
        (
            "I contacted 20 shops and nobody replied.",
            EvidenceTurnDisposition.ACTUAL_EVIDENCE,
            "extract_experiment_evidence",
        ),
        (
            "I completed the interviews. Is this enough?",
            EvidenceTurnDisposition.ACTUAL_EVIDENCE,
            "extract_experiment_evidence",
        ),
    ],
)
def test_existing_turn_classification_scenarios_preserved(
    mocker, base_evidence_state, founder_text, expected_disp, expected_route
):
    """Ensures existing boundary and routing tests remain fully intact."""
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.side_effect = RuntimeError("Offline unit test")
    mock_llm.with_structured_output.return_value = mock_structured
    mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
    mocker.patch("startary.graph.nodes.get_settings", return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"))

    base_evidence_state["founder_response"] = founder_text
    res = classify_waiting_for_evidence_turn(base_evidence_state)
    base_evidence_state.update(res)

    disp = get_evidence_turn_disposition(base_evidence_state)
    assert disp == expected_disp

    route = route_waiting_for_evidence_turn(base_evidence_state)
    assert route == expected_route
