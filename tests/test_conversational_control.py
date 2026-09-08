"""
Unit and integration tests for Milestone 3.2.1: Conversational Control Integrity Patch.

Tests verify:
1. NextTurnDecision domain model & authority
2. Composer semantic target enforcement
3. Stale question ID rejection & token matching
4. QuestionHistory prevents re-asking answered targets
5. Semantic repetition guard advances past known operational areas (e.g. TEAM -> CAPACITY)
6. "I already answered this" claim detection and replanning
7. Multi-answer satisfaction of planned targets
8. Explicit conflict detection (FactConflict) preventing silent fact mutation
9. Founder goal persistence in conversational context
10. Introductory summary suppression after turn 0
11. Deterministic fallback preserving canonical question
12. Active decision & advisor cache clearance in merge_founder_answer
"""

from unittest.mock import MagicMock
import pytest

from startary.domain.enums import WorkflowPhase
from startary.domain.turn_control import (
    FactConflict,
    FounderGoal,
    FounderGoalType,
    NextTurnDecision,
    QuestionHistoryItem,
)
from startary.graph.graph_state import StartaryGraphState
from startary.graph.nodes import (
    _merge_knowledge_field,
    compose_advisor_response,
    interpret_founder_turn,
    merge_execution_state,
    merge_founder_answer,
    plan_baseline_question,
)
from startary.llm.schemas import (
    AdvisorResponseLLMOutput,
    ExtractedKnowledgeFieldLLMOutput,
    FounderTurnInterpretationLLMOutput,
    StartupStateDeltaLLMOutput,
)


@pytest.fixture
def base_control_state() -> StartaryGraphState:
    return {
        "project_id": "proj-control-123",
        "phase": WorkflowPhase.UNDERSTAND,
        "iteration": 1,
        "raw_context": "TruckTaxi is a two-sided freight platform matching small shippers with truck drivers.",
        "weak_area": "Problem",
        "selection_confidence": "LOW",
        "baseline_sufficiency": {
            "is_sufficient": False,
            "missing_groups": ["TEAM", "CAPACITY", "PRODUCT_STAGE"],
        },
        "baseline_question_turns": 0,
        "strategic_clarification_turns": 0,
        "weak_link_clarification_turns": 0,
        "founder_messages": [
            {"role": "founder", "content": "TruckTaxi matches small shippers with independent truck drivers."}
        ],
        "execution_context": {
            "founder_count": {"value": None, "provenance": "UNKNOWN", "confidence": "LOW"},
            "team_members": [],
        },
        "startup_progress": {
            "development_stage": {"value": None, "provenance": "UNKNOWN", "confidence": "LOW"},
        },
        "question_history": [],
        "fact_conflicts": [],
        "founder_current_goal": None,
        "intro_summary_shown": False,
        "active_next_turn_decision": None,
        "latest_advisor_response": None,
        "pending_question": None,
    }


def test_next_turn_decision_model_validates():
    """1. NextTurnDecision model validates required fields and generates unique IDs."""
    decision = NextTurnDecision(
        source_phase=WorkflowPhase.CLARIFYING_CONTEXT,
        information_target="CAPACITY",
        canonical_question="Roughly how much time can each person give per week?",
        clarification_reason="Determine bandwidth",
        planner_version="baseline_v1",
    )
    assert decision.question_id.startswith("q_")
    assert decision.information_target == "CAPACITY"
    assert decision.should_ask_question is True
    assert decision.canonical_question == "Roughly how much time can each person give per week?"


def test_composer_target_must_match_planner_target(mocker, base_control_state):
    """2. If Advisor Composer outputs a different information target, it is rejected and falls back."""
    decision = NextTurnDecision(
        question_id="q_target_match_1",
        source_phase=WorkflowPhase.CLARIFYING_CONTEXT,
        information_target="CAPACITY",
        canonical_question="Roughly how much time can each of you give TruckTaxi each week?",
    )
    base_control_state["active_next_turn_decision"] = decision.model_dump()

    mock_llm = MagicMock()
    mock_structured = MagicMock()

    # Composer erroneously tries to ask about TEAM instead of CAPACITY
    bad_output = AdvisorResponseLLMOutput(
        question_id="q_target_match_1",
        next_question_target="TEAM",  # MISMATCH!
        next_question_rephrased="Who else is on the team with you?",
        acknowledge="Understood.",
        answer_to_founder=None,
        diagnosis_delta=None,
        should_ask_next_question=True,
        response_text="Who else is on the team with you?",
    )
    mock_structured.invoke.return_value = bad_output
    mock_llm.with_structured_output.return_value = mock_structured

    mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
    mocker.patch("startary.graph.nodes.get_settings", return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"))

    res = compose_advisor_response(base_control_state)
    advisor_resp = res["latest_advisor_response"]

    # Fallback must be triggered, preserving the planner's canonical question and target
    assert advisor_resp["next_question_target"] == "CAPACITY"
    assert advisor_resp["question_id"] == "q_target_match_1"
    assert "Roughly how much time" in advisor_resp["response_text"]


def test_stale_question_id_rejected(mocker, base_control_state):
    """3. Advisor response referencing an expired or mismatched question_id is rejected."""
    decision = NextTurnDecision(
        question_id="q_current_active_token",
        source_phase=WorkflowPhase.CLARIFYING_CONTEXT,
        information_target="CAPACITY",
        canonical_question="Roughly how much time can you spend each week?",
    )
    base_control_state["active_next_turn_decision"] = decision.model_dump()

    mock_llm = MagicMock()
    mock_structured = MagicMock()

    # Composer output has a stale question_id from a previous turn
    stale_output = AdvisorResponseLLMOutput(
        question_id="q_stale_previous_token",  # MISMATCH!
        next_question_target="CAPACITY",
        next_question_rephrased="Roughly how much time can you spend each week?",
        acknowledge="Understood.",
        answer_to_founder=None,
        diagnosis_delta=None,
        should_ask_next_question=True,
        response_text="Understood. Roughly how much time can you spend each week?",
    )
    mock_structured.invoke.return_value = stale_output
    mock_llm.with_structured_output.return_value = mock_structured

    mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
    mocker.patch("startary.graph.nodes.get_settings", return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"))

    res = compose_advisor_response(base_control_state)
    advisor_resp = res["latest_advisor_response"]

    # Fallback enforces active decision's question_id
    assert advisor_resp["question_id"] == "q_current_active_token"
    assert advisor_resp["next_question_target"] == "CAPACITY"


def test_answered_target_cannot_be_asked_again(mocker, base_control_state):
    """4. QuestionHistory blocks planner from asking already-answered targets."""
    base_control_state["question_history"] = [
        {
            "question_id": "q_base_1_team",
            "information_target": "TEAM",
            "canonical_question": "Who is on the team?",
            "turn": 1,
            "answered": True,
            "founder_answer_summary": "We are 3 people.",
        }
    ]
    base_control_state["baseline_question_turns"] = 1

    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.side_effect = Exception("Fallback branch should be used")
    mock_llm.with_structured_output.return_value = mock_structured

    mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
    mocker.patch("startary.graph.nodes.get_settings", return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"))

    res = plan_baseline_question(base_control_state)
    decision = res["active_next_turn_decision"]

    # Must NOT select TEAM since TEAM is in question_history as answered
    assert decision["information_target"] != "TEAM"
    assert decision["information_target"] == "CAPACITY"


def test_semantic_repetition_blocked(mocker, base_control_state):
    """5. When TEAM is already known in execution_context, planner advances to CAPACITY."""
    base_control_state["execution_context"]["founder_count"] = {
        "value": "3",
        "provenance": "EXPLICIT",
        "confidence": "HIGH",
    }
    base_control_state["execution_context"]["team_members"] = [
        {"role": "driver relations"},
        {"role": "mobile dev"},
        {"role": "business analysis"},
    ]

    mock_llm = MagicMock()
    mock_structured = MagicMock()
    # LLM mistakenly tries to return TEAM
    from startary.llm.schemas import BaselineQuestionPlanLLMOutput
    mock_structured.invoke.return_value = BaselineQuestionPlanLLMOutput(
        should_ask=True,
        target_group="TEAM",
        target_fields=["founder_count"],
        question="Who else is working on this?",
        reason="Check team",
    )
    mock_llm.with_structured_output.return_value = mock_structured

    mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
    mocker.patch("startary.graph.nodes.get_settings", return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"))

    res = plan_baseline_question(base_control_state)
    decision = res["active_next_turn_decision"]

    # Deterministic guard overrides LLM and selects CAPACITY
    assert decision["information_target"] == "CAPACITY"


def test_correction_i_already_answered_triggers_replanning(mocker, base_control_state):
    """6. Founder saying 'I already answered this' is classified as CORRECTION with HIGH priority."""
    base_control_state["founder_messages"].append(
        {"role": "founder", "content": "I already answered this. We are 3 people as I said earlier."}
    )

    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = FounderTurnInterpretationLLMOutput(
        intent_types=["CORRECTION"],
        direct_answer=False,
        direct_question=None,
        concern_or_goal=None,
        correction="Founder notes team was already provided",
        cannot_answer_now=False,
        cannot_act_now=False,
        is_already_answered_claim=True,
        response_priority="HIGH",
        interpretation_summary="Founder corrected repetition.",
    )
    mock_llm.with_structured_output.return_value = mock_structured

    mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
    mocker.patch("startary.graph.nodes.get_settings", return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"))

    res = interpret_founder_turn(base_control_state)
    interpretation = res["latest_founder_turn_interpretation"]

    assert "CORRECTION" in interpretation["intent_types"]
    assert interpretation["response_priority"] == "HIGH"
    assert interpretation.get("is_already_answered_claim") is True


def test_multi_answer_satisfies_planned_target(mocker, base_control_state):
    """7. Multi-answer providing team and capacity updates both and advances to product stage."""
    # Delta contains both team and capacity
    base_control_state["startup_state_delta"] = StartupStateDeltaLLMOutput(
        founder_count=ExtractedKnowledgeFieldLLMOutput(value="3", provenance="EXPLICIT", confidence="HIGH"),
        weekly_hours_per_member=ExtractedKnowledgeFieldLLMOutput(value="15", provenance="EXPLICIT", confidence="HIGH"),
    ).model_dump()
    base_control_state["startup_state_delta"]["_source_turn"] = 1

    merged = merge_execution_state(base_control_state)
    base_control_state["execution_context"] = merged["execution_context"]

    assert base_control_state["execution_context"]["founder_count"]["value"] == "3"
    assert base_control_state["execution_context"]["approximate_weekly_hours_per_member"]["value"] == "15"

    # Now planner should skip both TEAM and CAPACITY and target PRODUCT_STAGE
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.side_effect = Exception("Use fallback to test effective missing")
    mock_llm.with_structured_output.return_value = mock_structured

    mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
    mocker.patch("startary.graph.nodes.get_settings", return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"))

    res = plan_baseline_question(base_control_state)
    decision = res["active_next_turn_decision"]

    assert decision["information_target"] == "PRODUCT_STAGE"


def test_explicit_conflict_requires_clarification():
    """8. Confirmed 3 founders followed by 'I am alone' triggers FactConflict and does not overwrite."""
    state: StartaryGraphState = {
        "execution_context": {
            "founder_count": {
                "value": "3",
                "provenance": "EXPLICIT",
                "confidence": "HIGH",
                "source_turn": 0,
            }
        },
        "startup_progress": {},
        "startup_state_delta": {
            "_source_turn": 1,
            "founder_count": {
                "value": "1",
                "provenance": "EXPLICIT",
                "confidence": "HIGH",
            },
        },
        "fact_conflicts": [],
    }

    res = merge_execution_state(state)

    # Value remains "3" (not overwritten)
    assert res["execution_context"]["founder_count"]["value"] == "3"

    # Conflict is formally recorded
    conflicts = res["fact_conflicts"]
    assert len(conflicts) == 1
    assert conflicts[0]["field_name"] == "founder_count"
    assert conflicts[0]["existing_value"] == "3"
    assert conflicts[0]["conflicting_value"] == "1"
    assert conflicts[0]["resolved"] is False


def test_founder_goal_persists_short_term(mocker, base_control_state):
    """9. Founder goal FIND_INITIAL_MARKET is detected and persists across nearby turns."""
    base_control_state["founder_messages"].append(
        {"role": "founder", "content": "My intention is to find the right market or understand how to find it."}
    )

    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = FounderTurnInterpretationLLMOutput(
        intent_types=["ANSWER", "REQUEST_FOR_ADVICE"],
        direct_answer=True,
        direct_question=None,
        concern_or_goal="Find initial market",
        detected_founder_goal="FIND_INITIAL_MARKET",
        cannot_answer_now=False,
        cannot_act_now=False,
        response_priority="NORMAL",
        interpretation_summary="Founder wants to find the initial market.",
    )
    mock_llm.with_structured_output.return_value = mock_structured

    mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
    mocker.patch("startary.graph.nodes.get_settings", return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"))

    res = interpret_founder_turn(base_control_state)
    goal = res["founder_current_goal"]

    assert goal is not None
    assert goal["goal_type"] == "FIND_INITIAL_MARKET"
    assert "find the right market" in goal["raw_statement"]


def test_intro_summary_shown_once(mocker, base_control_state):
    """10. intro_summary_shown flag is set to True after turn 0 advisor response."""
    decision = NextTurnDecision(
        question_id="q_intro_test",
        source_phase=WorkflowPhase.CLARIFYING_CONTEXT,
        information_target="TEAM",
        canonical_question="Who is on the team?",
    )
    base_control_state["active_next_turn_decision"] = decision.model_dump()
    base_control_state["intro_summary_shown"] = False

    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = AdvisorResponseLLMOutput(
        question_id="q_intro_test",
        next_question_target="TEAM",
        next_question_rephrased="Who is on the team?",
        acknowledge="Understood.",
        answer_to_founder=None,
        diagnosis_delta=None,
        should_ask_next_question=True,
        response_text="Understood. Who is on the team?",
    )
    mock_llm.with_structured_output.return_value = mock_structured

    mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
    mocker.patch("startary.graph.nodes.get_settings", return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"))

    res = compose_advisor_response(base_control_state)
    assert res["intro_summary_shown"] is True


def test_deterministic_fallback_uses_canonical_question(mocker, base_control_state):
    """11. Deterministic fallback uses canonical_question and preserves target and qid."""
    decision = NextTurnDecision(
        question_id="q_fallback_exact",
        source_phase=WorkflowPhase.CLARIFYING_CONTEXT,
        information_target="RESOURCE",
        canonical_question="Roughly what budget can you realistically spend on testing?",
    )
    base_control_state["active_next_turn_decision"] = decision.model_dump()

    mock_llm = MagicMock()
    mock_structured = MagicMock()
    # Force exception to trigger fallback
    mock_structured.invoke.side_effect = RuntimeError("API failure")
    mock_llm.with_structured_output.return_value = mock_structured

    mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
    mocker.patch("startary.graph.nodes.get_settings", return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"))

    res = compose_advisor_response(base_control_state)
    advisor_resp = res["latest_advisor_response"]

    assert advisor_resp["question_id"] == "q_fallback_exact"
    assert advisor_resp["next_question_target"] == "RESOURCE"
    assert "Roughly what budget" in advisor_resp["response_text"]


def test_active_decision_cleared_on_founder_answer(base_control_state):
    """12. merge_founder_answer clears active_next_turn_decision and latest_advisor_response."""
    decision = NextTurnDecision(
        question_id="q_lifecycle_token_99",
        source_phase=WorkflowPhase.CLARIFYING_CONTEXT,
        information_target="TEAM",
        canonical_question="Who is on the team?",
    )
    base_control_state["active_next_turn_decision"] = decision.model_dump()
    base_control_state["latest_advisor_response"] = {
        "question_id": "q_lifecycle_token_99",
        "response_text": "Who is building TruckTaxi with you?",
    }
    base_control_state["founder_response"] = "We are 3 people."
    base_control_state["active_interrupt_type"] = "baseline"

    res = merge_founder_answer(base_control_state)

    # Active decision and latest advisor response MUST be cleared
    assert res["active_next_turn_decision"] is None
    assert res["latest_advisor_response"] is None
    assert res["answered_question_id"] == "q_lifecycle_token_99"

    # Question is recorded into question_history
    q_hist = res["question_history"]
    assert len(q_hist) == 1
    assert q_hist[0]["question_id"] == "q_lifecycle_token_99"
    assert q_hist[0]["information_target"] == "TEAM"
    assert q_hist[0]["answered"] is True

