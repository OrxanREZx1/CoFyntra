"""
Unit tests for Milestone 4.0.7: Evidence Routing Integrity.

Verifies:
1. Classifier writes canonical evidence_turn_assessment and preserves pending_post_evidence_question.
2. Router reads Pydantic EvidenceTurnAssessment correctly via get_evidence_turn_disposition.
3. Router reads serialized dict, string, and enum forms correctly.
4. ACTUAL_EVIDENCE routes to extract_experiment_evidence.
5. ACTUAL_EVIDENCE + QUESTION routes to extract_experiment_evidence (mixed-turn precedence).
6. respond_to_waiting_founder cannot override admitted evidence or reclassify as question.
7. Checkpoint round-trip (serialize/deserialize) preserves disposition and routing.
8. Missing disposition after classifier triggers evidence_routing_integrity_error and executes fail-safe.
9. Post-evidence question is answered after re-diagnosis using the new state.
10. Transient assessment and pending question are cleared after completed cycle.
"""

from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock
import pytest

from langgraph.checkpoint.memory import MemorySaver

from startary.domain.enums import (
    AssumptionStatus,
    EvidenceAnalysisConclusion,
    EvidenceStrength,
    ExperimentStatus,
    WorkflowPhase,
)
from startary.domain.evidence import (
    EvidenceTurnAssessment,
    EvidenceTurnDisposition,
    get_evidence_turn_disposition,
    is_actual_evidence_turn,
)
from startary.graph.builder import (
    route_waiting_for_evidence_turn,
    run_evidence_learning_pipeline,
)
from startary.graph.graph_state import StartaryGraphState
from startary.graph.nodes import (
    answer_post_evidence_question,
    classify_waiting_for_evidence_turn,
    respond_to_waiting_founder,
    wait_for_evidence,
)
from startary.llm.schemas import (
    EvidenceTurnAssessmentLLMOutput,
)


@pytest.fixture
def m407_base_state() -> StartaryGraphState:
    return {
        "project_id": "proj-shelfsense-m407",
        "phase": WorkflowPhase.WAITING_FOR_EVIDENCE,
        "iteration": 1,
        "raw_context": (
            "ShelfSense helps independent grocery store owners prevent produce and dairy "
            "spoilage with camera-based shelf tracking. Founded by two ML engineers."
        ),
        "weak_area": "Problem",
        "selection_confidence": "LOW",
        "active_assumption": {
            "id": "asmp-shelfsense-001",
            "statement": "Independent grocery store owners experience daily shelf stockouts and spoilage as an acute operational pain.",
            "area": "Problem",
            "status": "UNTESTED",
            "is_critical": True,
        },
        "active_experiment": {
            "id": "exp-shelfsense-001",
            "assumption_id": "asmp-shelfsense-001",
            "experiment_type": "CUSTOMER_INTERVIEWS",
            "objective": "Determine if independent grocery shop owners struggle with stockouts and spoilage.",
            "hypothesis": "Independent grocery store owners experience daily shelf stockouts and spoilage as an acute operational pain.",
            "founder_actions": [
                "Speak with 5-6 independent grocery shop owners.",
                "Ask who makes supplier ordering decisions and how often stockouts occur.",
            ],
            "target_participants_or_scope": "5-6 independent grocery shop owners",
            "evidence_to_collect": "Owner ordering control, stockout frequency, spoilage waste, current workarounds",
            "support_signal": "At least 4 of 6 owners personally decide supplier orders and report recent stockouts/waste.",
            "contradiction_signal": "Most owners do not control orders or have automated replenishment with zero issues.",
            "inconclusive_signal": "Mixed answers or unable to confirm who makes decisions.",
            "valid_inferences": ["Validates whether inventory imbalance is an active operational pain."],
            "invalid_inferences": ["Does not prove willingness to purchase camera hardware."],
            "status": ExperimentStatus.IN_PROGRESS.value,
        },
        "founder_messages": [
            {"role": "founder", "content": "I am building ShelfSense for local grocery store owners."}
        ],
    }


def test_classifier_writes_canonical_assessment_and_preserves_question(mocker, m407_base_state):
    """1. Classifier writes canonical evidence_turn_assessment and preserves pending_post_evidence_question."""
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
        evidence_candidate_text="I talked with 6 independent grocery shop owners. 5/6 owners decide most supplier orders.",
        rationale_summary="Founder completed interviews with 6 shop owners and reported detailed findings.",
    )
    mock_llm.with_structured_output.return_value = mock_structured
    mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
    mocker.patch("startary.graph.nodes.get_settings", return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"))

    shelf_msg = (
        "I talked with 6 independent grocery shop owners. "
        "5/6 owners personally control supplier ordering. "
        "4/6 had recent stockouts. 3/6 had over-ordering / waste. "
        "What should I do next?"
    )
    m407_base_state["founder_response"] = shelf_msg
    res = classify_waiting_for_evidence_turn(m407_base_state)

    assert "evidence_turn_assessment" in res
    assessment = res["evidence_turn_assessment"]
    assert assessment["disposition"] == "ACTUAL_EVIDENCE"
    assert assessment["contains_completed_external_action"] is True
    assert assessment["contains_results"] is True

    # Question is preserved
    assert res.get("pending_post_evidence_question") == "What should I do next?"


def test_router_reads_pydantic_assessment_correctly():
    """2. Router reads Pydantic EvidenceTurnAssessment correctly via get_evidence_turn_disposition."""
    assessment = EvidenceTurnAssessment(
        disposition=EvidenceTurnDisposition.ACTUAL_EVIDENCE,
        contains_completed_external_action=True,
        contains_results=True,
    )
    # Direct object test
    assert get_evidence_turn_disposition(assessment) == EvidenceTurnDisposition.ACTUAL_EVIDENCE
    assert is_actual_evidence_turn(assessment) is True

    # Nested state test
    state = {"evidence_turn_assessment": assessment}
    assert get_evidence_turn_disposition(state) == EvidenceTurnDisposition.ACTUAL_EVIDENCE
    assert is_actual_evidence_turn(state) is True
    assert route_waiting_for_evidence_turn(state) == "extract_experiment_evidence"


def test_router_reads_serialized_dict_and_string_forms():
    """3. Router reads serialized dict, string, and enum forms correctly without relying on identity."""
    # Dict with string disposition
    s_dict_str = {"evidence_turn_assessment": {"disposition": "ACTUAL_EVIDENCE"}}
    assert get_evidence_turn_disposition(s_dict_str) == EvidenceTurnDisposition.ACTUAL_EVIDENCE
    assert route_waiting_for_evidence_turn(s_dict_str) == "extract_experiment_evidence"

    # Dict with enum disposition
    s_dict_enum = {"evidence_turn_assessment": {"disposition": EvidenceTurnDisposition.ACTUAL_EVIDENCE}}
    assert get_evidence_turn_disposition(s_dict_enum) == EvidenceTurnDisposition.ACTUAL_EVIDENCE
    assert route_waiting_for_evidence_turn(s_dict_enum) == "extract_experiment_evidence"

    # Dict with qualified enum string
    s_dict_qual = {"evidence_turn_assessment": {"disposition": "EvidenceTurnDisposition.ACTUAL_EVIDENCE"}}
    assert get_evidence_turn_disposition(s_dict_qual) == EvidenceTurnDisposition.ACTUAL_EVIDENCE
    assert route_waiting_for_evidence_turn(s_dict_qual) == "extract_experiment_evidence"

    # String directly
    s_str = {"evidence_turn_assessment": "ACTUAL_EVIDENCE"}
    assert get_evidence_turn_disposition(s_str) == EvidenceTurnDisposition.ACTUAL_EVIDENCE
    assert route_waiting_for_evidence_turn(s_str) == "extract_experiment_evidence"

    # Top-level fallback
    s_top = {"disposition": "ACTUAL_EVIDENCE"}
    assert get_evidence_turn_disposition(s_top) == EvidenceTurnDisposition.ACTUAL_EVIDENCE
    assert route_waiting_for_evidence_turn(s_top) == "extract_experiment_evidence"


def test_actual_evidence_routes_to_extractor(m407_base_state):
    """4. ACTUAL_EVIDENCE routes strictly to extract_experiment_evidence."""
    m407_base_state["evidence_turn_assessment"] = {
        "disposition": "ACTUAL_EVIDENCE",
        "contains_completed_external_action": True,
        "contains_results": True,
    }
    route = route_waiting_for_evidence_turn(m407_base_state)
    assert route == "extract_experiment_evidence"


def test_mixed_evidence_plus_question_routes_to_extractor(mocker, m407_base_state):
    """5. Mixed evidence + question routes to extract_experiment_evidence (precedence rule)."""
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
        evidence_candidate_text="I completed the interviews with 5 shop owners. 4 had stockouts.",
        rationale_summary="Actual evidence combined with a question.",
    )
    mock_llm.with_structured_output.return_value = mock_structured
    mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
    mocker.patch("startary.graph.nodes.get_settings", return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"))

    m407_base_state["founder_response"] = "I completed the interviews with 5 shop owners. 4 had stockouts. What should I do next?"
    res = classify_waiting_for_evidence_turn(m407_base_state)
    m407_base_state.update(res)

    route = route_waiting_for_evidence_turn(m407_base_state)
    assert route == "extract_experiment_evidence"
    assert route != "respond_to_waiting_founder"


def test_respond_to_waiting_founder_cannot_override_admitted_evidence(m407_base_state, caplog):
    """6. respond_to_waiting_founder cannot reclassify admitted evidence or change disposition."""
    m407_base_state["evidence_turn_assessment"] = {
        "disposition": "ACTUAL_EVIDENCE",
        "contains_completed_external_action": True,
        "contains_results": True,
    }
    m407_base_state["founder_response"] = "I talked with 6 shop owners. 4 had stockouts."

    with caplog.at_level(logging.WARNING):
        res = respond_to_waiting_founder(m407_base_state)

    # Invariant: Evidence is preserved and not reclassified
    assert "disposition=ACTUAL_EVIDENCE reached unexpectedly; preserving evidence" in caplog.text
    assert "processing and analyzing" in res["waiting_response"].lower()
    # Ensure disposition was NOT altered in state
    assert get_evidence_turn_disposition(m407_base_state) == EvidenceTurnDisposition.ACTUAL_EVIDENCE


def test_checkpoint_roundtrip_preserves_disposition_and_routing(m407_base_state):
    """7. Checkpoint round-trip (JSON and MemorySaver) preserves disposition and routing."""
    m407_base_state["evidence_turn_assessment"] = {
        "disposition": "ACTUAL_EVIDENCE",
        "contains_completed_external_action": True,
        "contains_results": True,
        "rationale_summary": "Verified real results.",
    }
    m407_base_state["pending_post_evidence_question"] = "What should I do next?"

    # 1. JSON serialize / deserialize roundtrip
    serialized = json.dumps(m407_base_state, default=str)
    restored = json.loads(serialized)

    assert get_evidence_turn_disposition(restored) == EvidenceTurnDisposition.ACTUAL_EVIDENCE
    assert route_waiting_for_evidence_turn(restored) == "extract_experiment_evidence"

    # 2. LangGraph MemorySaver checkpoint roundtrip
    from startary.graph.builder import build_graph
    graph = build_graph(checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "thread-m407-test"}}
    graph.update_state(cfg, restored)
    snap = graph.get_state(cfg)
    retrieved_state = snap.values

    assert get_evidence_turn_disposition(retrieved_state) == EvidenceTurnDisposition.ACTUAL_EVIDENCE
    assert route_waiting_for_evidence_turn(retrieved_state) == "extract_experiment_evidence"


def test_missing_disposition_triggers_integrity_error_and_failsafe(caplog):
    """8. Missing disposition after classifier triggers evidence_routing_integrity_error and executes fail-safe."""
    # Case A: Missing assessment, but message text has completed action
    state_with_evidence = {
        "project_id": "proj-fail-safe",
        "evidence_turn_assessment": None,
        "founder_response": "I interviewed 6 independent grocery shop owners. 4 had stockouts.",
    }

    with caplog.at_level(logging.ERROR):
        route = route_waiting_for_evidence_turn(state_with_evidence)

    assert "evidence_routing_integrity_error" in caplog.text
    assert route == "extract_experiment_evidence"

    # Case B: Missing assessment, message text has no completed action (just question)
    caplog.clear()
    state_without_evidence = {
        "project_id": "proj-fail-safe-2",
        "evidence_turn_assessment": None,
        "founder_response": "What should I do now?",
    }

    with caplog.at_level(logging.ERROR):
        route = route_waiting_for_evidence_turn(state_without_evidence)

    assert "evidence_routing_integrity_error" in caplog.text
    assert route == "respond_to_waiting_founder"


def test_post_evidence_question_answered_after_learning_cycle():
    """9. Post-evidence question is answered after re-diagnosis using the new state."""
    state = {
        "project_id": "proj-post-q",
        "iteration": 2,
        "weak_area": "Target User",
        "active_assumption": {
            "statement": "Independent grocery store owners make purchasing decisions within 2 weeks.",
        },
        "evidence_analysis": {
            "conclusion": "SUPPORTED",
            "support_findings": ["5/6 owners personally decide supplier orders."],
            "valid_learnings": ["Store owners control inventory directly."],
        },
        "learning_delta": {
            "what_we_learned": ["Store owners have direct decision authority."],
            "what_became_more_certain": ["Problem friction is confirmed across 4/6 stores."],
            "what_remains_unknown": ["Willingness to pay for SaaS remains unvalidated."],
        },
        "pending_post_evidence_question": "What should I do next?",
        "founder_messages": [],
    }

    res = answer_post_evidence_question(state)

    assert "post_evidence_answer" in res
    answer = res["post_evidence_answer"]
    assert "What you learned:" in answer
    assert "Store owners have direct decision authority." in answer
    assert "What changed:" in answer
    assert "Problem friction is confirmed" in answer
    assert "Next:" in answer
    assert "Target User" in answer

    # Invariant: transient fields are cleared
    assert res["pending_post_evidence_question"] is None
    assert res["evidence_turn_assessment"] is None


def test_transient_evidence_state_cleared_after_cycle(mocker, m407_base_state):
    """10. Transient assessment and pending question are cleared after completed cycle."""
    m407_base_state["pending_post_evidence_question"] = "What should I do next?"
    m407_base_state["evidence_turn_assessment"] = {
        "disposition": "ACTUAL_EVIDENCE",
    }
    m407_base_state["post_evidence_answer"] = "Here is what we learned and what to do next."

    # Mock interrupt to return founder reply
    mocker.patch("startary.graph.nodes.interrupt", return_value="Sounds good, I will prepare.")

    # Call wait_for_evidence
    res = wait_for_evidence(m407_base_state)

    # Invariants: transient fields are reset to None
    assert res["evidence_turn_assessment"] is None
    assert res["pending_post_evidence_question"] is None
    assert res["post_evidence_answer"] is None
    assert res["waiting_response"] is None


@pytest.mark.parametrize(
    "founder_text,expected_disposition,expected_route",
    [
        (
            "I interviewed 6 shops. Here are the results. What should I do next?",
            EvidenceTurnDisposition.ACTUAL_EVIDENCE,
            "extract_experiment_evidence",
        ),
        (
            "I completed the interviews. Is this enough?",
            EvidenceTurnDisposition.ACTUAL_EVIDENCE,
            "extract_experiment_evidence",
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
    ],
)
def test_mixed_turn_precedence_scenarios_a_to_e(
    mocker, m407_base_state, founder_text, expected_disposition, expected_route
):
    """Section 14: Verify all 5 precedence scenarios A through E."""
    # Fast fallback: structured LLM raises exception to exercise deterministic classification rules
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.side_effect = RuntimeError("Offline unit test")
    mock_llm.with_structured_output.return_value = mock_structured
    mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
    mocker.patch("startary.graph.nodes.get_settings", return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"))

    m407_base_state["founder_response"] = founder_text
    res = classify_waiting_for_evidence_turn(m407_base_state)
    m407_base_state.update(res)

    disp = get_evidence_turn_disposition(m407_base_state)
    assert disp == expected_disposition

    route = route_waiting_for_evidence_turn(m407_base_state)
    assert route == expected_route

