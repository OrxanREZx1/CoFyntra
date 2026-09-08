"""
Milestone 4.0.4 Unit Tests: Grounding & Experiment Recovery Integrity.

Tests cover:
1. Coding capability does not set design capability false (remains UNKNOWN).
2. Coding capability does not set marketing capability false (remains UNKNOWN).
3. Explicit 'we don't know design' sets design capability explicitly False.
4. Unknown capability is omitted from execution constraints in diagnosis snapshot.
5. Long multi-intent founder message cannot create unrelated negative facts.
6. Second critic REVISE triggers QUALITY_BAR_NOT_MET.
7. Rejected experiment is never marked accepted / READY.
8. Terminal renderer displays EXPERIMENT DESIGN NEEDS REVISION with issues and assumption.
9. Diagnosis snapshot, weak link, and critical assumption survive experiment failure.
10. Simple capacity answer ('15 hours each') resolves to BRIEF_CONTINUE.
11. Material founder concern ('is this problem worth to ship?') resolves to DIRECT_ANSWER.
12. Advisor Composer preserves planner's question_id and information_target.
"""

import pytest
from unittest.mock import MagicMock

from startary.domain.enums import (
    ConversationalResponseMode,
    ExperimentDesignStatus,
    WorkflowPhase,
)
from startary.domain.provenance import (
    KnowledgeField,
    unknown_field,
    validate_state_grounding,
)
from startary.domain.terminal_state import (
    TerminalStatus,
    render_terminal_state,
    resolve_founder_facing_status,
)
from startary.domain.turn_control import (
    NextTurnDecision,
    determine_conversational_response_mode,
    should_surface_diagnosis_delta,
)
from startary.graph.builder import route_on_experiment_critique
from startary.graph.nodes import (
    compose_advisor_response,
    compose_startup_diagnosis,
    handle_experiment_failure_stop,
    merge_execution_state,
)
from startary.llm.schemas import (
    AdvisorResponseLLMOutput,
    ExtractedKnowledgeFieldLLMOutput,
    StartupDiagnosisSnapshotLLMOutput,
    StartupStateDeltaLLMOutput,
)


# ==============================================================================
# PART A: STARTUP STATE GROUNDING INTEGRITY
# ==============================================================================

def test_coding_capability_does_not_set_design_false():
    """1. Founder stating coding skill must NOT infer design_capability=false."""
    from startary.domain.execution_context import ExecutionContext
    founder_turn = "We're two founders and both know how to code."
    candidate_delta = {
        "has_technical_capability": {"value": "true", "provenance": "EXPLICIT", "confidence": "HIGH"},
        "has_design_capability": {"value": "false", "provenance": "INFERRED", "confidence": "LOW"},
    }
    validated = validate_state_grounding({}, candidate_delta, founder_turn)
    assert validated["has_technical_capability"] is not None
    assert validated["has_technical_capability"]["value"] == "true"
    assert validated["has_design_capability"] is None

    # Merge check
    state = {
        "founder_messages": [{"role": "founder", "content": founder_turn}],
        "startup_state_delta": validated,
        "execution_context": ExecutionContext().model_dump(),
    }
    merged = merge_execution_state(state)
    exec_ctx = merged["execution_context"]
    assert exec_ctx["technical_capability"]["value"] == "true"
    assert exec_ctx["technical_capability"]["provenance"] == "EXPLICIT"
    assert exec_ctx["design_capability"]["provenance"] == "UNKNOWN"
    assert exec_ctx["design_capability"]["value"] is None


def test_coding_capability_does_not_set_marketing_false():
    """2. Founder stating coding skill must NOT infer marketing_capability=false."""
    from startary.domain.execution_context import ExecutionContext
    founder_turn = "We are technical founders and built the backend ourselves."
    candidate_delta = {
        "has_technical_capability": {"value": "true", "provenance": "EXPLICIT"},
        "has_marketing_capability": {"value": "false", "provenance": "INFERRED"},
    }
    validated = validate_state_grounding({}, candidate_delta, founder_turn)
    assert validated["has_marketing_capability"] is None

    state = {
        "founder_messages": [{"role": "founder", "content": founder_turn}],
        "startup_state_delta": candidate_delta,
        "execution_context": ExecutionContext().model_dump(),
    }
    merged = merge_execution_state(state)
    exec_ctx = merged["execution_context"]
    assert exec_ctx["marketing_capability"]["provenance"] == "UNKNOWN"
    assert exec_ctx["marketing_capability"]["value"] is None


def test_explicit_absence_sets_known_absence():
    """3. Explicit 'Neither of us knows design' legitimately sets design=false with EXPLICIT provenance."""
    founder_turn = "We can code, but neither of us knows design or Figma."
    candidate_delta = {
        "has_technical_capability": {"value": "true", "provenance": "EXPLICIT"},
        "has_design_capability": {"value": "false", "provenance": "EXPLICIT"},
    }
    validated = validate_state_grounding({}, candidate_delta, founder_turn)
    assert validated["has_design_capability"] is not None
    assert validated["has_design_capability"]["value"] == "false"

    state = {
        "founder_messages": [{"role": "founder", "content": founder_turn}],
        "startup_state_delta": validated,
    }
    merged = merge_execution_state(state)
    exec_ctx = merged["execution_context"]
    assert exec_ctx["design_capability"]["provenance"] == "EXPLICIT"
    assert exec_ctx["design_capability"]["value"] == "false"


def test_unknown_capability_omitted_from_execution_constraints(mocker):
    """4. Unknown design capability must NOT appear under EXECUTION CONSTRAINTS in diagnosis snapshot."""
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_snapshot = StartupDiagnosisSnapshotLLMOutput(
        stage="RAW_IDEA",
        stage_explanation="Early concept stage.",
        grounded_areas=["Founder technical ability"],
        current_advantages=["In-house engineering"],
        execution_constraints=[
            "Neither founder has design capability",
            "Only 15 hours per week per founder",
            "Bootstrap budget under $500",
        ],
        major_uncertainties=["Merchant willingness to reduce returns"],
        weak_link="Problem",
        weak_link_reason="Problem needs verification.",
        weak_link_confidence="MEDIUM",
        weak_link_confidence_explanation="Competing risks exist.",
        competing_risks=["Demand Signal", "Value Proposition"],
        reality_summary="One store showed interest but broader pattern is unproven.",
        recommended_learning_goal="Verify whether other stores experience returns.",
        critical_assumption="Fashion merchants experience high return costs.",
    )
    mock_structured.invoke.return_value = mock_snapshot
    mock_llm.with_structured_output.return_value = mock_structured

    mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
    mocker.patch("startary.graph.nodes.get_settings", return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"))

    # In state, design_capability is UNKNOWN
    state = {
        "project_id": "proj-m404-test",
        "iteration": 1,
        "weak_area": "Problem",
        "active_assumption": {"statement": "Fashion merchants experience high return costs."},
        "weak_link_discrimination": {"confidence": "MEDIUM"},
        "reality_check": {"summary": "Early indicators exist."},
        "execution_context": {
            "design_capability": unknown_field().model_dump(),
            "technical_capability": {"value": True, "provenance": "EXPLICIT"},
        },
    }
    res = compose_startup_diagnosis(state)
    snapshot = res["diagnosis_snapshot"]
    constraints = snapshot["execution_constraints"]

    # Verify design constraint was stripped
    assert not any("design" in c.lower() for c in constraints)
    # Grounded constraints preserved
    assert any("15 hours" in c.lower() for c in constraints)
    assert any("budget" in c.lower() for c in constraints)


def test_multi_intent_message_does_not_expand_unrelated_facts():
    """5. Long multi-intent message must not populate speculative zero-metrics or unmentioned capabilities."""
    founder_turn = (
        "We talked to 1 store owner yesterday and our prototype is still very buggy. "
        "Do you think this return problem is even worth pursuing?"
    )
    candidate_delta = {
        "prototype_exists": {"value": "true", "provenance": "EXPLICIT"},
        "revenue": {"value": "0", "provenance": "INFERRED"},
        "active_users": {"value": "0", "provenance": "INFERRED"},
        "paying_customers": {"value": "0", "provenance": "INFERRED"},
        "has_marketing_capability": {"value": "false", "provenance": "INFERRED"},
        "has_design_capability": {"value": "false", "provenance": "INFERRED"},
    }
    validated = validate_state_grounding({}, candidate_delta, founder_turn)
    # Prototype was mentioned
    assert validated["prototype_exists"] is not None
    # Unmentioned / speculative negative claims are stripped
    assert validated["revenue"] is None
    assert validated["active_users"] is None
    assert validated["paying_customers"] is None
    assert validated["has_marketing_capability"] is None
    assert validated["has_design_capability"] is None


# ==============================================================================
# PART B: EXPERIMENT RECOVERY INTEGRITY
# ==============================================================================

def test_second_critic_revise_sets_quality_bar_not_met():
    """6. Revision budget exhaustion routes to handle_experiment_failure_stop and sets QUALITY_BAR_NOT_MET."""
    state = {
        "project_id": "proj-m404-b",
        "critic_revision_count": 2,
        "experiment_critic_output": {
            "verdict": "REVISE",
            "issues": ["Sampling size exceeds founder customer access", "Contradiction criteria require top-three priority"],
        },
    }
    # Builder routing
    route = route_on_experiment_critique(state)
    assert route == "handle_experiment_failure_stop"

    # Node execution
    updates = handle_experiment_failure_stop(state)
    assert updates["experiment_design_status"] == ExperimentDesignStatus.QUALITY_BAR_NOT_MET.value
    assert updates["phase"] == WorkflowPhase.WAITING_FOR_LEARNING_INPUT
    assert "unable to produce an accepted experiment" in updates["status_message"].lower()


def test_rejected_experiment_is_never_marked_accepted():
    """7. Exhausting revisions never marks the experiment as accepted or READY."""
    state = {
        "phase": WorkflowPhase.TEST,
        "experiment_design_status": ExperimentDesignStatus.QUALITY_BAR_NOT_MET.value,
        "status_message": "Experiment generation stopped: unable to produce an accepted experiment.",
        "active_experiment": None,
    }
    status = resolve_founder_facing_status(state)
    assert status == TerminalStatus.EXPERIMENT_DESIGN_NEEDS_REVISION
    assert status != TerminalStatus.EXPERIMENT_READY
    assert status != TerminalStatus.INITIAL_DIAGNOSIS_COMPLETE


def test_terminal_renderer_shows_experiment_design_needs_revision():
    """8. Terminal renderer outputs the exact EXPERIMENT DESIGN NEEDS REVISION format."""
    state = {
        "phase": WorkflowPhase.TEST,
        "experiment_design_status": ExperimentDesignStatus.QUALITY_BAR_NOT_MET.value,
        "active_assumption": {"statement": "Independent boutique merchants lose over 5 hours/week processing returns."},
        "experiment_critic_output": {
            "issues": [
                "Decision rule requires top-three business priority (overpowering assumption)",
                "Requires 10 store interviews in 3 days with no existing merchant network",
            ]
        },
    }
    output = render_terminal_state(state)
    assert "STARTARY — EXPERIMENT DESIGN NEEDS REVISION" in output
    assert "Independent boutique merchants lose over 5 hours/week processing returns." in output
    assert "Why:" in output
    assert "Decision rule requires top-three business priority" in output
    assert "Requires 10 store interviews in 3 days" in output
    assert "The diagnosis is still valid, but Startary has not yet produced a test reliable enough to run." in output
    assert "INITIAL DIAGNOSIS COMPLETE" not in output


def test_diagnosis_and_assumption_preserved_after_experiment_failure():
    """9. Diagnosis snapshot, weak link, and critical assumption survive experiment design failure."""
    ca_stmt = "Independent fashion stores experience significant pain from returns."
    state = {
        "project_id": "proj-m404-preserve",
        "phase": WorkflowPhase.TEST,
        "weak_area": "Problem",
        "active_assumption": {"id": "ca-001", "statement": ca_stmt},
        "diagnosis_snapshot": {"id": "snap-001", "weak_link": "Problem"},
        "experiment_critic_output": {"issues": ["Flawed test plan"]},
    }
    updates = handle_experiment_failure_stop(state)
    merged_state = {**state, **updates}

    assert merged_state["weak_area"] == "Problem"
    assert merged_state["active_assumption"]["statement"] == ca_stmt
    assert merged_state["diagnosis_snapshot"]["weak_link"] == "Problem"
    assert merged_state["experiment_design_status"] == ExperimentDesignStatus.QUALITY_BAR_NOT_MET.value


# ==============================================================================
# PART C: CONVERSATIONAL NATURALNESS
# ==============================================================================

def test_simple_capacity_answer_uses_brief_continue():
    """10. Simple factual answer ('15 hours each') resolves to BRIEF_CONTINUE and avoids delta commentary."""
    founder_turn = "15 hours each."
    state = {
        "phase": WorkflowPhase.UNDERSTAND,
        "startup_stage": "RAW_IDEA",
        "weak_area": "Problem",
        "active_next_turn_decision": {
            "question_id": "q_budget",
            "information_target": "RESOURCE",
            "canonical_question": "What budget can you allocate?",
        },
    }
    prev_state = dict(state)

    mode = determine_conversational_response_mode(
        state=state,
        founder_interpretation={"is_question": False, "is_correction": False},
        founder_turn=founder_turn,
        previous_state=prev_state,
    )
    assert mode == ConversationalResponseMode.BRIEF_CONTINUE

    surface_delta = should_surface_diagnosis_delta(prev_state, state, founder_turn)
    assert surface_delta is False


def test_material_founder_concern_triggers_direct_answer():
    """11. Founder asking 'is this problem worth to ship?' triggers DIRECT_ANSWER mode."""
    founder_turn = "is this problem worth to ship?"
    state = {
        "phase": WorkflowPhase.UNDERSTAND,
        "startup_stage": "RAW_IDEA",
        "active_next_turn_decision": {
            "question_id": "q_cust",
            "information_target": "CUSTOMER_ACCESS",
            "canonical_question": "How do you reach other stores?",
        },
    }
    mode = determine_conversational_response_mode(
        state=state,
        founder_interpretation={"is_question": True, "asks_for_guidance": True},
        founder_turn=founder_turn,
    )
    assert mode == ConversationalResponseMode.DIRECT_ANSWER


def test_composer_cannot_change_planner_target(mocker):
    """12. Advisor composer strictly preserves planner's question_id and information_target."""
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_advisor_out = AdvisorResponseLLMOutput(
        question_id="q_target_123",
        next_question_target="CAPACITY",
        next_question_rephrased="How many hours per week can you each dedicate?",
        acknowledge="Got it.",
        response_mode="BRIEF_CONTINUE",
        response_text="Got it. How many hours per week can you each dedicate?",
    )
    mock_structured.invoke.return_value = mock_advisor_out
    mock_llm.with_structured_output.return_value = mock_structured

    mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
    mocker.patch("startary.graph.nodes.get_settings", return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"))

    state = {
        "project_id": "proj-m404-comp",
        "phase": WorkflowPhase.UNDERSTAND,
        "active_next_turn_decision": {
            "question_id": "q_target_123",
            "information_target": "CAPACITY",
            "canonical_question": "How much weekly time can each founder spend?",
            "should_ask_question": True,
        },
        "founder_messages": [{"role": "founder", "content": "We both code full-time at our day jobs."}],
    }
    res = compose_advisor_response(state)
    adv = res["latest_advisor_response"]
    assert adv["question_id"] == "q_target_123"
    assert adv["next_question_target"] == "CAPACITY"
    assert adv["response_mode"] == "BRIEF_CONTINUE"
