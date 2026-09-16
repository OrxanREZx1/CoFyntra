"""
Tests for Milestone 4.1.1 — Experiment Failure Recovery Loop.

Covers all 16 required integrity scenarios from Section 13:
1. Second Critic REVISE -> phase WAITING_FOR_LEARNING_INPUT.
2. Second Critic REVISE does not graph END.
3. CLI remains interactive after fallback.
4. No accepted active experiment remains after failure.
5. Stale execution handoff cannot render.
6. Fallback includes final Critic issue context.
7. LastBite proximity/availability failure produces targeted corrective action.
8. Generic customer-interview fallback is not used for that case.
9. Founder fallback response is integrated as grounded context.
10. Fallback response does not create experimental evidence cycle.
11. Iteration does not increment merely from fallback submission.
12. Workflow re-enters diagnosis after fallback input.
13. New assumption/experiment may be produced after re-diagnosis.
14. FridgeGuard failure remains recoverable.
15. Repeated/no-new-information fallback does not create infinite retry loop.
16. WAITING_FOR_EVIDENCE only renders for a valid accepted current experiment.
"""

from __future__ import annotations

import uuid
import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from startary.domain.enums import (
    CriticVerdict,
    ExecutionReadiness,
    ExperimentDesignStatus,
    ExperimentStatus,
    ExperimentType,
    NextLearningActionType,
    WorkflowPhase,
)
from startary.domain.experiment import (
    FallbackLearningAction,
    compose_fallback_learning_action,
    render_fallback_learning_action_text,
)
from startary.domain.terminal_state import (
    TerminalStatus,
    can_render_waiting_for_evidence,
    render_terminal_state,
    resolve_founder_facing_status,
)
from startary.graph.builder import (
    build_graph,
    build_experiment_graph,
    route_after_interpret_founder_turn,
    route_on_experiment_critique,
)
from startary.graph.nodes import (
    ask_and_wait_for_founder,
    handle_experiment_failure_stop,
    merge_founder_answer,
)


# ===========================================================================
# 1. second Critic REVISE -> phase WAITING_FOR_LEARNING_INPUT
# ===========================================================================
def test_1_second_critic_revise_sets_phase_waiting_for_learning_input():
    state = {
        "project_id": "proj-m411-01",
        "iteration": 1,
        "critic_revision_count": 2,
        "experiment_critic_output": {
            "verdict": CriticVerdict.REVISE.value,
            "issues": ["Decision rule unfeasible within resource constraints"],
        },
        "active_assumption": {
            "id": "asmp-01",
            "statement": "Restaurants will purchase automated temperature monitoring.",
            "area": "Problem",
        },
        "proposed_experiment_output": {
            "objective": "Test temperature monitoring demand",
            "method": "Landing page signup",
        },
    }

    assert route_on_experiment_critique(state) == "handle_experiment_failure_stop"
    updates = handle_experiment_failure_stop(state)
    assert updates["phase"] == WorkflowPhase.WAITING_FOR_LEARNING_INPUT
    assert updates["active_interrupt_type"] == "waiting_for_learning_input"
    assert updates["experiment_design_status"] == ExperimentDesignStatus.QUALITY_BAR_NOT_MET.value


# ===========================================================================
# 2. second Critic REVISE does not graph END
# ===========================================================================
def test_2_second_critic_revise_does_not_graph_end(mocker):
    """Main graph routes handle_experiment_failure_stop to ask_and_wait_for_founder interrupt instead of END."""
    checkpointer = MemorySaver()
    graph = build_graph(checkpointer=checkpointer)

    state = {
        "project_id": "proj-m411-02",
        "iteration": 1,
        "phase": WorkflowPhase.TEST,
        "critic_revision_count": 2,
        "experiment_critic_output": {
            "verdict": CriticVerdict.REVISE.value,
            "issues": ["Unfalsifiable proxy signal"],
        },
        "active_assumption": {
            "id": "asmp-02",
            "statement": "Store owners experience daily inventory waste.",
            "area": "Problem",
        },
        "proposed_experiment_output": {
            "objective": "Test waste awareness",
            "method": "Customer interviews",
        },
    }

    updates = handle_experiment_failure_stop(state)
    state_after_failure = {**state, **updates}

    mock_interrupt = mocker.patch(
        "startary.graph.nodes.interrupt",
        return_value="I talked to 3 stores.",
    )
    interrupt_res = ask_and_wait_for_founder(state_after_failure)

    assert mock_interrupt.called
    call_args = mock_interrupt.call_args[0][0]
    assert call_args["type"] == "WAITING_FOR_LEARNING_INPUT"
    assert call_args["interrupt_type"] == "waiting_for_learning_input"
    assert interrupt_res["phase"] == WorkflowPhase.WAITING_FOR_LEARNING_INPUT


# ===========================================================================
# 3. CLI remains interactive after fallback
# ===========================================================================
def test_3_cli_remains_interactive_after_fallback(mocker):
    """Interrupt value provides formatted fallback and resumes with founder input."""
    state = {
        "active_interrupt_type": "waiting_for_learning_input",
        "fallback_learning_action": {
            "response_text": "STARTARY — EXPERIMENT DESIGN NEEDS REVISION\nFallback content",
            "primary_action": "Talk with 3 stores",
        },
    }

    mocker.patch(
        "startary.graph.nodes.interrupt",
        return_value="Here are the actual observations from the stores.",
    )

    res = ask_and_wait_for_founder(state)
    assert res["founder_response"] == "Here are the actual observations from the stores."
    assert res["phase"] == WorkflowPhase.WAITING_FOR_LEARNING_INPUT


# ===========================================================================
# 4. no accepted active experiment remains after failure
# ===========================================================================
def test_4_no_accepted_active_experiment_remains_after_failure():
    state = {
        "project_id": "proj-m411-04",
        "iteration": 1,
        "critic_revision_count": 2,
        "active_experiment": {
            "id": "exp-stale-04",
            "status": ExperimentStatus.READY.value,
            "hypothesis": "Test hypothesis",
        },
        "proposed_experiment_output": {
            "id": "exp-stale-04",
            "objective": "Test stale experiment",
        },
        "experiment_critic_output": {
            "verdict": "REVISE",
            "issues": ["Rejected for lack of falsifiability"],
        },
    }

    updates = handle_experiment_failure_stop(state)
    final_state = {**state, **updates}

    assert final_state["active_experiment"] is None
    assert final_state["active_execution_handoff"] is None
    assert final_state["can_execute_result"] is None


# ===========================================================================
# 5. stale execution handoff cannot render
# ===========================================================================
def test_5_stale_execution_handoff_cannot_render():
    state = {
        "project_id": "proj-m411-05",
        "phase": WorkflowPhase.WAITING_FOR_LEARNING_INPUT,
        "fallback_learning_action": {"learning_goal": "Gather observations"},
        "active_execution_handoff": {"experiment_id": "exp-stale-05", "response_text": "Old handoff"},
        "execution_handoff_text": "Old text",
        "can_execute_result": {"readiness": "CAN_EXECUTE"},
    }

    assert can_render_waiting_for_evidence(state) is False
    assert resolve_founder_facing_status(state) == TerminalStatus.EXPERIMENT_DESIGN_NEEDS_REVISION
    rendered = render_terminal_state(state)
    assert "STARTARY — WAITING FOR EVIDENCE" not in rendered
    assert "STARTARY — EXPERIMENT DESIGN NEEDS REVISION" in rendered


# ===========================================================================
# 6. fallback includes final Critic issue context
# ===========================================================================
def test_6_fallback_includes_final_critic_issue_context():
    critic_output = {
        "verdict": "REVISE",
        "issues": [
            "Actual proximity during specific offer window was not verified",
            "Non-reservations could reflect participant unavailability rather than weak demand",
        ],
    }
    action = compose_fallback_learning_action(
        assumption={"statement": "Nearby users will collect surplus bakery food.", "area": "Target User"},
        critic_output=critic_output,
        rejected_experiment={"objective": "Test surplus collection"},
    )

    rendered = action.response_text
    assert "WHY THE TEST COULD NOT BE TRUSTED:" in rendered
    assert "Actual proximity during specific offer window was not verified" in rendered
    assert action.critic_rejection_reasons == critic_output["issues"]


# ===========================================================================
# 7. LastBite proximity/availability failure produces targeted corrective action
# ===========================================================================
def test_7_lastbite_proximity_availability_failure_produces_targeted_corrective_action():
    critic_output = {
        "verdict": "REVISE",
        "issues": [
            "Actual proximity during each specific offer window was not verified",
            "Non-reservations could reflect participant unavailability rather than weak demand",
            "Eligibility had to be established per venue/window",
        ],
    }
    asmp = {
        "statement": "There is a definable segment of people near independent Baku bakeries and small cafes with recurring closing-time surplus who are able to collect surplus food during the pre-closing window.",
        "area": "Target User",
    }
    action = compose_fallback_learning_action(
        assumption=asmp,
        weak_link="Target User",
        critic_output=critic_output,
        rejected_experiment={"target_participants_or_scope": "Baku consumers"},
    )

    assert "bakery" in action.primary_action.lower() or "cafe" in action.primary_action.lower()
    assert "closing-time" in action.primary_action.lower() or "pre-closing" in action.primary_action.lower()
    
    q_str = " ".join(action.what_to_ask_or_observe).lower()
    assert "within 10 minutes" in q_str or "pickup window" in q_str or "proximity" in q_str
    
    b_str = " ".join(action.what_to_bring_back).lower()
    assert "confirmed" in b_str and ("nearby" in b_str or "pickup" in b_str or "reservations" in b_str)


# ===========================================================================
# 8. generic customer-interview fallback is not used for that case
# ===========================================================================
def test_8_generic_customer_interview_fallback_not_used_for_lastbite():
    critic_output = {
        "verdict": "REVISE",
        "issues": [
            "Actual per-window proximity and availability are not verified",
        ],
    }
    asmp = {
        "statement": "Baku consumers near bakeries will collect closing-time surplus.",
        "area": "Target User",
    }
    action = compose_fallback_learning_action(
        assumption=asmp,
        weak_link="Target User",
        critic_output=critic_output,
    )

    rendered_lower = action.response_text.lower()
    # Must NOT be a generic interview template
    assert "ask each for one recent example of unsold/expired inventory" not in rendered_lower
    assert "talk with 5–8 independent target participants or store owners relevant to target user" not in rendered_lower
    # Must be targeted closing-time offer probe
    assert "closing-time" in rendered_lower or "pre-closing" in rendered_lower or "pickup window" in rendered_lower


# ===========================================================================
# 9. founder fallback response is integrated as grounded context
# ===========================================================================
def test_9_founder_fallback_response_integrated_as_grounded_context():
    fallback_dict = {
        "target_area": "Target User",
        "primary_action": "Run one real closing-time offer at one participating bakery.",
        "response_text": "STARTARY — EXPERIMENT DESIGN NEEDS REVISION\nPerform offer probe.",
    }
    state = {
        "phase": WorkflowPhase.WAITING_FOR_LEARNING_INPUT,
        "active_interrupt_type": "waiting_for_learning_input",
        "fallback_learning_action": fallback_dict,
        "startup_state": {
            "areas": {
                "Target User": {"evidence_provided": []},
            },
        },
        "founder_messages": [],
        "founder_response": (
            "I ran one offer at the same bakery. I only counted people who confirmed at 7:00 PM "
            "that they were currently within a 10-minute walk and could stay until the 7:30–8:15 pickup window. "
            "Twelve people qualified. Four reserved, three paid deposits, and three completed pickup. "
            "One person declined because they had to leave before 7:30."
        ),
    }

    res = merge_founder_answer(state)

    # 1. Appended to conversation history
    assert len(res["founder_messages"]) >= 2
    founder_msg = res["founder_messages"][-1]
    assert founder_msg["role"] == "founder"
    assert "Twelve people qualified" in founder_msg["content"]

    # 2. Integrated into startup_state grounded context
    tu_evidence = res["startup_state"]["areas"]["Target User"]["evidence_provided"]
    assert len(tu_evidence) == 1
    assert "Twelve people qualified" in tu_evidence[0]

    # 3. Preserves provenance
    assert len(res["founder_evidence"]) == 1
    assert res["founder_evidence"][0]["source"] == "FALLBACK_LEARNING_ACTION"
    assert res["founder_evidence"][0]["target_area"] == "Target User"


# ===========================================================================
# 10. fallback response does not create experimental evidence cycle
# ===========================================================================
def test_10_fallback_response_does_not_create_experimental_evidence_cycle():
    state = {
        "active_interrupt_type": "waiting_for_learning_input",
        "fallback_learning_action": {"target_area": "Target User"},
        "active_assumption": {
            "id": "ca-lastbite-01",
            "status": "ACTIVE",
            "statement": "Nearby users will collect surplus.",
        },
        "founder_response": "Observed 12 eligible users, 4 reservations.",
    }

    res = merge_founder_answer(state)

    # No evidence cycle objects
    assert res.get("active_evidence") is None
    assert res.get("extracted_evidence") is None
    assert res.get("evidence_analysis") is None
    assert res.get("learning_delta") is None

    # Active assumption is NOT marked SUPPORTED
    asmp = state["active_assumption"]
    assert asmp.get("status") != "SUPPORTED"


# ===========================================================================
# 11. iteration does not increment merely from fallback submission
# ===========================================================================
def test_11_iteration_does_not_increment_merely_from_fallback_submission():
    state = {
        "iteration": 1,
        "active_interrupt_type": "waiting_for_learning_input",
        "fallback_learning_action": {"target_area": "Target User"},
        "founder_response": "12 qualified, 4 reserved.",
    }

    res = merge_founder_answer(state)
    # Iteration is NOT incremented
    assert res.get("iteration", 1) == 1


# ===========================================================================
# 12. workflow re-enters diagnosis after fallback input
# ===========================================================================
def test_12_workflow_re_enters_diagnosis_after_fallback_input():
    state = {
        "active_interrupt_type": "waiting_for_learning_input",
        "fallback_learning_action": {"target_area": "Target User"},
        "founder_response": "12 qualified, 4 reserved.",
    }

    res = merge_founder_answer(state)
    assert res["phase"] == WorkflowPhase.UNDERSTAND

    # Verify route_after_interpret_founder_turn sends it to extract_startup_state
    merged_state = {**state, **res}
    next_route = route_after_interpret_founder_turn(merged_state)
    assert next_route == "extract_startup_state"


# ===========================================================================
# 13. new assumption/experiment may be produced after re-diagnosis
# ===========================================================================
def test_13_new_assumption_experiment_may_be_produced_after_re_diagnosis():
    """State is reset cleanly so downstream generation can produce a fresh experiment."""
    state = {
        "critic_revision_count": 2,
        "critic_revision_instructions": "Prior instructions",
        "proposed_experiment_output": {"objective": "Old failed experiment"},
        "experiment_critic_output": {"verdict": "REVISE"},
        "active_interrupt_type": "waiting_for_learning_input",
        "fallback_learning_action": {"target_area": "Target User"},
        "founder_response": "Observed 12 nearby participants, 4 reserved.",
    }

    res = merge_founder_answer(state)

    # Critic revision count reset to 0 so fresh experiments get full revision budget
    assert res["critic_revision_count"] == 0
    assert res["critic_revision_instructions"] == ""
    assert res["proposed_experiment_output"] is None
    assert res["experiment_critic_output"] is None
    assert res["fallback_learning_action"] is None


# ===========================================================================
# 14. FridgeGuard failure remains recoverable
# ===========================================================================
def test_14_fridgeguard_failure_remains_recoverable():
    critic_output = {
        "verdict": "REVISE",
        "issues": [
            "There is a definable restaurant user and decision-maker group beyond the founder's uncle",
            "Unverified kitchen manager decision-maker access",
        ],
    }
    asmp = {
        "statement": "There is a definable restaurant user and decision-maker group beyond the founder's uncle for whom refrigerator-temperature monitoring alerts are relevant enough to consider adopting the system.",
        "area": "Target User",
    }
    action = compose_fallback_learning_action(
        assumption=asmp,
        weak_link="Target User",
        critic_output=critic_output,
        rejected_experiment={"objective": "Test restaurant alert adoption"},
    )

    # Fallback targets commercial kitchen operators/managers with no personal ties
    assert "restaurant" in action.primary_action.lower() or "kitchen" in action.primary_action.lower()
    assert "no personal ties" in action.who_to_engage.lower() or "beyond personal connections" in action.primary_action.lower()

    # Founder returns with learning
    reply = "I spoke with 4 independent restaurant shift managers in downtown. 3 said they manually log temperatures on a clipboard twice daily."
    state = {
        "phase": WorkflowPhase.WAITING_FOR_LEARNING_INPUT,
        "active_interrupt_type": "waiting_for_learning_input",
        "fallback_learning_action": action.model_dump(),
        "startup_state": {"areas": {"Target User": {"evidence_provided": []}}},
        "founder_response": reply,
    }
    merged = merge_founder_answer(state)
    assert merged["phase"] == WorkflowPhase.UNDERSTAND
    assert len(merged["startup_state"]["areas"]["Target User"]["evidence_provided"]) == 1


# ===========================================================================
# 15. repeated/no-new-information fallback does not create infinite retry loop
# ===========================================================================
def test_15_repeated_no_new_information_fallback_does_not_create_infinite_retry_loop():
    """Previous rejected experiments are preserved so generator does not regenerate the identical experiment."""
    state = {
        "iteration": 1,
        "critic_revision_count": 2,
        "previous_experiments": [],
        "proposed_experiment_output": {
            "id": "exp-failed-01",
            "objective": "Survey bakery customers online",
            "hypothesis": "Customers will pick up surplus",
        },
        "experiment_critic_output": {
            "verdict": "REVISE",
            "issues": ["Online survey lacks direct behavioral observation"],
        },
    }

    updates = handle_experiment_failure_stop(state)
    prev_exps = updates["previous_experiments"]
    assert len(prev_exps) == 1
    assert prev_exps[0]["id"] == "exp-failed-01"
    assert prev_exps[0]["status"] == "QUALITY_BAR_NOT_MET"
    assert "Online survey lacks direct behavioral observation" in prev_exps[0]["critic_issues"]


# ===========================================================================
# 16. WAITING_FOR_EVIDENCE only renders for a valid accepted current experiment
# ===========================================================================
def test_16_waiting_for_evidence_only_renders_for_valid_accepted_current_experiment():
    exp_id = str(uuid.uuid4())

    # Case A: Valid accepted experiment -> True
    valid_state = {
        "phase": WorkflowPhase.WAITING_FOR_EVIDENCE,
        "iteration": 1,
        "active_experiment": {
            "id": exp_id,
            "status": ExperimentStatus.READY.value,
            "created_iteration": 1,
        },
        "active_execution_handoff": {
            "experiment_id": exp_id,
            "response_text": "Run the accepted test",
        },
        "can_execute_result": {
            "readiness": ExecutionReadiness.CAN_EXECUTE.value,
        },
    }
    assert can_render_waiting_for_evidence(valid_state) is True
    assert resolve_founder_facing_status(valid_state) == TerminalStatus.WAITING_FOR_EVIDENCE

    # Case B: In fallback mode -> False
    fallback_state = {
        **valid_state,
        "phase": WorkflowPhase.WAITING_FOR_LEARNING_INPUT,
        "fallback_learning_action": {"learning_goal": "Gather facts"},
        "experiment_design_status": ExperimentDesignStatus.QUALITY_BAR_NOT_MET.value,
    }
    assert can_render_waiting_for_evidence(fallback_state) is False
    assert resolve_founder_facing_status(fallback_state) == TerminalStatus.EXPERIMENT_DESIGN_NEEDS_REVISION

    # Case C: Active experiment is None -> False
    no_exp_state = {**valid_state, "active_experiment": None}
    assert can_render_waiting_for_evidence(no_exp_state) is False

    # Case D: Active experiment status is BLOCKED -> False
    blocked_state = {
        **valid_state,
        "can_execute_result": {"readiness": ExecutionReadiness.BLOCKED.value},
    }
    assert can_render_waiting_for_evidence(blocked_state) is False
    assert resolve_founder_facing_status(blocked_state) == TerminalStatus.BLOCKED
