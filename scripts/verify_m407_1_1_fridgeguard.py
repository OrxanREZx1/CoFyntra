import sys
from startary.domain.enums import (
    CriticVerdict,
    ExecutionReadiness,
    ExperimentDesignStatus,
    ExperimentStatus,
    ExperimentType,
    WorkflowPhase,
)
from startary.domain.terminal_state import (
    TerminalStatus,
    can_render_waiting_for_evidence,
    render_terminal_state,
    resolve_founder_facing_status,
)
from startary.graph.builder import route_on_experiment_critique
from startary.graph.nodes import (
    handle_experiment_failure_stop,
    increment_iteration,
)

def run_fridgeguard_verification():
    print("=" * 50)
    print("STARTING FRIDGEGUARD LIFECYCLE VERIFICATION")
    print("=" * 50)

    # Iteration 1
    exp1_id = "exp-fg-001"
    asmp1_id = "asmp-fg-001"
    state_iter1 = {
        "project_id": "proj-fridgeguard-live",
        "iteration": 1,
        "phase": WorkflowPhase.WAITING_FOR_EVIDENCE,
        "weak_area": "Problem",
        "active_assumption": {
            "id": asmp1_id,
            "statement": "Consumers frequently throw away spoiled food from their refrigerator.",
            "area": "Problem",
        },
        "active_experiment": {
            "id": exp1_id,
            "assumption_id": asmp1_id,
            "status": ExperimentStatus.IN_PROGRESS.value,
            "objective": "Measure food waste frequency among urban professionals",
            "created_iteration": 1,
        },
        "active_execution_handoff": {
            "experiment_id": exp1_id,
            "assumption_id": asmp1_id,
            "what_we_are_testing": "Food spoilage frequency in refrigerators",
            "response_text": "Keep a fridge inventory log for 7 days with 6 households.",
        },
        "execution_handoff_text": "Keep a fridge inventory log for 7 days with 6 households.",
        "can_execute_result": {
            "readiness": ExecutionReadiness.CAN_EXECUTE.value,
            "reason": "Founder has household tester access.",
        },
        "previous_experiments": [
            {
                "id": exp1_id,
                "objective": "Measure food waste frequency among urban professionals",
                "status": ExperimentStatus.IN_PROGRESS.value,
                "created_iteration": 1,
            }
        ],
        "founder_evidence": [
            {
                "iteration": 1,
                "assumption_id": asmp1_id,
                "evidence_type": "USER_OBSERVATION",
                "content": "6/6 households recorded throwing away produce weekly.",
                "quality_score": 0.90,
            }
        ],
        "evidence_analysis": {
            "iteration": 1,
            "conclusion": "SUPPORTED",
            "findings": ["Severe recurring food waste confirmed."],
        },
        "startup_state": {
            "area_knowledge": {
                "Problem": {"status": "KNOWN", "confidence": "HIGH"},
            }
        },
    }

    print("\n[1] Iteration 1 completed with SUPPORTED evidence. Advancing iteration...")
    iter_updates = increment_iteration(state_iter1)
    state_iter2 = {**state_iter1, **iter_updates}

    assert state_iter2["iteration"] == 2, "Iteration did not advance to 2"
    assert state_iter2["phase"] == WorkflowPhase.LEARN, "Phase did not become LEARN"
    assert state_iter2["active_experiment"] is None, "active_experiment not reset"
    assert state_iter2["active_execution_handoff"] is None, "active_execution_handoff not reset"
    assert state_iter2["can_execute_result"] is None, "can_execute_result not reset"

    print("    Iteration 2 started cleanly: active execution artifacts reset to None.")

    # Iteration 2: Re-diagnosis selects Existing Alternatives
    print("\n[2] Re-diagnosis in Iteration 2 selects 'Existing Alternatives' as weak link.")
    state_iter2["weak_area"] = "Existing Alternatives"
    state_iter2["active_assumption"] = {
        "id": "asmp-fg-002",
        "statement": "Existing smart fridge apps and reminder lists fail because users abandon manual logging.",
        "area": "Existing Alternatives",
    }
    state_iter2["diagnosis_snapshot"] = {
        "weak_link": "Existing Alternatives",
        "critical_assumption": state_iter2["active_assumption"]["statement"],
        "stage": "PRE_MVP",
    }

    # First experiment proposed & Critic returns REVISE
    print("\n[3] Experiment 1 generated for Iteration 2. Critic returns REVISE (attempt 1).")
    state_iter2["proposed_experiment_output"] = {
        "objective": "Compare manual logging drop-off vs competitor apps",
        "method": "Online survey on Reddit",
    }
    state_iter2["valid_inferences"] = ["User frustration with manual entry"]
    state_iter2["invalid_inferences"] = ["Automated solution willingness to pay"]
    state_iter2["experiment_critic_output"] = {
        "verdict": CriticVerdict.REVISE.value,
        "issues": ["Reddit self-reporting lacks direct observation of abandonment triggers."],
    }
    state_iter2["critic_revision_count"] = 1
    assert route_on_experiment_critique(state_iter2) == "generate_experiment"

    # Revision proposed & Critic returns REVISE again
    print("\n[4] Revision generated. Critic returns REVISE (attempt 2, budget exhausted).")
    state_iter2["proposed_experiment_output"] = {
        "objective": "App store sentiment analysis of competitors",
        "method": "NLP review scraping",
    }
    state_iter2["experiment_critic_output"] = {
        "verdict": CriticVerdict.REVISE.value,
        "issues": [
            "App store reviews do not reliably distinguish a true market gap from weaknesses in a small or poorly selected competitor set."
        ],
    }
    state_iter2["critic_revision_count"] = 2

    # Router sends to handle_experiment_failure_stop
    next_node = route_on_experiment_critique(state_iter2)
    assert next_node == "handle_experiment_failure_stop", f"Expected handle_experiment_failure_stop, got {next_node}"
    print(f"    Router correctly dispatched to: {next_node}")

    # Handle experiment failure stop executes
    fail_updates = handle_experiment_failure_stop(state_iter2)
    final_state = {**state_iter2, **fail_updates}


    print("\n[5] Verifying final state invariants...")

    # Check 1: phase is WAITING_FOR_LEARNING_INPUT
    assert final_state["phase"] == WorkflowPhase.WAITING_FOR_LEARNING_INPUT, f"Phase was {final_state['phase']}"
    print(f"    [PASS] phase == WAITING_FOR_LEARNING_INPUT ({final_state['phase']})")

    # Check 2: no stale previous experiment or execution handoff active
    assert final_state["active_experiment"] is None, "active_experiment is not None"
    assert final_state["active_execution_handoff"] is None, "active_execution_handoff is not None"
    assert final_state["can_execute_result"] is None, "can_execute_result is not None"
    print("    [PASS] active_experiment, active_execution_handoff, can_execute_result are all None")

    # Check 3: fallback learning action is shown and active
    assert final_state.get("fallback_learning_action") is not None, "fallback_learning_action is missing"
    assert final_state.get("learning_input_mode") == "FALLBACK", "learning_input_mode != FALLBACK"
    print("    [PASS] fallback_learning_action is active in FALLBACK mode")

    # Check 4: can_render_waiting_for_evidence is False
    can_render_wfe = can_render_waiting_for_evidence(final_state)
    assert can_render_wfe is False, "can_render_waiting_for_evidence returned True!"
    print("    [PASS] can_render_waiting_for_evidence == False")

    # Check 5: resolve_founder_facing_status is EXPERIMENT_DESIGN_NEEDS_REVISION
    status = resolve_founder_facing_status(final_state)
    assert status == TerminalStatus.EXPERIMENT_DESIGN_NEEDS_REVISION, f"Status was {status}"
    assert status != TerminalStatus.WAITING_FOR_EVIDENCE
    print(f"    [PASS] resolve_founder_facing_status == {status.value} (NOT WAITING_FOR_EVIDENCE)")

    # Check 6: render_terminal_state output verification
    banner = render_terminal_state(final_state)
    assert "WAITING FOR EVIDENCE" not in banner, "Rendered stale WAITING FOR EVIDENCE banner!"
    assert "Your experiment is ready." not in banner, "Rendered 'Your experiment is ready.'!"
    assert "EXPERIMENT DESIGN NEEDS REVISION" in banner, "Missing EXPERIMENT DESIGN NEEDS REVISION banner"
    assert "Here is the safest next learning action:" in banner, "Missing fallback intro"
    assert "WHAT DO WE STILL NEED TO LEARN?" in banner, "Missing fallback section"
    print("    [PASS] Terminal banner contains Fallback Learning Action and NO 'WAITING FOR EVIDENCE'")

    # Check 7: historical iteration 1 records intact
    assert len(final_state["previous_assumptions"]) == 1
    assert final_state["previous_assumptions"][0]["id"] == asmp1_id
    assert len(final_state["previous_experiments"]) == 1
    assert final_state["previous_experiments"][0]["id"] == exp1_id
    assert len(final_state["founder_evidence"]) == 1
    print("    [PASS] Historical iteration 1 assumption, experiment, and evidence remain intact")

    print("\n" + "=" * 50)
    print("FRIDGEGUARD RENDERED TERMINAL BANNER:")
    print("=" * 50)
    print(banner)
    print("=" * 50)
    print("ALL 7 VERIFICATION CHECKS PASSED PERFECTLY!")
    print("=" * 50)

if __name__ == '__main__':
    run_fridgeguard_verification()
