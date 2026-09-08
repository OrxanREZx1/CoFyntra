"""
Verification script for Milestone 4.0.3: Execution Handoff & Evidence Admission Integrity.

Validates the three scenarios requested in Requirement 42:
Scenario A: LoopCart guidance question ("Then what should I do?")
  - accepted experiment rendered in full ExecutionHandoff
  - founder asks "Then what should I do?"
  - Startary answers conversationally
  - no Evidence created
  - experiment remains IN_PROGRESS
  - iteration remains 1
  - phase remains WAITING_FOR_EVIDENCE

Scenario B: Future intention ("Okay, I will speak to them tomorrow.")
  - acknowledged encouragingly
  - no Evidence created
  - remains WAITING_FOR_EVIDENCE
  - no analysis or iteration increment

Scenario C: Real results ("I spoke with 6 stores...")
  - classified as ACTUAL_EVIDENCE
  - enters Evidence Extractor & Analyzer
  - valid evidence cycle established
  - iteration increments to 2
  - LearningDelta eligible
"""

import sys
from typing import Any

from startary.domain.enums import (
    AssumptionStatus,
    ExecutionReadiness,
    ExperimentStatus,
    WorkflowPhase,
)
from startary.domain.evidence import (
    EvidenceTurnDisposition,
    has_valid_completed_evidence_cycle,
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
    can_compose_learning_delta,
    classify_waiting_for_evidence_turn,
    compose_execution_handoff,
    increment_iteration,
    interpret_founder_turn,
    respond_to_waiting_founder,
    validate_evidence_analysis,
    wait_for_evidence,
)


def create_loopcart_state() -> StartaryGraphState:
    return {
        "project_id": "proj-loopcart-verify",
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


def verify_scenario_a_guidance_question() -> bool:
    print("\n" + "=" * 60)
    print("SCENARIO A: LoopCart Guidance Question ('Then what should I do?')")
    print("=" * 60)

    state = create_loopcart_state()

    # 1. Verify CAN_EXECUTE routes to compose_execution_handoff
    route_exec = route_after_can_execute(state)
    print(f"1. route_after_can_execute: '{route_exec}'")
    assert route_exec == "compose_execution_handoff"

    # 2. Verify ExecutionHandoff is composed and non-None
    handoff_res = compose_execution_handoff(state)
    state.update(handoff_res)
    handoff_text = state.get("execution_handoff_text")
    print(f"2. ExecutionHandoff created (len={len(handoff_text)}):")
    print("   " + handoff_text.splitlines()[1])
    assert handoff_text is not None and len(handoff_text) > 100
    assert "YOUR NEXT TEST" in handoff_text
    assert "Speak with 5-8 small fashion" in handoff_text

    # 3. Enter wait_for_evidence
    state["phase"] = WorkflowPhase.WAITING_FOR_EVIDENCE
    state["active_experiment"]["status"] = ExperimentStatus.IN_PROGRESS.value

    # 4. Founder asks: "Then what should I do?"
    founder_msg = "Then what should I do?"
    state["founder_response"] = founder_msg
    state["founder_messages"].append({"role": "founder", "content": founder_msg})

    interp_res = interpret_founder_turn(state)
    state.update(interp_res)

    route_interp = route_after_interpret_founder_turn(state)
    print(f"3. route_after_interpret_founder_turn: '{route_interp}'")
    assert route_interp == "classify_waiting_for_evidence_turn"

    # 5. Classify turn
    class_res = classify_waiting_for_evidence_turn(state)
    state.update(class_res)
    assessment = state.get("evidence_turn_assessment", {})
    disp = assessment.get("disposition")
    print(f"4. classify_waiting_for_evidence_turn disposition: '{disp}'")
    assert disp == EvidenceTurnDisposition.QUESTION_ABOUT_EXPERIMENT.value
    assert assessment.get("contains_completed_external_action") is False

    # 6. Route on evidence admission
    route_gate = route_waiting_for_evidence_turn(state)
    print(f"5. route_waiting_for_evidence_turn: '{route_gate}'")
    assert route_gate == "respond_to_waiting_founder"

    # 7. Advisor responds conversationally
    resp_res = respond_to_waiting_founder(state)
    state.update(resp_res)
    waiting_reply = state.get("waiting_response")
    print(f"6. Advisor response (len={len(waiting_reply)}):")
    print(f"   '{waiting_reply[:120]}...'")
    assert waiting_reply is not None and len(waiting_reply) > 20

    # 8. Verify safety invariants
    assert has_valid_completed_evidence_cycle(state) is False
    assert state.get("active_experiment", {}).get("status") == ExperimentStatus.IN_PROGRESS.value
    assert state.get("iteration") == 1
    assert state.get("phase") == WorkflowPhase.WAITING_FOR_EVIDENCE
    assert not state.get("founder_evidence")

    print("PASS: Scenario A correctly explains the experiment, rejects fake evidence, and stays WAITING_FOR_EVIDENCE!")
    return True


def verify_scenario_b_future_intention() -> bool:
    print("\n" + "=" * 60)
    print("SCENARIO B: Future Intention ('Okay, I will speak to them tomorrow.')")
    print("=" * 60)

    state = create_loopcart_state()
    handoff_res = compose_execution_handoff(state)
    state.update(handoff_res)
    state["phase"] = WorkflowPhase.WAITING_FOR_EVIDENCE
    state["active_experiment"]["status"] = ExperimentStatus.IN_PROGRESS.value

    # Founder states future intention
    founder_msg = "Okay, I will speak to them tomorrow."
    state["founder_response"] = founder_msg
    state["founder_messages"].append({"role": "founder", "content": founder_msg})

    interp_res = interpret_founder_turn(state)
    state.update(interp_res)

    class_res = classify_waiting_for_evidence_turn(state)
    state.update(class_res)
    assessment = state.get("evidence_turn_assessment", {})
    disp = assessment.get("disposition")
    print(f"1. classify_waiting_for_evidence_turn disposition: '{disp}'")
    assert disp == EvidenceTurnDisposition.FUTURE_INTENTION.value
    assert assessment.get("contains_completed_external_action") is False

    route_gate = route_waiting_for_evidence_turn(state)
    print(f"2. route_waiting_for_evidence_turn: '{route_gate}'")
    assert route_gate == "respond_to_waiting_founder"

    resp_res = respond_to_waiting_founder(state)
    state.update(resp_res)
    waiting_reply = state.get("waiting_response")
    print(f"3. Advisor response:\n   '{waiting_reply}'")

    # Invariants
    assert has_valid_completed_evidence_cycle(state) is False
    assert state.get("active_experiment", {}).get("status") == ExperimentStatus.IN_PROGRESS.value
    assert state.get("iteration") == 1
    assert state.get("phase") == WorkflowPhase.WAITING_FOR_EVIDENCE

    print("PASS: Scenario B acknowledges intention, remains WAITING_FOR_EVIDENCE, and performs no analysis!")
    return True


def verify_scenario_c_real_results() -> bool:
    print("\n" + "=" * 60)
    print("SCENARIO C: Real Results ('I spoke with 6 small fashion-store owners...')")
    print("=" * 60)

    state = create_loopcart_state()
    handoff_res = compose_execution_handoff(state)
    state.update(handoff_res)
    state["phase"] = WorkflowPhase.WAITING_FOR_EVIDENCE
    state["active_experiment"]["status"] = ExperimentStatus.IN_PROGRESS.value

    founder_msg = (
        "I spoke to 6 small fashion-store owners. Five had at least several sizing-related returns last month. "
        "Three said return shipping and staff handling are painful, two said they just treat returns as normal business, "
        "and one said sizing is not a major issue for them."
    )
    state["founder_response"] = founder_msg
    state["founder_messages"].append({"role": "founder", "content": founder_msg})

    interp_res = interpret_founder_turn(state)
    state.update(interp_res)

    class_res = classify_waiting_for_evidence_turn(state)
    state.update(class_res)
    assessment = state.get("evidence_turn_assessment", {})
    disp = assessment.get("disposition")
    print(f"1. classify_waiting_for_evidence_turn disposition: '{disp}'")
    assert disp == EvidenceTurnDisposition.ACTUAL_EVIDENCE.value
    assert assessment.get("contains_completed_external_action") is True

    route_gate = route_waiting_for_evidence_turn(state)
    print(f"2. route_waiting_for_evidence_turn: '{route_gate}'")
    assert route_gate == "extract_experiment_evidence"

    # Simulate valid evidence analysis
    state["evidence_analysis"] = {
        "conclusion": "SUPPORTED",
        "evidence_strength": "MODERATE",
        "support_findings": ["5 of 6 store owners confirmed recurring sizing-related returns."],
        "contradiction_findings": ["Two treat it as ordinary cost of doing business."],
        "inconclusive_findings": [],
        "reasoning_summary": "5/6 positive responses meets directional threshold for problem validity.",
        "recommended_state_updates": {"Problem": "Verified sizing return friction in small fashion stores."},
    }

    # Verify canonical cycle validator
    assert has_valid_completed_evidence_cycle(state) is True

    # Validate analysis
    val_res = validate_evidence_analysis(state)
    state.update(val_res)
    assert state["active_assumption"]["status"] == AssumptionStatus.SUPPORTED.value
    assert state["active_experiment"]["status"] == ExperimentStatus.COMPLETED.value
    assert len(state["founder_evidence"]) == 1
    print(f"3. validate_evidence_analysis: assumption={state['active_assumption']['status']}, experiment={state['active_experiment']['status']}")

    # Apply updates & increment
    st_res = apply_validated_state_updates(state)
    state.update(st_res)

    iter_res = increment_iteration(state)
    state.update(iter_res)
    print(f"4. increment_iteration: new iteration={state.get('iteration')}, phase={state.get('phase')}")
    assert state.get("iteration") == 2
    assert state.get("phase") == WorkflowPhase.LEARN

    # LearningDelta eligibility
    eligible = can_compose_learning_delta(state)
    print(f"5. can_compose_learning_delta: {eligible}")
    assert eligible is True

    print("PASS: Scenario C admits real evidence, completes experiment, updates state, and advances iteration!")
    return True


if __name__ == "__main__":
    ok_a = verify_scenario_a_guidance_question()
    ok_b = verify_scenario_b_future_intention()
    ok_c = verify_scenario_c_real_results()

    if ok_a and ok_b and ok_c:
        print("\n" + "=" * 60)
        print("ALL M4.0.3 SCENARIOS VERIFIED SUCCESSFULLY!")
        print("=" * 60)
    else:
        print("\nFAIL: One or more scenarios did not pass verification.")
        sys.exit(1)
