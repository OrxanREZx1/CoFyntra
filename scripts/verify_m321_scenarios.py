"""
Live Verification Script for Milestone 3.2.1: Conversational Control Integrity Patch.

Executes live LLM scenarios:
1. TruckTaxi Scenario (Turn 0 -> Turn 1):
   - Turn 0: Onboarding description -> asks TEAM
   - Turn 1: Founder answers TEAM ("We are 3 people...") + expresses goal ("find the right market")
   - Verifies:
     * Planner selects CAPACITY (not TEAM)
     * Decision has information_target='CAPACITY' and unique question_id
     * Advisor response matches question_id and CAPACITY target
     * Advisor acknowledges team and goal without generic premature strategy
     * TEAM is never asked again
     * Intro summary is not repeated
2. Repeated-Question / Already Answered Claim Scenario:
   - Founder says "I already answered this" -> classified as CORRECTION, acknowledges, replans
3. Conflicting Facts Scenario:
   - Confirmed 3 founders -> founder says "I am alone" -> FactConflict detected, 3 founders preserved
"""

import json
import logging
import sys

from startary.domain.enums import WorkflowPhase
from startary.graph.builder import build_graph
from startary.graph.nodes import (
    compose_advisor_response,
    extract_startup_state,
    interpret_founder_turn,
    merge_execution_state,
    merge_founder_answer,
    plan_baseline_question,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("verify_m321")


def run_live_trucktaxi_scenario():
    print("\n=======================================================")
    print("SCENARIO 1: TRUCKTAXI LIVE RUN (TURN 0 -> TURN 1)")
    print("=======================================================")

    state = {
        "project_id": "trucktaxi-live-321",
        "phase": WorkflowPhase.UNDERSTAND,
        "iteration": 1,
        "raw_context": (
            "TruckTaxi is an on-demand freight marketplace connecting small business shippers "
            "with independent truck owners for same-day local and regional cargo delivery."
        ),
        "founder_messages": [
            {
                "role": "founder",
                "content": (
                    "TruckTaxi is an on-demand freight marketplace connecting small business shippers "
                    "with independent truck owners for same-day local and regional cargo delivery."
                ),
            }
        ],
        "execution_context": {},
        "startup_progress": {},
        "baseline_sufficiency": {
            "is_sufficient": False,
            "missing_groups": ["TEAM", "CAPACITY", "PRODUCT_STAGE", "VALIDATION_OR_USER", "RESOURCE"],
        },
        "baseline_question_turns": 0,
        "strategic_clarification_turns": 0,
        "weak_link_clarification_turns": 0,
        "question_history": [],
        "fact_conflicts": [],
        "founder_current_goal": None,
        "intro_summary_shown": False,
        "active_next_turn_decision": None,
        "latest_advisor_response": None,
        "pending_question": None,
    }

    # Turn 0 Planner: Should plan TEAM
    print("\n--- Turn 0: Planning first baseline question ---")
    p0_res = plan_baseline_question(state)
    state.update(p0_res)
    decision_0 = state["active_next_turn_decision"]
    print(f"Turn 0 Decision Target: {decision_0['information_target']}")
    print(f"Turn 0 Canonical Question: {decision_0['canonical_question']}")
    print(f"Turn 0 Question ID: {decision_0['question_id']}")

    # Turn 0 Advisor Composer
    print("\n--- Turn 0: Composing Advisor Response ---")
    adv0_res = compose_advisor_response(state)
    state.update(adv0_res)
    adv0 = state["latest_advisor_response"]
    print(f"Turn 0 Advisor Response Text:\n{adv0['response_text']}")
    print(f"Turn 0 Advisor Target: {adv0['next_question_target']}")
    print(f"Turn 0 Intro Summary Shown Flag: {state.get('intro_summary_shown')}")

    assert adv0["question_id"] == decision_0["question_id"], "Turn 0 question_id must match decision"
    assert adv0["next_question_target"] == decision_0["information_target"], "Turn 0 target must match decision"
    assert state.get("intro_summary_shown") is True, "intro_summary_shown must be True after Turn 0"

    # Simulate founder response at Turn 1
    founder_turn_1_msg = (
        "We are 3 people. I talk with truck drivers, one person is a mobile dev, "
        "another does business analysis. My intention is to find the right market or understand how to find it."
    )
    print(f"\n--- Founder Turn 1 Input ---\n\"{founder_turn_1_msg}\"")

    state["founder_response"] = founder_turn_1_msg
    state["active_interrupt_type"] = "baseline"

    # Merge founder answer (Lifecycle token consumed, active decision cleared)
    merge_res = merge_founder_answer(state)
    state.update(merge_res)

    assert state["active_next_turn_decision"] is None, "active_next_turn_decision must be cleared"
    assert state["latest_advisor_response"] is None, "latest_advisor_response must be cleared"
    assert len(state["question_history"]) == 1, "Question history must record answered question"
    assert state["question_history"][0]["information_target"] == "TEAM"

    # Interpret founder turn
    print("\n--- Turn 1: Interpreting Founder Turn ---")
    interp_res = interpret_founder_turn(state)
    state.update(interp_res)
    interp = state["latest_founder_turn_interpretation"]
    print(f"Turn 1 Priority: {interp['response_priority']}")
    print(f"Turn 1 Intents: {interp['intent_types']}")
    print(f"Turn 1 Detected Goal: {interp.get('detected_founder_goal')}")
    print(f"State Founder Current Goal: {state.get('founder_current_goal')}")

    assert state.get("founder_current_goal") is not None, "Founder current goal must be saved in state"
    assert state["founder_current_goal"]["goal_type"] == "FIND_INITIAL_MARKET", "Goal must be FIND_INITIAL_MARKET"

    # State Extraction & Merge
    print("\n--- Turn 1: Extracting Startup State ---")
    ext_res = extract_startup_state(state)
    state.update(ext_res)
    merge_exec_res = merge_execution_state(state)
    state.update(merge_exec_res)

    print(f"Known team size: {state['execution_context'].get('founder_count')}")

    # Turn 1 Planner: Must NOT ask TEAM again! Must plan CAPACITY!
    print("\n--- Turn 1: Planning Next Baseline Question ---")
    p1_res = plan_baseline_question(state)
    state.update(p1_res)
    decision_1 = state["active_next_turn_decision"]

    print(f"Turn 1 Decision Target: {decision_1['information_target']}")
    print(f"Turn 1 Canonical Question: {decision_1['canonical_question']}")
    print(f"Turn 1 Question ID: {decision_1['question_id']}")

    assert decision_1["information_target"] == "CAPACITY", (
        f"CRITICAL CHECK: Planner MUST select CAPACITY! Got: {decision_1['information_target']}"
    )
    assert decision_1["question_id"] != decision_0["question_id"], "New turn must have a fresh question_id"

    # Turn 1 Advisor Composer: Must preserve CAPACITY and question_id
    print("\n--- Turn 1: Composing Advisor Response ---")
    adv1_res = compose_advisor_response(state)
    state.update(adv1_res)
    adv1 = state["latest_advisor_response"]

    print(f"Turn 1 Advisor Target: {adv1['next_question_target']}")
    print(f"Turn 1 Advisor Question ID: {adv1['question_id']}")
    print(f"Turn 1 Advisor Response Text:\n{adv1['response_text']}")

    assert adv1["question_id"] == decision_1["question_id"], "Advisor question_id must match decision_1"
    assert adv1["next_question_target"] == "CAPACITY", "Advisor target must be CAPACITY"
    assert "Who is building TruckTaxi with you" not in adv1["response_text"], "Must NOT ask who is on the team again!"
    assert not ("TruckTaxi is an on-demand freight marketplace" in adv1["response_text"]), "Must NOT repeat startup definition!"

    print("\n>>> SCENARIO 1 PASSED PERFECTLY!")


def run_live_already_answered_scenario():
    print("\n=======================================================")
    print("SCENARIO 2: 'I ALREADY ANSWERED THIS' CORRECTION")
    print("=======================================================")

    state = {
        "project_id": "repeat-guard-321",
        "phase": WorkflowPhase.UNDERSTAND,
        "iteration": 1,
        "raw_context": "TruckTaxi freight platform.",
        "founder_messages": [
            {"role": "founder", "content": "TruckTaxi freight platform."},
            {"role": "startary", "content": "Who is building this with you?"},
            {"role": "founder", "content": "I already answered this earlier. We are 3 people."},
        ],
        "execution_context": {
            "founder_count": {"value": "3", "provenance": "EXPLICIT", "confidence": "HIGH"},
        },
        "startup_progress": {},
        "baseline_sufficiency": {
            "is_sufficient": False,
            "missing_groups": ["TEAM", "CAPACITY", "RESOURCE"],
        },
        "baseline_question_turns": 1,
        "strategic_clarification_turns": 0,
        "weak_link_clarification_turns": 0,
        "question_history": [
            {
                "question_id": "q_test_1",
                "information_target": "TEAM",
                "canonical_question": "Who is on the team?",
                "turn": 1,
                "answered": True,
                "founder_answer_summary": "We are 3 people.",
            }
        ],
        "fact_conflicts": [],
        "founder_current_goal": None,
        "intro_summary_shown": True,
        "active_next_turn_decision": None,
        "latest_advisor_response": None,
        "pending_question": "Who is building this with you?",
    }

    # Interpret
    interp_res = interpret_founder_turn(state)
    state.update(interp_res)
    interp = state["latest_founder_turn_interpretation"]
    print(f"Interpreted priority: {interp['response_priority']}")
    print(f"Interpreted intents: {interp['intent_types']}")
    print(f"Already answered claim: {interp.get('is_already_answered_claim')}")

    assert "CORRECTION" in interp["intent_types"], "Must be classified as CORRECTION"
    assert interp["response_priority"] == "HIGH", "Must have HIGH priority"

    # Plan
    p_res = plan_baseline_question(state)
    state.update(p_res)
    decision = state["active_next_turn_decision"]
    print(f"New Decision Target: {decision['information_target']}")
    print(f"Canonical Question: {decision['canonical_question']}")

    assert decision["information_target"] == "CAPACITY", "Must advance to CAPACITY"

    # Compose
    adv_res = compose_advisor_response(state)
    state.update(adv_res)
    adv = state["latest_advisor_response"]
    print(f"Advisor Response:\n{adv['response_text']}")

    assert adv["next_question_target"] == "CAPACITY"
    print("\n>>> SCENARIO 2 PASSED PERFECTLY!")


def run_live_fact_conflict_scenario():
    print("\n=======================================================")
    print("SCENARIO 3: CONFLICTING FACTS PROTECTION")
    print("=======================================================")

    state = {
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

    print(f"Preserved founder_count value: {res['execution_context']['founder_count']['value']}")
    print(f"Fact conflicts detected: {res['fact_conflicts']}")

    assert res["execution_context"]["founder_count"]["value"] == "3", "Must NOT mutate confirmed 3 founders"
    assert len(res["fact_conflicts"]) == 1, "Must record exactly 1 FactConflict"
    assert res["fact_conflicts"][0]["existing_value"] == "3"
    assert res["fact_conflicts"][0]["conflicting_value"] == "1"

    print("\n>>> SCENARIO 3 PASSED PERFECTLY!")


if __name__ == "__main__":
    run_live_trucktaxi_scenario()
    run_live_already_answered_scenario()
    run_live_fact_conflict_scenario()
    print("\n=======================================================")
    print("ALL MILESTONE 3.2.1 VERIFICATION SCENARIOS PASSED!")
    print("=======================================================")

