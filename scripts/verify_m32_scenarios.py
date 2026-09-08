"""
Verification script for Milestone 3.2: Founder Intent + Conversational Awareness.

Verifies the 3 required manual scenarios:
  Scenario A: Shiftly concern scenario ("what is your honest reaction?")
  Scenario B: Correction scenario ("I'm not the café manager; my friend is")
  Scenario C: One-word 'yes' state extraction scenario (no speculative leakage)
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from startary.domain.enums import WorkflowPhase
from startary.domain.execution_context import ExecutionContext
from startary.domain.startup_progress import StartupProgress
from startary.graph.builder import run_advisor_response, run_founder_turn_interpretation
from startary.graph.graph_state import StartaryGraphState
from startary.graph.nodes import extract_startup_state


def test_scenario_a_shiftly_concern():
    print("\n" + "=" * 70)
    print("SCENARIO A: SHIFTLY CONCERN & DIRECT QUESTION SCENARIO")
    print("=" * 70)

    state: StartaryGraphState = {
        "project_id": "test-shiftly-m32-a",
        "phase": WorkflowPhase.UNDERSTAND,
        "iteration": 1,
        "raw_context": "Shiftly helps independent café and restaurant managers manage shift swaps and last-minute cancellations.",
        "weak_area": "Problem",
        "selection_confidence": "LOW",
        "pending_question": "Have you spoken with other café managers besides your friend?",
        "active_interrupt_type": "baseline",
        "current_baseline_question": {
            "question": "Roughly how much personal budget can you risk on Shiftly right now?",
            "target_group": "RESOURCES",
            "information_goal": "Assess available testing budget",
        },
        "reality_check": {
            "summary": "One café manager confirmed cancellations create weekly stress, but this is a single contact and does not prove widespread demand.",
        },
        "diagnosis": {
            "summary": "Early build / idea stage with one informal problem signal and high strategic uncertainty.",
        },
        "founder_messages": [
            {"role": "founder", "content": "I'm building Shiftly to help café managers manage shift changes."},
            {"role": "startary", "content": "Have you spoken with other café managers besides your friend?"},
            {
                "role": "founder",
                "content": "No I haven't, my fear is whether this problem is worth my time and energy, what is your honest reaction?",
            },
        ],
        "execution_context": ExecutionContext().model_dump(),
        "startup_progress": StartupProgress().model_dump(),
        "traces": [],
    }

    # Step 1: Interpret Turn
    state_after_interp = run_founder_turn_interpretation(state)
    interp = state_after_interp.get("latest_founder_turn_interpretation", {})
    print("\n--- Founder Turn Interpretation ---")
    print(f"Intents:           {interp.get('intent_types')}")
    print(f"Direct Question:   {interp.get('direct_question')}")
    print(f"Concern/Fear:      {interp.get('concern_or_goal')}")
    print(f"Response Priority: {interp.get('response_priority')}")
    print(f"Summary:           {interp.get('interpretation_summary')}")

    # Step 2: Compose Advisor Response
    state_after_resp = run_advisor_response(state_after_interp)
    advisor_resp = state_after_resp.get("latest_advisor_response", {})
    print("\n--- Advisor Response ---")
    print(f"Response Text:\n{advisor_resp.get('response_text')}\n")
    print(f"Should Ask Next Question: {advisor_resp.get('should_ask_next_question')}")
    return state_after_resp


def test_scenario_b_correction():
    print("\n" + "=" * 70)
    print("SCENARIO B: CORRECTION SCENARIO")
    print("=" * 70)

    state: StartaryGraphState = {
        "project_id": "test-shiftly-m32-b",
        "phase": WorkflowPhase.UNDERSTAND,
        "iteration": 1,
        "raw_context": "Shiftly helps café managers manage shift cancellations.",
        "weak_area": "Problem",
        "selection_confidence": "LOW",
        "pending_question": "When you manage shift changes at the café, what usually breaks first?",
        "active_interrupt_type": "strategic",
        "current_clarifying_question": {
            "question": "From what your friend described, what part of their current process fails when someone cancels?",
            "target_area": "Problem",
            "information_goal": "Understand failure point from secondary observations",
        },
        "founder_messages": [
            {"role": "founder", "content": "Shiftly helps café managers with shift changes."},
            {"role": "startary", "content": "When you manage shift changes at the café, what usually breaks first?"},
            {
                "role": "founder",
                "content": "I'm not the café manager; my friend is.",
            },
        ],
        "execution_context": ExecutionContext().model_dump(),
        "startup_progress": StartupProgress().model_dump(),
        "traces": [],
    }

    state_after_interp = run_founder_turn_interpretation(state)
    interp = state_after_interp.get("latest_founder_turn_interpretation", {})
    print("\n--- Founder Turn Interpretation ---")
    print(f"Intents:           {interp.get('intent_types')}")
    print(f"Correction:        {interp.get('correction')}")
    print(f"Response Priority: {interp.get('response_priority')}")
    print(f"Summary:           {interp.get('interpretation_summary')}")

    state_after_resp = run_advisor_response(state_after_interp)
    advisor_resp = state_after_resp.get("latest_advisor_response", {})
    print("\n--- Advisor Response ---")
    print(f"Response Text:\n{advisor_resp.get('response_text')}\n")
    return state_after_resp


def test_scenario_c_one_word_yes():
    print("\n" + "=" * 70)
    print("SCENARIO C: ONE-WORD 'YES' STATE AUDIT SCENARIO")
    print("=" * 70)

    state: StartaryGraphState = {
        "project_id": "test-shiftly-m32-c",
        "phase": WorkflowPhase.UNDERSTAND,
        "iteration": 1,
        "pending_question": "Do you have direct personal access to local café owners who can test with you?",
        "founder_messages": [
            {"role": "founder", "content": "We are two co-founders building Shiftly."},
            {"role": "startary", "content": "Do you have direct personal access to local café owners who can test with you?"},
            {"role": "founder", "content": "yes"},
        ],
        "execution_context": ExecutionContext().model_dump(),
        "startup_progress": StartupProgress().model_dump(),
        "traces": [],
    }

    # Extract state with short-answer guard
    res = extract_startup_state(state)
    delta = res.get("startup_state_delta", {})

    print("\n--- Extracted Delta Keys from 'yes' ---")
    populated_keys = [k for k, v in delta.items() if v is not None and k != "_source_turn"]
    print(f"Populated Keys: {populated_keys}")
    for k in populated_keys:
        val = delta[k]
        print(f"  - {k}: {val}")

    # Check for speculative inference leakage
    leaked_keys = [k for k in ["revenue", "launched", "active_users", "has_design_capability", "has_marketing_capability"] if delta.get(k) is not None]
    print(f"\nSpeculative Leaked Keys (must be empty): {leaked_keys}")
    assert len(leaked_keys) == 0, f"Leaked speculative inferences found: {leaked_keys}"
    print("SUCCESS: Zero speculative inference leakage on short answer 'yes'!")


if __name__ == "__main__":
    test_scenario_a_shiftly_concern()
    test_scenario_b_correction()
    test_scenario_c_one_word_yes()

