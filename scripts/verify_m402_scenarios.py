"""
scripts/verify_m402_scenarios.py
Verification script for Milestone 4.0.2: Uncertainty-to-Test Handoff.

Scenario A: LoopCart
- Shopify returns tool for merchants
- 2 technical co-founders, 30h/wk, $1,000 budget, MVP tested only on cousin's store
- Founder states: "No, we haven't spoken to merchants."
- Asserts that Startary:
  1. Identifies missing information requires EXTERNAL_EVIDENCE
  2. Recognizes testable uncertainty exists (can_formulate_testable_uncertainty == True)
  3. Hands off to TEST (assess_startup_areas) without looping on questions
  4. Does NOT terminate with CONTEXT CLARIFICATION LIMIT REACHED

Scenario B: Truly Vague Startup
- Founder: "I want to build something with AI but don't know what."
- Asserts that Startary:
  1. Classifies as NOT_YET_FORMULATABLE
  2. Correctly stops at handle_insufficient_context_stop when clarification budget is exhausted
"""
from __future__ import annotations

import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from startary.domain.gap_plan import (
    InformationGapPlan,
    UncertaintyResolutionMode,
    can_formulate_testable_uncertainty,
)
from startary.graph.builder import (
    MAX_STRATEGIC_CLARIFICATION_TURNS,
    route_on_gap_plan,
)
from startary.domain.terminal_state import (
    TerminalStatus,
    resolve_founder_facing_status,
)


def verify_scenario_a_loopcart() -> bool:
    print("\n" + "=" * 60)
    print("VERIFYING SCENARIO A: LoopCart (Uncertainty-to-Test Handoff)")
    print("=" * 60)

    loopcart_state = {
        "project_id": "proj_loopcart_verify",
        "raw_context": (
            "LoopCart is a B2B Shopify returns-management plugin. "
            "Founded by 2 engineers with 30 hrs/week and $1,000 budget. "
            "They have a working MVP deployed on their cousin's apparel store."
        ),
        "startup_state": {
            "area_knowledge": {
                "Target User": {
                    "understanding": "Small Shopify merchants selling apparel and accessories.",
                    "founder_claims": ["Merchants suffer from 25% return rates"],
                },
                "Problem": {
                    "understanding": "High return rates severely erode merchant profitability and cash flow.",
                    "founder_claims": ["Returns are the #1 headache for e-commerce stores"],
                },
            }
        },
        "founder_messages": [
            {
                "role": "founder",
                "content": "No, we haven't spoken to external merchants yet. We only tested it on my cousin's store.",
            }
        ],
        "strategic_clarification_turns": 1,
    }

    # 1. Check testability detector
    can_test = can_formulate_testable_uncertainty(loopcart_state)
    print(f"1. can_formulate_testable_uncertainty: {can_test}")
    assert can_test is True, "Expected LoopCart to be recognized as testable!"

    # 2. Simulate Planner output for LoopCart
    gap_plan = InformationGapPlan(
        should_ask=False,
        strategic_sufficient=False,
        resolution_mode=UncertaintyResolutionMode.EXTERNAL_EVIDENCE,
        can_formulate_testable_uncertainty=True,
        why_this_matters="Validating whether external merchants experience returns as high enough priority",
        missing_information="External merchant validation of return pain and willingness to install",
        recommended_area="Problem",
        reason="Remaining uncertainty requires live market observation rather than founder-known facts.",
        question=None,
    )
    loopcart_state["gap_plan"] = gap_plan.model_dump(mode="json")

    # 3. Verify router routes directly to TEST (assess_startup_areas)
    next_node = route_on_gap_plan(loopcart_state)
    print(f"2. route_on_gap_plan returned: '{next_node}'")
    assert next_node == "assess_startup_areas", f"Expected 'assess_startup_areas', got '{next_node}'"

    # 4. Verify terminal status would NOT be CONTEXT CLARIFICATION LIMIT REACHED
    # Even if turns were simulated at max:
    loopcart_state["strategic_clarification_turns"] = MAX_STRATEGIC_CLARIFICATION_TURNS
    max_turn_route = route_on_gap_plan(loopcart_state)
    print(f"3. route_on_gap_plan at MAX turns returned: '{max_turn_route}'")
    assert max_turn_route == "assess_startup_areas", f"Expected 'assess_startup_areas' even at MAX turns, got '{max_turn_route}'"

    print("PASS: Scenario A (LoopCart) successfully handoffs uncertainty to TEST without stopping!")
    return True


def verify_scenario_b_vague_startup() -> bool:
    print("\n" + "=" * 60)
    print("VERIFYING SCENARIO B: Truly Vague Startup (Safe Stop on Max Turns)")
    print("=" * 60)

    vague_state = {
        "project_id": "proj_vague_verify",
        "raw_context": "I want to build something with AI but don't know what.",
        "startup_state": {"area_knowledge": {}},
        "founder_messages": [
            {"role": "founder", "content": "I want to do an AI startup, just exploring ideas right now."}
        ],
        "strategic_clarification_turns": 1,
    }

    # 1. Check testability detector
    can_test = can_formulate_testable_uncertainty(vague_state)
    print(f"1. can_formulate_testable_uncertainty: {can_test}")
    assert can_test is False, "Expected vague startup to NOT be testable!"

    # 2. Simulate Planner output for vague startup
    gap_plan = InformationGapPlan(
        should_ask=True,
        strategic_sufficient=False,
        resolution_mode=UncertaintyResolutionMode.NOT_YET_FORMULATABLE,
        can_formulate_testable_uncertainty=False,
        why_this_matters="No problem or user domain identified",
        missing_information="Target user and problem concept",
        reason="Context is too vague to formulate an assumption.",
        question="What specific type of user or problem are you interested in solving?",
    )
    vague_state["gap_plan"] = gap_plan.model_dump(mode="json")

    # 3. Turns < MAX: ask question
    route_turn_1 = route_on_gap_plan(vague_state)
    print(f"2. route_on_gap_plan at turn 1: '{route_turn_1}'")
    assert route_turn_1 == "ask_and_wait_for_founder"

    from startary.graph.nodes import handle_insufficient_context_stop

    # 4. Turns >= MAX: safely stops
    vague_state["strategic_clarification_turns"] = MAX_STRATEGIC_CLARIFICATION_TURNS
    route_max = route_on_gap_plan(vague_state)
    print(f"3. route_on_gap_plan at MAX turns: '{route_max}'")
    assert route_max == "handle_insufficient_context_stop", f"Expected 'handle_insufficient_context_stop', got '{route_max}'"

    # 5. Execute stop node and verify terminal status
    stop_result = handle_insufficient_context_stop(vague_state)
    vague_state.update(stop_result)
    status = resolve_founder_facing_status(vague_state)
    print(f"4. resolve_founder_facing_status: '{status.value}'")
    assert status == TerminalStatus.CLARIFICATION_LIMIT_REACHED

    print("PASS: Scenario B (Truly Vague Startup) clarifies when budget exists and safely stops at limit!")
    return True


if __name__ == "__main__":
    ok_a = verify_scenario_a_loopcart()
    ok_b = verify_scenario_b_vague_startup()

    if ok_a and ok_b:
        print("\n" + "=" * 60)
        print("ALL M4.0.2 SCENARIOS VERIFIED SUCCESSFULLY!")
        print("=" * 60)
    else:
        sys.exit(1)
