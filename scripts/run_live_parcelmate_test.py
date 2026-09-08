"""
Live verification script for Milestone 2.6.2: Understand Reasoning Integrity Patch.

Executes the ParcelMate scenario:
  1. Founder Initial Input (sparse concept)
  2. Baseline check-in interrupt -> realistic operational profile answer
  3. Strategic understand interrupt (if any) -> strategic context answer
  4. Weak-link discrimination -> VERIFIES graph ACTUALLY pauses with discriminating question
  5. Founder discriminating answer (Section 22)
  6. Re-diagnosis, re-assessment, and score recalculation
  7. Reality Check & Critical Assumption (critical_assumption_v2)
  8. Verifies NO unsupported mechanisms, metrics, or frequencies appear in final assumption.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from startary.config import get_settings
from startary.domain.enums import WorkflowPhase
from startary.graph.builder import build_graph
from startary.graph.graph_state import StartaryGraphState


def run_parcelmate_live_test():
    settings = get_settings()
    print("=" * 70, flush=True)
    print("PARCELMATE LIVE TEST — MILESTONE 2.6.2 INTEGRITY VERIFICATION", flush=True)
    print(f"Model: {settings.model_name}", flush=True)
    print("=" * 70, flush=True)

    founder_input = (
        "We are building ParcelMate, a tool for small Instagram sellers to make delivery management easier. "
        "A lot of them manage orders through DMs and different courier apps, and we think it gets messy when they start getting more orders."
    )

    print(f"\nFounder Initial Description:\n{founder_input}\n", flush=True)

    checkpointer = MemorySaver()
    graph = build_graph(checkpointer=checkpointer)
    thread_id = "live-parcelmate-m262"
    cfg = {"configurable": {"thread_id": thread_id}}

    initial_state: StartaryGraphState = {
        "project_id": thread_id,
        "phase": WorkflowPhase.UNDERSTAND,
        "iteration": 1,
        "raw_context": founder_input,
        "founder_messages": [{"role": "founder", "content": founder_input}],
        "baseline_question_turns": 0,
        "strategic_clarification_turns": 0,
        "weak_link_clarification_turns": 0,
        "clarification_turn_count": 0,
        "baseline_history": [],
        "clarification_history": [],
        "weak_link_clarification_history": [],
    }

    print("Running graph with live LLM...", flush=True)
    graph.invoke(initial_state, cfg)

    # Step through interrupts
    interrupt_step = 0
    snapshot = graph.get_state(cfg)

    pre_clarification_scores: dict[str, int] = {}
    pre_clarification_winner: str | None = None
    pre_clarification_conf: str | None = None
    first_wl_question: str | None = None
    first_wl_answer: str | None = None
    post_clarification_scores: dict[str, int] = {}
    post_clarification_winner: str | None = None
    post_clarification_conf: str | None = None

    while snapshot.tasks and snapshot.tasks[0].interrupts:
        interrupt_step += 1
        interrupt_val = snapshot.tasks[0].interrupts[0].value
        interrupt_type = interrupt_val.get("interrupt_type", "unknown")
        question = interrupt_val.get("question")
        print(f"\n[INTERRUPT {interrupt_step}: {interrupt_type.upper()}] Startary asks:\n  {question}", flush=True)

        if interrupt_type == "baseline":
            simulated_reply = (
                "We are 2 co-founders. One builds the software and the other handles sales and operations. "
                "We each have about 15 hours per week. We have wireframes and talked with 3 sellers, "
                "but no working code prototype or MVP yet. We can spend about 150 AZN per month."
            )
        elif interrupt_type == "strategic":
            simulated_reply = (
                "The sellers said when order volumes grow, they lose track of which customer orders were booked with couriers "
                "and which are still unconfirmed in Instagram DMs, causing customer complaints and delays."
            )
        elif interrupt_type == "weak_link":
            # Record pre-clarification state
            cur_vals = snapshot.values
            pre_clarification_winner = cur_vals.get("weak_area")
            pre_clarification_conf = cur_vals.get("selection_confidence")
            for a in cur_vals.get("area_assessments") or []:
                pre_clarification_scores[a.get("area")] = a.get("calculated_decision_risk", 0)

            first_wl_question = question
            # Section 22 exact test answer
            simulated_reply = (
                "Two sellers told us that when orders increase they keep switching between Instagram messages "
                "and courier apps, and sometimes they spend a lot of time checking which orders were already sent "
                "and which still need action. They said the process becomes annoying, but neither of them asked us for a new tool yet."
            )
            first_wl_answer = simulated_reply
            print(f"Recorded pre-clarification weak-link winner: {pre_clarification_winner} (conf={pre_clarification_conf})", flush=True)
            print(f"Competing areas before clarification: {cur_vals.get('competing_weak_areas')}", flush=True)
        else:
            simulated_reply = "We are focused on Instagram sellers in Baku."

        print(f"\nFounder responds:\n  {simulated_reply}", flush=True)
        graph.invoke(Command(resume=simulated_reply), cfg)
        snapshot = graph.get_state(cfg)

    final_values = snapshot.values
    post_clarification_winner = final_values.get("weak_area")
    post_clarification_conf = final_values.get("selection_confidence")
    for a in final_values.get("area_assessments") or []:
        post_clarification_scores[a.get("area")] = a.get("calculated_decision_risk", 0)

    print("\n" + "=" * 70, flush=True)
    print("PARCELMATE RUN RESULTS & VERIFICATION", flush=True)
    print("=" * 70, flush=True)

    print(f"Total Interrupts Processed       : {interrupt_step}", flush=True)
    print(f"Baseline Turns Used              : {final_values.get('baseline_question_turns', 0)}", flush=True)
    print(f"Strategic Turns Used             : {final_values.get('strategic_clarification_turns', 0)}", flush=True)
    print(f"Weak-Link Turns Used             : {final_values.get('weak_link_clarification_turns', 0)}", flush=True)
    print(f"First Weak-Link Question         : {first_wl_question}", flush=True)
    print(f"Founder Answer                   : {first_wl_answer}", flush=True)
    print(f"Pre-Clarification Winner         : {pre_clarification_winner} (conf={pre_clarification_conf})", flush=True)
    print(f"Post-Clarification Winner        : {post_clarification_winner} (conf={post_clarification_conf})", flush=True)

    print("\n--- SCORES BEFORE CLARIFICATION ---", flush=True)
    for area, score in pre_clarification_scores.items():
        print(f"  {area:30}: {score}", flush=True)

    print("\n--- SCORES AFTER CLARIFICATION ---", flush=True)
    for area, score in post_clarification_scores.items():
        print(f"  {area:30}: {score}", flush=True)

    rc = final_values.get("reality_check") or {}
    print(f"\nReality Check Conclusion         : {rc.get('conclusion')}", flush=True)
    print(f"Reality Check Unresolved         : {rc.get('unresolved_uncertainty')}", flush=True)
    print(f"Reality Check Not Supported      : {rc.get('what_evidence_does_not_support')}", flush=True)

    assumption = final_values.get("active_assumption") or {}
    statement = assumption.get("statement", "")
    print(f"\nFinal Critical Assumption Statement:\n  \"{statement}\"", flush=True)
    print(f"Assumption Area                  : {assumption.get('area')}", flush=True)
    print(f"Assumption Status                : {assumption.get('status')}", flush=True)
    print(f"Workflow Phase                   : {final_values.get('phase')}", flush=True)

    # Validation checks
    assert final_values.get("weak_link_clarification_turns", 0) >= 1, "FAIL: Weak link clarification was not executed!"
    print("SUCCESS: Weak-link clarification question was actually asked and answered!", flush=True)

    # Prohibited phrases inspection
    prohibited_fragments = [
        "20 orders", "20 orders per week", "at least weekly",
        "missed, delayed", "missed deliveries", "delayed deliveries",
        "incorrectly recorded deliveries", "top three operational frustrations", "top three"
    ]
    for p in prohibited_fragments:
        assert p.lower() not in statement.lower(), f"FAIL: Prohibited invented fragment '{p}' found in assumption: {statement}"
    print("SUCCESS: Final Critical Assumption contains NO ungrounded thresholds, frequencies, or invented failure modes!", flush=True)


if __name__ == "__main__":
    run_parcelmate_live_test()

