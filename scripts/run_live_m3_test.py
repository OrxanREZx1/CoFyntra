"""
Live verification script for Milestone 3: Experiment Generation + Experiment Critic.

Executes the full pipeline for ParcelMate with live OpenAI model:
  1. Founder Initial Input (sparse concept)
  2. Baseline check-in interrupt -> operational answer
  3. Strategic understand interrupt (if any) -> strategic context answer
  4. Weak-link discrimination -> discriminating question -> founder answer
  5. Reality Check & Grounded Critical Assumption
  6. Milestone 3:
     - generate_experiment (experiment_generator_v1)
     - validate_proposed_experiment (deterministic validator)
     - critique_experiment (experiment_critic_v1)
     - save_final_experiment (ExperimentStatus.READY)
     - assess_can_execute (CAN_EXECUTE / NEEDS_HELP / BLOCKED)
"""

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
from startary.cli import _print_experiment, _print_critical_assumption, _print_startup_profile


def run_m3_live_test():
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    settings = get_settings()
    print("=" * 70, flush=True)
    print("PARCELMATE LIVE TEST — MILESTONE 3 VERIFICATION", flush=True)
    print(f"Model: {settings.model_name}", flush=True)
    print("=" * 70, flush=True)

    founder_input = (
        "We are building ParcelMate, a tool for small Instagram sellers to make delivery management easier. "
        "A lot of them manage orders through DMs and different courier apps, and we think it gets messy when they start getting more orders."
    )

    print(f"\nFounder Initial Description:\n{founder_input}\n", flush=True)

    checkpointer = MemorySaver()
    graph = build_graph(checkpointer=checkpointer)
    thread_id = "live-parcelmate-m3"
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
        "critic_revision_count": 0,
        "previous_experiments": [],
    }

    graph.invoke(initial_state, cfg)

    # Human-in-the-loop responses for baseline & weak link
    while True:
        snapshot = graph.get_state(cfg)
        if not snapshot.tasks:
            break
        task = snapshot.tasks[0]
        if not task.interrupts:
            break

        interrupt_data = task.interrupts[0].value
        interrupt_type = interrupt_data.get("interrupt_type") or interrupt_data.get("type", "unknown")
        question = interrupt_data.get("question", "")

        print(f"\n[INTERRUPT: {interrupt_type.upper()}]")
        print(f"Startary asks: \"{question}\"")

        if interrupt_type == "baseline":
            answer = (
                "We are two co-founders — one technical developer and one handling sales and operations. "
                "We each have about 15-20 hours per week available. We have basic wireframes and initial customer contacts, "
                "but no working software prototype or MVP yet. We have a small self-funded budget of around 300 AZN (about $175). "
                "We have direct personal relationships with about 5-6 Instagram boutique owners in Baku."
            )
        elif interrupt_type == "strategic":
            answer = (
                "The sellers said when order volumes grow, they lose track of which customer orders were booked with couriers "
                "and which are still unconfirmed in Instagram DMs, causing customer complaints and delays."
            )
        elif interrupt_type == "weak_link":
            answer = (
                "When talking informally with 2 boutique sellers, both said they often forget to update customers on tracking numbers "
                "because they copy-paste manually between Instagram and Telegram delivery channels. They said it takes 30-45 minutes daily. "
                "They haven't paid for any tools yet, but they complain that it is their most tedious daily chore."
            )
        else:
            answer = (
                "We are two founders working 15-20 hours weekly, with 5 direct seller contacts, $175 budget, and no MVP yet."
            )

        print(f"Founder answers: \"{answer}\"\n")
        graph.invoke(Command(resume=answer), cfg)

    final_state = graph.get_state(cfg).values
    print("\n" + "=" * 70)
    print("PIPELINE EXECUTION COMPLETE")
    print("=" * 70)

    _print_startup_profile(final_state)
    _print_critical_assumption(final_state)
    _print_experiment(final_state)

    # Verification assertions
    active_exp = final_state.get("active_experiment")
    assert active_exp is not None, "active_experiment missing in final state!"
    assert active_exp.get("status") == "READY", f"Status expected READY, got {active_exp.get('status')}"
    assert active_exp.get("assumption_id") == final_state["active_assumption"]["id"]
    assert len(active_exp.get("valid_inferences", [])) > 0
    assert len(active_exp.get("invalid_inferences", [])) > 0
    assert active_exp.get("support_signal") is not None
    assert active_exp.get("contradiction_signal") is not None
    assert active_exp.get("inconclusive_signal") is not None

    critic = final_state.get("experiment_critic_output")
    assert critic is not None, "experiment_critic_output missing!"
    print(f"\nCRITIC VERDICT: {critic.get('verdict')}")

    can_exec = final_state.get("can_execute_result")
    assert can_exec is not None, "can_execute_result missing!"
    print(f"CAN EXECUTE READINESS: {can_exec.get('readiness')} ({can_exec.get('reason')})")

    print("\n[SUCCESS] ParcelMate Milestone 3 live verification passed!\n")


if __name__ == "__main__":
    run_m3_live_test()

