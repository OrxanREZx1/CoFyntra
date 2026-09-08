"""
Live verification script for Milestone 2.6.1: Baseline Startup Check-In + Weak-Link Discrimination Loop.

Tests the two required real-world scenarios with live OpenAI integration:
  Scenario 1: Pre-Filled Rich Baseline Input -> Skips baseline questions completely (0 questions asked!)
  Scenario 2: Sparse Mynera Input -> Triggers compact baseline question, integrates answer, and proceeds.
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


def test_scenario_1_prefilled():
    print("\n" + "=" * 70, flush=True)
    print("SCENARIO 1: PRE-FILLED RICH BASELINE INPUT (BASELINE SHORTCUT VERIFICATION)", flush=True)
    print("=" * 70, flush=True)

    founder_input = (
        "We are 3 students. Two developers and one designer. We each have 10 hours per week. "
        "We built a prototype but no MVP. We spoke with 15 students but nobody has tested the product. "
        "We are self-funded and can spend around 300 AZN per month. "
        "We are building Mynera, an AI tool that helps students choose the right IT branch."
    )

    print(f"\nFounder Initial Description:\n{founder_input}\n", flush=True)

    checkpointer = MemorySaver()
    graph = build_graph(checkpointer=checkpointer)
    thread_id = "live-test-prefilled"
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

    # Check if interrupted
    snapshot = graph.get_state(cfg)
    while snapshot.tasks and snapshot.tasks[0].interrupts:
        interrupt_val = snapshot.tasks[0].interrupts[0].value
        interrupt_type = interrupt_val.get("interrupt_type", "unknown")
        question = interrupt_val.get("question")
        print(f"\n[INTERRUPT: {interrupt_type.upper()}] Startary asks: {question}", flush=True)

        if interrupt_type == "strategic":
            simulated_reply = "Students mainly struggle because universities give generic curriculum and online courses are overwhelming. Most end up choosing whatever their friends pick."
        elif interrupt_type == "weak_link":
            simulated_reply = "When students face this problem, they usually spend weeks agonizing and watching YouTube, but they don't buy anything because there are no clear tools."
        else:
            simulated_reply = "We are fully focused on university students in Azerbaijan."

        print(f"Founder simulated answer: {simulated_reply}", flush=True)
        graph.invoke(Command(resume=simulated_reply), cfg)
        snapshot = graph.get_state(cfg)

    final_values = snapshot.values
    baseline_turns = final_values.get("baseline_question_turns", 0)
    print(f"\n--- SCENARIO 1 RESULTS ---", flush=True)
    print(f"Baseline Question Turns Used: {baseline_turns} (Expected: 0)", flush=True)
    assert baseline_turns == 0, f"Expected 0 baseline turns for rich input, but got {baseline_turns}"
    print("SUCCESS: Rich input bypassed baseline questions completely!", flush=True)

    print(f"Selected Weak Link        : {final_values.get('weak_area')}", flush=True)
    print(f"Selection Confidence      : {final_values.get('selection_confidence')}", flush=True)
    print(f"Competing Weak Areas      : {final_values.get('competing_weak_areas')}", flush=True)
    print(f"Active Assumption         : {final_values.get('active_assumption', {}).get('statement')}", flush=True)
    print(f"Final Workflow Phase      : {final_values.get('phase')}", flush=True)


def test_scenario_2_sparse():
    print("\n" + "=" * 70, flush=True)
    print("SCENARIO 2: SPARSE MYNERA INPUT (BASELINE CHECK-IN VERIFICATION)", flush=True)
    print("=" * 70, flush=True)

    founder_input = (
        "Mynera is an AI agent that helps students and freshmen find the most appropriate IT branch "
        "with resources and tasks before they waste time and money trying different courses."
    )

    print(f"\nFounder Initial Description:\n{founder_input}\n", flush=True)

    checkpointer = MemorySaver()
    graph = build_graph(checkpointer=checkpointer)
    thread_id = "live-test-sparse"
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
    turn = 0
    snapshot = graph.get_state(cfg)
    while snapshot.tasks and snapshot.tasks[0].interrupts:
        turn += 1
        interrupt_val = snapshot.tasks[0].interrupts[0].value
        interrupt_type = interrupt_val.get("interrupt_type", "unknown")
        question = interrupt_val.get("question")
        print(f"\n[INTERRUPT {turn}: {interrupt_type.upper()}] Startary asks:\n  {question}", flush=True)

        if interrupt_type == "baseline":
            # Give a comprehensive operational response answering the baseline question
            simulated_reply = (
                "I am working on this alone right now, spending around 15 hours a week. "
                "I have Figma wireframes and interviewed 10 freshmen, but no live code prototype yet. "
                "I have zero external funding and can spend about 100 AZN a month from personal savings."
            )
        elif interrupt_type == "strategic":
            simulated_reply = (
                "Freshmen get overwhelmed by 10 different roadmap options. Many switch majors or drop out "
                "of bootcamps after spending 500 AZN on web dev when they actually prefer data analysis."
            )
        elif interrupt_type == "weak_link":
            simulated_reply = (
                "They actively search free roadmaps on YouTube, but give up due to lack of practical feedback."
            )
        else:
            simulated_reply = "We are focused on undergraduate CS students in Baku."

        print(f"Founder responds:\n  {simulated_reply}", flush=True)
        graph.invoke(Command(resume=simulated_reply), cfg)
        snapshot = graph.get_state(cfg)

    final_values = snapshot.values
    baseline_turns = final_values.get("baseline_question_turns", 0)
    print(f"\n--- SCENARIO 2 RESULTS ---", flush=True)
    print(f"Baseline Question Turns Used: {baseline_turns} (Expected: >= 1)", flush=True)
    assert baseline_turns >= 1, f"Expected at least 1 baseline turn for sparse input, but got {baseline_turns}"
    print("SUCCESS: Sparse input triggered baseline question and integrated answer!", flush=True)

    print(f"Selected Weak Link        : {final_values.get('weak_area')}", flush=True)
    print(f"Selection Confidence      : {final_values.get('selection_confidence')}", flush=True)
    print(f"Competing Weak Areas      : {final_values.get('competing_weak_areas')}", flush=True)
    print(f"Active Assumption         : {final_values.get('active_assumption', {}).get('statement')}", flush=True)
    print(f"Final Workflow Phase      : {final_values.get('phase')}", flush=True)


if __name__ == "__main__":
    settings = get_settings()
    print(f"Running M2.6.1 live test with model: {settings.model_name}", flush=True)
    test_scenario_1_prefilled()
    test_scenario_2_sparse()
    print("\n" + "=" * 70, flush=True)
    print("ALL M2.6.1 LIVE TESTS COMPLETED SUCCESSFULLY!", flush=True)
    print("=" * 70 + "\n", flush=True)

