"""
Live test runner for Milestone 2.6 acceptance verification.

Runs the exact founder context specified in Section 36:
'We are 3 students building an AI startup that helps university students choose an IT specialization. Two of us code and one handles design. We have a basic prototype but no MVP yet. We can spend around 300 AZN per month and each of us can give about 10 hours a week. I personally struggled to choose between backend and AI, and I have spoken with several classmates who had similar confusion.'
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


def run_live_test():
    settings = get_settings()
    print("==================================================")
    print("STARTARY MILESTONE 2.6 — LIVE MANUAL TEST")
    print(f"Model: {settings.model_name}")
    print("==================================================\n")

    founder_input = (
        "We are 3 students building an AI startup that helps university students choose an IT specialization. "
        "Two of us code and one handles design. We have a basic prototype but no MVP yet. "
        "We can spend around 300 AZN per month and each of us can give about 10 hours a week. "
        "I personally struggled to choose between backend and AI, and I have spoken with several classmates who had similar confusion."
    )

    print("FOUNDER INPUT:")
    print(founder_input)
    print("\nExecuting graph with live LLM...\n")

    checkpointer = MemorySaver()
    graph = build_graph(checkpointer=checkpointer)
    thread_id = "live-m26-test"
    cfg = {"configurable": {"thread_id": thread_id}}

    initial_state: StartaryGraphState = {
        "project_id": thread_id,
        "phase": WorkflowPhase.UNDERSTAND,
        "iteration": 1,
        "raw_context": founder_input,
        "founder_messages": [{"role": "founder", "content": founder_input}],
        "clarification_turn_count": 0,
        "clarification_history": [],
    }

    graph.invoke(initial_state, cfg)

    snapshot = graph.get_state(cfg)
    values = snapshot.values

    # 1. Extracted Strategic State
    diagnosis = values.get("diagnosis") or {}
    print("\n1. EXTRACTED STRATEGIC STATE:")
    print(f"  Summary: {diagnosis.get('summary')}")
    print(f"  Clear areas: {diagnosis.get('clear_areas')}")
    print(f"  Unclear areas: {diagnosis.get('unclear_areas')}")
    print(f"  Missing areas: {diagnosis.get('missing_areas')}")
    print(f"  Claims: {diagnosis.get('unsupported_claims')}")

    # 2. Extracted Execution Context
    exec_ctx = values.get("execution_context") or {}
    print("\n2. EXTRACTED EXECUTION CONTEXT:")
    for k, v in exec_ctx.items():
        if isinstance(v, dict) and "provenance" in v:
            print(f"  {k:<35}: value={v.get('value')} | prov={v.get('provenance')} | conf={v.get('confidence')}")
        elif isinstance(v, list) and v:
            print(f"  {k:<35}: {v}")

    # 3. Extracted Startup Progress
    prog = values.get("startup_progress") or {}
    print("\n3. EXTRACTED STARTUP PROGRESS:")
    for k, v in prog.items():
        if isinstance(v, dict) and "provenance" in v:
            print(f"  {k:<35}: value={v.get('value')} | prov={v.get('provenance')} | conf={v.get('confidence')}")
        elif isinstance(v, list) and v:
            print(f"  {k:<35}: {v}")

    # 4. Information Gap Plan & Clarification Decision
    gap_plan = values.get("gap_plan") or {}
    print("\n4. INFORMATION GAP PLAN:")
    print(f"  should_ask          : {gap_plan.get('should_ask')}")
    print(f"  strategic_sufficient: {gap_plan.get('strategic_sufficient')}")
    print(f"  target_domain       : {gap_plan.get('target_domain')}")
    print(f"  target_field_or_area: {gap_plan.get('target_field_or_area')}")
    print(f"  priority            : {gap_plan.get('priority')}")
    print(f"  information_goal    : {gap_plan.get('information_goal')}")
    print(f"  reason              : {gap_plan.get('reason')}")
    print(f"  question            : {gap_plan.get('question')}")

    # Check if interrupted
    if snapshot.tasks and snapshot.tasks[0].interrupts:
        q = snapshot.tasks[0].interrupts[0].value
        print(f"\nINTERRUPT TRIGGERED (Question asked to founder):")
        print(f"  Question: {q.get('question')}")
        print(f"  Target  : {q.get('target_area')}")
    else:
        print("\nNO INTERRUPT: Graph proceeded to downstream Weak Link / Critical Assumption.")
        weak_area = values.get("weak_area")
        weak_details = values.get("weak_link_details") or {}
        rc = values.get("reality_check") or {}
        assumption = values.get("active_assumption") or {}
        print(f"  Weak Link: {weak_area} (score: {weak_details.get('score')})")
        print(f"  Reality Check: {rc.get('conclusion')}")
        print(f"  Critical Assumption: {assumption.get('statement')}")

    # Save results to json for detailed analysis
    results_path = Path("scratch/live_m26_test_result.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "founder_input": founder_input,
                "diagnosis": diagnosis,
                "execution_context": exec_ctx,
                "startup_progress": prog,
                "gap_plan": gap_plan,
                "interrupted": bool(snapshot.tasks and snapshot.tasks[0].interrupts),
                "question": snapshot.tasks[0].interrupts[0].value if (snapshot.tasks and snapshot.tasks[0].interrupts) else None,
                "weak_area": values.get("weak_area"),
                "critical_assumption": values.get("active_assumption"),
            },
            f,
            indent=2,
            default=str,
        )
    print(f"\nFull results saved to {results_path}")


if __name__ == "__main__":
    run_live_test()

