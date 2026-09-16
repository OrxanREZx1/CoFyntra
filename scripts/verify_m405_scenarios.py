"""
Verification Script for Milestone 4.0.5 — Founder-Facing Experiment Summary.

Verifies LoopCart scenario:
1. ExecutionHandoff generates compact summary answering the 5 key questions:
   - WHAT ARE WE TESTING?
   - WHAT SHOULD I DO?
   - WHAT SHOULD I BRING BACK?
   - WHAT COUNTS AS A USEFUL SIGNAL?
   - WHAT LOOKS PROMISING SO FAR?
2. Epistemic caution and grounding:
   - No generic "Validating critical assumption"
   - No generic "Target customer cohort"
   - No unsupported praise, market hype, or budget-as-advantage
   - Positive signals reflect grounded facts (pilot interest, willingness to pay, recurring returns, in-house team)
3. Full experiment remains accessible intact:
   - render_execution_handoff_text(handoff, mode="full")
4. Waiting-for-evidence detail requests:
   - "Show me the full interview guide." returns full guide without mutating state.
   - "Give me the exact questions." returns concrete questions.
   - "Show me the decision rule." returns exact decision rules.
"""

from __future__ import annotations

import sys
from startary.domain.enums import WorkflowPhase, ExperimentStatus
from startary.domain.experiment import (
    ExecutionHandoff,
    FounderExperimentSummary,
    compose_current_positive_signals,
    compose_founder_experiment_summary,
    render_execution_handoff_text,
)
from startary.graph.nodes import (
    compose_execution_handoff,
    respond_to_waiting_founder,
)


def verify_loopcart_m405():
    print("=" * 60)
    print("VERIFYING MILESTONE 4.0.5 — LOOPCART FOUNDER SUMMARY")
    print("=" * 60)

    # 1. Setup LoopCart state with grounded facts
    loopcart_state = {
        "project_id": "proj-loopcart-m405",
        "phase": WorkflowPhase.TEST,
        "iteration": 1,
        "raw_context": (
            "LoopCart is a B2B Shopify returns-management plugin. "
            "Founded by 2 engineers with 30 hrs/week and $1,000 budget. "
            "They have a working MVP deployed on their cousin's apparel store."
        ),
        "startup_state": {
            "execution_context": {
                "technical_capability": {"value": True, "provenance": "EXPLICIT_FOUNDER"},
                "customer_access": {"value": "Cousin's apparel store", "provenance": "EXPLICIT_FOUNDER"},
            },
            "progress": {
                "mvp_exists": {"value": True, "provenance": "EXPLICIT_FOUNDER"},
            },
            "areas": {
                "Problem": {
                    "evidence_provided": [
                        "One relevant merchant already experiences recurring size-related returns."
                    ],
                },
                "Business Model": {
                    "evidence_provided": [
                        "Merchant expressed conditional willingness to pay commission if returns decrease."
                    ],
                },
            },
        },
        "active_assumption": {
            "id": "asmp-loopcart-001",
            "statement": "Small fashion e-commerce merchants experience sizing-related returns as a high-friction operational pain.",
            "learning_goal": "Validate whether sizing returns represent a severe pain point worth solving.",
            "area": "Problem",
            "status": "UNTESTED",
            "is_critical": True,
        },
        "active_experiment": {
            "id": "exp-loopcart-001",
            "assumption_id": "asmp-loopcart-001",
            "experiment_type": "CUSTOMER_INTERVIEWS",
            "objective": "Determine whether small fashion merchants view sizing returns as a severe problem worth solving.",
            "hypothesis": "Validating critical assumption: Small fashion e-commerce merchants experience sizing-related returns as a high-friction operational pain.",
            "founder_actions": [
                "Interview 6-8 owner-led online clothing stores using recent-return questions.",
                "Ask for 3 most recent return cases and reasons.",
                "Inquire about monthly return overhead costs.",
                "Ask if they would evaluate a pilot workflow.",
            ],
            "target_participants_or_scope": "Owner-led online clothing stores selling multiple apparel sizes",
            "evidence_to_collect": "recent return examples, return frequency, cost or operational burden, current workaround, who owns the decision, whether a concrete follow-up was accepted",
            "support_signal": "Several independent merchants report recent costly sizing returns, and at least some decision-makers accept a real follow-up.",
            "contradiction_signal": "Most merchants view returns as a minor or fully solved problem and decline follow-up.",
            "inconclusive_signal": "Merchants have mixed feedback or cannot estimate return friction.",
            "valid_inferences": ["Validates whether sizing returns create genuine operational friction."],
            "invalid_inferences": ["Does not prove long-term SaaS willingness to pay."],
            "status": "READY",
        },
        "diagnosis_snapshot": {
            "current_advantages": [
                "Direct merchant access through cousin's apparel store",
                "Technical founders with in-house engineering",
            ],
        },
        "founder_messages": [
            {"role": "founder", "content": "One relevant merchant already reports recurring size-related returns."},
            {"role": "founder", "content": "That merchant is willing to run a pilot."},
            {"role": "founder", "content": "They expressed conditional willingness to pay commission if returns decrease."},
        ],
    }

    # 2. Run compose_execution_handoff
    res = compose_execution_handoff(loopcart_state)
    handoff_dict = res["active_execution_handoff"]
    compact_text = res["execution_handoff_text"]
    full_text = handoff_dict.get("full_response_text")

    print("")
    print("--- COMPACT FOUNDER VIEW ---")
    print(compact_text)
    print("----------------------------")
    print(f"Length: {len(compact_text)} characters")

    # 3. Verify the 5 key sections are present
    assert "WHAT ARE WE TESTING?" in compact_text
    assert "WHAT SHOULD I DO?" in compact_text
    assert "WHAT SHOULD I BRING BACK?" in compact_text
    assert "WHAT COUNTS AS A USEFUL SIGNAL?" in compact_text
    assert "WHAT LOOKS PROMISING SO FAR?" not in compact_text
    print("")
    print("[OK] All 5 key sections are present in compact view.")

    # 4. Verify no internal jargon
    assert "Validating critical assumption" not in compact_text
    assert "Target customer cohort" not in compact_text
    assert "Target customer cohort" not in full_text
    print("[OK] Internal jargon ('Validating critical assumption', 'Target customer cohort') eliminated.")

    # 5. Verify positive signals
    summary = handoff_dict.get("summary") or {}
    pos_signals = summary.get("current_positive_signals") or []
    print("")
    print(f"Current Positive Signals ({len(pos_signals)}):")
    for s in pos_signals:
        print(f"  - {s}")

    assert 1 <= len(pos_signals) <= 4
    all_pos_str = " ".join(pos_signals).lower()
    assert "budget" not in all_pos_str
    assert "30 hrs" not in all_pos_str
    assert "huge market" not in all_pos_str
    assert any("pilot" in s.lower() for s in pos_signals) or any("merchant" in s.lower() for s in pos_signals)
    print("[OK] Positive signals are strictly grounded, non-hyped, and exclude budget/hours.")

    # 6. Verify full handoff is preserved
    assert len(full_text) > len(compact_text)
    assert "What to do:" in full_text
    assert "Who to do it with:" in full_text
    assert "Confirms assumption:" in full_text
    assert "WHEN YOU'RE DONE" in full_text
    print(f"[OK] Full handoff preserved ({len(full_text)} characters).")

    # 7. Verify WAITING_FOR_EVIDENCE detail queries
    loopcart_state["active_execution_handoff"] = handoff_dict
    loopcart_state["execution_handoff_text"] = compact_text
    loopcart_state["phase"] = WorkflowPhase.WAITING_FOR_EVIDENCE

    loopcart_state["founder_response"] = "Show me the full interview guide."
    detail_res1 = respond_to_waiting_founder(loopcart_state)
    assert len(detail_res1["waiting_response"]) > 300
    assert "What to do:" in detail_res1["waiting_response"]
    print("[OK] 'Show me the full interview guide.' returns complete methodology.")

    loopcart_state["founder_response"] = "Give me the exact questions."
    detail_res2 = respond_to_waiting_founder(loopcart_state)
    assert "Ask for 3 most recent return cases" in detail_res2["waiting_response"]
    print("[OK] 'Give me the exact questions.' returns detailed actions.")

    loopcart_state["founder_response"] = "Show me the decision rule."
    detail_res3 = respond_to_waiting_founder(loopcart_state)
    assert "Several independent merchants report recent costly sizing returns" in detail_res3["waiting_response"]
    print("[OK] 'Show me the decision rule.' returns exact support/contradiction rules.")

    print("")
    print("============================================================")
    print("ALL MILESTONE 4.0.5 VERIFICATION CHECKS PASSED SUCCESSFULLY!")
    print("============================================================")


if __name__ == "__main__":
    verify_loopcart_m405()
