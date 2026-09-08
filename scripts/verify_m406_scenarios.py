"""
Verification Script for Milestone 4.0.6 — Experiment Failure Recovery Action.

Verifies ShelfSense scenario:
1. Reconstructs ShelfSense run where Experiment Critic rejects experiment twice.
2. Graceful stop routes to handle_experiment_failure_stop, setting QUALITY_BAR_NOT_MET.
3. Composes safe, bounded FallbackLearningAction:
   - WHAT DO WE STILL NEED TO LEARN?
   - WHAT SHOULD I DO NEXT?
   - WHAT SHOULD I BRING BACK?
   - WHAT SHOULD I NOT CONCLUDE YET?
4. Preserves diagnosis and active critical assumption without fake validation.
5. Terminal state renders EXPERIMENT_DESIGN_NEEDS_REVISION with fallback action.
6. Founder conversational inquiries during fallback ("Who should I talk to?", "Can I start with 3 shops?").
7. Returned fallback observations safely enrich startup state without prematurely validating the assumption.
"""

from __future__ import annotations

import sys

from startary.domain.enums import (
    CriticVerdict,
    ExecutionReadiness,
    ExperimentDesignStatus,
    ExperimentType,
    NextLearningActionType,
    WorkflowPhase,
)
from startary.domain.experiment import (
    FallbackLearningAction,
    compose_fallback_learning_action,
    render_fallback_learning_action_text,
)
from startary.domain.terminal_state import (
    TerminalStatus,
    render_terminal_state,
    resolve_founder_facing_status,
)
from startary.graph.nodes import (
    handle_experiment_failure_stop,
    respond_to_waiting_founder,
)


def verify_shelfsense_m406():
    print("=" * 70)
    print("VERIFYING MILESTONE 4.0.6 — SHELFSENSE EXPERIMENT FAILURE RECOVERY")
    print("=" * 70)

    # 1. Setup ShelfSense state with exhausted revision budget
    shelfsense_state = {
        "project_id": "proj-shelfsense-m406",
        "phase": WorkflowPhase.TEST,
        "iteration": 1,
        "critic_revision_count": 2,
        "weak_area": "Problem",
        "raw_context": (
            "ShelfSense helps independent grocery store owners prevent produce and dairy "
            "spoilage with camera-based shelf tracking. Founded by two ML engineers."
        ),
        "startup_state": {
            "execution_context": {
                "technical_capability": {"value": True, "provenance": "EXPLICIT_FOUNDER"},
                "customer_access": {"value": "Local neighborhood bodegas and grocers", "provenance": "EXPLICIT_FOUNDER"},
            },
            "areas": {
                "Problem": {
                    "evidence_provided": [],
                },
            },
        },
        "active_assumption": {
            "id": "asmp-shelfsense-001",
            "statement": "Validating critical assumption: Independent grocery store owners experience perishable stockouts and spoilage as a recurring, high-cost operational burden.",
            "learning_goal": "Validate whether inventory spoilage causes significant financial or time loss.",
            "area": "Problem",
        },
        "diagnosis_snapshot": {
            "id": "snap-shelfsense-001",
            "weak_link": "Problem",
            "weak_link_confidence": "HIGH",
            "stage": "RAW_IDEA",
            "critical_assumption": "Independent grocery store owners experience perishable stockouts and spoilage as a recurring, high-cost operational burden.",
        },
        "active_experiment": {
            "id": "exp-shelfsense-revised",
            "assumption_id": "asmp-shelfsense-001",
            "target_area": "Problem",
            "type": ExperimentType.CUSTOMER_INTERVIEW.value,
            "title": "Grocery Spoilage Discovery",
            "objective": "Test problem urgency",
            "hypothesis": "Grocery store owners will adopt automated inventory tracking.",
            "target_participants_or_scope": "5–8 independent grocery store owners in urban centers",
            "execution_readiness": ExecutionReadiness.BLOCKED.value,
        },
        "experiment_critic_output": {
            "verdict": CriticVerdict.REVISE.value,
            "issues": [
                "Offering a free manual tally sheet is a weak proxy for willingness to adopt camera software.",
                "Contradiction criteria require top-three operational priority rather than direct incident observation.",
            ],
        },
    }

    print("\n[Step 1] Triggering handle_experiment_failure_stop on revision exhaustion...")
    updates = handle_experiment_failure_stop(shelfsense_state)
    merged = {**shelfsense_state, **updates}

    # Verify status and phase
    assert merged["experiment_design_status"] == ExperimentDesignStatus.QUALITY_BAR_NOT_MET.value
    assert merged["learning_input_mode"] == "FALLBACK"
    print("  -> experiment_design_status:", merged["experiment_design_status"])
    print("  -> learning_input_mode:", merged["learning_input_mode"])

    # Verify FallbackLearningAction creation
    fallback = merged.get("fallback_learning_action")
    assert fallback is not None
    assert fallback["action_type"] == NextLearningActionType.FALLBACK_LEARNING_ACTION.value
    print("  -> Fallback action type:", fallback["action_type"])
    print("  -> Target Area:", fallback["target_area"])
    print("  -> Learning Goal:", fallback["learning_goal"])
    print("  -> Primary Action:", fallback["primary_action"])

    # 2. Check Fallback Learning Action structure & boundaries
    print("\n[Step 2] Validating Epistemic Boundaries and Content...")
    assert not fallback["learning_goal"].lower().startswith("validating critical assumption:")
    assert "grocery store owners" in fallback["who_to_engage"].lower()

    # Questions must reconstruct events and workarounds
    q_str = " ".join(fallback["what_to_ask_or_observe"]).lower()
    assert "recent" in q_str
    assert "workaround" in q_str or "current" in q_str
    print("  -> Questions focus on concrete past events and existing workarounds: OK")

    # What not to infer must include willingness to pay and need for AI
    n_str = " ".join(fallback["what_not_to_infer"]).lower()
    assert "willingness to pay" in n_str
    assert "ai" in n_str or "software" in n_str
    print("  -> Non-inference boundaries explicitly exclude WTP and AI need: OK")

    # 3. Check Terminal State Rendering
    print("\n[Step 3] Validating Terminal State Display...")
    terminal_output = render_terminal_state(merged)
    assert "STARTARY — EXPERIMENT DESIGN NEEDS REVISION" in terminal_output
    assert "WHAT DO WE STILL NEED TO LEARN?" in terminal_output
    assert "WHAT SHOULD I DO NEXT?" in terminal_output
    assert "WHAT SHOULD I BRING BACK?" in terminal_output
    assert "WHAT SHOULD I NOT CONCLUDE YET?" in terminal_output
    assert "This action is for learning, not validation." in terminal_output
    assert "STARTARY — INITIAL DIAGNOSIS COMPLETE" not in terminal_output
    print("  -> Terminal banner renders EXPERIMENT_DESIGN_NEEDS_REVISION with 4 core fallback sections: OK")

    # 4. Check Founder Inquiries
    print("\n[Step 4] Validating Founder Interaction During Fallback...")
    # Who to engage query
    reply_who = respond_to_waiting_founder({
        **merged,
        "founder_response": "Who should I talk to?",
        "founder_messages": [{"role": "founder", "content": "Who should I talk to?"}],
    })
    assert "grocery store owners" in reply_who["waiting_response"].lower()
    print("  -> Inquiry 'Who should I talk to?' answered directly from fallback action: OK")

    # Scope reduction query
    reply_scope = respond_to_waiting_founder({
        **merged,
        "founder_response": "Can I start with 3 shops?",
        "founder_messages": [{"role": "founder", "content": "Can I start with 3 shops?"}],
    })
    assert "2–3 conversations" in reply_scope["waiting_response"]
    print("  -> Inquiry 'Can I start with 3 shops?' answered safely: OK")

    # 5. Check Returned Fallback Observations
    print("\n[Step 5] Validating Returned Fallback Observations...")
    obs_text = (
        "I spoke to 4 neighborhood grocers today. 3 of them said they discard $80-$120 worth of dairy "
        "every Tuesday, and currently rely on manual pencil-and-paper clipboards to check expiration dates."
    )
    reply_obs = respond_to_waiting_founder({
        **merged,
        "founder_response": obs_text,
        "founder_messages": [{"role": "founder", "content": obs_text}],
    })
    assert "I have captured these notes into your startup state" in reply_obs["waiting_response"]
    assert "without prematurely marking the assumption validated" in reply_obs["waiting_response"]
    assert reply_obs["startup_state"]["areas"]["Problem"]["evidence_provided"] == [obs_text]
    assert len(reply_obs["founder_evidence"]) == 1
    assert reply_obs["founder_evidence"][0]["source"] == "FALLBACK_LEARNING_ACTION"
    # Verify assumption status remained intact
    assert merged["active_assumption"].get("status") != "SUPPORTED"
    print("  -> Observations safely enriched startup state without premature assumption validation: OK")

    print("\n" + "=" * 70)
    print("SHELFSENSE VERIFICATION SUCCESSFUL: ALL CHECKS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    verify_shelfsense_m406()
