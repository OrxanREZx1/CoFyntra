"""
Tests for Milestone 4.0.5 — Founder-Facing Experiment Summary.

Covers:
1. FounderExperimentSummary validates and serializes required fields.
2. Compact handoff contains exactly the 5 key sections and meets concise length targets.
3. Actual critical uncertainty in founder language replaces generic "Validating critical assumption".
4. Concrete participant description replaces generic "Target customer cohort".
5. Positive signals only use grounded facts (commitments, problem evidence, access, capabilities).
6. Unknown / unvalidated strengths and forbidden terms (budget, hours, hype) are strictly excluded.
7. Full handoff remains available and intact with complete methodology.
8. Waiting-for-evidence detail queries ("show full guide", "exact questions", "decision rule", "what to avoid")
   work deterministically without mutating state or regenerating methodology.
"""

from __future__ import annotations

import pytest

from startary.domain.enums import WorkflowPhase
from startary.domain.experiment import (
    ExecutionHandoff,
    FounderExperimentSummary,
    compose_current_positive_signals,
    compose_founder_experiment_summary,
    render_execution_handoff_text,
)
from startary.graph.graph_state import StartaryGraphState
from startary.graph.nodes import (
    compose_execution_handoff,
    respond_to_waiting_founder,
)


def test_founder_experiment_summary_validates():
    """FounderExperimentSummary validates and serializes the 5 core fields."""
    summary = FounderExperimentSummary(
        what_are_we_testing="Whether online clothing stores repeatedly suffer sizing returns.",
        what_should_i_do="Interview 6–8 owner-led clothing stores using recent-return questions.",
        what_should_i_bring_back="- Return frequency\n- Operational costs\n- Willingness to pilot",
        useful_signal="Multiple merchants report recurring costly returns and accept a follow-up.",
        current_positive_signals=[
            "One merchant already reports recurring returns.",
            "That merchant is willing to pilot the product.",
        ],
        short_title="Customer Interviews",
        estimated_effort="5-8 hours",
        participant_target="Owner-led apparel stores",
    )

    data = summary.model_dump()
    assert data["what_are_we_testing"].startswith("Whether")
    assert "Interview 6–8" in data["what_should_i_do"]
    assert len(data["current_positive_signals"]) == 2
    assert data["short_title"] == "Customer Interviews"


def test_compact_handoff_contains_exactly_5_key_sections():
    """Compact handoff renders exactly the 5 key sections and is concise."""
    summary = FounderExperimentSummary(
        what_are_we_testing="Whether owner-led online clothing stores suffer costly returns.",
        what_should_i_do="Interview 6–8 stores using structured questions.",
        what_should_i_bring_back="- return counts\n- costs\n- follow-up acceptance",
        useful_signal="Several merchants confirm pain and accept follow-up.",
        current_positive_signals=["One store reports 25% return rate."],
    )
    handoff = ExecutionHandoff(
        experiment_id="exp-test-01",
        assumption_id="asmp-test-01",
        experiment_type="CUSTOMER_INTERVIEWS",
        learning_goal="Validate return problem",
        what_we_are_testing="Whether owner-led online clothing stores suffer costly returns.",
        founder_actions=["Interview 6-8 store owners."],
        target_participants_or_context="Online apparel stores",
        evidence_to_collect="Return counts, handling costs",
        summary=summary,
    )

    rendered = render_execution_handoff_text(handoff, mode="compact")

    assert "YOUR NEXT TEST" in rendered
    assert "WHAT ARE WE TESTING?" in rendered
    assert "WHAT SHOULD I DO?" in rendered
    assert "WHAT SHOULD I BRING BACK?" in rendered
    assert "WHAT COUNTS AS A USEFUL SIGNAL?" in rendered
    assert "WHAT LOOKS PROMISING SO FAR?" not in rendered

    # Target roughly 500-900 chars (allow slight margin, strictly < 1200)
    assert 300 < len(rendered) < 1200


def test_actual_critical_uncertainty_replaces_generic_label():
    """Generic 'Validating critical assumption' is stripped and replaced with founder language."""
    handoff = ExecutionHandoff(
        experiment_id="exp-test-02",
        assumption_id="asmp-test-02",
        experiment_type="CUSTOMER_INTERVIEWS",
        learning_goal="Validate critical assumption",
        what_we_are_testing="Validating critical assumption: Small fashion merchants experience sizing returns as high friction.",
        founder_actions=["Interview merchants"],
        target_participants_or_context="Fashion merchants",
        evidence_to_collect="Return rates",
    )

    rendered = render_execution_handoff_text(handoff, mode="compact")
    assert "Validating critical assumption" not in rendered
    assert "Whether small fashion merchants experience sizing returns as high friction." in rendered


def test_concrete_participant_description_replaces_generic_cohort():
    """Generic 'Target customer cohort' is not exposed in handoff."""
    handoff = ExecutionHandoff(
        experiment_id="exp-test-03",
        assumption_id="asmp-test-03",
        experiment_type="CUSTOMER_INTERVIEWS",
        learning_goal="Problem severity",
        what_we_are_testing="Whether merchants experience return friction.",
        founder_actions=["Interview 5 stores"],
        target_participants_or_context="Target customer cohort",
        evidence_to_collect="Quotes",
    )

    full_rendered = render_execution_handoff_text(handoff, mode="full")
    assert "Target customer cohort" not in full_rendered

    compact_rendered = render_execution_handoff_text(handoff, mode="compact")
    assert "Target customer cohort" not in compact_rendered


def test_positive_signals_only_use_grounded_facts():
    """Positive signals extract grounded facts across commitments, problem evidence, and capability."""
    state = {
        "raw_context": "LoopCart returns tool for Shopify apparel.",
        "startup_state": {
            "execution_context": {
                "technical_capability": {"value": True, "provenance": "EXPLICIT_FOUNDER"},
                "customer_access": {"value": "Cousin's apparel store", "provenance": "EXPLICIT_FOUNDER"},
            },
            "progress": {
                "mvp_exists": {"value": True, "provenance": "EXPLICIT_FOUNDER"},
            },
        },
        "founder_messages": [
            {"role": "founder", "content": "One merchant is willing to pilot the product and pay commission if returns drop."},
            {"role": "founder", "content": "That merchant has recurring size-related returns."},
        ],
    }
    snapshot = {
        "current_advantages": [
            "Direct merchant access through cousin's apparel store",
            "In-house engineering team",
        ]
    }

    signals = compose_current_positive_signals(state=state, diagnosis_snapshot=snapshot)

    assert 1 <= len(signals) <= 4
    # Must contain grounded items
    sig_text = " ".join(signals).lower()
    assert "pilot" in sig_text or "merchant" in sig_text or "prototype" in sig_text or "access" in sig_text


def test_unknown_and_unvalidated_strengths_excluded():
    """Financial budgets, hours, and generic hype are excluded from positive signals."""
    state = {
        "raw_context": "We have $1,000 budget and 30 hrs/week.",
        "startup_state": {
            "execution_context": {
                "technical_capability": {"value": None, "provenance": "UNKNOWN"},
                "customer_access": {"value": None, "provenance": "UNKNOWN"},
            }
        },
    }
    snapshot = {
        "current_advantages": [
            "$1,000 budget",
            "30 hrs/week combined",
            "Huge market opportunity",
            "Strong product-market fit",
        ]
    }

    signals = compose_current_positive_signals(state=state, diagnosis_snapshot=snapshot)

    # All items were budget, hours, or hype -> should be filtered out
    assert signals == ["No strong positive signal is established yet."]


def test_full_handoff_remains_available_and_unchanged():
    """Full mode preserves all detailed methodology, actions, constraints, and decision rules."""
    handoff = ExecutionHandoff(
        experiment_id="exp-full-01",
        assumption_id="asmp-full-01",
        experiment_type="CUSTOMER_INTERVIEWS",
        learning_goal="Validate sizing return pain",
        what_we_are_testing="Merchants care about returns",
        founder_actions=[
            "Step 1: Contact stores",
            "Step 2: Ask return questions",
            "Step 3: Measure pain",
        ],
        target_participants_or_context="Shopify apparel merchants",
        evidence_to_collect="Return counts, handling costs, quotes",
        support_signal="4 of 6 merchants report high pain",
        contradiction_signal="Most merchants report no friction",
        inconclusive_signal="Unclear responses",
        important_constraints=["Do not pitch the solution in the first 15 minutes"],
        valid_inferences=["Proves whether return friction is experienced"],
        invalid_inferences=["Does not prove pricing willingness"],
    )

    full_text = render_execution_handoff_text(handoff, mode="full")

    assert "YOUR NEXT TEST" in full_text
    assert "What we're testing:" in full_text
    assert "Step 1: Contact stores" in full_text
    assert "Step 2: Ask return questions" in full_text
    assert "Step 3: Measure pain" in full_text
    assert "Shopify apparel merchants" in full_text
    assert "Confirms assumption: 4 of 6 merchants report high pain" in full_text
    assert "Weakens/disproves: Most merchants report no friction" in full_text
    assert "Inconclusive signal: Unclear responses" in full_text
    assert "Do not pitch the solution in the first 15 minutes" in full_text
    assert "Does not prove pricing willingness" in full_text
    assert "WHEN YOU'RE DONE" in full_text


def test_waiting_for_evidence_detail_requests():
    """Founder requests for full guide, exact questions, decision rules, and guardrails return full detail."""
    base_state: StartaryGraphState = {
        "project_id": "proj-verify-detail",
        "phase": WorkflowPhase.WAITING_FOR_EVIDENCE,
        "iteration": 1,
        "active_experiment": {
            "id": "exp-det-01",
            "assumption_id": "asmp-det-01",
            "experiment_type": "CUSTOMER_INTERVIEWS",
            "objective": "Determine sizing return burden",
            "hypothesis": "Merchants suffer return burden",
            "founder_actions": [
                "Ask stores about return percentages",
                "Record who handles restocking overhead",
            ],
            "target_participants_or_scope": "Owner-led apparel stores",
            "evidence_to_collect": "Quotes, return percentages",
            "support_signal": "4+ stores report major return burden",
            "contradiction_signal": "Stores view returns as trivial",
            "inconclusive_signal": "No store can estimate return burden",
            "valid_inferences": ["Establishes merchant problem severity"],
            "invalid_inferences": ["Does not establish SaaS conversion"],
        },
        "active_execution_handoff": {
            "experiment_id": "exp-det-01",
            "assumption_id": "asmp-det-01",
            "important_constraints": ["Do not offer discounts or pitch features"],
            "full_response_text": "FULL EXPERIMENT METHODOLOGY:\nStep 1... Step 2...",
            "response_text": "COMPACT HANDOFF",
        },
    }

    # 1. Full guide request
    base_state["founder_response"] = "Show me the full interview guide."
    res1 = respond_to_waiting_founder(base_state)
    assert "FULL EXPERIMENT METHODOLOGY" in res1["waiting_response"]
    assert res1["phase"] == WorkflowPhase.WAITING_FOR_EVIDENCE

    # 2. Exact questions request
    base_state["founder_response"] = "Give me the exact questions."
    res2 = respond_to_waiting_founder(base_state)
    assert "Ask stores about return percentages" in res2["waiting_response"]
    assert "Record who handles restocking overhead" in res2["waiting_response"]

    # 3. Decision rules request
    base_state["founder_response"] = "Show me the decision rule."
    res3 = respond_to_waiting_founder(base_state)
    assert "4+ stores report major return burden" in res3["waiting_response"]
    assert "Stores view returns as trivial" in res3["waiting_response"]

    # 4. Guardrails / what to avoid request
    base_state["founder_response"] = "What should I avoid?"
    res4 = respond_to_waiting_founder(base_state)
    assert "Do not offer discounts or pitch features" in res4["waiting_response"]
    assert "Does not establish SaaS conversion" in res4["waiting_response"]
