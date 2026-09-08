"""
Tests for Milestone 4.0.7.1 — Founder Summary Integrity.

Covers:
1. what_are_we_testing never renders blank under any circumstance.
2. Active critical assumption appears explicitly in compact mode above WHAT ARE WE TESTING.
3. Empty summary field falls back to experiment objective.
4. Objective absence falls back to active critical assumption.
5. Grounded positive problem signals are included (single-shop and interview signals).
6. Unsupported praise, market hype, and financial budget are strictly excluded.
7. 'No strong positive signal is established yet.' appears only when genuinely empty.
8. Full mode remains unchanged and intact with complete methodology.
9. validate_founder_experiment_summary enforces strict integrity rules.
"""

from __future__ import annotations

import pytest

from startary.domain.enums import WorkflowPhase
from startary.domain.experiment import (
    ExecutionHandoff,
    FounderExperimentSummary,
    clean_assumption_statement,
    clean_testing_question,
    compose_current_positive_signals,
    compose_founder_experiment_summary,
    render_execution_handoff_text,
    resolve_what_are_we_testing,
    validate_founder_experiment_summary,
)


def test_what_are_we_testing_never_renders_blank():
    """Fallback chain ensures WHAT ARE WE TESTING? never renders blank, even with empty inputs."""
    handoff = ExecutionHandoff(
        experiment_id="exp-blank-test",
        assumption_id="asmp-blank-test",
        critical_assumption="Independent grocery-shop owners experience recurring stockouts.",
        experiment_type="CUSTOMER_INTERVIEWS",
        learning_goal="",
        what_we_are_testing="Validating critical assumption",
        founder_actions=["Speak with owners"],
        target_participants_or_context="Grocery store owners",
        evidence_to_collect="Stockout counts",
        summary=FounderExperimentSummary(
            critical_assumption="Independent grocery-shop owners experience recurring stockouts.",
            what_are_we_testing="Validating critical assumption",  # Strips to empty
            what_should_i_do="Speak with 5 owners.",
            what_should_i_bring_back="- Notes",
            useful_signal="Owners confirm stockouts.",
            current_positive_signals=["One shop owner reports recurring stockouts."],
        ),
    )

    rendered = render_execution_handoff_text(handoff, mode="compact")

    assert "WHAT ARE WE TESTING?" in rendered
    parts = rendered.split("WHAT ARE WE TESTING?")
    after_testing = parts[1].split("WHAT SHOULD I DO?")[0].strip()
    assert len(after_testing) > 5, "WHAT ARE WE TESTING? rendered blank or trivial!"
    assert "Whether independent grocery-shop owners experience recurring stockouts." in after_testing


def test_active_critical_assumption_appears_in_compact_mode():
    """Active critical assumption is explicitly displayed right above WHAT ARE WE TESTING? in compact mode."""
    handoff = ExecutionHandoff(
        experiment_id="exp-asmp-visible",
        assumption_id="asmp-asmp-visible",
        critical_assumption="Independent grocery-shop owners experience recurring stockouts or perishable waste strongly enough that improving ordering decisions matters to them.",
        experiment_type="CUSTOMER_INTERVIEWS",
        learning_goal="Validate ordering recommendation utility",
        what_we_are_testing="Whether concrete ordering recommendations are credible and useful enough for an owner to change a supplier order.",
        founder_actions=["Shadow 2 grocery owners"],
        target_participants_or_context="Independent grocery stores",
        evidence_to_collect="Order changes, quotes",
        summary=FounderExperimentSummary(
            critical_assumption="Independent grocery-shop owners experience recurring stockouts or perishable waste strongly enough that improving ordering decisions matters to them.",
            what_are_we_testing="Whether concrete ordering recommendations are credible and useful enough for an owner to change a supplier order.",
            what_should_i_do="Shadow 2 grocery owners during supplier ordering.",
            what_should_i_bring_back="- Order adjustments\n- Owner feedback",
            useful_signal="At least one owner modifies an order based on recommendations.",
            current_positive_signals=[
                "Multiple independent shop owners reported recent stockouts.",
                "Multiple owners reported recurring perishable waste.",
            ],
        ),
    )

    rendered = render_execution_handoff_text(handoff, mode="compact")

    assert "CRITICAL ASSUMPTION" in rendered
    assert "Independent grocery-shop owners experience recurring stockouts or perishable waste" in rendered
    assert "WHAT ARE WE TESTING?" in rendered
    assert "Whether concrete ordering recommendations are credible" in rendered

    crit_pos = rendered.find("CRITICAL ASSUMPTION")
    test_pos = rendered.find("WHAT ARE WE TESTING?")
    do_pos = rendered.find("WHAT SHOULD I DO?")
    assert crit_pos < test_pos < do_pos


def test_empty_summary_field_falls_back_to_experiment_objective():
    """When what_are_we_testing is empty or stripped, fallback chain picks experiment objective."""
    cand1 = clean_testing_question("Validating critical assumption")
    assert cand1 == ""

    cand2 = "Determine whether independent grocery-shop owners experience daily stockouts."
    cand3 = "Validating critical assumption: grocery stockouts"
    cand4 = "Owners experience daily stockouts."

    result = resolve_what_are_we_testing(
        candidates=[cand1, cand2, cand3, cand4],
        fallback="Fallback statement.",
    )

    assert result == "Whether independent grocery-shop owners experience daily stockouts."


def test_objective_absence_falls_back_to_active_assumption():
    """When both summary field and objective are absent, fallback chain uses active critical assumption."""
    cand1 = ""
    cand2 = None
    cand3 = "Validating critical assumption"
    cand4 = "Independent grocery-shop owners experience daily shelf stockouts and spoilage."

    result = resolve_what_are_we_testing(
        candidates=[cand1, cand2, cand3, cand4],
        fallback="Fallback statement.",
    )

    assert result == "Whether independent grocery-shop owners experience daily shelf stockouts and spoilage."


def test_grounded_positive_problem_signal_included_shelfsense():
    """ShelfSense single-shop operational pain and workarounds are recognized as grounded positive signals."""
    shelfsense_state = {
        "raw_context": (
            "ShelfSense helps independent grocery store owners prevent produce and dairy spoilage. "
            "One grocery owner already reports recurring stockouts and spoilage multiple times per month. "
            "The owner considers the stockout and waste issue worth changing. "
            "Current POS workflow is insufficient for ordering decisions."
        ),
        "startup_state": {
            "execution_context": {
                "customer_access": {"value": "Cousin's grocery store", "provenance": "EXPLICIT_FOUNDER"},
                "technical_capability": {"value": True, "provenance": "EXPLICIT_FOUNDER"},
            },
            "progress": {
                "mvp_exists": {"value": False, "provenance": "UNKNOWN"},
            },
        },
        "diagnosis_snapshot": {
            "current_advantages": [
                "Direct store access through cousin's grocery store",
                "Technical team can build prototype internally",
            ]
        },
    }

    signals = compose_current_positive_signals(
        state=shelfsense_state,
        diagnosis_snapshot=shelfsense_state["diagnosis_snapshot"],
    )

    assert 1 <= len(signals) <= 4
    assert "No strong positive signal is established yet." not in signals
    assert any("stockout" in s.lower() or "spoilage" in s.lower() for s in signals)
    assert any("worth changing" in s.lower() or "worth solving" in s.lower() for s in signals)
    assert any("pos" in s.lower() or "inventory" in s.lower() for s in signals)
    assert any("cousin" in s.lower() or "access" in s.lower() for s in signals)


def test_unsupported_praise_and_budget_excluded():
    """Vague hype and financial resources are excluded from positive signals."""
    state = {
        "raw_context": "We have $5,000 budget and 40 hrs/week with massive market opportunity.",
        "startup_state": {
            "execution_context": {
                "technical_capability": {"value": None, "provenance": "UNKNOWN"},
                "customer_access": {"value": None, "provenance": "UNKNOWN"},
            }
        },
    }
    snapshot = {
        "current_advantages": [
            "$5,000 budget",
            "40 hrs/week",
            "Huge market",
            "Visionary team",
            "Strong product-market fit",
            "Validated demand",
        ]
    }

    signals = compose_current_positive_signals(state=state, diagnosis_snapshot=snapshot)
    assert signals == ["No strong positive signal is established yet."]


def test_no_positive_signal_appears_only_when_truly_empty():
    """'No strong positive signal is established yet.' is rendered ONLY when truly no grounded facts exist."""
    state_with_cap = {
        "startup_state": {
            "execution_context": {
                "technical_capability": {"value": True, "provenance": "EXPLICIT_FOUNDER"},
            }
        }
    }
    sigs1 = compose_current_positive_signals(state=state_with_cap)
    assert "No strong positive signal is established yet." not in sigs1
    assert "The team can build the prototype internally." in sigs1

    empty_state = {}
    sigs2 = compose_current_positive_signals(state=empty_state)
    assert sigs2 == ["No strong positive signal is established yet."]


def test_full_mode_remains_unchanged_and_intact():
    """Full mode preserves all methodology, actions, constraints, and completion instructions."""
    handoff = ExecutionHandoff(
        experiment_id="exp-full-check",
        assumption_id="asmp-full-check",
        critical_assumption="Independent grocery store owners make ordering decisions.",
        experiment_type="CUSTOMER_INTERVIEWS",
        learning_goal="Validate ordering authority",
        what_we_are_testing="Validating critical assumption: Store owners control supplier ordering.",
        founder_actions=[
            "Interview 5 grocery owners",
            "Inspect weekly invoice logs",
        ],
        target_participants_or_context="Independent grocery store owners",
        evidence_to_collect="Invoice logs, supplier contracts",
        support_signal="4 of 5 owners personally sign purchase orders",
        contradiction_signal="Ordering is delegated to offsite corporate distributor",
        inconclusive_signal="Unclear decision rights",
        important_constraints=["Do not pitch SaaS features during the interview"],
        valid_inferences=["Confirms local ordering authority"],
        invalid_inferences=["Does not prove software pricing willingness"],
        completion_instruction="Return to Startary with the interview notes.",
    )

    rendered = render_execution_handoff_text(handoff, mode="full")

    assert "YOUR NEXT TEST" in rendered
    assert "What we're testing:" in rendered
    assert "Whether store owners control supplier ordering." in rendered
    assert "Interview 5 grocery owners" in rendered
    assert "Inspect weekly invoice logs" in rendered
    assert "Independent grocery store owners" in rendered
    assert "Confirms assumption: 4 of 5 owners personally sign purchase orders" in rendered
    assert "Weakens/disproves: Ordering is delegated to offsite corporate distributor" in rendered
    assert "Inconclusive signal: Unclear decision rights" in rendered
    assert "Do not:" in rendered
    assert "Do not pitch SaaS features during the interview" in rendered
    assert "Does not prove software pricing willingness" in rendered
    assert "WHEN YOU'RE DONE" in rendered
    assert "Return here and tell me what actually happened." in rendered


def test_validate_founder_experiment_summary_integrity():
    """validate_founder_experiment_summary enforces strict integrity rules."""
    valid_summary = FounderExperimentSummary(
        critical_assumption="Independent grocery-shop owners experience recurring stockouts.",
        what_are_we_testing="Whether owners find ordering recommendations credible.",
        what_should_i_do="Interview 5-6 store owners.",
        what_should_i_bring_back="- Ordering notes\n- Objections",
        useful_signal="Multiple owners report pain and accept a follow-up.",
        current_positive_signals=["One owner reports recurring stockouts."],
    )
    assert validate_founder_experiment_summary(
        valid_summary,
        assumption={"statement": "Independent grocery-shop owners experience recurring stockouts."},
    ) is True

    # 1. Blank what_are_we_testing raises ValueError
    with pytest.raises(ValueError, match="what_are_we_testing"):
        invalid_testing = valid_summary.model_copy(update={"what_are_we_testing": "   "})
        validate_founder_experiment_summary(invalid_testing)

    # 2. Missing critical_assumption when active assumption exists raises ValueError
    with pytest.raises(ValueError, match="critical_assumption"):
        invalid_asmp = valid_summary.model_copy(update={"critical_assumption": None})
        validate_founder_experiment_summary(
            invalid_asmp,
            assumption={"statement": "Active assumption statement exists."},
        )

    # 3. More than 4 positive signals raises ValueError
    with pytest.raises(ValueError, match="maximum 4"):
        too_many_sigs = valid_summary.model_copy(
            update={"current_positive_signals": ["sig1", "sig2", "sig3", "sig4", "sig5"]}
        )
        validate_founder_experiment_summary(too_many_sigs)

    # 4. Forbidden term in positive signal raises ValueError
    with pytest.raises(ValueError, match="forbidden term"):
        hype_sig = valid_summary.model_copy(
            update={"current_positive_signals": ["Huge market opportunity for this product."]}
        )
        validate_founder_experiment_summary(hype_sig)
