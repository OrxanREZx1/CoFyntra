"""
Unit Tests for Milestone 4.0.1 — Integration Integrity Patch.

Verifies:
1. LearningDelta eligibility guard (skipped on Iteration 1 and when no completed evidence exists)
2. LearningDelta generated after valid evidence cycle
3. Clean state initialization on Iteration 1 (stale state protection)
4. Customer access evidence resolver (MVP testers count as access)
5. Distinguishing lack of access from sample recruitment uncertainty (NEEDS_HELP without 'no access')
6. Factual consistency guard on readiness reasons (forbidding contradictions)
7. Terminal state resolution and rendering (WAITING_FOR_EVIDENCE, NEEDS_HELP, BLOCKED)
"""

from __future__ import annotations

import pytest

from startary.domain.customer_access import (
    CustomerAccessAssessment,
    CustomerAccessStatus,
    CustomerAccessStrength,
)
from startary.domain.enums import ExecutionReadiness, WorkflowPhase
from startary.domain.terminal_state import (
    TerminalStatus,
    render_terminal_state,
    resolve_founder_facing_status,
)
from startary.graph.nodes import (
    assess_can_execute,
    can_compose_learning_delta,
    compose_learning_delta,
    initialize_state,
    resolve_customer_access,
)


# ---------------------------------------------------------------------------
# Test 1 & 2 & 3: LearningDelta Eligibility Guard
# ---------------------------------------------------------------------------

def test_learning_delta_skipped_on_iteration_1():
    """Iteration 1 must never compose a LearningDelta or log false INCONCLUSIVE."""
    state = {
        "project_id": "test-iter1",
        "iteration": 1,
        "phase": WorkflowPhase.TEST,
        "weak_area": "Problem",
        "founder_evidence": [],
        "evidence_analysis": None,
        "previous_assumptions": [],
    }
    assert not can_compose_learning_delta(state)
    res = compose_learning_delta(state)
    assert res.get("learning_delta") is None


def test_learning_delta_skipped_when_no_completed_evidence():
    """Iteration 2 without a completed evidence cycle must not compose a LearningDelta."""
    state = {
        "project_id": "test-iter2-no-ev",
        "iteration": 2,
        "phase": WorkflowPhase.LEARN,
        "weak_area": "Problem",
        "founder_evidence": [],  # Empty
        "evidence_analysis": {"conclusion": "SUPPORTED"},
        "previous_assumptions": [{"statement": "Old assumption", "area": "Problem"}],
    }
    assert not can_compose_learning_delta(state)
    res = compose_learning_delta(state)
    assert res.get("learning_delta") is None


def test_learning_delta_generated_after_valid_evidence_cycle():
    """When all 6 conditions are met after evidence intake, compose_learning_delta executes."""
    state = {
        "project_id": "test-iter2-valid",
        "iteration": 2,
        "phase": WorkflowPhase.LEARN,
        "weak_area": "Value Proposition",
        "previous_weak_area": "Problem",
        "founder_evidence": [
            {"id": "ev-1", "iteration": 1, "result": "ANALYZED"}
        ],
        "evidence_analysis": {
            "conclusion": "SUPPORTED",
            "evidence_strength": "MODERATE",
            "support_findings": ["5/7 managers confirmed problem"],
            "contradiction_findings": [],
            "inconclusive_findings": [],
            "valid_learnings": ["Managers experience cancellation pain"],
        },
        "previous_assumptions": [
            {"statement": "Café managers struggle with shift cancellations", "area": "Problem"}
        ],
        "startup_state": {
            "area_knowledge": {
                "Problem": {
                    "evidence_provided": ["Interviewed 7 café managers; 5 confirmed disruption."],
                }
            }
        },
    }
    assert can_compose_learning_delta(state)
    res = compose_learning_delta(state)
    delta = res.get("learning_delta")
    assert delta is not None
    assert delta["evidence_conclusion"] == "SUPPORTED"
    assert delta["weak_link_changed"] is True
    assert delta["previous_weak_link"] == "Problem"
    assert delta["new_weak_link"] == "Value Proposition"


def test_stale_m4_state_cleaned_on_iteration_1_init():
    """Fresh Iteration 1 state must clear any surviving or accidental M4 fields."""
    dirty_state = {
        "project_id": "clean-test",
        "iteration": 1,
        "learning_delta": {"fake": "stale_data"},
        "previous_assumptions": [{"statement": "stale"}],
        "evidence_analysis": {"conclusion": "INCONCLUSIVE"},
    }
    updates = initialize_state(dirty_state)
    assert updates["learning_delta"] is None
    assert updates["previous_assumptions"] == []
    assert updates["evidence_analysis"] is None


# ---------------------------------------------------------------------------
# Test 4 & 5 & 6: Customer Access Resolver & Can Execute Grounding
# ---------------------------------------------------------------------------

def test_historical_mvp_testers_count_as_customer_access():
    """MVP tested by students and user feedback must resolve to demonstrated customer access (LIMITED or ESTABLISHED), not NONE."""
    state = {
        "execution_context": {
            "customer_access": {"value": "UNKNOWN", "provenance": "UNKNOWN"},
        },
        "startup_progress": {
            "mvp_exists": {"value": True, "provenance": "FOUNDER_MENTIONED"},
        },
        "startup_state": {
            "area_knowledge": {
                "Target User": {
                    "understanding": "University students exploring IT careers",
                    "founder_claims": ["Students tested the early prototype."],
                },
                "Demand Signal": {
                    "founder_claims": ["Feedback from students who tried the tool was collected."],
                }
            }
        },
        "founder_messages": [
            {"role": "user", "content": "Students already tested our MVP and gave feedback on why they would not pay."}
        ],
    }

    access = resolve_customer_access(state)
    assert access.access_status in (CustomerAccessStatus.LIMITED, CustomerAccessStatus.ESTABLISHED)
    assert access.access_status != CustomerAccessStatus.NONE
    assert len(access.evidence_sources) > 0


def test_limited_access_produces_needs_help_without_saying_no_access():
    """
    When founder has demonstrated past testers but experiment requires recruiting 8 qualifying students,
    system should conclude NEEDS_HELP while recognizing the existing access.
    """
    state = {
        "project_id": "mynera-test",
        "active_experiment": {
            "experiment_type": "PROBLEM_INTERVIEW",
            "required_access": ["students who recently changed IT specialization"],
            "sample_size": "8 students",
            "estimated_cost": "$0",
        },
        "execution_context": {
            "available_budget_amount": {"value": 100, "provenance": "FOUNDER_REPORTED"},
        },
        "startup_progress": {
            "mvp_exists": {"value": True},
        },
        "founder_messages": [
            {"role": "user", "content": "Our MVP was tested by university students and they gave us feedback."}
        ],
        "startup_state": {
            "area_knowledge": {
                "Target User": {
                    "understanding": "Students",
                    "founder_claims": ["Tested with students"],
                }
            }
        }
    }

    res = assess_can_execute(state)
    can_exec = res["can_execute_result"]
    assert can_exec["readiness"] == ExecutionReadiness.NEEDS_HELP.value

    reason = can_exec["reason"].lower()
    # Must NOT say founder has no customer access
    assert "founder has no established customer access" not in reason
    assert "founder has no demonstrated customer access" not in reason
    # Must acknowledge existing access
    assert "existing customer access is demonstrated" in reason or "existing access is limited" in reason


def test_readiness_reason_cannot_contradict_known_access():
    """Factual consistency guard: reason cannot claim 'no access' if access evidence exists."""
    state = {
        "project_id": "test-contradiction-guard",
        "active_experiment": {
            "experiment_type": "PROBLEM_INTERVIEW",
            "required_access": ["Independent café managers"],
            "sample_size": "6 managers",
            "estimated_cost": "$0",
        },
        "execution_context": {
            "customer_access": {"value": "Interviewed 5 café managers previously", "provenance": "FOUNDER_REPORTED"},
            "available_budget_amount": {"value": 500, "provenance": "FOUNDER_REPORTED"},
        },
        "startup_progress": {
            "mvp_exists": {"value": False},
        },
        "startup_state": {"area_knowledge": {}},
    }

    res = assess_can_execute(state)
    reason = res["can_execute_result"]["reason"]
    assert "no established customer access" not in reason
    assert "no demonstrated customer access" not in reason


# ---------------------------------------------------------------------------
# Test 7 & 8: Centralized Terminal State Rendering
# ---------------------------------------------------------------------------

def test_final_cli_heading_for_needs_help():
    """NEEDS_HELP state renders 'STARTARY — EXPERIMENT READY, SUPPORT NEEDED'."""
    state = {
        "phase": WorkflowPhase.TEST,
        "can_execute_result": {
            "readiness": "NEEDS_HELP",
            "reason": "Existing access is demonstrated, but sample recruitment requires assistance.",
        },
        "status_message": "Old clarification limit message that should be ignored",
    }
    status = resolve_founder_facing_status(state)
    assert status == TerminalStatus.NEEDS_HELP

    rendered = render_terminal_state(state)
    assert "STARTARY — EXPERIMENT READY, SUPPORT NEEDED" in rendered
    assert "CONTEXT CLARIFICATION LIMIT REACHED" not in rendered
    assert "Existing access is demonstrated" in rendered


def test_final_cli_heading_for_waiting_for_evidence():
    """WAITING_FOR_EVIDENCE renders 'STARTARY — WAITING FOR EVIDENCE'."""
    exp_id = "exp-integ-test-01"
    state = {
        "phase": WorkflowPhase.WAITING_FOR_EVIDENCE,
        "active_experiment": {
            "id": exp_id,
            "status": "IN_PROGRESS",
        },
        "active_execution_handoff": {
            "experiment_id": exp_id,
        },
        "can_execute_result": {
            "readiness": "CAN_EXECUTE",
            "reason": "Ready to execute.",
        },
    }
    status = resolve_founder_facing_status(state)
    assert status == TerminalStatus.WAITING_FOR_EVIDENCE

    rendered = render_terminal_state(state)
    assert "STARTARY — WAITING FOR EVIDENCE" in rendered
    assert "Run it in the real world and return with what happened." in rendered

