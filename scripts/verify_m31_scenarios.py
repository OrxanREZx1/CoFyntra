"""
Verification script for Milestone 3.1: Founder Diagnosis Snapshot Scenarios.

Tests two live scenarios with real LLM:
  Scenario A: Shiftly (Early Build / Pre-MVP, Solo founder, Weak link: Problem)
  Scenario B: PulseFlow (Live MVP, 35 active beta users, Weak link: Acquisition)
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from startary.cli import _print_diagnosis_snapshot
from startary.config import get_settings
from startary.domain.enums import WorkflowPhase
from startary.graph.builder import run_diagnosis_snapshot
from startary.graph.graph_state import StartaryGraphState


def test_scenario_a_shiftly():
    print("\n" + "=" * 70)
    print("RUNNING SCENARIO A: SHIFTLY (PROBLEM WEAK LINK)")
    print("=" * 70)

    state: StartaryGraphState = {
        "project_id": "test-shiftly-m31",
        "phase": WorkflowPhase.UNDERSTAND,
        "iteration": 1,
        "weak_area": "Problem",
        "selection_confidence": "LOW",
        "active_assumption": {
            "id": "ca-shiftly-001",
            "project_id": "test-shiftly-m31",
            "area": "Problem",
            "statement": "Small independent café and restaurant managers experience recurring operational disruption from staff cancellations or shift swaps that their current coordination processes fail to handle.",
            "is_critical": True,
            "status": "UNTESTED",
        },
        "reality_check": {
            "summary": "One café owner confirmed that cancellations and no-shows create operational strain, which provides an initial problem signal. However, this evidence comes from only one business and does not yet establish whether informal scheduling workflows are the main cause or whether the problem generalizes across other cafés.",
            "unresolved_uncertainty": "Whether scheduling disruption is widespread and costly across other independent cafés.",
            "what_evidence_supports": ["One café manager described repeated scheduling disruption."],
            "what_evidence_does_not_support": ["Claims that managers actively seek mobile apps for shift swaps"],
        },
        "area_assessments": [
            {"area": "Problem", "calculated_decision_risk": 81},
            {"area": "Demand Signal", "calculated_decision_risk": 81},
            {"area": "Target User", "calculated_decision_risk": 81},
            {"area": "Value Proposition", "calculated_decision_risk": 72},
            {"area": "Acquisition", "calculated_decision_risk": 54},
        ],
        "execution_context": {
            "team_size": {"value": "Solo founder", "provenance": "FOUNDER_STATED"},
            "total_weekly_capacity_hours": {"value": 19, "provenance": "FOUNDER_STATED"},
            "available_budget_amount": {"value": 200, "provenance": "FOUNDER_STATED"},
            "customer_access": {"value": "Direct access to 1 café owner", "provenance": "FOUNDER_STATED"},
        },
        "startup_progress": {
            "development_stage": {"value": "idea", "provenance": "FOUNDER_STATED"},
            "mvp_exists": {"value": False, "provenance": "FOUNDER_STATED"},
        },
        "founder_messages": [
            {"role": "founder", "content": "Shiftly helps café managers manage shift changes easily."}
        ],
        "previous_experiments": [],
        "traces": [],
    }

    result = run_diagnosis_snapshot(state)
    _print_diagnosis_snapshot(result)
    return result


def test_scenario_b_contrasting():
    print("\n" + "=" * 70)
    print("RUNNING SCENARIO B: PULSEFLOW (ACQUISITION WEAK LINK, LIVE MVP)")
    print("=" * 70)

    state: StartaryGraphState = {
        "project_id": "test-pulseflow-m31",
        "phase": WorkflowPhase.UNDERSTAND,
        "iteration": 1,
        "weak_area": "Acquisition",
        "selection_confidence": "HIGH",
        "active_assumption": {
            "id": "ca-pulseflow-001",
            "project_id": "test-pulseflow-m31",
            "area": "Acquisition",
            "statement": "B2B engineering team leads can be acquired cost-effectively through targeted LinkedIn cold outreach and tech community discussions without a dedicated sales team.",
            "is_critical": True,
            "status": "UNTESTED",
        },
        "reality_check": {
            "summary": "PulseFlow has a working web MVP with 35 active engineering managers who report weekly satisfaction. However, all current users were recruited from the founders' immediate former colleagues, with zero repeatable customer acquisition channels established.",
            "unresolved_uncertainty": "Whether team leads outside the founders' personal network will convert from cold outreach.",
            "what_evidence_supports": ["35 active weekly users on working MVP", "Positive retention among initial cohort"],
            "what_evidence_does_not_support": ["Claims of scalable inbound organic growth"],
        },
        "area_assessments": [
            {"area": "Acquisition", "calculated_decision_risk": 81},
            {"area": "Business Model", "calculated_decision_risk": 54},
            {"area": "Problem", "calculated_decision_risk": 27},
            {"area": "Solution / MVP", "calculated_decision_risk": 18},
        ],
        "execution_context": {
            "team_size": {"value": "2 technical co-founders", "provenance": "FOUNDER_STATED"},
            "total_weekly_capacity_hours": {"value": 40, "provenance": "FOUNDER_STATED"},
            "available_budget_amount": {"value": 1500, "provenance": "FOUNDER_STATED"},
            "customer_access": {"value": "Exhausted personal network; no established cold channel", "provenance": "FOUNDER_STATED"},
        },
        "startup_progress": {
            "development_stage": {"value": "early_traction", "provenance": "FOUNDER_STATED"},
            "mvp_exists": {"value": True, "provenance": "FOUNDER_STATED"},
        },
        "founder_messages": [
            {"role": "founder", "content": "PulseFlow has 35 active teams using our MVP, but we don't know how to reach stranger teams."}
        ],
        "previous_experiments": [],
        "traces": [],
    }

    result = run_diagnosis_snapshot(state)
    _print_diagnosis_snapshot(result)
    return result


if __name__ == "__main__":
    test_scenario_a_shiftly()
    test_scenario_b_contrasting()

