"""
Tests: Milestone 3 Execution Readiness (Can Execute?) Evaluator.

Verification:
- Returns CAN_EXECUTE when founder capacity, access, and budget are sufficient.
- Returns NEEDS_HELP when prototype test is proposed but mvp_exists is False.
- Returns NEEDS_HELP when customer access is required but founder has no established access.
- Returns BLOCKED when estimated costs exceed zero available budget.
"""

import pytest

from startary.domain.enums import ExecutionReadiness, ExperimentType
from startary.graph.nodes import assess_can_execute
from startary.graph.graph_state import StartaryGraphState


@pytest.fixture
def can_execute_base_state() -> StartaryGraphState:
    return {
        "project_id": "proj-exec-123",
        "active_experiment": {
            "experiment_type": ExperimentType.PROBLEM_INTERVIEW.value,
            "objective": "Test problem severity",
            "required_access": ["Local store owners"],
            "estimated_cost": "Near-zero / $0",
            "estimated_time": "3-5 days",
        },
        "execution_context": {
            "customer_access": {"value": "Personal contacts with 10 store owners", "provenance": "FOUNDER_STATED"},
            "available_budget_amount": {"value": 200, "provenance": "FOUNDER_STATED"},
            "weekly_capacity": {"value": "20 hrs/week", "provenance": "FOUNDER_STATED"},
        },
        "startup_progress": {
            "mvp_exists": {"value": False, "provenance": "FOUNDER_STATED"},
            "development_stage": {"value": "idea", "provenance": "FOUNDER_STATED"},
        },
    }


class TestCanExecuteEvaluator:
    def test_can_execute_happy_path(self, can_execute_base_state):
        res = assess_can_execute(can_execute_base_state)
        assessment = res["can_execute_result"]
        assert assessment["readiness"] == ExecutionReadiness.CAN_EXECUTE
        assert len(assessment["missing_requirements"]) == 0
        assert len(assessment["blockers"]) == 0

    def test_prototype_test_without_mvp_yields_needs_help(self, can_execute_base_state):
        can_execute_base_state["active_experiment"]["experiment_type"] = ExperimentType.PROTOTYPE_TEST.value
        can_execute_base_state["startup_progress"]["mvp_exists"]["value"] = False

        res = assess_can_execute(can_execute_base_state)
        assessment = res["can_execute_result"]
        assert assessment["readiness"] == ExecutionReadiness.NEEDS_HELP
        assert any("working prototype" in req for req in assessment["missing_requirements"])

    def test_missing_customer_access_yields_needs_help(self, can_execute_base_state):
        can_execute_base_state["execution_context"]["customer_access"] = {"value": "None", "provenance": "FOUNDER_STATED"}
        can_execute_base_state["execution_context"]["existing_audience_or_community"] = {"value": "None", "provenance": "FOUNDER_STATED"}

        res = assess_can_execute(can_execute_base_state)
        assessment = res["can_execute_result"]
        assert assessment["readiness"] == ExecutionReadiness.NEEDS_HELP
        assert any("customer" in req.lower() for req in assessment["missing_requirements"])

    def test_expensive_test_with_zero_budget_yields_blocked(self, can_execute_base_state):
        can_execute_base_state["active_experiment"]["estimated_cost"] = "$500 ad spend"
        can_execute_base_state["execution_context"]["available_budget_amount"] = {"value": 0, "provenance": "FOUNDER_STATED"}

        res = assess_can_execute(can_execute_base_state)
        assessment = res["can_execute_result"]
        assert assessment["readiness"] == ExecutionReadiness.BLOCKED
        assert any("budget" in b.lower() for b in assessment["blockers"])

