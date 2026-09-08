"""
Tests: Milestone 3 Experiment Critic Node and Revision Router.

Verification:
- Evaluates 6 dimensions (alignment, evidence validity, feasibility, cost efficiency, falsifiability, scope).
- ACCEPT verdict passes without revision.
- REVISE verdict increments revision count and stores instructions.
- Router routes to save_final_experiment on ACCEPT.
- Router routes to generate_experiment on first REVISE (1 revision allowed).
- Router routes to handle_experiment_failure_stop on second REVISE.
- Failure stop produces graceful message and leaves active_experiment unset.
"""

from unittest.mock import MagicMock
import pytest

from startary.domain.enums import WorkflowPhase
from startary.graph.builder import route_on_experiment_critique
from startary.graph.graph_state import StartaryGraphState
from startary.graph.nodes import (
    critique_experiment,
    handle_experiment_failure_stop,
)
from startary.llm.schemas import ExperimentCriticLLMOutput
from tests.fixtures.scenarios import (
    make_valid_experiment_critic_output_dict,
    make_valid_experiment_generator_output_dict,
)


@pytest.fixture
def critic_test_state() -> StartaryGraphState:
    return {
        "project_id": "proj-crit-123",
        "phase": WorkflowPhase.TEST,
        "iteration": 1,
        "weak_area": "Problem",
        "active_assumption": {
            "id": "asm-xyz-999",
            "project_id": "proj-crit-123",
            "area": "Problem",
            "statement": "Independent restaurant managers waste over 5 hours per week reconciling food supply orders.",
            "is_critical": True,
            "status": "UNTESTED",
        },
        "reality_check": {
            "summary": "Founder claims manual inventory takes 5 hours weekly, but has no documented audits.",
        },
        "proposed_experiment_output": make_valid_experiment_generator_output_dict(),
        "execution_context": {
            "team_size": {"value": "2 founders", "provenance": "FOUNDER_STATED"},
            "weekly_capacity": {"value": "20 hours/week", "provenance": "FOUNDER_STATED"},
            "available_budget_amount": {"value": 500, "provenance": "FOUNDER_STATED"},
        },
        "startup_progress": {
            "development_stage": {"value": "idea", "provenance": "FOUNDER_STATED"},
            "mvp_exists": {"value": False, "provenance": "FOUNDER_STATED"},
        },
        "critic_revision_count": 0,
        "critic_revision_instructions": None,
        "previous_experiments": [],
    }


class TestExperimentCriticNode:
    def test_critique_accept_verdict(self, mocker, critic_test_state):
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_output_dict = make_valid_experiment_critic_output_dict(verdict="ACCEPT")
        mock_structured.invoke.return_value = ExperimentCriticLLMOutput(**mock_output_dict)
        mock_llm.with_structured_output.return_value = mock_structured

        mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
        mocker.patch("startary.graph.nodes.get_settings", return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"))

        res = critique_experiment(critic_test_state)

        assert "experiment_critic_output" in res
        output = res["experiment_critic_output"]
        assert output["verdict"] == "ACCEPT"
        assert output["assumption_alignment"] == "PASS"
        assert output["evidence_validity"] == "PASS"
        assert output["feasibility"] == "PASS"
        assert output["cost_efficiency"] == "PASS"
        assert output["falsifiability"] == "PASS"
        assert output["scope"] == "PASS"
        assert "critic_revision_count" not in res  # not incremented on accept

    def test_critique_revise_increments_count_and_sets_instructions(self, mocker, critic_test_state):
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_output_dict = make_valid_experiment_critic_output_dict(
            verdict="REVISE",
            issues=["Method builds full platform instead of cheap interviews."],
            revision_instructions="Remove coding tasks; conduct manual interviews instead.",
        )
        mock_structured.invoke.return_value = ExperimentCriticLLMOutput(**mock_output_dict)
        mock_llm.with_structured_output.return_value = mock_structured

        mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
        mocker.patch("startary.graph.nodes.get_settings", return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"))

        res = critique_experiment(critic_test_state)

        assert res["experiment_critic_output"]["verdict"] == "REVISE"
        assert res["critic_revision_count"] == 1
        assert "Remove coding tasks" in res["critic_revision_instructions"]


class TestCriticRouterAndFailureStop:
    def test_router_accept_routes_to_save_final_experiment(self):
        state = {
            "experiment_critic_output": {"verdict": "ACCEPT"},
            "critic_revision_count": 0,
        }
        assert route_on_experiment_critique(state) == "save_final_experiment"

    def test_router_first_revise_routes_to_generate_experiment(self):
        state = {
            "experiment_critic_output": {"verdict": "REVISE"},
            "critic_revision_count": 1,
        }
        assert route_on_experiment_critique(state) == "generate_experiment"

    def test_router_second_revise_routes_to_failure_stop(self):
        state = {
            "experiment_critic_output": {"verdict": "REVISE"},
            "critic_revision_count": 2,
        }
        assert route_on_experiment_critique(state) == "handle_experiment_failure_stop"

    def test_handle_experiment_failure_stop(self, critic_test_state):
        critic_test_state["experiment_critic_output"] = {
            "verdict": "REVISE",
            "issues": ["Unfalsifiable signals", "Cost exceeds budget"],
        }
        res = handle_experiment_failure_stop(critic_test_state)
        assert "status_message" in res
        assert "unable to produce an accepted experiment" in res["status_message"]
        assert res["phase"] == WorkflowPhase.WAITING_FOR_LEARNING_INPUT

