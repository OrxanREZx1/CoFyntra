"""
Tests: Milestone 3 Experiment Generator Node and Validator.

Verification:
- Single experiment draft generated with active assumption as hypothesis.
- Rejection if active assumption is missing.
- Negative constraints and critic revision instructions integrated in user prompt.
- Deterministic validator checks (objective, method, signals, inferences, actions).
- Normalization of hypothesis.
- Parameter provenance tracking and fake precision detection.
- Deterministic saving as ExperimentStatus.READY linked to assumption_id.
"""

from unittest.mock import MagicMock
import pytest

from startary.domain.enums import (
    ExperimentStatus,
    ExperimentType,
    ParameterProvenance,
    WorkflowPhase,
)
from startary.graph.graph_state import StartaryGraphState
from startary.graph.nodes import (
    generate_experiment,
    save_final_experiment,
    validate_proposed_experiment,
)
from startary.llm.schemas import ExperimentGeneratorLLMOutput
from tests.fixtures.scenarios import make_valid_experiment_generator_output_dict


@pytest.fixture
def base_test_state() -> StartaryGraphState:
    return {
        "project_id": "proj-gen-123",
        "phase": WorkflowPhase.TEST,
        "iteration": 1,
        "weak_area": "Problem",
        "active_assumption": {
            "id": "asm-xyz-999",
            "project_id": "proj-gen-123",
            "area": "Problem",
            "statement": "Independent restaurant managers waste over 5 hours per week reconciling food supply orders.",
            "is_critical": True,
            "status": "UNTESTED",
        },
        "reality_check": {
            "summary": "Founder claims manual inventory takes 5 hours weekly, but has no documented customer audits.",
            "unresolved_uncertainty": "Exact hours wasted and whether this friction is top-of-mind.",
            "what_evidence_supports": ["Founder noticed manual clipboards in 2 restaurants."],
            "what_evidence_does_not_support": ["Claims of exact 5 hours wasted", "Claims that managers are actively looking for AI"],
        },
        "execution_context": {
            "team_size": {"value": "2 founders", "provenance": "FOUNDER_STATED"},
            "weekly_capacity": {"value": "20 hours/week", "provenance": "FOUNDER_STATED"},
            "available_budget_amount": {"value": 500, "provenance": "FOUNDER_STATED"},
            "customer_access": {"value": "Personal contacts with 3 local restaurant owners", "provenance": "FOUNDER_STATED"},
        },
        "startup_progress": {
            "development_stage": {"value": "idea", "provenance": "FOUNDER_STATED"},
            "mvp_exists": {"value": False, "provenance": "FOUNDER_STATED"},
        },
        "founder_messages": [
            {"role": "founder", "content": "We want to solve inventory reconciliation for restaurants."}
        ],
        "previous_experiments": [],
        "critic_revision_count": 0,
        "critic_revision_instructions": None,
    }


class TestExperimentGeneratorNode:
    def test_generate_experiment_invokes_llm_and_returns_output(self, mocker, base_test_state):
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_output_dict = make_valid_experiment_generator_output_dict(
            hypothesis=base_test_state["active_assumption"]["statement"]
        )
        mock_structured.invoke.return_value = ExperimentGeneratorLLMOutput(**mock_output_dict)
        mock_llm.with_structured_output.return_value = mock_structured

        mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
        mocker.patch("startary.graph.nodes.get_settings", return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"))

        res = generate_experiment(base_test_state)

        assert "proposed_experiment_output" in res
        output = res["proposed_experiment_output"]
        assert output["experiment_type"] == "PROBLEM_INTERVIEW"
        assert output["hypothesis"] == base_test_state["active_assumption"]["statement"]
        assert len(output["founder_actions"]) >= 1
        assert len(output["valid_inferences"]) >= 1
        assert len(output["invalid_inferences"]) >= 1
        assert "last_trace" in res

    def test_generate_experiment_fails_if_active_assumption_missing(self, base_test_state):
        base_test_state["active_assumption"] = None
        with pytest.raises(ValueError, match="active_assumption missing"):
            generate_experiment(base_test_state)

    def test_generate_experiment_includes_critic_instructions_if_revising(self, mocker, base_test_state):
        base_test_state["critic_revision_instructions"] = "Do not build an app; test with manual interviews only."
        base_test_state["critic_revision_count"] = 1

        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_output_dict = make_valid_experiment_generator_output_dict(
            hypothesis=base_test_state["active_assumption"]["statement"]
        )
        mock_structured.invoke.return_value = ExperimentGeneratorLLMOutput(**mock_output_dict)
        mock_llm.with_structured_output.return_value = mock_structured

        mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
        mocker.patch("startary.graph.nodes.get_settings", return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"))

        generate_experiment(base_test_state)

        called_messages = mock_structured.invoke.call_args[0][0]
        user_msg = called_messages[1]["content"]
        assert "CRITIC REVISION INSTRUCTIONS" in user_msg
        assert "Do not build an app; test with manual interviews only." in user_msg


class TestValidateProposedExperiment:
    def test_validator_accepts_valid_experiment(self, base_test_state):
        base_test_state["proposed_experiment_output"] = make_valid_experiment_generator_output_dict(
            hypothesis=base_test_state["active_assumption"]["statement"]
        )
        res = validate_proposed_experiment(base_test_state)
        assert "proposed_experiment_output" in res

    # Milestone 6.2.5: structural draft defects no longer raise (a raise aborted the graph and /retry
    # re-validated the same committed draft forever). The validator passes the draft through and the
    # critic's deterministic contract check turns the defect into a bounded REVISE without an LLM call.
    @pytest.mark.parametrize("field, empty_value", [
        ("objective", ""),
        ("founder_actions", []),
        ("valid_inferences", []),
        ("support_signal", "   "),
    ])
    def test_validator_defect_becomes_deterministic_revise(self, base_test_state, field, empty_value):
        from unittest.mock import patch
        from startary.graph.nodes import critique_experiment

        exp = make_valid_experiment_generator_output_dict()
        exp[field] = empty_value
        base_test_state["proposed_experiment_output"] = exp
        validated = validate_proposed_experiment(base_test_state)  # does not raise
        state = {**base_test_state, **validated}
        with patch("startary.graph.nodes.get_llm_client", side_effect=AssertionError("critic LLM must not be called")):
            res = critique_experiment(state)
        assert res["experiment_critic_output"]["verdict"] == "REVISE"
        assert res["experiment_critic_output"]["contract_violation_codes"] == ["REQUIRED_FIELD_EMPTY"]
        assert field in res["critic_revision_instructions"]
        assert res["critic_revision_count"] == 1

    def test_validator_normalizes_divergent_hypothesis(self, base_test_state):
        exp = make_valid_experiment_generator_output_dict()
        exp["hypothesis"] = "Slightly different phrasing"
        base_test_state["proposed_experiment_output"] = exp

        res = validate_proposed_experiment(base_test_state)
        assert res["proposed_experiment_output"]["hypothesis"] == base_test_state["active_assumption"]["statement"]


class TestSaveFinalExperimentNode:
    def test_save_final_experiment_instantiates_domain_model(self, base_test_state):
        base_test_state["proposed_experiment_output"] = make_valid_experiment_generator_output_dict(
            hypothesis=base_test_state["active_assumption"]["statement"]
        )
        res = save_final_experiment(base_test_state)

        assert "active_experiment" in res
        exp = res["active_experiment"]
        assert exp["assumption_id"] == "asm-xyz-999"
        assert exp["status"] == ExperimentStatus.READY
        assert exp["experiment_type"] == ExperimentType.PROBLEM_INTERVIEW
        assert exp["hypothesis"] == base_test_state["active_assumption"]["statement"]
        assert len(res["previous_experiments"]) == 1
        assert res["phase"] == WorkflowPhase.TEST

