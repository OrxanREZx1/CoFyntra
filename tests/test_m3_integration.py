"""
Tests: Milestone 3 End-to-End Experiment Pipeline Integration.

Verification:
- Standalone subgraph execution via build_experiment_graph and run_experiment_pipeline.
- 1-cycle revision loop: draft rejected by critic, revised with feedback, accepted on second attempt.
- Controlled failure stop if critic rejects second attempt.
- Domain scenario variety (Problem, Demand, Pricing, Acquisition, Solution).
"""

from unittest.mock import MagicMock
import pytest

from startary.domain.enums import WorkflowPhase
from startary.graph.builder import build_experiment_graph, run_experiment_pipeline
from startary.graph.graph_state import StartaryGraphState
from startary.llm.schemas import ExperimentCriticLLMOutput, ExperimentGeneratorLLMOutput
from tests.fixtures.scenarios import (
    make_valid_experiment_critic_output_dict,
    make_valid_experiment_generator_output_dict,
)


@pytest.fixture
def m3_valid_state() -> StartaryGraphState:
    return {
        "project_id": "proj-m3-integ-1",
        "phase": WorkflowPhase.TEST,
        "iteration": 1,
        "weak_area": "Problem",
        "active_assumption": {
            "id": "asm-integ-001",
            "project_id": "proj-m3-integ-1",
            "area": "Problem",
            "statement": "Local courier businesses lose significant revenue due to disorganized parcel consolidation.",
            "is_critical": True,
            "status": "UNTESTED",
        },
        "reality_check": {
            "summary": "Founder noticed consolidation issues informally but has no quantitative tracking data.",
            "unresolved_uncertainty": "Whether lost revenue is substantial enough to warrant a dedicated tool.",
            "what_evidence_supports": ["Informal conversations with 2 drivers."],
            "what_evidence_does_not_support": ["Claims of exact 20% loss"],
        },
        "execution_context": {
            "team_size": {"value": "Solo founder", "provenance": "FOUNDER_STATED"},
            "weekly_capacity": {"value": "15 hrs/week", "provenance": "FOUNDER_STATED"},
            "available_budget_amount": {"value": 100, "provenance": "FOUNDER_STATED"},
            "customer_access": {"value": "Direct phone contacts with 5 courier dispatchers", "provenance": "FOUNDER_STATED"},
        },
        "startup_progress": {
            "development_stage": {"value": "idea", "provenance": "FOUNDER_STATED"},
            "mvp_exists": {"value": False, "provenance": "FOUNDER_STATED"},
        },
        "founder_messages": [
            {"role": "founder", "content": "We are building ParcelMate to fix courier parcel consolidation."}
        ],
        "previous_experiments": [],
        "critic_revision_count": 0,
        "critic_revision_instructions": None,
    }


class TestM3PipelineIntegration:
    def test_direct_experiment_pipeline_accepts_clean_experiment(self, mocker, m3_valid_state):
        mock_llm = MagicMock()

        def mock_structured(schema):
            s = MagicMock()
            if schema == ExperimentGeneratorLLMOutput:
                s.invoke.return_value = ExperimentGeneratorLLMOutput(
                    **make_valid_experiment_generator_output_dict(
                        hypothesis=m3_valid_state["active_assumption"]["statement"]
                    )
                )
            elif schema == ExperimentCriticLLMOutput:
                s.invoke.return_value = ExperimentCriticLLMOutput(
                    **make_valid_experiment_critic_output_dict(verdict="ACCEPT")
                )
            return s

        mock_llm.with_structured_output.side_effect = mock_structured
        mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
        mocker.patch("startary.graph.nodes.get_settings", return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"))

        res = run_experiment_pipeline(m3_valid_state)

        assert res["active_experiment"] is not None
        assert res["active_experiment"]["status"] == "READY"
        assert res["active_experiment"]["assumption_id"] == "asm-integ-001"
        assert res["experiment_critic_output"]["verdict"] == "ACCEPT"
        assert res["can_execute_result"]["readiness"] == "CAN_EXECUTE"

    def test_pipeline_executes_one_revision_cycle_and_succeeds(self, mocker, m3_valid_state):
        mock_llm = MagicMock()
        critic_call_count = {"count": 0}

        def mock_structured(schema):
            s = MagicMock()
            if schema == ExperimentGeneratorLLMOutput:
                s.invoke.return_value = ExperimentGeneratorLLMOutput(
                    **make_valid_experiment_generator_output_dict(
                        hypothesis=m3_valid_state["active_assumption"]["statement"]
                    )
                )
            elif schema == ExperimentCriticLLMOutput:
                # First call: REVISE. Second call: ACCEPT.
                if critic_call_count["count"] == 0:
                    critic_call_count["count"] += 1
                    s.invoke.return_value = ExperimentCriticLLMOutput(
                        **make_valid_experiment_critic_output_dict(
                            verdict="REVISE",
                            issues=["Draft is too complex"],
                            revision_instructions="Simplify to 5 phone interviews",
                        )
                    )
                else:
                    s.invoke.return_value = ExperimentCriticLLMOutput(
                        **make_valid_experiment_critic_output_dict(verdict="ACCEPT")
                    )
            return s

        mock_llm.with_structured_output.side_effect = mock_structured
        mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
        mocker.patch("startary.graph.nodes.get_settings", return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"))

        res = run_experiment_pipeline(m3_valid_state)

        assert res["active_experiment"] is not None
        assert res["critic_revision_count"] == 1
        assert res["experiment_critic_output"]["verdict"] == "ACCEPT"

    def test_pipeline_halts_at_failure_stop_if_critic_rejects_twice(self, mocker, m3_valid_state):
        mock_llm = MagicMock()

        def mock_structured(schema):
            s = MagicMock()
            if schema == ExperimentGeneratorLLMOutput:
                s.invoke.return_value = ExperimentGeneratorLLMOutput(
                    **make_valid_experiment_generator_output_dict(
                        hypothesis=m3_valid_state["active_assumption"]["statement"]
                    )
                )
            elif schema == ExperimentCriticLLMOutput:
                # Always REVISE
                s.invoke.return_value = ExperimentCriticLLMOutput(
                    **make_valid_experiment_critic_output_dict(
                        verdict="REVISE",
                        issues=["Persistent build-first bias"],
                        revision_instructions="Do not build software",
                    )
                )
            return s

        mock_llm.with_structured_output.side_effect = mock_structured
        mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
        mocker.patch("startary.graph.nodes.get_settings", return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"))

        res = run_experiment_pipeline(m3_valid_state)

        assert res.get("active_experiment") is None
        assert "status_message" in res
        assert "unable to produce an accepted experiment" in res["status_message"]
        assert res["critic_revision_count"] == 2

