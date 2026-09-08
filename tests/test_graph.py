"""
Tests: LangGraph graph compilation and execution — updated for Milestone 2.6.

M2.6 graph flow:
  initialize_state
  → verify_state
  → extract_startup_state (LLM — M2.6)
  → merge_execution_state (Deterministic — M2.6)
  → diagnose_startup (LLM)
  → merge_diagnosis (Deterministic)
  → plan_information_gap (LLM — M2.6)
  → assess_startup_areas (LLM)
  → select_weak_link (Deterministic)
  → reality_check (LLM)
  → generate_critical_assumption (LLM)
  → save_critical_assumption (Deterministic)
  → END

All tests mock LLM responses based on the bound schema.
"""

import os
import uuid
from unittest.mock import MagicMock

import pytest

from startary.domain.enums import WorkflowPhase
from startary.domain.taxonomy import StartupArea
from startary.graph.builder import build_graph
from startary.graph.graph_state import StartaryGraphState
from startary.llm.schemas import (
    AdvisorResponseLLMOutput,
    BaselineQuestionPlanLLMOutput,
    BaselineSufficiencyLLMOutput,
    CriticalAssumptionLLMOutput,
    DiagnosisLLMOutput,
    ExperimentCriticLLMOutput,
    ExperimentGeneratorLLMOutput,
    FounderTurnInterpretationLLMOutput,
    InformationGapPlanLLMOutput,
    RealityCheckLLMOutput,
    StartupAreaAssessmentBatchLLMOutput,
    StartupDiagnosisSnapshotLLMOutput,
    StartupStateDeltaLLMOutput,
    WeakLinkClarifyingQuestionLLMOutput,
    WeakLinkDiscriminationLLMOutput,
)
from tests.fixtures.scenarios import (
    make_minimal_valid_llm_output_dict,
    make_valid_advisor_response_output_dict,
    make_valid_area_assessments_batch_output_dict,
    make_valid_critical_assumption_output_dict,
    make_valid_diagnosis_snapshot_output_dict,
    make_valid_experiment_critic_output_dict,
    make_valid_experiment_generator_output_dict,
    make_valid_founder_turn_interpretation_output_dict,
    make_valid_gap_plan_output_dict,
    make_valid_reality_check_output_dict,
)


def _project_id() -> str:
    return str(uuid.uuid4())


def _make_initial_state(
    project_id: str | None = None,
    raw_context: str = "I want to make an AI app for restaurants.",
) -> StartaryGraphState:
    return StartaryGraphState(
        project_id=project_id or _project_id(),
        phase=WorkflowPhase.UNDERSTAND,
        iteration=1,
        raw_context=raw_context,
        founder_evidence=[],
        external_evidence=[],
        traces=[],
    )


def _patch_all_m26_llms(mocker):
    """
    Patch get_llm_client to return structured outputs corresponding to whichever
    schema is passed to with_structured_output. Supports M2.6 schemas.
    """
    mock_llm = MagicMock()

    def mock_with_structured_output(schema):
        structured = MagicMock()
        if schema == DiagnosisLLMOutput:
            structured.invoke.return_value = DiagnosisLLMOutput(**make_minimal_valid_llm_output_dict())
        elif schema == StartupStateDeltaLLMOutput:
            # Empty delta — no execution context extracted
            structured.invoke.return_value = StartupStateDeltaLLMOutput()
        elif schema == InformationGapPlanLLMOutput:
            # Planner says: sufficient, don't ask
            structured.invoke.return_value = InformationGapPlanLLMOutput(
                **make_valid_gap_plan_output_dict(should_ask=False, strategic_sufficient=True)
            )
        elif schema == StartupAreaAssessmentBatchLLMOutput:
            structured.invoke.return_value = StartupAreaAssessmentBatchLLMOutput(
                **make_valid_area_assessments_batch_output_dict(StartupArea.PROBLEM)
            )
        elif schema == RealityCheckLLMOutput:
            structured.invoke.return_value = RealityCheckLLMOutput(
                **make_valid_reality_check_output_dict(StartupArea.PROBLEM)
            )
        elif schema == CriticalAssumptionLLMOutput:
            structured.invoke.return_value = CriticalAssumptionLLMOutput(
                **make_valid_critical_assumption_output_dict(StartupArea.PROBLEM)
            )
        elif schema == BaselineSufficiencyLLMOutput:
            from tests.fixtures.scenarios import make_valid_baseline_sufficiency_output_dict
            structured.invoke.return_value = BaselineSufficiencyLLMOutput(
                **make_valid_baseline_sufficiency_output_dict(is_sufficient=True)
            )
        elif schema == BaselineQuestionPlanLLMOutput:
            from tests.fixtures.scenarios import make_valid_baseline_question_plan_output_dict
            structured.invoke.return_value = BaselineQuestionPlanLLMOutput(
                **make_valid_baseline_question_plan_output_dict(should_ask=False)
            )
        elif schema == WeakLinkDiscriminationLLMOutput:
            from tests.fixtures.scenarios import make_valid_weak_link_discrimination_output_dict
            structured.invoke.return_value = WeakLinkDiscriminationLLMOutput(
                **make_valid_weak_link_discrimination_output_dict(selection_confidence="HIGH", should_clarify=False)
            )
        elif schema == WeakLinkClarifyingQuestionLLMOutput:
            from tests.fixtures.scenarios import make_valid_weak_link_clarifying_question_output_dict
            structured.invoke.return_value = WeakLinkClarifyingQuestionLLMOutput(
                **make_valid_weak_link_clarifying_question_output_dict()
            )
        elif schema == ExperimentGeneratorLLMOutput:
            structured.invoke.return_value = ExperimentGeneratorLLMOutput(
                **make_valid_experiment_generator_output_dict()
            )
        elif schema == ExperimentCriticLLMOutput:
            structured.invoke.return_value = ExperimentCriticLLMOutput(
                **make_valid_experiment_critic_output_dict()
            )
        elif schema == StartupDiagnosisSnapshotLLMOutput:
            structured.invoke.return_value = StartupDiagnosisSnapshotLLMOutput(
                **make_valid_diagnosis_snapshot_output_dict()
            )
        elif schema == FounderTurnInterpretationLLMOutput:
            structured.invoke.return_value = FounderTurnInterpretationLLMOutput(
                **make_valid_founder_turn_interpretation_output_dict()
            )
        elif schema == AdvisorResponseLLMOutput:
            structured.invoke.return_value = AdvisorResponseLLMOutput(
                **make_valid_advisor_response_output_dict()
            )
        return structured

    mock_llm.with_structured_output.side_effect = mock_with_structured_output

    mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
    mocker.patch(
        "startary.graph.nodes.get_settings",
        return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"),
    )
    return mock_llm


class TestGraphCompilation:
    def test_build_graph_compiles(self):
        """Graph compiles without error and without API key."""
        os.environ.pop("OPENAI_API_KEY", None)
        graph = build_graph()
        assert graph is not None

    def test_compiled_graph_is_invocable(self):
        graph = build_graph()
        assert callable(getattr(graph, "invoke", None))


class TestGraphExecutionM2:
    def test_full_m2_graph_executes(self, mocker):
        """End-to-end M2.6 graph run with fully mocked LLMs."""
        _patch_all_m26_llms(mocker)
        graph = build_graph()
        state = _make_initial_state()
        result = graph.invoke(state)

        assert result is not None
        # Identity
        assert result["iteration"] == 1
        assert result["phase"] == WorkflowPhase.TEST

        # M1 Outputs
        assert "startup_state" in result
        assert "diagnosis" in result

        # M2 Outputs
        assert result["weak_area"] == str(StartupArea.PROBLEM)
        assert "weak_link_details" in result
        assert result["weak_link_details"]["score"] == 81
        assert "reality_check" in result
        assert "active_assumption" in result

        # Assumption domain model checks
        assumption = result["active_assumption"]
        assert assumption["area"] == str(StartupArea.PROBLEM)
        assert assumption["is_critical"] is True
        assert assumption["status"] == "UNTESTED"

        # M2.6 Outputs
        assert "execution_context" in result
        assert "startup_progress" in result
        assert "gap_plan" in result

        # Traces — M2.6 adds 2 more LLM calls (extractor + gap planner) + M3 generator + critic
        assert "traces" in result
        assert len(result["traces"]) >= 5

    def test_m3_experiment_created_in_full_pipeline(self, mocker):
        """M3 pipeline executes through to active_experiment, critic, and can_execute."""
        _patch_all_m26_llms(mocker)
        graph = build_graph()
        state = _make_initial_state()
        result = graph.invoke(state)

        assert "active_experiment" in result and result["active_experiment"] is not None
        assert result["active_experiment"]["status"] == "READY"
        assert result["active_experiment"]["assumption_id"] == result["active_assumption"]["id"]
        assert "experiment_critic_output" in result
        assert result["experiment_critic_output"]["verdict"] == "ACCEPT"
        assert "can_execute_result" in result
        assert result["can_execute_result"]["readiness"] in ["CAN_EXECUTE", "NEEDS_HELP", "BLOCKED"]
