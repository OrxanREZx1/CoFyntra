"""
Tests: diagnose_startup and merge_diagnosis nodes — Milestone 1.

All tests use mocked LLM client. No real API calls are made.

Tests verify:
  1. Raw context is NOT stored as a Problem area value
  2. Valid diagnosis output is accepted and state is mutated correctly
  3. Missing areas remain explicitly UNKNOWN
  4. Invalid LLM output causes retry (max once) then safe failure
  5. Malformed diagnosis_output does not mutate startup_state
  6. No Assumption or Experiment objects are created by M1 nodes
  7. AITrace is produced for successful diagnosis calls
  8. Empty raw_context returns safe empty diagnosis without API call
"""

import uuid
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from startary.domain.area_knowledge import AreaKnowledge, AreaStatus
from startary.domain.diagnosis import Diagnosis
from startary.domain.enums import WorkflowPhase
from startary.domain.state import StartupState
from startary.domain.taxonomy import StartupArea
from startary.graph.graph_state import StartaryGraphState
from startary.graph.nodes import diagnose_startup, merge_diagnosis
from startary.llm.schemas import AreaDiagnosisOutput, DiagnosisLLMOutput
from tests.fixtures.scenarios import (
    make_minimal_valid_llm_output_dict,
    make_rich_valid_llm_output_dict,
)


def _project_id() -> str:
    return str(uuid.uuid4())


def _base_state(raw_context: str = "", project_id: str | None = None) -> StartaryGraphState:
    pid = project_id or _project_id()
    from startary.domain.enums import WorkflowPhase
    return StartaryGraphState(
        project_id=pid,
        phase=WorkflowPhase.UNDERSTAND,
        iteration=1,
        raw_context=raw_context,
        startup_state=StartupState().model_dump(),
        founder_evidence=[],
        external_evidence=[],
    )


# ---------------------------------------------------------------------------
# diagnose_startup node tests
# ---------------------------------------------------------------------------

class TestDiagnoseStartupNode:

    def test_empty_raw_context_returns_safe_empty_diagnosis(self):
        """If no raw context, diagnosis returns empty output without API call."""
        state = _base_state(raw_context="")
        # No API key needed — should not attempt to call LLM
        result = diagnose_startup(state)
        assert "diagnosis_output" in result
        output = result["diagnosis_output"]
        assert output["diagnosis_summary"] == "No founder context was provided."

    def test_empty_context_does_not_call_llm(self, mocker):
        """Empty raw_context must not invoke any LLM client."""
        mock_client = mocker.patch("startary.graph.nodes.get_llm_client")
        mocker.patch(
            "startary.graph.nodes.get_settings",
            return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"),
        )
        state = _base_state(raw_context="")
        diagnose_startup(state)
        mock_client.assert_not_called()

    def test_valid_llm_output_returned_as_dict(self, mocker):
        """Valid structured LLM output is returned as diagnosis_output dict."""
        valid_output = DiagnosisLLMOutput(**make_minimal_valid_llm_output_dict())
        mock_structured_llm = MagicMock()
        mock_structured_llm.invoke.return_value = valid_output
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured_llm

        mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
        mocker.patch(
            "startary.graph.nodes.get_settings",
            return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"),
        )

        state = _base_state(raw_context="I want to make an AI app for restaurants.")
        result = diagnose_startup(state)

        assert "diagnosis_output" in result
        assert isinstance(result["diagnosis_output"], dict)
        # Only one call — no per-area calls
        assert mock_structured_llm.invoke.call_count == 1

    def test_ai_trace_produced_on_success(self, mocker):
        """Successful diagnosis produces an AITrace."""
        valid_output = DiagnosisLLMOutput(**make_minimal_valid_llm_output_dict())
        mock_structured_llm = MagicMock()
        mock_structured_llm.invoke.return_value = valid_output
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured_llm

        mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
        mocker.patch(
            "startary.graph.nodes.get_settings",
            return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"),
        )

        state = _base_state(raw_context="Some startup context.")
        result = diagnose_startup(state)

        from startary.llm.prompts import DIAGNOSE_STARTUP_PROMPT_VERSION
        assert "last_trace" in result
        trace = result["last_trace"]
        assert trace["module_name"] == "diagnose_startup"
        assert trace["prompt_version"] == DIAGNOSE_STARTUP_PROMPT_VERSION
        assert "project_id" in trace
        assert trace["latency_ms"] >= 0.0

    def test_llm_failure_retried_exactly_once(self, mocker):
        """LLM failure triggers exactly one retry (2 total attempts) then raises."""
        mock_structured_llm = MagicMock()
        mock_structured_llm.invoke.side_effect = RuntimeError("LLM unavailable")
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured_llm

        mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
        mocker.patch(
            "startary.graph.nodes.get_settings",
            return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"),
        )

        state = _base_state(raw_context="Some context.")
        with pytest.raises(RuntimeError, match="failed after 2 attempts"):
            diagnose_startup(state)

        # Must have been called exactly 2 times (original + 1 retry)
        assert mock_structured_llm.invoke.call_count == 2

    def test_phase_updated_to_diagnose_on_success(self, mocker):
        """Phase is updated to DIAGNOSE when diagnosis succeeds."""
        valid_output = DiagnosisLLMOutput(**make_minimal_valid_llm_output_dict())
        mock_structured_llm = MagicMock()
        mock_structured_llm.invoke.return_value = valid_output
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured_llm

        mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
        mocker.patch(
            "startary.graph.nodes.get_settings",
            return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"),
        )

        state = _base_state(raw_context="Context.")
        result = diagnose_startup(state)
        assert result.get("phase") == WorkflowPhase.DIAGNOSE

    def test_no_assumption_or_experiment_created(self, mocker):
        """M1 must not create Assumption or Experiment objects."""
        valid_output = DiagnosisLLMOutput(**make_minimal_valid_llm_output_dict())
        mock_structured_llm = MagicMock()
        mock_structured_llm.invoke.return_value = valid_output
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured_llm

        mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
        mocker.patch(
            "startary.graph.nodes.get_settings",
            return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"),
        )

        state = _base_state(raw_context="Context.")
        result = diagnose_startup(state)

        assert "active_assumption" not in result
        assert "active_experiment" not in result


# ---------------------------------------------------------------------------
# merge_diagnosis node tests
# ---------------------------------------------------------------------------

class TestMergeDiagnosisNode:

    def _state_with_diagnosis_output(
        self, output_dict: dict, project_id: str | None = None
    ) -> StartaryGraphState:
        from startary.domain.enums import WorkflowPhase
        pid = project_id or _project_id()
        return StartaryGraphState(
            project_id=pid,
            phase=WorkflowPhase.DIAGNOSE,
            iteration=1,
            raw_context="I want to make an AI app for restaurants.",
            startup_state=StartupState().model_dump(),
            diagnosis_output=output_dict,
            founder_evidence=[],
            external_evidence=[],
        )

    def test_valid_minimal_output_produces_startup_state(self):
        """Valid minimal diagnosis output builds a complete StartupState."""
        state = self._state_with_diagnosis_output(make_minimal_valid_llm_output_dict())
        result = merge_diagnosis(state)

        assert "startup_state" in result
        ss_dict = result["startup_state"]
        areas = ss_dict.get("areas", {})

        # Should have all 10 areas (UNKNOWN for unmentioned)
        assert len(areas) == 10
        # Problem was in the LLM output
        problem = areas.get("Problem", {})
        assert problem.get("status") == AreaStatus.PARTIALLY_KNOWN

    def test_missing_areas_remain_unknown(self):
        """Areas not mentioned by LLM are filled in as UNKNOWN."""
        state = self._state_with_diagnosis_output(make_minimal_valid_llm_output_dict())
        result = merge_diagnosis(state)

        areas = result["startup_state"]["areas"]
        # Target User was not in minimal output
        target_user = areas.get("Target User", {})
        assert target_user.get("status") == AreaStatus.UNKNOWN

    def test_valid_rich_output_captures_evidence(self):
        """Rich output with evidence preserves evidence_provided lists."""
        state = self._state_with_diagnosis_output(make_rich_valid_llm_output_dict())
        result = merge_diagnosis(state)

        areas = result["startup_state"]["areas"]
        problem = areas.get("Problem", {})
        assert len(problem.get("evidence_provided", [])) >= 1

    def test_diagnosis_domain_object_built(self):
        """A Diagnosis domain object is built and serialized into state."""
        state = self._state_with_diagnosis_output(make_minimal_valid_llm_output_dict())
        result = merge_diagnosis(state)

        assert "diagnosis" in result
        diag_dict = result["diagnosis"]

        # Must be reconstructable as Diagnosis
        diag = Diagnosis(**diag_dict)
        assert diag.project_id == state["project_id"]
        assert diag.iteration == 1

    def test_missing_diagnosis_output_raises(self):
        """merge_diagnosis raises if diagnosis_output is missing from state."""
        from startary.domain.enums import WorkflowPhase
        state = StartaryGraphState(
            project_id=_project_id(),
            phase=WorkflowPhase.DIAGNOSE,
            iteration=1,
            startup_state=StartupState().model_dump(),
            founder_evidence=[],
            external_evidence=[],
        )
        with pytest.raises(ValueError, match="diagnosis_output is missing"):
            merge_diagnosis(state)

    def test_malformed_diagnosis_output_raises_without_state_mutation(self):
        """Malformed LLM output raises and does NOT mutate startup_state."""
        from startary.domain.enums import WorkflowPhase
        original_state_dict = StartupState().model_dump()
        state = StartaryGraphState(
            project_id=_project_id(),
            phase=WorkflowPhase.DIAGNOSE,
            iteration=1,
            startup_state=original_state_dict,
            diagnosis_output={"completely": "wrong", "schema": True},
            founder_evidence=[],
            external_evidence=[],
        )
        with pytest.raises((ValueError, ValidationError)):
            merge_diagnosis(state)

        # Original startup_state must be unchanged
        assert state["startup_state"] == original_state_dict

    def test_raw_context_not_assigned_to_problem_area(self):
        """
        Critical M1 correctness test.
        Raw context must NOT appear as the Problem area value.
        """
        raw = "I want to make an AI app for restaurants."
        state = self._state_with_diagnosis_output(make_minimal_valid_llm_output_dict())
        result = merge_diagnosis(state)

        areas = result["startup_state"]["areas"]
        problem = areas.get("Problem", {})

        # The raw context string should NOT be the understanding value
        # (the LLM returned a proper understanding, not the raw text)
        raw_as_understanding = problem.get("understanding", "") == raw
        assert not raw_as_understanding, (
            "Raw context was incorrectly stored as the Problem understanding. "
            "The LLM interpretation should be used, not the raw string."
        )

    def test_weak_areas_empty_in_m1(self):
        """weak_areas must be empty in M1 — Milestone 2 responsibility."""
        state = self._state_with_diagnosis_output(make_minimal_valid_llm_output_dict())
        result = merge_diagnosis(state)

        diag = result["diagnosis"]
        assert diag.get("weak_areas", []) == []

    def test_no_assumption_or_experiment_in_output(self):
        """merge_diagnosis must not create assumptions or experiments."""
        state = self._state_with_diagnosis_output(make_minimal_valid_llm_output_dict())
        result = merge_diagnosis(state)

        assert "active_assumption" not in result
        assert "active_experiment" not in result
