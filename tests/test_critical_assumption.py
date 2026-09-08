"""
Tests: Critical Assumption generation and validation — Milestone 2.

All tests use mocked LLM clients without requiring live network calls.
"""

from unittest.mock import MagicMock
import pytest

from startary.domain.assumption import Assumption
from startary.domain.enums import AssumptionStatus, WorkflowPhase
from startary.domain.taxonomy import StartupArea
from startary.graph.graph_state import StartaryGraphState
from startary.graph.nodes import generate_critical_assumption, save_critical_assumption
from startary.llm.schemas import CriticalAssumptionLLMOutput
from tests.fixtures.scenarios import make_valid_critical_assumption_output_dict


class TestCriticalAssumptionNode:

    def _create_state(self, weak_area: str = "Problem") -> StartaryGraphState:
        return StartaryGraphState(
            project_id="test-ca-project",
            phase=WorkflowPhase.DIAGNOSE,
            iteration=1,
            weak_area=weak_area,
            reality_check={
                "weak_link_area": weak_area,
                "conclusion": "UNSUPPORTED",
                "summary": "Founder has no validation for restaurant AI adoption.",
                "unresolved_uncertainty": "Whether restaurant owners will adopt AI.",
            },
            startup_state={"areas": {}},
            traces=[],
        )

    def test_missing_weak_area_or_reality_check_raises(self):
        """generate_critical_assumption raises if weak_area is absent."""
        state = StartaryGraphState(
            project_id="test-proj",
            phase=WorkflowPhase.DIAGNOSE,
            iteration=1,
        )
        with pytest.raises(ValueError, match="weak_area or reality_check missing"):
            generate_critical_assumption(state)

    def test_valid_assumption_generation_and_storage(self, mocker):
        """Generates structured output and saves a first-class Assumption domain model."""
        ca_dict = make_valid_critical_assumption_output_dict(area=StartupArea.PROBLEM)
        valid_output = CriticalAssumptionLLMOutput(**ca_dict)

        mock_structured_llm = MagicMock()
        mock_structured_llm.invoke.return_value = valid_output
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured_llm

        mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
        mocker.patch(
            "startary.graph.nodes.get_settings",
            return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"),
        )

        state = self._create_state(weak_area=str(StartupArea.PROBLEM))

        # 1. LLM node
        llm_result = generate_critical_assumption(state)
        assert "critical_assumption_output" in llm_result

        # Update working state
        state["critical_assumption_output"] = llm_result["critical_assumption_output"]

        # 2. Deterministic save node
        save_result = save_critical_assumption(state)
        assert "active_assumption" in save_result

        assumption_dict = save_result["active_assumption"]
        assumption = Assumption(**assumption_dict)

        assert assumption.area == StartupArea.PROBLEM
        assert assumption.status == AssumptionStatus.UNTESTED
        assert assumption.is_critical is True
        assert assumption.iteration_created == 1
        assert len(assumption.statement) >= 10
        assert save_result["phase"] == WorkflowPhase.TEST

    def test_area_normalized_if_mismatched(self, mocker):
        """If LLM outputs a different area, it is normalized to match the weak link."""
        ca_dict = make_valid_critical_assumption_output_dict(area=StartupArea.TARGET_USER)
        valid_output = CriticalAssumptionLLMOutput(**ca_dict)

        mock_structured_llm = MagicMock()
        mock_structured_llm.invoke.return_value = valid_output
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured_llm

        mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
        mocker.patch(
            "startary.graph.nodes.get_settings",
            return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"),
        )

        # Weak area is ACQUISITION, but LLM output was TARGET_USER
        state = self._create_state(weak_area=str(StartupArea.ACQUISITION))
        llm_result = generate_critical_assumption(state)
        state["critical_assumption_output"] = llm_result["critical_assumption_output"]

        save_result = save_critical_assumption(state)
        assumption = Assumption(**save_result["active_assumption"])
        assert assumption.area == StartupArea.ACQUISITION

    def test_empty_statement_fails_validation(self):
        """Empty assumption statement raises ValueError in save_critical_assumption."""
        state = self._create_state(weak_area=str(StartupArea.PROBLEM))
        state["critical_assumption_output"] = {
            "area": "Problem",
            "statement": "   ",
        }
        with pytest.raises(ValueError, match="statement cannot be empty"):
            save_critical_assumption(state)

