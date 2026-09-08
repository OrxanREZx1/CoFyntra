"""
Tests: Reality Check node & epistemic evidence evaluation — Milestone 2.

All tests use mocked LLM clients without requiring live network calls.
"""

from unittest.mock import MagicMock
import pytest

from startary.domain.enums import WorkflowPhase
from startary.domain.reality_check import EvidenceConclusion, RealityCheck
from startary.domain.taxonomy import StartupArea
from startary.graph.graph_state import StartaryGraphState
from startary.graph.nodes import reality_check
from startary.llm.schemas import RealityCheckLLMOutput
from tests.fixtures.scenarios import make_valid_reality_check_output_dict


class TestRealityCheckNode:

    def _create_state(self, weak_area: str = "Problem") -> StartaryGraphState:
        return StartaryGraphState(
            project_id="test-rc-project",
            phase=WorkflowPhase.DIAGNOSE,
            iteration=1,
            weak_area=weak_area,
            startup_state={
                "areas": {
                    weak_area: {
                        "status": "PARTIALLY_KNOWN",
                        "understanding": "Founder observed workflow friction in restaurants.",
                        "founder_claims": ["All restaurants will buy this software."],
                        "evidence_provided": ["Interviewed 2 restaurant owners."],
                        "missing_information": "Frequency and cost of friction.",
                    }
                }
            },
            diagnosis={"summary": "Initial diagnosis summary."},
            traces=[],
        )

    def test_missing_weak_area_raises(self):
        """reality_check must raise if weak_area is missing from state."""
        state = StartaryGraphState(
            project_id="test-proj",
            phase=WorkflowPhase.DIAGNOSE,
            iteration=1,
        )
        with pytest.raises(ValueError, match="weak_area is missing"):
            reality_check(state)

    def test_valid_reality_check_execution(self, mocker):
        """Valid LLM reality check output is parsed into a RealityCheck domain object."""
        rc_dict = make_valid_reality_check_output_dict(
            area=StartupArea.PROBLEM,
            conclusion=EvidenceConclusion.PARTIALLY_SUPPORTED,
        )
        valid_output = RealityCheckLLMOutput(**rc_dict)

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
        result = reality_check(state)

        assert "reality_check" in result
        rc_obj = RealityCheck(**result["reality_check"])
        assert rc_obj.weak_link_area == StartupArea.PROBLEM
        assert rc_obj.conclusion == EvidenceConclusion.PARTIALLY_SUPPORTED
        assert "traces" in result
        assert len(result["traces"]) == 1

    def test_preserves_no_evidence_conclusion(self, mocker):
        """NO_EVIDENCE conclusion is preserved without forcing false validation."""
        rc_dict = make_valid_reality_check_output_dict(
            area=StartupArea.BUSINESS_MODEL,
            conclusion=EvidenceConclusion.NO_EVIDENCE,
        )
        valid_output = RealityCheckLLMOutput(**rc_dict)

        mock_structured_llm = MagicMock()
        mock_structured_llm.invoke.return_value = valid_output
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured_llm

        mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
        mocker.patch(
            "startary.graph.nodes.get_settings",
            return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"),
        )

        state = self._create_state(weak_area=str(StartupArea.BUSINESS_MODEL))
        result = reality_check(state)

        rc_obj = RealityCheck(**result["reality_check"])
        assert rc_obj.conclusion == EvidenceConclusion.NO_EVIDENCE

    def test_preserves_inconclusive_conclusion(self, mocker):
        """INCONCLUSIVE conclusion is preserved for mixed evidence."""
        rc_dict = make_valid_reality_check_output_dict(
            area=StartupArea.PROBLEM,
            conclusion=EvidenceConclusion.INCONCLUSIVE,
        )
        valid_output = RealityCheckLLMOutput(**rc_dict)

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
        result = reality_check(state)

        rc_obj = RealityCheck(**result["reality_check"])
        assert rc_obj.conclusion == EvidenceConclusion.INCONCLUSIVE

    def test_llm_failure_retried_once_then_raises(self, mocker):
        """LLM failure triggers exactly one retry before raising a RuntimeError."""
        mock_structured_llm = MagicMock()
        mock_structured_llm.invoke.side_effect = RuntimeError("API error")
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured_llm

        mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
        mocker.patch(
            "startary.graph.nodes.get_settings",
            return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"),
        )

        state = self._create_state(weak_area=str(StartupArea.PROBLEM))
        with pytest.raises(RuntimeError, match="reality_check failed after 2 attempts"):
            reality_check(state)

        assert mock_structured_llm.invoke.call_count == 2

