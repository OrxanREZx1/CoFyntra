"""
Automated unit tests for Critical Assumption Grounding Discipline — Milestone 2.6.2.

Verifies:
  1. Rejects ungrounded percentages (e.g. 30%, 50%)
  2. Allows founder-supplied percentages
  3. Rejects ungrounded order thresholds (e.g. 20 orders/week)
  4. Allows founder-supplied order thresholds
  5. Rejects invented frequencies (e.g. at least weekly, daily)
  6. Rejects ungrounded ranking claims (e.g. top three)
  7. Rejects claims explicitly marked as unsupported in Reality Check (what_evidence_does_not_support)
  8. Unresolved uncertainty correctly formulated as the testable proposition
  9. Qualitative assumption without numeric threshold validates cleanly
  10. Invalid first generation triggers corrective retry
  11. Double-failure raises controlled error (invalid assumption never saved)
  12. Valid assumption saved with status=UNTESTED, is_critical=True, matching weak area
"""

from __future__ import annotations

from unittest.mock import MagicMock
import pytest

from startary.domain.assumption import Assumption
from startary.domain.enums import AssumptionStatus, WorkflowPhase
from startary.domain.taxonomy import StartupArea
from startary.graph.graph_state import StartaryGraphState
from startary.graph.nodes import (
    generate_critical_assumption,
    save_critical_assumption,
    validate_critical_assumption_grounding,
)
from startary.llm.schemas import CriticalAssumptionLLMOutput


# ---------------------------------------------------------------------------
# 1. Deterministic Grounding Validator Tests
# ---------------------------------------------------------------------------

class TestCriticalAssumptionGroundingValidator:
    def test_rejects_invented_percentage(self):
        """Assumption with invented percentage (e.g. 30%) not in founder text is rejected."""
        ca = CriticalAssumptionLLMOutput(
            area=StartupArea.PROBLEM,
            statement="At least 30% of small sellers experience delivery coordination errors.",
            rationale="Core problem risk.",
            what_falsifies_it="Fewer than 10% report errors.",
        )
        rc_dict = {"what_evidence_does_not_support": []}
        founder_source = "Small sellers say managing orders across DMs gets messy as they grow."

        is_valid, reason = validate_critical_assumption_grounding(ca, rc_dict, founder_source, "Problem")
        assert is_valid is False
        assert "percentage" in reason.lower()

    def test_allows_founder_supplied_percentage(self):
        """If founder explicitly stated a percentage (e.g. 50%), it is permitted."""
        ca = CriticalAssumptionLLMOutput(
            area=StartupArea.PROBLEM,
            statement="At least 50% of our current student users struggle with selecting an IT branch.",
            rationale="Validating founder segment claim.",
            what_falsifies_it="Most students easily choose.",
        )
        rc_dict = {"what_evidence_does_not_support": []}
        founder_source = "We noticed around 50% of the students we polled had difficulty choosing a track."

        is_valid, reason = validate_critical_assumption_grounding(ca, rc_dict, founder_source, "Problem")
        assert is_valid is True
        assert reason == ""

    def test_rejects_invented_order_threshold(self):
        """Assumption with invented order threshold (e.g. '20 orders per week') is rejected."""
        ca = CriticalAssumptionLLMOutput(
            area=StartupArea.PROBLEM,
            statement="Instagram sellers handling more than 20 orders per week experience delivery confusion.",
            rationale="Operational bottleneck threshold.",
            what_falsifies_it="Sellers with 20+ orders report no confusion.",
        )
        rc_dict = {"what_evidence_does_not_support": []}
        founder_source = "Small sellers told us order volume makes managing Instagram DMs messy."

        is_valid, reason = validate_critical_assumption_grounding(ca, rc_dict, founder_source, "Problem")
        assert is_valid is False
        assert "order" in reason.lower() or "threshold" in reason.lower()

    def test_allows_founder_supplied_order_threshold(self):
        """If founder explicitly stated an order volume (e.g. '20 orders'), it is allowed."""
        ca = CriticalAssumptionLLMOutput(
            area=StartupArea.PROBLEM,
            statement="Sellers receiving more than 20 orders encounter delivery tracking friction.",
            rationale="Testing founder's explicit volume threshold.",
            what_falsifies_it="High volume sellers have zero friction.",
        )
        rc_dict = {"what_evidence_does_not_support": []}
        founder_source = "When sellers reach 20 orders a day, things become chaotic."

        is_valid, reason = validate_critical_assumption_grounding(ca, rc_dict, founder_source, "Problem")
        assert is_valid is True

    def test_rejects_invented_frequency(self):
        """Assumption with invented frequency (e.g. 'at least weekly') is rejected."""
        ca = CriticalAssumptionLLMOutput(
            area=StartupArea.PROBLEM,
            statement="Sellers experience delivery coordination friction at least weekly across tools.",
            rationale="Recurring frequency.",
            what_falsifies_it="Errors occur rarely.",
        )
        rc_dict = {"what_evidence_does_not_support": []}
        founder_source = "Sellers switch between WhatsApp and courier apps to coordinate deliveries."

        is_valid, reason = validate_critical_assumption_grounding(ca, rc_dict, founder_source, "Problem")
        assert is_valid is False
        assert "frequency" in reason.lower()

    def test_rejects_invented_ranking_claim(self):
        """Assumption with invented ranking claim (e.g. 'top three operational frustrations') is rejected."""
        ca = CriticalAssumptionLLMOutput(
            area=StartupArea.PROBLEM,
            statement="Delivery management is among the top three operational frustrations for small sellers.",
            rationale="Priority ranking.",
            what_falsifies_it="Sellers rank it low priority.",
        )
        rc_dict = {"what_evidence_does_not_support": []}
        founder_source = "Delivery coordination is annoying for small sellers."

        is_valid, reason = validate_critical_assumption_grounding(ca, rc_dict, founder_source, "Problem")
        assert is_valid is False
        assert "ranking" in reason.lower()

    def test_rejects_reality_check_unsupported_mechanisms(self):
        """Claims explicitly marked as unevidenced in Reality Check cannot be asserted as facts."""
        ca = CriticalAssumptionLLMOutput(
            area=StartupArea.PROBLEM,
            statement="Small sellers suffer from missed deliveries when coordinating orders across DMs.",
            rationale="Core delivery failure mode.",
            what_falsifies_it="No missed deliveries occur.",
        )
        rc_dict = {
            "what_evidence_does_not_support": [
                "No evidence of missed, delayed, or incorrectly recorded deliveries",
            ]
        }
        founder_source = "Sellers say they spend too much time switching apps."

        is_valid, reason = validate_critical_assumption_grounding(ca, rc_dict, founder_source, "Problem")
        assert is_valid is False
        assert "missed" in reason.lower()

    def test_qualitative_testable_assumption_without_numbers_validates(self):
        """Qualitative falsifiable assumption without invented numbers or mechanisms passes validation cleanly."""
        ca = CriticalAssumptionLLMOutput(
            area=StartupArea.PROBLEM,
            statement=(
                "Small Instagram sellers managing growing order volumes across Instagram DMs "
                "and courier tools experience enough delivery-coordination friction that they "
                "would prefer a simpler way to manage the workflow."
            ),
            rationale="Directly tests core problem hypothesis without invented thresholds.",
            what_falsifies_it="Sellers report DM and courier switching is effortless and satisfactory.",
            grounding_points=["Founder spoke with small Instagram sellers.", "Sellers coordinate via DMs."],
            unresolved_uncertainty_addressed="Whether delivery coordination friction is severe enough to motivate seeking another workflow.",
            introduced_test_parameters=[],
        )
        rc_dict = {
            "what_evidence_does_not_support": ["Specific failure mechanisms (missed deliveries) unknown."],
            "unresolved_uncertainty": "Whether delivery coordination friction is severe enough to motivate seeking another workflow.",
        }
        founder_source = "Small Instagram sellers use DMs and courier apps, and it gets messy as orders grow."

        is_valid, reason = validate_critical_assumption_grounding(ca, rc_dict, founder_source, "Problem")
        assert is_valid is True
        assert reason == ""


# ---------------------------------------------------------------------------
# 2. Node Generation & Retry Logic Tests
# ---------------------------------------------------------------------------

class TestCriticalAssumptionNodeExecution:
    def test_retry_on_invalid_output_and_succeeds_on_second_attempt(self, mocker):
        """When the first LLM generation introduces unsupported numbers, the node retries with corrective feedback and succeeds."""
        invalid_output = CriticalAssumptionLLMOutput(
            area=StartupArea.PROBLEM,
            statement="Sellers with at least 20 orders per week experience errors weekly.",
            rationale="Initial attempt with invented numbers",
            what_falsifies_it="No errors observed.",
        )
        valid_output = CriticalAssumptionLLMOutput(
            area=StartupArea.PROBLEM,
            statement="Small Instagram sellers managing growing order volume experience recurring delivery coordination friction.",
            rationale="Clean grounded assumption without numbers",
            what_falsifies_it="Sellers report managing delivery via DMs is smooth.",
        )

        mock_llm = MagicMock()
        mock_structured = MagicMock()
        # First call returns invalid, second call returns valid
        mock_structured.invoke.side_effect = [invalid_output, valid_output]
        mock_llm.with_structured_output.return_value = mock_structured

        mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
        mocker.patch("startary.graph.nodes.get_settings", return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"))

        state: StartaryGraphState = {
            "project_id": "test-retry",
            "weak_area": "Problem",
            "reality_check": {
                "summary": "Delivery coordination is messy.",
                "conclusion": "INCONCLUSIVE",
                "unresolved_uncertainty": "Whether coordination friction is significant.",
                "what_evidence_does_not_support": [],
            },
            "startup_state": {"areas": {}},
            "founder_messages": [
                {"role": "founder", "content": "Small Instagram sellers tell us managing orders across DMs gets messy."}
            ],
        }

        result = generate_critical_assumption(state)
        assert result["critical_assumption_output"]["statement"] == valid_output.statement
        assert mock_structured.invoke.call_count == 2

    def test_double_failure_raises_controlled_error(self, mocker):
        """If both generation attempts introduce ungrounded metrics, a controlled error is raised and nothing is saved."""
        invalid_output = CriticalAssumptionLLMOutput(
            area=StartupArea.PROBLEM,
            statement="At least 30% of sellers experience at least 20 order errors weekly.",
            rationale="Persistently invalid",
            what_falsifies_it="No errors.",
        )

        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.side_effect = [invalid_output, invalid_output]
        mock_llm.with_structured_output.return_value = mock_structured

        mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
        mocker.patch("startary.graph.nodes.get_settings", return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"))
        mocker.patch("startary.graph.nodes.build_grounded_critical_assumption_fallback", return_value=None)

        state: StartaryGraphState = {
            "project_id": "test-fail",
            "weak_area": "Problem",
            "reality_check": {
                "summary": "Summary",
                "conclusion": "INCONCLUSIVE",
                "unresolved_uncertainty": "Uncertainty",
                "what_evidence_does_not_support": [],
            },
            "startup_state": {"areas": {}},
            "founder_messages": [
                {"role": "founder", "content": "Sellers say it is messy."}
            ],
        }

        with pytest.raises(ValueError) as excinfo:
            generate_critical_assumption(state)
        assert "failed grounding validation after retry" in str(excinfo.value)

    def test_save_critical_assumption_persists_single_untested_assumption(self):
        """save_critical_assumption instantiates first-class Assumption with UNTESTED status and matching area."""
        state: StartaryGraphState = {
            "project_id": "test-save",
            "weak_area": "Problem",
            "iteration": 1,
            "critical_assumption_output": {
                "statement": "Small Instagram sellers experience recurring delivery coordination friction across tools.",
                "area": "Problem",
            },
        }

        result = save_critical_assumption(state)
        assumption = result["active_assumption"]
        assert assumption["statement"] == "Small Instagram sellers experience recurring delivery coordination friction across tools."
        assert assumption["area"] == "Problem"
        assert assumption["status"] == AssumptionStatus.UNTESTED
        assert assumption["is_critical"] is True
        assert result["phase"] == WorkflowPhase.TEST
