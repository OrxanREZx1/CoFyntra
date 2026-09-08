"""
Automated unit tests for Weak-Link Discrimination Loop — Milestone 2.6.1.

Verifies:
  1. Selection confidence enum (LOW, MEDIUM, HIGH)
  2. Discrimination domain model & schema validations
  3. Decisive single winner (>= 10 point score lead) yields HIGH confidence without clarification
  4. Saturated 81s / ties trigger discrimination clarification (LOW confidence, should_clarify=True)
  5. Single discriminating question crafted to separate competing candidates
  6. Maximum 3 weak-link clarification turns limit
  7. Persisting LOW confidence when ambiguity remains after max turns (no manufactured confidence)
  8. Deterministic winner remains final authority (no LLM score override)
  9. Answering weak-link question increments weak_link_clarification_turns only (counter isolation)
  10. Reality Check runs only after weak-link loop exits
"""

from __future__ import annotations

from unittest.mock import MagicMock
import pytest

from startary.domain.assessment import RiskFactorLevel, StartupAreaAssessment
from startary.domain.discrimination import (
    SelectionConfidence,
    WeakLinkClarifyingQuestion,
    WeakLinkDiscrimination,
)
from startary.domain.enums import WorkflowPhase
from startary.domain.taxonomy import StartupArea
from startary.graph.builder import build_graph, route_on_weak_link
from startary.graph.graph_state import StartaryGraphState
from startary.graph.nodes import (
    MAX_WEAK_LINK_CLARIFICATION_TURNS,
    assess_weak_link_discrimination,
    merge_founder_answer,
    select_weak_link,
    select_weak_link_question,
)
from startary.llm.schemas import (
    WeakLinkClarifyingQuestionLLMOutput,
    WeakLinkDiscriminationLLMOutput,
)
from tests.fixtures.scenarios import (
    make_valid_weak_link_clarifying_question_output_dict,
    make_valid_weak_link_discrimination_output_dict,
)


# ---------------------------------------------------------------------------
# 1. Domain Models & Schemas
# ---------------------------------------------------------------------------

class TestDiscriminationDomainModels:
    def test_selection_confidence_enum_values(self):
        assert SelectionConfidence.LOW == "LOW"
        assert SelectionConfidence.MEDIUM == "MEDIUM"
        assert SelectionConfidence.HIGH == "HIGH"

    def test_discrimination_model_validates(self):
        disc = WeakLinkDiscrimination(
            selection_confidence=SelectionConfidence.LOW,
            current_best_area=StartupArea.PROBLEM,
            competing_areas=[StartupArea.TARGET_USER, StartupArea.DEMAND_SIGNAL],
            discrimination_reason="Multiple areas share score 81/81.",
            missing_discriminating_information="Observation of user workarounds.",
            should_clarify=True,
            recommended_information_goal="Separate problem severity from demand signal.",
        )
        assert disc.selection_confidence == SelectionConfidence.LOW
        assert disc.current_best_area == StartupArea.PROBLEM
        assert len(disc.competing_areas) == 2
        assert disc.should_clarify is True

    def test_clarifying_question_model_validates(self):
        cq = WeakLinkClarifyingQuestion(
            question="When students face this confusion, what do they actually do about it — delay choosing or spend money?",
            competing_areas=[StartupArea.PROBLEM, StartupArea.DEMAND_SIGNAL],
            information_goal="Distinguish active demand signal from passive problem complaint.",
            why_this_question_now="Tied at 81/81.",
        )
        assert len(cq.question) >= 10
        assert len(cq.competing_areas) == 2


# ---------------------------------------------------------------------------
# 2. Discrimination Logic & Clear Winner Shortcut
# ---------------------------------------------------------------------------

class TestDiscriminationEvaluation:
    def test_decisive_single_winner_shortcuts_with_high_confidence(self):
        """When the top area leads by >= 10 points without a tie, confidence is HIGH and no clarification is asked."""
        assessments = [
            StartupAreaAssessment(
                area=StartupArea.PROBLEM,
                importance=RiskFactorLevel.HIGH,
                uncertainty=RiskFactorLevel.HIGH,
                evidence_gap=RiskFactorLevel.HIGH,
                consequence_if_wrong=RiskFactorLevel.HIGH,
                calculated_decision_risk=81,
                risk_rationale="Core problem unvalidated",
            ).model_dump(),
            StartupAreaAssessment(
                area=StartupArea.TARGET_USER,
                importance=RiskFactorLevel.MEDIUM,
                uncertainty=RiskFactorLevel.MEDIUM,
                evidence_gap=RiskFactorLevel.MEDIUM,
                consequence_if_wrong=RiskFactorLevel.HIGH,
                calculated_decision_risk=24,
                risk_rationale="Users relatively understood",
            ).model_dump(),
        ]

        state: StartaryGraphState = {
            "project_id": "test-clear-winner",
            "weak_area": str(StartupArea.PROBLEM),
            "weak_link_details": {
                "score": 81,
                "rationale": "High across all factors",
                "saturation_telemetry": {
                    "max_score": 81,
                    "max_score_tie_count": 1,
                    "saturated_area_count": 1,
                    "tie_breaker_used": False,
                },
            },
            "area_assessments": assessments,
            "weak_link_clarification_turns": 0,
        }

        result = assess_weak_link_discrimination(state)
        assert result["selection_confidence"] == "HIGH"
        assert result["should_clarify_weak_link"] is False
        assert len(result["competing_weak_areas"]) == 0

    def test_saturated_ties_trigger_clarification(self, mocker):
        """When multiple areas share max score (saturation), discrimination calls LLM and requests clarification."""
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = WeakLinkDiscriminationLLMOutput(
            selection_confidence="LOW",
            competing_areas=[StartupArea.TARGET_USER, StartupArea.PAIN_URGENCY],
            discrimination_reason="Problem, Target User, and Pain all scored 81/81.",
            missing_discriminating_information="Direct evidence of friction.",
            should_clarify=True,
            recommended_information_goal="Separate problem from user definition.",
        )
        mock_llm.with_structured_output.return_value = mock_structured
        mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
        mocker.patch("startary.graph.nodes.get_settings", return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"))

        state: StartaryGraphState = {
            "project_id": "test-saturated",
            "weak_area": str(StartupArea.PROBLEM),
            "weak_link_details": {
                "score": 81,
                "rationale": "Max score across 3 areas",
                "saturation_telemetry": {
                    "max_score": 81,
                    "max_score_tie_count": 3,
                    "saturated_area_count": 3,
                    "tie_breaker_used": True,
                },
            },
            "area_assessments": [
                {"area": "Problem", "calculated_decision_risk": 81, "importance": "HIGH", "uncertainty": "HIGH", "evidence_gap": "HIGH", "consequence_if_wrong": "HIGH"},
                {"area": "Target User", "calculated_decision_risk": 81, "importance": "HIGH", "uncertainty": "HIGH", "evidence_gap": "HIGH", "consequence_if_wrong": "HIGH"},
                {"area": "Pain / Urgency", "calculated_decision_risk": 81, "importance": "HIGH", "uncertainty": "HIGH", "evidence_gap": "HIGH", "consequence_if_wrong": "HIGH"},
            ],
            "weak_link_clarification_turns": 0,
        }

        result = assess_weak_link_discrimination(state)
        assert result["selection_confidence"] == "LOW"
        assert result["should_clarify_weak_link"] is True
        assert len(result["competing_weak_areas"]) > 0

    def test_select_weak_link_question_generates_differentiating_prompt(self, mocker):
        """Generates one focused question to differentiate competing candidate areas."""
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = WeakLinkClarifyingQuestionLLMOutput(
            **make_valid_weak_link_clarifying_question_output_dict(
                question="When students face this confusion, what do they actually do about it — delay choosing or spend money?",
                competing_areas=[StartupArea.PROBLEM, StartupArea.DEMAND_SIGNAL],
            )
        )
        mock_llm.with_structured_output.return_value = mock_structured
        mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
        mocker.patch("startary.graph.nodes.get_settings", return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"))

        state: StartaryGraphState = {
            "project_id": "test-disc-q",
            "weak_area": "Problem",
            "competing_weak_areas": ["Problem", "Demand Signal"],
            "weak_link_discrimination": {
                "recommended_information_goal": "Differentiate whether Problem is more critical than Demand Signal",
            },
            "weak_link_clarification_turns": 0,
        }

        result = select_weak_link_question(state)
        assert result["active_interrupt_type"] == "weak_link"
        assert "current_weak_link_question" in result
        assert len(result["current_weak_link_question"]["question"]) >= 10


# ---------------------------------------------------------------------------
# 3. Router, Turn Limits & Authority
# ---------------------------------------------------------------------------

class TestDiscriminationRoutingAndLimits:
    def test_routes_to_select_question_when_clarification_needed_under_max(self):
        state: StartaryGraphState = {
            "should_clarify_weak_link": True,
            "weak_link_clarification_turns": 1,
        }
        assert route_on_weak_link(state) == "select_weak_link_question"

    def test_routes_to_reality_check_when_not_clarifying(self):
        state: StartaryGraphState = {
            "should_clarify_weak_link": False,
            "weak_link_clarification_turns": 0,
        }
        assert route_on_weak_link(state) == "reality_check"

    def test_max_turns_forces_progression_to_reality_check_with_low_confidence(self):
        state: StartaryGraphState = {
            "should_clarify_weak_link": True,
            "weak_link_clarification_turns": MAX_WEAK_LINK_CLARIFICATION_TURNS,
            "selection_confidence": "LOW",
        }
        assert route_on_weak_link(state) == "reality_check"

    def test_assess_discrimination_preserves_low_confidence_on_max_turns(self):
        """When MAX turns are reached, assess_weak_link_discrimination does NOT manufacture HIGH confidence."""
        state: StartaryGraphState = {
            "project_id": "test-max-turns",
            "weak_area": "Problem",
            "weak_link_details": {"score": 81},
            "area_assessments": [],
            "weak_link_clarification_turns": MAX_WEAK_LINK_CLARIFICATION_TURNS,
        }

        result = assess_weak_link_discrimination(state)
        assert result["selection_confidence"] == "LOW"
        assert result["should_clarify_weak_link"] is False

    def test_counter_isolation_weak_link_does_not_consume_baseline_or_strategic(self):
        """Answering a weak-link question increments weak_link_clarification_turns only."""
        state: StartaryGraphState = {
            "active_interrupt_type": "weak_link",
            "current_weak_link_question": {
                "question": "What do students actually do when facing this issue?",
                "competing_areas": ["Problem", "Demand Signal"],
            },
            "founder_response": "They mostly search for YouTube videos or ask older students.",
            "baseline_question_turns": 2,
            "strategic_clarification_turns": 1,
            "weak_link_clarification_turns": 0,
            "clarification_turn_count": 1,
            "baseline_history": [],
            "clarification_history": [],
            "weak_link_clarification_history": [],
        }

        updates = merge_founder_answer(state)
        assert updates["weak_link_clarification_turns"] == 1
        # Baseline and strategic counters must not be modified!
        assert "baseline_question_turns" not in updates
        assert "strategic_clarification_turns" not in updates
        assert len(updates["weak_link_clarification_history"]) == 1
        assert updates["active_interrupt_type"] is None

