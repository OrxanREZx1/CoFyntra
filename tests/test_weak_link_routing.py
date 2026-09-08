"""
Automated unit tests for Weak-Link Router & Loop Integrity — Milestone 2.6.2.

Verifies:
  1. should_clarify=True routes to select_weak_link_question (Case A)
  2. should_clarify=True, turns=2 routes to select_weak_link_question (Case B)
  3. should_clarify=True, turns=3 routes to reality_check (Case C, max turns limit)
  4. should_clarify=False, turns=0 routes to reality_check (Case D)
  5. selection_confidence=LOW but should_clarify=False routes to reality_check (Case E)
  6. selection_confidence=HIGH but should_clarify=True routes to select_weak_link_question (Case F)
  7. Router reads top-level should_clarify and nested weak_link_discrimination.should_clarify
  8. Founder answer re-executes diagnosis and area assessment loop
  9. Weak-link winner and scores can change after clarification
  10. Competing areas update on re-assessment
"""

from __future__ import annotations

from unittest.mock import MagicMock
import pytest

from startary.domain.assessment import RiskFactorLevel, StartupAreaAssessment
from startary.domain.discrimination import SelectionConfidence
from startary.domain.enums import WorkflowPhase
from startary.domain.taxonomy import StartupArea
from startary.graph.builder import (
    MAX_WEAK_LINK_CLARIFICATION_TURNS,
    build_graph,
    route_after_merge_diagnosis,
    route_on_weak_link,
)
from startary.graph.graph_state import StartaryGraphState
from startary.graph.nodes import (
    assess_weak_link_discrimination,
    merge_founder_answer,
    select_weak_link,
)
from startary.llm.schemas import (
    DiagnosisLLMOutput,
    StartupAreaAssessmentBatchLLMOutput,
    WeakLinkDiscriminationLLMOutput,
)
from tests.fixtures.scenarios import (
    make_minimal_valid_llm_output_dict,
    make_valid_area_assessments_batch_output_dict,
    make_valid_weak_link_discrimination_output_dict,
)


# ---------------------------------------------------------------------------
# 1. Router Authority Tests (Cases A through F)
# ---------------------------------------------------------------------------

class TestWeakLinkRouterAuthority:
    def test_case_a_should_clarify_true_turn_0(self):
        """Case A: should_clarify=True, turns=0 -> select_weak_link_question."""
        state: StartaryGraphState = {
            "should_clarify": True,
            "weak_link_clarification_turns": 0,
            "selection_confidence": "LOW",
        }
        assert route_on_weak_link(state) == "select_weak_link_question"

    def test_case_b_should_clarify_true_turn_2(self):
        """Case B: should_clarify=True, turns=2 -> select_weak_link_question."""
        state: StartaryGraphState = {
            "should_clarify": True,
            "weak_link_clarification_turns": 2,
            "selection_confidence": "LOW",
        }
        assert route_on_weak_link(state) == "select_weak_link_question"

    def test_case_c_should_clarify_true_turn_3_max_limit(self):
        """Case C: should_clarify=True, turns=3 -> reality_check (guardrail)."""
        state: StartaryGraphState = {
            "should_clarify": True,
            "weak_link_clarification_turns": MAX_WEAK_LINK_CLARIFICATION_TURNS,
            "selection_confidence": "LOW",
        }
        assert route_on_weak_link(state) == "reality_check"

    def test_case_d_should_clarify_false_turn_0(self):
        """Case D: should_clarify=False, turns=0 -> reality_check."""
        state: StartaryGraphState = {
            "should_clarify": False,
            "weak_link_clarification_turns": 0,
            "selection_confidence": "HIGH",
        }
        assert route_on_weak_link(state) == "reality_check"

    def test_case_e_low_confidence_alone_does_not_route_to_clarify(self):
        """Case E: selection_confidence=LOW, but should_clarify=False -> reality_check."""
        state: StartaryGraphState = {
            "should_clarify": False,
            "weak_link_clarification_turns": 0,
            "selection_confidence": "LOW",
        }
        assert route_on_weak_link(state) == "reality_check"

    def test_case_f_high_confidence_with_should_clarify_routes_to_clarify(self):
        """Case F: selection_confidence=HIGH, but should_clarify=True -> select_weak_link_question."""
        state: StartaryGraphState = {
            "should_clarify": True,
            "weak_link_clarification_turns": 0,
            "selection_confidence": "HIGH",
        }
        assert route_on_weak_link(state) == "select_weak_link_question"

    def test_nested_weak_link_discrimination_should_clarify_is_read(self):
        """Router successfully reads should_clarify from nested weak_link_discrimination dict."""
        state: StartaryGraphState = {
            "weak_link_discrimination": {"should_clarify": True},
            "weak_link_clarification_turns": 1,
        }
        assert route_on_weak_link(state) == "select_weak_link_question"

    def test_fallback_alias_should_clarify_weak_link_is_read(self):
        """Router successfully falls back to should_clarify_weak_link if top-level should_clarify is None."""
        state: StartaryGraphState = {
            "should_clarify_weak_link": True,
            "weak_link_clarification_turns": 0,
        }
        assert route_on_weak_link(state) == "select_weak_link_question"


# ---------------------------------------------------------------------------
# 2. Weak-Link Loop Re-Execution & Progression Tests
# ---------------------------------------------------------------------------

class TestWeakLinkLoopReExecution:
    def test_route_after_merge_diagnosis_routes_to_assess_areas_during_weak_link_loop(self):
        """During weak-link loop (weak_link_clarification_turns > 0), merge_diagnosis routes directly to assess_startup_areas."""
        state: StartaryGraphState = {
            "weak_link_clarification_turns": 1,
            "is_strategic_sufficient": True,
        }
        assert route_after_merge_diagnosis(state) == "assess_startup_areas"

    def test_route_after_merge_diagnosis_routes_to_gap_plan_initially(self):
        """When not yet in weak-link loop and strategic sufficiency is False, routes to plan_information_gap."""
        state: StartaryGraphState = {
            "weak_link_clarification_turns": 0,
            "is_strategic_sufficient": False,
        }
        assert route_after_merge_diagnosis(state) == "plan_information_gap"

    def test_merge_founder_answer_sets_sufficiency_and_increments_turns(self):
        """Founder answer to weak-link question increments weak_link_clarification_turns and marks sufficiency."""
        state: StartaryGraphState = {
            "active_interrupt_type": "weak_link",
            "current_weak_link_question": {
                "question": "What delivery problems do sellers actually describe?",
                "competing_areas": ["Problem", "Demand Signal"],
            },
            "founder_response": "Two sellers told us switching between DMs and courier apps wastes an hour every day.",
            "weak_link_clarification_turns": 0,
            "baseline_question_turns": 1,
            "strategic_clarification_turns": 1,
            "weak_link_clarification_history": [],
            "founder_messages": [],
        }

        updates = merge_founder_answer(state)
        assert updates["weak_link_clarification_turns"] == 1
        assert updates["is_strategic_sufficient"] is True
        assert updates["is_baseline_sufficient"] is True
        assert len(updates["weak_link_clarification_history"]) == 1

    def test_recalculated_scores_can_alter_weak_link_winner(self):
        """When assessments change after re-diagnosis, select_weak_link deterministically selects the new winner."""
        # Initial: PROBLEM is winner with score 81, DEMAND_SIGNAL is 24
        initial_output = {
            "assessments": [
                {"area": "Problem", "importance": "HIGH", "uncertainty": "HIGH", "evidence_gap": "HIGH", "consequence_of_being_wrong": "HIGH", "rationale": "High problem risk"},
                {"area": "Demand Signal", "importance": "MEDIUM", "uncertainty": "MEDIUM", "evidence_gap": "MEDIUM", "consequence_of_being_wrong": "HIGH", "rationale": "Moderate demand risk"},
            ]
        }
        state_1: StartaryGraphState = {
            "project_id": "test-rerun",
            "area_assessments_output": initial_output,
            "startup_state": {"areas": {}},
        }
        res_1 = select_weak_link(state_1)
        assert res_1["weak_area"] == StartupArea.PROBLEM

        # After founder clarification: Problem uncertainty dropped (now score 24), but Demand Signal became score 81
        updated_output = {
            "assessments": [
                {"area": "Problem", "importance": "HIGH", "uncertainty": "LOW", "evidence_gap": "LOW", "consequence_of_being_wrong": "HIGH", "rationale": "Problem validated by interviews"},
                {"area": "Demand Signal", "importance": "HIGH", "uncertainty": "HIGH", "evidence_gap": "HIGH", "consequence_of_being_wrong": "HIGH", "rationale": "Zero willingness to pay reported"},
            ]
        }
        state_2: StartaryGraphState = {
            "project_id": "test-rerun",
            "area_assessments_output": updated_output,
            "startup_state": {"areas": {}},
        }
        res_2 = select_weak_link(state_2)
        assert res_2["weak_area"] == StartupArea.DEMAND_SIGNAL
        assert res_2["weak_link_details"]["score"] == 81

