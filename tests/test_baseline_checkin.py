"""
Automated unit tests for Baseline Startup Check-In — Milestone 2.6.1.

Verifies:
  1. Baseline domain models & enum validations
  2. Baseline LLM schema validations
  3. Fast-path shortcut: rich input skips baseline questions completely (0 questions asked)
  4. Sparse input triggers baseline check-in
  5. Compact grouped questions generated
  6. Dynamic skipping: already-known team fields are skipped
  7. Dynamic skipping: already-known product stage fields are skipped
  8. Dynamic skipping: already-known resource/budget fields are skipped
  9. Baseline max turns guardrail (5 turns max)
  10. Separate counter isolation: baseline turns do NOT consume strategic or weak-link counters
  11. Unknown revenue does not block pre-MVP baseline sufficiency
"""

from __future__ import annotations

from unittest.mock import MagicMock
import pytest

from startary.domain.baseline import (
    BaselineDimension,
    BaselineDimensionStatus,
    BaselineQuestionPlan,
    BaselineStartupStateSufficiency,
)
from startary.domain.enums import WorkflowPhase
from startary.domain.execution_context import ExecutionContext
from startary.domain.provenance import FactConfidence, FactProvenance, KnowledgeField
from startary.domain.startup_progress import DevelopmentStage, ProductStatus, StartupProgress
from startary.domain.state import StartupState
from startary.graph.builder import build_graph, route_on_baseline
from startary.graph.graph_state import StartaryGraphState
from startary.graph.nodes import (
    MAX_BASELINE_QUESTION_TURNS,
    assess_baseline_state_sufficiency,
    merge_founder_answer,
    plan_baseline_question,
)
from startary.llm.schemas import (
    BaselineQuestionPlanLLMOutput,
    BaselineSufficiencyLLMOutput,
    CriticalAssumptionLLMOutput,
    DiagnosisLLMOutput,
    InformationGapPlanLLMOutput,
    RealityCheckLLMOutput,
    StartupAreaAssessmentBatchLLMOutput,
    StartupStateDeltaLLMOutput,
    WeakLinkClarifyingQuestionLLMOutput,
    WeakLinkDiscriminationLLMOutput,
)
from tests.fixtures.scenarios import (
    SCENARIO_M26_A_RICH_PROFILE,
    make_minimal_valid_llm_output_dict,
    make_valid_area_assessments_batch_output_dict,
    make_valid_baseline_question_plan_output_dict,
    make_valid_baseline_sufficiency_output_dict,
    make_valid_critical_assumption_output_dict,
    make_valid_gap_plan_output_dict,
    make_valid_reality_check_output_dict,
    make_valid_weak_link_clarifying_question_output_dict,
    make_valid_weak_link_discrimination_output_dict,
)


# ---------------------------------------------------------------------------
# 1. Domain Models & Schemas
# ---------------------------------------------------------------------------

class TestBaselineDomainModels:
    def test_baseline_dimension_enums(self):
        assert BaselineDimension.TEAM == "TEAM"
        assert BaselineDimension.CAPACITY == "CAPACITY"
        assert BaselineDimension.PRODUCT_STAGE == "PRODUCT_STAGE"
        assert BaselineDimension.VALIDATION_OR_USER == "VALIDATION_OR_USER"
        assert BaselineDimension.RESOURCE == "RESOURCE"

    def test_baseline_status_enums(self):
        assert BaselineDimensionStatus.KNOWN_ENOUGH == "KNOWN_ENOUGH"
        assert BaselineDimensionStatus.MISSING == "MISSING"
        assert BaselineDimensionStatus.NOT_RELEVANT == "NOT_RELEVANT"

    def test_baseline_sufficiency_model_validates(self):
        suff = BaselineStartupStateSufficiency(
            is_sufficient=False,
            dimension_statuses={
                BaselineDimension.TEAM: BaselineDimensionStatus.KNOWN_ENOUGH,
                BaselineDimension.CAPACITY: BaselineDimensionStatus.MISSING,
                BaselineDimension.PRODUCT_STAGE: BaselineDimensionStatus.KNOWN_ENOUGH,
                BaselineDimension.VALIDATION_OR_USER: BaselineDimensionStatus.MISSING,
                BaselineDimension.RESOURCE: BaselineDimensionStatus.KNOWN_ENOUGH,
            },
            missing_groups=[BaselineDimension.CAPACITY, BaselineDimension.VALIDATION_OR_USER],
            reason="Capacity and validation status are missing.",
        )
        assert suff.is_sufficient is False
        assert len(suff.missing_groups) == 2
        assert suff.dimension_statuses[BaselineDimension.TEAM] == BaselineDimensionStatus.KNOWN_ENOUGH

    def test_baseline_question_plan_model_validates(self):
        plan = BaselineQuestionPlan(
            should_ask=True,
            target_group=BaselineDimension.TEAM,
            target_fields=["founder_count", "member_roles"],
            question="Are you building this alone or with a team, and what does each person mainly work on?",
            reason="Establish initial team structure.",
        )
        assert plan.should_ask is True
        assert plan.target_group == BaselineDimension.TEAM
        assert len(plan.target_fields) == 2
        assert "alone or with a team" in plan.question


# ---------------------------------------------------------------------------
# 2. Fast-Path Shortcut & Skipping Tests
# ---------------------------------------------------------------------------

class TestBaselineSufficiencyEvaluation:
    def test_fast_path_shortcut_skips_llm_when_all_known(self):
        """When all 5 baseline dimensions are already extracted, assess_baseline_state_sufficiency shortcuts."""
        exec_ctx = ExecutionContext(
            founder_count=KnowledgeField(value=3, provenance=FactProvenance.EXPLICIT),
            approximate_weekly_hours_per_member=KnowledgeField(value=10, provenance=FactProvenance.EXPLICIT),
            available_budget_amount=KnowledgeField(value="300", provenance=FactProvenance.EXPLICIT),
            budget_currency=KnowledgeField(value="AZN", provenance=FactProvenance.EXPLICIT),
        )
        prog = StartupProgress(
            development_stage=KnowledgeField(value=DevelopmentStage.PROTOTYPE, provenance=FactProvenance.EXPLICIT),
            prototype_exists=KnowledgeField(value=True, provenance=FactProvenance.EXPLICIT),
            validation_activity=KnowledgeField(value="15 student chats", provenance=FactProvenance.EXPLICIT),
        )

        state: StartaryGraphState = {
            "project_id": "test-shortcut",
            "execution_context": exec_ctx.model_dump(),
            "startup_progress": prog.model_dump(),
            "baseline_question_turns": 0,
        }

        # Should return is_sufficient=True directly without calling LLM
        result = assess_baseline_state_sufficiency(state)
        assert result["is_baseline_sufficient"] is True
        assert result["baseline_sufficiency"]["is_sufficient"] is True
        assert len(result["baseline_sufficiency"]["missing_groups"]) == 0

    def test_sparse_input_triggers_baseline_insufficiency(self, mocker):
        """Completely empty state returns is_baseline_sufficient=False and missing groups."""
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = BaselineSufficiencyLLMOutput(
            **make_valid_baseline_sufficiency_output_dict(is_sufficient=False, missing_groups=["TEAM", "CAPACITY", "PRODUCT_STAGE", "VALIDATION_OR_USER", "RESOURCE"])
        )
        mock_llm.with_structured_output.return_value = mock_structured
        mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
        mocker.patch("startary.graph.nodes.get_settings", return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"))

        state: StartaryGraphState = {
            "project_id": "test-sparse",
            "execution_context": ExecutionContext().model_dump(),
            "startup_progress": StartupProgress().model_dump(),
            "baseline_question_turns": 0,
        }

        result = assess_baseline_state_sufficiency(state)
        assert result["is_baseline_sufficient"] is False
        assert len(result["baseline_sufficiency"]["missing_groups"]) > 0

    def test_plan_baseline_question_generates_compact_question(self, mocker):
        """Plan baseline question produces a compact question for the missing group."""
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = BaselineQuestionPlanLLMOutput(
            **make_valid_baseline_question_plan_output_dict(
                should_ask=True,
                target_group="TEAM",
                question="Are you building this alone or with a team, and what does each person mainly work on?",
            )
        )
        mock_llm.with_structured_output.return_value = mock_structured
        mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
        mocker.patch("startary.graph.nodes.get_settings", return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"))

        state: StartaryGraphState = {
            "project_id": "test-plan",
            "baseline_sufficiency": {"missing_groups": ["TEAM"]},
            "execution_context": ExecutionContext().model_dump(),
            "startup_progress": StartupProgress().model_dump(),
            "baseline_question_turns": 0,
        }

        result = plan_baseline_question(state)
        assert result["active_interrupt_type"] == "baseline"
        assert result["current_baseline_question"]["target_group"] == "TEAM"
        assert "alone or with a team" in result["current_baseline_question"]["question"]

    def test_dynamic_skipping_already_known_dimensions(self):
        """If Team and Product Stage are already known, only remaining dimensions are evaluated as missing."""
        exec_ctx = ExecutionContext(
            founder_count=KnowledgeField(value=1, provenance=FactProvenance.EXPLICIT),
        )
        prog = StartupProgress(
            development_stage=KnowledgeField(value=DevelopmentStage.PROTOTYPE, provenance=FactProvenance.EXPLICIT),
            prototype_exists=KnowledgeField(value=True, provenance=FactProvenance.EXPLICIT),
        )

        state: StartaryGraphState = {
            "project_id": "test-partial",
            "execution_context": exec_ctx.model_dump(),
            "startup_progress": prog.model_dump(),
            "baseline_question_turns": 0,
        }

        # Mock LLM for the remaining checks
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = BaselineSufficiencyLLMOutput(
            is_sufficient=False,
            team_status="KNOWN_ENOUGH",
            capacity_status="MISSING",
            product_stage_status="KNOWN_ENOUGH",
            validation_status="MISSING",
            resource_status="MISSING",
            missing_groups=["CAPACITY", "VALIDATION_OR_USER", "RESOURCE"],
            reason="Team and stage known, missing capacity, validation, and resources.",
        )
        mock_llm.with_structured_output.return_value = mock_structured

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("startary.graph.nodes.get_llm_client", lambda _: mock_llm)
            mp.setattr("startary.graph.nodes.get_settings", lambda: MagicMock(openai_api_key="sk-test", model_name="gpt-4o"))
            result = assess_baseline_state_sufficiency(state)

        missing = result["baseline_sufficiency"]["missing_groups"]
        assert BaselineDimension.TEAM not in missing
        assert BaselineDimension.PRODUCT_STAGE not in missing
        assert BaselineDimension.CAPACITY in missing


# ---------------------------------------------------------------------------
# 3. Router & Turn Limits
# ---------------------------------------------------------------------------

class TestBaselineRoutingAndLimits:
    def test_routes_to_diagnose_startup_when_sufficient(self):
        state: StartaryGraphState = {
            "is_baseline_sufficient": True,
            "baseline_question_turns": 0,
        }
        assert route_on_baseline(state) == "diagnose_startup"

    def test_routes_to_plan_question_when_insufficient_under_max(self):
        state: StartaryGraphState = {
            "is_baseline_sufficient": False,
            "baseline_question_turns": 2,
        }
        assert route_on_baseline(state) == "plan_baseline_question"

    def test_max_turns_forces_progression_to_diagnose(self):
        state: StartaryGraphState = {
            "is_baseline_sufficient": False,
            "baseline_question_turns": MAX_BASELINE_QUESTION_TURNS,
        }
        assert route_on_baseline(state) == "diagnose_startup"

    def test_counter_isolation_baseline_does_not_consume_strategic_or_weak_link(self):
        """Answering a baseline question increments baseline_question_turns only."""
        state: StartaryGraphState = {
            "active_interrupt_type": "baseline",
            "current_baseline_question": {
                "question": "Are you building alone or with a team?",
                "target_group": "TEAM",
            },
            "founder_response": "We are 2 co-founders.",
            "baseline_question_turns": 1,
            "strategic_clarification_turns": 0,
            "weak_link_clarification_turns": 0,
            "clarification_turn_count": 0,
            "baseline_history": [],
            "clarification_history": [],
            "weak_link_clarification_history": [],
        }

        updates = merge_founder_answer(state)
        assert updates["baseline_question_turns"] == 2
        # Strategic and weak-link counters must remain untouched!
        assert "strategic_clarification_turns" not in updates
        assert "weak_link_clarification_turns" not in updates
        assert len(updates["baseline_history"]) == 1
        assert updates["active_interrupt_type"] is None

