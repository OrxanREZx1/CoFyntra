"""
Tests: Rich Startup State Domain Models & Provenance — Milestone 2.6.

Tests:
  - FactProvenance enum (EXPLICIT, INFERRED, UNKNOWN, CONFLICTING)
  - FactConfidence enum (LOW, MEDIUM, HIGH)
  - KnowledgeField generic fact envelope
  - ExecutionContext domain model
  - StartupProgress domain model
  - InformationGapPlan domain model
  - StartupState integration with new layers
  - Overwrite rules and conflict handling
  - Serialization / deserialization roundtrip
"""

import pytest
from pydantic import ValidationError

from startary.domain.execution_context import ExecutionContext, TeamMember
from startary.domain.gap_plan import GapPriority, GapTargetDomain, InformationGapPlan
from startary.domain.provenance import (
    FactConfidence,
    FactProvenance,
    KnowledgeField,
    explicit_field,
    inferred_field,
    unknown_field,
)
from startary.domain.startup_progress import DevelopmentStage, ProductStatus, StartupProgress
from startary.domain.state import StartupState
from startary.domain.taxonomy import StartupArea


class TestProvenanceAndConfidenceEnums:
    def test_provenance_enum_values(self):
        assert FactProvenance.EXPLICIT == "EXPLICIT"
        assert FactProvenance.INFERRED == "INFERRED"
        assert FactProvenance.UNKNOWN == "UNKNOWN"
        assert FactProvenance.CONFLICTING == "CONFLICTING"

    def test_confidence_enum_values(self):
        assert FactConfidence.LOW == "LOW"
        assert FactConfidence.MEDIUM == "MEDIUM"
        assert FactConfidence.HIGH == "HIGH"


class TestKnowledgeField:
    def test_unknown_field_defaults(self):
        f = unknown_field()
        assert f.value is None
        assert f.provenance == FactProvenance.UNKNOWN
        assert f.confidence == FactConfidence.LOW
        assert f.source_turn is None
        assert f.is_known() is False

    def test_explicit_field_creation(self):
        f = explicit_field(value=3, confidence=FactConfidence.HIGH, source_turn=1)
        assert f.value == 3
        assert f.provenance == FactProvenance.EXPLICIT
        assert f.confidence == FactConfidence.HIGH
        assert f.source_turn == 1
        assert f.is_known() is True

    def test_inferred_field_creation(self):
        f = inferred_field(value="30", confidence=FactConfidence.MEDIUM, source_turn=1)
        assert f.value == "30"
        assert f.provenance == FactProvenance.INFERRED
        assert f.is_known() is True


class TestProvenanceMergeRules:
    """Test the overwrite and conflict resolution rules."""

    def test_explicit_overwrites_unknown(self):
        current = unknown_field()
        incoming = explicit_field(value="3", source_turn=1)
        assert current.can_be_overwritten_by(incoming) is True

    def test_explicit_overwrites_inferred(self):
        current = inferred_field(value="2", source_turn=1)
        incoming = explicit_field(value="3", source_turn=1)
        assert current.can_be_overwritten_by(incoming) is True

    def test_inferred_cannot_overwrite_explicit(self):
        current = explicit_field(value="3", source_turn=1)
        incoming = inferred_field(value="4", source_turn=2)
        assert current.can_be_overwritten_by(incoming) is False

    def test_newer_explicit_overwrites_older_explicit(self):
        current = explicit_field(value="2", source_turn=1)
        incoming = explicit_field(value="3", source_turn=2)
        assert current.can_be_overwritten_by(incoming) is True

    def test_unknown_cannot_overwrite_known(self):
        current = explicit_field(value="3", source_turn=1)
        incoming = unknown_field()
        assert current.can_be_overwritten_by(incoming) is False


class TestExecutionContextDomain:
    def test_default_execution_context_is_all_unknown(self):
        ctx = ExecutionContext()
        assert ctx.founder_count.provenance == FactProvenance.UNKNOWN
        assert ctx.active_team_size.provenance == FactProvenance.UNKNOWN
        assert ctx.available_budget_amount.provenance == FactProvenance.UNKNOWN
        assert ctx.funding_status.provenance == FactProvenance.UNKNOWN
        assert ctx.customer_access.provenance == FactProvenance.UNKNOWN
        assert ctx.team_members == []
        assert ctx.existing_assets == []

    def test_team_member_model(self):
        tm = TeamMember(role="Lead Developer", capabilities=["Python", "AI"], weekly_hours=15.0)
        assert tm.role == "Lead Developer"
        assert len(tm.capabilities) == 2
        assert tm.weekly_hours == 15.0

    def test_populated_execution_context(self):
        ctx = ExecutionContext(
            founder_count=explicit_field(3, FactConfidence.HIGH, source_turn=1),
            total_weekly_capacity_hours=inferred_field(30.0, FactConfidence.HIGH, source_turn=1),
            available_budget_amount=explicit_field(300.0, FactConfidence.HIGH, source_turn=1),
            budget_currency=explicit_field("AZN", FactConfidence.HIGH, source_turn=1),
            budget_period=explicit_field("month", FactConfidence.HIGH, source_turn=1),
            existing_assets=["prototype", "telegram_community"],
        )
        assert ctx.founder_count.value == 3
        assert ctx.founder_count.provenance == FactProvenance.EXPLICIT
        assert ctx.total_weekly_capacity_hours.provenance == FactProvenance.INFERRED
        assert ctx.budget_currency.value == "AZN"
        assert "prototype" in ctx.existing_assets


class TestStartupProgressDomain:
    def test_default_progress_is_unknown(self):
        prog = StartupProgress()
        assert prog.development_stage.provenance == FactProvenance.UNKNOWN
        assert prog.product_status.provenance == FactProvenance.UNKNOWN
        assert prog.mvp_exists.provenance == FactProvenance.UNKNOWN
        assert prog.active_users.provenance == FactProvenance.UNKNOWN
        assert prog.revenue.provenance == FactProvenance.UNKNOWN

    def test_development_stage_enum(self):
        assert DevelopmentStage.RAW_IDEA == "RAW_IDEA"
        assert DevelopmentStage.PROTOTYPE == "PROTOTYPE"
        assert DevelopmentStage.PRE_MVP == "PRE_MVP"
        assert DevelopmentStage.MVP == "MVP"
        assert DevelopmentStage.EARLY_USAGE == "EARLY_USAGE"
        assert DevelopmentStage.GROWTH == "GROWTH"

    def test_product_status_enum(self):
        assert ProductStatus.NOTHING == "NOTHING"
        assert ProductStatus.CONCEPT == "CONCEPT"
        assert ProductStatus.PROTOTYPE == "PROTOTYPE"
        assert ProductStatus.MVP_BUILT == "MVP_BUILT"
        assert ProductStatus.LAUNCHED == "LAUNCHED"

    def test_mvp_false_does_not_require_specific_stage(self):
        prog = StartupProgress(
            mvp_exists=explicit_field(False, FactConfidence.HIGH, source_turn=1),
            development_stage=unknown_field(),
        )
        assert prog.mvp_exists.value is False
        assert prog.mvp_exists.provenance == FactProvenance.EXPLICIT
        assert prog.development_stage.provenance == FactProvenance.UNKNOWN


class TestInformationGapPlanDomain:
    def test_gap_plan_validates(self):
        plan = InformationGapPlan(
            should_ask=True,
            strategic_sufficient=False,
            target_domain=GapTargetDomain.STRATEGIC,
            target_field_or_area="Problem",
            priority=GapPriority.REQUIRED_NOW,
            information_goal="Understand specific student friction",
            reason="Core problem lacks concrete observations.",
            question="What specific difficulty do students report when picking their branch?",
        )
        assert plan.should_ask is True
        assert plan.strategic_sufficient is False
        assert plan.target_domain == GapTargetDomain.STRATEGIC
        assert plan.priority == GapPriority.REQUIRED_NOW
        assert len(plan.question) > 10

    def test_gap_plan_proceed(self):
        plan = InformationGapPlan(
            should_ask=False,
            strategic_sufficient=True,
            reason="Context is grounded enough to proceed.",
        )
        assert plan.should_ask is False
        assert plan.strategic_sufficient is True
        assert plan.question is None


class TestStartupStateIntegration:
    def test_startup_state_includes_execution_and_progress_layers(self):
        state = StartupState(raw_context="Testing rich startup state.")
        assert state.execution_context is not None
        assert state.progress is not None
        assert state.areas == {}

    def test_startup_state_model_dump_roundtrip(self):
        state = StartupState(
            raw_context="My startup context.",
            execution_context=ExecutionContext(
                founder_count=explicit_field(2, FactConfidence.HIGH, source_turn=1)
            ),
            progress=StartupProgress(
                mvp_exists=explicit_field(False, FactConfidence.HIGH, source_turn=1)
            ),
        )
        data = state.model_dump()
        rebuilt = StartupState(**data)
        assert rebuilt.execution_context.founder_count.value == 2
        assert rebuilt.progress.mvp_exists.value is False
        assert rebuilt.raw_context == "My startup context."

