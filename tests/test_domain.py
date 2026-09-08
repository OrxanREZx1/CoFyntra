"""
Tests: core domain schemas — updated for Milestone 1.

Changes from M0:
  - StartupState now uses AreaKnowledge (not raw strings)
  - AreaStatus enum tested
  - AreaKnowledge validation tested
  - Other domain schemas (Assumption, Experiment, Evidence) unchanged
"""

import uuid

import pytest
from pydantic import ValidationError

from startary.domain.area_knowledge import AreaKnowledge, AreaStatus
from startary.domain.assessment import StartupAreaAssessment
from startary.domain.assumption import Assumption
from startary.domain.enums import (
    AssumptionStatus,
    EvidenceSourceType,
    ExperimentStatus,
    WorkflowPhase,
)
from startary.domain.evidence import Evidence
from startary.domain.experiment import Experiment
from startary.domain.project import StartupProject
from startary.domain.state import StartupState
from startary.domain.taxonomy import StartupArea


def _project_id() -> str:
    return str(uuid.uuid4())


def _assumption_id() -> str:
    return str(uuid.uuid4())


# ── AreaStatus ────────────────────────────────────────────────────────────────

class TestAreaStatus:
    def test_all_statuses_present(self):
        assert AreaStatus.UNKNOWN == "UNKNOWN"
        assert AreaStatus.PARTIALLY_KNOWN == "PARTIALLY_KNOWN"
        assert AreaStatus.KNOWN == "KNOWN"
        assert AreaStatus.UNCLEAR == "UNCLEAR"

    def test_area_status_is_str(self):
        assert isinstance(AreaStatus.UNKNOWN, str)

    def test_invalid_status_rejected(self):
        with pytest.raises(ValueError):
            AreaKnowledge(status="INVENTED_STATUS")


# ── AreaKnowledge ─────────────────────────────────────────────────────────────

class TestAreaKnowledge:
    def test_default_is_unknown(self):
        k = AreaKnowledge()
        assert k.status == AreaStatus.UNKNOWN
        assert k.understanding is None
        assert k.founder_claims == []
        assert k.evidence_provided == []
        assert k.missing_information is None

    def test_valid_partially_known(self):
        k = AreaKnowledge(
            status=AreaStatus.PARTIALLY_KNOWN,
            understanding="Founders struggle with prioritization.",
            founder_claims=["Everyone needs this"],
            evidence_provided=["8 interviews conducted"],
            missing_information="Frequency of the problem.",
        )
        assert k.status == AreaStatus.PARTIALLY_KNOWN
        assert len(k.founder_claims) == 1
        assert len(k.evidence_provided) == 1

    def test_known_with_no_claims(self):
        k = AreaKnowledge(
            status=AreaStatus.KNOWN,
            understanding="Founder has 2 teammates, no marketing budget.",
            evidence_provided=["Directly stated by founder"],
        )
        assert k.status == AreaStatus.KNOWN
        assert k.founder_claims == []

    def test_unclear_preserves_ambiguity(self):
        k = AreaKnowledge(
            status=AreaStatus.UNCLEAR,
            understanding="Founder mentioned both students and accelerator founders.",
            missing_information="Which segment is the actual target.",
        )
        assert k.status == AreaStatus.UNCLEAR


# ── StartupState (M1) ─────────────────────────────────────────────────────────

class TestStartupState:
    def test_empty_state_is_valid(self):
        s = StartupState()
        assert s.is_empty()
        assert s.missing_areas() == list(StartupArea)
        assert s.raw_context is None

    def test_raw_context_stored_separately(self):
        """Raw context must NOT be stored as a startup area value."""
        s = StartupState(raw_context="I want to build an AI app for restaurants.")
        # raw_context is in state
        assert s.raw_context == "I want to build an AI app for restaurants."
        # But it is NOT assigned to any startup area
        assert s.is_empty()
        for area in StartupArea:
            assert area not in s.areas

    def test_area_knowledge_stored_correctly(self):
        k = AreaKnowledge(
            status=AreaStatus.PARTIALLY_KNOWN,
            understanding="Student founders struggle with priorities.",
        )
        s = StartupState(areas={StartupArea.PROBLEM: k})
        assert StartupArea.PROBLEM in s.known_areas()
        assert len(s.missing_areas()) == 9

    def test_known_areas_excludes_unknown(self):
        areas = {
            StartupArea.PROBLEM: AreaKnowledge(status=AreaStatus.KNOWN),
            StartupArea.TARGET_USER: AreaKnowledge(status=AreaStatus.UNKNOWN),
            StartupArea.PAIN_URGENCY: AreaKnowledge(status=AreaStatus.PARTIALLY_KNOWN),
        }
        s = StartupState(areas=areas)
        known = s.known_areas()
        assert StartupArea.PROBLEM in known
        assert StartupArea.PAIN_URGENCY in known
        assert StartupArea.TARGET_USER not in known

    def test_unknown_areas_method(self):
        areas = {
            StartupArea.PROBLEM: AreaKnowledge(status=AreaStatus.UNKNOWN),
        }
        s = StartupState(areas=areas)
        assert StartupArea.PROBLEM in s.unknown_areas()

    def test_unclear_areas_method(self):
        areas = {
            StartupArea.TARGET_USER: AreaKnowledge(status=AreaStatus.UNCLEAR),
        }
        s = StartupState(areas=areas)
        assert StartupArea.TARGET_USER in s.unclear_areas()

    def test_get_area_returns_unknown_for_missing(self):
        s = StartupState()
        k = s.get_area(StartupArea.PROBLEM)
        assert k.status == AreaStatus.UNKNOWN

    def test_raw_context_not_required(self):
        s = StartupState()
        assert s.raw_context is None


# ── StartupProject ────────────────────────────────────────────────────────────

class TestStartupProject:
    def test_valid_project_creation(self):
        p = StartupProject(title="TestCo")
        assert p.title == "TestCo"
        assert p.phase == WorkflowPhase.UNDERSTAND
        assert p.iteration == 1
        assert len(p.id) == 36

    def test_empty_title_rejected(self):
        with pytest.raises(ValidationError):
            StartupProject(title="")

    def test_iteration_must_be_positive(self):
        with pytest.raises(ValidationError):
            StartupProject(title="X", iteration=0)

    def test_invalid_phase_rejected(self):
        with pytest.raises(ValidationError):
            StartupProject(title="X", phase="INVALID_PHASE")


# ── Assumption ────────────────────────────────────────────────────────────────

class TestAssumption:
    def test_valid_assumption(self):
        a = Assumption(
            project_id=_project_id(),
            statement="Founders lack structured feedback on their assumptions.",
            area=StartupArea.PROBLEM,
            iteration_created=1,
        )
        assert a.status == AssumptionStatus.UNTESTED
        assert a.is_critical is False

    def test_all_valid_statuses_accepted(self):
        pid = _project_id()
        for status in AssumptionStatus:
            a = Assumption(
                project_id=pid,
                statement="Valid statement for testing.",
                area=StartupArea.DEMAND_SIGNAL,
                status=status,
                iteration_created=1,
            )
            assert a.status == status

    def test_invalid_status_rejected(self):
        with pytest.raises(ValidationError):
            Assumption(
                project_id=_project_id(),
                statement="Some statement.",
                area=StartupArea.PROBLEM,
                status="MAYBE",
                iteration_created=1,
            )

    def test_invalid_area_rejected(self):
        with pytest.raises(ValidationError):
            Assumption(
                project_id=_project_id(),
                statement="Some statement.",
                area="Made Up Area",
                iteration_created=1,
            )

    def test_short_statement_rejected(self):
        with pytest.raises(ValidationError):
            Assumption(
                project_id=_project_id(),
                statement="Hi",
                area=StartupArea.PROBLEM,
                iteration_created=1,
            )

    def test_iteration_must_be_positive(self):
        with pytest.raises(ValidationError):
            Assumption(
                project_id=_project_id(),
                statement="Valid statement here.",
                area=StartupArea.PROBLEM,
                iteration_created=0,
            )


# ── Experiment ────────────────────────────────────────────────────────────────

class TestExperiment:
    def test_valid_experiment_requires_assumption_id(self):
        exp = Experiment(
            assumption_id=_assumption_id(),
            project_id=_project_id(),
            objective="Validate that founders will pay for structured advice.",
        )
        assert exp.status == ExperimentStatus.DRAFT
        assert exp.assumption_id is not None

    def test_experiment_without_assumption_id_rejected(self):
        with pytest.raises(ValidationError):
            Experiment(
                assumption_id="",
                project_id=_project_id(),
                objective="Some objective here.",
            )

    def test_experiment_missing_assumption_id_field_rejected(self):
        with pytest.raises((ValidationError, TypeError)):
            Experiment(
                project_id=_project_id(),
                objective="Some objective here.",
            )

    def test_short_objective_rejected(self):
        with pytest.raises(ValidationError):
            Experiment(
                assumption_id=_assumption_id(),
                project_id=_project_id(),
                objective="No",
            )


# ── Evidence ──────────────────────────────────────────────────────────────────

class TestEvidence:
    def test_valid_founder_evidence(self):
        e = Evidence(
            project_id=_project_id(),
            source_type=EvidenceSourceType.FOUNDER_PROVIDED,
            content="Interviewed 5 founders; all confirmed the pain point.",
        )
        assert e.source_type == EvidenceSourceType.FOUNDER_PROVIDED
        from startary.domain.enums import EvidenceResult
        assert e.result == EvidenceResult.PENDING

    def test_valid_external_evidence(self):
        e = Evidence(
            project_id=_project_id(),
            source_type=EvidenceSourceType.EXTERNALLY_RETRIEVED,
            content="CB Insights report: 42% of startups fail due to no market need.",
        )
        assert e.source_type == EvidenceSourceType.EXTERNALLY_RETRIEVED

    def test_short_content_rejected(self):
        with pytest.raises(ValidationError):
            Evidence(
                project_id=_project_id(),
                source_type=EvidenceSourceType.FOUNDER_PROVIDED,
                content="ok",
            )

    def test_invalid_source_type_rejected(self):
        with pytest.raises(ValidationError):
            Evidence(
                project_id=_project_id(),
                source_type="INVENTED",
                content="Some content here that is long enough.",
            )

    def test_evidence_with_assumption_linkage(self):
        aid = _assumption_id()
        e = Evidence(
            project_id=_project_id(),
            source_type=EvidenceSourceType.FOUNDER_PROVIDED,
            content="Customer interviews confirmed the assumption.",
            assumption_id=aid,
        )
        assert e.assumption_id == aid


# ── StartupAreaAssessment (M2) ───────────────────────────────────────────────

class TestStartupAreaAssessment:
    def test_minimal_assessment(self):
        a = StartupAreaAssessment(area=StartupArea.PROBLEM)
        assert a.area == StartupArea.PROBLEM
        assert a.importance is None
        assert a.uncertainty is None
        assert a.status is None

    def test_assessment_with_status(self):
        a = StartupAreaAssessment(
            area=StartupArea.PROBLEM,
            status=AreaStatus.PARTIALLY_KNOWN,
            summary="Some understanding exists.",
        )
        assert a.status == AreaStatus.PARTIALLY_KNOWN

    def test_valid_full_assessment(self):
        from startary.domain.assessment import RiskFactorLevel
        a = StartupAreaAssessment(
            area=StartupArea.DEMAND_SIGNAL,
            status=AreaStatus.UNKNOWN,
            summary="No demand signals identified.",
            importance=RiskFactorLevel.HIGH,
            uncertainty=RiskFactorLevel.HIGH,
            evidence_gap=RiskFactorLevel.HIGH,
            consequence_if_wrong=RiskFactorLevel.HIGH,
            rationale="No customer interviews conducted yet.",
        )
        assert a.importance == RiskFactorLevel.HIGH
        assert a.uncertainty == RiskFactorLevel.HIGH
        score = a.calculate_risk()
        assert score == 81
        assert a.calculated_decision_risk == 81


# ── RealityCheck (M2) ─────────────────────────────────────────────────────────

class TestRealityCheckDomain:
    def test_valid_reality_check_creation(self):
        from startary.domain.reality_check import EvidenceConclusion, RealityCheck
        rc = RealityCheck(
            weak_link_area=StartupArea.PROBLEM,
            relevant_founder_claims=["Students need this."],
            available_founder_evidence=["8 interviews"],
            what_evidence_supports=["Problem exists."],
            what_evidence_does_not_support=["Willingness to pay."],
            conclusion=EvidenceConclusion.PARTIALLY_SUPPORTED,
            unresolved_uncertainty="Severity of problem.",
            summary="8 interviews validate problem existence but not willingness to pay.",
        )
        assert rc.weak_link_area == StartupArea.PROBLEM
        assert rc.conclusion == EvidenceConclusion.PARTIALLY_SUPPORTED
        assert len(rc.what_evidence_supports) == 1
