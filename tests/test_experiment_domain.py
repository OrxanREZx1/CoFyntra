"""
Tests: Milestone 3 Experiment Domain Models and Enums.

Verification:
- Strict Assumption linkage (assumption_id required).
- ExperimentType enum coverage.
- ExperimentStatus READY upon saving.
- Valid and invalid inference boundary fields.
- 3-directional interpretation signals (support, contradiction, inconclusive).
- Time and cost aliases synchronization.
- ExperimentParameter with ParameterProvenance (AI_PROPOSED vs FOUNDER_SUPPLIED).
- ExperimentCriticReport with 6 evaluation dimensions.
- CanExecuteAssessment with ExecutionReadiness.
"""

import pytest
from pydantic import ValidationError

from startary.domain.enums import (
    CriticDimensionResult,
    CriticVerdict,
    ExecutionReadiness,
    ExperimentStatus,
    ExperimentType,
    ParameterProvenance,
)
from startary.domain.experiment import (
    CanExecuteAssessment,
    Experiment,
    ExperimentCriticReport,
    ExperimentParameter,
)


class TestExperimentEnums:
    def test_all_experiment_types_present(self):
        expected = [
            "CUSTOMER_INTERVIEW",
            "PROBLEM_INTERVIEW",
            "OBSERVATION",
            "CONCIERGE_TEST",
            "FAKE_DOOR",
            "LANDING_PAGE",
            "PROTOTYPE_TEST",
            "MANUAL_SERVICE",
            "PREORDER_OR_COMMITMENT",
            "PRICING_TEST",
            "OUTREACH_TEST",
            "CHANNEL_TEST",
            "OTHER",
        ]
        for exp_type in expected:
            assert exp_type in ExperimentType.__members__

    def test_parameter_provenance_enums(self):
        assert ParameterProvenance.AI_PROPOSED == "AI_PROPOSED"
        assert ParameterProvenance.FOUNDER_SUPPLIED == "FOUNDER_SUPPLIED"

    def test_critic_verdict_and_dimensions(self):
        assert CriticVerdict.ACCEPT == "ACCEPT"
        assert CriticVerdict.REVISE == "REVISE"
        assert CriticDimensionResult.PASS == "PASS"
        assert CriticDimensionResult.FAIL == "FAIL"

    def test_execution_readiness_enums(self):
        assert ExecutionReadiness.CAN_EXECUTE == "CAN_EXECUTE"
        assert ExecutionReadiness.NEEDS_HELP == "NEEDS_HELP"
        assert ExecutionReadiness.BLOCKED == "BLOCKED"


class TestExperimentModel:
    def test_experiment_requires_assumption_id(self):
        with pytest.raises(ValidationError):
            Experiment(
                project_id="proj-123",
                objective="Test problem severity",
                assumption_id="",  # empty
            )

    def test_valid_experiment_creation(self):
        exp = Experiment(
            assumption_id="asm-123",
            project_id="proj-456",
            experiment_type=ExperimentType.PROBLEM_INTERVIEW,
            objective="Determine if boutique owners actively seek automated packaging.",
            hypothesis="Boutique owners spend over 2 hours weekly on manual packaging.",
            method="Conduct 6-8 retrospective interviews with local boutique owners.",
            founder_actions=[
                "List 10 candidate boutique shops.",
                "Visit 6 shops during afternoon hours.",
                "Ask open questions regarding packaging workflows.",
            ],
            evidence_to_collect="Documented hours spent packaging and specific delays experienced.",
            support_signal="At least 5 of 8 owners report packaging delays causing missed shipments.",
            contradiction_signal="Fewer than 2 owners report packaging as a meaningful friction.",
            inconclusive_signal="Owners report packaging is trivial or completed by casual staff without issues.",
            valid_inferences=[
                "Whether packaging is a recurring operational bottleneck.",
                "What manual tools they currently use.",
            ],
            invalid_inferences=[
                "Whether they will purchase our proprietary hardware box.",
                "Long-term customer lifetime value.",
            ],
            estimated_time="3 days",
            estimated_cost="$0",
            status=ExperimentStatus.READY,
        )
        assert exp.assumption_id == "asm-123"
        assert exp.experiment_type == ExperimentType.PROBLEM_INTERVIEW
        assert exp.status == ExperimentStatus.READY
        assert len(exp.valid_inferences) == 2
        assert len(exp.invalid_inferences) == 2
        assert exp.time_constraint == "3 days"
        assert exp.cost_constraint == "$0"

    def test_time_and_cost_alias_sync(self):
        exp = Experiment(
            assumption_id="asm-123",
            project_id="proj-456",
            objective="Test demand via landing page",
            time_constraint="1 week",
            cost_constraint="$50",
        )
        assert exp.estimated_time == "1 week"
        assert exp.estimated_cost == "$50"


class TestExperimentParameterModel:
    def test_parameter_creation_with_provenance(self):
        param = ExperimentParameter(
            name="sample_size",
            value="6-8 sellers",
            provenance=ParameterProvenance.AI_PROPOSED,
            notes="Standard exploratory interview threshold.",
        )
        assert param.name == "sample_size"
        assert param.value == "6-8 sellers"
        assert param.provenance == ParameterProvenance.AI_PROPOSED


class TestExperimentCriticReportModel:
    def test_critic_report_validation(self):
        report = ExperimentCriticReport(
            verdict=CriticVerdict.ACCEPT,
            issues=[],
            assumption_alignment=CriticDimensionResult.PASS,
            evidence_validity=CriticDimensionResult.PASS,
            feasibility=CriticDimensionResult.PASS,
            cost_efficiency=CriticDimensionResult.PASS,
            falsifiability=CriticDimensionResult.PASS,
            scope=CriticDimensionResult.PASS,
            rationale="Experiment is tightly aligned, cheap, and falsifiable.",
        )
        assert report.verdict == CriticVerdict.ACCEPT
        assert report.assumption_alignment == CriticDimensionResult.PASS
        assert len(report.issues) == 0


class TestCanExecuteAssessmentModel:
    def test_can_execute_model_validation(self):
        assessment = CanExecuteAssessment(
            readiness=ExecutionReadiness.NEEDS_HELP,
            reason="Requires access to boutique owners which founder has not established.",
            missing_requirements=["Customer access to boutique owners"],
            blockers=[],
        )
        assert assessment.readiness == ExecutionReadiness.NEEDS_HELP
        assert len(assessment.missing_requirements) == 1

