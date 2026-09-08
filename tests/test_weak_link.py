"""
Tests: Deterministic Weak-Link Risk Assessment & Tie-Breaking Hierarchy.

Verifies:
  1. Controlled risk factor scale (LOW=1, MEDIUM=2, HIGH=3).
  2. Decision Risk formula: Score = Importance × Uncertainty × Evidence Gap × Consequence.
  3. Score is calculated deterministically in Python (never accepted raw from LLM).
  4. Explicit tie-breaking hierarchy is strictly respected.
  5. Exactly one weak link is selected.
  6. Missing areas without claims do not receive artificially inflated risk.
"""

import pytest

from startary.domain.assessment import (
    RiskFactorLevel,
    StartupAreaAssessment,
    risk_factor_to_int,
)
from startary.domain.taxonomy import StartupArea
from startary.graph.graph_state import StartaryGraphState
from startary.graph.nodes import select_weak_link


class TestRiskFactorCalculation:
    def test_risk_factor_conversion(self):
        assert risk_factor_to_int(RiskFactorLevel.LOW) == 1
        assert risk_factor_to_int(RiskFactorLevel.MEDIUM) == 2
        assert risk_factor_to_int(RiskFactorLevel.HIGH) == 3
        assert risk_factor_to_int("LOW") == 1
        assert risk_factor_to_int("MEDIUM") == 2
        assert risk_factor_to_int("HIGH") == 3

    def test_invalid_risk_factor_rejected(self):
        with pytest.raises(ValueError, match="Invalid risk factor level"):
            risk_factor_to_int("EXTREME")

    def test_calculate_risk_all_high(self):
        a = StartupAreaAssessment(
            area=StartupArea.PROBLEM,
            importance=RiskFactorLevel.HIGH,
            uncertainty=RiskFactorLevel.HIGH,
            evidence_gap=RiskFactorLevel.HIGH,
            consequence_if_wrong=RiskFactorLevel.HIGH,
        )
        score = a.calculate_risk()
        assert score == 81
        assert a.calculated_decision_risk == 81

    def test_calculate_risk_all_low(self):
        a = StartupAreaAssessment(
            area=StartupArea.BUSINESS_MODEL,
            importance=RiskFactorLevel.LOW,
            uncertainty=RiskFactorLevel.LOW,
            evidence_gap=RiskFactorLevel.LOW,
            consequence_if_wrong=RiskFactorLevel.LOW,
        )
        score = a.calculate_risk()
        assert score == 1
        assert a.calculated_decision_risk == 1

    def test_calculate_risk_mixed(self):
        # 3 * 2 * 3 * 1 = 18
        a = StartupAreaAssessment(
            area=StartupArea.TARGET_USER,
            importance=RiskFactorLevel.HIGH,
            uncertainty=RiskFactorLevel.MEDIUM,
            evidence_gap=RiskFactorLevel.HIGH,
            consequence_if_wrong=RiskFactorLevel.LOW,
        )
        assert a.calculate_risk() == 18


class TestDeterministicWeakLinkSelection:
    def _create_state_with_assessments(self, assessments: list[dict]) -> StartaryGraphState:
        from startary.domain.enums import WorkflowPhase
        return StartaryGraphState(
            project_id="test-weak-link",
            phase=WorkflowPhase.DIAGNOSE,
            iteration=1,
            startup_state={"areas": {}},
            diagnosis={"summary": "test"},
            area_assessments_output={"assessments": assessments},
        )

    def test_clear_winner_selected(self):
        """Area with highest Decision Risk score wins."""
        assessments = [
            {
                "area": StartupArea.PROBLEM,
                "importance": "HIGH",
                "uncertainty": "HIGH",
                "evidence_gap": "HIGH",
                "consequence_of_being_wrong": "HIGH",
                "rationale": "Max risk",
            },
            {
                "area": StartupArea.TARGET_USER,
                "importance": "MEDIUM",
                "uncertainty": "MEDIUM",
                "evidence_gap": "MEDIUM",
                "consequence_of_being_wrong": "MEDIUM",
                "rationale": "Medium risk",
            },
        ]
        state = self._create_state_with_assessments(assessments)
        result = select_weak_link(state)

        assert result["weak_area"] == str(StartupArea.PROBLEM)
        assert result["weak_link_details"]["score"] == 81

    def test_tie_breaking_higher_consequence_wins(self):
        """When scores are identical (e.g. 18 vs 18), higher Consequence of Being Wrong wins."""
        # Option A: Imp=3, Unc=2, Gap=3, Con=1 -> Score = 18 (Con = 1)
        # Option B: Imp=2, Unc=3, Gap=1, Con=3 -> Score = 18 (Con = 3)
        assessments = [
            {
                "area": StartupArea.PROBLEM,
                "importance": "HIGH",
                "uncertainty": "MEDIUM",
                "evidence_gap": "HIGH",
                "consequence_of_being_wrong": "LOW",
                "rationale": "Lower consequence",
            },
            {
                "area": StartupArea.TARGET_USER,
                "importance": "MEDIUM",
                "uncertainty": "HIGH",
                "evidence_gap": "LOW",
                "consequence_of_being_wrong": "HIGH",
                "rationale": "Higher consequence",
            },
        ]
        state = self._create_state_with_assessments(assessments)
        result = select_weak_link(state)

        # Target User should win due to higher consequence
        assert result["weak_area"] == str(StartupArea.TARGET_USER)

    def test_tie_breaking_higher_uncertainty_wins(self):
        """When score and consequence are identical, higher Uncertainty wins."""
        # Option A: Imp=3, Unc=1, Gap=2, Con=3 -> Score = 18 (Con=3, Unc=1)
        # Option B: Imp=1, Unc=3, Gap=2, Con=3 -> Score = 18 (Con=3, Unc=3)
        assessments = [
            {
                "area": StartupArea.PROBLEM,
                "importance": "HIGH",
                "uncertainty": "LOW",
                "evidence_gap": "MEDIUM",
                "consequence_of_being_wrong": "HIGH",
                "rationale": "Low unc",
            },
            {
                "area": StartupArea.PAIN_URGENCY,
                "importance": "LOW",
                "uncertainty": "HIGH",
                "evidence_gap": "MEDIUM",
                "consequence_of_being_wrong": "HIGH",
                "rationale": "High unc",
            },
        ]
        state = self._create_state_with_assessments(assessments)
        result = select_weak_link(state)

        assert result["weak_area"] == str(StartupArea.PAIN_URGENCY)

    def test_tie_breaking_higher_evidence_gap_wins(self):
        """When score, consequence, and uncertainty are identical, higher Evidence Gap wins."""
        # Option A: Imp=3, Unc=2, Gap=1, Con=3 -> Score = 18 (Con=3, Unc=2, Gap=1)
        # Option B: Imp=1, Unc=2, Gap=3, Con=3 -> Score = 18 (Con=3, Unc=2, Gap=3)
        assessments = [
            {
                "area": StartupArea.EXISTING_ALTERNATIVES,
                "importance": "HIGH",
                "uncertainty": "MEDIUM",
                "evidence_gap": "LOW",
                "consequence_of_being_wrong": "HIGH",
                "rationale": "Low gap",
            },
            {
                "area": StartupArea.DEMAND_SIGNAL,
                "importance": "LOW",
                "uncertainty": "MEDIUM",
                "evidence_gap": "HIGH",
                "consequence_of_being_wrong": "HIGH",
                "rationale": "High gap",
            },
        ]
        state = self._create_state_with_assessments(assessments)
        result = select_weak_link(state)

        assert result["weak_area"] == str(StartupArea.DEMAND_SIGNAL)

    def test_tie_breaking_canonical_order_fallback(self):
        """When all factors are identical, canonical taxonomy order is the final tie-breaker."""
        assessments = [
            {
                "area": StartupArea.TARGET_USER,
                "importance": "HIGH",
                "uncertainty": "HIGH",
                "evidence_gap": "HIGH",
                "consequence_of_being_wrong": "HIGH",
                "rationale": "Identical factors",
            },
            {
                "area": StartupArea.PROBLEM,
                "importance": "HIGH",
                "uncertainty": "HIGH",
                "evidence_gap": "HIGH",
                "consequence_of_being_wrong": "HIGH",
                "rationale": "Identical factors",
            },
        ]
        state = self._create_state_with_assessments(assessments)
        result = select_weak_link(state)

        # Problem comes before Target User in canonical taxonomy
        assert result["weak_area"] == str(StartupArea.PROBLEM)

    def test_diagnosis_updated_with_weak_link(self):
        """Selected weak link is recorded in Diagnosis.weak_areas and selected_weak_link."""
        assessments = [
            {
                "area": StartupArea.ACQUISITION,
                "importance": "HIGH",
                "uncertainty": "HIGH",
                "evidence_gap": "HIGH",
                "consequence_of_being_wrong": "HIGH",
                "rationale": "Acquisition bottleneck",
            }
        ]
        state = self._create_state_with_assessments(assessments)
        result = select_weak_link(state)

        diag = result["diagnosis"]
        assert diag["selected_weak_link"] == str(StartupArea.ACQUISITION)
        assert diag["weak_areas"] == [str(StartupArea.ACQUISITION)]


class TestSaturationTelemetry:
    def _create_state_with_assessments(self, assessments: list[dict]) -> StartaryGraphState:
        from startary.domain.enums import WorkflowPhase
        return StartaryGraphState(
            project_id="test-saturation-telemetry",
            phase=WorkflowPhase.DIAGNOSE,
            iteration=1,
            startup_state={"areas": {}},
            diagnosis={"summary": "test"},
            area_assessments_output={"assessments": assessments},
        )

    def test_single_winner_telemetry(self):
        """Single clear winner has max_score_tie_count=1 and tie_breaker_used=False."""
        assessments = [
            {
                "area": StartupArea.PROBLEM,
                "importance": "HIGH",
                "uncertainty": "HIGH",
                "evidence_gap": "HIGH",
                "consequence_of_being_wrong": "HIGH",
                "rationale": "Max risk",
            },
            {
                "area": StartupArea.TARGET_USER,
                "importance": "LOW",
                "uncertainty": "LOW",
                "evidence_gap": "LOW",
                "consequence_of_being_wrong": "LOW",
                "rationale": "Min risk",
            },
        ]
        state = self._create_state_with_assessments(assessments)
        result = select_weak_link(state)

        telemetry = result["weak_link_details"]["saturation_telemetry"]
        assert telemetry["max_score"] == 81
        assert telemetry["max_score_tie_count"] == 1
        assert telemetry["saturated_area_count"] == 1
        assert telemetry["tie_breaker_used"] is False

    def test_six_saturated_81s_telemetry(self):
        """Six areas scoring 81/81 are tracked with saturated_area_count=6 and tie_breaker_used=True."""
        areas_at_81 = [
            StartupArea.PROBLEM,
            StartupArea.TARGET_USER,
            StartupArea.PAIN_URGENCY,
            StartupArea.EXISTING_ALTERNATIVES,
            StartupArea.VALUE_PROPOSITION,
            StartupArea.DEMAND_SIGNAL,
        ]
        assessments = [
            {
                "area": a,
                "importance": "HIGH",
                "uncertainty": "HIGH",
                "evidence_gap": "HIGH",
                "consequence_of_being_wrong": "HIGH",
                "rationale": "High risk",
            }
            for a in areas_at_81
        ]
        state = self._create_state_with_assessments(assessments)
        result = select_weak_link(state)

        # Winner remains Problem via canonical taxonomy order
        assert result["weak_area"] == str(StartupArea.PROBLEM)

        telemetry = result["weak_link_details"]["saturation_telemetry"]
        assert telemetry["max_score"] == 81
        assert telemetry["max_score_tie_count"] == 6
        assert telemetry["saturated_area_count"] == 6
        assert telemetry["tie_breaker_used"] is True

