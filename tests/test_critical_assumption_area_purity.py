"""
Milestone 4.0.7.1.3 Test Suite: Critical Assumption Area Purity.

Tests:
1. Target User assumption containing "willing to pay" is rejected.
2. Target User assumption requiring adoption is rejected.
3. Target User assumption about repeatable segment + identifiable decision-maker passes.
4. Existing Alternatives assumption can discuss alternative adequacy without implying our solution is validated.
5. Business Model assumption may legitimately include payment/willingness-to-pay.
6. Value Proposition assumption may discuss usefulness/preference without automatically requiring payment.
7. Area-purity failure feeds exact reason into existing retry.
8. Deterministic fallback must pass area-purity validation.
9. Existing valid assumptions from previous milestones continue to pass.
10. End-to-end FridgeGuard Target User flow: clean non-contaminated assumption.
"""

import logging
from unittest.mock import MagicMock

import pytest

from startary.domain.taxonomy import StartupArea
from startary.graph.graph_state import StartaryGraphState
from startary.graph.nodes import (
    build_grounded_critical_assumption_fallback,
    generate_critical_assumption,
    validate_critical_assumption_area_purity,
    validate_critical_assumption_grounding,
)
from startary.llm.schemas import CriticalAssumptionLLMOutput


class TestCriticalAssumptionAreaPurityValidator:
    def test_1_target_user_with_willing_to_pay_rejected(self):
        """1. Target User assumption containing 'willing to pay' is rejected."""
        ca = CriticalAssumptionLLMOutput(
            area=StartupArea.TARGET_USER,
            statement="Independent restaurant owners are willing to pay $100 per month for temperature monitoring.",
            rationale="Test payment requirement",
            what_falsifies_it="Owners refuse to pay.",
            introduced_test_parameters=[],
        )
        is_valid, reason = validate_critical_assumption_area_purity(ca, StartupArea.TARGET_USER)
        assert is_valid is False
        assert "willing to pay" in reason
        assert "Business Model / Revenue" in reason

    def test_2_target_user_requiring_adoption_rejected(self):
        """2. Target User assumption requiring adoption is rejected."""
        # Exact problematic wording from live FridgeGuard run
        ca = CriticalAssumptionLLMOutput(
            area=StartupArea.TARGET_USER,
            statement="There is a definable restaurant user and decision-maker group beyond the founder's uncle for whom refrigerator-temperature monitoring alerts are relevant enough to consider adopting the system.",
            rationale="Test adoption requirement",
            what_falsifies_it="Users do not consider adopting.",
            introduced_test_parameters=[],
        )
        is_valid, reason = validate_critical_assumption_area_purity(ca, StartupArea.TARGET_USER)
        assert is_valid is False
        assert "solution adoption" in reason
        assert "consider adopting" in reason

    def test_3_target_user_repeatable_segment_and_role_passes(self):
        """3. Target User assumption about repeatable segment + identifiable decision-maker passes."""
        ca = CriticalAssumptionLLMOutput(
            area=StartupArea.TARGET_USER,
            statement="There is a repeatable restaurant segment beyond the founder's uncle in which a clearly identifiable role owns the refrigeration-monitoring problem and/or the decision to address it.",
            rationale="Tests core customer role and segment boundary",
            what_falsifies_it="No repeatable segment or role ownership found across restaurants.",
            introduced_test_parameters=[],
        )
        is_valid, reason = validate_critical_assumption_area_purity(ca, StartupArea.TARGET_USER)
        assert is_valid is True
        assert reason == ""

    def test_4_existing_alternatives_discusses_adequacy_without_solution_validation(self):
        """4. Existing Alternatives assumption can discuss alternative adequacy without implying our solution is validated."""
        # Valid: examines status quo limitations
        ca_valid = CriticalAssumptionLLMOutput(
            area=StartupArea.EXISTING_ALTERNATIVES,
            statement="Manual twice-daily temperature logs leave unmonitored overnight gaps that permit refrigeration failures to go undetected until food is spoiled.",
            rationale="Tests inadequacy of status quo",
            what_falsifies_it="Manual logs reliably catch all critical refrigeration temperature anomalies.",
            introduced_test_parameters=[],
        )
        is_valid, reason = validate_critical_assumption_area_purity(ca_valid, StartupArea.EXISTING_ALTERNATIVES)
        assert is_valid is True
        assert reason == ""

        # Invalid: asserts that inadequacy proves our solution
        ca_invalid = CriticalAssumptionLLMOutput(
            area=StartupArea.EXISTING_ALTERNATIVES,
            statement="Manual temperature logs are inadequate, which proves our solution will be adopted by restaurant operators.",
            rationale="Contaminated with solution validation",
            what_falsifies_it="Users keep manual logs.",
            introduced_test_parameters=[],
        )
        is_valid_inv, reason_inv = validate_critical_assumption_area_purity(ca_invalid, StartupArea.EXISTING_ALTERNATIVES)
        assert is_valid_inv is False
        assert "improperly implies solution validation" in reason_inv

    def test_5_business_model_legitimately_includes_payment(self):
        """5. Business Model assumption may legitimately include payment/willingness-to-pay."""
        ca = CriticalAssumptionLLMOutput(
            area=StartupArea.BUSINESS_MODEL,
            statement="Commercial kitchen operators derive sufficient financial value from avoiding inventory loss that they are willing to pay a monthly subscription fee for automated monitoring.",
            rationale="Tests economic viability and payment willingness",
            what_falsifies_it="Operators refuse to allocate budget for monitoring software.",
            introduced_test_parameters=[],
        )
        is_valid, reason = validate_critical_assumption_area_purity(ca, StartupArea.BUSINESS_MODEL)
        assert is_valid is True
        assert reason == ""

    def test_6_value_proposition_discusses_usefulness_without_requiring_payment(self):
        """6. Value Proposition assumption may discuss usefulness/preference without automatically requiring payment."""
        ca = CriticalAssumptionLLMOutput(
            area=StartupArea.VALUE_PROPOSITION,
            statement="Restaurant operators perceive real-time automated refrigeration alerts as significantly more reliable and actionable than manual twice-daily paper logs.",
            rationale="Tests perceived utility and preference of outcome over status quo",
            what_falsifies_it="Operators view automated alerts as no more useful than paper logs.",
            introduced_test_parameters=[],
        )
        is_valid, reason = validate_critical_assumption_area_purity(ca, StartupArea.VALUE_PROPOSITION)
        assert is_valid is True
        assert reason == ""

    def test_7_area_purity_failure_feeds_exact_reason_into_retry(self, mocker):
        """7. Area-purity failure on attempt 1 feeds exact rejection reason and blocked concepts into attempt 2."""
        invalid_output = CriticalAssumptionLLMOutput(
            area=StartupArea.TARGET_USER,
            statement="There is a definable restaurant group for whom alerts are relevant enough to consider adopting the system.",
            rationale="Contaminated",
            what_falsifies_it="No adoption",
            introduced_test_parameters=[],
        )
        valid_output = CriticalAssumptionLLMOutput(
            area=StartupArea.TARGET_USER,
            statement="There is a repeatable restaurant segment beyond the founder's uncle with an identifiable decision-maker for refrigeration equipment.",
            rationale="Pure Target User",
            what_falsifies_it="No repeatable segment",
            introduced_test_parameters=[],
        )

        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.side_effect = [invalid_output, valid_output]
        mock_llm.with_structured_output.return_value = mock_structured

        mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
        mocker.patch("startary.graph.nodes.get_settings", return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"))

        state: StartaryGraphState = {
            "project_id": "test-area-purity-retry",
            "weak_area": "Target User",
            "raw_context": "Founder's uncle runs a restaurant. Founder wants to know if other restaurants have this problem.",
            "founder_messages": [
                {"role": "founder", "content": "Founder's uncle runs a restaurant. Founder wants to know if other restaurants have this problem."}
            ],
            "reality_check": {
                "summary": "Customer access is currently restricted to uncle's restaurant.",
                "conclusion": "INCONCLUSIVE",
                "unresolved_uncertainty": "Whether restaurant operators outside warm contacts have a repeatable decision-maker role.",
                "what_evidence_supports": ["Uncle confirmed the issue."],
                "what_evidence_does_not_support": [],
            },
            "startup_state": {"areas": {}},
            "traces": [],
        }

        result = generate_critical_assumption(state)

        # Confirm attempt 2 was invoked
        assert mock_structured.invoke.call_count == 2
        attempt_2_messages = mock_structured.invoke.call_args_list[1][0][0]
        retry_prompt = attempt_2_messages[-1]["content"]

        # Check that rejection reason and scope guidance were passed
        assert "improperly requires solution adoption" in retry_prompt
        assert "Target User" in retry_prompt
        assert "Do NOT import validation requirements from neighboring areas" in retry_prompt
        assert result["critical_assumption_output"]["statement"] == valid_output.statement

    def test_8_deterministic_fallback_must_pass_area_purity(self):
        """8. Deterministic fallback builder produces assumptions that pass area-purity validation for all 10 areas."""
        state: StartaryGraphState = {
            "project_id": "test-fallback-purity",
            "weak_area": "Target User",
            "raw_context": "Commercial refrigeration for kitchens.",
            "reality_check": {
                "summary": "Summary",
                "conclusion": "INCONCLUSIVE",
                "unresolved_uncertainty": "Whether a repeatable restaurant customer segment exists outside personal network.",
                "what_evidence_supports": [],
                "what_evidence_does_not_support": [],
            },
            "startup_state": {"areas": {}},
        }

        # Test for all 10 areas
        for area in StartupArea:
            fallback = build_grounded_critical_assumption_fallback(
                area.value,
                state["reality_check"],
                state["raw_context"],
                state,
            )
            assert fallback is not None, f"Fallback failed to build for {area.value}"
            is_valid_g, g_reason = validate_critical_assumption_grounding(
                fallback, state["reality_check"], state["raw_context"], area.value
            )
            assert is_valid_g is True, f"Fallback for {area.value} failed grounding: {g_reason}"
            is_valid_p, p_reason = validate_critical_assumption_area_purity(
                fallback, area.value
            )
            assert is_valid_p is True, f"Fallback for {area.value} failed area purity: {p_reason}"

    def test_9_existing_valid_assumptions_continue_to_pass(self):
        """9. Existing valid assumptions from previous milestones (e.g. Problem, Acquisition) continue to pass."""
        # Problem area assumption
        problem_ca = CriticalAssumptionLLMOutput(
            area=StartupArea.PROBLEM,
            statement="Commercial kitchens experience refrigeration failures occurring between manual inspection logs that cause significant avoidable food loss.",
            rationale="Tests core problem occurrence",
            what_falsifies_it="No failures occur between logs.",
            introduced_test_parameters=[],
        )
        is_valid_p, reason_p = validate_critical_assumption_area_purity(problem_ca, StartupArea.PROBLEM)
        assert is_valid_p is True
        assert reason_p == ""

        # Acquisition area assumption
        acq_ca = CriticalAssumptionLLMOutput(
            area=StartupArea.ACQUISITION,
            statement="Early-stage founders outside the founder's university network can be reached and engaged through direct outbound communication.",
            rationale="Tests outbound channel reach",
            what_falsifies_it="Zero response from cold outreach.",
            introduced_test_parameters=[],
        )
        is_valid_a, reason_a = validate_critical_assumption_area_purity(acq_ca, StartupArea.ACQUISITION)
        assert is_valid_a is True
        assert reason_a == ""

    def test_10_fridgeguard_target_user_generates_pure_assumption(self, mocker):
        """10. FridgeGuard with Target User weak link generates a clean, non-contaminated assumption."""
        fridgeguard_target_user_assumption = CriticalAssumptionLLMOutput(
            area=StartupArea.TARGET_USER,
            statement="There is a repeatable restaurant segment beyond the founder's uncle in which a clearly identifiable role owns the refrigeration-monitoring problem and/or the decision to address it.",
            rationale="Directly tests segment and role boundary without requiring adoption",
            what_falsifies_it="Restaurant roles are heterogeneous and no repeatable problem owner exists.",
            introduced_test_parameters=[],
        )

        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = fridgeguard_target_user_assumption
        mock_llm.with_structured_output.return_value = mock_structured

        mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
        mocker.patch("startary.graph.nodes.get_settings", return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"))

        state: StartaryGraphState = {
            "project_id": "test-fg-target-user",
            "weak_area": "Target User",
            "raw_context": "Founder built IoT sensors for his uncle's restaurant kitchen to monitor commercial fridges. He wants to know if other restaurant owners care.",
            "founder_messages": [
                {
                    "role": "founder",
                    "content": "Founder built IoT sensors for his uncle's restaurant kitchen to monitor commercial fridges. He wants to know if other restaurant owners care.",
                }
            ],
            "reality_check": {
                "summary": "Customer access is currently restricted to one warm contact (uncle's restaurant).",
                "conclusion": "INCONCLUSIVE",
                "unresolved_uncertainty": "Whether an identifiable operational role in other restaurants owns the refrigeration problem.",
                "what_evidence_supports": ["Uncle confirmed he cares about temp monitoring."],
                "what_evidence_does_not_support": [],
                "relevant_founder_claims": ["Other restaurant owners might have this problem."],
                "available_founder_evidence": ["One restaurant tested."],
            },
            "startup_state": {"areas": {}},
            "traces": [],
        }

        result = generate_critical_assumption(state)
        ca_stmt = result["critical_assumption_output"]["statement"]

        # Verify assumption is pure Target User
        assert "adopting" not in ca_stmt.lower()
        assert "adopt" not in ca_stmt.lower()
        assert "pay" not in ca_stmt.lower()
        assert "prefer" not in ca_stmt.lower()
        assert "repeatable" in ca_stmt.lower() or "role" in ca_stmt.lower()

