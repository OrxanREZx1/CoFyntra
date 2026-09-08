"""
Milestone 4.0.7.1.2 Test Suite: Critical Assumption Recovery & Error Reporting Integrity.

Tests:
1. first invalid + second valid -> succeeds
2. retry receives validator rejection reason
3. unsupported concept from attempt 1 explicitly blocked on retry
4. two invalid attempts -> deterministic fallback attempted
5. valid fallback -> workflow continues
6. invalid fallback -> clean reasoning failure
7. internal validation failure does NOT print API-key warning
8. missing API key still prints configuration guidance
9. provider/authentication errors distinguishable from reasoning errors
"""

import logging
from unittest.mock import MagicMock

import pytest

from startary.cli import ErrorCategory, classify_cli_error
from startary.domain.assumption import CriticalAssumptionGroundingError
from startary.domain.taxonomy import StartupArea
from startary.graph.graph_state import StartaryGraphState
from startary.graph.nodes import (
    build_grounded_critical_assumption_fallback,
    extract_blocked_grounding_concepts,
    generate_critical_assumption,
    validate_critical_assumption_grounding,
)
from startary.llm.schemas import CriticalAssumptionLLMOutput


@pytest.fixture
def base_fridgeguard_state() -> StartaryGraphState:
    return {
        "project_id": "test-fg-recovery",
        "weak_area": "Problem",
        "raw_context": "We run commercial refrigeration sensors for restaurants. Chefs log temps manually twice a day. Food spoils if fridges fail between checks.",
        "founder_messages": [
            {
                "role": "founder",
                "content": "We run commercial refrigeration sensors for restaurants. Chefs log temps manually twice a day. Food spoils if fridges fail between checks.",
            }
        ],
        "reality_check": {
            "summary": "Core refrigeration failure risk is plausible but unmeasured between checks.",
            "conclusion": "INCONCLUSIVE",
            "unresolved_uncertainty": "Whether refrigeration failures between staff checks cause enough disruption or food loss to warrant continuous monitoring.",
            "what_evidence_supports": ["Manual temperature checks are performed only twice daily."],
            "what_evidence_does_not_support": ["delayed deliveries"],
            "relevant_founder_claims": ["Food spoils if fridge fails between checks."],
            "available_founder_evidence": ["Chefs do manual logs."],
        },
        "startup_state": {"areas": {}},
        "traces": [],
    }


class TestCriticalAssumptionRecovery:
    def test_attempt_1_invalid_attempt_2_valid_succeeds(self, mocker, base_fridgeguard_state):
        """Scenario 1: First attempt fails grounding (unsupported concept), second attempt succeeds."""
        invalid_output = CriticalAssumptionLLMOutput(
            area=StartupArea.PROBLEM,
            statement="Commercial kitchens experience delayed deliveries causing food spoilage.",
            rationale="Delayed deliveries are critical",
            what_falsifies_it="No delayed deliveries.",
            introduced_test_parameters=[],
        )
        valid_output = CriticalAssumptionLLMOutput(
            area=StartupArea.PROBLEM,
            statement="Refrigeration failures that occur between manual staff checks cause avoidable food spoilage.",
            rationale="Directly tests core unmonitored window.",
            what_falsifies_it="No failures occur between manual checks.",
            introduced_test_parameters=[],
        )

        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.side_effect = [invalid_output, valid_output]
        mock_llm.with_structured_output.return_value = mock_structured

        mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
        mocker.patch("startary.graph.nodes.get_settings", return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"))

        result = generate_critical_assumption(base_fridgeguard_state)
        assert result["critical_assumption_output"]["statement"] == valid_output.statement
        assert mock_structured.invoke.call_count == 2

    def test_retry_receives_validator_rejection_reason(self, mocker, base_fridgeguard_state):
        """Scenario 2: Retry prompt includes the exact rejection reason from the grounding validator."""
        invalid_output = CriticalAssumptionLLMOutput(
            area=StartupArea.PROBLEM,
            statement="Commercial kitchens suffer delayed deliveries between checks.",
            rationale="Delayed deliveries",
            what_falsifies_it="No delays",
            introduced_test_parameters=[],
        )
        valid_output = CriticalAssumptionLLMOutput(
            area=StartupArea.PROBLEM,
            statement="Commercial kitchens suffer undetected temperature spikes between manual checks.",
            rationale="Valid hypothesis",
            what_falsifies_it="No spikes",
            introduced_test_parameters=[],
        )

        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.side_effect = [invalid_output, valid_output]
        mock_llm.with_structured_output.return_value = mock_structured

        mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
        mocker.patch("startary.graph.nodes.get_settings", return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"))

        generate_critical_assumption(base_fridgeguard_state)

        # Inspect messages passed on attempt 2
        attempt_2_messages = mock_structured.invoke.call_args_list[1][0][0]
        retry_prompt = attempt_2_messages[-1]["content"]

        assert "GROUNDING VALIDATION FAILED:" in retry_prompt
        assert "delayed deliveries" in retry_prompt
        assert "Reality Check explicitly confirmed is unsupported by evidence" in retry_prompt

    def test_unsupported_concept_blocked_on_retry(self, mocker, caplog, base_fridgeguard_state):
        """Scenario 3: Unsupported concept from attempt 1 is explicitly blocked and logged."""
        invalid_output = CriticalAssumptionLLMOutput(
            area=StartupArea.PROBLEM,
            statement="Commercial kitchens suffer delayed deliveries weekly.",
            rationale="Delayed deliveries weekly",
            what_falsifies_it="No delays",
            introduced_test_parameters=[],
        )
        valid_output = CriticalAssumptionLLMOutput(
            area=StartupArea.PROBLEM,
            statement="Commercial kitchens experience refrigeration failures between manual checks.",
            rationale="Valid hypothesis",
            what_falsifies_it="No failures",
            introduced_test_parameters=[],
        )

        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.side_effect = [invalid_output, valid_output]
        mock_llm.with_structured_output.return_value = mock_structured

        mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
        mocker.patch("startary.graph.nodes.get_settings", return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"))

        with caplog.at_level(logging.WARNING):
            generate_critical_assumption(base_fridgeguard_state)

        # Verify structured retry log
        assert "critical_assumption_retry | attempt=2" in caplog.text
        assert "blocked_concepts" in caplog.text
        assert "delayed deliveries" in caplog.text

        # Verify retry message contains blocked concepts list
        attempt_2_messages = mock_structured.invoke.call_args_list[1][0][0]
        retry_prompt = attempt_2_messages[-1]["content"]
        assert "BLOCKED CONCEPTS / PHRASES:" in retry_prompt
        assert "'delayed deliveries'" in retry_prompt

    def test_two_invalid_attempts_triggers_fallback(self, mocker, base_fridgeguard_state):
        """Scenario 4: When two attempts fail grounding validation, deterministic fallback is attempted."""
        invalid_output = CriticalAssumptionLLMOutput(
            area=StartupArea.PROBLEM,
            statement="Kitchens lose 30% of their inventory weekly due to delayed deliveries.",
            rationale="Ungrounded",
            what_falsifies_it="No loss",
            introduced_test_parameters=[],
        )

        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.side_effect = [invalid_output, invalid_output]
        mock_llm.with_structured_output.return_value = mock_structured

        mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
        mocker.patch("startary.graph.nodes.get_settings", return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"))

        import startary.graph.nodes as nodes

        spy_fallback = mocker.spy(
            nodes, "build_grounded_critical_assumption_fallback"
        )

        result = generate_critical_assumption(base_fridgeguard_state)

        assert spy_fallback.call_count == 1
        assert result["critical_assumption_output"]["statement"] is not None

    def test_valid_fallback_allows_workflow_to_continue(self, mocker, caplog, base_fridgeguard_state):
        """Scenario 5: Valid fallback succeeds and workflow continues without crashing."""
        invalid_output = CriticalAssumptionLLMOutput(
            area=StartupArea.PROBLEM,
            statement="Commercial kitchens lose 50% of stock weekly.",
            rationale="Invalid",
            what_falsifies_it="No loss",
            introduced_test_parameters=[],
        )

        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.side_effect = [invalid_output, invalid_output]
        mock_llm.with_structured_output.return_value = mock_structured

        mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
        mocker.patch("startary.graph.nodes.get_settings", return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"))

        with caplog.at_level(logging.INFO):
            result = generate_critical_assumption(base_fridgeguard_state)

        # Confirm fallback log was emitted
        assert "critical_assumption_fallback | used=True" in caplog.text
        ca_out = result["critical_assumption_output"]
        assert ca_out["area"] == StartupArea.PROBLEM
        assert len(ca_out["statement"]) >= 10
        # Confirm trace records fallback_used
        assert result["last_trace"]["input_data"]["fallback_used"] is True

    def test_invalid_fallback_raises_clean_reasoning_error(self, mocker, caplog, base_fridgeguard_state):
        """Scenario 6: When both attempts fail and fallback fails/returns None, CriticalAssumptionGroundingError is raised."""
        invalid_output = CriticalAssumptionLLMOutput(
            area=StartupArea.PROBLEM,
            statement="Kitchens lose 30% weekly.",
            rationale="Invalid",
            what_falsifies_it="None",
            introduced_test_parameters=[],
        )

        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.side_effect = [invalid_output, invalid_output]
        mock_llm.with_structured_output.return_value = mock_structured

        mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
        mocker.patch("startary.graph.nodes.get_settings", return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"))
        mocker.patch("startary.graph.nodes.build_grounded_critical_assumption_fallback", return_value=None)

        with caplog.at_level(logging.ERROR):
            with pytest.raises(CriticalAssumptionGroundingError) as excinfo:
                generate_critical_assumption(base_fridgeguard_state)

        assert "failed grounding validation after retry" in str(excinfo.value)
        assert "critical_assumption_recovery_failed" in caplog.text

    def test_internal_validation_failure_does_not_print_api_key_warning(self):
        """Scenario 7: CriticalAssumptionGroundingError classifies as REASONING_ERROR and does NOT print API key warning."""
        exc = CriticalAssumptionGroundingError(
            "generate_critical_assumption failed grounding validation after retry: Assumption asserts 'delayed deliveries'"
        )
        category, title, guidance = classify_cli_error(exc)

        assert category == ErrorCategory.REASONING_ERROR
        assert "Reasoning" in title
        assert "Ensure OPENAI_API_KEY is configured in .env." not in guidance
        assert "not an OpenAI API key or configuration issue" in guidance

    def test_missing_api_key_prints_configuration_guidance(self):
        """Scenario 8: Missing/placeholder API key error classifies as CONFIGURATION_ERROR and gives API key guidance."""
        exc = ValueError("OpenAI API key is not configured. Open the project .env file and replace PASTE_YOUR_OPENAI_API_KEY_HERE")
        category, title, guidance = classify_cli_error(exc)

        assert category == ErrorCategory.CONFIGURATION_ERROR
        assert "Configuration" in title
        assert "Ensure OPENAI_API_KEY is configured in .env." in guidance

    def test_provider_errors_distinguishable_from_reasoning_errors(self):
        """Scenario 9: Provider errors (timeout, network, auth) classify as PROVIDER_ERROR, distinguishable from REASONING_ERROR."""
        net_exc = ConnectionError("Connection timed out to api.openai.com:443")
        net_cat, _, net_guidance = classify_cli_error(net_exc)

        auth_exc = RuntimeError("Error code: 401 - {'error': {'message': 'Incorrect API key provided'}}")
        auth_cat, _, auth_guidance = classify_cli_error(auth_exc)

        assert net_cat == ErrorCategory.PROVIDER_ERROR
        assert auth_cat == ErrorCategory.PROVIDER_ERROR
        assert net_cat != ErrorCategory.REASONING_ERROR
        assert auth_cat != ErrorCategory.REASONING_ERROR
        assert "network" in net_guidance.lower() or "service" in net_guidance.lower()
        assert "authentication" in auth_guidance.lower() or "active and valid" in auth_guidance.lower()
