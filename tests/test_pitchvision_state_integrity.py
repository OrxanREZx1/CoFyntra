"""
Regression tests for PitchVision AI state integrity and latency-safe fixes.

Validates the 5 confirmed defects discovered during PitchVision testing:
1. extract_startup_state token exhaustion recovery via structured retry (1,000 -> 2,500 tokens).
2. assess_baseline_state_sufficiency invariant guards against failed extraction / unpersisted state.
3. resolve_customer_access and assess_can_execute preserving explicit access (two amateur teams, three coaches).
4. decide_next_strategic_action stage purity: idea-stage with 0 users / 0 revenue NEVER receives RETENTION_TEST.
5. interpret_founder_turn intent classification: initial intake facts are ANSWER, NOT NEW_EVIDENCE.
"""

from unittest.mock import MagicMock, patch
import pytest

from startary.config import Settings
from startary.domain.customer_access import CustomerAccessStatus, CustomerAccessStrength
from startary.domain.enums import ExecutionReadiness, ExperimentType, WorkflowPhase
from startary.domain.strategic_action import NextStrategicActionType
from startary.graph.graph_state import StartaryGraphState
from startary.graph.nodes import (
    assess_baseline_state_sufficiency,
    assess_can_execute,
    decide_next_strategic_action,
    extract_startup_state,
    interpret_founder_turn,
    resolve_customer_access,
)
from startary.llm.provider import TokenBudgetExhaustedError
from startary.llm.schemas import (
    ExtractedKnowledgeFieldLLMOutput,
    FounderTurnInterpretationLLMOutput,
    StartupStateDeltaLLMOutput,
)


class MockLengthFinishReasonError(Exception):
    """Simulates an OpenAI / LangChain LengthFinishReasonError on token limit."""
    pass


@pytest.fixture
def pitchvision_raw_message() -> str:
    return (
        "We are PitchVision AI. We use computer vision to analyze amateur football matches and provide tactical insights. "
        "We are two co-founders, spending about 15 hours each per week. We have a budget of $1,500. "
        "We have direct access to two amateur football teams and three coaches who agreed to test our prototype once ready. "
        "We have not built the MVP yet, we have zero paying customers, zero revenue, and haven't run structured customer interviews yet."
    )


@pytest.fixture
def pitchvision_base_state(pitchvision_raw_message) -> StartaryGraphState:
    return {
        "project_id": "proj-pitchvision-001",
        "phase": WorkflowPhase.UNDERSTAND,
        "raw_context": pitchvision_raw_message,
        "founder_messages": [
            {"role": "founder", "content": pitchvision_raw_message}
        ],
        "pending_question": None,
        "clarification_turn_count": 0,
        "execution_context": {
            "customer_access": {
                "value": "Two amateur football teams and three coaches",
                "provenance": "EXPLICIT",
            },
            "available_budget_amount": {"value": 1500, "provenance": "EXPLICIT"},
            "weekly_capacity": {"value": "30 hrs/week (15 hrs/founder)", "provenance": "EXPLICIT"},
            "founder_count": {"value": 2, "provenance": "EXPLICIT"},
            "active_team_size": {"value": 2, "provenance": "EXPLICIT"},
        },
        "startup_progress": {
            "mvp_exists": {"value": False, "provenance": "EXPLICIT"},
            "prototype_exists": {"value": False, "provenance": "EXPLICIT"},
            "launched": {"value": False, "provenance": "EXPLICIT"},
            "development_stage": {"value": "idea_validation", "provenance": "EXPLICIT"},
            "paying_customers_count": 0,
            "active_users_count": 0,
            "revenue": {"value": "$0", "provenance": "EXPLICIT"},
        },
        "weak_area": "Problem",
        "critical_assumption": {
            "statement": "Amateur football coaches feel enough pain with post-match tactical analysis that they will adopt computer-vision insights.",
            "area": "Problem",
        },
        "active_experiment": {
            "experiment_type": ExperimentType.PROBLEM_INTERVIEW.value,
            "objective": "Validate whether coaches actively spend 3+ hours reviewing match video",
            "required_access": ["Amateur football coaches"],
            "estimated_cost": "$0",
            "estimated_time": "3-5 days",
        },
    }


# ---------------------------------------------------------------------------
# Defect 1: extract_startup_state token exhaustion recovery
# ---------------------------------------------------------------------------

class TestExtractStartupStateTokenRecovery:
    @patch("startary.graph.nodes._safe_get_llm_client")
    def test_extract_startup_state_succeeds_on_first_attempt_without_retry(
        self, mock_get_client, pitchvision_base_state, pitchvision_raw_message
    ):
        """
        Milestone 6.2.1: Verify that extract_startup_state configures Attempt 1 with 2500 tokens
        and reasoning_effort='low', succeeding directly in a single attempt without triggering retry.
        """
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        expected_output = StartupStateDeltaLLMOutput(
            founder_count=ExtractedKnowledgeFieldLLMOutput(value="2", provenance="EXPLICIT"),
            active_team_size=ExtractedKnowledgeFieldLLMOutput(value="2", provenance="EXPLICIT"),
            available_budget_amount=ExtractedKnowledgeFieldLLMOutput(value="1500", provenance="EXPLICIT"),
            customer_access=ExtractedKnowledgeFieldLLMOutput(
                value="Two amateur football teams and three coaches", provenance="EXPLICIT"
            ),
            mvp_exists=ExtractedKnowledgeFieldLLMOutput(value="false", provenance="EXPLICIT"),
            paying_customers=ExtractedKnowledgeFieldLLMOutput(value="0", provenance="EXPLICIT"),
            revenue=ExtractedKnowledgeFieldLLMOutput(value="0", provenance="EXPLICIT"),
        )
        mock_structured.invoke.return_value = expected_output
        mock_llm.with_structured_output.return_value = mock_structured
        mock_get_client.return_value = mock_llm

        result = extract_startup_state(pitchvision_base_state)

        # 1. Delta was successfully extracted on attempt 1
        assert "startup_state_delta" in result
        delta = result["startup_state_delta"]
        assert delta["founder_count"]["value"] == "2"
        assert delta["customer_access"]["value"] == "Two amateur football teams and three coaches"
        assert delta["available_budget_amount"]["value"] == "1500"
        assert result.get("extraction_status") == "SUCCESS"

        # 2. Verify ONLY 1 attempt was made (no retry needed)
        assert mock_get_client.call_count == 1
        attempt_1_kwargs = mock_get_client.call_args[1]
        assert attempt_1_kwargs["max_completion_tokens"] == 2500
        assert attempt_1_kwargs["reasoning_effort"] == "low"
        assert attempt_1_kwargs["tier"] == "fast"

    @patch("startary.graph.nodes._safe_get_llm_client")
    def test_extract_startup_state_recovers_from_token_exhaustion_with_expanded_budget(
        self, mock_get_client, pitchvision_base_state, pitchvision_raw_message
    ):
        """
        Verify that extract_startup_state catches token budget exhaustion if base call (2500 tokens)
        hits length limit, retries with expanded budget (4000 tokens) and reasoning_effort='low',
        and successfully returns delta.
        """
        # First call fails with token exhaustion
        mock_llm_base = MagicMock()
        mock_structured_base = MagicMock()
        mock_structured_base.invoke.side_effect = MockLengthFinishReasonError("finish_reason='length' - token limit exceeded")
        mock_llm_base.with_structured_output.return_value = mock_structured_base

        # Second call (retry) succeeds with valid extracted output
        mock_llm_retry = MagicMock()
        mock_structured_retry = MagicMock()
        expected_output = StartupStateDeltaLLMOutput(
            founder_count=ExtractedKnowledgeFieldLLMOutput(value="2", provenance="EXPLICIT"),
            active_team_size=ExtractedKnowledgeFieldLLMOutput(value="2", provenance="EXPLICIT"),
            available_budget_amount=ExtractedKnowledgeFieldLLMOutput(value="1500", provenance="EXPLICIT"),
            customer_access=ExtractedKnowledgeFieldLLMOutput(
                value="Two amateur football teams and three coaches", provenance="EXPLICIT"
            ),
        )
        mock_structured_retry.invoke.return_value = expected_output
        mock_llm_retry.with_structured_output.return_value = mock_structured_retry

        # Return base client on first call, retry client on second call
        mock_get_client.side_effect = [mock_llm_base, mock_llm_retry]

        result = extract_startup_state(pitchvision_base_state)

        # Invariant checks:
        # 1. Delta was successfully extracted
        assert "startup_state_delta" in result
        delta = result["startup_state_delta"]
        assert delta["founder_count"]["value"] == "2"
        assert delta["customer_access"]["value"] == "Two amateur football teams and three coaches"
        assert result.get("extraction_status") == "SUCCESS"

        # 2. Verify retry used expanded budget 4000 tokens and reasoning_effort='low'
        assert mock_get_client.call_count == 2
        attempt_1_kwargs = mock_get_client.call_args_list[0][1]
        assert attempt_1_kwargs["max_completion_tokens"] == 2500
        assert attempt_1_kwargs["reasoning_effort"] == "low"

        retry_kwargs = mock_get_client.call_args_list[1][1]
        assert retry_kwargs["max_completion_tokens"] == 4000
        assert retry_kwargs["reasoning_effort"] == "low"

    @patch("startary.graph.nodes._safe_get_llm_client")
    def test_extract_startup_state_non_token_error_fails_safely_without_blind_retry(
        self, mock_get_client, pitchvision_base_state
    ):
        """
        Verify that non-token errors (e.g. ValueError or syntax) fail safely without triggering
        a token budget expansion retry, preserving latency.
        """
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.side_effect = ValueError("Invalid schema payload")
        mock_llm.with_structured_output.return_value = mock_structured
        mock_get_client.return_value = mock_llm

        result = extract_startup_state(pitchvision_base_state)

        assert result.get("extraction_status") == "FAILED"
        assert result.get("startup_state_delta") == {}
        # Only 1 attempt made - no blind token retry
        assert mock_get_client.call_count == 1


# ---------------------------------------------------------------------------
# Defect 2: assess_baseline_state_sufficiency invariant guards
# ---------------------------------------------------------------------------

class TestBaselineSufficiencyInvariants:
    def test_baseline_sufficiency_fails_immediately_when_extraction_failed(
        self, pitchvision_base_state
    ):
        """
        If extraction_status == 'FAILED', baseline sufficiency MUST return False,
        preventing unpersisted raw founder text from declaring the baseline complete.
        """
        pitchvision_base_state["extraction_status"] = "FAILED"
        pitchvision_base_state["extraction_error"] = "LengthFinishReasonError: token exhausted"

        res = assess_baseline_state_sufficiency(pitchvision_base_state)

        assert res["is_baseline_sufficient"] is False
        assert res["baseline_sufficiency"]["is_sufficient"] is False
        assert "extraction_status=FAILED" in res["baseline_sufficiency"]["reason"]

    @patch("startary.graph.nodes._safe_get_llm_client")
    def test_baseline_sufficiency_fails_when_operational_facts_not_persisted(
        self, mock_get_client, pitchvision_base_state
    ):
        """
        Even if the LLM says 'sufficient' based on raw founder text, if fewer than 2
        core operational dimensions are actually persisted in state, sufficiency MUST be False.
        """
        # Extraction failed: facts never persisted into state
        pitchvision_base_state["execution_context"] = {}
        pitchvision_base_state["startup_progress"] = {}
        pitchvision_base_state["extraction_status"] = "FAILED"
        pitchvision_base_state["extraction_error"] = "LengthFinishReasonError: token budget exhausted"

        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_res = MagicMock()
        mock_res.is_sufficient = True
        mock_res.team_status = "KNOWN"
        mock_res.capabilities_status = "KNOWN"
        mock_res.stage_status = "KNOWN"
        mock_res.resources_status = "KNOWN"
        mock_res.access_status = "KNOWN"
        mock_res.reason = "Founder mentioned team and access in raw text."
        mock_structured.invoke.return_value = mock_res
        mock_llm.with_structured_output.return_value = mock_structured
        mock_get_client.return_value = mock_llm

        res = assess_baseline_state_sufficiency(pitchvision_base_state)

        # Invariant enforced: facts are not in state, so cannot be sufficient
        assert res["is_baseline_sufficient"] is False
        assert res["baseline_sufficiency"]["is_sufficient"] is False


# ---------------------------------------------------------------------------
# Defect 3: resolve_customer_access & assess_can_execute preservation
# ---------------------------------------------------------------------------

class TestCustomerAccessPreservation:
    def test_resolve_customer_access_identifies_teams_and_coaches_as_established(
        self, pitchvision_base_state
    ):
        """
        Verify that explicit access to 'Two amateur football teams and three coaches'
        is correctly resolved as established access (not NONE).
        """
        access = resolve_customer_access(pitchvision_base_state)

        assert access.access_status != CustomerAccessStatus.NONE
        assert access.access_status == CustomerAccessStatus.ESTABLISHED

    def test_assess_can_execute_pitchvision_problem_interview_passes(
        self, pitchvision_base_state
    ):
        """
        For PitchVision (problem interview, $1,500 budget, 30 hrs/wk, 2 teams + 3 coaches),
        assess_can_execute MUST return CAN_EXECUTE without blocking on customer access.
        """
        res = assess_can_execute(pitchvision_base_state)
        assessment = res["can_execute_result"]

        assert assessment["readiness"] == ExecutionReadiness.CAN_EXECUTE
        assert len(assessment["missing_requirements"]) == 0
        assert len(assessment["blockers"]) == 0

    def test_resolve_customer_access_dialogue_fallback_detects_agreed_indicators(
        self, pitchvision_base_state
    ):
        """
        Even if execution_context is missing the field, dialogue mentioning
        'access to two amateur teams and three coaches' or 'teams agreed' must resolve established access.
        """
        pitchvision_base_state["execution_context"]["customer_access"] = None
        pitchvision_base_state["founder_messages"] = [
            {"role": "founder", "content": "We have direct access to two amateur football teams and coaches agreed to test."}
        ]

        access = resolve_customer_access(pitchvision_base_state)
        assert access.access_status != CustomerAccessStatus.NONE
        assert access.access_status == CustomerAccessStatus.ESTABLISHED


# ---------------------------------------------------------------------------
# Defect 4: decide_next_strategic_action stage & metric purity
# ---------------------------------------------------------------------------

class TestDecideNextStrategicActionStagePurity:
    def test_idea_stage_with_zero_users_and_revenue_never_selects_retention_test(
        self, pitchvision_base_state
    ):
        """
        PitchVision AI has 0 paying customers, 0 active users, and no live product.
        decide_next_strategic_action MUST NOT select RETENTION_TEST even if the founder
        mentions recurring terms like 'matches played weekly' or 'coaches review weekly'.
        """
        # Inject tricky founder phrasing that previously fooled naive substring matching
        pitchvision_base_state["raw_context"] = (
            "We are PitchVision. Matches are played weekly, coaches do weekly video reviews. "
            "We are not paying yet and have zero revenue, no live product deployed yet."
        )
        pitchvision_base_state["founder_messages"] = [
            {"role": "founder", "content": pitchvision_base_state["raw_context"]}
        ]

        res = decide_next_strategic_action(pitchvision_base_state)
        action_type = res["next_strategic_action"]["action_type"]

        assert action_type != NextStrategicActionType.RETENTION_TEST.value
        assert action_type != "RETENTION_TEST"
        # Must be an early validation / discovery action
        assert action_type in (
            NextStrategicActionType.NEW_VALIDATION_TEST.value,
            NextStrategicActionType.BUILD_MINIMAL_PROTOTYPE.value,
            NextStrategicActionType.MORE_FOUNDER_KNOWLEDGE.value,
        )

    def test_retention_test_only_selectable_with_verified_live_traction(
        self, pitchvision_base_state
    ):
        """
        RETENTION_TEST requires verified launched product AND paying/active users > 0.
        """
        # Configure verified live traction
        pitchvision_base_state["startup_progress"]["mvp_exists"] = {"value": True, "provenance": "EXPLICIT"}
        pitchvision_base_state["startup_progress"]["development_stage"] = {"value": "LIVE_PRODUCT", "provenance": "EXPLICIT"}
        pitchvision_base_state["startup_progress"]["paying_customers_count"] = 8
        pitchvision_base_state["startup_progress"]["active_users_count"] = 14
        pitchvision_base_state["founder_messages"] = [
            {"role": "founder", "content": "We have 8 paying teams and 14 active teams using weekly, but churn is high."}
        ]
        pitchvision_base_state["critical_assumption"] = {
            "statement": "Teams will continue using weekly and not churn after initial matches",
            "area": "Retention",
        }
        pitchvision_base_state["weak_area"] = "Retention"

        res = decide_next_strategic_action(pitchvision_base_state)
        assert res["next_strategic_action"]["action_type"] == NextStrategicActionType.RETENTION_TEST.value


# ---------------------------------------------------------------------------
# Defect 5: interpret_founder_turn intake intent classification
# ---------------------------------------------------------------------------

class TestInterpretFounderTurnIntakePurity:
    @patch("startary.graph.nodes._safe_get_llm_client")
    def test_initial_intake_message_is_not_classified_as_new_evidence(
        self, mock_get_client, pitchvision_base_state, pitchvision_raw_message
    ):
        """
        When a founder provides initial startup profile information (team, budget, hours, access),
        interpret_founder_turn MUST classify it as ANSWER, not NEW_EVIDENCE.
        """
        # Simulate LLM returning NEW_EVIDENCE due to operational facts mentioned
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_output = FounderTurnInterpretationLLMOutput(
            intent_types=["ANSWER", "NEW_EVIDENCE"],
            direct_answer=True,
            new_evidence_summary="Founder mentioned two teams and three coaches.",
            response_priority="NORMAL",
            interpretation_summary="Founder shared background context on PitchVision AI.",
        )
        mock_structured.invoke.return_value = mock_output
        mock_llm.with_structured_output.return_value = mock_structured
        mock_get_client.return_value = mock_llm

        # Set up state at turn 0 / intake
        pitchvision_base_state["pending_question"] = None
        pitchvision_base_state["clarification_turn_count"] = 0
        pitchvision_base_state["active_experiment"] = None
        pitchvision_base_state["evidence_submissions"] = []

        res = interpret_founder_turn(pitchvision_base_state)
        turn_interp = res["latest_founder_turn_interpretation"]

        # Invariant checks:
        # 1. NEW_EVIDENCE is sanitized on turn 0 / intake
        assert "NEW_EVIDENCE" not in turn_interp["intent_types"]
        assert "ANSWER" in turn_interp["intent_types"]
        # 2. new_evidence_summary is None
        assert turn_interp["new_evidence_summary"] is None
