"""
Tests: Startup State Extractor & Deterministic Merge — Milestone 2.6.

Tests:
  - extract_startup_state node
  - merge_execution_state node
  - Scenarios A through I (from Section 28 of M2.6 spec)
  - Semantic separation: users != paying_customers != revenue
  - Semantic separation: funding != available budget
  - Currency preservation
  - Multi-turn state accumulation
  - AI messages excluded from extraction source
"""

from unittest.mock import MagicMock
import pytest

from startary.domain.enums import WorkflowPhase
from startary.domain.execution_context import ExecutionContext
from startary.domain.provenance import FactConfidence, FactProvenance
from startary.domain.startup_progress import DevelopmentStage, ProductStatus, StartupProgress
from startary.graph.graph_state import StartaryGraphState
from startary.graph.nodes import (
    _merge_knowledge_field,
    extract_startup_state,
    merge_execution_state,
)
from startary.llm.schemas import (
    ExtractedKnowledgeFieldLLMOutput,
    ExtractedTeamMemberLLMOutput,
    StartupStateDeltaLLMOutput,
)
from tests.fixtures.scenarios import (
    SCENARIO_M26_A_RICH_PROFILE,
    SCENARIO_M26_B_MVP_FALSE_ONLY,
    SCENARIO_M26_C_USERS_NO_REVENUE,
    SCENARIO_M26_D_REVENUE_NO_USER_COUNT,
    SCENARIO_M26_E_TEAM_UPDATE,
    SCENARIO_M26_F_AMBIGUOUS_TEAM,
    SCENARIO_M26_G_GEOGRAPHY_EXPLICIT,
    SCENARIO_M26_H_EXISTING_USER_ACCESS,
    SCENARIO_M26_I_STRATEGIC_REVEALS_PROFILE,
)


def _make_state_for_extraction(messages: list[dict[str, str]], turn: int = 0) -> StartaryGraphState:
    return StartaryGraphState(
        project_id="test-extractor-proj",
        phase=WorkflowPhase.UNDERSTAND,
        iteration=1,
        founder_messages=messages,
        clarification_turn_count=turn,
        execution_context=ExecutionContext().model_dump(),
        startup_progress=StartupProgress().model_dump(),
        startup_state_delta={},
        traces=[],
    )


class TestDeterministicMergeLogic:
    def test_merge_explicit_into_unknown(self):
        existing = {"value": None, "provenance": "UNKNOWN", "confidence": "LOW", "source_turn": None}
        incoming = {"value": "3", "provenance": "EXPLICIT", "confidence": "HIGH"}
        merged = _merge_knowledge_field(existing, incoming, source_turn=1)
        assert merged["value"] == "3"
        assert merged["provenance"] == "EXPLICIT"
        assert merged["source_turn"] == 1

    def test_inferred_cannot_overwrite_explicit(self):
        existing = {"value": "3", "provenance": "EXPLICIT", "confidence": "HIGH", "source_turn": 1}
        incoming = {"value": "4", "provenance": "INFERRED", "confidence": "MEDIUM"}
        merged = _merge_knowledge_field(existing, incoming, source_turn=2)
        assert merged["value"] == "3"
        assert merged["provenance"] == "EXPLICIT"

    def test_newer_explicit_overwrites_older_explicit(self):
        existing = {"value": "2", "provenance": "EXPLICIT", "confidence": "HIGH", "source_turn": 1}
        incoming = {"value": "3", "provenance": "EXPLICIT", "confidence": "HIGH"}
        merged = _merge_knowledge_field(existing, incoming, source_turn=2)
        assert merged["value"] == "3"
        assert merged["provenance"] == "EXPLICIT"
        assert merged["source_turn"] == 2

    def test_conflicting_explicit_same_turn(self):
        existing = {"value": "2", "provenance": "EXPLICIT", "confidence": "HIGH", "source_turn": 1}
        incoming = {"value": "4", "provenance": "EXPLICIT", "confidence": "HIGH"}
        merged = _merge_knowledge_field(existing, incoming, source_turn=1)
        assert merged["provenance"] == "CONFLICTING"
        assert merged["value"] is None

    def test_unknown_never_overwrites_known(self):
        existing = {"value": "300", "provenance": "EXPLICIT", "confidence": "HIGH", "source_turn": 1}
        incoming = {"value": None, "provenance": "UNKNOWN", "confidence": "LOW"}
        merged = _merge_knowledge_field(existing, incoming, source_turn=2)
        assert merged["value"] == "300"


class TestMandatoryScenarios:
    """Tests covering Scenarios A through I from Milestone 2.6 spec."""

    def test_scenario_a_rich_profile_in_one_message(self, mocker):
        """Scenario A: 3 founders, 2 devs + 1 designer, 10h/wk, prototype but no MVP, 300 AZN/mo."""
        delta = StartupStateDeltaLLMOutput(
            founder_count=ExtractedKnowledgeFieldLLMOutput(value="3", provenance="EXPLICIT", confidence="HIGH"),
            active_team_size=ExtractedKnowledgeFieldLLMOutput(value="3", provenance="EXPLICIT", confidence="HIGH"),
            weekly_hours_per_member=ExtractedKnowledgeFieldLLMOutput(value="10", provenance="EXPLICIT", confidence="HIGH"),
            total_weekly_capacity_hours=ExtractedKnowledgeFieldLLMOutput(value="30", provenance="INFERRED", confidence="HIGH"),
            has_technical_capability=ExtractedKnowledgeFieldLLMOutput(value="true", provenance="EXPLICIT", confidence="HIGH"),
            has_design_capability=ExtractedKnowledgeFieldLLMOutput(value="true", provenance="EXPLICIT", confidence="HIGH"),
            prototype_exists=ExtractedKnowledgeFieldLLMOutput(value="true", provenance="EXPLICIT", confidence="HIGH"),
            mvp_exists=ExtractedKnowledgeFieldLLMOutput(value="false", provenance="EXPLICIT", confidence="HIGH"),
            available_budget_amount=ExtractedKnowledgeFieldLLMOutput(value="300", provenance="EXPLICIT", confidence="HIGH"),
            budget_currency=ExtractedKnowledgeFieldLLMOutput(value="AZN", provenance="EXPLICIT", confidence="HIGH"),
            budget_period=ExtractedKnowledgeFieldLLMOutput(value="month", provenance="EXPLICIT", confidence="HIGH"),
            existing_assets=["prototype"],
        )

        state = _make_state_for_extraction([{"role": "founder", "content": SCENARIO_M26_A_RICH_PROFILE}])
        state["startup_state_delta"] = delta.model_dump()
        state["startup_state_delta"]["_source_turn"] = 0

        merged = merge_execution_state(state)
        exec_ctx = merged["execution_context"]
        prog = merged["startup_progress"]

        assert exec_ctx["active_team_size"]["value"] == "3"
        assert exec_ctx["active_team_size"]["provenance"] == "EXPLICIT"
        assert exec_ctx["total_weekly_capacity_hours"]["value"] == "30"
        assert exec_ctx["total_weekly_capacity_hours"]["provenance"] == "INFERRED"
        assert exec_ctx["available_budget_amount"]["value"] == "300"
        assert exec_ctx["budget_currency"]["value"] == "AZN"
        assert prog["mvp_exists"]["value"] == "false"
        assert prog["mvp_exists"]["provenance"] == "EXPLICIT"
        assert "prototype" in exec_ctx["existing_assets"]

    def test_scenario_b_no_mvp_only_does_not_infer_unrelated(self):
        """Scenario B: 'No MVP yet' safely implies mvp_exists=false, nothing else."""
        delta = StartupStateDeltaLLMOutput(
            mvp_exists=ExtractedKnowledgeFieldLLMOutput(value="false", provenance="EXPLICIT", confidence="HIGH"),
        )
        state = _make_state_for_extraction([{"role": "founder", "content": SCENARIO_M26_B_MVP_FALSE_ONLY}])
        state["startup_state_delta"] = delta.model_dump()

        merged = merge_execution_state(state)
        prog = merged["startup_progress"]
        exec_ctx = merged["execution_context"]

        assert prog["mvp_exists"]["value"] == "false"
        assert prog["mvp_exists"]["provenance"] == "EXPLICIT"
        # Must NOT infer unrelated fields
        assert prog["development_stage"]["provenance"] == "UNKNOWN"
        assert prog["active_users"]["provenance"] == "UNKNOWN"
        assert prog["revenue"]["provenance"] == "UNKNOWN"
        assert exec_ctx["founder_count"]["provenance"] == "UNKNOWN"
        assert exec_ctx["funding_status"]["provenance"] == "UNKNOWN"

    def test_scenario_c_users_without_revenue_does_not_invent_paying_customers(self):
        """Scenario C: launched=true, 40 active users, no revenue -> paying customers not invented."""
        delta = StartupStateDeltaLLMOutput(
            launched=ExtractedKnowledgeFieldLLMOutput(value="true", provenance="EXPLICIT", confidence="HIGH"),
            active_users=ExtractedKnowledgeFieldLLMOutput(value="40", provenance="EXPLICIT", confidence="HIGH"),
            revenue=ExtractedKnowledgeFieldLLMOutput(value="0", provenance="EXPLICIT", confidence="HIGH", notes="no revenue"),
        )
        state = _make_state_for_extraction([{"role": "founder", "content": SCENARIO_M26_C_USERS_NO_REVENUE}])
        state["startup_state_delta"] = delta.model_dump()

        merged = merge_execution_state(state)
        prog = merged["startup_progress"]

        assert prog["launched"]["value"] == "true"
        assert prog["active_users"]["value"] == "40"
        assert prog["revenue"]["value"] == "0"
        # Paying customers MUST remain UNKNOWN — not invented
        assert prog["paying_customers"]["provenance"] == "UNKNOWN"

    def test_scenario_d_revenue_without_user_count(self):
        """Scenario D: Revenue known, user count remains UNKNOWN."""
        delta = StartupStateDeltaLLMOutput(
            revenue=ExtractedKnowledgeFieldLLMOutput(value="$500/month", provenance="EXPLICIT", confidence="HIGH"),
        )
        state = _make_state_for_extraction([{"role": "founder", "content": SCENARIO_M26_D_REVENUE_NO_USER_COUNT}])
        state["startup_state_delta"] = delta.model_dump()

        merged = merge_execution_state(state)
        prog = merged["startup_progress"]

        assert prog["revenue"]["value"] == "$500/month"
        assert prog["active_users"]["provenance"] == "UNKNOWN"

    def test_scenario_e_team_update_across_turns(self):
        """Scenario E: Team size updates cleanly across clarification turns."""
        # Turn 1: 2 founders
        state = _make_state_for_extraction([], turn=0)
        state["startup_state_delta"] = StartupStateDeltaLLMOutput(
            founder_count=ExtractedKnowledgeFieldLLMOutput(value="2", provenance="EXPLICIT", confidence="HIGH"),
            active_team_size=ExtractedKnowledgeFieldLLMOutput(value="2", provenance="EXPLICIT", confidence="HIGH"),
        ).model_dump()
        state["startup_state_delta"]["_source_turn"] = 0

        turn1_merged = merge_execution_state(state)
        state["execution_context"] = turn1_merged["execution_context"]
        assert state["execution_context"]["active_team_size"]["value"] == "2"

        # Turn 2: Designer joins -> 3 active members
        state["startup_state_delta"] = StartupStateDeltaLLMOutput(
            active_team_size=ExtractedKnowledgeFieldLLMOutput(value="3", provenance="EXPLICIT", confidence="HIGH"),
            has_design_capability=ExtractedKnowledgeFieldLLMOutput(value="true", provenance="EXPLICIT", confidence="HIGH"),
        ).model_dump()
        state["startup_state_delta"]["_source_turn"] = 1

        turn2_merged = merge_execution_state(state)
        exec_ctx = turn2_merged["execution_context"]
        assert exec_ctx["active_team_size"]["value"] == "3"
        assert exec_ctx["active_team_size"]["source_turn"] == 1
        # Original founder_count remains 2 from turn 0
        assert exec_ctx["founder_count"]["value"] == "2"

    def test_scenario_f_ambiguous_team_does_not_infer_founder_count(self):
        """Scenario F: 'Three people helping' should NOT set founder_count=3."""
        delta = StartupStateDeltaLLMOutput(
            active_team_size=ExtractedKnowledgeFieldLLMOutput(value="3", provenance="EXPLICIT", confidence="MEDIUM"),
            # founder_count left None because ambiguous
        )
        state = _make_state_for_extraction([{"role": "founder", "content": SCENARIO_M26_F_AMBIGUOUS_TEAM}])
        state["startup_state_delta"] = delta.model_dump()

        merged = merge_execution_state(state)
        exec_ctx = merged["execution_context"]
        assert exec_ctx["active_team_size"]["value"] == "3"
        assert exec_ctx["founder_count"]["provenance"] == "UNKNOWN"

    def test_scenario_g_geography_market_only(self):
        """Scenario G: 'targeting university students in Azerbaijan' -> market geography only."""
        delta = StartupStateDeltaLLMOutput(
            geographic_market=ExtractedKnowledgeFieldLLMOutput(value="Azerbaijan", provenance="EXPLICIT", confidence="HIGH"),
        )
        state = _make_state_for_extraction([{"role": "founder", "content": SCENARIO_M26_G_GEOGRAPHY_EXPLICIT}])
        state["startup_state_delta"] = delta.model_dump()

        merged = merge_execution_state(state)
        exec_ctx = merged["execution_context"]
        assert exec_ctx["geographic_market"]["value"] == "Azerbaijan"
        assert exec_ctx["geographic_market"]["provenance"] == "EXPLICIT"

    def test_scenario_h_existing_user_access_asset(self):
        """Scenario H: 1,200 member community -> audience and asset extracted."""
        delta = StartupStateDeltaLLMOutput(
            existing_audience_or_community=ExtractedKnowledgeFieldLLMOutput(
                value="Telegram community with 1,200 members", provenance="EXPLICIT", confidence="HIGH"
            ),
            customer_access=ExtractedKnowledgeFieldLLMOutput(
                value="Telegram student community", provenance="EXPLICIT", confidence="HIGH"
            ),
            existing_assets=["telegram_community_1200_members"],
        )
        state = _make_state_for_extraction([{"role": "founder", "content": SCENARIO_M26_H_EXISTING_USER_ACCESS}])
        state["startup_state_delta"] = delta.model_dump()

        merged = merge_execution_state(state)
        exec_ctx = merged["execution_context"]
        assert "1,200" in exec_ctx["existing_audience_or_community"]["value"]
        assert "telegram_community_1200_members" in exec_ctx["existing_assets"]

    def test_scenario_i_strategic_answer_with_hidden_profile_info(self):
        """Scenario I: Answering a problem question reveals validation activity + team info."""
        delta = StartupStateDeltaLLMOutput(
            customer_access=ExtractedKnowledgeFieldLLMOutput(value="university club", provenance="EXPLICIT", confidence="HIGH"),
            founder_count=ExtractedKnowledgeFieldLLMOutput(value="2", provenance="EXPLICIT", confidence="HIGH"),
            validation_activity=ExtractedKnowledgeFieldLLMOutput(value="10 student interviews", provenance="EXPLICIT", confidence="HIGH"),
            interviews_completed=ExtractedKnowledgeFieldLLMOutput(value="10", provenance="EXPLICIT", confidence="HIGH"),
        )
        state = _make_state_for_extraction([{"role": "founder", "content": SCENARIO_M26_I_STRATEGIC_REVEALS_PROFILE}])
        state["startup_state_delta"] = delta.model_dump()

        merged = merge_execution_state(state)
        exec_ctx = merged["execution_context"]
        prog = merged["startup_progress"]

        assert exec_ctx["customer_access"]["value"] == "university club"
        assert exec_ctx["founder_count"]["value"] == "2"
        assert prog["interviews_completed"]["value"] == "10"


class TestSourceSeparationAndSafety:
    def test_ai_messages_excluded_from_source(self, mocker):
        """System and Startary messages must not be passed to extractor."""
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = StartupStateDeltaLLMOutput()
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured

        mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
        mocker.patch(
            "startary.graph.nodes.get_settings",
            return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"),
        )

        messages = [
            {"role": "founder", "content": "I am building an app alone."},
            {"role": "startary", "content": "What is your funding status and runway?"},
            {"role": "founder", "content": "I have no funding, just savings."},
        ]
        state = _make_state_for_extraction(messages)
        extract_startup_state(state)

        call_args = mock_structured.invoke.call_args[0][0]
        user_msg = call_args[1]["content"]

        # Founder content must be present
        assert "I am building an app alone." in user_msg
        assert "I have no funding" in user_msg
        # Startary message text must NOT be in the source messages
        assert "What is your funding status and runway?" not in user_msg

