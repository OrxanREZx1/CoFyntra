"""
Tests: Adaptive Information Gap Planner — Milestone 2.6.

Tests:
  - plan_information_gap node
  - Single question guarantee (asks at most ONE question)
  - Unknown fields alone do NOT automatically trigger questions
  - REQUIRED_NOW gap triggers a question
  - USEFUL_LATER gap does not force a question
  - Strategic question priority over profile questions
  - Routing logic with route_on_gap_plan
  - Backward compatibility shims
"""

from unittest.mock import MagicMock
import pytest

from startary.domain.enums import WorkflowPhase
from startary.domain.execution_context import ExecutionContext
from startary.domain.gap_plan import GapPriority, GapTargetDomain, InformationGapPlan
from startary.domain.startup_progress import StartupProgress
from startary.domain.taxonomy import StartupArea
from startary.graph.builder import route_on_context_sufficiency, route_on_gap_plan
from startary.graph.graph_state import StartaryGraphState
from startary.graph.nodes import plan_information_gap
from startary.llm.schemas import InformationGapPlanLLMOutput
from tests.fixtures.scenarios import make_valid_gap_plan_output_dict


def _make_state_for_gap_planner(
    turn: int = 0,
    strategic_summary: str = "Problem partially known.",
    exec_ctx: dict | None = None,
    progress: dict | None = None,
) -> StartaryGraphState:
    return StartaryGraphState(
        project_id="test-gap-planner-proj",
        phase=WorkflowPhase.UNDERSTAND,
        iteration=1,
        raw_context="Testing gap planner.",
        founder_messages=[{"role": "founder", "content": "Testing gap planner."}],
        clarification_turn_count=turn,
        clarification_history=[],
        startup_state={"areas": {}},
        diagnosis={"summary": strategic_summary},
        execution_context=exec_ctx or ExecutionContext().model_dump(),
        startup_progress=progress or StartupProgress().model_dump(),
        traces=[],
    )


class TestInformationGapPlannerNode:
    def test_plan_information_gap_returns_single_question_when_needed(self, mocker):
        mock_output = InformationGapPlanLLMOutput(
            should_ask=True,
            strategic_sufficient=False,
            target_domain="STRATEGIC",
            target_field_or_area="Problem",
            priority="REQUIRED_NOW",
            information_goal="Clarify specific student friction.",
            reason="Problem hypothesis is ungrounded.",
            question="What specific difficulty do students report when choosing their branch?",
        )
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = mock_output
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured

        mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
        mocker.patch(
            "startary.graph.nodes.get_settings",
            return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"),
        )

        state = _make_state_for_gap_planner()
        result = plan_information_gap(state)

        assert "gap_plan" in result
        gap_plan = result["gap_plan"]
        assert gap_plan["should_ask"] is True
        assert gap_plan["strategic_sufficient"] is False
        assert gap_plan["target_domain"] == "STRATEGIC"
        assert gap_plan["priority"] == "REQUIRED_NOW"
        assert result["current_clarifying_question"] is not None
        assert len(result["current_clarifying_question"]["question"]) > 10
        # Backward compatibility check
        assert result["is_sufficient"] is False
        assert "context_sufficiency" in result

    def test_plan_information_gap_proceeds_when_sufficient(self, mocker):
        mock_output = InformationGapPlanLLMOutput(
            should_ask=False,
            strategic_sufficient=True,
            reason="Context is grounded enough with 15 customer interviews.",
        )
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = mock_output
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured

        mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
        mocker.patch(
            "startary.graph.nodes.get_settings",
            return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"),
        )

        state = _make_state_for_gap_planner()
        result = plan_information_gap(state)

        assert result["gap_plan"]["should_ask"] is False
        assert result["gap_plan"]["strategic_sufficient"] is True
        assert result["current_clarifying_question"] is None
        assert result["is_sufficient"] is True

    def test_unknown_fields_alone_do_not_trigger_question(self, mocker):
        """Even if budget, users, and funding are UNKNOWN, planner proceeds if not REQUIRED_NOW."""
        mock_output = InformationGapPlanLLMOutput(
            should_ask=False,
            strategic_sufficient=True,
            priority="USEFUL_LATER",
            reason="Budget and funding are UNKNOWN but not REQUIRED_NOW during early problem diagnosis.",
        )
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = mock_output
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured

        mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
        mocker.patch(
            "startary.graph.nodes.get_settings",
            return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"),
        )

        # State has unknown execution fields
        state = _make_state_for_gap_planner(exec_ctx=ExecutionContext().model_dump())
        result = plan_information_gap(state)

        assert result["gap_plan"]["should_ask"] is False
        assert result["is_sufficient"] is True

    def test_strategic_gap_has_higher_priority_than_profile_gap(self, mocker):
        """When strategic grounding is weak, strategic question is chosen over team/budget."""
        mock_output = InformationGapPlanLLMOutput(
            should_ask=True,
            strategic_sufficient=False,
            target_domain="STRATEGIC",
            target_field_or_area="Problem",
            priority="REQUIRED_NOW",
            information_goal="Establish whether problem was directly observed.",
            reason="Strategic grounding is LEVEL 1 (conceptual); resolves core uncertainty before asking team.",
            question="Is this based on your own experience or conversations with students?",
        )
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = mock_output
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured

        mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
        mocker.patch(
            "startary.graph.nodes.get_settings",
            return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"),
        )

        state = _make_state_for_gap_planner()
        result = plan_information_gap(state)

        assert result["gap_plan"]["target_domain"] == "STRATEGIC"
        assert result["gap_plan"]["priority"] == "REQUIRED_NOW"


class TestGapPlanRouting:
    def test_routes_to_assess_startup_areas_when_no_ask_and_sufficient(self):
        state: StartaryGraphState = {
            "gap_plan": {"should_ask": False, "strategic_sufficient": True},
            "clarification_turn_count": 0,
        }
        route = route_on_gap_plan(state)
        assert route == "assess_startup_areas"

    def test_routes_to_ask_when_should_ask_under_limit(self):
        state: StartaryGraphState = {
            "gap_plan": {"should_ask": True, "strategic_sufficient": False},
            "clarification_turn_count": 2,
        }
        route = route_on_gap_plan(state)
        assert route == "ask_and_wait_for_founder"

    def test_routes_to_stop_when_turn_limit_reached(self):
        state: StartaryGraphState = {
            "gap_plan": {"should_ask": True, "strategic_sufficient": False},
            "clarification_turn_count": 5,
        }
        route = route_on_gap_plan(state)
        assert route == "handle_insufficient_context_stop"

    def test_legacy_backward_compat_routing_alias(self):
        state: StartaryGraphState = {
            "context_sufficiency": {"sufficient": True},
            "clarification_turn_count": 0,
        }
        route = route_on_context_sufficiency(state)
        assert route == "assess_startup_areas"

