"""
Tests: Milestone 3.2 Founder Intent & Conversational Awareness.

Verification:
- FounderTurnInterpretation domain model validation.
- Multi-intent classification and priority escalation (HIGH for questions/fears/corrections).
- Correction handling.
- Does-not-know and cannot-act-now handling.
- One-word 'yes' extraction guard against speculative inference leakage.
- Advisor response composition (answers founder before next question, no experiment instructions in UNDERSTAND).
- Diagnosis delta generation only on material new evidence.
- Direct development entry points.
"""

from unittest.mock import MagicMock
import pytest

from startary.domain.enums import WorkflowPhase
from startary.domain.execution_context import ExecutionContext
from startary.domain.founder_intent import (
    AdvisorResponse,
    FounderIntentType,
    FounderResponsePriority,
    FounderTurnInterpretation,
)
from startary.domain.startup_progress import StartupProgress
from startary.graph.builder import run_advisor_response, run_founder_turn_interpretation
from startary.graph.graph_state import StartaryGraphState
from startary.graph.nodes import (
    compose_advisor_response,
    extract_startup_state,
    interpret_founder_turn,
)
from startary.llm.schemas import (
    AdvisorResponseLLMOutput,
    ExtractedKnowledgeFieldLLMOutput,
    FounderTurnInterpretationLLMOutput,
    StartupStateDeltaLLMOutput,
)
from tests.fixtures.scenarios import (
    make_valid_advisor_response_output_dict,
    make_valid_founder_turn_interpretation_output_dict,
)


@pytest.fixture
def intent_base_state() -> StartaryGraphState:
    return {
        "project_id": "proj-intent-123",
        "phase": WorkflowPhase.UNDERSTAND,
        "iteration": 1,
        "raw_context": "Shiftly helps small café managers manage shift changes easily.",
        "weak_area": "Problem",
        "selection_confidence": "LOW",
        "pending_question": "Have you spoken with other café owners?",
        "active_interrupt_type": "strategic",
        "current_clarifying_question": {
            "question": "What is your approximate budget for testing?",
            "target_area": "Founder Constraints",
            "information_goal": "Determine resource runway",
        },
        "founder_messages": [
            {"role": "founder", "content": "Shiftly helps café managers manage shift swaps."},
            {"role": "startary", "content": "Have you spoken with other café owners?"},
            {
                "role": "founder",
                "content": "No I haven't, my fear is whether this problem is worth my time and energy, what is your honest reaction?",
            },
        ],
        "execution_context": ExecutionContext().model_dump(),
        "startup_progress": StartupProgress().model_dump(),
        "traces": [],
    }


class TestFounderTurnInterpretationDomain:
    def test_interpretation_schema_validates(self):
        interp = FounderTurnInterpretation(
            project_id="proj-123",
            turn_index=2,
            intent_types=[FounderIntentType.ANSWER, FounderIntentType.CONCERN, FounderIntentType.QUESTION],
            direct_answer=True,
            direct_question="What is your honest reaction?",
            concern_or_goal="Fear of wasting time on an invalid problem",
            response_priority=FounderResponsePriority.HIGH,
            interpretation_summary="Founder answered and asked for candid evaluation due to fear of wasting time.",
        )
        assert interp.project_id == "proj-123"
        assert interp.response_priority == FounderResponsePriority.HIGH
        assert FounderIntentType.QUESTION in interp.intent_types
        assert interp.direct_question == "What is your honest reaction?"


class TestInterpretFounderTurnNode:
    def test_multi_intent_message_gets_high_priority(self, mocker, intent_base_state):
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_out = make_valid_founder_turn_interpretation_output_dict(
            intent_types=["ANSWER", "CONCERN", "QUESTION", "REQUEST_FOR_ADVICE"],
            direct_question="What is your honest reaction?",
            concern_or_goal="Fear of wasting time and energy on a weak problem",
            response_priority="HIGH",
        )
        mock_structured.invoke.return_value = FounderTurnInterpretationLLMOutput(**mock_out)
        mock_llm.with_structured_output.return_value = mock_structured

        mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
        mocker.patch("startary.graph.nodes.get_settings", return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"))

        res = interpret_founder_turn(intent_base_state)
        interpretation = res["latest_founder_turn_interpretation"]

        assert interpretation["response_priority"] == "HIGH"
        assert "QUESTION" in interpretation["intent_types"]
        assert "CONCERN" in interpretation["intent_types"]
        assert interpretation["direct_question"] == "What is your honest reaction?"
        assert res["founder_current_concern"] == "Fear of wasting time and energy on a weak problem"

    def test_correction_gets_classified_and_recorded(self, mocker, intent_base_state):
        intent_base_state["founder_messages"][-1] = {
            "role": "founder",
            "content": "I'm not the manager; my friend is.",
        }
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_out = make_valid_founder_turn_interpretation_output_dict(
            intent_types=["CORRECTION"],
            correction="Founder is not a manager; reporting friend's experience",
            response_priority="HIGH",
        )
        mock_structured.invoke.return_value = FounderTurnInterpretationLLMOutput(**mock_out)
        mock_llm.with_structured_output.return_value = mock_structured

        mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
        mocker.patch("startary.graph.nodes.get_settings", return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"))

        res = interpret_founder_turn(intent_base_state)
        assert res["latest_founder_turn_interpretation"]["response_priority"] == "HIGH"
        assert "CORRECTION" in res["latest_founder_turn_interpretation"]["intent_types"]
        assert len(res["recent_corrections"]) >= 1

    def test_does_not_know_classification(self, mocker, intent_base_state):
        intent_base_state["founder_messages"][-1] = {
            "role": "founder",
            "content": "I don't know.",
        }
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_out = make_valid_founder_turn_interpretation_output_dict(
            intent_types=["DOES_NOT_KNOW"],
            cannot_answer_now=True,
            response_priority="HIGH",
        )
        mock_structured.invoke.return_value = FounderTurnInterpretationLLMOutput(**mock_out)
        mock_llm.with_structured_output.return_value = mock_structured

        mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
        mocker.patch("startary.graph.nodes.get_settings", return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"))

        res = interpret_founder_turn(intent_base_state)
        interp = res["latest_founder_turn_interpretation"]
        assert interp["cannot_answer_now"] is True
        assert "DOES_NOT_KNOW" in interp["intent_types"]

    def test_cannot_act_now_sets_context(self, mocker, intent_base_state):
        intent_base_state["founder_messages"][-1] = {
            "role": "founder",
            "content": "Right now I cannot make conversation or do outreach, but I will later.",
        }
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_out = make_valid_founder_turn_interpretation_output_dict(
            intent_types=["CANNOT_ACT_NOW"],
            cannot_act_now=True,
            response_priority="HIGH",
        )
        mock_structured.invoke.return_value = FounderTurnInterpretationLLMOutput(**mock_out)
        mock_llm.with_structured_output.return_value = mock_structured

        mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
        mocker.patch("startary.graph.nodes.get_settings", return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"))

        res = interpret_founder_turn(intent_base_state)
        assert res["cannot_act_now_context"] is True


class TestOneWordAnswerExtractionGuard:
    def test_one_word_yes_does_not_leak_unrelated_inferred_fields(self, mocker, intent_base_state):
        """
        Section 16 / 40: When founder replies 'yes' to customer access,
        it must not populate unrelated fields like revenue=0, design=false, launched=false.
        """
        intent_base_state["pending_question"] = "Do you have direct access to local café owners?"
        intent_base_state["founder_messages"] = [
            {"role": "startary", "content": "Do you have direct access to local café owners?"},
            {"role": "founder", "content": "yes"},
        ]

        # Simulate an over-zealous LLM that hallucinates unasked negative inferences alongside access
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_delta = StartupStateDeltaLLMOutput(
            customer_access=ExtractedKnowledgeFieldLLMOutput(
                value="Direct personal access to café owners",
                provenance="INFERRED",
                confidence="HIGH",
            ),
            has_design_capability=ExtractedKnowledgeFieldLLMOutput(
                value="false",
                provenance="INFERRED",
                confidence="LOW",
            ),
            revenue=ExtractedKnowledgeFieldLLMOutput(
                value="0",
                provenance="INFERRED",
                confidence="LOW",
            ),
            launched=ExtractedKnowledgeFieldLLMOutput(
                value="false",
                provenance="INFERRED",
                confidence="LOW",
            ),
        )
        mock_structured.invoke.return_value = mock_delta
        mock_llm.with_structured_output.return_value = mock_structured

        mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
        mocker.patch("startary.graph.nodes.get_settings", return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"))

        res = extract_startup_state(intent_base_state)
        delta = res["startup_state_delta"]

        # customer_access is related to pending question -> preserved
        assert delta["customer_access"] is not None
        # design, revenue, launched were NOT mentioned or asked -> dropped by short-answer guard
        assert delta["has_design_capability"] is None
        assert delta["revenue"] is None
        assert delta["launched"] is None


class TestComposeAdvisorResponseNode:
    def test_advisor_response_answers_founder_first(self, mocker, intent_base_state):
        intent_base_state["latest_founder_turn_interpretation"] = {
            "direct_question": "What is your honest reaction?",
            "concern_or_goal": "Fear of wasting time",
            "response_priority": "HIGH",
        }

        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_out = make_valid_advisor_response_output_dict(
            response_text=(
                "I wouldn't call Shiftly validated yet, but I also wouldn't stop based on what you know. "
                "One real café gives you an initial signal, but avoid heavy build investment until we test whether others share it. "
                "To see what testing is feasible, roughly how much budget can you risk right now?"
            ),
            next_question="Roughly how much budget can you risk right now?",
        )
        mock_structured.invoke.return_value = AdvisorResponseLLMOutput(**mock_out)
        mock_llm.with_structured_output.return_value = mock_structured

        mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
        mocker.patch("startary.graph.nodes.get_settings", return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"))

        res = compose_advisor_response(intent_base_state)
        advisor_resp = res["latest_advisor_response"]

        assert "validated yet" in advisor_resp["response_text"]
        assert "budget" in advisor_resp["response_text"]
        assert advisor_resp["should_ask_next_question"] is True

    def test_no_experiment_instructions_during_understand(self, mocker, intent_base_state):
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_out = make_valid_advisor_response_output_dict(
            response_text="From the café manager you know, what manual tools do they currently use when staff cancel?",
            next_question="What manual tools do they currently use?",
        )
        mock_structured.invoke.return_value = AdvisorResponseLLMOutput(**mock_out)
        mock_llm.with_structured_output.return_value = mock_structured

        mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
        mocker.patch("startary.graph.nodes.get_settings", return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"))

        res = compose_advisor_response(intent_base_state)
        text = res["latest_advisor_response"]["response_text"].lower()

        # UNDERSTAND questions must not order the founder to execute experiments
        assert "run an experiment" not in text
        assert "charge them $20" not in text
        assert "launch a pilot" not in text

    def test_direct_entry_points(self, mocker, intent_base_state):
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = FounderTurnInterpretationLLMOutput(
            **make_valid_founder_turn_interpretation_output_dict()
        )
        mock_llm.with_structured_output.return_value = mock_structured

        mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
        mocker.patch("startary.graph.nodes.get_settings", return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"))

        res_state = run_founder_turn_interpretation(intent_base_state)
        assert "latest_founder_turn_interpretation" in res_state

