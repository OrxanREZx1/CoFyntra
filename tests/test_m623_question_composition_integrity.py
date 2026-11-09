"""
Milestone 6.2.3 — Question composition, interrupt safety, and transcript attribution.

LangGraph re-executes an interrupted node from the top on resume. Before M6.2.3 the advisor
composer ran inside ask_and_wait_for_founder, so it was called once before the founder saw the
question and again on resume, and the transcript stored the regenerated wording. These tests run
the REAL compiled graph (real checkpointer, interrupt/resume, deterministic nodes) on the
PitchVision AI first turn with a scripted offline LLM, and prove:

  1. exactly one question-generation call before the interrupt;
  2. zero question-regeneration calls on resume;
  3. the saved question equals the displayed question, byte for byte;
  4. the founder answer attaches to the question it answered;
  5. checkpoint restore / retry / repeated resume never duplicate transcript entries.
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import patch

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from startary.config import Settings
from startary.domain.enums import WorkflowPhase
from startary.graph.builder import build_graph
from startary.graph.nodes import ask_and_wait_for_founder, compose_founder_question, merge_founder_answer
from tests.fixtures.pitchvision_scenario import (
    PITCHVISION_ANSWER_1,
    PITCHVISION_FIRST_MESSAGE,
    make_pitchvision_llm,
)

COMPOSER = "AdvisorResponseLLMOutput"
OFFLINE_SETTINGS = Settings(
    _env_file=None,
    openai_api_key="sk-offline-test",
    fast_model_name="scripted-fast",
    reasoning_model_name="scripted-reasoning",
)


@contextmanager
def offline_llm(llm):
    with patch("startary.graph.nodes.get_llm_client", side_effect=llm.get_llm_client), \
         patch("startary.graph.nodes.get_settings", return_value=OFFLINE_SETTINGS):
        yield


def _initial_state(thread_id: str) -> dict:
    return {
        "project_id": thread_id,
        "phase": WorkflowPhase.UNDERSTAND,
        "iteration": 1,
        "raw_context": PITCHVISION_FIRST_MESSAGE,
        "founder_messages": [{"role": "founder", "content": PITCHVISION_FIRST_MESSAGE}],
    }


def _cfg(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _interrupt_payload(graph, cfg) -> dict:
    snap = graph.get_state(cfg)
    assert snap.tasks and snap.tasks[0].interrupts, f"graph is not interrupted (next={snap.next})"
    return snap.tasks[0].interrupts[0].value


def _run_first_turn(graph, thread_id: str) -> dict:
    cfg = _cfg(thread_id)
    graph.invoke(_initial_state(thread_id), cfg)
    return _interrupt_payload(graph, cfg)


# ---------------------------------------------------------------------------
# 1–4: single composition, zero regeneration, exact transcript, correct attribution
# ---------------------------------------------------------------------------

class TestSingleCompositionAndExactTranscript:
    def test_one_question_generation_call_before_interrupt(self):
        llm = make_pitchvision_llm()
        with offline_llm(llm):
            graph = build_graph(checkpointer=MemorySaver())
            payload = _run_first_turn(graph, "pv-once")
            state = graph.get_state(_cfg("pv-once")).values

        assert llm.unscripted == []
        assert llm.count(COMPOSER) == 1
        # The composer runs after the question planner, and is the last LLM call before the interrupt.
        schemas = [c.schema for c in llm.calls]
        assert schemas[-2:] == ["WeakLinkClarifyingQuestionLLMOutput", COMPOSER]
        # The displayed text IS the single composer output, persisted in the checkpoint before the interrupt.
        assert payload["question"].startswith("[composer call 1]")
        assert payload["interrupt_type"] == "weak_link"
        prompt = state["pending_founder_prompt"]
        assert prompt["question"] == payload["question"]
        assert prompt["question_id"] == payload["question_id"]
        assert prompt["source"] == "advisor_composer"

    def test_resume_makes_zero_question_regeneration_calls(self):
        llm = make_pitchvision_llm()
        with offline_llm(llm):
            graph = build_graph(checkpointer=MemorySaver())
            _run_first_turn(graph, "pv-resume")
            calls_before = len(llm.calls)
            graph.invoke(Command(resume=PITCHVISION_ANSWER_1), _cfg("pv-resume"), interrupt_after=["merge_founder_answer"])
            snap = graph.get_state(_cfg("pv-resume"))

        assert snap.next == ("interpret_founder_turn",)
        # Resume -> ask_and_wait_for_founder -> merge_founder_answer: no LLM call of any kind.
        assert llm.calls[calls_before:] == []
        assert llm.count(COMPOSER) == 1

    def test_saved_question_exactly_matches_displayed_question(self):
        llm = make_pitchvision_llm()
        with offline_llm(llm):
            graph = build_graph(checkpointer=MemorySaver())
            payload = _run_first_turn(graph, "pv-exact")
            graph.invoke(Command(resume=PITCHVISION_ANSWER_1), _cfg("pv-exact"), interrupt_after=["merge_founder_answer"])
            state = graph.get_state(_cfg("pv-exact")).values

        displayed = payload["question"]
        assert state["founder_messages"][1] == {"role": "startary", "content": displayed}
        assert state["weak_link_clarification_history"][0]["question"] == displayed
        assert state["pending_question"] == displayed
        assert not any("[composer call 2]" in m["content"] for m in state["founder_messages"])

    def test_founder_response_attaches_to_correct_question(self):
        llm = make_pitchvision_llm()
        with offline_llm(llm):
            graph = build_graph(checkpointer=MemorySaver())
            payload = _run_first_turn(graph, "pv-attr")
            graph.invoke(Command(resume=PITCHVISION_ANSWER_1), _cfg("pv-attr"), interrupt_after=["merge_founder_answer"])
            state = graph.get_state(_cfg("pv-attr")).values

        assert [m["role"] for m in state["founder_messages"]] == ["founder", "startary", "founder"]
        assert state["founder_messages"][2]["content"] == PITCHVISION_ANSWER_1
        assert state["answered_question_id"] == payload["question_id"]
        assert state["question_history"][-1]["question_id"] == payload["question_id"]
        wl_entry = state["weak_link_clarification_history"][0]
        assert wl_entry["founder_answer"] == PITCHVISION_ANSWER_1
        assert state["weak_link_clarification_turns"] == 1
        # The answered prompt is consumed and can never be re-displayed.
        assert state["pending_founder_prompt"] is None
        assert state["latest_advisor_response"] is None


# ---------------------------------------------------------------------------
# 5: checkpoint restoration, retry, repeated resume
# ---------------------------------------------------------------------------

class TestCheckpointResumeSafety:
    def test_checkpoint_restoration_resumes_without_composer_or_duplicates(self):
        llm = make_pitchvision_llm()
        saver = MemorySaver()
        with offline_llm(llm):
            first_process_graph = build_graph(checkpointer=saver)
            payload = _run_first_turn(first_process_graph, "pv-restore")

            # Simulated restart: a freshly compiled graph restores the thread from the checkpointer.
            restored_graph = build_graph(checkpointer=saver)
            assert _interrupt_payload(restored_graph, _cfg("pv-restore")) == payload
            calls_before = len(llm.calls)
            restored_graph.invoke(Command(resume=PITCHVISION_ANSWER_1), _cfg("pv-restore"))
            payload_2 = _interrupt_payload(restored_graph, _cfg("pv-restore"))
            state = restored_graph.get_state(_cfg("pv-restore")).values

        resumed_composer_calls = [c for c in llm.calls[calls_before:] if c.schema == COMPOSER]
        # The only composer call after resume phrases the NEXT question, never the answered one.
        assert len(resumed_composer_calls) == 1
        assert f"Question ID: {payload_2['question_id']}" in resumed_composer_calls[0].messages[-1]["content"]
        assert payload_2["question_id"] != payload["question_id"]
        assert [m["content"] for m in state["founder_messages"]] == [
            PITCHVISION_FIRST_MESSAGE, payload["question"], PITCHVISION_ANSWER_1,
        ]
        assert len(state["weak_link_clarification_history"]) == 1

    def test_retry_after_downstream_failure_does_not_duplicate_transcript(self):
        assess_calls = {"n": 0}

        def flaky_assess(messages, rec):
            from tests.fixtures.pitchvision_scenario import tied_area_assessments
            assess_calls["n"] += 1
            if assess_calls["n"] in (2, 3):  # both attempts of the post-resume assessment fail
                raise RuntimeError("simulated provider outage")
            return tied_area_assessments()

        llm = make_pitchvision_llm(StartupAreaAssessmentBatchLLMOutput=flaky_assess)
        with offline_llm(llm):
            graph = build_graph(checkpointer=MemorySaver())
            payload = _run_first_turn(graph, "pv-retry")
            with pytest.raises(RuntimeError, match="assess_startup_areas failed"):
                graph.invoke(Command(resume=PITCHVISION_ANSWER_1), _cfg("pv-retry"))

            failed_state = graph.get_state(_cfg("pv-retry")).values
            assert len(failed_state["founder_messages"]) == 3
            composer_calls_before_retry = llm.count(COMPOSER)

            graph.invoke(None, _cfg("pv-retry"))  # /retry from the last good checkpoint
            payload_2 = _interrupt_payload(graph, _cfg("pv-retry"))
            state = graph.get_state(_cfg("pv-retry")).values

        assert composer_calls_before_retry == 1      # failure path never re-composed the answered question
        assert llm.count(COMPOSER) == 2              # retry composed only the next question
        assert [m["content"] for m in state["founder_messages"]] == [
            PITCHVISION_FIRST_MESSAGE, payload["question"], PITCHVISION_ANSWER_1,
        ]
        assert len(state["weak_link_clarification_history"]) == 1
        assert state["weak_link_clarification_turns"] == 1
        assert payload_2["turn"] == 2

    def test_repeated_resume_of_same_checkpoint_is_idempotent(self):
        """A double-submitted / client-retried resume against the same interrupted checkpoint
        replays the stored resume value: no LLM call, no second Q/A pair, no overwritten answer."""
        llm = make_pitchvision_llm()
        with offline_llm(llm):
            graph = build_graph(checkpointer=MemorySaver())
            payload = _run_first_turn(graph, "pv-fork")
            interrupted_cfg = graph.get_state(_cfg("pv-fork")).config
            calls_before = len(llm.calls)

            outcomes = []
            for answer in ("The coaches.", "The coaches, resent."):
                graph.invoke(Command(resume=answer), interrupted_cfg, interrupt_after=["merge_founder_answer"])
                outcomes.append(graph.get_state(_cfg("pv-fork")).values)

        assert llm.calls[calls_before:] == []
        for state in outcomes:
            assert [m["content"] for m in state["founder_messages"]] == [PITCHVISION_FIRST_MESSAGE, payload["question"], "The coaches."]
            assert state["answered_question_id"] == payload["question_id"]
            assert len(state["question_history"]) == 1
            assert len(state["weak_link_clarification_history"]) == 1

    def test_conversational_repair_reasks_without_composer_and_attributes_answer(self):
        llm = make_pitchvision_llm()
        with offline_llm(llm):
            graph = build_graph(checkpointer=MemorySaver())
            payload = _run_first_turn(graph, "pv-repair")
            graph.invoke(Command(resume="I don't understand"), _cfg("pv-repair"))
            repair_payload = _interrupt_payload(graph, _cfg("pv-repair"))
            graph.invoke(Command(resume=PITCHVISION_ANSWER_1), _cfg("pv-repair"), interrupt_after=["merge_founder_answer"])
            state = graph.get_state(_cfg("pv-repair")).values

        assert llm.count(COMPOSER) == 1          # repair path never re-runs the advisor composer
        assert llm.count("TEXT") == 1            # one rephrasing call by the repair node
        assert repair_payload["question_id"] == payload["question_id"]
        assert repair_payload["question"] != payload["question"]
        assert [m["content"] for m in state["founder_messages"]] == [
            PITCHVISION_FIRST_MESSAGE,
            payload["question"],
            "I don't understand",
            repair_payload["question"],
            PITCHVISION_ANSWER_1,
        ]
        assert state["weak_link_clarification_history"][0]["question"] == repair_payload["question"]
        assert state["answered_question_id"] == payload["question_id"]


def _baseline_path_llm():
    """Nothing extracted -> baseline gate asks a grouped baseline question."""
    return make_pitchvision_llm(
        turn_deltas=[{}],
        BaselineSufficiencyLLMOutput=lambda m, r: {
            "is_sufficient": False, "team_status": "MISSING", "capacity_status": "MISSING",
            "product_stage_status": "MISSING", "validation_status": "MISSING", "resource_status": "MISSING",
            "missing_groups": ["TEAM"], "reason": "Team unknown.",
        },
        BaselineQuestionPlanLLMOutput=lambda m, r: {
            "should_ask": True, "target_group": "TEAM", "target_fields": ["active_team_size"],
            "question": "Who is on the team and how many hours can each person give per week?",
            "reason": "Team is unknown.",
        },
    )


def _strategic_path_llm():
    """Baseline known, but the gap planner needs one strategic clarification."""
    return make_pitchvision_llm(
        InformationGapPlanLLMOutput=lambda m, r: {
            "should_ask": True, "strategic_sufficient": False, "target_domain": "STRATEGIC",
            "target_field_or_area": "Problem", "priority": "CRITICAL_NOW",
            "information_goal": "How coaches review matches today.",
            "reason": "Problem severity is ungrounded.",
            "question": "How do the coaches review matches today, if at all?",
        },
    )


@pytest.mark.parametrize("make_llm, interrupt_type", [
    (_baseline_path_llm, "baseline"),
    (_strategic_path_llm, "strategic"),
    (make_pitchvision_llm, "weak_link"),
], ids=["baseline", "strategic", "weak_link"])
def test_every_question_path_composes_once_and_never_on_resume(make_llm, interrupt_type):
    llm = make_llm()
    with offline_llm(llm):
        graph = build_graph(checkpointer=MemorySaver())
        payload = _run_first_turn(graph, f"pv-path-{interrupt_type}")
        assert payload["interrupt_type"] == interrupt_type
        assert llm.count(COMPOSER) == 1
        calls_before = len(llm.calls)
        graph.invoke(Command(resume="Two founders, 15 hours each."), _cfg(f"pv-path-{interrupt_type}"),
                     interrupt_after=["merge_founder_answer"])
        state = graph.get_state(_cfg(f"pv-path-{interrupt_type}")).values

    assert llm.unscripted == []
    assert llm.calls[calls_before:] == []
    assert state["founder_messages"][1] == {"role": "startary", "content": payload["question"]}
    assert state["answered_question_id"] == payload["question_id"]


# ---------------------------------------------------------------------------
# Node-level contracts
# ---------------------------------------------------------------------------

def _weak_link_question_state(**extra) -> dict:
    state = {
        "project_id": "pv-unit",
        "active_interrupt_type": "weak_link",
        "weak_link_clarification_turns": 0,
        "active_next_turn_decision": {
            "question_id": "q_wl_1_abc123",
            "information_target": "WEAK_LINK_Problem",
            "canonical_question": "Who struggles more today with reviewing matches?",
        },
        "current_weak_link_question": {"question": "Who struggles more today with reviewing matches?", "information_goal": "Whose pain is direct."},
        "founder_messages": [{"role": "founder", "content": PITCHVISION_FIRST_MESSAGE}],
    }
    state.update(extra)
    return state


class TestNodeContracts:
    def test_ask_node_never_calls_the_composer(self):
        """Even with no persisted prompt, the interrupt node is LLM-free (it re-runs on every resume)."""
        with patch("startary.graph.nodes.compose_advisor_response", side_effect=AssertionError("composer called")), \
             patch("startary.graph.nodes.interrupt", return_value="The coaches.") as mock_interrupt:
            res = ask_and_wait_for_founder(_weak_link_question_state())

        assert mock_interrupt.call_args[0][0]["question"] == "Who struggles more today with reviewing matches?"
        assert res["founder_response"] == "The coaches."
        assert res["pending_founder_prompt"]["source"] == "planner_fallback"

    def test_ask_node_displays_persisted_prompt_verbatim(self):
        prompt = {
            "interrupt_type": "weak_link", "question_id": "q_wl_1_abc123",
            "raw_planner_question": "Who struggles more today with reviewing matches?",
            "target_area": "WEAK_LINK_Problem", "information_goal": "", "turn": 1,
            "question": "Thanks. Who struggles more today with reviewing matches?", "source": "advisor_composer",
        }
        with patch("startary.graph.nodes.interrupt", return_value="x") as mock_interrupt:
            res = ask_and_wait_for_founder(_weak_link_question_state(pending_founder_prompt=prompt))
        assert mock_interrupt.call_args[0][0]["question"] == prompt["question"]
        assert "pending_founder_prompt" not in res  # nothing to re-persist

    def test_stale_prompt_for_another_question_is_never_displayed(self):
        stale = {"interrupt_type": "weak_link", "question_id": "q_wl_0_old", "question": "OLD QUESTION", "raw_planner_question": "OLD"}
        with patch("startary.graph.nodes.interrupt", return_value="x") as mock_interrupt:
            ask_and_wait_for_founder(_weak_link_question_state(pending_founder_prompt=stale))
        assert mock_interrupt.call_args[0][0]["question"] != "OLD QUESTION"

    def test_compose_node_calls_composer_once_and_persists_prompt(self):
        composed = {"latest_advisor_response": {"question_id": "q_wl_1_abc123", "response_text": "Composed text?"}, "intro_summary_shown": True}
        with patch("startary.graph.nodes.compose_advisor_response", return_value=composed) as mock_compose:
            res = compose_founder_question(_weak_link_question_state())
        assert mock_compose.call_count == 1
        assert res["pending_founder_prompt"]["question"] == "Composed text?"
        assert res["latest_advisor_response"]["response_text"] == "Composed text?"

    def test_compose_node_reuses_response_already_bound_to_question(self):
        bound = {"question_id": "q_wl_1_abc123", "response_text": "Already composed?"}
        with patch("startary.graph.nodes.compose_advisor_response", side_effect=AssertionError("re-composed")):
            res = compose_founder_question(_weak_link_question_state(latest_advisor_response=bound))
        assert res["pending_founder_prompt"]["question"] == "Already composed?"

    def test_compose_failure_falls_back_to_planner_question(self):
        with patch("startary.graph.nodes.compose_advisor_response", side_effect=RuntimeError("timeout")):
            res = compose_founder_question(_weak_link_question_state())
        assert res["pending_founder_prompt"]["question"] == "Who struggles more today with reviewing matches?"
        assert res["pending_founder_prompt"]["source"] == "planner_fallback"

    def test_merge_records_persisted_prompt_not_stale_advisor_text(self):
        prompt = {
            "interrupt_type": "weak_link", "question_id": "q_wl_1_abc123",
            "raw_planner_question": "Who struggles more today with reviewing matches?",
            "target_area": "WEAK_LINK_Problem", "information_goal": "", "turn": 1,
            "question": "Displayed text?", "source": "advisor_composer",
        }
        state = _weak_link_question_state(
            pending_founder_prompt=prompt,
            latest_advisor_response={"question_id": "q_wl_1_abc123", "response_text": "Regenerated text?"},
            founder_response="The coaches.",
            raw_context=PITCHVISION_FIRST_MESSAGE,
        )
        res = merge_founder_answer(state)
        assert res["founder_messages"][-2] == {"role": "startary", "content": "Displayed text?"}
        assert res["weak_link_clarification_history"][-1]["question"] == "Displayed text?"
        assert res["pending_founder_prompt"] is None


def test_graph_topology_composes_before_interrupt():
    builder = build_graph().builder
    edges = set(builder.edges)
    assert ("compose_founder_question", "ask_and_wait_for_founder") in edges
    assert ("select_weak_link_question", "compose_founder_question") in edges
    assert ("plan_baseline_question", "compose_founder_question") in edges
    assert ("select_weak_link_question", "ask_and_wait_for_founder") not in edges
    gap_branch = builder.branches["plan_information_gap"]
    (branch,) = gap_branch.values()
    assert branch.ends["ask_and_wait_for_founder"] == "compose_founder_question"
