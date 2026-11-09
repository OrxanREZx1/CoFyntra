"""
Milestone 6.2.7 — safe first-turn parallel understanding (STARTARY_PARALLEL_UNDERSTANDING).

Offline only (ScriptedLLM; conftest blocks the network). Covers:
  flag OFF = unchanged sequential graph; flag ON = interpret_founder_turn and extract_startup_state
  run concurrently (proven with controlled delays and thread ids), behind an explicit barrier
  (join_founder_turn) that merges shared keys deterministically before any downstream routing;
  equivalence of ON vs OFF state, prompts and LLM calls; provenance / corrections / negative
  evidence; initial, clarification, repair and evidence routes (no speculative extraction);
  interrupt/resume, duplicate resume, branch failure + checkpoint restore without repeating
  completed work; bounded retries; privacy of the new LangSmith metadata; run-tree overlap;
  independence from STARTARY_EXPERIMENT_REVISION_V2.

The M6.2.3 graph test classes are re-run at the bottom with the flag ON and their assertions
inherited unchanged (one topology-name assertion has an ON counterpart; see that class).
"""

from __future__ import annotations

import re
import threading
import time
from collections import Counter
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

import startary.graph.builder as builder
import startary.graph.nodes as nodes
from startary.config import Settings
from startary.domain.enums import WorkflowPhase
from startary.domain.telemetry import TurnTelemetrySummary
from startary.graph.builder import build_experiment_graph, build_graph, route_after_interpret_founder_turn
from startary.graph.graph_state import merge_understanding_branches
from startary.graph.nodes import founder_turn_route, interpret_founder_turn, join_founder_turn
from startary.observability.langsmith import sanitize_metadata
from tests import test_m623_question_composition_integrity as m623_composition
from tests import test_m623_state_integrity_and_observability as m623_state
from tests.fixtures.pitchvision_scenario import (
    PITCHVISION_ANSWER_1,
    PITCHVISION_FIRST_MESSAGE,
    PITCHVISION_NEVER_STATED,
    make_pitchvision_llm,
)
from tests.fixtures.scripted_llm import ScriptedLLM

INTERPRET = "FounderTurnInterpretationLLMOutput"
EXTRACT = "StartupStateDeltaLLMOutput"
OFFLINE_SETTINGS = Settings(
    _env_file=None,
    openai_api_key="sk-offline-test",
    fast_model_name="scripted-fast",
    reasoning_model_name="scripted-reasoning",
)
PARALLEL_KEYS = ("founder_turn_dispatch", "understanding_branches")
SHARED_KEYS = ("traces", "last_trace", "turn_telemetry")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

@contextmanager
def offline(llm: ScriptedLLM):
    with patch("startary.graph.nodes.get_llm_client", side_effect=llm.get_llm_client), \
         patch("startary.graph.nodes.get_settings", return_value=OFFLINE_SETTINGS):
        yield


def _cfg(thread: str) -> dict:
    return {"configurable": {"thread_id": thread}}


def _initial_state(thread: str) -> dict:
    return {
        "project_id": thread,
        "phase": WorkflowPhase.UNDERSTAND,
        "iteration": 1,
        "raw_context": PITCHVISION_FIRST_MESSAGE,
        "founder_messages": [{"role": "founder", "content": PITCHVISION_FIRST_MESSAGE}],
    }


def _payload(graph, thread: str) -> dict:
    snap = graph.get_state(_cfg(thread))
    assert snap.tasks and snap.tasks[0].interrupts, f"graph is not interrupted (next={snap.next})"
    return snap.tasks[0].interrupts[0].value


def _run(parallel: bool, llm: ScriptedLLM, thread: str, resumes: tuple[str, ...] = (), saver=None):
    """Run the PitchVision first turn (+ optional resumes). Returns (graph, [snapshot per interrupt])."""
    with offline(llm):
        graph = build_graph(checkpointer=saver or MemorySaver(), parallel_understanding=parallel)
        graph.invoke(_initial_state(thread), _cfg(thread))
        snaps = [graph.get_state(_cfg(thread))]
        for answer in resumes:
            graph.invoke(Command(resume=answer), _cfg(thread))
            snaps.append(graph.get_state(_cfg(thread)))
    return graph, snaps


def _delay(llm: ScriptedLLM, schema: str, seconds: float, windows: dict[str, list]) -> None:
    """Wrap a responder: sleep for a controlled time and record (start, end, thread id)."""
    inner = llm.responders[schema]

    def responder(messages, rec):
        start = time.perf_counter()
        time.sleep(seconds)
        out = inner(messages, rec)
        windows.setdefault(schema, []).append((start, time.perf_counter(), threading.get_ident()))
        return out

    llm.responders[schema] = responder


_VOLATILE = {"id", "created_at", "timestamp", "start_timestamp", "end_timestamp", "latency_ms", "duration_ms",
             "start_time", "end_time", "turn_total_latency_ms", "total_duration_ms"}
_QID = re.compile(r"(q_[a-z_]+_\d+)_[0-9a-f]{6}")
_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(\.\d+)?(\+00:00|Z)?")


def _norm(value: Any) -> Any:
    """Drop per-run noise (uuid ids, timestamps, latencies, random question-id suffixes), also
    where they are embedded in prompt text (the composer prompt carries the decision JSON)."""
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    if isinstance(value, dict):
        return {k: _norm(v) for k, v in value.items() if k not in _VOLATILE}
    if isinstance(value, (list, tuple)):
        return [_norm(v) for v in value]
    if isinstance(value, str):
        return _TIMESTAMP.sub("<ts>", _UUID.sub("<uuid>", _QID.sub(r"\1_*", value)))
    return value


def _authoritative(values: dict) -> dict:
    return _norm({k: v for k, v in values.items() if k not in PARALLEL_KEYS + SHARED_KEYS})


def _telemetry_nodes(values: dict) -> list[str]:
    return [r.node_name for r in values["turn_telemetry"].records]


def _join_meta(values: dict) -> list[dict]:
    return [r.details for r in values["turn_telemetry"].records if r.node_name == "join_founder_turn"]


class _RunTimeline(BaseCallbackHandler):
    """Records every chain run's (name, parent, start, end) — the same run tree LangSmith renders."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.runs: dict[Any, dict[str, Any]] = {}

    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, **kwargs):
        with self.lock:
            self.runs[run_id] = {"name": kwargs.get("name"), "parent": parent_run_id, "start": time.perf_counter(), "end": None}

    def on_chain_end(self, outputs, *, run_id, **kwargs):
        with self.lock:
            if run_id in self.runs:
                self.runs[run_id]["end"] = time.perf_counter()

    def named(self, name: str) -> list[dict[str, Any]]:
        return [r for r in self.runs.values() if r["name"] == name]


# ---------------------------------------------------------------------------
# 1–2: feature flag
# ---------------------------------------------------------------------------

class TestFeatureFlag:
    def test_flag_defaults_false_and_only_explicit_true_enables(self, monkeypatch):
        assert Settings(_env_file=None).parallel_understanding is False
        monkeypatch.setenv("STARTARY_PARALLEL_UNDERSTANDING", "true")
        assert Settings(_env_file=None).parallel_understanding is True
        # MagicMock settings (common in tests) never enable it by accident.
        with patch("startary.graph.builder.get_settings", return_value=MagicMock()):
            assert builder.parallel_understanding_enabled() is False
        with patch("startary.graph.builder.get_settings", return_value=Settings(_env_file=None, parallel_understanding=True)):
            assert builder.parallel_understanding_enabled() is True

    def test_flag_off_builds_the_unchanged_sequential_topology(self):
        graph = build_graph()  # conftest pins the flag false
        g = graph.get_graph()
        assert "dispatch_founder_turn" not in g.nodes and "join_founder_turn" not in g.nodes
        edges = {(e.source, e.target) for e in g.edges}
        assert {("verify_state", "interpret_founder_turn"), ("merge_founder_answer", "interpret_founder_turn"),
                ("wait_for_evidence", "interpret_founder_turn"), ("extract_startup_state", "merge_execution_state"),
                ("interpret_founder_turn", "extract_startup_state"), ("interpret_founder_turn", "handle_conversational_repair"),
                ("interpret_founder_turn", "classify_waiting_for_evidence_turn")} <= edges
        # The registered callables are the original node functions, not the wrappers.
        assert graph.builder.nodes["interpret_founder_turn"].runnable.func is nodes.interpret_founder_turn
        assert graph.builder.nodes["extract_startup_state"].runnable.func is nodes.extract_startup_state
        assert {(e.source, e.target) for e in build_graph(parallel_understanding=False).get_graph().edges} == edges

    def test_flag_on_inserts_dispatch_and_barrier_only_around_understanding(self):
        off = {(e.source, e.target) for e in build_graph(parallel_understanding=False).get_graph().edges}
        on = {(e.source, e.target) for e in build_graph(parallel_understanding=True).get_graph().edges}
        assert off - on == {("verify_state", "interpret_founder_turn"), ("merge_founder_answer", "interpret_founder_turn"),
                            ("wait_for_evidence", "interpret_founder_turn"), ("extract_startup_state", "merge_execution_state")}
        assert on - off == {
            ("verify_state", "dispatch_founder_turn"), ("merge_founder_answer", "dispatch_founder_turn"),
            ("wait_for_evidence", "dispatch_founder_turn"), ("dispatch_founder_turn", "interpret_founder_turn"),
            ("dispatch_founder_turn", "extract_startup_state"), ("interpret_founder_turn", "join_founder_turn"),
            ("extract_startup_state", "join_founder_turn"), ("join_founder_turn", "merge_execution_state"),
            ("join_founder_turn", "handle_conversational_repair"),
        }
        # Node names are unchanged (CLI error mapping and telemetry key off them).
        graph = build_graph(parallel_understanding=True)
        assert graph.builder.nodes["interpret_founder_turn"].runnable.func is nodes.parallel_interpret_founder_turn


# ---------------------------------------------------------------------------
# 3–5: sequential when off, concurrent when on, barrier before routing
# ---------------------------------------------------------------------------

class TestConcurrency:
    def test_flag_off_runs_interpretation_strictly_before_extraction(self):
        llm, windows = make_pitchvision_llm(), {}
        _delay(llm, INTERPRET, 0.2, windows)
        _delay(llm, EXTRACT, 0.2, windows)
        _, (snap,) = _run(False, llm, "off-seq")
        (i_start, i_end, _), (e_start, _, _) = windows[INTERPRET][0], windows[EXTRACT][0]
        assert e_start >= i_end
        assert "founder_turn_dispatch" not in snap.values
        assert "join_founder_turn" not in _telemetry_nodes(snap.values)

    def test_flag_on_runs_both_branches_concurrently(self):
        delay = 0.5
        llm, windows = make_pitchvision_llm(), {}
        _delay(llm, INTERPRET, delay, windows)
        _delay(llm, EXTRACT, delay, windows)
        _, (snap,) = _run(True, llm, "on-overlap")
        (i_start, i_end, i_thread), (e_start, e_end, e_thread) = windows[INTERPRET][0], windows[EXTRACT][0]
        overlap = min(i_end, e_end) - max(i_start, e_start)
        assert i_thread != e_thread
        assert overlap >= 0.6 * delay
        (meta,) = _join_meta(snap.values)
        assert meta["branch_overlap_ms"] >= 0.6 * delay * 1000
        # Region wall clock is close to one branch, well under the sum of both.
        assert meta["parallel_region_ms"] < meta["sum_of_branches_ms"] - 0.3 * delay * 1000
        assert llm.count(INTERPRET) == 1 and llm.count(EXTRACT) == 1

    def test_barrier_waits_for_the_slower_branch_before_routing(self):
        llm, windows = make_pitchvision_llm(), {}
        _delay(llm, INTERPRET, 0.6, windows)   # interpretation is the slow branch
        _delay(llm, EXTRACT, 0.05, windows)
        steps: dict[str, list[int]] = {}
        with offline(llm):
            graph = build_graph(checkpointer=MemorySaver(), parallel_understanding=True)
            for event in graph.stream(_initial_state("on-barrier"), _cfg("on-barrier"), stream_mode="debug"):
                if event["type"] == "task_result":
                    steps.setdefault(event["payload"]["name"], []).append(event["step"])
            values = graph.get_state(_cfg("on-barrier")).values
        assert steps["interpret_founder_turn"] == steps["extract_startup_state"]       # same superstep
        assert steps["join_founder_turn"] == [steps["interpret_founder_turn"][0] + 1]  # exactly once, after both
        assert steps["merge_execution_state"] == [steps["join_founder_turn"][0] + 1]  # routing only after the barrier
        (meta,) = _join_meta(values)
        assert meta["sync_wait_ms"] >= 400 and meta["sync_status"] == "SUCCESS"
        assert meta["next_route"] == "merge_execution_state"


# ---------------------------------------------------------------------------
# 6–9: deterministic merge; ON == OFF; provenance, corrections, negative evidence
# ---------------------------------------------------------------------------

class TestDeterministicMerge:
    def test_on_matches_off_state_prompts_and_calls_over_two_turns(self):
        off_llm, on_llm = make_pitchvision_llm(), make_pitchvision_llm()
        off_graph, off_snaps = _run(False, off_llm, "eq", resumes=(PITCHVISION_ANSWER_1,))
        on_graph, on_snaps = _run(True, on_llm, "eq", resumes=(PITCHVISION_ANSWER_1,))

        for off, on in zip(off_snaps, on_snaps):
            assert on.next == off.next == ("ask_and_wait_for_founder",)
            assert _norm(on.tasks[0].interrupts[0].value) == _norm(off.tasks[0].interrupts[0].value)
            assert _authoritative(on.values) == _authoritative(off.values)
        # Same calls, same prompts, same tiers (only the interleaving of the two parallel calls differs).
        assert Counter(c.schema for c in on_llm.calls) == Counter(c.schema for c in off_llm.calls)
        for schema in {c.schema for c in off_llm.calls}:
            assert [_norm(c.messages) for c in on_llm.calls_for(schema)] == [_norm(c.messages) for c in off_llm.calls_for(schema)], schema
            assert [c.tier for c in on_llm.calls_for(schema)] == [c.tier for c in off_llm.calls_for(schema)], schema

    def test_shared_keys_merge_in_sequential_order_without_loss_or_duplicates(self):
        off_llm, on_llm = make_pitchvision_llm(), make_pitchvision_llm()
        _, (off,) = _run(False, off_llm, "shared")
        _, (on,) = _run(True, on_llm, "shared")
        # traces: same entries, same order (interpretation's before extraction's), nothing duplicated.
        assert [_norm(t) for t in on.values["traces"]] == [_norm(t) for t in off.values["traces"]]
        # telemetry: identical records in identical order, plus the barrier's own deterministic record.
        off_nodes, on_nodes = _telemetry_nodes(off.values), _telemetry_nodes(on.values)
        assert on_nodes == off_nodes[:2] + ["join_founder_turn"] + off_nodes[2:]
        s_off, s_on = off.values["turn_telemetry"], on.values["turn_telemetry"]
        assert isinstance(s_on, TurnTelemetrySummary)
        assert (s_on.llm_call_count, s_on.total_tokens, s_on.execution_path) == (s_off.llm_call_count, s_off.total_tokens, s_off.execution_path)
        assert _norm(on.values["last_trace"]) == _norm(off.values["last_trace"])
        # The branch scratch area is cleared by the barrier; the transcript has no duplicates.
        assert on.values["understanding_branches"] == {}
        assert on.values["founder_messages"] == off.values["founder_messages"]

    def test_founder_fact_provenance_is_preserved(self):
        llm = make_pitchvision_llm()
        _, (snap,) = _run(True, llm, "provenance")
        startup_state = snap.values["startup_state"]
        for layer, field, value, provenance in m623_state.PITCHVISION_REQUIRED_FACTS:
            fact = startup_state[layer][field]
            assert (fact["value"], fact["provenance"]) == (value, provenance), field

    def test_negative_evidence_and_unknowns_are_preserved(self):
        llm = make_pitchvision_llm()
        _, (snap,) = _run(True, llm, "negative")
        startup_state = snap.values["startup_state"]
        for layer, field, value in (("progress", "paying_customers", "0"), ("progress", "revenue", "0"),
                                    ("progress", "mvp_exists", "false"), ("progress", "interviews_completed", "0")):
            assert (startup_state[layer][field]["value"], startup_state[layer][field]["provenance"]) == (value, "EXPLICIT"), field
        for layer, fields in PITCHVISION_NEVER_STATED.items():
            for field in fields:
                assert startup_state[layer][field]["provenance"] == "UNKNOWN" and startup_state[layer][field]["value"] is None

    def test_correction_and_goal_from_interpretation_survive_alongside_extraction(self):
        def correcting(messages, rec):
            return {
                "intent_types": ["ANSWER", "CORRECTION"], "direct_answer": True, "direct_question": None,
                "concern_or_goal": "Worried about wasting the budget", "correction": "Budget is for the whole year",
                "cannot_answer_now": False, "cannot_act_now": False, "uncertainty_statement": None,
                "new_evidence_summary": None, "founder_preference_or_constraint": None,
                "detected_founder_goal": "VALIDATE_PROBLEM", "response_priority": "HIGH",
                "interpretation_summary": "Founder corrected the budget horizon.",
            }

        results = {}
        for parallel in (False, True):
            llm = make_pitchvision_llm(FounderTurnInterpretationLLMOutput=correcting)
            _, (snap,) = _run(parallel, llm, "correction")
            results[parallel] = snap.values
        on = results[True]
        assert on["recent_corrections"] == ["Budget is for the whole year"]
        assert on["founder_current_concern"] == "Worried about wasting the budget"
        assert on["founder_current_goal"]["goal_type"] == "VALIDATE_PROBLEM"
        assert on["startup_state"]["execution_context"]["available_budget_amount"]["value"] == "1500"
        assert _authoritative(on) == _authoritative(results[False])


# ---------------------------------------------------------------------------
# 10–13: routing and readiness gating (no speculative extraction)
# ---------------------------------------------------------------------------

_ROUTE_CASES = [
    ("plain intake", WorkflowPhase.UNDERSTAND, "We sell match analysis to amateur coaches.", {}),
    ("repair phrase", WorkflowPhase.UNDERSTAND, "What do you mean?", {}),
    ("repair starter", WorkflowPhase.UNDERSTAND, "Can you explain why that matters", {}),
    ("third-party question is not repair", WorkflowPhase.UNDERSTAND, "What do you mean by customer segment", {}),
    ("repair flag already set", WorkflowPhase.UNDERSTAND, "The coaches.", {"is_conversational_repair": True}),
    ("repair execution path", WorkflowPhase.UNDERSTAND, "The coaches.", {"execution_path": "CONVERSATIONAL_REPAIR"}),
    ("evidence phase", WorkflowPhase.WAITING_FOR_EVIDENCE, "We interviewed six coaches.", {}),
    ("evidence phase string", "WAITING_FOR_EVIDENCE", "We interviewed six coaches.", {}),
    ("evidence phase repair", WorkflowPhase.WAITING_FOR_EVIDENCE, "I don't understand", {}),
    ("no founder message", WorkflowPhase.UNDERSTAND, None, {}),
]


class TestRoutingAndGating:
    @pytest.mark.parametrize("label,phase,text,extra", _ROUTE_CASES, ids=[c[0] for c in _ROUTE_CASES])
    def test_dispatch_prediction_equals_post_interpretation_route(self, label, phase, text, extra):
        messages = [{"role": "founder", "content": text}] if text is not None else [{"role": "startary", "content": "Hi"}]
        state = {"project_id": "route", "iteration": 1, "phase": phase, "founder_messages": messages, **extra}
        llm = make_pitchvision_llm()
        with offline(llm):
            interpreted = {**state, **interpret_founder_turn(dict(state))}
        assert founder_turn_route(state) == route_after_interpret_founder_turn(interpreted), label

    def test_initial_and_clarification_turns_dispatch_parallel(self):
        llm = make_pitchvision_llm()
        _, snaps = _run(True, llm, "dispatch", resumes=(PITCHVISION_ANSWER_1,))
        for snap, turns in zip(snaps, (1, 2)):
            dispatch = snap.values["founder_turn_dispatch"]
            assert (dispatch["mode"], dispatch["reason"], dispatch["founder_turn_count"]) == ("PARALLEL", "UNDERSTAND_TURN", turns)
            assert len(_join_meta(snap.values)) == turns
        assert llm.count(INTERPRET) == 2 and llm.count(EXTRACT) == 2

    def test_repair_turn_stays_sequential_and_never_extracts(self):
        results = {}
        for parallel in (False, True):
            llm = make_pitchvision_llm()
            graph, (_, repair_snap) = _run(parallel, llm, "repair", resumes=("I don't understand",))
            results[parallel] = (llm, repair_snap)
        (off_llm, off), (on_llm, on) = results[False], results[True]
        assert on_llm.count(EXTRACT) == off_llm.count(EXTRACT) == 1          # only the first turn extracted
        assert Counter(c.schema for c in on_llm.calls) == Counter(c.schema for c in off_llm.calls)
        assert on.values["founder_turn_dispatch"]["mode"] == "SEQUENTIAL"
        assert on.values["founder_turn_dispatch"]["reason"] == "CONVERSATIONAL_REPAIR"
        assert len(_join_meta(on.values)) == 1                                 # no barrier on the repair turn
        assert _norm(on.tasks[0].interrupts[0].value) == _norm(off.tasks[0].interrupts[0].value)
        assert _authoritative(on.values) == _authoritative(off.values)

    def test_evidence_turn_stays_sequential_and_never_extracts(self):
        state = {
            "project_id": "evidence", "iteration": 1, "phase": WorkflowPhase.WAITING_FOR_EVIDENCE,
            "raw_context": PITCHVISION_FIRST_MESSAGE,
            "founder_messages": [{"role": "founder", "content": "We interviewed six coaches and four would pay."}],
        }
        outcomes = {}
        for parallel in (False, True):
            llm = make_pitchvision_llm()
            with offline(llm):
                graph = build_graph(checkpointer=MemorySaver(), parallel_understanding=parallel)
                graph.update_state(_cfg("evidence"), state, as_node="wait_for_evidence")
                graph.invoke(None, _cfg("evidence"), interrupt_before=["classify_waiting_for_evidence_turn"])
                outcomes[parallel] = (llm, graph.get_state(_cfg("evidence")))
        (off_llm, off), (on_llm, on) = outcomes[False], outcomes[True]
        assert on.next == off.next == ("classify_waiting_for_evidence_turn",)
        assert on_llm.count(EXTRACT) == off_llm.count(EXTRACT) == 0
        assert [c.schema for c in on_llm.calls] == [c.schema for c in off_llm.calls] == [INTERPRET]
        assert on.values["founder_turn_dispatch"]["reason"] == "WAITING_FOR_EVIDENCE"
        assert _norm(on.values["latest_founder_turn_interpretation"]) == _norm(off.values["latest_founder_turn_interpretation"])


# ---------------------------------------------------------------------------
# 14–17: interrupts, duplicate resume, failure + checkpoint restore, bounded retries
# ---------------------------------------------------------------------------

class TestInterruptsCheckpointsAndFailures:
    def test_resume_reaches_dispatch_with_zero_llm_calls_and_no_regeneration(self):
        llm = make_pitchvision_llm()
        with offline(llm):
            graph = build_graph(checkpointer=MemorySaver(), parallel_understanding=True)
            graph.invoke(_initial_state("resume"), _cfg("resume"))
            calls_before = len(llm.calls)
            graph.invoke(Command(resume=PITCHVISION_ANSWER_1), _cfg("resume"), interrupt_after=["merge_founder_answer"])
            snap = graph.get_state(_cfg("resume"))
            assert snap.next == ("dispatch_founder_turn",)
            assert llm.calls[calls_before:] == []
            graph.invoke(None, _cfg("resume"))
        assert llm.count("AdvisorResponseLLMOutput") == 2   # first question + next question, never the answered one
        assert llm.count(INTERPRET) == 2 and llm.count(EXTRACT) == 2

    def test_duplicate_resume_of_the_same_checkpoint_is_idempotent(self):
        llm = make_pitchvision_llm()
        with offline(llm):
            graph = build_graph(checkpointer=MemorySaver(), parallel_understanding=True)
            graph.invoke(_initial_state("dup"), _cfg("dup"))
            payload = _payload(graph, "dup")
            interrupted_cfg = graph.get_state(_cfg("dup")).config
            outcomes = []
            for answer in ("The coaches.", "The coaches, resent."):
                graph.invoke(Command(resume=answer), interrupted_cfg, interrupt_after=["join_founder_turn"])
                outcomes.append(graph.get_state(_cfg("dup")).values)
        # Each resume forks the same checkpoint; the stored resume value is replayed, so every fork
        # understands the same single answer exactly once.
        for state in outcomes:
            assert [m["content"] for m in state["founder_messages"]] == [PITCHVISION_FIRST_MESSAGE, payload["question"], "The coaches."]
            assert len(_join_meta(state)) == 2
            assert len(state["question_history"]) == 1
        assert llm.count(INTERPRET) == 3 and llm.count(EXTRACT) == 3   # turn 1 + one per fork, no duplicates within a fork

    @pytest.mark.parametrize("failing", ["interpret_founder_turn", "extract_startup_state"])
    def test_branch_failure_and_checkpoint_restore_rerun_only_the_failed_branch(self, failing):
        """One branch crashes while the other completes: the completed branch's writes are kept in the
        checkpoint, a freshly built graph (simulated restart) resumes, and only the failed branch re-runs."""
        llm, windows = make_pitchvision_llm(), {}
        _delay(llm, INTERPRET, 0.3, windows)
        _delay(llm, EXTRACT, 0.3, windows)
        original = getattr(nodes, failing)
        crashes = {"n": 0}

        def crash_once(state):
            if crashes["n"] == 0:
                crashes["n"] += 1
                raise RuntimeError(f"simulated {failing} crash")
            return original(state)

        saver = MemorySaver()
        with offline(llm), patch.object(nodes, failing, crash_once):
            graph = build_graph(checkpointer=saver, parallel_understanding=True)
            with pytest.raises(RuntimeError, match="simulated"):
                graph.invoke(_initial_state("crash"), _cfg("crash"))
            failed = graph.get_state(_cfg("crash"))
            assert failing in failed.next and "join_founder_turn" not in failed.next   # barrier never ran
            counts_after_crash = (llm.count(INTERPRET), llm.count(EXTRACT))

            restored = build_graph(checkpointer=saver, parallel_understanding=True)
            restored.invoke(None, _cfg("crash"))
            values = restored.get_state(_cfg("crash")).values

        survivor = EXTRACT if failing == "interpret_founder_turn" else INTERPRET
        # (interpretation calls, extraction calls) when the crash surfaced: the other branch had completed.
        assert counts_after_crash == ((0, 1) if failing == "interpret_founder_turn" else (1, 0))
        assert llm.count(survivor) == 1                         # completed work was not repeated
        assert llm.count(INTERPRET) == 1 and llm.count(EXTRACT) == 1
        assert len(_join_meta(values)) == 1
        reference_llm = make_pitchvision_llm()
        _, (reference,) = _run(True, reference_llm, "crash")
        assert _authoritative(values) == _authoritative(reference.values)

    def test_interpretation_retry_is_bounded_and_reported(self):
        attempts = {"n": 0}
        base = make_pitchvision_llm().responders[INTERPRET]

        def flaky(messages, rec):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise RuntimeError("transient provider error")
            return base(messages, rec)

        llm = make_pitchvision_llm(FounderTurnInterpretationLLMOutput=flaky)
        _, (snap,) = _run(True, llm, "retry")
        (meta,) = _join_meta(snap.values)
        assert llm.count(INTERPRET) == 2
        assert (meta["interpretation_retry_count"], meta["interpretation_status"], meta["extraction_retry_count"]) == (1, "SUCCESS", 0)

    def test_interpretation_exhaustion_uses_the_existing_fallback_visibly_and_identically(self):
        def outage(messages, rec):
            raise RuntimeError("provider outage")

        results = {}
        for parallel in (False, True):
            llm = make_pitchvision_llm(FounderTurnInterpretationLLMOutput=outage)
            _, (snap,) = _run(parallel, llm, "outage")
            results[parallel] = (llm, snap)
        (off_llm, off), (on_llm, on) = results[False], results[True]
        assert on_llm.count(INTERPRET) == off_llm.count(INTERPRET) == 2      # bounded: 2 attempts, no extra
        (meta,) = _join_meta(on.values)
        assert meta["interpretation_status"] == "FALLBACK" and meta["interpretation_retry_count"] == 1
        assert _authoritative(on.values) == _authoritative(off.values)

    def test_extraction_failure_propagates_exactly_as_in_sequential_mode(self):
        def outage(messages, rec):
            raise RuntimeError("provider outage")

        outcomes = {}
        for parallel in (False, True):
            llm = make_pitchvision_llm(StartupStateDeltaLLMOutput=outage)
            with offline(llm):
                graph = build_graph(checkpointer=MemorySaver(), parallel_understanding=parallel)
                try:
                    graph.invoke(_initial_state("extract-fail"), _cfg("extract-fail"))
                    error = None
                except Exception as exc:  # compare whatever the sequential graph does
                    error = (type(exc), str(exc))
                outcomes[parallel] = (llm, graph.get_state(_cfg("extract-fail")), error)
        (off_llm, off, off_err), (on_llm, on, on_err) = outcomes[False], outcomes[True]
        assert on_err == off_err
        assert on.next == off.next
        assert on.values.get("extraction_status") == off.values.get("extraction_status") == "FAILED"
        assert Counter(c.schema for c in on_llm.calls) == Counter(c.schema for c in off_llm.calls)
        assert _authoritative(on.values) == _authoritative(off.values)
        if _join_meta(on.values):
            assert _join_meta(on.values)[0]["extraction_status"] == "FAILED"


# ---------------------------------------------------------------------------
# barrier / reducer contracts (unit level)
# ---------------------------------------------------------------------------

class TestBarrierContracts:
    def test_reducer_merges_branches_and_reset_replaces(self):
        assert merge_understanding_branches(None, {"__reset__": True}) == {}
        merged = merge_understanding_branches(merge_understanding_branches({}, {"interpret": {"a": 1}}), {"extract": {"b": 2}})
        assert merged == {"interpret": {"a": 1}, "extract": {"b": 2}}
        assert merge_understanding_branches(merged, {"__reset__": True}) == {}
        assert merge_understanding_branches(merged, None) == merged

    def test_join_fails_loudly_when_a_branch_result_is_missing(self):
        state = {"founder_turn_dispatch": {"mode": "PARALLEL", "dispatched_at": time.time()},
                 "understanding_branches": {"interpret": {"started_at": 0, "ended_at": 0, "new_traces": [], "last_trace": None, "telemetry_records": []}}}
        with pytest.raises(RuntimeError, match=r"branch result missing: \['extract'\]"):
            join_founder_turn(state)

    def test_join_is_a_no_op_outside_the_parallel_region(self):
        assert join_founder_turn({"founder_turn_dispatch": {"mode": "SEQUENTIAL"}}) == {}
        assert join_founder_turn({}) == {}

    def test_join_discards_extraction_on_route_mismatch(self):
        branch = {"started_at": 1.0, "ended_at": 2.0, "new_traces": [], "last_trace": None, "telemetry_records": []}
        state = {"founder_turn_dispatch": {"mode": "PARALLEL", "dispatched_at": 1.0}, "is_conversational_repair": True,
                 "understanding_branches": {"interpret": branch, "extract": dict(branch)}}
        with patch("startary.graph.nodes.emit_decision_span") as span:
            out = join_founder_turn(state)
        assert out["startup_state_delta"] == {}
        meta = span.call_args[0][1]
        assert (meta["sync_status"], meta["next_route"]) == ("ROUTE_MISMATCH", "handle_conversational_repair")
        assert builder.route_after_join_founder_turn(state) == "handle_conversational_repair"


# ---------------------------------------------------------------------------
# 18–19: LangSmith metadata privacy and run-tree overlap
# ---------------------------------------------------------------------------

class TestObservability:
    REQUIRED = {"parallel_understanding", "dispatch_mode", "interpretation_ms", "extraction_ms", "parallel_region_ms",
                "branch_overlap_ms", "sync_wait_ms", "sum_of_branches_ms", "interpretation_retry_count",
                "extraction_retry_count", "interpretation_status", "extraction_status", "sync_status", "next_route",
                "fallback_to_sequential"}

    def test_span_metadata_is_complete_and_private(self):
        spans: list[tuple[str, dict]] = []

        def capture(name, metadata, tags=None):
            spans.append((name, dict(metadata)))
            return sanitize_metadata(metadata)

        llm = make_pitchvision_llm()
        with patch("startary.graph.nodes.emit_decision_span", side_effect=capture):
            _run(True, llm, "privacy", resumes=("I don't understand",))
        by_name: dict[str, list[dict]] = {}
        for name, meta in spans:
            by_name.setdefault(name, []).append(meta)
        dispatches, regions = by_name["understanding_dispatch"], by_name["understanding_parallel_region"]
        assert [d["dispatch_mode"] for d in dispatches] == ["PARALLEL", "SEQUENTIAL"]
        assert dispatches[1]["fallback_to_sequential"] is True and dispatches[1]["dispatch_reason"] == "CONVERSATIONAL_REPAIR"
        (region,) = regions
        assert self.REQUIRED <= set(region)
        for meta in dispatches + regions:
            assert sanitize_metadata(meta) == meta                    # nothing is stripped: no sensitive key names
            for value in meta.values():
                assert value is None or isinstance(value, (bool, int, float)) or re.fullmatch(r"[A-Z_a-z]+", value), value
        serialized = repr(spans).lower()
        for phrase in ("pitchvision", "amateur football", "coaches", "1,500", "don't understand"):
            assert phrase not in serialized

    def test_root_run_carries_the_understanding_mode_for_ab_filtering(self):
        from startary.observability.langsmith import build_trace_config

        for flag, tag in ((True, "understanding:parallel"), (False, "understanding:sequential")):
            settings = Settings(_env_file=None, parallel_understanding=flag, langsmith_tracing=False)
            config = build_trace_config("proj-ab-0001", turn_number=1, settings=settings)
            assert config["metadata"]["parallel_understanding"] is flag
            assert tag in config["tags"]
            assert sanitize_metadata(config["metadata"]) == config["metadata"]

    def test_run_tree_shows_overlapping_understanding_runs_under_one_parent(self):
        delay = 0.4
        llm, windows = make_pitchvision_llm(), {}
        _delay(llm, INTERPRET, delay, windows)
        _delay(llm, EXTRACT, delay, windows)
        timeline = _RunTimeline()
        with offline(llm):
            graph = build_graph(checkpointer=MemorySaver(), parallel_understanding=True)
            graph.invoke(_initial_state("waterfall"), {**_cfg("waterfall"), "callbacks": [timeline]})
        (interp,), (extract,) = timeline.named("interpret_founder_turn"), timeline.named("extract_startup_state")
        (join,) = timeline.named("join_founder_turn")
        assert interp["parent"] == extract["parent"] == join["parent"]
        assert min(interp["end"], extract["end"]) - max(interp["start"], extract["start"]) >= 0.6 * delay
        assert join["start"] >= max(interp["end"], extract["end"])
        # The decision span nests under the barrier node run.
        (region,) = timeline.named("understanding_parallel_region")
        assert timeline.runs[region["parent"]]["name"] == "join_founder_turn"


# ---------------------------------------------------------------------------
# 20: independence from the experiment revision V2 flag
# ---------------------------------------------------------------------------

class TestIndependence:
    def test_parallel_flag_is_independent_of_experiment_revision_v2(self, monkeypatch):
        monkeypatch.setenv("STARTARY_PARALLEL_UNDERSTANDING", "true")
        monkeypatch.setenv("STARTARY_EXPERIMENT_REVISION_V2", "false")
        s = Settings(_env_file=None)
        assert (s.parallel_understanding, s.experiment_revision_v2) == (True, False)
        monkeypatch.setenv("STARTARY_PARALLEL_UNDERSTANDING", "false")
        monkeypatch.setenv("STARTARY_EXPERIMENT_REVISION_V2", "true")
        s = Settings(_env_file=None)
        assert (s.parallel_understanding, s.experiment_revision_v2) == (False, True)

        def topology(g):
            return sorted(g.get_graph().nodes), sorted((e.source, e.target, e.conditional) for e in g.get_graph().edges)

        # The experiment subgraph has no understanding nodes and is built identically either way.
        with patch("startary.graph.builder.get_settings", return_value=Settings(_env_file=None, parallel_understanding=True)):
            exp_on = topology(build_experiment_graph())
        assert exp_on == topology(build_experiment_graph())
        # The experiment region of the main graph is untouched by the parallel flag.
        off_nodes, off_edges = topology(build_graph(parallel_understanding=False))
        on_nodes, on_edges = topology(build_graph(parallel_understanding=True))
        assert set(on_nodes) - set(off_nodes) == {"dispatch_founder_turn", "join_founder_turn"}
        exp_edges = lambda edges: {e for e in edges if "experiment" in e[0] or "experiment" in e[1]}  # noqa: E731
        assert exp_edges(on_edges) == exp_edges(off_edges)


# ---------------------------------------------------------------------------
# Existing M6.2.3 graph tests re-run with the flag ON (assertions inherited unchanged)
# ---------------------------------------------------------------------------

class _ParallelOn:
    @pytest.fixture(autouse=True)
    def _parallel_understanding_on(self, monkeypatch):
        built: list[bool] = []
        monkeypatch.setattr(builder, "parallel_understanding_enabled", lambda: built.append(True) or True)
        yield
        assert built, "test did not build the graph through build_graph()"


class TestM623CompositionUnderParallel(_ParallelOn, m623_composition.TestSingleCompositionAndExactTranscript):
    def test_resume_makes_zero_question_regeneration_calls(self):
        """ON counterpart of the M6.2.3 test: its only topology-specific assertion is the name of the
        node after merge_founder_answer, which is dispatch_founder_turn in the parallel graph. The
        behavioural assertions (no LLM call of any kind on resume, one composer call) are identical."""
        llm = make_pitchvision_llm()
        with m623_composition.offline_llm(llm):
            graph = build_graph(checkpointer=MemorySaver())
            m623_composition._run_first_turn(graph, "pv-resume")
            calls_before = len(llm.calls)
            graph.invoke(Command(resume=PITCHVISION_ANSWER_1), _cfg("pv-resume"), interrupt_after=["merge_founder_answer"])
            snap = graph.get_state(_cfg("pv-resume"))

        assert snap.next == ("dispatch_founder_turn",)
        assert llm.calls[calls_before:] == []
        assert llm.count("AdvisorResponseLLMOutput") == 1


class TestM623CheckpointResumeUnderParallel(_ParallelOn, m623_composition.TestCheckpointResumeSafety):
    pass


class TestM623FactPropagationUnderParallel(_ParallelOn, m623_state.TestPitchVisionFactPropagation):
    pass


class TestM623ObservabilityUnderParallel(_ParallelOn, m623_state.TestObservabilityCoverage):
    @pytest.fixture(autouse=True)
    def _parallel_understanding_on(self, monkeypatch):
        # Some tests in this class call nodes directly (no graph build), so the build assertion is skipped.
        monkeypatch.setattr(builder, "parallel_understanding_enabled", lambda: True)
        yield
