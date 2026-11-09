"""
Milestone 6.2.5 — Experiment loop correctness, persistence, and observability.

Graph-level tests run the REAL compiled graph and MemorySaver checkpointer. The state is seeded as
if decide_next_strategic_action had just run (graph.update_state(..., as_node=...)), so the real
generate → validate → critique → route → save → can_execute → handoff / failure-stop path executes.
Only get_llm_client is replaced by the offline ScriptedLLM.
"""

from __future__ import annotations

import json
import typing
from contextlib import contextmanager
from unittest.mock import patch

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

import startary.graph.nodes as nodes
from startary.config import Settings
from startary.domain.enums import ExperimentDesignStatus
from startary.domain.execution_context import ExecutionContext
from startary.domain.experiment_contract import (
    CRITIC_DIMENSIONS,
    BudgetStatus,
    ContractCode,
    PrototypeStatus,
    assess_prototype_requirement,
    check_privacy_consent,
    compare_cost_to_budget,
    evaluate_experiment_contract,
    experiment_critique_route,
    interpret_critic_output,
    parse_money,
)
from startary.domain.startup_progress import StartupProgress
from startary.domain.terminal_state import TerminalStatus, can_render_waiting_for_evidence, resolve_founder_facing_status
from startary.graph.builder import build_graph, route_on_experiment_critique
from startary.graph.graph_state import StartaryGraphState
from startary.graph.nodes import (
    assess_can_execute,
    compose_execution_handoff,
    critique_experiment,
    generate_experiment,
    handle_experiment_failure_stop,
    merge_execution_state,
    save_final_experiment,
    validate_proposed_experiment,
)
from startary.llm.provider import LLMFailureKind, TokenBudgetExhaustedError, classify_llm_failure
from startary.llm.prompts import EXPERIMENT_CRITIC_SYSTEM_PROMPT, EXPERIMENT_GENERATOR_SYSTEM_PROMPT
from startary.llm.schemas import ExperimentCriticLLMOutput
from startary.observability.langsmith import sanitize_metadata
from tests.fixtures.pitchvision_scenario import PITCHVISION_DELTA, PITCHVISION_FIRST_MESSAGE
from tests.fixtures.scenarios import (
    make_valid_experiment_critic_output_dict,
    make_valid_experiment_generator_output_dict,
)
from tests.fixtures.scripted_llm import ScriptedLLM

OFFLINE_SETTINGS = Settings(_env_file=None, openai_api_key="sk-offline-test", fast_model_name="scripted-fast", reasoning_model_name="scripted-reasoning")
GEN, CRIT = "ExperimentGeneratorLLMOutput", "ExperimentCriticLLMOutput"
ASSUMPTION = {"id": "asmp-pv-1", "statement": "Amateur coaches lose hours to post-match review and want faster insight.", "area": "Problem"}
SENTINEL = "zq-sentinel-founder-phrase"


def _fact(value):
    return {"value": value, "provenance": "EXPLICIT", "confidence": "HIGH", "source_turn": 0, "notes": None}


def pitchvision_fact_stores() -> tuple[dict, dict]:
    """PitchVision facts exactly as merge_execution_state stores them (M6.2.3 regression scenario)."""
    out = merge_execution_state({
        "execution_context": ExecutionContext().model_dump(),
        "startup_progress": StartupProgress().model_dump(),
        "startup_state_delta": PITCHVISION_DELTA,
        "founder_messages": [{"role": "founder", "content": PITCHVISION_FIRST_MESSAGE}],
    })
    return out["execution_context"], out["startup_progress"]


def loop_state(**overrides) -> dict:
    exec_ctx, prog = pitchvision_fact_stores()
    state = {
        "project_id": "pv-loop", "iteration": 1, "phase": "TEST",
        "active_assumption": dict(ASSUMPTION), "weak_area": "Problem",
        "reality_check": {"summary": "No direct evidence of coach behaviour yet."},
        "execution_context": exec_ctx, "startup_progress": prog,
        "founder_messages": [{"role": "founder", "content": PITCHVISION_FIRST_MESSAGE}],
        "raw_context": PITCHVISION_FIRST_MESSAGE,
        "critic_revision_count": 0, "critic_revision_instructions": "", "previous_experiments": [],
    }
    state.update(overrides)
    return state


def experiment(**overrides) -> dict:
    exp = make_valid_experiment_generator_output_dict(hypothesis=ASSUMPTION["statement"])
    exp.update(overrides)
    return exp


def critic(verdict="ACCEPT", **overrides) -> dict:
    out = make_valid_experiment_critic_output_dict(verdict)
    out.update(overrides)
    return out


def sequence(*items):
    it = iter(items)
    return lambda messages, rec: next(it)


@contextmanager
def offline(llm: ScriptedLLM):
    with patch("startary.graph.nodes.get_llm_client", side_effect=llm.get_llm_client), \
         patch("startary.graph.nodes.get_settings", return_value=OFFLINE_SETTINGS):
        yield


def run_loop(llm: ScriptedLLM, thread: str, saver=None, callbacks=None, **state_overrides):
    graph = build_graph(checkpointer=saver or MemorySaver())
    cfg = {"configurable": {"thread_id": thread}}
    if callbacks:
        cfg["callbacks"] = callbacks
    graph.update_state(cfg, loop_state(**state_overrides), as_node="decide_next_strategic_action")
    graph.invoke(None, cfg)
    return graph, cfg


class SpanRecorder(BaseCallbackHandler):
    def __init__(self):
        self.runs: dict = {}

    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, tags=None, metadata=None, **kwargs):
        self.runs[run_id] = {"name": kwargs.get("name"), "parent": parent_run_id, "metadata": dict(metadata or {})}

    def named(self, name):
        return [r for r in self.runs.values() if r["name"] == name]


# ---------------------------------------------------------------------------
# 1. Graph state persistence (real LangGraph transitions, checkpointing, resume)
# ---------------------------------------------------------------------------

class TestStatePersistence:
    def test_declared_keys_cover_every_experiment_loop_write(self):
        declared = set(typing.get_type_hints(StartaryGraphState))
        state = loop_state(proposed_experiment_output=experiment(), experiment_critic_output=critic("REVISE"))
        llm = ScriptedLLM({CRIT: lambda m, r: critic("ACCEPT")})
        written: set[str] = set()
        with offline(llm):
            for node in (validate_proposed_experiment, critique_experiment, save_final_experiment):
                out = node(dict(state)); written |= set(out); state.update(out)
            for node in (assess_can_execute, compose_execution_handoff, handle_experiment_failure_stop):
                out = node(dict(state)); written |= set(out); state.update(out)
        assert written - declared == set()

    def test_handoff_and_waiting_reply_survive_transitions_checkpoint_and_resume(self):
        llm = ScriptedLLM({
            GEN: lambda m, r: experiment(),
            CRIT: lambda m, r: critic("ACCEPT"),
            "FounderTurnInterpretationLLMOutput": lambda m, r: {
                "intent_types": ["QUESTION"], "direct_answer": False, "direct_question": "start smaller?",
                "concern_or_goal": None, "correction": None, "cannot_answer_now": False, "cannot_act_now": False,
                "uncertainty_statement": None, "new_evidence_summary": None, "founder_preference_or_constraint": None,
                "response_priority": "NORMAL", "interpretation_summary": "question",
            },
            "EvidenceTurnAssessmentLLMOutput": lambda m, r: {"disposition": "QUESTION_ABOUT_EXPERIMENT", "rationale_summary": "question"},
        })
        saver = MemorySaver()
        with offline(llm):
            graph, cfg = run_loop(llm, "pv-handoff", saver=saver)
            snap = graph.get_state(cfg)
            assert snap.next == ("wait_for_evidence",)
            values = snap.values
            assert values["active_execution_handoff"]["experiment_id"] == values["active_experiment"]["id"]
            assert values["execution_handoff_text"]
            # The founder sees the composed handoff, not a fallback re-render.
            assert snap.tasks[0].interrupts[0].value["message"] == values["execution_handoff_text"]

            restored = build_graph(checkpointer=saver)  # simulated process restart
            assert restored.get_state(cfg).values["active_execution_handoff"] == values["active_execution_handoff"]
            restored.invoke(Command(resume="Can I start with just the three coaches this week?"), cfg)
            snap2 = restored.get_state(cfg)

        assert snap2.next == ("wait_for_evidence",)
        reply = snap2.values["waiting_response"]
        assert reply and snap2.tasks[0].interrupts[0].value["message"] == reply  # advisor reply is displayed
        assert snap2.values["founder_messages"][-1] == {"role": "assistant", "content": reply}
        assert can_render_waiting_for_evidence(snap2.values) is True
        assert resolve_founder_facing_status(snap2.values) == TerminalStatus.WAITING_FOR_EVIDENCE

    def test_failure_status_persists_and_terminal_shows_needs_revision(self):
        llm = ScriptedLLM({GEN: lambda m, r: experiment(), CRIT: lambda m, r: critic("REVISE")})
        saver = MemorySaver()
        with offline(llm):
            graph, cfg = run_loop(llm, "pv-fail", saver=saver)
            snap = build_graph(checkpointer=saver).get_state(cfg)  # read back through a restored graph
        values = snap.values
        assert values["experiment_design_status"] == ExperimentDesignStatus.QUALITY_BAR_NOT_MET.value
        assert values["fallback_learning_action"]
        assert snap.tasks[0].interrupts[0].value["interrupt_type"] == "waiting_for_learning_input"
        assert resolve_founder_facing_status(values) == TerminalStatus.EXPERIMENT_DESIGN_NEEDS_REVISION
        assert resolve_founder_facing_status({"experiment_design_status": "QUALITY_BAR_NOT_MET"}) == TerminalStatus.EXPERIMENT_DESIGN_NEEDS_REVISION
        assert (llm.count(GEN), llm.count(CRIT)) == (2, 2)

    def test_founder_learning_input_clears_failure_status_and_counter(self):
        llm = ScriptedLLM({GEN: lambda m, r: experiment(), CRIT: lambda m, r: critic("REVISE")})
        with offline(llm):
            graph, cfg = run_loop(llm, "pv-fail-resume")
            graph.invoke(Command(resume="I spoke to two coaches."), cfg, interrupt_after=["merge_founder_answer"])
            values = graph.get_state(cfg).values
        assert values["experiment_design_status"] is None
        assert values["critic_revision_count"] == 0
        assert values["fallback_learning_action"] is None


# ---------------------------------------------------------------------------
# 2. ACCEPT / REVISE routing
# ---------------------------------------------------------------------------

class TestRouting:
    @pytest.mark.parametrize("verdict, count, expected", [
        ("ACCEPT", 0, "save_final_experiment"),
        ("ACCEPT", 1, "save_final_experiment"),
        ("REVISE", 1, "generate_experiment"),
        ("REVISE", 2, "handle_experiment_failure_stop"),
        ("ACCEPTED", 1, "generate_experiment"),               # not an exact token
        ("accept with minor changes", 1, "generate_experiment"),
        (None, 1, "generate_experiment"),
        (" accept ", 0, "save_final_experiment"),             # trimming/case only
    ])
    def test_route_table(self, verdict, count, expected):
        assert experiment_critique_route(verdict, count) == expected
        assert route_on_experiment_critique({"experiment_critic_output": {"verdict": verdict}, "critic_revision_count": count}) == expected

    @pytest.mark.parametrize("critiques, expected_calls, expected_end", [
        ([critic("ACCEPT")], (1, 1), "wait_for_evidence"),
        ([critic("REVISE"), critic("ACCEPT")], (2, 2), "wait_for_evidence"),
        ([critic("REVISE"), critic("REVISE")], (2, 2), "ask_and_wait_for_founder"),
    ], ids=["accept-first", "revise-then-accept", "revise-twice"])
    def test_graph_loop_bounds(self, critiques, expected_calls, expected_end):
        llm = ScriptedLLM({GEN: lambda m, r: experiment(), CRIT: sequence(*critiques)})
        with offline(llm):
            graph, cfg = run_loop(llm, f"pv-route-{len(critiques)}-{critiques[-1]['verdict']}")
            snap = graph.get_state(cfg)
        assert (llm.count(GEN), llm.count(CRIT)) == expected_calls
        assert snap.next == (expected_end,)

    def test_revision_round_two_receives_critic_instructions(self):
        llm = ScriptedLLM({
            GEN: lambda m, r: experiment(),
            CRIT: sequence(critic("REVISE", revision_instructions="Remove the demo; interview coaches about last week's match."), critic("ACCEPT")),
        })
        with offline(llm):
            run_loop(llm, "pv-round2")
        second_gen = llm.calls_for(GEN)[1]
        assert "Remove the demo" in second_gen.messages[-1]["content"]
        assert second_gen.config["metadata"]["revision_round"] == 2


# ---------------------------------------------------------------------------
# 3. Six-dimension verdict interpretation
# ---------------------------------------------------------------------------

ALL_PASS = {d: "PASS" for d in CRITIC_DIMENSIONS}


class TestVerdictInterpretation:
    def test_accept_requires_all_dimensions_pass(self):
        d = interpret_critic_output({"verdict": "ACCEPT", **ALL_PASS})
        assert (d.verdict, d.verdict_flag_conflict, d.failed_dimensions) == ("ACCEPT", False, [])

    def test_accept_with_failed_dimension_is_revise_conflict(self):
        d = interpret_critic_output({"verdict": "ACCEPT", **ALL_PASS, "feasibility": "FAIL"})
        assert (d.verdict, d.verdict_flag_conflict, d.failed_dimensions) == ("REVISE", True, ["feasibility"])

    def test_revise_with_all_pass_stays_revise_and_is_flagged(self):
        d = interpret_critic_output({"verdict": "REVISE", **ALL_PASS})
        assert (d.verdict, d.verdict_flag_conflict) == ("REVISE", True)

    def test_missing_dimension_never_accepts(self):
        flags = dict(ALL_PASS); flags.pop("scope")
        assert interpret_critic_output({"verdict": "ACCEPT", **flags}).verdict == "REVISE"

    @pytest.mark.parametrize("bad", ["ACCEPTED", "Accept with changes", "MAYBE", ""])
    def test_schema_rejects_non_token_verdicts(self, bad):
        with pytest.raises(Exception):
            ExperimentCriticLLMOutput(**{**critic("ACCEPT"), "verdict": bad})

    def test_schema_normalizes_case_and_whitespace_only(self):
        out = ExperimentCriticLLMOutput(**{**critic("ACCEPT"), "verdict": " accept ", "scope": "pass"})
        assert out.model_dump(mode="json")["verdict"] == "ACCEPT"

    def test_node_conflict_accept_with_fail_revises_with_actionable_instructions(self):
        llm = ScriptedLLM({CRIT: lambda m, r: critic("ACCEPT", feasibility="FAIL")})
        with offline(llm):
            res = critique_experiment(loop_state(proposed_experiment_output=experiment()))
        out = res["experiment_critic_output"]
        assert (out["verdict"], out["model_verdict"], out["verdict_flag_conflict"]) == ("REVISE", "ACCEPT", True)
        assert res["critic_revision_count"] == 1
        assert "feasibility" in res["critic_revision_instructions"]

    def test_node_revise_all_pass_without_issues_uses_rationale(self):
        llm = ScriptedLLM({CRIT: lambda m, r: {**critic("ACCEPT"), "verdict": "REVISE", "issues": [], "revision_instructions": "", "rationale": "Tighten the decision rule."}})
        with offline(llm):
            res = critique_experiment(loop_state(proposed_experiment_output=experiment()))
        assert res["critic_revision_instructions"] == "Tighten the decision rule."


# ---------------------------------------------------------------------------
# 4. BUILD / MVP compatibility
# ---------------------------------------------------------------------------

BUILD = {"decision": "BUILD_MINIMAL_PROTOTYPE", "minimum_build_scope": ["single-match clip summary for one coach"], "must_not_build": ["AI recommendations"]}
BUILD_NO_SCOPE = {"decision": "BUILD_MINIMAL_PROTOTYPE", "minimum_build_scope": [], "must_not_build": []}
MANUAL = {"decision": "CONTINUE_MANUAL"}
ARTIFACTS = {
    "none": {"mvp_exists": _fact("false")},
    "prototype": {"prototype_exists": _fact("true"), "mvp_exists": _fact("false")},
    "mvp": {"mvp_exists": _fact("true")},
}


@pytest.mark.parametrize("decision", ["BUILD", "BUILD_NO_SCOPE", "MANUAL", "NONE"])
@pytest.mark.parametrize("artifact", ["none", "prototype", "mvp"])
@pytest.mark.parametrize("exp_type", ["PROTOTYPE_TEST", "PROBLEM_INTERVIEW"])
def test_build_mvp_matrix(decision, artifact, exp_type):
    br = {"BUILD": BUILD, "BUILD_NO_SCOPE": BUILD_NO_SCOPE, "MANUAL": MANUAL, "NONE": None}[decision]
    status = assess_prototype_requirement(exp_type, ARTIFACTS[artifact], br).status
    if exp_type != "PROTOTYPE_TEST":
        expected = PrototypeStatus.NOT_APPLICABLE
    elif artifact != "none":
        expected = PrototypeStatus.EXISTING_ARTIFACT             # usability test on an existing prototype/MVP
    elif decision == "BUILD":
        expected = PrototypeStatus.BUILD_AUTHORIZED              # bounded build-and-test probe
    elif decision == "BUILD_NO_SCOPE":
        expected = PrototypeStatus.BUILD_SCOPE_UNDEFINED         # never unconstrained building
    else:
        expected = PrototypeStatus.UNAVAILABLE
    assert status == expected
    codes = evaluate_experiment_contract(experiment(experiment_type=exp_type), loop_state(startup_progress=ARTIFACTS[artifact], build_readiness=br)).violation_codes
    blocked = expected in (PrototypeStatus.UNAVAILABLE, PrototypeStatus.BUILD_SCOPE_UNDEFINED)
    assert bool({"PROTOTYPE_NOT_AVAILABLE", "BUILD_SCOPE_UNDEFINED"} & set(codes)) == blocked


class TestBuildMinimalPrototypeGraph:
    def test_authorized_build_reaches_llm_critic_with_scope_and_executes(self):
        """Before M6.2.5 this combination was a guaranteed dead end (two deterministic REVISEs)."""
        llm = ScriptedLLM({GEN: lambda m, r: experiment(), CRIT: lambda m, r: critic("ACCEPT")})
        with offline(llm):
            graph, cfg = run_loop(llm, "pv-build", build_readiness=BUILD, startup_progress=ARTIFACTS["none"])
            values = graph.get_state(cfg).values
        assert (llm.count(GEN), llm.count(CRIT)) == (1, 1)
        critic_msg = llm.calls_for(CRIT)[0].messages[-1]["content"]
        assert "DETERMINISTIC CHECK NOTES" in critic_msg and "single-match clip summary for one coach" in critic_msg
        assert values["active_experiment"]["experiment_type"] == "PROTOTYPE_TEST"
        assert not any("no prototype exists" in m for m in values["can_execute_result"]["missing_requirements"])

    def test_prototype_test_without_artifact_or_authorization_revises_deterministically(self):
        llm = ScriptedLLM({GEN: lambda m, r: experiment(experiment_type="PROTOTYPE_TEST"), CRIT: lambda m, r: critic("ACCEPT")})
        with offline(llm):
            graph, cfg = run_loop(llm, "pv-proto-manual", build_readiness=MANUAL, startup_progress=ARTIFACTS["none"])
            values = graph.get_state(cfg).values
        assert llm.count(CRIT) == 0
        assert values["experiment_critic_output"]["contract_violation_codes"] == ["PROTOTYPE_NOT_AVAILABLE"]


# ---------------------------------------------------------------------------
# 5. Budget checks
# ---------------------------------------------------------------------------

def _budget(value, provenance="EXPLICIT", currency=None):
    ctx = {"available_budget_amount": {"value": value, "provenance": provenance}}
    if currency:
        ctx["budget_currency"] = {"value": currency, "provenance": "EXPLICIT"}
    return ctx


@pytest.mark.parametrize("cost, ctx, expected", [
    ("$300", _budget("1500"), BudgetStatus.WITHIN_BUDGET),
    ("$2,000", _budget("1500"), BudgetStatus.EXCEEDS_BUDGET),
    ("$50", _budget("0"), BudgetStatus.EXCEEDS_BUDGET),
    ("Near-zero / $0", _budget("0"), BudgetStatus.NO_COST),
    ("$50", {}, BudgetStatus.BUDGET_UNKNOWN),
    ("$50", _budget(None, "UNKNOWN"), BudgetStatus.BUDGET_UNKNOWN),
    ("€50", _budget("1500", currency="USD"), BudgetStatus.NOT_COMPARABLE),
    ("$100-$2,000", _budget("1500"), BudgetStatus.WITHIN_BUDGET),     # lower bound fits
    ("$50 over 2 weeks", _budget("40"), BudgetStatus.EXCEEDS_BUDGET), # 2 is not money
    ("Low", _budget("0"), BudgetStatus.NO_COST),
])
def test_budget_comparison(cost, ctx, expected):
    assert compare_cost_to_budget(cost, ctx).status == expected


def test_money_parsing_handles_thousands_separators():
    assert parse_money("$1,500").low == 1500
    assert parse_money("1500", allow_bare_number=True).low == 1500


def test_budget_violation_vs_advisory():
    over = evaluate_experiment_contract(experiment(estimated_cost="$2,000"), loop_state())
    assert "COST_EXCEEDS_BUDGET" in over.violation_codes
    unknown_ctx = loop_state(execution_context={})
    unknown = evaluate_experiment_contract(experiment(estimated_cost="$80"), unknown_ctx)
    assert unknown.violation_codes == [] and "BUDGET_UNKNOWN" in unknown.advisory_codes


def test_can_execute_reads_thousands_correctly():
    """The old parser read "$1,200" as 1, and "$50 over 2 weeks" against $0 as 50 or 2 by position."""
    state = loop_state(active_experiment={"id": "e", "experiment_type": "OBSERVATION", "estimated_cost": "$1,200", "required_access": []})
    assert assess_can_execute(state)["can_execute_result"]["readiness"] == "CAN_EXECUTE"   # 1,200 <= 1,500
    state["active_experiment"]["estimated_cost"] = "$1,800"
    res = assess_can_execute(state)["can_execute_result"]
    assert res["readiness"] == "NEEDS_HELP" and "1800" in res["reason"]


# ---------------------------------------------------------------------------
# 6. Critical-assumption alignment
# ---------------------------------------------------------------------------

class TestAssumptionAlignment:
    def test_generator_stamps_draft_with_active_assumption(self):
        llm = ScriptedLLM({GEN: lambda m, r: experiment()})
        with offline(llm):
            res = generate_experiment(loop_state())
        assert res["proposed_experiment_output"]["assumption_id"] == ASSUMPTION["id"]

    def test_stale_draft_for_another_assumption_revises_without_llm(self):
        llm = ScriptedLLM({})
        with offline(llm):
            res = critique_experiment(loop_state(proposed_experiment_output=experiment(assumption_id="asmp-old")))
        assert llm.calls == []
        assert res["experiment_critic_output"]["contract_violation_codes"] == ["ASSUMPTION_MISMATCH"]
        assert res["experiment_critic_output"]["assumption_alignment"] == "FAIL"

    def test_validator_binds_hypothesis_to_active_statement(self):
        res = validate_proposed_experiment(loop_state(proposed_experiment_output=experiment(hypothesis="Coaches like apps.")))
        assert res["proposed_experiment_output"]["hypothesis"] == ASSUMPTION["statement"]


# ---------------------------------------------------------------------------
# 7. Validator errors and /retry
# ---------------------------------------------------------------------------

class TestValidatorAndRetry:
    def test_whitespace_draft_is_bounded_revise_not_a_graph_error(self):
        llm = ScriptedLLM({GEN: lambda m, r: experiment(support_signal="     "), CRIT: lambda m, r: critic("ACCEPT")})
        with offline(llm):
            graph, cfg = run_loop(llm, "pv-blank")  # no exception
            values = graph.get_state(cfg).values
        assert llm.count(CRIT) == 0                               # deterministic revise, twice
        assert llm.count(GEN) == 2
        assert values["experiment_design_status"] == "QUALITY_BAR_NOT_MET"

    def test_retry_after_blank_draft_regenerates_instead_of_repeating(self):
        llm = ScriptedLLM({GEN: sequence(experiment(support_signal="     "), experiment()), CRIT: lambda m, r: critic("ACCEPT")})
        with offline(llm):
            graph, cfg = run_loop(llm, "pv-blank-then-ok")
            snap = graph.get_state(cfg)
        assert snap.next == ("wait_for_evidence",)
        assert "support_signal" in llm.calls_for(GEN)[1].messages[1]["content"]  # revision names the defect

    def test_invalid_structured_output_retries_once_with_schema_feedback(self):
        llm = ScriptedLLM({GEN: sequence({"objective": "x"}, experiment())})
        with offline(llm):
            res = generate_experiment(loop_state())
        assert llm.count(GEN) == 2
        assert "did not satisfy the required output schema" in llm.calls_for(GEN)[1].messages[-1]["content"]
        assert res["proposed_experiment_output"]["objective"]


# ---------------------------------------------------------------------------
# 8. Retry taxonomy and token-exhaustion bounds
# ---------------------------------------------------------------------------

class LengthFinishReasonError(Exception):
    pass


class APITimeoutError(Exception):
    pass


class PermissionDenied(Exception):
    status_code = 403


def _raise(exc):
    def responder(messages, rec):
        raise exc
    return responder


class TestRetryBounds:
    @pytest.mark.parametrize("exc, kind", [
        (LengthFinishReasonError("length limit was reached"), LLMFailureKind.TOKEN_BUDGET_EXHAUSTED),
        (TokenBudgetExhaustedError("x"), LLMFailureKind.TOKEN_BUDGET_EXHAUSTED),
        (APITimeoutError("Request timed out"), LLMFailureKind.TRANSIENT_API_ERROR),
        (PermissionDenied("project does not have access to model"), LLMFailureKind.PERMANENT_API_ERROR),
        (ValueError("weird"), LLMFailureKind.UNKNOWN_ERROR),
    ])
    def test_failure_classification(self, exc, kind):
        assert classify_llm_failure(exc) == kind

    def test_critic_persistent_token_exhaustion_makes_two_calls_not_four(self):
        llm = ScriptedLLM({CRIT: _raise(LengthFinishReasonError("length limit was reached"))})
        state = loop_state(proposed_experiment_output=experiment())
        with offline(llm), pytest.raises(TokenBudgetExhaustedError):
            critique_experiment(state)
        assert llm.count(CRIT) == 2  # base call + one expanded-budget recovery
        (rec,) = state["turn_telemetry"].records
        assert (rec.status, rec.details["llm_failure_kinds"]) == ("FAILED", ["TOKEN_BUDGET_EXHAUSTED"])

    def test_generator_persistent_token_exhaustion_makes_two_calls(self):
        llm = ScriptedLLM({GEN: _raise(LengthFinishReasonError("length limit was reached"))})
        with offline(llm), pytest.raises(TokenBudgetExhaustedError):
            generate_experiment(loop_state())
        assert llm.count(GEN) == 2

    def test_transient_error_retries_once_unchanged(self):
        llm = ScriptedLLM({CRIT: sequence_raise_then(APITimeoutError("Request timed out"), critic("ACCEPT"))})
        with offline(llm):
            res = critique_experiment(loop_state(proposed_experiment_output=experiment()))
        calls = llm.calls_for(CRIT)
        assert len(calls) == 2 and calls[0].messages == calls[1].messages
        assert res["experiment_critic_output"]["verdict"] == "ACCEPT"

    def test_permanent_error_is_not_retried(self):
        llm = ScriptedLLM({CRIT: _raise(PermissionDenied("project does not have access to model"))})
        with offline(llm), pytest.raises(RuntimeError):
            critique_experiment(loop_state(proposed_experiment_output=experiment()))
        assert llm.count(CRIT) == 1

    def test_deterministic_violation_never_reaches_the_llm(self):
        llm = ScriptedLLM({})
        with offline(llm):
            res = critique_experiment(loop_state(proposed_experiment_output=experiment(estimated_cost="$5,000")))
        assert llm.calls == [] and res["experiment_critic_output"]["decision_source"] == "DETERMINISTIC"


def sequence_raise_then(exc, value):
    state = {"n": 0}

    def responder(messages, rec):
        state["n"] += 1
        if state["n"] == 1:
            raise exc
        return value
    return responder


# ---------------------------------------------------------------------------
# 9. Critic telemetry, LangSmith span, sanitization, privacy of metadata
# ---------------------------------------------------------------------------

DECISION_KEYS = {
    "critic_verdict", "critic_model_verdict", "verdict_flag_conflict", "failed_dimensions", "revision_trigger",
    "revision_round", "issue_count", "issue_codes", "contract_violation_codes", "contract_advisory_codes",
    "contract_finding_bases", "critic_attempts", "llm_failure_kinds", "next_route",
} | {f"dim_{d}" for d in CRITIC_DIMENSIONS}


class TestCriticObservability:
    def _run_revise_then_accept(self, recorder=None):
        issues = [f"The draft pitches the prototype to coaches before validating the problem ({SENTINEL}).", "No falsification threshold."]
        llm = ScriptedLLM({
            GEN: lambda m, r: experiment(method=f"Interview coaches about last week ({SENTINEL})."),
            CRIT: sequence(critic("REVISE", issues=issues, revision_instructions=f"Remove the demo {SENTINEL}."), critic("ACCEPT")),
        })
        with offline(llm):
            graph, cfg = run_loop(llm, "pv-observe", callbacks=[recorder] if recorder else None)
        return llm, graph.get_state(cfg).values

    def test_local_telemetry_records_structured_decisions(self):
        _, values = self._run_revise_then_accept()
        recs = [r for r in values["turn_telemetry"].records if r.node_name == "critique_experiment"]
        assert len(recs) == 2
        first, second = recs[0].details, recs[1].details
        assert set(first) == DECISION_KEYS
        assert (first["critic_verdict"], first["revision_trigger"], first["revision_round"], first["next_route"]) == ("REVISE", "LLM", 1, "generate_experiment")
        assert first["issue_count"] == 2 and {"CONTAMINATION", "FALSIFIABILITY"} <= set(first["issue_codes"])
        assert first["failed_dimensions"] == ["assumption_alignment", "evidence_validity"]
        assert (second["critic_verdict"], second["revision_round"], second["next_route"]) == ("ACCEPT", 2, "save_final_experiment")
        assert recs[0].model_tier == "reasoning" and recs[0].resolved_model == "scripted-reasoning"
        assert recs[0].logical_stage == "experiment_critique" and recs[0].latency_ms >= 0
        assert recs[0].reasoning_effort is None  # Sol effort unchanged (provider default)

    def test_decision_metadata_survives_sanitization_and_has_no_free_text(self):
        _, values = self._run_revise_then_accept()
        for rec in values["turn_telemetry"].records:
            if rec.node_name == "critique_experiment":
                assert sanitize_metadata(rec.details) == rec.details
        dumped = json.dumps([r.to_dict() for r in values["turn_telemetry"].records], default=str)
        assert SENTINEL not in dumped
        assert "pitches the prototype" not in dumped

    def test_decision_span_is_emitted_under_critic_node(self):
        recorder = SpanRecorder()
        llm, _ = self._run_revise_then_accept(recorder)
        spans = recorder.named("experiment_critic_decision")
        assert [s["metadata"]["critic_verdict"] for s in spans] == ["REVISE", "ACCEPT"]
        parents = {recorder.runs[s["parent"]]["name"] for s in spans}
        assert parents == {"critique_experiment"}
        assert SENTINEL not in json.dumps([s["metadata"] for s in spans], default=str)
        assert llm.calls_for(CRIT)[1].config["metadata"]["revision_round"] == 2

    def test_deterministic_decision_is_observable(self):
        state = loop_state(proposed_experiment_output=experiment(estimated_cost="$5,000"))
        with offline(ScriptedLLM({})):
            res = critique_experiment(state)
        (rec,) = res["turn_telemetry"].records
        assert rec.status == "SKIPPED" and rec.model_tier == "none"
        assert rec.details["revision_trigger"] == "DETERMINISTIC"
        assert rec.details["contract_violation_codes"] == ["COST_EXCEEDS_BUDGET"]
        assert rec.details["dim_feasibility"] == "FAIL" and rec.details["dim_scope"] == "NOT_EVALUATED"
        assert sanitize_metadata(rec.details) == rec.details


# ---------------------------------------------------------------------------
# 10. Privacy and consent safeguards
# ---------------------------------------------------------------------------

CONSENT_ACTION = "Ask each coach and player for permission before recording; store videos privately and delete them after the test."


class TestPrivacyConsent:
    def test_ordinary_interview_has_no_privacy_findings(self):
        # The shared fixture mentions "video interviews" (calls) and "Record whether ..." (note-taking).
        assert check_privacy_consent(make_valid_experiment_generator_output_dict(), "") == []

    @pytest.mark.parametrize("method", [
        "Film two matches per team and review the footage with each coach.",
        "Ask coaches to upload their match videos for analysis.",
        "Collect screenshots of the coaches' current analysis tools.",
        "Audio-record the coach debriefs after each match.",
    ])
    def test_recording_without_consent_is_a_violation(self, method):
        codes = [f.code for f in check_privacy_consent(experiment(method=method), "")]
        assert ContractCode.CONSENT_STEP_MISSING in codes

    def test_recording_with_consent_passes_with_minors_advisory(self):
        exp = experiment(method="Film two matches per team.", founder_actions=[CONSENT_ACTION, "Review the footage with each coach."])
        findings = check_privacy_consent(exp, "")
        assert [f.code for f in findings] == [ContractCode.POSSIBLE_MINORS]

    def test_known_minors_require_guardian_authorization(self):
        # M6.2.5.1: minors are established by the founder's participant-access fact, not general conversation.
        exp = experiment(method="Film two matches per team.", founder_actions=[CONSENT_ACTION])
        codes = [f.code for f in check_privacy_consent(exp, "", "Our two U14 youth teams")]
        assert codes == [ContractCode.GUARDIAN_CONSENT_MISSING]
        exp["founder_actions"].append("Get written parental consent through the club before filming.")
        assert check_privacy_consent(exp, "", "Our two U14 youth teams") == []

    def test_personal_data_is_advisory_only(self):
        exp = experiment(method="Collect email addresses of interested coaches at the end of each interview.")
        findings = check_privacy_consent(exp, "")
        assert [f.code for f in findings] == [ContractCode.PERSONAL_DATA_PRIVACY]

    def test_critic_revises_unconsented_recording_with_specific_instruction(self):
        with offline(ScriptedLLM({})):
            res = critique_experiment(loop_state(proposed_experiment_output=experiment(method="Film two matches per team and review the footage.")))
        assert res["experiment_critic_output"]["contract_violation_codes"] == ["CONSENT_STEP_MISSING"]
        assert "permission" in res["critic_revision_instructions"]

    def test_requirement_is_explicit_to_both_models_without_compliance_claims(self):
        assert "PRIVACY & CONSENT" in EXPERIMENT_GENERATOR_SYSTEM_PROMPT and "PRIVACY & CONSENT" in EXPERIMENT_CRITIC_SYSTEM_PROMPT
        assert "never claim legal compliance" in EXPERIMENT_GENERATOR_SYSTEM_PROMPT
        assert "ordinary interviews" in EXPERIMENT_CRITIC_SYSTEM_PROMPT.lower()


# ---------------------------------------------------------------------------
# 11. PitchVision founder-fact grounding in the experiment loop
# ---------------------------------------------------------------------------

class TestPitchVisionGrounding:
    def test_generator_and_critic_receive_pitchvision_facts(self):
        llm = ScriptedLLM({GEN: lambda m, r: experiment(), CRIT: lambda m, r: critic("ACCEPT")})
        with offline(llm):
            run_loop(llm, "pv-grounding")
        gen_msg = llm.calls_for(GEN)[0].messages[-1]["content"]
        crit_msg = llm.calls_for(CRIT)[0].messages[-1]["content"]
        for text in (gen_msg, crit_msg):
            assert "Direct access to two amateur football teams and three coaches" in text
            assert "1500" in text
        crit_ctx = crit_msg.split("Execution Context:")[1].split("Existing Founder Evidence:")[0]
        assert '"founder_count": "2"' in crit_ctx and '"paying_customers": "0"' in crit_ctx and '"revenue": "0"' in crit_ctx
        assert '"mvp_exists": "false"' in crit_ctx and '"interviews_completed": "0"' in crit_ctx
        assert "funding_status" not in crit_ctx  # unknown facts are not presented as known

    def test_pitchvision_budget_is_enforced(self):
        assert "COST_EXCEEDS_BUDGET" in evaluate_experiment_contract(experiment(estimated_cost="$2,000"), loop_state()).violation_codes
        assert evaluate_experiment_contract(experiment(estimated_cost="$300 for printed sheets"), loop_state()).violation_codes == []
