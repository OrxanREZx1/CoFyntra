"""
Milestone 6.2.3 — Founder-fact propagation, small confirmed defects, and observability coverage.

Task 2: merge_diagnosis rebuilt StartupState from scratch, so the risk evaluator
(assess_startup_areas) received UNKNOWN placeholders for every execution/progress fact even when
extraction had captured them. The graph-level tests run the real compiled graph on the PitchVision
AI first turn (scripted offline LLM) and inspect the exact startup-state JSON handed to the risk
evaluator's prompt builder.
Task 5: "zero paying customers" must stay negative evidence; testable-uncertainty must read `areas`.
Task 4: every first-turn LLM node records node / stage / tier / model / effort / retries / status.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from unittest.mock import patch

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

import startary.graph.nodes as nodes
from startary.config import Settings
from startary.domain.enums import WorkflowPhase
from startary.domain.evidence import contains_unnegated_keyword
from startary.domain.gap_plan import can_formulate_testable_uncertainty
from startary.domain.telemetry import TurnTelemetrySummary
from startary.graph.builder import build_graph
from startary.graph.nodes import (
    _startup_state_with_canonical_facts,
    assess_build_readiness,
    assess_startup_areas,
    assess_weak_link_discrimination,
    merge_diagnosis,
    plan_information_gap,
)
from startary.observability.langsmith import sanitize_metadata
from tests.fixtures.pitchvision_scenario import (
    PITCHVISION_ANSWER_1,
    PITCHVISION_FIRST_MESSAGE,
    PITCHVISION_NEVER_STATED,
    make_pitchvision_llm,
    pitchvision_diagnosis,
    tied_area_assessments,
)
from tests.fixtures.scripted_llm import ScriptedLLM

OFFLINE_SETTINGS = Settings(
    _env_file=None,
    openai_api_key="sk-offline-test",
    fast_model_name="scripted-fast",
    reasoning_model_name="scripted-reasoning",
)

# Every PitchVision fact the brief requires to survive, as (layer, field, expected value, provenance).
PITCHVISION_REQUIRED_FACTS = [
    ("execution_context", "founder_count", "2", "EXPLICIT"),                       # team composition
    ("execution_context", "active_team_size", "2", "EXPLICIT"),
    ("execution_context", "approximate_weekly_hours_per_member", "15", "EXPLICIT"),  # weekly availability
    ("execution_context", "total_weekly_capacity_hours", "30", "INFERRED"),
    ("execution_context", "available_budget_amount", "1500", "EXPLICIT"),          # budget
    ("execution_context", "customer_access", "Direct access to two amateur football teams and three coaches who agreed to test", "EXPLICIT"),
    ("progress", "development_stage", "IDEA", "INFERRED"),                          # idea stage
    ("progress", "mvp_exists", "false", "EXPLICIT"),                                # no MVP
    ("progress", "validation_activity", "No structured customer interviews yet", "EXPLICIT"),
    ("progress", "interviews_completed", "0", "EXPLICIT"),                          # no structured interviews
    ("progress", "paying_customers", "0", "EXPLICIT"),                              # zero paying customers
    ("progress", "revenue", "0", "EXPLICIT"),                                       # zero revenue
]


@contextmanager
def offline_graph(llm: ScriptedLLM):
    with patch("startary.graph.nodes.get_llm_client", side_effect=llm.get_llm_client), \
         patch("startary.graph.nodes.get_settings", return_value=OFFLINE_SETTINGS), \
         patch("startary.graph.nodes.build_area_assessment_user_message", wraps=nodes.build_area_assessment_user_message) as spy:
        yield build_graph(checkpointer=MemorySaver()), spy


def _risk_prompt_states(spy) -> list[dict]:
    """The startup-state JSON exactly as embedded in each assess_startup_areas prompt."""
    return [json.loads(call.kwargs["startup_state_summary"]) for call in spy.call_args_list]


def _run(graph, thread: str, resume: str | None = None) -> dict:
    cfg = {"configurable": {"thread_id": thread}}
    graph.invoke({
        "project_id": thread,
        "phase": WorkflowPhase.UNDERSTAND,
        "iteration": 1,
        "raw_context": PITCHVISION_FIRST_MESSAGE,
        "founder_messages": [{"role": "founder", "content": PITCHVISION_FIRST_MESSAGE}],
    }, cfg)
    if resume is not None:
        graph.invoke(Command(resume=resume), cfg)
    return graph.get_state(cfg).values


# ---------------------------------------------------------------------------
# Task 2 — founder facts survive diagnosis merging and reach the risk evaluator
# ---------------------------------------------------------------------------

class TestPitchVisionFactPropagation:
    def test_extracted_facts_survive_diagnosis_merging(self):
        llm = make_pitchvision_llm()
        with offline_graph(llm) as (graph, _):
            state = _run(graph, "pv-facts")

        assert llm.unscripted == []
        startup_state = state["startup_state"]
        for layer, field, value, provenance in PITCHVISION_REQUIRED_FACTS:
            fact = startup_state[layer][field]
            assert (fact["value"], fact["provenance"]) == (value, provenance), field

    def test_risk_assessment_receives_accurate_known_facts(self):
        llm = make_pitchvision_llm()
        with offline_graph(llm) as (graph, spy):
            _run(graph, "pv-risk")

        (risk_state,) = _risk_prompt_states(spy)
        for layer, field, value, provenance in PITCHVISION_REQUIRED_FACTS:
            fact = risk_state[layer][field]
            assert (fact["value"], fact["provenance"]) == (value, provenance), field
        # Target customers come from the diagnosed Target User area, with claims and evidence kept apart.
        target_user = risk_state["areas"]["Target User"]
        assert target_user["understanding"] == "Amateur football teams and their coaches."
        problem = risk_state["areas"]["Problem"]
        assert problem["founder_claims"] == ["Coaches want tactical insights from match video"]   # hypothesis
        assert problem["evidence_provided"] == []                                                 # not validated
        # Raw founder message is retained for downstream reasoning.
        assert PITCHVISION_FIRST_MESSAGE in risk_state["raw_context"]

    def test_unknown_information_is_not_invented(self):
        llm = make_pitchvision_llm()
        with offline_graph(llm) as (graph, spy):
            state = _run(graph, "pv-unknown")

        (risk_state,) = _risk_prompt_states(spy)
        for layer, fields in PITCHVISION_NEVER_STATED.items():
            for field in fields:
                for container in (state["startup_state"][layer], risk_state[layer]):
                    assert container[field]["provenance"] == "UNKNOWN", field
                    assert container[field]["value"] is None, field

    def test_facts_persist_into_follow_up_turn_with_empty_extraction(self):
        """Turn 2 extracts nothing new and diagnosis is cache-reused: facts must still reach the evaluator."""
        llm = make_pitchvision_llm()  # turn 2 delta is empty
        with offline_graph(llm) as (graph, spy):
            state = _run(graph, "pv-turn2", resume=PITCHVISION_ANSWER_1)

        risk_states = _risk_prompt_states(spy)
        assert len(risk_states) == 2
        for layer, field, value, provenance in PITCHVISION_REQUIRED_FACTS:
            assert risk_states[1][layer][field]["value"] == value, field
            assert state["startup_state"][layer][field]["value"] == value, field
        # The founder's answer reaches the evaluator through raw_context in the weak-link loop.
        assert PITCHVISION_ANSWER_1 in risk_states[1]["raw_context"]


class TestMergeDiagnosisNeverDowngradesFacts:
    def _state(self, **extra) -> dict:
        state = {
            "project_id": "pv-merge",
            "raw_context": PITCHVISION_FIRST_MESSAGE,
            "diagnosis_output": pitchvision_diagnosis(),
        }
        state.update(extra)
        return state

    def test_known_mirrored_fact_survives_when_canonical_store_is_empty(self):
        prior = {"execution_context": {"available_budget_amount": {"value": "1500", "provenance": "EXPLICIT"}},
                 "progress": {"paying_customers": {"value": "0", "provenance": "EXPLICIT"}}}
        res = merge_diagnosis(self._state(startup_state=prior, execution_context={}, startup_progress={}))
        assert res["startup_state"]["execution_context"]["available_budget_amount"]["value"] == "1500"
        assert res["startup_state"]["progress"]["paying_customers"]["value"] == "0"

    def test_canonical_unknown_never_overwrites_known_and_newer_known_wins(self):
        prior = {"execution_context": {
            "founder_count": {"value": "1", "provenance": "EXPLICIT"},
            "available_budget_amount": {"value": "1500", "provenance": "EXPLICIT"},
        }}
        canonical = {
            "founder_count": {"value": "2", "provenance": "EXPLICIT", "source_turn": 1},
            "available_budget_amount": {"value": None, "provenance": "UNKNOWN"},
        }
        res = merge_diagnosis(self._state(startup_state=prior, execution_context=canonical))
        exec_ctx = res["startup_state"]["execution_context"]
        assert exec_ctx["founder_count"]["value"] == "2"
        assert exec_ctx["available_budget_amount"]["value"] == "1500"

    def test_no_canonical_stores_means_no_injected_placeholders(self):
        st = {"areas": {}}
        assert _startup_state_with_canonical_facts({"startup_state": st}) == st


# ---------------------------------------------------------------------------
# Task 5 — confirmed small defects
# ---------------------------------------------------------------------------

PAYING = ["paying", "paid", "repeat payment", "repeat payments", "paying teams", "paying customers", "paid pilot", "subscription revenue", "mrr", "8 paying"]


@pytest.mark.parametrize("text, expected", [
    (PITCHVISION_FIRST_MESSAGE, False),
    ("We have no paying customers.", False),
    ("0 paying customers so far", False),
    ("paying customers: 0", False),
    ("We don't have any paying customers yet", False),
    ("nobody is paying yet", False),
    ("We have 10 paying customers", True),
    ("zero revenue and 8 paying teams", True),
    ("no revenue, but 3 paid pilots signed", True),
])
def test_paying_keyword_detection_respects_negation(text, expected):
    assert contains_unnegated_keyword(text, PAYING) is expected


class TestZeroPayingCustomersRemainsNegativeEvidence:
    def _assess(self, founder_text: str) -> dict:
        llm = ScriptedLLM({"StartupAreaAssessmentBatchLLMOutput": lambda m, r: tied_area_assessments()})
        state = {
            "project_id": "pv-traction",
            "raw_context": founder_text,
            "founder_messages": [{"role": "founder", "content": founder_text}],
            "startup_state": {"areas": {}},
            "diagnosis": {"summary": "Idea stage.", "startup_stage": "idea"},
        }
        with patch("startary.graph.nodes.get_llm_client", side_effect=llm.get_llm_client), \
             patch("startary.graph.nodes.get_settings", return_value=OFFLINE_SETTINGS):
            out = assess_startup_areas(state)
        return next(a for a in out["area_assessments_output"]["assessments"] if a["area"] == "Problem")

    def test_zero_paying_and_zero_active_users_do_not_reduce_problem_risk(self):
        problem = self._assess(PITCHVISION_FIRST_MESSAGE + " We have zero active users.")
        assert problem["uncertainty"] == "HIGH"
        assert problem["evidence_gap"] == "HIGH"
        assert "Adjusted for verified" not in (problem["rationale"] or "")

    def test_real_paying_traction_still_reduces_problem_risk(self):
        problem = self._assess("Our app is live and 8 paying teams record full matches weekly.")
        assert problem["uncertainty"] == "MEDIUM"
        assert "Adjusted for verified" in problem["rationale"]

    def test_build_readiness_does_not_claim_paying_customers_for_pitchvision(self):
        state = {
            "raw_context": PITCHVISION_FIRST_MESSAGE,
            "founder_messages": [{"role": "founder", "content": PITCHVISION_FIRST_MESSAGE}],
            "active_assumption": {"statement": "Amateur coaches will rewatch matches if analysis is automatic.", "area": "Problem"},
        }
        decision = assess_build_readiness(state)["build_readiness"]
        assert "already has a deployed product" not in decision["reason"]

    def test_build_readiness_live_paying_shortcut_still_applies(self):
        text = "Our app is live and 8 paying teams use it; 14 recorded full matches so far."
        decision = assess_build_readiness({"raw_context": text, "founder_messages": [], "active_assumption": {"statement": "x"}})["build_readiness"]
        assert "already has a deployed product" in decision["reason"]


class TestTestableUncertaintyReadsAreas:
    def _state(self, status: str) -> dict:
        return {
            "raw_context": "PitchVision analyses football matches.",  # no helper keywords present
            "founder_messages": [],
            "startup_state": {"areas": {
                "Target User": {"status": status, "understanding": "Amateur football coaches and their teams.", "founder_claims": []},
                "Problem": {"status": status, "understanding": "Post-match video review takes hours.", "founder_claims": []},
            }},
        }

    def test_diagnosed_areas_key_is_read(self):
        assert can_formulate_testable_uncertainty(self._state("PARTIALLY_KNOWN")) is True

    def test_unknown_area_narration_does_not_count_as_grounding(self):
        assert can_formulate_testable_uncertainty(self._state("UNKNOWN")) is False


# ---------------------------------------------------------------------------
# Task 4 — observability coverage
# ---------------------------------------------------------------------------

FIRST_TURN_LLM_NODES = {
    # node: (logical stage, tier, reasoning effort recorded)
    "interpret_founder_turn": ("intake", "fast", "low"),
    "extract_startup_state": ("intake", "fast", "low"),
    "diagnose_startup": ("diagnosis", "reasoning", None),
    "plan_information_gap": ("gap_planning", "reasoning", None),
    "assess_startup_areas": ("risk_assessment", "reasoning", None),
    "assess_weak_link_discrimination": ("weak_link_selection", "reasoning", None),
    "select_weak_link_question": ("question_planning", "reasoning", None),
    "compose_advisor_response": ("question_composition", "fast", "low"),
}


class TestObservabilityCoverage:
    def test_every_first_turn_llm_node_records_consistent_telemetry(self):
        llm = make_pitchvision_llm()
        with offline_graph(llm) as (graph, _):
            state = _run(graph, "pv-telemetry")

        summary = state["turn_telemetry"]
        assert isinstance(summary, TurnTelemetrySummary)
        records = {r.node_name: r for r in summary.records}
        for node, (stage, tier, effort) in FIRST_TURN_LLM_NODES.items():
            rec = records[node]
            assert rec.logical_stage == stage, node
            assert rec.model_tier == tier, node
            assert rec.resolved_model == f"scripted-{tier}", node
            assert rec.reasoning_effort == effort, node  # None = provider default (Sol effort unchanged)
            assert rec.status == "SUCCESS", node
            assert rec.retry_count == 0, node
            assert rec.latency_ms >= 0.0, node

    def test_telemetry_records_contain_no_founder_text(self):
        llm = make_pitchvision_llm()
        with offline_graph(llm) as (graph, _):
            state = _run(graph, "pv-privacy")
        serialized = json.dumps([r.to_dict() for r in state["turn_telemetry"].records], default=str).lower()
        for phrase in ("pitchvision", "amateur football", "coaches", "1,500"):
            assert phrase not in serialized

    def test_llm_call_run_metadata_carries_node_and_stage_and_survives_sanitization(self):
        llm = make_pitchvision_llm()
        with offline_graph(llm) as (graph, _):
            _run(graph, "pv-metadata")
        by_schema = {c.schema: c for c in llm.calls}
        for schema, node in [
            ("InformationGapPlanLLMOutput", "plan_information_gap"),
            ("StartupAreaAssessmentBatchLLMOutput", "assess_startup_areas"),
            ("WeakLinkDiscriminationLLMOutput", "assess_weak_link_discrimination"),
            ("WeakLinkClarifyingQuestionLLMOutput", "select_weak_link_question"),
            ("AdvisorResponseLLMOutput", "compose_advisor_response"),
        ]:
            meta = by_schema[schema].config["metadata"]
            assert meta["node_name"] == node
            assert meta["logical_stage"] == FIRST_TURN_LLM_NODES[node][0]
            assert sanitize_metadata(meta)["logical_stage"] == meta["logical_stage"]

    def test_failed_node_records_failed_status(self):
        def outage(messages, rec):
            raise RuntimeError("provider outage")

        llm = ScriptedLLM({"InformationGapPlanLLMOutput": outage})
        state = {"project_id": "pv-fail", "founder_messages": [], "raw_context": "x"}
        with patch("startary.graph.nodes.get_llm_client", side_effect=llm.get_llm_client), \
             patch("startary.graph.nodes.get_settings", return_value=OFFLINE_SETTINGS):
            with pytest.raises(RuntimeError, match="plan_information_gap failed"):
                plan_information_gap(state)
        (rec,) = state["turn_telemetry"].records
        assert (rec.node_name, rec.status, rec.retry_count, rec.model_tier) == ("plan_information_gap", "FAILED", 1, "reasoning")

    def test_discrimination_fast_path_records_skipped_without_llm(self):
        state = {
            "project_id": "pv-fast",
            "weak_area": "Problem",
            "weak_link_details": {"score": 81, "saturation_telemetry": {"max_score_tie_count": 1, "tie_breaker_used": False}},
            "area_assessments": [
                {"area": "Problem", "calculated_decision_risk": 81},
                {"area": "Target User", "calculated_decision_risk": 36},
            ],
        }
        with patch("startary.graph.nodes.get_llm_client", side_effect=AssertionError("LLM called")):
            res = assess_weak_link_discrimination(state)
        (rec,) = res["turn_telemetry"].records
        assert (rec.node_name, rec.status, rec.model_tier, rec.logical_stage) == (
            "assess_weak_link_discrimination", "SKIPPED", "none", "weak_link_selection",
        )
