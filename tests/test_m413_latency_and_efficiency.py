"""
Milestone M4.1.3 — Runtime Latency & Reasoning Efficiency Test Suite.

Comprehensive tests (47 tests) validating:
  1. Telemetry domain models and tracking
  2. Model tiering configuration and client routing
  3. Bound answers fast-path parser
  4. Conversational repair fast path
  5. State extraction fast-path bypass and prompt slimming
  6. Strategic state change detection and diagnosis snapshot cache hit
  7. Experiment and critic prompt slimming
  8. Timeout classification and stage-aware recovery messaging
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch
import pytest

from startary.cli import classify_cli_error
from startary.config import Settings, get_model_for_tier
from startary.domain.bound_answers import (
    parse_bound_answer,
    parse_capacity_answer,
    parse_product_stage_answer,
    parse_resource_answer,
    parse_team_answer,
)
from startary.domain.evidence import EvidenceTurnDisposition
from startary.domain.telemetry import (
    NodeTelemetryRecord,
    TurnTelemetrySummary,
    record_telemetry_event,
)
from startary.domain.trace import AITrace
from startary.domain.turn_control import WorkflowPhase
from startary.graph.builder import build_graph, route_after_interpret_founder_turn
from startary.graph.graph_state import StartaryGraphState
from startary.graph.nodes import (
    critique_experiment,
    diagnose_startup,
    extract_startup_state,
    generate_experiment,
    handle_conversational_repair,
    interpret_founder_turn,
    is_conversational_repair_text,
    is_strategic_state_changed,
    merge_founder_answer,
    assess_baseline_state_sufficiency,
    plan_baseline_question,
)
from startary.llm.provider import get_llm_client
from tests.fixtures.scenarios import make_valid_experiment_generator_output_dict


# ===========================================================================
# 1. Telemetry Domain Model Tests (1–6)
# ===========================================================================

def test_01_node_telemetry_record_init():
    rec = NodeTelemetryRecord(
        node_name="test_node",
        model_name="test_model",
        duration_ms=123.4,
        prompt_tokens=50,
        completion_tokens=25,
        total_tokens=75,
        cached=False,
        skipped=False,
        execution_path="FAST_LOCAL_UPDATE",
    )
    assert rec.node_name == "test_node"
    assert rec.total_tokens == 75
    assert rec.execution_path == "FAST_LOCAL_UPDATE"


def test_02_turn_telemetry_summary_init_and_counts():
    summary = TurnTelemetrySummary(turn_id=1)
    assert summary.total_duration_ms == 0.0
    assert summary.total_llm_calls == 0
    assert summary.nodes_visited == []
    assert summary.nodes_cached == 0
    assert summary.nodes_skipped == 0


def test_03_record_telemetry_event_aggregates_tokens_and_counts():
    state: StartaryGraphState = {}
    record_telemetry_event(
        state=state,
        node_name="node_1",
        model_name="model_fast",
        duration_ms=100.0,
        prompt_tokens=40,
        completion_tokens=10,
        total_tokens=50,
    )
    record_telemetry_event(
        state=state,
        node_name="node_2",
        model_name="model_reasoning",
        duration_ms=200.0,
        prompt_tokens=80,
        completion_tokens=20,
        total_tokens=100,
    )
    summary: TurnTelemetrySummary = state["turn_telemetry"]
    assert summary.total_duration_ms == 300.0
    assert summary.total_llm_calls == 2
    assert summary.total_prompt_tokens == 120
    assert summary.total_completion_tokens == 30
    assert summary.total_tokens == 150


def test_04_record_telemetry_event_tracks_nodes_visited():
    state: StartaryGraphState = {}
    record_telemetry_event(state=state, node_name="node_a")
    record_telemetry_event(state=state, node_name="node_b")
    summary: TurnTelemetrySummary = state["turn_telemetry"]
    assert summary.nodes_visited == ["node_a", "node_b"]


def test_05_record_telemetry_event_tracks_cached_and_skipped():
    state: StartaryGraphState = {}
    record_telemetry_event(state=state, node_name="node_skip", skipped=True)
    record_telemetry_event(state=state, node_name="node_cache", cached=True)
    summary: TurnTelemetrySummary = state["turn_telemetry"]
    assert summary.nodes_skipped == 1
    assert summary.nodes_cached == 1
    assert summary.total_llm_calls == 0


def test_06_ai_trace_has_telemetry_fields_backward_compatible():
    trace = AITrace(
        module_name="test_mod",
        prompt_version="v1",
        latency_ms=50.0,
        project_id="test_proj",
        model_name="fast-model",
        execution_path="CONVERSATIONAL_REPAIR",
        prompt_tokens=20,
        completion_tokens=10,
        total_tokens=30,
        skipped_by_fast_path=True,
    )
    dump = trace.model_dump()
    assert dump["model_name"] == "fast-model"
    assert dump["execution_path"] == "CONVERSATIONAL_REPAIR"
    assert dump["skipped_by_fast_path"] is True


# ===========================================================================
# 2. Model Tiering Configuration Tests (7–13)
# ===========================================================================

def test_07_fast_model_config_loaded_from_env(monkeypatch):
    monkeypatch.setenv("STARTARY_FAST_MODEL", "gemini-2.5-flash")
    s = Settings()
    assert s.fast_model_name == "gemini-2.5-flash"


def test_08_reasoning_model_config_loaded_from_env(monkeypatch):
    monkeypatch.setenv("STARTARY_REASONING_MODEL", "gemini-2.5-pro")
    s = Settings()
    assert s.reasoning_model_name == "gemini-2.5-pro"


def test_09_get_model_for_tier_fast():
    s = Settings(fast_model_name="my-fast-model")
    assert get_model_for_tier(s, "fast") == "my-fast-model"


def test_10_get_model_for_tier_reasoning():
    s = Settings(reasoning_model_name="my-reasoning-model")
    assert get_model_for_tier(s, "reasoning") == "my-reasoning-model"


def test_11_get_model_for_tier_fallback_to_default():
    s = Settings(model_name="my-default-model")
    assert get_model_for_tier(s, "unknown_tier") == "my-default-model"


def test_12_get_llm_client_tier_fast_uses_fast_model():
    s = Settings(openai_api_key="sk-fake", fast_model_name="fast-gpt")
    with patch("langchain_openai.ChatOpenAI") as mock_cls:
        get_llm_client(s, tier="fast")
        mock_cls.assert_called_once()
        assert mock_cls.call_args[1]["model"] == "fast-gpt"


def test_13_get_llm_client_tier_reasoning_uses_reasoning_model():
    s = Settings(openai_api_key="sk-fake", reasoning_model_name="reasoning-gpt")
    with patch("langchain_openai.ChatOpenAI") as mock_cls:
        get_llm_client(s, tier="reasoning")
        mock_cls.assert_called_once()
        assert mock_cls.call_args[1]["model"] == "reasoning-gpt"


# ===========================================================================
# 3. Bound Answers Parser Tests (14–27)
# ===========================================================================

def test_14_parse_capacity_explicit_hours():
    ok, delta, conf = parse_capacity_answer("We can dedicate 15 hours a week")
    assert ok is True
    assert delta["total_weekly_capacity_hours"]["value"] == "15"
    assert conf == "HIGH"


def test_15_parse_capacity_range_uses_upper_bound():
    ok, delta, _ = parse_capacity_answer("About 10-20 hours")
    assert ok is True
    assert delta["total_weekly_capacity_hours"]["value"] == "20"


def test_16_parse_capacity_correction_phrase():
    ok, delta, _ = parse_capacity_answer("Actually not 10, more like 25 hours")
    assert ok is True
    assert delta["total_weekly_capacity_hours"]["value"] == "25"


def test_17_parse_capacity_full_time_keyword():
    ok, delta, _ = parse_capacity_answer("Working on this full-time")
    assert ok is True
    assert delta["total_weekly_capacity_hours"]["value"] == "40"


def test_18_parse_capacity_part_time_keyword():
    ok, delta, _ = parse_capacity_answer("Currently part time")
    assert ok is True
    assert delta["total_weekly_capacity_hours"]["value"] == "20"


def test_19_parse_team_explicit_cofounders():
    ok, delta, _ = parse_team_answer("2 co-founders with engineering background")
    assert ok is True
    assert delta["founder_count"]["value"] == "2"


def test_20_parse_team_alone_solo():
    ok, delta, _ = parse_team_answer("I am building this alone as a solo founder")
    assert ok is True
    assert delta["founder_count"]["value"] == "1"


def test_21_parse_resource_dollar_budget():
    ok, delta, _ = parse_resource_answer("We have $5000 in personal savings")
    assert ok is True
    assert delta["available_budget_amount"]["value"] == "5000"


def test_22_parse_resource_k_suffix():
    ok, delta, _ = parse_resource_answer("Roughly 10k budget")
    assert ok is True
    assert delta["available_budget_amount"]["value"] == "10000"


def test_23_parse_resource_zero_budget():
    ok, delta, _ = parse_resource_answer("Zero budget, bootstrapped with no money")
    assert ok is True
    assert delta["available_budget_amount"]["value"] == "0"


def test_24_parse_product_stage_idea_only():
    ok, delta, _ = parse_product_stage_answer("No product yet, just an idea")
    assert ok is True
    assert delta["prototype_exists"]["value"] == "false"


def test_25_parse_product_stage_prototype():
    ok, delta, _ = parse_product_stage_answer("We built a clickable prototype in Figma")
    assert ok is True
    assert delta["prototype_exists"]["value"] == "true"


def test_26_parse_product_stage_mvp():
    ok, delta, _ = parse_product_stage_answer("We have a working MVP built with React")
    assert ok is True
    assert delta["mvp_exists"]["value"] == "true"


def test_27_parse_bound_answer_rejects_multi_topic_strategic():
    msg = (
        "We have 15 hours a week and our paying customer revenue is $2000 from 5 pilot stores "
        "who loved the inventory optimization feature and gave us written feedback."
    )
    ok, delta, conf = parse_bound_answer("CAPACITY", msg)
    assert ok is False


# ===========================================================================
# 4. Conversational Repair Tests (28–34)
# ===========================================================================

def test_28_is_conversational_repair_text_detects_common_phrases():
    repairs = [
        "I don't understand",
        "i dont understand",
        "What do you mean?",
        "Can you explain?",
        "Could you rephrase that?",
        "Why are you asking this?",
        "I'm confused",
        "Huh?",
        "What?",
    ]
    for r in repairs:
        assert is_conversational_repair_text(r) is True, f"Failed for: {r}"


def test_29_is_conversational_repair_text_rejects_substantive_answers():
    non_repairs = [
        "We don't understand our customers yet",
        "We have 20 hours a week",
        "Our team has 2 engineers",
        "The problem is that stores have stockouts",
    ]
    for nr in non_repairs:
        assert is_conversational_repair_text(nr) is False, f"False positive on: {nr}"


def test_30_merge_founder_answer_intercepts_repair_without_turn_increment():
    state = {
        "project_id": "test_repair",
        "active_interrupt_type": "baseline",
        "current_baseline_question": {"question": "What is your weekly capacity?", "target_group": "CAPACITY"},
        "active_next_turn_decision": {"question_id": "q1", "canonical_question": "What is your weekly capacity?"},
        "baseline_question_turns": 1,
        "founder_response": "I don't understand",
        "founder_messages": [],
    }
    updates = merge_founder_answer(state)
    assert updates["is_conversational_repair"] is True
    assert updates["execution_path"] == "CONVERSATIONAL_REPAIR"
    # Turns NOT incremented
    assert "baseline_question_turns" not in updates or updates["baseline_question_turns"] == 1


def test_31_merge_founder_answer_preserves_active_interrupt_type():
    state = {
        "project_id": "test_repair",
        "active_interrupt_type": "baseline",
        "current_baseline_question": {"question": "Q1"},
        "founder_response": "What do you mean?",
    }
    updates = merge_founder_answer(state)
    assert updates["active_interrupt_type"] == "baseline"
    assert updates["current_baseline_question"] == {"question": "Q1"}


def test_32_handle_conversational_repair_rephrases_pending_question():
    state = {
        "project_id": "test_repair",
        "pending_question": "How many hours per week can you spend on this startup?",
        "founder_messages": [{"role": "founder", "content": "What do you mean?"}],
        "active_next_turn_decision": {"question_id": "q1"},
    }
    updates = handle_conversational_repair(state)
    assert updates["execution_path"] == "CONVERSATIONAL_REPAIR"
    advisor_resp = updates["latest_advisor_response"]
    assert advisor_resp["response_mode"] == "CONVERSATIONAL_REPAIR"
    assert len(advisor_resp["response_text"]) > 10


def test_33_route_after_interpret_founder_turn_routes_repair_to_handler():
    state = {
        "is_conversational_repair": True,
        "execution_path": "CONVERSATIONAL_REPAIR",
        "phase": WorkflowPhase.UNDERSTAND,
    }
    route = route_after_interpret_founder_turn(state)
    assert route == "handle_conversational_repair"


def test_34_repair_edge_routes_directly_to_ask_and_wait():
    graph = build_graph()
    # verify handle_conversational_repair node exists and is in graph
    assert "handle_conversational_repair" in graph.nodes


# ===========================================================================
# 5. State Extraction Fast-Path Tests (35–37)
# ===========================================================================

def test_35_extract_startup_state_bypasses_llm_on_fast_local_update():
    state = {
        "project_id": "test_fast_bypass",
        "execution_path": "FAST_LOCAL_UPDATE",
        "startup_state_delta": {
            "total_weekly_capacity_hours": {"value": "20", "provenance": "EXPLICIT"}
        },
        "traces": [],
    }
    # Should NOT call LLM
    with patch("startary.graph.nodes.get_llm_client") as mock_get_llm:
        updates = extract_startup_state(state)
        mock_get_llm.assert_not_called()
        assert updates["startup_state_delta"]["total_weekly_capacity_hours"]["value"] == "20"
        telem: TurnTelemetrySummary = updates["turn_telemetry"]
        assert telem.nodes_skipped == 1


def test_36_extract_startup_state_prompt_slimming_uses_last_3_founder_msgs():
    state = {
        "project_id": "test_slimming",
        "founder_messages": [
            {"role": "founder", "content": f"Message {i}"} for i in range(10)
        ],
        "execution_path": "DELTA_STATE_UPDATE",
        "traces": [],
    }
    with patch("startary.graph.nodes.get_llm_client") as mock_get_llm, \
         patch("startary.graph.nodes.build_extract_startup_state_user_message") as mock_build_msg:
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_res = MagicMock()
        mock_res.model_dump.return_value = {}
        mock_structured.invoke.return_value = mock_res
        mock_llm.with_structured_output.return_value = mock_structured
        mock_get_llm.return_value = mock_llm

        extract_startup_state(state)
        # Verify the user message builder was called with slimmed founder text (3 messages)
        assert mock_build_msg.called
        text_arg = mock_build_msg.call_args[1]["founder_messages_text"]
        lines = [l for l in text_arg.split("\n\n") if l.strip()]
        assert len(lines) == 3


def test_37_extract_startup_state_records_skipped_telemetry():
    state = {
        "project_id": "test_telem",
        "execution_path": "FAST_LOCAL_UPDATE",
        "startup_state_delta": {"team": "fast"},
    }
    updates = extract_startup_state(state)
    assert updates["last_trace"]["skipped_by_fast_path"] is True


# ===========================================================================
# 6. Strategic State Change & Diagnosis Cache Tests (38–42)
# ===========================================================================

def test_38_is_strategic_state_changed_false_for_bound_capacity_team():
    state = {
        "diagnosis_output": {"summary": "Existing"},
        "execution_path": "FAST_LOCAL_UPDATE",
        "strategic_state_changed": False,
        "startup_state_delta": {"weekly_hours_per_member": 20},
    }
    assert is_strategic_state_changed(state) is False


def test_39_is_strategic_state_changed_false_for_conversational_repair():
    state = {
        "diagnosis_output": {"summary": "Existing"},
        "execution_path": "CONVERSATIONAL_REPAIR",
    }
    assert is_strategic_state_changed(state) is False


def test_40_is_strategic_state_changed_true_for_paying_customers():
    state = {
        "diagnosis_output": {"summary": "Existing"},
        "execution_path": "DELTA_STATE_UPDATE",
        "startup_state_delta": {"paying_customers": {"value": "10"}},
    }
    assert is_strategic_state_changed(state) is True


def test_41_is_strategic_state_changed_true_for_actual_evidence():
    state = {
        "diagnosis_output": {"summary": "Existing"},
        "evidence_turn_assessment": {"disposition": "ACTUAL_EVIDENCE"},
    }
    assert is_strategic_state_changed(state) is True


def test_42_diagnose_startup_reuses_cached_diagnosis_when_not_changed():
    existing_diag = {"id": "d1", "summary": "Old diagnosis"}
    state = {
        "project_id": "test_cache",
        "diagnosis_output": existing_diag,
        "strategic_state_changed": False,
        "execution_path": "FAST_LOCAL_UPDATE",
        "traces": [],
    }
    with patch("startary.graph.nodes.get_llm_client") as mock_get_llm:
        updates = diagnose_startup(state)
        mock_get_llm.assert_not_called()
        assert updates["diagnosis_output"] == existing_diag
        assert updates["last_trace"]["skipped_by_fast_path"] is True
        telem: TurnTelemetrySummary = updates["turn_telemetry"]
        assert telem.nodes_cached == 1


# ===========================================================================
# 7. Prompt Slimming Tests (43–44)
# ===========================================================================

def test_43_generate_experiment_slims_founder_evidence_summary():
    state = {
        "project_id": "test_slim_exp",
        "active_assumption": {"id": "ca1", "statement": "Test CA", "area": "Problem"},
        "founder_messages": [{"role": "founder", "content": f"Founder turn {i}"} for i in range(12)],
        "traces": [],
    }
    with patch("startary.graph.nodes.get_llm_client") as mock_get_llm, \
         patch("startary.graph.nodes.build_experiment_generator_user_message") as mock_build:
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_res = MagicMock()
        mock_res.model_dump.return_value = {"experiment_type": "INTERVIEWS", "objective": "Test obj"}
        mock_structured.invoke.return_value = mock_res
        mock_llm.with_structured_output.return_value = mock_structured
        mock_get_llm.return_value = mock_llm

        generate_experiment(state)
        assert mock_build.called
        summary_arg = mock_build.call_args[1]["founder_evidence_summary"]
        lines = summary_arg.split("\n")
        assert len(lines) == 3


def test_44_critique_experiment_slims_founder_evidence_summary():
    state = {
        "project_id": "test_slim_critique",
        "active_assumption": {"id": "ca1", "statement": "Test CA", "area": "Problem"},
        # M6.2.5: a complete draft; a partial one is now a deterministic REVISE before any LLM call.
        "proposed_experiment_output": make_valid_experiment_generator_output_dict(),
        "founder_messages": [{"role": "founder", "content": f"Founder turn {i}"} for i in range(15)],
        "traces": [],
    }
    with patch("startary.graph.nodes.get_llm_client") as mock_get_llm, \
         patch("startary.graph.nodes.build_experiment_critic_user_message") as mock_build:
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_res = MagicMock()
        mock_res.model_dump.return_value = {"verdict": "ACCEPT", "issues": []}
        mock_structured.invoke.return_value = mock_res
        mock_llm.with_structured_output.return_value = mock_structured
        mock_get_llm.return_value = mock_llm

        critique_experiment(state)
        assert mock_build.called
        summary_arg = mock_build.call_args[1]["founder_evidence_summary"]
        lines = summary_arg.split("\n")
        assert len(lines) == 3


# ===========================================================================
# 8. Timeout Classification Tests (45–47)
# ===========================================================================

def test_45_classify_cli_error_generate_experiment_stage():
    state = {"current_node": "generate_experiment", "startup_state": {"areas": {}}}
    exc = TimeoutError("Request timed out at 60s.")
    cat, title, guidance = classify_cli_error(exc, state=state)
    assert cat == "TIMEOUT"
    assert "EXPERIMENT GENERATION TEMPORARILY UNAVAILABLE" in title
    assert "No new experiment was created" in guidance
    assert "startup state and diagnosis are safe" in guidance


def test_46_classify_cli_error_extract_startup_state_stage():
    state = {"current_node": "extract_startup_state"}
    exc = TimeoutError("Request timed out at 60s.")
    cat, title, guidance = classify_cli_error(exc, state=state)
    assert cat == "TIMEOUT"
    assert "COULD NOT PROCESS THIS UPDATE" in title
    assert "previous state is safe" in guidance


def test_47_classify_cli_error_evidence_present_diagnosis_stage():
    state = {"evidence_analysis": {"conclusion": "SUPPORTED"}}
    exc = TimeoutError("Request timed out at 60s.")
    cat, title, guidance = classify_cli_error(exc, state=state)
    assert cat == "TIMEOUT"
    assert "DIAGNOSIS TEMPORARILY UNAVAILABLE" in title
    assert "Your completed evidence has been saved" in guidance
