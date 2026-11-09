"""
Milestone 6.2.6.1 — Controlled live experiment-revision probe (offline verification only).

The probe seeds the PRODUCTION experiment subgraph right after generate_experiment and lets the real
nodes run. Offline, model responses come from the ScriptedLLM; the live path is only reachable with
--live --confirm-paid plus every preflight condition, and is never exercised here.
"""

from __future__ import annotations

import importlib
import json
import pathlib
from unittest.mock import patch

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

import startary.devtools.revision_probe as probe
from startary.config import Settings
from startary.devtools.revision_scenarios import (
    SCENARIOS,
    SHIFTRELAY_ASSUMPTION,
    SHIFTRELAY_FOUNDER_MESSAGE,
    scenario_state,
)
from startary.domain.experiment_contract import (
    BudgetStatus,
    CRITIC_DIMENSIONS,
    compare_cost_to_budget,
    evaluate_experiment_contract,
    required_authorizations,
)
from startary.llm.prompts import EXPERIMENT_CRITIC_REVISION_SCOPE_ADDENDUM, EXPERIMENT_CRITIC_SYSTEM_PROMPT
from startary.llm.schemas import ExperimentGeneratorLLMOutput
from startary.observability.langsmith import sanitize_metadata
from tests.test_m625_experiment_loop_correctness import SpanRecorder

CRIT_V2, FOCUSED, PATCH, GEN = "ExperimentCriticV2LLMOutput", "ExperimentFocusedCriticLLMOutput", "ExperimentRevisionPatchLLMOutput", "ExperimentGeneratorLLMOutput"


def smoke(path="targeted", scenario="shiftrelay", **kw):
    return probe.run_offline_smoke(scenario, path, **kw)


def live_settings(**overrides) -> Settings:
    values = dict(
        _env_file=None, openai_api_key="sk-live-looking-test-key-0000000000", reasoning_model_name="gpt-5.6-sol",
        fast_model_name="gpt-5.6-luna", experiment_revision_v2=True, langsmith_tracing=True,
        langsmith_api_key="lsv2_pt_test", langsmith_project="startary-ai-latency",
        langsmith_hide_inputs=True, langsmith_hide_outputs=True,
    )
    values.update(overrides)
    return Settings(**values)


# ---------------------------------------------------------------------------
# 1–2. Fixtures
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_seeded_draft_satisfies_the_generator_schema(scenario):
    draft = scenario_state(scenario)["proposed_experiment_output"]
    ExperimentGeneratorLLMOutput(**{k: draft[k] for k in ExperimentGeneratorLLMOutput.model_fields if k in draft})
    assert draft["hypothesis"] == SHIFTRELAY_ASSUMPTION["statement"]
    assert draft["assumption_id"] == SHIFTRELAY_ASSUMPTION["id"]


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_seeded_draft_has_no_deterministic_findings(scenario):
    state = scenario_state(scenario)
    draft = state["proposed_experiment_output"]
    contract = evaluate_experiment_contract(draft, state)
    assert (contract.violation_codes, contract.advisory_codes) == ([], [])
    assert required_authorizations(draft, state) == []                       # no recording / personal data / minors
    assert compare_cost_to_budget(draft["estimated_cost"], state["execution_context"]).status == BudgetStatus.WITHIN_BUDGET
    assert draft["experiment_type"] == "PROBLEM_INTERVIEW"                    # no prototype required
    assert "do not request schedules, payroll data or private messages" in draft["method"]


def test_shiftrelay_founder_facts_are_seeded_exactly():
    state = scenario_state("shiftrelay")
    ex, pr = state["execution_context"], state["startup_progress"]
    assert (ex["founder_count"]["value"], ex["approximate_weekly_hours_per_member"]["value"], ex["available_budget_amount"]["value"]) == ("2", "8", "350")
    assert "Three independent café/restaurant managers and four part-time workers" in ex["customer_access"]["value"]
    assert (pr["mvp_exists"]["value"], pr["prototype_exists"]["value"], pr["paying_customers"]["value"], pr["revenue"]["value"]) == ("false", "false", "0", "0")
    assert "at most two weeks" in SHIFTRELAY_FOUNDER_MESSAGE and "KakaoTalk" in SHIFTRELAY_FOUNDER_MESSAGE
    assert state["build_readiness"]["decision"] == "CONTINUE_MANUAL"
    assert state["experiment_revision"] == {"draft_origin": "INITIAL"} and state["critic_revision_count"] == 0


def test_primary_flaw_is_in_the_decision_criteria_not_the_method():
    draft = SCENARIOS["shiftrelay"]["draft"]
    assert "AI scheduling assistant" in draft["support_signal"] and "AI scheduling assistant" in draft["contradiction_signal"]
    assert "reconstruct" in draft["method"] and "eight weeks" in draft["method"]
    second = SCENARIOS["shiftrelay-b"]["draft"]
    assert "AI scheduling assistant" not in json.dumps(second)
    assert second["method"] == draft["method"]


# ---------------------------------------------------------------------------
# 3–10. Production graph through the probe (scripted, offline)
# ---------------------------------------------------------------------------

def test_probe_invokes_the_production_critic_on_the_seeded_draft():
    recorder = SpanRecorder()
    summary, llm = smoke("targeted", extra_callbacks=[recorder])
    first_call = llm.calls[0]
    assert first_call.schema == CRIT_V2                                          # generation skipped, real critic node first
    system, user = (m["content"] for m in first_call.messages)
    assert system == EXPERIMENT_CRITIC_SYSTEM_PROMPT + "\n\n" + EXPERIMENT_CRITIC_REVISION_SCOPE_ADDENDUM
    assert SCENARIOS["shiftrelay"]["draft"]["support_signal"] in user
    assert llm.count(GEN) == 0
    names = [r["name"] for r in recorder.runs.values()]
    for node in ("validate_proposed_experiment", "critique_experiment", "revise_experiment", "save_final_experiment", "assess_can_execute"):
        assert node in names, node
    root = next(r for r in recorder.runs.values() if r["name"] == "revision_probe_shiftrelay")
    assert root["parent"] is None and root["metadata"]["run_type"] == "revision_probe"


def test_scripted_revise_reaches_targeted_edit_and_focused_critic():
    summary, llm = smoke("targeted")
    assert summary["outcome"] == "TARGETED_EDIT_ACCEPTED" and summary["revision_path_exercised"] is True
    assert summary["model_calls"] == {"critique_experiment_llm": 2, "revise_experiment": 1, "generate_experiment_regeneration": 0}
    assert [c.schema for c in llm.calls] == [CRIT_V2, PATCH, FOCUSED]
    assert summary["second_critique"]["critic_mode"] == "FOCUSED"
    assert summary["revision_checks"]["second_critic_mode_as_expected"] is True
    assert summary["input_audit"] == {
        "reviser_called": True, "reviser_saw_original_method": True, "reviser_saw_original_support_signal": True,
        "reviser_saw_all_first_critic_issues": True, "focused_critic_saw_original_issues": True,
    }


def test_scripted_accept_terminates_without_revision():
    summary, llm = smoke("accept")
    assert summary["outcome"] == "FIRST_PASS_ACCEPT" and summary["revision_path_exercised"] is False
    assert [c.schema for c in llm.calls] == [CRIT_V2]
    assert summary["second_critique"] is None and summary["input_audit"] is None
    assert summary["final_route"] == "save_final_experiment" and summary["experiment_saved"] is True


def test_targeted_edit_changes_only_the_fields_the_reviser_returned():
    summary, llm = smoke("targeted")
    checks = summary["revision_checks"]
    patch_fields = set(probe.offline_smoke_responders("targeted")[PATCH](None, None))
    assert set(checks["changed_fields_vs_seeded_draft"]) == patch_fields
    assert checks["only_reported_fields_changed"] is True and checks["unchanged_fields_identical"] is True
    assert checks["revised_draft_schema_valid"] is True and checks["revised_draft_contract_violations"] == []


def test_full_regeneration_receives_a_full_critique():
    summary, llm = smoke("redesign")
    assert summary["outcome"] == "FULL_REGENERATION_ACCEPTED"
    assert [c.schema for c in llm.calls] == [CRIT_V2, GEN, CRIT_V2]
    assert summary["second_critique"]["critic_mode"] == "FULL"
    assert summary["input_audit"]["reviser_called"] is False


def test_failure_fallback_remains_reachable():
    summary, _ = smoke("reject")
    assert summary["outcome"] == "TARGETED_EDIT_REJECTED"
    assert summary["final_route"] == "handle_experiment_failure_stop"
    assert summary["experiment_design_status"] == "QUALITY_BAR_NOT_MET" and summary["experiment_saved"] is False
    assert summary["second_critique"]["strict_acceptance_rule_respected"] is True


def test_probe_is_bounded_even_when_the_reviser_changes_nothing():
    responders = dict(probe.offline_smoke_responders("targeted"))
    responders[PATCH] = lambda m, r: {}
    summary, llm = smoke(responders=responders)
    assert summary["final_route"] == "handle_experiment_failure_stop"
    assert llm.count(FOCUSED) == 0 and llm.count(CRIT_V2) == 1 and llm.count(PATCH) == 1
    assert sum(summary["model_calls"].values()) <= 4


def test_second_critique_rescored_all_six_dimensions():
    summary, _ = smoke("targeted")
    second = summary["second_critique"]
    assert second["all_six_dimensions_rescored"] is True
    assert set(second["dimensions"]) == set(CRITIC_DIMENSIONS)
    assert second["strict_acceptance_rule_respected"] is True and second["original_issues_resolved"] is True


# ---------------------------------------------------------------------------
# 11. Live calls impossible without explicit opt-in
# ---------------------------------------------------------------------------

@pytest.fixture
def no_live_calls():
    with patch("startary.graph.nodes.get_llm_client", side_effect=AssertionError("LLM client requested")), \
         patch.object(probe, "run_probe", side_effect=AssertionError("probe executed")) as run:
        yield run


@pytest.mark.parametrize("argv", [[], ["--live"], ["--live", "--scenario", "shiftrelay"], ["--live", "--offline-smoke", "--confirm-paid"]])
def test_live_requires_explicit_opt_in(no_live_calls, argv):
    assert probe.main(argv, settings=live_settings()) == probe.EXIT_REFUSED
    no_live_calls.assert_not_called()


@pytest.mark.parametrize("overrides, fragment", [
    ({"experiment_revision_v2": False}, "STARTARY_EXPERIMENT_REVISION_V2"),
    ({"openai_api_key": None}, "OPENAI_API_KEY"),
    ({"reasoning_model_name": "gpt-4o"}, "Reasoning model"),
    ({"langsmith_tracing": False}, "LANGSMITH_TRACING"),
    ({"langsmith_api_key": None}, "LANGSMITH_API_KEY"),
    ({"langsmith_project": "other"}, "LANGSMITH_PROJECT"),
    ({"langsmith_hide_inputs": False}, "HIDE_INPUTS"),
    ({"langsmith_hide_outputs": False}, "HIDE_OUTPUTS"),
])
def test_live_preflight_refuses_unsafe_configuration(no_live_calls, overrides, fragment, capsys):
    assert probe.main(["--live", "--confirm-paid"], settings=live_settings(**overrides)) == probe.EXIT_REFUSED
    assert fragment in capsys.readouterr().err
    no_live_calls.assert_not_called()


def test_live_preflight_passes_only_with_every_condition():
    assert probe.live_preflight(live_settings()) == []


def test_import_has_no_side_effects():
    with patch("startary.graph.builder.build_experiment_graph", side_effect=AssertionError("graph built on import")), \
         patch("startary.graph.nodes.get_llm_client", side_effect=AssertionError("LLM on import")):
        importlib.reload(probe)
    importlib.reload(probe)


def test_offline_smoke_never_constructs_a_real_model_client():
    with patch("langchain_openai.ChatOpenAI", side_effect=AssertionError("real client constructed")):
        summary, _ = smoke("targeted")
    assert summary["mode"] == "offline-smoke-targeted"


def test_live_input_auditor_reads_tags_and_round_from_real_langchain_callbacks():
    auditor = probe.PromptInputAuditor()
    FakeListChatModel(responses=["ok"]).invoke(
        [("system", "SYS"), ("user", "DRAFT")],
        config={"tags": ["node:revise_experiment", "attempt:1"], "metadata": {"revision_round": 2}, "callbacks": [auditor]},
    )
    assert probe.first_input(auditor.inputs, "revise_experiment", 2) == "SYS\nDRAFT"
    assert probe.first_input(auditor.inputs, "critique_experiment", 2) is None


# ---------------------------------------------------------------------------
# 12. LangSmith metadata safe and complete; local artifact
# ---------------------------------------------------------------------------

REQUIRED_SUMMARY_KEYS = {
    "probe_scenario", "probe_mode", "probe_outcome", "revision_path_exercised", "first_verdict", "first_failed_dimensions",
    "revision_strategy", "changed_fields", "second_critic_mode", "original_issues_resolved", "new_blocking_issue_count",
    "final_verdict", "final_route", "critic_llm_calls", "revise_calls", "regeneration_calls", "wall_clock_s",
    # M6.2.6.2 additions (critic scope vs. selected strategy, patch inspection, ShiftRelay-C rule check)
    "first_revision_scope", "strategy_reason", "patch_fields_returned", "changed_field_count",
    "criteria_overlap_before", "criteria_overlap_after",
} | {f"first_dim_{d}" for d in CRITIC_DIMENSIONS} | {f"second_dim_{d}" for d in CRITIC_DIMENSIONS}


def test_langsmith_summary_metadata_is_complete_and_safe():
    recorder = SpanRecorder()
    summary, _ = smoke("targeted", extra_callbacks=[recorder])
    (span,) = [r for r in recorder.runs.values() if r["name"] == "revision_probe_summary"]
    assert REQUIRED_SUMMARY_KEYS <= set(span["metadata"])
    emitted = probe.langsmith_summary_metadata(summary)
    assert sanitize_metadata(emitted) == emitted and set(emitted) == REQUIRED_SUMMARY_KEYS
    decision_spans = [r["metadata"] for r in recorder.runs.values() if r["name"] == "experiment_critic_decision"]
    assert [m["critic_mode"] for m in decision_spans] == ["FULL", "FOCUSED"]
    everything = json.dumps([r["metadata"] for r in recorder.runs.values()], default=str)
    for private in (SHIFTRELAY_FOUNDER_MESSAGE[:60], SCENARIOS["shiftrelay"]["draft"]["support_signal"], "sk-offline-smoke"):
        assert private not in everything


def test_summary_and_artifact_contain_no_secrets(tmp_path):
    summary, _ = smoke("targeted", artifact_dir=tmp_path)
    path = pathlib.Path(summary["artifact_path"])
    artifact = json.loads(path.read_text(encoding="utf-8"))
    assert artifact["seeded_draft"]["support_signal"] == SCENARIOS["shiftrelay"]["draft"]["support_signal"]
    assert artifact["final_draft"]["support_signal"] != artifact["seeded_draft"]["support_signal"]
    assert artifact["first_critique_feedback"]["failed_dimensions"] == ["evidence_validity", "falsifiability"]
    text = path.read_text(encoding="utf-8") + json.dumps(summary, default=str)
    assert "sk-" not in text and "lsv2_" not in text


def test_default_artifact_directory_is_git_ignored():
    assert probe.DEFAULT_ARTIFACT_DIR.parts[0] == ".startary_dev"
    assert ".startary_dev/" in pathlib.Path(".gitignore").read_text(encoding="utf-8").splitlines()


def test_cli_offline_smoke_prints_json_summary(capsys):
    assert probe.main(["--offline-smoke", "--no-artifact"]) == probe.EXIT_OK
    printed = json.loads(capsys.readouterr().out)
    assert printed["outcome"] == "TARGETED_EDIT_ACCEPTED" and printed["mode"] == "offline-smoke-targeted"
