"""
Milestone 6.2.6.2 — ShiftRelay-C: narrow decision-rule defect for TARGETED_EDIT + FOCUSED critique.

All runs are offline (ScriptedLLM through the production experiment subgraph). The scripted critic
outcomes below are the POSSIBLE live outcomes; none of them is forced on the real critic.
"""

from __future__ import annotations

import json

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

import startary.devtools.revision_probe as probe
from startary.devtools.revision_scenarios import SCENARIOS, SHIFTRELAY_FOUNDER_MESSAGE, scenario_state
from startary.domain.experiment_contract import CRITIC_DIMENSIONS, evaluate_experiment_contract, required_authorizations
from startary.domain.experiment_revision import select_revision_strategy, select_revision_strategy_with_reason
from startary.observability.langsmith import sanitize_metadata
from tests.test_m625_experiment_loop_correctness import SpanRecorder

CRIT_V2, FOCUSED, PATCH, GEN = "ExperimentCriticV2LLMOutput", "ExperimentFocusedCriticLLMOutput", "ExperimentRevisionPatchLLMOutput", "ExperimentGeneratorLLMOutput"
DRAFT = SCENARIOS["shiftrelay-c"]["draft"]


def smoke(path="targeted", responders=None, **kw):
    return probe.run_offline_smoke("shiftrelay-c", path, responders=responders, **kw)


def scripted(path="targeted", **critic_overrides):
    responders = dict(probe.offline_smoke_responders(path, "shiftrelay-c"))
    if critic_overrides:
        first = {**responders[CRIT_V2](None, None), **critic_overrides}
        responders[CRIT_V2] = lambda m, r: first
        if critic_overrides.get("revision_scope") == "REDESIGN" or critic_overrides.get("assumption_alignment") == "FAIL" or \
                sum(1 for d in CRITIC_DIMENSIONS if first.get(d) == "FAIL") > 3:
            accept = {**{d: "PASS" for d in CRITIC_DIMENSIONS}, "verdict": "ACCEPT", "issues": [], "revision_instructions": "",
                      "rationale": "ok", "revision_scope": "NONE"}
            calls = iter([first, accept])
            responders[CRIT_V2] = lambda m, r: next(calls)
            responders[GEN] = lambda m, r: {**DRAFT, **probe.SHIFTRELAY_C_REPAIR}
    return responders


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

def test_fixture_preserves_the_shiftrelay_setup():
    state = scenario_state("shiftrelay-c")
    assert DRAFT["method"] == SCENARIOS["shiftrelay"]["draft"]["method"]                 # same retrospective method
    assert "previous eight weeks" in DRAFT["method"] and "anonymized written notes" in DRAFT["method"]
    assert "do not request schedules, payroll data or private messages" in DRAFT["method"]
    assert DRAFT["target_sample_size"] == 3 and DRAFT["experiment_type"] == "PROBLEM_INTERVIEW"
    assert state["execution_context"]["available_budget_amount"]["value"] == "350"
    assert state["startup_progress"]["mvp_exists"]["value"] == "false" and "at most two weeks" in SHIFTRELAY_FOUNDER_MESSAGE
    assert "AI scheduling assistant" not in json.dumps(DRAFT)                              # otherwise high quality


def test_fixture_is_contract_clean():
    state = scenario_state("shiftrelay-c")
    contract = evaluate_experiment_contract(DRAFT, state)
    assert (contract.violation_codes, contract.advisory_codes, required_authorizations(DRAFT, state)) == ([], [], [])


def test_fixture_contains_exactly_the_intended_overlap_and_the_repair_removes_it():
    seeded = probe.decision_rule_check(DRAFT)
    assert seeded["overlap_detected"] is True
    assert seeded["support_requires_two_or_more"] is True
    assert seeded["contradiction_requires_zero_qualifying"] is False
    repaired = probe.decision_rule_check({**DRAFT, **probe.SHIFTRELAY_C_REPAIR})
    assert repaired == {
        "heuristic": True, "support_requires_two_or_more": True, "contradiction_requires_zero_qualifying": True,
        "contradiction_requires_completed_interviews": True, "contradiction_admits_exactly_one": False,
        "inconclusive_covers_exactly_one": True, "inconclusive_covers_incomplete_evidence": True, "overlap_detected": False,
    }


@pytest.mark.parametrize("contradiction, admits_one, requires_zero", [
    ("Fewer than 2 of the 3 managers meet the qualifying pattern.", True, False),
    ("At most one manager meets the pattern.", True, False),
    ("0 or 1 managers qualify.", True, False),
    ("No manager meets the qualifying pattern after all three interviews.", False, True),
    ("Zero qualifying managers once all three interviews are completed.", False, True),
    ("None of the three managers qualifies.", False, True),
])
def test_decision_rule_heuristic_phrasings(contradiction, admits_one, requires_zero):
    result = probe.decision_rule_check({**DRAFT, "contradiction_signal": contradiction})
    assert (result["contradiction_admits_exactly_one"], result["contradiction_requires_zero_qualifying"]) == (admits_one, requires_zero)


# ---------------------------------------------------------------------------
# Targeted edit + focused critique (the hypothesis under test)
# ---------------------------------------------------------------------------

def test_targeted_edit_changes_only_the_decision_rules_and_gets_a_focused_critique():
    summary, llm = smoke("targeted")
    assert [c.schema for c in llm.calls] == [CRIT_V2, PATCH, FOCUSED]
    assert summary["outcome"] == "TARGETED_EDIT_ACCEPTED" and summary["probe"] == "M6.2.6.2"
    first, checks, second = summary["first_critique"], summary["revision_checks"], summary["second_critique"]
    assert (first["critic_revision_scope"], first["revision_strategy"], first["strategy_reason"]) == ("TARGETED", "TARGETED_EDIT", "CRITIC_SCOPE_TARGETED")
    assert first["failed_dimensions"] == ["falsifiability"]
    assert checks["changed_fields_vs_seeded_draft"] == ["contradiction_signal", "inconclusive_signal"]
    assert checks["patch_fields_returned_but_unchanged"] == ["support_signal"]         # returned, identical, not counted
    assert checks["changed_field_count"] == 2 and checks["editable_field_count"] == 17
    assert all(checks["core_fields_preserved"].values())
    assert checks["revised_draft_schema_valid"] and checks["revised_draft_contract_violations"] == []
    assert second["critic_mode"] == "FOCUSED" and checks["second_critic_mode_as_expected"] is True
    assert second["all_six_dimensions_rescored"] and second["strict_acceptance_rule_respected"]
    assert summary["decision_rule_check"]["seeded"]["overlap_detected"] is True
    assert summary["decision_rule_check"]["final"]["overlap_detected"] is False
    assert all(summary["input_audit"].values())


def test_reviser_and_focused_critic_receive_the_real_feedback():
    summary, llm = smoke("targeted")
    reviser_user = llm.calls_for(PATCH)[0].messages[-1]["content"]
    first_out = llm.calls_for(CRIT_V2)[0].output
    assert first_out["revision_instructions"] in reviser_user
    assert all(i in reviser_user for i in first_out["issues"])
    assert "Failed dimensions: falsifiability" in reviser_user
    focused_user = llm.calls_for(FOCUSED)[0].messages[-1]["content"]
    assert '"field": "contradiction_signal"' in focused_user and '"field": "inconclusive_signal"' in focused_user
    assert "Fewer than 2 of the 3 managers" in focused_user                         # original value shown as "before"
    assert all(i in focused_user for i in first_out["issues"])


# ---------------------------------------------------------------------------
# Every routing possibility the real critic could produce (reported, never overridden)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("overrides, strategy, reason", [
    ({"revision_scope": "REDESIGN"}, "FULL_REGENERATION", "CRITIC_SCOPE_NOT_TARGETED"),
    ({"assumption_alignment": "FAIL"}, "FULL_REGENERATION", "ASSUMPTION_ALIGNMENT_FAILED"),
    ({"evidence_validity": "FAIL", "feasibility": "FAIL", "scope": "FAIL"}, "FULL_REGENERATION", "TOO_MANY_FAILED_DIMENSIONS"),
    ({"evidence_validity": "FAIL"}, "TARGETED_EDIT", "CRITIC_SCOPE_TARGETED"),
])
def test_strategy_follows_production_rules_for_each_critic_outcome(overrides, strategy, reason):
    summary, llm = smoke(responders=scripted(**overrides))
    first = summary["first_critique"]
    assert (first["revision_strategy"], first["strategy_reason"]) == (strategy, reason)
    if strategy == "FULL_REGENERATION":
        assert summary["second_critique"]["critic_mode"] == "FULL" and llm.count(PATCH) == 0
    else:
        assert summary["second_critique"]["critic_mode"] == "FOCUSED" and llm.count(GEN) == 0


def test_first_pass_accept_is_reported_honestly():
    summary, llm = smoke("accept")
    assert summary["outcome"] == "FIRST_PASS_ACCEPT" and summary["revision_path_exercised"] is False
    assert summary["first_critique"]["critic_revision_scope"] == "NONE"
    assert summary["decision_rule_check"]["final"]["overlap_detected"] is True       # the flaw is still there


def test_substantial_edit_is_sent_to_the_full_critic_and_reported():
    responders = scripted()
    responders[PATCH] = lambda m, r: {**probe.SHIFTRELAY_C_REPAIR,
                                      "method": "Run a two-week shadowing study in each café and log every absence live."}
    full_ok = {**{d: "PASS" for d in CRITIC_DIMENSIONS}, "verdict": "ACCEPT", "issues": [], "revision_instructions": "",
               "rationale": "ok", "revision_scope": "NONE"}
    first = responders[CRIT_V2](None, None)
    calls = iter([first, full_ok])
    responders[CRIT_V2] = lambda m, r: next(calls)
    summary, llm = smoke(responders=responders)
    checks = summary["revision_checks"]
    assert checks["expected_second_critic_mode"] == "FULL" and summary["second_critique"]["critic_mode"] == "FULL"
    assert checks["second_critic_mode_as_expected"] is True and checks["core_fields_preserved"]["method"] is False
    assert llm.count(FOCUSED) == 0


def test_focused_critic_new_blocking_issue_keeps_the_strict_rule():
    responders = scripted()
    responders[FOCUSED] = lambda m, r: {**{d: "PASS" for d in CRITIC_DIMENSIONS}, "verdict": "ACCEPT", "issues": [],
                                        "revision_instructions": "", "rationale": "x", "original_issues_resolved": True,
                                        "new_blocking_issues": ["The new rule requires all four part-time workers."]}
    summary, _ = smoke(responders=responders)
    assert summary["final_verdict"] == "REVISE" and summary["final_route"] == "handle_experiment_failure_stop"
    assert summary["second_critique"]["strict_acceptance_rule_respected"] is True


def test_rejected_targeted_edit_reaches_the_failure_fallback():
    summary, _ = smoke("reject")
    assert summary["outcome"] == "TARGETED_EDIT_REJECTED" and summary["experiment_design_status"] == "QUALITY_BAR_NOT_MET"


def test_strategy_reason_never_changes_the_strategy():
    for args in [("LLM", ["falsifiability"], [], "TARGETED"), ("LLM", ["assumption_alignment"], [], "TARGETED"),
                 ("LLM", ["evidence_validity"], [], "REDESIGN"), ("DETERMINISTIC", [], ["COST_EXCEEDS_BUDGET"], None),
                 ("DETERMINISTIC", [], ["PROTOTYPE_NOT_AVAILABLE"], None), ("LLM", ["falsifiability"], [], None)]:
        assert select_revision_strategy_with_reason(*args)[0] == select_revision_strategy(*args)


# ---------------------------------------------------------------------------
# Patch inspection (live path parses Sol's structured output locally)
# ---------------------------------------------------------------------------

def test_live_auditor_captures_the_reviser_patch_from_json_content():
    auditor = probe.PromptInputAuditor()
    FakeListChatModel(responses=[json.dumps(probe.SHIFTRELAY_C_REPAIR)]).invoke(
        [("user", "x")], config={"tags": ["node:revise_experiment"], "metadata": {"revision_round": 2}, "callbacks": [auditor]})
    assert probe.first_output(auditor.outputs, "revise_experiment", 2) == probe.SHIFTRELAY_C_REPAIR


def test_structured_output_parser_handles_tool_calls():
    msg = AIMessage(content="", tool_calls=[{"name": "ExperimentRevisionPatchLLMOutput", "args": {"inconclusive_signal": "x"}, "id": "1"}])
    assert probe.parse_structured_output(LLMResult(generations=[[ChatGeneration(message=msg)]])) == {"inconclusive_signal": "x"}
    assert probe.parse_structured_output(LLMResult(generations=[[ChatGeneration(message=AIMessage(content="not json"))]])) is None


# ---------------------------------------------------------------------------
# Observability and CLI
# ---------------------------------------------------------------------------

def test_langsmith_metadata_records_scope_strategy_patch_and_rule_check():
    recorder = SpanRecorder()
    summary, _ = smoke("targeted", extra_callbacks=[recorder])
    meta = probe.langsmith_summary_metadata(summary)
    assert sanitize_metadata(meta) == meta
    assert (meta["first_revision_scope"], meta["revision_strategy"], meta["strategy_reason"]) == ("TARGETED", "TARGETED_EDIT", "CRITIC_SCOPE_TARGETED")
    assert meta["changed_fields"] == ["contradiction_signal", "inconclusive_signal"] and meta["changed_field_count"] == 2
    assert meta["patch_fields_returned"] == ["contradiction_signal", "inconclusive_signal", "support_signal"]
    assert (meta["criteria_overlap_before"], meta["criteria_overlap_after"], meta["second_critic_mode"]) == (True, False, "FOCUSED")
    decisions = [r["metadata"] for r in recorder.runs.values() if r["name"] == "experiment_critic_decision"]
    assert decisions[0]["critic_revision_scope"] == "TARGETED"                       # production span carries the scope
    everything = json.dumps([r["metadata"] for r in recorder.runs.values()], default=str)
    for private in (DRAFT["contradiction_signal"], probe.SHIFTRELAY_C_REPAIR["inconclusive_signal"], SHIFTRELAY_FOUNDER_MESSAGE[:50]):
        assert private not in everything


def test_artifact_keeps_the_patch_and_checklist_locally(tmp_path):
    summary, _ = smoke("targeted", artifact_dir=tmp_path)
    artifact = json.loads(open(summary["artifact_path"], encoding="utf-8").read())
    assert artifact["reviser_patch_returned"]["contradiction_signal"] == probe.SHIFTRELAY_C_REPAIR["contradiction_signal"]
    assert len(artifact["human_evaluation_checklist"]) == 4
    assert artifact["final_draft"]["method"] == DRAFT["method"]


def test_cli_offline_smoke_for_shiftrelay_c(capsys):
    assert probe.main(["--offline-smoke", "--scenario", "shiftrelay-c", "--no-artifact"]) == probe.EXIT_OK
    printed = json.loads(capsys.readouterr().out)
    assert printed["outcome"] == "TARGETED_EDIT_ACCEPTED" and printed["scenario"] == "shiftrelay-c"


def test_cli_live_shiftrelay_c_still_requires_explicit_opt_in(capsys):
    assert probe.main(["--live", "--scenario", "shiftrelay-c"]) == probe.EXIT_REFUSED
    assert "--confirm-paid" in capsys.readouterr().err
