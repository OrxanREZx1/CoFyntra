"""
Milestone 6.2.6 — Efficient experiment revision (STARTARY_EXPERIMENT_REVISION_V2).

Graph-level tests run the REAL compiled graph + MemorySaver, seeded as if decide_next_strategic_action
had just run, with only get_llm_client replaced by the offline ScriptedLLM. Schemas identify the call:
  ExperimentGeneratorLLMOutput      full generation (initial or FULL_REGENERATION)
  ExperimentRevisionPatchLLMOutput  TARGETED_EDIT (revise_experiment)
  ExperimentCriticLLMOutput         flag OFF critic (M6.2.5.1)
  ExperimentCriticV2LLMOutput       flag ON full critic
  ExperimentFocusedCriticLLMOutput  flag ON focused re-critique
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from unittest.mock import patch

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from startary.config import Settings
from startary.domain.experiment_contract import CRITIC_DIMENSIONS, CriticDecision, evaluate_experiment_contract, pending_authorizations
from startary.domain.experiment_revision import (
    MAX_SEMANTIC_REVISIONS,
    REMEDIATION_ACTIONS,
    apply_focused_acceptance_rule,
    diff_experiment,
    is_substantial_revision,
    remediate_consent_violations,
    select_critic_mode,
    select_revision_strategy,
)
from startary.graph.builder import build_graph
from startary.graph.nodes import validate_proposed_experiment
from startary.observability.langsmith import sanitize_metadata
from tests.fixtures.scripted_llm import ScriptedLLM
from tests.test_m625_experiment_loop_correctness import ASSUMPTION, SpanRecorder, experiment, loop_state

V2 = Settings(_env_file=None, openai_api_key="sk-offline-test", fast_model_name="scripted-fast",
              reasoning_model_name="scripted-reasoning", experiment_revision_v2=True)
V1 = Settings(_env_file=None, openai_api_key="sk-offline-test", fast_model_name="scripted-fast",
              reasoning_model_name="scripted-reasoning", experiment_revision_v2=False)
GEN, PATCH = "ExperimentGeneratorLLMOutput", "ExperimentRevisionPatchLLMOutput"
CRIT_V1, CRIT_V2, FOCUSED = "ExperimentCriticLLMOutput", "ExperimentCriticV2LLMOutput", "ExperimentFocusedCriticLLMOutput"
SENTINEL = "zq-sentinel-founder-phrase"
CONSENT_STEP = REMEDIATION_ACTIONS["CONSENT_STEP_MISSING"]
GUARDIAN_STEP = REMEDIATION_ACTIONS["GUARDIAN_CONSENT_MISSING"]
SHARP_SIGNALS = {
    "support_signal": "At least 4 of 6 coaches show a review session they skipped in the last 3 weeks because it took over 2 hours.",
    "contradiction_signal": "3 or more of 6 coaches review every match within a day and spend under 45 minutes.",
}


def crit(verdict="ACCEPT", fails=(), scope=None, issues=None, instructions="", **extra) -> dict:
    out = {
        "verdict": verdict,
        "issues": issues if issues is not None else ([] if verdict == "ACCEPT" else ["A blocking issue."]),
        **{d: ("FAIL" if d in fails else "PASS") for d in CRITIC_DIMENSIONS},
        "revision_instructions": instructions,
        "rationale": "Critic rationale.",
    }
    if scope is not None:
        out["revision_scope"] = scope
    out.update(extra)
    return out


def full(verdict="ACCEPT", fails=(), scope=None, **kw) -> dict:
    return crit(verdict, fails, scope or ("NONE" if verdict == "ACCEPT" else "TARGETED"), **kw)


def focused(verdict="ACCEPT", fails=(), resolved=True, new=None, **kw) -> dict:
    return crit(verdict, fails, original_issues_resolved=resolved, new_blocking_issues=new or [], **kw)


FALSIFIABILITY_REVISE = full("REVISE", fails=("falsifiability",), scope="TARGETED",
                             issues=["Support threshold is vague.", "No contradiction threshold."],
                             instructions="Add numeric support and contradiction thresholds.")
EVIDENCE_REDESIGN = full("REVISE", fails=("evidence_validity",), scope="REDESIGN",
                         issues=["Opinion interviews cannot show whether coaches change behaviour."],
                         instructions="Observe actual review behaviour instead of asking opinions.")


def seq(*items):
    it = iter(items)
    return lambda m, r: next(it)


@contextmanager
def offline(llm: ScriptedLLM, settings: Settings = V2):
    with patch("startary.graph.nodes.get_llm_client", side_effect=llm.get_llm_client), \
         patch("startary.graph.nodes.get_settings", return_value=settings):
        yield


def run(llm, thread, settings=V2, saver=None, callbacks=None, interrupt_after=None, **overrides):
    graph = build_graph(checkpointer=saver or MemorySaver())
    cfg = {"configurable": {"thread_id": thread}}
    if callbacks:
        cfg["callbacks"] = callbacks
    with offline(llm, settings):
        graph.update_state(cfg, loop_state(**overrides), as_node="decide_next_strategic_action")
        graph.invoke(None, cfg, interrupt_after=interrupt_after)
    return graph, cfg


def calls(llm) -> dict:
    return {k: llm.count(k) for k in (GEN, PATCH, CRIT_V1, CRIT_V2, FOCUSED)}


def critic_records(values) -> list[dict]:
    return [r.details for r in values["turn_telemetry"].records if r.node_name == "critique_experiment"]


# ---------------------------------------------------------------------------
# Pure rules
# ---------------------------------------------------------------------------

class TestStrategyRules:
    @pytest.mark.parametrize("source, failed, codes, scope, expected", [
        ("LLM", ["falsifiability"], [], "TARGETED", "TARGETED_EDIT"),
        ("LLM", ["evidence_validity", "falsifiability"], [], "TARGETED", "TARGETED_EDIT"),
        ("LLM", ["evidence_validity"], [], "REDESIGN", "FULL_REGENERATION"),
        ("LLM", ["assumption_alignment"], [], "TARGETED", "FULL_REGENERATION"),     # wrong thing tested
        ("LLM", ["falsifiability"], [], None, "FULL_REGENERATION"),                  # missing scope: conservative
        ("LLM", ["evidence_validity", "feasibility", "falsifiability", "scope"], [], "TARGETED", "FULL_REGENERATION"),
        ("DETERMINISTIC", [], ["COST_EXCEEDS_BUDGET"], None, "TARGETED_EDIT"),
        ("DETERMINISTIC", [], ["REQUIRED_FIELD_EMPTY", "INVALID_SAMPLE_SIZE"], None, "TARGETED_EDIT"),
        ("DETERMINISTIC", [], ["PROTOTYPE_NOT_AVAILABLE"], None, "FULL_REGENERATION"),
        ("DETERMINISTIC", [], ["ASSUMPTION_MISMATCH"], None, "FULL_REGENERATION"),
        ("DETERMINISTIC", [], ["COST_EXCEEDS_BUDGET", "CONSENT_STEP_MISSING"], None, "FULL_REGENERATION"),
    ])
    def test_strategy_table(self, source, failed, codes, scope, expected):
        assert select_revision_strategy(source, failed, codes, scope) == expected

    def test_substantial_change_rules(self):
        base = experiment()
        tweak = dict(base, support_signal=SHARP_SIGNALS["support_signal"])
        assert diff_experiment(base, tweak) == ["support_signal"]
        assert is_substantial_revision(base, tweak, ["support_signal"]) is False
        new_method = dict(base, method="Run a 2-week concierge analysis service for three coaches and log hours saved.")
        assert is_substantial_revision(base, new_method, diff_experiment(base, new_method)) is True
        new_type = dict(base, experiment_type="CONCIERGE_TEST")
        assert is_substantial_revision(base, new_type, ["experiment_type"]) is True

    def test_focused_mode_requires_light_targeted_edit_with_prior_critique(self):
        rev = {"draft_origin": "TARGETED_EDIT", "substantial_change": False, "rejected_draft": {"a": 1}, "rejected_critique": {"v": 1}}
        assert select_critic_mode(rev) == "FOCUSED"
        assert select_critic_mode({**rev, "substantial_change": True}) == "FULL"
        assert select_critic_mode({**rev, "draft_origin": "FULL_REGENERATION"}) == "FULL"
        assert select_critic_mode(None) == "FULL"

    def test_focused_acceptance_is_stricter_than_verdict_and_flags(self):
        ok = CriticDecision("ACCEPT", "ACCEPT", {d: "PASS" for d in CRITIC_DIMENSIONS}, [], False, "LLM")
        assert apply_focused_acceptance_rule(ok, {"original_issues_resolved": True, "new_blocking_issues": []}).verdict == "ACCEPT"
        assert apply_focused_acceptance_rule(ok, {"original_issues_resolved": False}).verdict == "REVISE"
        assert apply_focused_acceptance_rule(ok, {"original_issues_resolved": True, "new_blocking_issues": ["x"]}).verdict == "REVISE"

    def test_bounds_are_unchanged(self):
        assert MAX_SEMANTIC_REVISIONS == 1


# ---------------------------------------------------------------------------
# 1, 6. First-pass ACCEPT and full regeneration
# ---------------------------------------------------------------------------

def test_first_draft_accept_uses_one_generation_and_one_full_critique():
    llm = ScriptedLLM({GEN: lambda m, r: experiment(), CRIT_V2: lambda m, r: full("ACCEPT")})
    graph, cfg = run(llm, "v2-accept")
    values = graph.get_state(cfg).values
    assert calls(llm) == {GEN: 1, PATCH: 0, CRIT_V1: 0, CRIT_V2: 1, FOCUSED: 0}
    assert graph.get_state(cfg).next == ("wait_for_evidence",)
    (rec,) = critic_records(values)
    assert (rec["critic_mode"], rec["draft_origin"], rec["revision_mode"]) == ("FULL", "INITIAL", "V2")
    assert len(values["previous_experiments"]) == 1


def test_evidence_validity_redesign_triggers_full_regeneration_and_full_critique():
    llm = ScriptedLLM({
        GEN: seq(experiment(), experiment(method="Observe each coach's next post-match review session and log the time spent.")),
        CRIT_V2: seq(EVIDENCE_REDESIGN, full("ACCEPT")),
    })
    graph, cfg = run(llm, "v2-redesign")
    values = graph.get_state(cfg).values
    assert calls(llm) == {GEN: 2, PATCH: 0, CRIT_V1: 0, CRIT_V2: 2, FOCUSED: 0}
    first, second = critic_records(values)
    assert first["revision_strategy"] == "FULL_REGENERATION" and first["next_route"] == "generate_experiment"
    assert (second["critic_mode"], second["draft_origin"], second["original_failed_dimensions"]) == ("FULL", "FULL_REGENERATION", ["evidence_validity"])
    assert "Observe actual review behaviour" in llm.calls_for(GEN)[1].messages[-1]["content"]


# ---------------------------------------------------------------------------
# 2, 3, 5. Targeted revision and focused re-critique
# ---------------------------------------------------------------------------

def test_targeted_edit_fixes_falsifiability_preserves_other_fields_and_uses_focused_critic():
    llm = ScriptedLLM({
        GEN: lambda m, r: experiment(),
        CRIT_V2: lambda m, r: FALSIFIABILITY_REVISE,
        PATCH: lambda m, r: dict(SHARP_SIGNALS),
        FOCUSED: lambda m, r: focused("ACCEPT"),
    })
    graph, cfg = run(llm, "v2-targeted")
    values = graph.get_state(cfg).values
    assert calls(llm) == {GEN: 1, PATCH: 1, CRIT_V1: 0, CRIT_V2: 1, FOCUSED: 1}
    assert graph.get_state(cfg).next == ("wait_for_evidence",)
    original, saved = experiment(), values["active_experiment"]
    for unchanged in ("method", "founder_actions", "evidence_to_collect", "target_participants_or_scope", "inconclusive_signal", "valid_inferences"):
        assert saved[unchanged] == original[unchanged], unchanged
    assert saved["support_signal"] == SHARP_SIGNALS["support_signal"]
    # The focused critic sees the original critique, the change list, and re-evaluates all six dimensions.
    focused_msg = llm.calls_for(FOCUSED)[0].messages[-1]["content"]
    assert "Support threshold is vague." in focused_msg and '"field": "support_signal"' in focused_msg
    assert "falsifiability=FAIL" in focused_msg
    assert "Do not assume a dimension still passes" in llm.calls_for(FOCUSED)[0].messages[0]["content"]
    # The reviser received the rejected draft, the failed dimensions, issues, and founder facts.
    patch_msg = llm.calls_for(PATCH)[0].messages[-1]["content"]
    assert original["method"] in patch_msg and "Failed dimensions: falsifiability" in patch_msg and "1500" in patch_msg
    first, second = critic_records(values)
    assert first["revision_strategy"] == "TARGETED_EDIT" and first["next_route"] == "revise_experiment"
    assert (second["critic_mode"], second["draft_origin"]) == ("FOCUSED", "TARGETED_EDIT")
    assert second["changed_fields"] == ["support_signal", "contradiction_signal"]
    assert (second["original_failed_dimensions"], second["remaining_failed_dimensions"]) == (["falsifiability"], [])
    assert second["original_issues_resolved"] is True


@pytest.mark.parametrize("focused_output, reason", [
    (focused("ACCEPT", new=["The new threshold needs a paid ad campaign the founders cannot fund."]), "new blocking issue"),
    (focused("REVISE", fails=("feasibility",), issues=["Now requires 20 coaches."]), "regressed dimension"),
    (focused("ACCEPT", resolved=False), "original issue unresolved"),
], ids=["new-blocking", "regression", "unresolved"])
def test_focused_critique_rejects_regressions_and_loop_ends_bounded(focused_output, reason):
    llm = ScriptedLLM({GEN: lambda m, r: experiment(), CRIT_V2: lambda m, r: FALSIFIABILITY_REVISE,
                       PATCH: lambda m, r: dict(SHARP_SIGNALS), FOCUSED: lambda m, r: focused_output})
    graph, cfg = run(llm, f"v2-regression-{reason[:5]}")
    snap = graph.get_state(cfg)
    assert snap.next == ("ask_and_wait_for_founder",)                       # failure fallback reachable
    assert snap.values["experiment_design_status"] == "QUALITY_BAR_NOT_MET"
    assert calls(llm) == {GEN: 1, PATCH: 1, CRIT_V1: 0, CRIT_V2: 1, FOCUSED: 1}
    assert critic_records(snap.values)[1]["critic_verdict"] == "REVISE"


def test_substantial_targeted_edit_gets_a_full_critique():
    llm = ScriptedLLM({GEN: lambda m, r: experiment(), CRIT_V2: seq(FALSIFIABILITY_REVISE, full("ACCEPT")),
                       PATCH: lambda m, r: {"method": "Run a two-week concierge analysis service for three coaches and log hours saved."},
                       FOCUSED: lambda m, r: focused("ACCEPT")})
    graph, cfg = run(llm, "v2-substantial")
    assert calls(llm) == {GEN: 1, PATCH: 1, CRIT_V1: 0, CRIT_V2: 2, FOCUSED: 0}
    assert critic_records(graph.get_state(cfg).values)[1]["substantial_change"] is True


def test_targeted_edit_with_no_change_is_rejected_without_a_critic_call():
    llm = ScriptedLLM({GEN: lambda m, r: experiment(), CRIT_V2: lambda m, r: FALSIFIABILITY_REVISE,
                       PATCH: lambda m, r: {}, FOCUSED: lambda m, r: focused("ACCEPT")})
    graph, cfg = run(llm, "v2-noop")
    values = graph.get_state(cfg).values
    assert calls(llm)[FOCUSED] == 0
    assert critic_records(values)[1]["revision_reason_codes"] == ["NO_EFFECTIVE_CHANGE"]
    assert values["experiment_design_status"] == "QUALITY_BAR_NOT_MET"


# ---------------------------------------------------------------------------
# 7–12. Deterministic remediation and privacy safeguards
# ---------------------------------------------------------------------------

FILMED_NO_CONSENT = dict(
    method="Film two matches per team and review the footage with each coach.",
    target_participants_or_scope="Three coaches and their two amateur teams",
    founder_actions=["Agree match dates with the three coaches.", "Film two matches per team.", "Review the footage with each coach."],
)


def test_consent_remediation_inserts_step_once_before_recording_and_keeps_permission_pending():
    llm = ScriptedLLM({GEN: lambda m, r: experiment(**FILMED_NO_CONSENT), CRIT_V2: lambda m, r: full("ACCEPT")})
    graph, cfg = run(llm, "v2-remediate")
    values = graph.get_state(cfg).values
    actions = values["active_experiment"]["founder_actions"]
    assert actions.count(CONSENT_STEP) == 1
    assert actions == [FILMED_NO_CONSENT["founder_actions"][0], CONSENT_STEP] + FILMED_NO_CONSENT["founder_actions"][1:]
    assert calls(llm) == {GEN: 1, PATCH: 0, CRIT_V1: 0, CRIT_V2: 1, FOCUSED: 0}   # no generation spent on it
    assert values["critic_revision_count"] == 0                                     # semantic allowance untouched
    # Adding a permission step does not mean permission exists.
    exp = values["active_experiment"]
    assert (exp["authorization_status"], exp["required_authorizations"]) == ("PENDING", ["RECORDING_PERMISSION"])
    assert pending_authorizations(exp) == ["RECORDING_PERMISSION"]
    assert "not yet obtained" in values["can_execute_result"]["reason"]
    assert values["execution_handoff_text"].startswith("BEFORE YOU START")
    rem = [r.details for r in values["turn_telemetry"].records if r.node_name == "validate_proposed_experiment"]
    assert rem[0]["remediation_codes"] == ["CONSENT_STEP_MISSING"] and rem[0]["revision_strategy"] == "DETERMINISTIC_REMEDIATION"
    assert critic_records(values)[0]["remediation_applied"] is True


def test_remediation_is_idempotent():
    state = loop_state(proposed_experiment_output=experiment(**FILMED_NO_CONSENT))
    with offline(ScriptedLLM({})):
        once = validate_proposed_experiment(state)
        twice = validate_proposed_experiment({**state, **once})
    assert once["proposed_experiment_output"]["founder_actions"] == twice["proposed_experiment_output"]["founder_actions"]
    assert twice["proposed_experiment_output"]["founder_actions"].count(CONSENT_STEP) == 1


def test_guardian_safeguard_for_actual_minors_is_remediated_and_stays_pending():
    minors = dict(method="Film two U14 academy matches and review the footage with each coach.",
                  founder_actions=["Get the club's permission to film.", "Film the matches.", "Review the footage with the coaches."])
    llm = ScriptedLLM({GEN: lambda m, r: experiment(**minors), CRIT_V2: lambda m, r: full("ACCEPT")})
    graph, cfg = run(llm, "v2-minors")
    exp = graph.get_state(cfg).values["active_experiment"]
    assert exp["founder_actions"].count(GUARDIAN_STEP) == 1
    assert exp["founder_actions"].index(GUARDIAN_STEP) < exp["founder_actions"].index("Film the matches.")
    assert "GUARDIAN_AUTHORIZATION" in exp["required_authorizations"] and exp["authorization_status"] == "PENDING"


def test_covert_recording_of_minors_is_not_remediated():
    covert = dict(method="Secretly film U14 training sessions to see how coaches review play.",
                  founder_actions=["Film the U14 training from the stands."])
    llm = ScriptedLLM({GEN: seq(experiment(**covert), experiment(**covert)), CRIT_V2: lambda m, r: full("ACCEPT")})
    graph, cfg = run(llm, "v2-covert")
    values = graph.get_state(cfg).values
    assert calls(llm)[CRIT_V2] == 0                                   # never reaches the LLM critic
    assert values["experiment_design_status"] == "QUALITY_BAR_NOT_MET"
    rem = [r.details for r in values["turn_telemetry"].records if r.node_name == "validate_proposed_experiment"]
    assert rem[0]["remediation_outcome"] == "UNREMEDIABLE_METHOD"
    assert critic_records(values)[0]["revision_strategy"] == "FULL_REGENERATION"


def test_adult_coach_interviews_do_not_trigger_guardian_requirements():
    adults = dict(method="Interview the three coaches and record the calls.",
                  founder_actions=["Ask each coach for permission to record the call.", "Record the calls."])
    llm = ScriptedLLM({GEN: lambda m, r: experiment(**adults), CRIT_V2: lambda m, r: full("ACCEPT")})
    graph, cfg = run(llm, "v2-adults", raw_context="Long term we sell to youth football academies.")
    exp = graph.get_state(cfg).values["active_experiment"]
    assert GUARDIAN_STEP not in exp["founder_actions"]
    assert exp["required_authorizations"] == ["RECORDING_PERMISSION"]


def test_unknown_ages_are_not_turned_into_facts():
    llm = ScriptedLLM({GEN: lambda m, r: experiment(**FILMED_NO_CONSENT), CRIT_V2: lambda m, r: full("ACCEPT")})
    graph, cfg = run(llm, "v2-unknown-ages", raw_context="Long term we sell to youth football academies.")
    exp = graph.get_state(cfg).values["active_experiment"]
    assert GUARDIAN_STEP not in exp["founder_actions"]                # no invented minors
    assert "GUARDIAN_AUTHORIZATION" not in exp["required_authorizations"]
    assert not any("adult" in a.lower() for a in exp["founder_actions"])  # no invented adults either
    critic_msg = llm.calls_for(CRIT_V2)[0].messages[-1]["content"]
    assert "Confirm everyone recorded is 18 or older" in critic_msg   # ambiguity handed to the critic


@pytest.mark.parametrize("overrides, code", [
    ({"estimated_cost": "$5,000"}, "COST_EXCEEDS_BUDGET"),
    ({"experiment_type": "PROTOTYPE_TEST"}, "PROTOTYPE_NOT_AVAILABLE"),
])
def test_budget_and_prototype_violations_are_never_auto_repaired(overrides, code):
    state = loop_state(proposed_experiment_output=experiment(**overrides), build_readiness={"decision": "CONTINUE_MANUAL"})
    with offline(ScriptedLLM({})):
        out = validate_proposed_experiment(state)
    assert out["proposed_experiment_output"]["founder_actions"] == experiment()["founder_actions"]
    assert out["turn_telemetry"].records[-1].details["remediation_outcome"] == "NON_REMEDIABLE_VIOLATION_PRESENT"
    assert evaluate_experiment_contract(out["proposed_experiment_output"], state).violation_codes == [code]


def test_mixed_consent_and_budget_violations_are_not_partially_repaired():
    result = remediate_consent_violations(
        experiment(**FILMED_NO_CONSENT, estimated_cost="$5,000"),
        evaluate_experiment_contract(experiment(**FILMED_NO_CONSENT, estimated_cost="$5,000"), loop_state()),
    )
    assert not result.applied and result.skipped_reason == "NON_REMEDIABLE_VIOLATION_PRESENT"


def test_budget_violation_gets_a_targeted_edit_with_the_finding():
    llm = ScriptedLLM({GEN: lambda m, r: experiment(estimated_cost="$5,000"), CRIT_V2: lambda m, r: full("ACCEPT"),
                       PATCH: lambda m, r: {"estimated_cost": "$0 (founder time only)"}, FOCUSED: lambda m, r: focused("ACCEPT")})
    graph, cfg = run(llm, "v2-budget")
    assert graph.get_state(cfg).next == ("wait_for_evidence",)
    assert "exceeds the founder's available budget" in llm.calls_for(PATCH)[0].messages[-1]["content"]
    assert calls(llm) == {GEN: 1, PATCH: 1, CRIT_V1: 0, CRIT_V2: 0, FOCUSED: 1}


# ---------------------------------------------------------------------------
# 13–16. Bounds, fallback, checkpoint restoration, duplicate resume
# ---------------------------------------------------------------------------

def test_always_rejecting_critic_is_bounded_and_falls_back():
    llm = ScriptedLLM({GEN: lambda m, r: experiment(), CRIT_V2: lambda m, r: FALSIFIABILITY_REVISE,
                       PATCH: lambda m, r: dict(SHARP_SIGNALS), FOCUSED: lambda m, r: focused("REVISE", fails=("falsifiability",))})
    graph, cfg = run(llm, "v2-bounded")
    snap = graph.get_state(cfg)
    assert calls(llm) == {GEN: 1, PATCH: 1, CRIT_V1: 0, CRIT_V2: 1, FOCUSED: 1}
    assert snap.next == ("ask_and_wait_for_founder",) and snap.values["fallback_learning_action"]


def test_revision_state_survives_checkpoint_restoration_and_saves_once():
    llm = ScriptedLLM({GEN: lambda m, r: experiment(), CRIT_V2: lambda m, r: FALSIFIABILITY_REVISE,
                       PATCH: lambda m, r: dict(SHARP_SIGNALS), FOCUSED: lambda m, r: focused("ACCEPT")})
    saver = MemorySaver()
    graph, cfg = run(llm, "v2-restore", saver=saver, interrupt_after=["revise_experiment"])
    restored = build_graph(checkpointer=saver)
    rev = restored.get_state(cfg).values["experiment_revision"]
    assert rev["draft_origin"] == "TARGETED_EDIT" and rev["changed_fields"] == ["support_signal", "contradiction_signal"]
    assert rev["rejected_draft"]["support_signal"] == experiment()["support_signal"]   # original preserved
    assert rev["rejected_critique"]["failed_dimensions"] == ["falsifiability"]
    assert restored.get_state(cfg).values["critic_revision_count"] == 1
    with offline(llm):
        restored.invoke(None, cfg)
    values = restored.get_state(cfg).values
    assert calls(llm) == {GEN: 1, PATCH: 1, CRIT_V1: 0, CRIT_V2: 1, FOCUSED: 1}
    assert len(values["previous_experiments"]) == 1
    assert values["previous_experiments"][0]["id"] == values["active_experiment"]["id"]


def test_duplicate_resume_does_not_duplicate_remediation_or_llm_calls():
    llm = ScriptedLLM({GEN: lambda m, r: experiment(**FILMED_NO_CONSENT), CRIT_V2: lambda m, r: full("ACCEPT")})
    graph, cfg = run(llm, "v2-dup-resume", interrupt_after=["validate_proposed_experiment"])
    with offline(llm):
        graph.invoke(None, cfg)          # continue to the evidence wait
        before = dict(calls(llm))
        graph.invoke(None, cfg)          # duplicate resume while waiting for evidence
    values = graph.get_state(cfg).values
    assert calls(llm) == before
    assert values["active_experiment"]["founder_actions"].count(CONSENT_STEP) == 1
    assert len(values["previous_experiments"]) == 1


# ---------------------------------------------------------------------------
# 17–19. Feature flag and metadata privacy
# ---------------------------------------------------------------------------

def test_flag_off_preserves_m6251_behaviour():
    from tests.test_m625_experiment_loop_correctness import DECISION_KEYS
    llm = ScriptedLLM({GEN: lambda m, r: experiment(**FILMED_NO_CONSENT),
                       CRIT_V1: lambda m, r: crit("ACCEPT")})
    graph, cfg = run(llm, "v1-off", settings=V1)
    values = graph.get_state(cfg).values
    assert calls(llm)[CRIT_V1] == 0 and calls(llm)[GEN] == 2           # consent violation -> deterministic REVISE x2, no remediation
    assert values["experiment_design_status"] == "QUALITY_BAR_NOT_MET"
    assert values.get("experiment_revision") is None
    assert all(set(d) == DECISION_KEYS for d in critic_records(values))
    assert "required_authorizations" not in (values["proposed_experiment_output"] or {})


def test_flag_off_revise_still_regenerates_with_the_original_critic():
    llm = ScriptedLLM({GEN: lambda m, r: experiment(), CRIT_V1: seq(crit("REVISE", fails=("falsifiability",)), crit("ACCEPT"))})
    graph, cfg = run(llm, "v1-revise", settings=V1)
    assert calls(llm) == {GEN: 2, PATCH: 0, CRIT_V1: 2, CRIT_V2: 0, FOCUSED: 0}
    assert graph.get_state(cfg).next == ("wait_for_evidence",)


def test_flag_on_uses_v2_schemas_and_records_v2_metadata():
    llm = ScriptedLLM({GEN: lambda m, r: experiment(), CRIT_V2: lambda m, r: full("ACCEPT")})
    graph, cfg = run(llm, "v2-flag")
    assert calls(llm)[CRIT_V1] == 0 and calls(llm)[CRIT_V2] == 1
    assert "REVISION SCOPE" in llm.calls_for(CRIT_V2)[0].messages[0]["content"]
    assert critic_records(graph.get_state(cfg).values)[0]["revision_mode"] == "V2"


def test_v2_metadata_has_no_sensitive_text_and_survives_sanitization():
    recorder = SpanRecorder()
    sensitive_revise = full("REVISE", fails=("falsifiability",), scope="TARGETED",
                            issues=[f"Threshold vague for {SENTINEL}."], instructions=f"Fix it {SENTINEL}.")
    llm = ScriptedLLM({
        GEN: lambda m, r: experiment(**FILMED_NO_CONSENT, objective=f"Learn about {SENTINEL} coaches."),
        CRIT_V2: lambda m, r: sensitive_revise,
        PATCH: lambda m, r: {"support_signal": f"At least 4 of 6 coaches ({SENTINEL}) skipped a review."},
        FOCUSED: lambda m, r: focused("ACCEPT", new=[]),
    })
    graph, cfg = run(llm, "v2-privacy", callbacks=[recorder])
    values = graph.get_state(cfg).values
    details = [r.details for r in values["turn_telemetry"].records if r.details]
    span_meta = [r["metadata"] for r in recorder.runs.values()
                 if r["name"] in ("experiment_critic_decision", "experiment_targeted_revision", "experiment_remediation")]
    assert {"experiment_critic_decision", "experiment_targeted_revision", "experiment_remediation"} <= {
        r["name"] for r in recorder.runs.values()}
    # What Startary emits (telemetry details == span payload before LangGraph adds its own run keys).
    for meta in details:
        assert sanitize_metadata(meta) == meta
    assert all(m.get("revision_mode") == "V2" for m in span_meta)
    assert SENTINEL not in json.dumps(details + span_meta, default=str)


def test_revision_prompt_stays_grounded_in_pitchvision_facts():
    llm = ScriptedLLM({GEN: lambda m, r: experiment(), CRIT_V2: lambda m, r: FALSIFIABILITY_REVISE,
                       PATCH: lambda m, r: dict(SHARP_SIGNALS), FOCUSED: lambda m, r: focused("ACCEPT")})
    graph, cfg = run(llm, "v2-grounding")
    system, user = (m["content"] for m in llm.calls_for(PATCH)[0].messages)
    assert "Direct access to two amateur football teams and three coaches" in user
    assert '"paying_customers": "0"' in user and '"mvp_exists": "false"' in user
    assert "CONTINUE_MANUAL" in user and "does not mean permission exists" in system
    assert ASSUMPTION["statement"] in user
    assert graph.get_state(cfg).values["active_experiment"]["hypothesis"] == ASSUMPTION["statement"]
