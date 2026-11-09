"""
Milestone 6.2.5.1 — Guardian-consent false positive.

Live PitchVision trace: critique 1 (LLM) REVISE on evidence_validity + falsifiability; critique 2
deterministic REVISE with only GUARDIAN_CONSENT_MISSING (so the draft recorded people and already
had a consent step) -> handle_experiment_failure_stop.

Root cause (confirmed in code): check_privacy_consent treated minors as "known" whenever ANY youth
word appeared in the founder's whole conversation or anywhere in the experiment text, so recording
an interview with an adult coach of a startup whose market is "youth football academies" required
guardian authorization. The guardian safeguard now applies when the experiment itself involves minors.
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import patch

import pytest
from langgraph.checkpoint.memory import MemorySaver

from startary.config import Settings
from startary.domain.experiment_contract import ContractCode, check_privacy_consent, evaluate_experiment_contract
from startary.graph.builder import build_graph
from tests.fixtures.scenarios import make_valid_experiment_critic_output_dict, make_valid_experiment_generator_output_dict
from tests.fixtures.scripted_llm import ScriptedLLM
from tests.test_m625_experiment_loop_correctness import ASSUMPTION, loop_state

OFFLINE_SETTINGS = Settings(_env_file=None, openai_api_key="sk-offline-test", fast_model_name="scripted-fast", reasoning_model_name="scripted-reasoning")
GEN, CRIT = "ExperimentGeneratorLLMOutput", "ExperimentCriticLLMOutput"
YOUTH_MARKET_CONTEXT = "PitchVision AI analyses amateur football. Long term we want to sell to youth football academies."
ADULT_ACCESS = "Direct access to two amateur football teams and three coaches who agreed to test"


def exp(**fields) -> dict:
    base = make_valid_experiment_generator_output_dict(hypothesis=ASSUMPTION["statement"])
    base.update(fields)
    return base


def findings(experiment: dict, founder_context: str = "", participant_access: str = "") -> list[tuple[str, str, str]]:
    return [(f.code.value, f.severity.value, f.basis) for f in check_privacy_consent(experiment, founder_context, participant_access)]


RECORDED_COACH_INTERVIEWS = exp(
    method="Interview the three coaches about last week's match review and record the interviews.",
    target_participants_or_scope="Three amateur football coaches",
    founder_actions=["Ask each coach for permission to record the call.", "Record the interviews.", "Delete recordings after notes are written."],
)
FILMED_U14_MATCHES = exp(
    method="Film two U14 academy matches and review the footage with each coach.",
    founder_actions=["Get the club's permission to film.", "Film the matches.", "Review the footage with the coaches."],
)
FILMED_TEAM_MATCHES = exp(
    method="Film two matches per team and review the footage with each coach.",
    founder_actions=["Get each team's permission to film.", "Film the matches."],
)


# ---------------------------------------------------------------------------
# Unit cases: adult interviews vs. recording minors vs. minors' data vs. youth as a market
# ---------------------------------------------------------------------------

class TestAdultOnlyInterviews:
    def test_recorded_coach_interviews_with_youth_market_context_are_not_a_guardian_violation(self):
        """The live-run false-positive class: adults recorded, youth only in founder context."""
        result = findings(RECORDED_COACH_INTERVIEWS, YOUTH_MARKET_CONTEXT, ADULT_ACCESS)
        assert ("GUARDIAN_CONSENT_MISSING", "VIOLATION", "EXPERIMENT_PARTICIPANTS") not in result
        assert all(severity == "ADVISORY" for _, severity, _ in result)
        assert result == [("POSSIBLE_MINORS", "ADVISORY", "YOUTH_MENTION")]

    def test_youth_academies_named_as_market_in_experiment_text(self):
        e = exp(
            objective="Test whether coaches at youth football academies lose hours to post-match review.",
            method="Interview three coaches and record the interviews with their permission.",
            founder_actions=["Get each coach's consent to record."],
        )
        assert [c for c, s, _ in findings(e) if s == "VIOLATION"] == []

    def test_coaches_of_youth_teams_are_adult_participants(self):
        e = exp(target_participants_or_scope="Coaches of youth teams in the local league",
                method="Interview five coaches of youth teams; record the calls with permission.",
                founder_actions=["Ask permission to record each call."])
        assert [c for c, s, _ in findings(e) if s == "VIOLATION"] == []

    def test_explicitly_adult_coaches_need_no_minors_advisory(self):
        e = exp(target_participants_or_scope="Three adult amateur coaches",
                method="Record interviews with three coaches.", founder_actions=["Ask each coach for consent to record."])
        assert findings(e, YOUTH_MARKET_CONTEXT) == []

    def test_unrecorded_coach_interviews_have_no_privacy_findings(self):
        assert findings(make_valid_experiment_generator_output_dict(), YOUTH_MARKET_CONTEXT, ADULT_ACCESS) == []

    def test_recording_consent_is_still_required_for_adults(self):
        e = exp(method="Record interviews with three coaches.", founder_actions=["Record the interviews."],
                target_participants_or_scope="Three adult coaches")
        assert findings(e) == [("CONSENT_STEP_MISSING", "VIOLATION", "EXPERIMENT_TEXT")]


class TestRecordingMinors:
    def test_filming_named_youth_players_requires_guardian_authorization(self):
        assert findings(FILMED_U14_MATCHES) == [("GUARDIAN_CONSENT_MISSING", "VIOLATION", "EXPERIMENT_TEXT")]

    def test_guardian_step_satisfies_the_safeguard(self):
        e = dict(FILMED_U14_MATCHES, founder_actions=FILMED_U14_MATCHES["founder_actions"] + ["Collect written parental consent via the club for every player."])
        assert findings(e) == []

    def test_no_consent_at_all_raises_both_violations(self):
        e = exp(method="Film two U14 academy matches.", founder_actions=["Film the matches."])
        assert {c for c, _, _ in findings(e)} == {"CONSENT_STEP_MISSING", "GUARDIAN_CONSENT_MISSING"}

    def test_filming_teams_the_founder_documented_as_youth_requires_guardian(self):
        result = findings(FILMED_TEAM_MATCHES, "", "Two U12 and U14 club squads and their three coaches")
        assert result == [("GUARDIAN_CONSENT_MISSING", "VIOLATION", "PARTICIPANT_ACCESS")]

    def test_experiment_stating_adults_overrides_youth_access(self):
        e = dict(FILMED_TEAM_MATCHES, method="Film two matches of the adult first team and review the footage.")
        assert [c for c, s, _ in findings(e, "", "Two U14 squads and the adult first team")] == []

    def test_minors_as_direct_participants_without_recording(self):
        e = exp(target_participants_or_scope="Ten academy players aged 13-15", method="Run a 10-minute survey after training.",
                founder_actions=["Ask the players three questions after training."])
        assert findings(e) == [("GUARDIAN_CONSENT_MISSING", "VIOLATION", "EXPERIMENT_PARTICIPANTS")]


class TestIdentifiableDataAboutMinors:
    def test_collecting_youth_players_identifiable_data_requires_guardian(self):
        e = exp(method="Collect the U14 players' names and dates of birth from the club roster to tag clips.",
                founder_actions=["Export the roster."])
        assert findings(e) == [("GUARDIAN_CONSENT_MISSING", "VIOLATION", "EXPERIMENT_TEXT")]

    def test_collecting_adult_coach_contacts_is_advisory_only(self):
        e = exp(method="Collect email addresses of interested coaches at the end of each interview.")
        assert findings(e, YOUTH_MARKET_CONTEXT) == [("PERSONAL_DATA_PRIVACY", "ADVISORY", "EXPERIMENT_TEXT")]


class TestAmbiguousAges:
    def test_filming_matches_with_unstated_ages_is_advisory(self):
        assert findings(FILMED_TEAM_MATCHES, "", ADULT_ACCESS) == [("POSSIBLE_MINORS", "ADVISORY", "UNSTATED_AGES")]

    def test_filming_matches_with_youth_market_context_is_advisory_not_violation(self):
        assert findings(FILMED_TEAM_MATCHES, YOUTH_MARKET_CONTEXT, ADULT_ACCESS) == [("POSSIBLE_MINORS", "ADVISORY", "YOUTH_MENTION")]

    def test_advisory_reaches_the_critic_as_a_note(self):
        state = loop_state(raw_context=YOUTH_MARKET_CONTEXT)
        contract = evaluate_experiment_contract(FILMED_TEAM_MATCHES, state)
        assert contract.violation_codes == []
        assert "POSSIBLE_MINORS" in contract.advisory_codes
        assert "POSSIBLE_MINORS:YOUTH_MENTION" in contract.finding_bases


# ---------------------------------------------------------------------------
# Graph-level replay of the live failure sequence (real graph + MemorySaver)
# ---------------------------------------------------------------------------

@contextmanager
def offline(llm):
    with patch("startary.graph.nodes.get_llm_client", side_effect=llm.get_llm_client), \
         patch("startary.graph.nodes.get_settings", return_value=OFFLINE_SETTINGS):
        yield


def _replay(second_draft: dict, thread: str):
    """Round 1: LLM REVISE failing evidence_validity + falsifiability (as measured). Round 2: `second_draft`."""
    first_critique = make_valid_experiment_critic_output_dict("REVISE") | {
        "assumption_alignment": "PASS", "evidence_validity": "FAIL", "falsifiability": "FAIL",
        "issues": ["Signals rely on opinions.", "No falsification threshold."],
    }
    drafts = iter([exp(), second_draft])
    critiques = iter([first_critique, make_valid_experiment_critic_output_dict("ACCEPT")])
    llm = ScriptedLLM({GEN: lambda m, r: next(drafts), CRIT: lambda m, r: next(critiques)})
    state = loop_state(
        raw_context=YOUTH_MARKET_CONTEXT,
        founder_messages=[{"role": "founder", "content": YOUTH_MARKET_CONTEXT}],
    )
    with offline(llm):
        graph = build_graph(checkpointer=MemorySaver())
        cfg = {"configurable": {"thread_id": thread}}
        graph.update_state(cfg, state, as_node="decide_next_strategic_action")
        graph.invoke(None, cfg)
        snap = graph.get_state(cfg)
    decisions = [r.details for r in snap.values["turn_telemetry"].records if r.node_name == "critique_experiment"]
    return llm, snap, decisions


def test_adult_interview_redraft_now_reaches_critic_and_is_accepted():
    llm, snap, decisions = _replay(RECORDED_COACH_INTERVIEWS, "pv-guardian-false-positive")
    assert llm.count(CRIT) == 2                                   # round 2 judged by the critic, not short-circuited
    assert snap.next == ("wait_for_evidence",)
    assert decisions[1]["critic_verdict"] == "ACCEPT"
    assert decisions[1]["contract_violation_codes"] == []
    assert "POSSIBLE_MINORS:YOUTH_MENTION" in decisions[1]["contract_finding_bases"]
    assert "confirm everyone recorded is 18 or older" in llm.calls_for(CRIT)[1].messages[-1]["content"].lower()


def test_genuine_minor_recording_still_fails_closed_with_recorded_basis():
    llm, snap, decisions = _replay(FILMED_U14_MATCHES, "pv-guardian-genuine")
    assert llm.count(CRIT) == 1                                   # round 2 is a deterministic REVISE
    assert snap.next == ("ask_and_wait_for_founder",)             # failure stop -> learning input
    assert snap.values["experiment_design_status"] == "QUALITY_BAR_NOT_MET"
    assert decisions[1]["contract_violation_codes"] == ["GUARDIAN_CONSENT_MISSING"]
    assert decisions[1]["contract_finding_bases"] == ["GUARDIAN_CONSENT_MISSING:EXPERIMENT_TEXT"]
    assert decisions[1]["revision_trigger"] == "DETERMINISTIC"
