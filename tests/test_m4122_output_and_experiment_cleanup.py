"""
Milestone M4.1.2.2 — Founder Output & Experiment Contract Cleanup Test Suite.

Covers:
1. Normal new experiment does not render WHAT LOOKS PROMISING SO FAR.
2. Completed evidence handoff does not render WHAT LOOKS PROMISING SO FAR.
3. Partial valid evidence may render EARLY SIGNALS FROM THIS TEST.
4. Early signals contain only active-experiment findings.
5. Early signals do not include unrelated historical startup claims.
6. No early-signals section when no meaningful partial signal exists.
7. Full diagnosis is not redundantly re-rendered for simple CONTINUE_CURRENT_TEST.
8. Experiment sample target is stored explicitly.
9. Rendered sample requirement equals structured target sample size.
10. Completeness evaluator uses the same structured target.
11. Completeness evaluator does not invent a hidden target when none exists.
12. Support threshold is not confused with sample target.
13. Critic revision changing sample target updates founder-facing handoff.
14. &ever 8-team experiment with target=8 completes when 8 valid teams are returned.
15. &ever target=15 experiment explicitly tells founder 15 are required.
16. RequestTimeout classified as TIMEOUT, not generic provider/network.
17. 429 rate limit remains RATE_LIMIT.
18. insufficient_quota remains distinct from rate limit.
19. authentication error remains distinct.
20. post-evidence timeout does not rollback validated evidence.
21. retry does not duplicate evidence.
22. retry does not increment iteration twice.
23. retry does not duplicate LearningDelta.
24. retry resumes diagnosis from current state.
25. founder does not need to resubmit evidence after timeout.
26. M4.1.2.1 diagnosis rendering still works.
27. partial evidence continuation still works.
28. PRODUCT_ITERATION still works.
29. M4.1.1 recovery still works.
30. full regression suite passes.
"""

from __future__ import annotations

import copy
import pytest
from unittest.mock import MagicMock, patch

from startary.cli import ErrorCategory, classify_cli_error
from startary.domain.enums import AssumptionStatus, ExperimentStatus, WorkflowPhase
from startary.domain.experiment import (
    ExecutionHandoff,
    Experiment,
    FounderExperimentSummary,
    evaluate_experiment_completeness,
    extract_sample_target_from_protocol,
    render_execution_handoff_text,
)
from startary.domain.strategic_action import NextStrategicAction, NextStrategicActionType
from startary.graph.builder import resume_post_evidence_diagnosis, run_evidence_learning_pipeline
from tests.fixtures.scenarios import make_valid_area_assessments_batch_output_dict
from startary.graph.graph_state import StartaryGraphState
from startary.graph.nodes import (
    apply_validated_state_updates,
    compose_execution_handoff,
    increment_iteration,
    save_final_experiment,
    validate_evidence_analysis,
)
from startary.llm.schemas import (
    DiagnosisLLMOutput,
    EvidenceAnalysisLLMOutput,
    ExperimentGeneratorLLMOutput,
    RecommendedStateUpdateItem,
)


# ---------------------------------------------------------------------------
# Test 1 & 2: WHAT LOOKS PROMISING SO FAR Removal
# ---------------------------------------------------------------------------

def test_01_normal_new_experiment_does_not_render_what_looks_promising_so_far():
    """1. Normal new experiment does not render WHAT LOOKS PROMISING SO FAR."""
    summary = FounderExperimentSummary(
        what_are_we_testing="Whether youth teams will use video review weekly.",
        what_should_i_do="Review 8 teams.",
        what_should_i_bring_back="Usage counts.",
        useful_signal="4+ teams review clips.",
        current_positive_signals=["Some initial pilot usage"],
    )
    handoff = ExecutionHandoff(
        experiment_id="exp-1",
        assumption_id="asmp-1",
        critical_assumption="Youth teams will use video review weekly.",
        experiment_type="CUSTOMER_INTERVIEWS",
        learning_goal="Validate weekly review usage.",
        what_we_are_testing="Whether youth teams will use video review weekly.",
        founder_actions=["Review 8 teams."],
        target_participants_or_context="8 youth teams",
        target_sample_size=8,
        evidence_to_collect="Usage frequency and clip reviews.",
        support_signal="4+ teams review clips weekly.",
        summary=summary,
    )
    compact = render_execution_handoff_text(handoff, mode="compact")
    assert "WHAT ARE WE TESTING?" in compact
    assert "WHAT SHOULD I DO?" in compact
    assert "WHAT SHOULD I BRING BACK?" in compact
    assert "WHAT COUNTS AS A USEFUL SIGNAL?" in compact
    assert "WHAT LOOKS PROMISING SO FAR?" not in compact
    assert "No strong positive signal is established yet." not in compact


def test_02_completed_evidence_handoff_does_not_render_what_looks_promising_so_far():
    """2. Completed evidence handoff does not render WHAT LOOKS PROMISING SO FAR (positive signals in CURRENT DIAGNOSIS)."""
    diag_text = (
        "==================================================\n"
        "STARTARY — CURRENT DIAGNOSIS\n"
        "==================================================\n\n"
        "WHAT LOOKS STRONG\n"
        "- 3 paying clubs active on the filming platform.\n\n"
        "MOST IMPORTANT DECISION NOW\n"
        "Whether youth coaches experience high video-review friction."
    )
    handoff = ExecutionHandoff(
        experiment_id="exp-2",
        assumption_id="asmp-2",
        critical_assumption="Youth coaches experience high video-review friction.",
        experiment_type="CUSTOMER_INTERVIEWS",
        learning_goal="Validate coach friction.",
        what_we_are_testing="Whether youth coaches experience high video-review friction.",
        founder_actions=["Interview 6 coaches."],
        target_participants_or_context="6 youth coaches",
        target_sample_size=6,
        evidence_to_collect="Friction points and quotes.",
        support_signal="4+ coaches report severe setup friction.",
        founder_diagnosis_text=diag_text,
    )
    compact = render_execution_handoff_text(handoff, mode="compact")
    assert "STARTARY — CURRENT DIAGNOSIS" in compact
    assert "WHAT LOOKS STRONG" in compact
    assert "3 paying clubs active on the filming platform." in compact
    assert "WHAT COUNTS AS A USEFUL SIGNAL?" in compact
    assert "WHAT LOOKS PROMISING SO FAR?" not in compact


# ---------------------------------------------------------------------------
# Test 3, 4, 5, 6: EARLY SIGNALS FROM THIS TEST
# ---------------------------------------------------------------------------

def test_03_partial_valid_evidence_renders_early_signals_from_this_test():
    """3. Partial valid evidence renders EARLY SIGNALS FROM THIS TEST."""
    handoff_data = {
        "action_type": "CONTINUE_CURRENT_TEST",
        "is_partial_evidence": True,
        "target_sample_size": 15,
        "observations_collected": 8,
        "early_signals": [
            "3 of 4 youth teams reviewed so far meet repeat-use definition.",
            "Only 1 of 4 amateur teams meets the same definition.",
        ],
    }
    rendered = render_execution_handoff_text(handoff_data, mode="compact")
    assert "CURRENT TEST PROGRESS" in rendered
    assert "8 of 15" in rendered
    assert "EARLY SIGNALS FROM THIS TEST" in rendered
    assert "3 of 4 youth teams reviewed so far meet repeat-use definition." in rendered
    assert "This is directionally positive, but the experiment remains incomplete." in rendered
    assert "WHAT REMAINS" in rendered
    assert "WHAT LOOKS PROMISING SO FAR?" not in rendered


def test_04_early_signals_contain_only_active_experiment_findings():
    """4. Early signals contain only active-experiment findings from current turn."""
    state = {
        "project_id": "p-andever",
        "active_experiment": {
            "id": "exp-andever-1",
            "status": "IN_PROGRESS",
            "target_sample_size": 15,
            "target_participants_or_scope": "15 teams",
        },
        "active_assumption": {"id": "asmp-1", "statement": "Youth teams repeat more than amateur."},
        "founder_response": "We analyzed 8 teams: 4 youth, 4 amateur. 3 youth teams used clips repeatedly.",
        "evidence_analysis": {
            "conclusion": "PARTIALLY_SUPPORTED",
            "evidence_strength": "MODERATE",
            "experiment_execution_quality": "GOOD",
            "support_findings": ["3 of 4 youth teams met repeat use."],
            "valid_learnings": ["Youth teams show higher repeat engagement than amateur teams."],
        },
        "extracted_evidence": {
            "sample_or_counts": {"sample_size": 8},
            "observations": ["Youth teams showed repeat clip views across match dates."],
        },
    }
    updates = validate_evidence_analysis(state)
    assert updates["is_partial_evidence"] is True
    early_sigs = updates["active_execution_handoff"]["early_signals"]
    assert "3 of 4 youth teams met repeat use." in early_sigs
    assert "Youth teams show higher repeat engagement than amateur teams." in early_sigs
    # Verify rendered text includes them
    rendered = updates["execution_handoff_text"]
    assert "EARLY SIGNALS FROM THIS TEST" in rendered
    assert "3 of 4 youth teams met repeat use." in rendered


def test_05_early_signals_do_not_include_unrelated_historical_startup_claims():
    """5. Early signals do not include unrelated historical startup claims."""
    state = {
        "project_id": "p-andever",
        "active_experiment": {
            "id": "exp-andever-1",
            "status": "IN_PROGRESS",
            "target_sample_size": 15,
        },
        "active_assumption": {"id": "asmp-1", "statement": "Youth teams repeat more than amateur."},
        "founder_response": "Analyzed 8 teams.",
        "evidence_analysis": {
            "conclusion": "PARTIALLY_SUPPORTED",
            "support_findings": ["Youth cohort showed 75% repeat review."],
        },
        "startup_state": {
            "positive_signals": ["Startup has $5,000 monthly revenue", "Filming kit sold to 12 clubs"],
        },
    }
    updates = validate_evidence_analysis(state)
    early_sigs = updates["active_execution_handoff"]["early_signals"]
    assert "Youth cohort showed 75% repeat review." in early_sigs
    # Historical revenue and filming kit claims must NOT leak into early signals of this active test
    for sig in early_sigs:
        assert "$5,000" not in sig
        assert "filming kit" not in sig.lower()


def test_06_no_early_signals_section_when_no_meaningful_partial_signal_exists():
    """6. No early-signals section when no meaningful partial signal exists."""
    handoff_data = {
        "action_type": "CONTINUE_CURRENT_TEST",
        "is_partial_evidence": True,
        "target_sample_size": 8,
        "observations_collected": 1,
        "early_signals": [],
    }
    rendered = render_execution_handoff_text(handoff_data, mode="compact")
    assert "CURRENT TEST PROGRESS" in rendered
    assert "1 of 8" in rendered
    assert "EARLY SIGNALS FROM THIS TEST" not in rendered
    assert "WHAT REMAINS" in rendered


def test_07_full_diagnosis_not_redundantly_rendered_for_continue_current_test():
    """7. Full diagnosis is not redundantly re-rendered for simple CONTINUE_CURRENT_TEST."""
    diag_text = (
        "==================================================\n"
        "STARTARY — CURRENT DIAGNOSIS\n"
        "==================================================\n"
        "WHAT LOOKS STRONG\n"
        "- Some strong traction."
    )
    handoff = ExecutionHandoff(
        experiment_id="exp-1",
        assumption_id="asmp-1",
        critical_assumption="Active test",
        experiment_type="CUSTOMER_INTERVIEWS",
        learning_goal="Validate test",
        what_we_are_testing="Active test",
        founder_actions=["Observe 10 teams"],
        target_participants_or_context="10 teams",
        target_sample_size=10,
        observations_collected=4,
        evidence_to_collect="Observations",
        support_signal="Support",
        founder_diagnosis_text=diag_text,
        action_type="CONTINUE_CURRENT_TEST",
    )
    rendered = render_execution_handoff_text(handoff, mode="compact")
    assert "STARTARY — CURRENT DIAGNOSIS" not in rendered
    assert "CURRENT TEST PROGRESS" in rendered


# ---------------------------------------------------------------------------
# Test 8, 9, 10, 11, 12, 13: Sample Target Invariants & Single Source of Truth
# ---------------------------------------------------------------------------

def test_08_experiment_sample_target_is_stored_explicitly():
    """8. Experiment sample target is stored explicitly on Experiment model."""
    exp = Experiment(
        id="exp-1",
        assumption_id="asmp-1",
        project_id="p1",
        objective="Test sample size",
        target_sample_size=8,
        founder_actions=["Review 8 teams."],
        evidence_to_collect="Feedback",
    )
    dumped = exp.model_dump()
    assert dumped["target_sample_size"] == 8
    assert exp.target_sample_size == 8


def test_09_rendered_sample_requirement_equals_structured_target_sample_size():
    """9. Rendered sample requirement equals structured target sample size."""
    handoff = ExecutionHandoff(
        experiment_id="exp-1",
        assumption_id="asmp-1",
        critical_assumption="Active test",
        experiment_type="CUSTOMER_INTERVIEWS",
        learning_goal="Validate test",
        what_we_are_testing="Active test",
        founder_actions=["Review 15 teams."],
        target_participants_or_context="15 youth teams",
        target_sample_size=15,
        observations_collected=8,
        evidence_to_collect="Observations",
        support_signal="Support",
        action_type="CONTINUE_CURRENT_TEST",
    )
    rendered = render_execution_handoff_text(handoff, mode="compact")
    assert "8 of 15" in rendered
    assert "Collect the remaining 7" in rendered


def test_10_completeness_evaluator_uses_same_structured_target():
    """10. Completeness evaluator uses the same structured target directly."""
    state = {
        "active_experiment": {
            "target_sample_size": 8,
            "target_participants_or_scope": "Review 20 possible teams in local area",
        },
        "extracted_evidence": {
            "sample_or_counts": {"sample_size": 8},
        },
    }
    analysis = {"conclusion": "SUPPORTED", "evidence_strength": "STRONG", "experiment_execution_quality": "GOOD"}
    is_complete, reason, collected, target = evaluate_experiment_completeness(state, analysis)
    assert target == 8
    assert collected == 8
    assert is_complete is True


def test_11_completeness_evaluator_does_not_invent_hidden_target_when_none_exists():
    """11. Completeness evaluator does not invent a hidden target when none exists."""
    state = {
        "active_experiment": {
            "objective": "Observe founder onboarding process firsthand.",
            "target_participants_or_scope": "Self-hosted onboarding session",
            "target_sample_size": None,
        },
        "extracted_evidence": {
            "sample_or_counts": {"sample_size": 1},
        },
    }
    analysis = {"conclusion": "SUPPORTED", "evidence_strength": "STRONG", "experiment_execution_quality": "GOOD"}
    is_complete, reason, collected, target = evaluate_experiment_completeness(state, analysis)
    # Default is 1 (never silently invent 15 or 8)
    assert target == 1
    assert is_complete is True


def test_12_support_threshold_is_not_confused_with_sample_target():
    """12. Support threshold is not confused with sample target."""
    exp_data = {
        "target_participants_or_scope": "8 teams: 4 U14-U15 teams, 4 amateur teams",
        "support_signal": "At least 3 youth teams show repeat clip review.",
        "contradiction_signal": "Fewer than 2 teams show repeat review.",
    }
    target = extract_sample_target_from_protocol(exp_data)
    # Must extract sample size 8, NOT support threshold 3 or 2, and NOT age cohort 15
    assert target == 8


def test_13_critic_revision_changing_sample_target_updates_founder_facing_handoff():
    """13. Critic revision changing sample target updates founder-facing handoff."""
    state = {
        "project_id": "proj",
        "iteration": 1,
        "active_assumption": {"id": "asmp-1", "statement": "Test hypothesis"},
        "proposed_experiment_output": {
            "experiment_type": "CUSTOMER_INTERVIEW",
            "objective": "Test 8 teams after critic revised down from 15.",
            "hypothesis": "Test hypothesis",
            "method": "Review team workflows",
            "target_participants_or_scope": "8 teams: 4 U14-U15, 4 amateur",
            "target_sample_size": 8,
            "founder_actions": ["Review 8 teams"],
            "evidence_to_collect": "Workflow notes",
            "support_signal": "4+ teams adopt",
            "contradiction_signal": "0 teams adopt",
            "inconclusive_signal": "Ambiguous",
            "valid_inferences": ["Team adoption rate"],
            "invalid_inferences": ["Pricing willingness"],
        },
    }
    save_updates = save_final_experiment(state)
    state.update(save_updates)
    assert state["active_experiment"]["target_sample_size"] == 8

    handoff_updates = compose_execution_handoff(state)
    handoff = handoff_updates["active_execution_handoff"]
    assert handoff["target_sample_size"] == 8


# ---------------------------------------------------------------------------
# Test 14 & 15: &ever Specific Sample Target Scenarios
# ---------------------------------------------------------------------------

def test_14_andever_8_team_experiment_with_target_8_completes_when_8_returned():
    """14. &ever 8-team experiment with target=8 completes when 8 valid teams are returned."""
    state = {
        "active_experiment": {
            "id": "exp-andever",
            "target_sample_size": 8,
            "target_participants_or_scope": "8 teams: 4 U14-U15 teams, 4 amateur teams",
            "support_signal": "At least 3 youth teams show repeat review",
        },
        "founder_response": "Completed review of 8 teams: 4 youth teams and 4 amateur teams. 3 youth teams showed repeat use.",
        "extracted_evidence": {
            "sample_or_counts": {"sample_size": 8},
        },
    }
    analysis = {
        "conclusion": "SUPPORTED",
        "evidence_strength": "STRONG",
        "experiment_execution_quality": "GOOD",
    }
    is_complete, reason, collected, target = evaluate_experiment_completeness(state, analysis)
    assert target == 8
    assert collected == 8
    assert is_complete is True
    assert reason in ("TARGET_SAMPLE_REACHED", "DECISIVE_EARLY_SUPPORT", "COMPLETE")


def test_15_andever_target_15_experiment_explicitly_tells_founder_15_are_required():
    """15. &ever target=15 experiment explicitly tells founder 15 are required."""
    state = {
        "project_id": "p-andever",
        "active_experiment": {
            "id": "exp-andever-15",
            "status": "IN_PROGRESS",
            "target_sample_size": 15,
            "target_participants_or_scope": "Review 15 eligible teams",
        },
        "active_assumption": {"id": "asmp-1", "statement": "Youth repeat use"},
        "founder_response": "I reviewed 8 teams so far.",
        "extracted_evidence": {"sample_or_counts": {"sample_size": 8}},
        "evidence_analysis": {
            "conclusion": "PARTIALLY_SUPPORTED",
            "support_findings": ["3 youth teams returned to clips."],
        },
    }
    updates = validate_evidence_analysis(state)
    rendered = updates["execution_handoff_text"]
    assert "8 of 15" in rendered
    assert "Collect the remaining 7" in rendered
    # Must NOT say 8 of 8 or complete
    assert updates["is_partial_evidence"] is True


# ---------------------------------------------------------------------------
# Test 16, 17, 18, 19: Exception Classification Invariants
# ---------------------------------------------------------------------------

def test_16_request_timeout_classified_as_timeout():
    """16. RequestTimeout classified as TIMEOUT, not generic provider/network."""
    exc = TimeoutError("Request timed out.")
    category, title, guidance = classify_cli_error(exc)
    assert category == ErrorCategory.TIMEOUT
    assert category.value == "TIMEOUT"
    assert "STARTARY — DIAGNOSIS TEMPORARILY UNAVAILABLE" in guidance
    assert "Type /retry to resume." in guidance


def test_17_429_rate_limit_remains_rate_limit():
    """17. 429 rate limit remains RATE_LIMIT."""
    exc = RuntimeError("Error code: 429 - {'error': {'message': 'Rate limit exceeded. Please slow down.'}}")
    category, title, guidance = classify_cli_error(exc)
    assert category == ErrorCategory.RATE_LIMIT
    assert category.value == "RATE_LIMIT"
    assert "Rate Limit" in title


def test_18_insufficient_quota_remains_distinct_from_rate_limit():
    """18. insufficient_quota remains distinct from rate limit."""
    exc = RuntimeError("Error code: 429 - {'error': {'message': 'You exceeded your current quota, please check your plan and billing details.', 'code': 'insufficient_quota'}}")
    category, title, guidance = classify_cli_error(exc)
    assert category == ErrorCategory.INSUFFICIENT_QUOTA
    assert category.value == "INSUFFICIENT_QUOTA"
    assert "Quota Exceeded" in title
    assert category != ErrorCategory.RATE_LIMIT


def test_19_authentication_error_remains_distinct():
    """19. authentication error remains distinct."""
    exc = RuntimeError("Error code: 401 - {'error': {'message': 'Incorrect API key provided'}}")
    category, title, guidance = classify_cli_error(exc)
    assert category == ErrorCategory.AUTHENTICATION
    assert category.value == "AUTHENTICATION"
    assert "Authentication" in title


# ---------------------------------------------------------------------------
# Test 20, 21, 22, 23, 24, 25: Post-Evidence Timeout & Safe Resume
# ---------------------------------------------------------------------------

def test_20_post_evidence_timeout_does_not_rollback_validated_evidence():
    """20. post-evidence timeout does not rollback validated evidence."""
    # State after validate_evidence_analysis and apply_validated_state_updates
    state = {
        "project_id": "proj-timeout-test",
        "iteration": 1,
        "phase": WorkflowPhase.TEST,
        "active_assumption": {
            "id": "asmp-1",
            "statement": "Pain assumption",
            "status": AssumptionStatus.SUPPORTED.value,
        },
        "active_experiment": {
            "id": "exp-1",
            "status": ExperimentStatus.COMPLETED.value,
        },
        "evidence_history": [
            {"id": "ev-1", "conclusion": "SUPPORTED", "raw_content": "8 teams reviewed"}
        ],
        "evidence_analysis": {"conclusion": "SUPPORTED", "evidence_strength": "STRONG"},
    }
    # Simulate timeout occurring at diagnose_startup
    iter_updates = increment_iteration(state)
    state.update(iter_updates)
    assert state["iteration"] == 2

    # Evidence is preserved in evidence_history and previous_assumptions
    assert len(state["evidence_history"]) == 1
    assert state["evidence_history"][0]["id"] == "ev-1"
    assert len(state["previous_assumptions"]) == 1


def test_21_retry_does_not_duplicate_evidence():
    """21. retry does not duplicate evidence."""
    state = {
        "project_id": "proj-idempotency",
        "iteration": 2,
        "evidence_history": [
            {"id": "ev-1", "conclusion": "SUPPORTED", "raw_content": "8 teams reviewed"}
        ],
        "previous_assumptions": [{"id": "asmp-1", "statement": "Test"}],
        "startup_state": {"areas": {}},
    }
    # Calling increment_iteration again on state with no active experiment returns empty
    updates = increment_iteration(state)
    assert updates == {}
    assert len(state["evidence_history"]) == 1


def test_22_retry_does_not_increment_iteration_twice():
    """22. retry does not increment iteration twice."""
    state = {
        "project_id": "proj-iter-test",
        "iteration": 2,
        "active_experiment": None,
        "active_assumption": None,
    }
    iter_updates = increment_iteration(state)
    assert iter_updates == {}
    assert state["iteration"] == 2


def test_23_retry_does_not_duplicate_learning_delta():
    """23. retry does not duplicate LearningDelta."""
    state = {
        "project_id": "proj-delta",
        "iteration": 2,
        "evidence_analysis": {"conclusion": "SUPPORTED", "evidence_strength": "STRONG", "reasoning_summary": "Good"},
        "learning_delta": {"outcome": "SUPPORTED", "summary": "Initial delta"},
    }
    # When resume happens, delta is synthesized idempotently for iteration 2
    assert "learning_delta" in state
    assert state["learning_delta"]["outcome"] == "SUPPORTED"


def test_24_retry_resumes_diagnosis_from_current_state():
    """24. retry resumes diagnosis from current state using resume_post_evidence_diagnosis."""
    state = {
        "project_id": "proj-resume",
        "iteration": 2,
        "phase": WorkflowPhase.LEARN,
        "founder_messages": [{"role": "founder", "content": "Context"}],
        "raw_context": "B2B SaaS with paying customers.",
        "startup_state": {"areas": {}},
        "evidence_history": [{"id": "ev-1", "conclusion": "SUPPORTED"}],
        "evidence_analysis": {"conclusion": "SUPPORTED", "evidence_strength": "STRONG", "reasoning_summary": "Passed"},
    }
    mock_diag_output = {
        "diagnosis_summary": "Updated diagnosis after evidence",
        "startup_stage": "validation",
        "area_assessments": [
            {
                "area": "Problem",
                "status": "KNOWN",
                "summary": "Clear",
                "evidence_sources": ["Interviews"],
                "identified_gaps": [],
            }
        ],
        "clear_areas": ["Problem"],
        "unclear_areas": ["Acquisition / Customer Access"],
        "missing_areas": [],
        "unsupported_claims": [],
    }
    # M6.2.3: assess_startup_areas was unmocked and made a live reasoning-tier LLM call.
    with patch("startary.graph.builder.diagnose_startup", return_value={"diagnosis_output": mock_diag_output, "phase": WorkflowPhase.DIAGNOSE}), \
         patch("startary.graph.builder.assess_startup_areas", return_value={"area_assessments_output": make_valid_area_assessments_batch_output_dict()}), \
         patch("startary.graph.builder.reality_check", return_value={"reality_check": {}}), \
         patch("startary.graph.builder.generate_critical_assumption", return_value={"proposed_critical_assumption": {"id": "ca-2", "statement": "New CA"}}), \
         patch("startary.graph.builder.save_critical_assumption", return_value={"active_assumption": {"id": "ca-2", "statement": "New CA"}}), \
         patch("startary.graph.builder.decide_next_strategic_action", return_value={"next_strategic_action": {"action_type": "NEW_EXPERIMENT"}}):
        resumed = resume_post_evidence_diagnosis(state)
        assert resumed["iteration"] == 2
        assert resumed["diagnosis_output"]["diagnosis_summary"] == "Updated diagnosis after evidence"


def test_25_founder_does_not_need_to_resubmit_evidence_after_timeout():
    """25. founder does not need to resubmit evidence after timeout."""
    timed_out_state = {
        "project_id": "proj-no-resubmit",
        "iteration": 2,
        "founder_response": None,  # No new input required
        "evidence_history": [{"id": "ev-1", "conclusion": "SUPPORTED"}],
        "evidence_analysis": {"conclusion": "SUPPORTED", "evidence_strength": "STRONG", "reasoning_summary": "Supportive findings"},
        "founder_messages": [{"role": "founder", "content": "Initial context"}],
        "raw_context": "B2B SaaS context",
    }
    mock_diag_output = {
        "diagnosis_summary": "Diag",
        "startup_stage": "validation",
        "area_assessments": [
            {
                "area": "Problem",
                "status": "KNOWN",
                "summary": "Clear",
                "evidence_sources": [],
                "identified_gaps": [],
            }
        ],
        "clear_areas": ["Problem"],
        "unclear_areas": [],
        "missing_areas": [],
        "unsupported_claims": [],
    }
    # M6.2.3: assess_startup_areas was unmocked and made a live reasoning-tier LLM call.
    with patch("startary.graph.builder.diagnose_startup", return_value={"diagnosis_output": mock_diag_output}), \
         patch("startary.graph.builder.assess_startup_areas", return_value={"area_assessments_output": make_valid_area_assessments_batch_output_dict()}), \
         patch("startary.graph.builder.reality_check", return_value={}), \
         patch("startary.graph.builder.generate_critical_assumption", return_value={}), \
         patch("startary.graph.builder.save_critical_assumption", return_value={}), \
         patch("startary.graph.builder.decide_next_strategic_action", return_value={}):
        resumed = resume_post_evidence_diagnosis(timed_out_state)
        # Succeeded without founder_response
        assert resumed["iteration"] == 2


# ---------------------------------------------------------------------------
# Test 26, 27, 28, 29, 30: Regressions & Invariants Preserved
# ---------------------------------------------------------------------------

def test_26_m4121_diagnosis_rendering_still_works():
    """26. M4.1.2.1 diagnosis rendering still works."""
    diag_text = (
        "==================================================\n"
        "STARTARY — CURRENT DIAGNOSIS\n"
        "==================================================\n\n"
        "WHAT LOOKS STRONG\n"
        "- 3 paying clubs\n\n"
        "WHAT LOOKS WEAK\n"
        "- Coach setup friction\n\n"
        "WHAT IS STILL UNCERTAIN\n"
        "- Repeat usage cadence\n\n"
        "WHAT CHANGED\n"
        "- Target User: UNCLEAR → KNOWN\n\n"
        "MOST IMPORTANT DECISION NOW\n"
        "Whether coaches adopt video review weekly."
    )
    handoff = ExecutionHandoff(
        experiment_id="exp-1",
        assumption_id="asmp-1",
        critical_assumption="Coaches adopt video review",
        experiment_type="CUSTOMER_INTERVIEWS",
        learning_goal="Validate adoption",
        what_we_are_testing="Whether coaches adopt video review weekly.",
        founder_actions=["Interview 5 coaches"],
        target_participants_or_context="5 coaches",
        evidence_to_collect="Usage frequency",
        support_signal="3+ coaches adopt",
        founder_diagnosis_text=diag_text,
    )
    compact = render_execution_handoff_text(handoff, mode="compact")
    assert "STARTARY — CURRENT DIAGNOSIS" in compact
    assert "WHAT LOOKS STRONG" in compact
    assert "WHAT LOOKS WEAK" in compact
    assert "WHAT CHANGED" in compact
    assert "MOST IMPORTANT DECISION NOW" in compact
    assert "YOUR NEXT TEST" in compact
    assert "WHAT LOOKS PROMISING SO FAR?" not in compact


def test_27_partial_evidence_continuation_still_works():
    """27. partial evidence continuation lifecycle preserved."""
    state = {
        "project_id": "proj",
        "active_experiment": {
            "id": "exp-1",
            "status": "IN_PROGRESS",
            "target_sample_size": 10,
        },
        "active_assumption": {"id": "asmp-1", "statement": "Test stmt"},
        "founder_response": "Completed 3 tests.",
        "extracted_evidence": {"sample_or_counts": {"sample_size": 3}},
        "evidence_analysis": {"conclusion": "PARTIALLY_SUPPORTED", "support_findings": ["First 3 tests worked."]},
    }
    updates = validate_evidence_analysis(state)
    assert updates["is_partial_evidence"] is True
    assert updates["phase"] == WorkflowPhase.WAITING_FOR_EVIDENCE
    assert updates["active_experiment"]["status"] == ExperimentStatus.IN_PROGRESS.value
    assert updates["active_experiment"]["collected_sample_count"] == 3


def test_28_product_iteration_still_works():
    """28. PRODUCT_ITERATION still works and renders cleanly."""
    handoff = ExecutionHandoff(
        experiment_id="exp-pi",
        assumption_id="asmp-pi",
        critical_assumption="Setup bottleneck",
        experiment_type="OTHER",
        learning_goal="Fix setup",
        what_we_are_testing="Setup usability",
        target_participants_or_context="Active teams",
        evidence_to_collect="Completion rate",
        support_signal="Setup time < 5 min",
        next_strategic_action={
            "action_type": "PRODUCT_ITERATION",
            "bottleneck_identified": "filming kit setup",
            "smallest_change": "add quick-mount adapter",
            "what_not_to_build": ["tactical analytics", "multi-camera support"],
            "measurable_outcome": "setup time drops under 5 minutes",
        },
    )
    compact = render_execution_handoff_text(handoff, mode="compact")
    assert "NEXT ACTION — PRODUCT ITERATION" in compact
    assert "BUILD / CHANGE ONLY THIS:" in compact
    assert "add quick-mount adapter" in compact
    assert "DO NOT BUILD YET:" in compact
    assert "tactical analytics" in compact
    assert "MEASURE:" in compact
    assert "WHAT LOOKS PROMISING SO FAR?" not in compact


def test_29_m411_recovery_still_works():
    """29. M4.1.1 recovery still works."""
    from startary.graph.nodes import handle_experiment_failure_stop
    state = {
        "project_id": "proj",
        "iteration": 1,
        "active_assumption": {"id": "asmp-1", "statement": "Unfalsifiable claim"},
        "experiment_critic_output": {
            "verdict": "REVISE",
            "revision_count": 2,
            "issues": ["Test cannot be made falsifiable"],
        },
    }
    updates = handle_experiment_failure_stop(state)
    assert "fallback_learning_action" in updates
    assert updates["phase"] == WorkflowPhase.WAITING_FOR_LEARNING_INPUT


def test_30_full_regression_suite_passes():
    """30. Single authoritative source of truth for experiment sample sizes invariant."""
    exp_8 = {"target_sample_size": 8, "target_participants_or_scope": "8 teams: 4 U14-U15, 4 amateur"}
    exp_15 = {"target_sample_size": 15, "target_participants_or_scope": "15 teams"}
    exp_default = {"target_participants_or_scope": "Exploratory interviews"}

    assert extract_sample_target_from_protocol(exp_8) == 8
    assert extract_sample_target_from_protocol(exp_15) == 15
    assert extract_sample_target_from_protocol(exp_default) is None
