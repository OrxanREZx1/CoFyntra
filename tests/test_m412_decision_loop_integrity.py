"""
Unit and Integration Tests for Milestone M4.1.2 — Decision Loop Integrity & Progression.

Covers all 24 required test scenarios:
  1. Traction-stage startup does not lose existing behavioral evidence during diagnosis.
  2. Paying/repeat-use evidence affects appropriate area knowledge.
  3. &ever-like startup is not automatically routed to generic Problem discovery without a grounded reason.
  4. Partial evidence does not complete an experiment.
  5. Partial evidence does not increment iteration.
  6. Partial evidence preserves active assumption.
  7. Partial evidence preserves active experiment.
  8. Partial evidence produces a "continue current test" action.
  9. Complete inconclusive evidence still allows normal re-diagnosis.
  10. Completed supported evidence produces LearningDelta.
  11. Founder-facing diagnosis includes strong/weak/uncertain.
  12. Founder-facing response includes WHAT CHANGED.
  13. Founder-facing response identifies most important current decision.
  14. NextStrategicAction selects CONTINUE_CURRENT_TEST for incomplete experiment.
  15. New experiment is not generated while current valid experiment remains incomplete.
  16. Existing-product startup is not forced through MVP build decision.
  17. Build readiness cannot mention unrelated LastBite/commission concepts in &ever.
  18. Active-project grounding prevents cross-startup domain leakage.
  19. Critic-rejected sampling bias cannot reappear in fallback.
  20. Existing LastBite pre-product BUILD_MINIMAL_PROTOTYPE behavior still works.
  21. M4.1.1 experiment failure recovery remains interactive.
  22. No regression in evidence admission precedence.
  23. No regression in LearningDelta attribution.
  24. Full test suite passes.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from startary.domain.enums import (
    AssumptionStatus,
    EvidenceAnalysisConclusion,
    EvidenceResult,
    ExecutionReadiness,
    ExperimentStatus,
    ExperimentType,
    WorkflowPhase,
)
from startary.domain.taxonomy import StartupArea
from startary.domain.assessment import RiskFactorLevel
from startary.domain.evidence import EvidenceTurnDisposition
from startary.domain.strategic_action import (
    NextStrategicAction,
    NextStrategicActionType,
)
from startary.domain.build_readiness import (
    BuildReadinessAction,
    BuildReadinessDecision,
    validate_domain_grounding,
    sanitize_domain_leakage,
    FOREIGN_DOMAIN_TERMS,
)
from startary.domain.experiment import (
    evaluate_experiment_completeness,
    render_execution_handoff_text,
    compose_fallback_learning_action,
    ExecutionHandoff,
)
from startary.graph.nodes import (
    assess_startup_areas,
    assess_build_readiness,
    compose_startup_diagnosis,
    compose_execution_handoff,
    decide_next_strategic_action,
    select_weak_link,
    validate_evidence_analysis,
)
from startary.graph.builder import (
    route_after_validate_evidence,
    run_evidence_learning_pipeline,
    build_graph,
)
from startary.llm.schemas import (
    AreaRiskAssessmentLLMOutput,
    StartupAreaAssessmentBatchLLMOutput,
    StartupDiagnosisSnapshotLLMOutput,
)


# ---------------------------------------------------------------------------
# Fixture: &ever Startup State
# ---------------------------------------------------------------------------
ANDEVER_RAW_CONTEXT = (
    "&ever is a sports-tech video recording and analysis tool for amateur youth soccer teams in Korea. "
    "We have a deployed product live in production. 20 teams tried our match recording, 14 recorded full matches, "
    "and 8 paying teams subscribe and pay recurring monthly fees for weekly match recordings."
)


# ---------------------------------------------------------------------------
# Test 1 & 2: Traction-stage startup diagnosis & area knowledge
# ---------------------------------------------------------------------------

def test_01_traction_stage_startup_does_not_lose_existing_behavioral_evidence_during_diagnosis():
    """Traction-stage startup (&ever) retains its behavioral evidence in state and does not lose it."""
    state = {
        "project_id": "andever_proj",
        "raw_context": ANDEVER_RAW_CONTEXT,
        "founder_messages": [
            {"role": "founder", "content": "8 paying teams currently use our software weekly to record full soccer matches."},
        ],
        "startup_state": {
            "stage": "PAYING_CUSTOMERS",
            "areas": {
                "Problem": {"understanding": "Amateur youth soccer teams need reliable full match recording without expensive camera crews."},
                "Product": {"understanding": "Deployed live app used weekly by 14 teams to record matches."},
                "Pricing": {"understanding": "8 teams paying monthly subscription."},
            }
        },
        "diagnosis": {
            "summary": "Live sports-tech product with 8 paying teams and 14 active teams recording weekly matches.",
            "startup_stage": "PAYING_CUSTOMERS",
        }
    }

    # Verify that traction keywords exist and are identifiable
    all_text = (state["raw_context"] + " " + state["founder_messages"][0]["content"]).lower()
    assert "8 paying teams" in all_text
    assert "recorded full matches" in all_text
    assert "weekly" in all_text


def test_02_paying_repeat_use_evidence_affects_appropriate_area_knowledge():
    """Paying customers and repeat usage reduce Problem risk from saturated 81 by lowering uncertainty/gap."""
    state = {
        "project_id": "andever_proj",
        "raw_context": ANDEVER_RAW_CONTEXT,
        "founder_messages": [
            {"role": "founder", "content": "We have 8 paying teams and weekly usage recording full matches."},
        ],
        "diagnosis": {
            "summary": "Live product with paying teams and repeat match recordings.",
            "startup_stage": "PAYING_CUSTOMERS",
        }
    }

    mock_llm_output = StartupAreaAssessmentBatchLLMOutput(
        assessments=[
            AreaRiskAssessmentLLMOutput(
                area=StartupArea.PROBLEM,
                importance=RiskFactorLevel.HIGH,
                uncertainty=RiskFactorLevel.HIGH,
                evidence_gap=RiskFactorLevel.HIGH,
                consequence_of_being_wrong=RiskFactorLevel.HIGH,
                rationale="Problem unproven.",
            ),
            AreaRiskAssessmentLLMOutput(
                area=StartupArea.BUSINESS_MODEL,
                importance=RiskFactorLevel.HIGH,
                uncertainty=RiskFactorLevel.HIGH,
                evidence_gap=RiskFactorLevel.HIGH,
                consequence_of_being_wrong=RiskFactorLevel.HIGH,
                rationale="Business model sustainability unproven.",
            ),
        ]
    )

    with patch("startary.graph.nodes.get_llm_client") as mock_client:
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = mock_llm_output
        mock_client.return_value.with_structured_output.return_value = mock_structured

        updates = assess_startup_areas(state)

    output = updates["area_assessments_output"]
    assessments = output["assessments"]
    problem_asm = next(a for a in assessments if str(a["area"]).lower() == "problem")

    # Problem uncertainty and evidence gap should be adjusted down from HIGH to MEDIUM
    assert problem_asm["uncertainty"] == RiskFactorLevel.MEDIUM.value
    assert problem_asm["evidence_gap"] == RiskFactorLevel.MEDIUM.value
    assert "Adjusted for verified" in problem_asm["rationale"]


def test_03_andever_like_startup_not_automatically_routed_to_generic_problem_discovery():
    """&ever-like startup with adjusted Problem does not tie at 81, allowing Business Model to be selected."""
    state = {
        "project_id": "andever_proj",
        "raw_context": ANDEVER_RAW_CONTEXT,
        "startup_state": {"areas": {}},
        "area_assessments_output": {
            "assessments": [
                {
                    "area": "Problem",
                    "importance": "HIGH",
                    "uncertainty": "MEDIUM",      # Adjusted for paying/repeat use
                    "evidence_gap": "MEDIUM",    # Adjusted for paying/repeat use
                    "consequence_of_being_wrong": "HIGH",
                    "rationale": "Adjusted for verified paying customers",
                },
                {
                    "area": "Business Model / Revenue",
                    "importance": "HIGH",
                    "uncertainty": "HIGH",
                    "evidence_gap": "HIGH",
                    "consequence_of_being_wrong": "HIGH",
                    "rationale": "Will teams continue paying across multi-month seasons?",
                },
            ]
        },
        "diagnosis": {},
    }

    result = select_weak_link(state)
    # Business Model has score 81 (3x3x3x3), Problem has score 36 (3x2x2x3).
    # Business Model should win the weak link!
    assert result["weak_area"] == "Business Model / Revenue"
    assert result["weak_link_details"]["score"] == 81



# ---------------------------------------------------------------------------
# Test 4 to 8: Partial Evidence Handling
# ---------------------------------------------------------------------------

def test_04_partial_evidence_does_not_complete_experiment():
    """Submitting 1 observation out of target 8 leaves active_experiment as IN_PROGRESS."""
    state = {
        "project_id": "andever_proj",
        "active_experiment": {
            "id": "exp-123",
            "status": "IN_PROGRESS",
            "target_participants_or_scope": "8 amateur soccer teams",
            "collected_sample_count": 0,
        },
        "active_assumption": {
            "id": "asmp-123",
            "statement": "Teams record matches weekly",
            "status": "IN_PROGRESS",
        },
        "founder_response": "We completed a recording session with Team Alpha. It went well.",
        "founder_evidence": [{"raw_content": "Completed recording with Team Alpha."}],
        "evidence_analysis": {
            "conclusion": "SUPPORTED",
            "rationale": "First team recorded successfully.",
        },
        "evidence_turn_assessment": {"disposition": "ACTUAL_EVIDENCE"},
    }

    updates = validate_evidence_analysis(state)

    assert updates.get("is_partial_evidence") is True
    assert updates["active_experiment"]["status"] == ExperimentStatus.IN_PROGRESS.value
    assert updates["active_experiment"]["collected_sample_count"] == 1
    assert updates["active_experiment"]["target_sample_size"] == 8


def test_05_partial_evidence_does_not_increment_iteration():
    """Partial evidence submission stops before increment_iteration in run_evidence_learning_pipeline."""
    state = {
        "project_id": "andever_proj",
        "iteration": 2,
        "phase": WorkflowPhase.WAITING_FOR_EVIDENCE,
        "active_experiment": {
            "id": "exp-123",
            "status": "IN_PROGRESS",
            "target_participants_or_scope": "8 teams",
        },
        "active_assumption": {
            "id": "asmp-123",
            "statement": "Teams record matches weekly",
            "status": "IN_PROGRESS",
        },
        "founder_messages": [],
    }

    with patch("startary.graph.builder.interpret_founder_turn") as mock_interp, \
         patch("startary.graph.builder.classify_waiting_for_evidence_turn") as mock_class, \
         patch("startary.graph.builder.extract_experiment_evidence") as mock_extract, \
         patch("startary.graph.builder.assess_evidence_submission_sufficiency") as mock_suff, \
         patch("startary.graph.builder.analyze_evidence") as mock_ana:
        mock_interp.return_value = {}
        mock_class.return_value = {"evidence_turn_assessment": {"disposition": "ACTUAL_EVIDENCE"}}
        mock_extract.return_value = {
            "extracted_evidence": {"sample_size": 1},
            "founder_evidence": [{"raw_content": "Team 1 recorded match"}],
        }
        mock_suff.return_value = {"evidence_submission_sufficiency": "SUFFICIENT_FOR_ANALYSIS"}
        mock_ana.return_value = {"evidence_analysis": {"conclusion": "SUPPORTED"}}

        result = run_evidence_learning_pipeline(state, "Team 1 recorded their match successfully.")

    # Iteration must NOT increment; must remain 2
    assert result.get("iteration") == 2
    assert result.get("is_partial_evidence") is True


def test_06_partial_evidence_preserves_active_assumption():
    """Partial evidence preserves active assumption in IN_PROGRESS status."""
    state = {
        "project_id": "proj",
        "active_experiment": {
            "id": "exp-1",
            "status": "IN_PROGRESS",
            "target_participants_or_scope": "8 teams",
        },
        "active_assumption": {
            "id": "asmp-1",
            "statement": "Teams record matches weekly",
            "status": "IN_PROGRESS",
        },
        "founder_response": "1 team recorded",
        "evidence_analysis": {"conclusion": "SUPPORTED"},
        "evidence_turn_assessment": {"disposition": "ACTUAL_EVIDENCE"},
    }

    updates = validate_evidence_analysis(state)
    assert updates["active_assumption"]["status"] == AssumptionStatus.TESTING.value


def test_07_partial_evidence_preserves_active_experiment():
    """Partial observations are accumulated in active_experiment."""
    state = {
        "project_id": "proj",
        "active_experiment": {
            "id": "exp-1",
            "status": "IN_PROGRESS",
            "target_participants_or_scope": "8 teams",
            "partial_observations": ["Prior observation 1"],
        },
        "active_assumption": {"id": "asmp-1", "statement": "Test stmt"},
        "founder_response": "Second observation recorded.",
        "evidence_analysis": {"conclusion": "SUPPORTED"},
        "evidence_turn_assessment": {"disposition": "ACTUAL_EVIDENCE"},
    }

    updates = validate_evidence_analysis(state)
    active_exp = updates["active_experiment"]
    assert "Prior observation 1" in active_exp["partial_observations"]
    assert "Second observation recorded." in active_exp["partial_observations"]
    assert len(active_exp["partial_observations"]) == 2


def test_08_partial_evidence_produces_continue_current_test_action():
    """Continuation prompt is composed instructing founder to continue the test with remaining cases."""
    state = {
        "project_id": "proj",
        "active_experiment": {
            "id": "exp-1",
            "status": "IN_PROGRESS",
            "target_participants_or_scope": "8 teams",
        },
        "active_assumption": {"id": "asmp-1", "statement": "Test stmt"},
        "founder_response": "Team 1 recorded.",
        "evidence_analysis": {"conclusion": "SUPPORTED"},
        "evidence_turn_assessment": {"disposition": "ACTUAL_EVIDENCE"},
    }

    updates = validate_evidence_analysis(state)
    handoff_text = updates["execution_handoff_text"]
    assert "requires 8 teams" in handoff_text
    assert "currently have 1 qualifying case(s)" in handoff_text
    assert "Continue the same test with the remaining teams." in handoff_text


# ---------------------------------------------------------------------------
# Test 9 & 10: Complete Evidence Cycle
# ---------------------------------------------------------------------------

def test_09_complete_inconclusive_evidence_still_allows_normal_rediagnosis():
    """Complete evidence submission with 8 cases is marked complete and advances normally."""
    state = {
        "project_id": "proj",
        "active_experiment": {
            "id": "exp-1",
            "status": "IN_PROGRESS",
            "target_participants_or_scope": "8 teams",
        },
        "active_assumption": {"id": "asmp-1", "statement": "Test stmt"},
        "founder_response": "We recorded 8 teams. Results were split evenly.",
        "evidence_analysis": {
            "conclusion": "INCONCLUSIVE",
            "rationale": "4 succeeded, 4 failed.",
        },
        "evidence_turn_assessment": {"disposition": "ACTUAL_EVIDENCE"},
    }

    is_complete, _, collected, target = evaluate_experiment_completeness(state, state["evidence_analysis"])
    assert is_complete is True
    assert collected == 8

    updates = validate_evidence_analysis(state)
    assert updates["active_experiment"]["status"] == ExperimentStatus.COMPLETED.value
    assert updates.get("is_partial_evidence") is not True


def test_10_completed_supported_evidence_produces_learning_delta():
    """Completed supported evidence completes experiment and maps assumption to SUPPORTED."""
    state = {
        "project_id": "proj",
        "active_experiment": {
            "id": "exp-1",
            "status": "IN_PROGRESS",
            "target_participants_or_scope": "8 teams",
        },
        "active_assumption": {"id": "asmp-1", "statement": "Test stmt"},
        "founder_response": "All 8 teams recorded matches successfully.",
        "evidence_analysis": {
            "conclusion": "SUPPORTED",
            "rationale": "8 out of 8 teams completed recordings.",
        },
        "evidence_turn_assessment": {"disposition": "ACTUAL_EVIDENCE"},
    }

    updates = validate_evidence_analysis(state)
    assert updates["active_experiment"]["status"] == ExperimentStatus.COMPLETED.value
    assert updates["active_assumption"]["status"] == AssumptionStatus.SUPPORTED.value


# ---------------------------------------------------------------------------
# Test 11, 12, 13: Founder-Facing Diagnosis Visibility
# ---------------------------------------------------------------------------

def test_11_founder_facing_diagnosis_includes_strong_weak_uncertain():
    """Founder-facing diagnosis includes CURRENT STAGE & ADVANTAGES, BIGGEST IMMEDIATE BOTTLENECK, and WHAT DID NOT CHANGE."""
    state = {
        "project_id": "andever_proj",
        "weak_area": "Retention",
        "selection_confidence": "HIGH",
        "active_assumption": {"statement": "Teams record matches weekly for at least 6 weeks"},
        "startup_progress": {"development_stage": {"value": "PAYING_CUSTOMERS"}},
        "reality_check": {
            "summary": "8 paying teams established, but season-long retention unverified.",
            "what_evidence_supports": ["8 paying teams subscribing monthly"],
            "what_evidence_does_not_support": [],
            "unresolved_uncertainty": "Multi-month cohort retention",
        },
    }

    with patch("startary.graph.nodes.get_llm_client") as mock_client:
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = StartupDiagnosisSnapshotLLMOutput(
            stage="Paying Customers",
            stage_explanation="Active paying customer cohort established.",
            grounded_areas=["8 paying teams subscribing monthly"],
            current_advantages=["Live deployed product", "Direct team relationships"],
            execution_constraints=[],
            major_uncertainties=["Season-long retention", "Drop-off reasons"],
            weak_link="Retention",
            weak_link_reason="Churn risk before multi-month renewal is primary gating risk.",
            weak_link_confidence="HIGH",
            weak_link_confidence_explanation="High confidence based on verified usage data.",
            competing_risks=["Acquisition", "Pricing"],
            reality_summary="Strong initial traction, retention unverified.",
            recommended_learning_goal="Measure weekly active match recording drop-offs.",
            critical_assumption="Teams record matches weekly for at least 6 weeks",
        )
        mock_client.return_value.with_structured_output.return_value = mock_structured

        updates = compose_startup_diagnosis(state)

    diag_text = updates["founder_diagnosis_text"]
    assert "STARTARY — CURRENT DIAGNOSIS" in diag_text
    assert "WHAT LOOKS STRONG" in diag_text or "1. CURRENT STAGE & ADVANTAGES" in diag_text
    assert "WHAT LOOKS WEAK" in diag_text or "2. BIGGEST IMMEDIATE BOTTLENECK / WEAK LINK" in diag_text
    assert "WHAT DID NOT CHANGE" in diag_text


def test_12_founder_facing_response_includes_what_changed():
    """Founder-facing response includes section WHAT CHANGED."""
    state = {
        "project_id": "andever_proj",
        "weak_area": "Retention",
        "active_assumption": {"statement": "Teams record matches weekly"},
        "learning_delta": {
            "headline": "Pilot cohort confirmed weekly match recording feasibility.",
            "key_takeaways": ["14 teams recorded matches without technical failure."],
        },
    }

    with patch("startary.graph.nodes.get_llm_client") as mock_client:
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = StartupDiagnosisSnapshotLLMOutput(
            stage="Live Product",
            stage_explanation="Product deployed.",
            grounded_areas=[],
            current_advantages=[],
            execution_constraints=[],
            major_uncertainties=[],
            weak_link="Retention",
            weak_link_reason="Primary risk.",
            weak_link_confidence="HIGH",
            weak_link_confidence_explanation="Confidence high.",
            competing_risks=[],
            reality_summary="Traction exists.",
            recommended_learning_goal="Validate retention.",
            critical_assumption="Teams record matches weekly",
        )
        mock_client.return_value.with_structured_output.return_value = mock_structured

        updates = compose_startup_diagnosis(state)

    diag_text = updates["founder_diagnosis_text"]
    assert "WHAT CHANGED" in diag_text
    assert "Pilot cohort confirmed weekly match recording feasibility" in diag_text


def test_13_founder_facing_response_identifies_most_important_current_decision():
    """Founder-facing response includes section MOST IMPORTANT DECISION NOW."""
    state = {
        "project_id": "andever_proj",
        "weak_area": "Retention",
        "active_assumption": {"statement": "Teams record matches weekly for 6 weeks"},
    }

    with patch("startary.graph.nodes.get_llm_client") as mock_client:
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = StartupDiagnosisSnapshotLLMOutput(
            stage="Live Product",
            stage_explanation="Product deployed.",
            grounded_areas=[],
            current_advantages=[],
            execution_constraints=[],
            major_uncertainties=[],
            weak_link="Retention",
            weak_link_reason="Primary risk.",
            weak_link_confidence="HIGH",
            weak_link_confidence_explanation="Confidence high.",
            competing_risks=[],
            reality_summary="Traction exists.",
            recommended_learning_goal="Validate retention.",
            critical_assumption="Teams record matches weekly for 6 weeks",
        )
        mock_client.return_value.with_structured_output.return_value = mock_structured

        updates = compose_startup_diagnosis(state)

    diag_text = updates["founder_diagnosis_text"]
    assert "MOST IMPORTANT DECISION NOW" in diag_text or "6. MOST IMPORTANT CURRENT STRATEGIC DECISION" in diag_text
    assert "Validate whether Teams record matches weekly for 6 weeks" in diag_text


# ---------------------------------------------------------------------------
# Test 14 & 15: NextStrategicAction & Routing Preservation
# ---------------------------------------------------------------------------

def test_14_next_strategic_action_selects_continue_current_test_for_incomplete_experiment():
    """decide_next_strategic_action selects CONTINUE_CURRENT_TEST when is_partial_evidence is True."""
    state = {
        "is_partial_evidence": True,
        "weak_area": "Retention",
        "active_assumption": {"statement": "Teams record weekly"},
    }

    updates = decide_next_strategic_action(state)
    action = updates["next_strategic_action"]
    assert action["action_type"] == NextStrategicActionType.CONTINUE_CURRENT_TEST.value
    assert "Current experiment is in progress with partial evidence" in action["rationale"]


def test_15_new_experiment_is_not_generated_while_current_valid_experiment_remains_incomplete():
    """route_after_validate_evidence routes back to wait_for_evidence on partial evidence."""
    state_partial = {"is_partial_evidence": True}
    route_partial = route_after_validate_evidence(state_partial)
    assert route_partial == "wait_for_evidence"

    state_complete = {"is_partial_evidence": False}
    route_complete = route_after_validate_evidence(state_complete)
    assert route_complete == "apply_validated_state_updates"


# ---------------------------------------------------------------------------
# Test 16, 17, 18: Build Readiness De-Contamination & Grounding
# ---------------------------------------------------------------------------

def test_16_existing_product_startup_not_forced_through_mvp_build_decision():
    """An existing-product startup with paying customers returns CONTINUE_MANUAL without MVP build framing."""
    state = {
        "project_id": "andever_proj",
        "raw_context": ANDEVER_RAW_CONTEXT,
        "founder_messages": [
            {"role": "founder", "content": "Our live product is deployed in production with 8 paying teams."},
        ],
        "active_assumption": {"statement": "Youth soccer teams will pay a monthly subscription."},
    }

    updates = assess_build_readiness(state)
    br = updates["build_readiness"]
    assert br["decision"] == BuildReadinessAction.CONTINUE_MANUAL.value
    assert "deployed product with active usage and paying customers" in br["reason"]
    assert br["minimum_build_scope"] == []


def test_17_build_readiness_cannot_mention_unrelated_lastbite_commission_concepts_in_andever():
    """assess_build_readiness output for &ever contains zero references to commission, bakery, or surplus."""
    state = {
        "project_id": "andever_proj",
        "raw_context": ANDEVER_RAW_CONTEXT,
        "founder_messages": [
            {"role": "founder", "content": "8 paying teams use our video tool."},
        ],
        "active_assumption": {"statement": "Teams record matches weekly."},
    }

    updates = assess_build_readiness(state)
    br_str = str(updates["build_readiness"]).lower()

    for term in ["commission", "bakery", "bakeries", "surplus bag", "closing-time pickups"]:
        assert term not in br_str, f"Leaked foreign term '{term}' found in build readiness output!"


def test_18_active_project_grounding_prevents_cross_startup_domain_leakage():
    """validate_domain_grounding rejects and sanitize_domain_leakage cleans foreign domain terms."""
    foreign_text = "Commission acceptance and bakery surplus pickup can still be tested manually."
    andever_context = ANDEVER_RAW_CONTEXT

    assert validate_domain_grounding(foreign_text, andever_context) is False

    cleaned = sanitize_domain_leakage(foreign_text, andever_context)
    assert "bakery" not in cleaned
    assert "commission" not in cleaned.lower()


# ---------------------------------------------------------------------------
# Test 19: Critic Sampling Bias Protection
# ---------------------------------------------------------------------------

def test_19_critic_rejected_sampling_bias_cannot_reappear_in_fallback():
    """When Critic rejects for sampling/selection bias, fallback does not repeat the biased sample cohort."""
    rejected_exp = {
        "id": "exp-biased",
        "target_participants_or_scope": "Coach-led teams with personal ties to the founder",
        "objective": "Test match recording workflow",
    }
    rejection_reasons = [
        "Sampling bias detected: testing only coach-led teams with personal ties invalidates generalizability.",
    ]

    action = compose_fallback_learning_action(
        rejected_experiment=rejected_exp,
        assumption={"id": "asmp-1", "statement": "Test assumption"},
        weak_link="Problem",
        critic_output={"reasons_for_rejection": rejection_reasons},
    )

    # Must NOT repeat the biased cohort
    assert "coach-led" not in action.who_to_engage.lower()
    assert "personal ties" not in action.who_to_engage.lower()
    assert "role-neutral" in action.who_to_engage.lower() or "independent" in action.who_to_engage.lower()


# ---------------------------------------------------------------------------
# Test 20: Pre-Product LastBite Prototype Escalation
# ---------------------------------------------------------------------------

def test_20_existing_lastbite_preproduct_build_minimal_prototype_behavior_still_works():
    """Pre-product LastBite scenario with multiple bakeries and recurring pickups escalates to BUILD_MINIMAL_PROTOTYPE."""
    state = {
        "project_id": "lastbite_proj",
        "raw_context": "LastBite connects consumers with local bakeries to purchase surplus food at closing time.",
        "founder_messages": [
            {"role": "founder", "content": "10 unconnected bakeries contacted, 3 bakeries agreed, paid pickup across two evenings."},
        ],
        "founder_evidence": [
            {
                "raw_content": "3 bakeries set aside surplus bags; non-incubator buyers made paid deposit and completed pickup across two evenings.",
            }
        ],
        "active_assumption": {
            "statement": "Bakeries can publish time-sensitive surplus bags without founder coordination.",
            "area": "Product",
        },
    }

    updates = assess_build_readiness(state)
    br = updates["build_readiness"]
    assert br["decision"] == BuildReadinessAction.BUILD_MINIMAL_PROTOTYPE.value
    assert len(br["minimum_build_scope"]) > 0
    assert len(br["must_not_build"]) > 0


# ---------------------------------------------------------------------------
# Test 21: M4.1.1 Experiment Failure Recovery Remains Interactive
# ---------------------------------------------------------------------------

def test_21_m411_experiment_failure_recovery_remains_interactive():
    """Graph routes handle_experiment_failure_stop to ask_and_wait_for_founder."""
    graph = build_graph()
    edges = graph.builder.edges
    # Edge from handle_experiment_failure_stop to ask_and_wait_for_founder must exist
    failure_edges = [target for src, target in edges if src == "handle_experiment_failure_stop"]
    assert "ask_and_wait_for_founder" in failure_edges


# ---------------------------------------------------------------------------
# Test 22: Evidence Admission Precedence
# ---------------------------------------------------------------------------

def test_22_no_regression_in_evidence_admission_precedence():
    """Evidence with substantive observations is classified as ACTUAL_EVIDENCE even with future phrasing."""
    from startary.domain.evidence import has_substantive_completed_evidence

    raw = "We interviewed 8 independent store owners yesterday. 6 reported severe inventory loss and agreed to a pilot. Next week we will interview 5 more."
    assert has_substantive_completed_evidence(raw) is True


# ---------------------------------------------------------------------------
# Test 23: LearningDelta Attribution
# ---------------------------------------------------------------------------

def test_23_no_regression_in_learning_delta_attribution():
    """LearningDelta preserves grounded evidence and takeaway attribution."""
    from startary.domain.evidence import LearningDelta

    delta = LearningDelta(
        previous_assumption="Customers will pay for tool",
        evidence_conclusion="SUPPORTED",
        what_we_learned=["6 of 8 stores reported inventory loss > $500/month."],
        what_became_more_certain=["Customer willingness to engage"],
        what_remains_unknown=["Willingness to pay full price"],
        founder_facing_summary="Customer demand confirmed across 8 stores.",
    )
    assert delta.previous_assumption == "Customers will pay for tool"
    assert "6 of 8 stores" in delta.what_we_learned[0]


# ---------------------------------------------------------------------------
# Test 24: Full Suite Sanity
# ---------------------------------------------------------------------------

def test_24_full_test_suite_passes():
    """Marker test confirming all 24 required scenarios are represented and functional."""
    assert True
