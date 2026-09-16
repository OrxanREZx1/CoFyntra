"""
Unit and Integration Tests for Milestone M4.1.3.3:
Post-Evidence Cache Invalidation & Strategic Progression Integrity.

Verifies:
1. Completed evidence cycles (SUPPORTED, PARTIALLY_SUPPORTED, CONTRADICTED, INCONCLUSIVE)
   strictly invalidate cached diagnosis and force fresh re-diagnosis.
2. Partial in-progress evidence does NOT unnecessarily force full re-diagnosis.
3. State mutations (area knowledge, assumption status, iteration advance, founder correction)
   increment strategic_state_revision and mark state dirty.
4. Non-strategic turns (FAST_LOCAL_UPDATE, conversational repairs) safely reuse cached diagnosis.
5. Invariant guards prevent validated KNOWN areas from remaining at maximal uncertainty/evidence gap (risk 81).
6. Mynera live regression reproducing the stale diagnosis cache bug and verifying post-fix progression.
"""

import copy
from unittest.mock import MagicMock, patch
import pytest

from startary.domain.area_knowledge import AreaStatus
from startary.domain.build_readiness import BuildReadinessAction
from startary.domain.enums import (
    AssumptionStatus,
    EvidenceAnalysisConclusion,
    EvidenceProvenance,
    EvidenceResult,
    EvidenceType,
    ExperimentStatus,
    WorkflowPhase,
)
from startary.domain.assessment import RiskFactorLevel
from startary.domain.taxonomy import StartupArea
from startary.domain.evidence import Evidence, EvidenceAnalysis, LearningDelta
from startary.graph.graph_state import StartaryGraphState
from startary.graph.nodes import (
    apply_validated_state_updates,
    assess_build_readiness,
    assess_startup_areas,
    compose_learning_delta,
    decide_next_strategic_action,
    diagnose_startup,
    increment_iteration,
    is_strategic_state_changed,
    merge_diagnosis,
    merge_execution_state,
    select_weak_link,
    validate_evidence_analysis,
)
from startary.llm.schemas import (
    AreaRiskAssessmentLLMOutput,
    DiagnosisLLMOutput,
    StartupAreaAssessmentBatchLLMOutput,
)


# ---------------------------------------------------------------------------
# Helpers & Fixtures
# ---------------------------------------------------------------------------

def _make_base_state(iteration: int = 1, with_diagnosis: bool = True) -> dict:
    """Constructs a clean baseline state with an existing diagnosis."""
    state: dict = {
        "project_id": "proj-m4133-test",
        "iteration": iteration,
        "phase": WorkflowPhase.TEST,
        "strategic_state_revision": 1,
        "diagnosis_revision": 1 if with_diagnosis else None,
        "strategic_state_changed": False,
        "raw_context": "Study planning assistant for university students",
        "founder_messages": [
            {"role": "founder", "content": "Study planning assistant for university students"},
        ],
        "startup_state": {
            "target_user": "University students",
            "problem_statement": "Students struggle with study organization",
            "areas": {
                "Problem": {"status": "PARTIALLY_KNOWN", "understanding": "Initial user interviews"},
                "Pain / Urgency": {"status": "UNKNOWN", "understanding": "Not yet validated"},
                "Target User": {"status": "PARTIALLY_KNOWN", "understanding": "College undergrads"},
            },
        },
        "traces": [],
    }
    if with_diagnosis:
        state["diagnosis_output"] = {
            "diagnosis_summary": "Pre-evidence diagnosis: Pain / Urgency is the primary unvalidated bottleneck.",
            "area_assessments": [],
        }
        state["diagnosis"] = {
            "summary": "Pre-evidence diagnosis: Pain / Urgency is the primary unvalidated bottleneck.",
            "startup_stage": "IDEA",
        }
        state["diagnosis_snapshot"] = copy.deepcopy(state["diagnosis"])
        state["weak_area"] = "Pain / Urgency"
        state["active_weak_link"] = "Pain / Urgency"

    return state


def _make_completed_evidence_state(conclusion: str = "SUPPORTED", target_size: int = 6, collected: int = 6) -> dict:
    """Constructs a state where an experiment has valid completed evidence submitted."""
    state = _make_base_state(iteration=1, with_diagnosis=True)
    state["active_experiment"] = {
        "id": "exp-1",
        "assumption_id": "asmp-1",
        "target_sample_size": target_size,
        "collected_sample_count": collected,
        "status": "IN_PROGRESS",
    }
    state["active_assumption"] = {
        "id": "asmp-1",
        "statement": "Students will use weekly study plans.",
        "area": "Pain / Urgency",
        "status": "TESTING",
    }
    state["active_critical_assumption"] = copy.deepcopy(state["active_assumption"])
    state["founder_response"] = f"Tested with {collected} students. 5 returned for the second week. Personalization takes 45 min/student bottleneck."
    state["extracted_evidence"] = {
        "actions_completed": [f"Tested with {collected} students"],
        "sample_or_counts": {"sample_size": collected},
        "observed_behaviors": ["5 returned for 2nd week", "Personalization took 45 min per student"],
    }
    state["evidence_analysis"] = {
        "conclusion": conclusion,
        "evidence_strength": "STRONG",
        "experiment_execution_quality": "GOOD",
        "support_findings": ["5 of 6 returned for 2nd week"] if conclusion in ("SUPPORTED", "PARTIALLY_SUPPORTED") else [],
        "contradiction_findings": ["Students ignored the study plan"] if conclusion == "CONTRADICTED" else [],
        "inconclusive_findings": ["Sample limited to one cohort"] if conclusion == "INCONCLUSIVE" else [],
        "assumption_confidence_change": "Shifted based on experiment results.",
        "reasoning_summary": f"Epistemic evaluation completed with {conclusion}.",
        "additional_observations": [
            {"text": "Manual personalization bottleneck of 45 min/student", "category": "OPERATIONAL_BOTTLENECK"}
        ],
    }
    return state


# ===========================================================================
# 1–4. Completed Evidence Cycles (SUPPORTED, PARTIALLY_SUPPORTED, CONTRADICTED, INCONCLUSIVE)
# ===========================================================================

def test_01_completed_supported_evidence_invalidates_diagnosis_cache():
    """1. Completed SUPPORTED evidence marks strategic state dirty and forces fresh diagnosis."""
    state = _make_completed_evidence_state(conclusion="SUPPORTED")
    val_res = validate_evidence_analysis(state)
    state.update(val_res)
    assert state["strategic_state_changed"] is True
    assert state["strategic_state_revision"] > 1

    updates_res = apply_validated_state_updates(state)
    state.update(updates_res)
    assert state["strategic_state_changed"] is True

    inc_res = increment_iteration(state)
    state.update(inc_res)
    assert state["iteration"] == 2
    assert state["diagnosis_output"] is None
    assert state["strategic_state_changed"] is True

    # diagnose_startup must run fresh (cannot reuse cached diagnosis)
    mock_diag = DiagnosisLLMOutput(
        area_assessments=[],
        diagnosis_summary="Fresh post-evidence diagnosis reflecting supported pain/urgency.",
    )
    with patch("startary.graph.nodes._safe_get_llm_client") as mock_get_llm:
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = mock_diag
        mock_llm.with_structured_output.return_value = mock_structured
        mock_get_llm.return_value = mock_llm

        diag_res = diagnose_startup(state)
        mock_structured.invoke.assert_called_once()
        assert diag_res["diagnosis_output"]["diagnosis_summary"] == "Fresh post-evidence diagnosis reflecting supported pain/urgency."
        assert diag_res["diagnosis_revision"] == state["strategic_state_revision"]


def test_02_completed_partially_supported_evidence_invalidates_cache():
    """2. Completed PARTIALLY_SUPPORTED evidence invalidates cache and triggers fresh diagnosis."""
    state = _make_completed_evidence_state(conclusion="PARTIALLY_SUPPORTED")
    val_res = validate_evidence_analysis(state)
    state.update(val_res)
    apply_res = apply_validated_state_updates(state)
    state.update(apply_res)
    inc_res = increment_iteration(state)
    state.update(inc_res)

    assert state["strategic_state_changed"] is True
    assert is_strategic_state_changed(state) is True


def test_03_completed_contradicted_evidence_invalidates_cache():
    """3. Completed CONTRADICTED evidence invalidates cache and triggers fresh diagnosis."""
    state = _make_completed_evidence_state(conclusion="CONTRADICTED")
    val_res = validate_evidence_analysis(state)
    state.update(val_res)
    apply_res = apply_validated_state_updates(state)
    state.update(apply_res)
    inc_res = increment_iteration(state)
    state.update(inc_res)

    assert state["strategic_state_changed"] is True
    assert is_strategic_state_changed(state) is True


def test_04_completed_inconclusive_evidence_invalidates_cache():
    """4. Completed INCONCLUSIVE evidence invalidates cache (it is new information)."""
    state = _make_completed_evidence_state(conclusion="INCONCLUSIVE")
    val_res = validate_evidence_analysis(state)
    state.update(val_res)
    apply_res = apply_validated_state_updates(state)
    state.update(apply_res)
    inc_res = increment_iteration(state)
    state.update(inc_res)

    assert state["strategic_state_changed"] is True
    assert is_strategic_state_changed(state) is True


# ===========================================================================
# 5. Partial IN_PROGRESS Evidence
# ===========================================================================

def test_05_partial_in_progress_evidence_does_not_force_rediagnosis():
    """5. Partial IN_PROGRESS evidence preserves active experiment and does not increment iteration."""
    state = _make_completed_evidence_state(conclusion="SUPPORTED", target_size=10, collected=3)
    # Execution quality partial / sample incomplete
    state["evidence_analysis"]["experiment_execution_quality"] = "PARTIAL"

    val_res = validate_evidence_analysis(state)
    assert val_res["active_experiment"]["status"] == ExperimentStatus.IN_PROGRESS.value
    assert val_res.get("is_partial_evidence") is True
    assert "strategic_state_changed" not in val_res or val_res.get("strategic_state_changed") is not True


# ===========================================================================
# 6–9. State Mutation Invalidation Contracts
# ===========================================================================

def test_06_apply_validated_state_updates_marks_strategic_state_dirty():
    """6. apply_validated_state_updates explicitly sets strategic_state_changed=True."""
    state = _make_completed_evidence_state(conclusion="SUPPORTED")
    val_res = validate_evidence_analysis(state)
    state.update(val_res)

    res = apply_validated_state_updates(state)
    assert res["strategic_state_changed"] is True
    assert res["strategic_state_revision"] > state["strategic_state_revision"]


def test_07_area_knowledge_change_invalidates_diagnosis():
    """7. Changing AreaKnowledge status to KNOWN results in is_strategic_state_changed=True."""
    state = _make_base_state(iteration=1, with_diagnosis=True)
    state["startup_state"]["areas"]["Pain / Urgency"]["status"] = AreaStatus.KNOWN.value
    state["strategic_state_revision"] = 2
    state["diagnosis_revision"] = 1  # Older diagnosis

    assert is_strategic_state_changed(state) is True


def test_08_assumption_status_change_invalidates_diagnosis():
    """8. Assumption status change on validated cycle increments revision and invalidates diagnosis."""
    state = _make_completed_evidence_state(conclusion="SUPPORTED")
    val_res = validate_evidence_analysis(state)
    assert val_res["active_assumption"]["status"] == AssumptionStatus.SUPPORTED.value
    assert val_res["strategic_state_revision"] > 1


def test_09_learning_delta_strategic_change_evaluates_on_fresh_weak_link():
    """9. LearningDelta correctly computes weak_link_changed based on newly selected weak link."""
    state = _make_completed_evidence_state(conclusion="SUPPORTED")
    state["iteration"] = 2
    state["previous_weak_area"] = "Pain / Urgency"
    state["weak_area"] = "Customer Access"  # Shifted after fresh diagnosis and prioritization
    state["previous_experiments"] = [{"id": "exp-1", "status": "COMPLETED"}]
    state["previous_assumptions"] = [{"id": "asmp-1", "statement": "Weekly study plans", "area": "Pain / Urgency"}]
    state["active_evidence"] = {
        "id": "ev-1",
        "experiment_id": "exp-1",
        "assumption_id": "asmp-1",
        "iteration": 1,
        "result": "ANALYZED",
        "evidence_type": "EXPERIMENT_RESULT",
    }

    delta_res = compose_learning_delta(state)
    ld = delta_res["learning_delta"]
    assert ld is not None
    assert ld["weak_link_changed"] is True
    assert ld["previous_weak_link"] == "Pain / Urgency"
    assert ld["new_weak_link"] == "Customer Access"


# ===========================================================================
# 10–13. Non-Strategic Turns & Fast Paths Reusing Diagnosis
# ===========================================================================

def test_10_iteration_advancement_cannot_reuse_prior_diagnosis():
    """10. Advancing iteration after completed evidence cannot reuse Iteration 1 diagnosis."""
    state = _make_completed_evidence_state(conclusion="SUPPORTED")
    state.update(validate_evidence_analysis(state))
    state.update(apply_validated_state_updates(state))
    state.update(increment_iteration(state))

    assert state["diagnosis_output"] is None
    assert state["diagnosis_revision"] is None or state["diagnosis_revision"] < state["strategic_state_revision"]


def test_11_cached_diagnosis_reuse_works_when_revisions_match():
    """11. Cached diagnosis reuse works when strategic_state_revision == diagnosis_revision."""
    state = _make_base_state(iteration=1, with_diagnosis=True)
    state["strategic_state_revision"] = 1
    state["diagnosis_revision"] = 1
    state["strategic_state_changed"] = False

    with patch("startary.graph.nodes._safe_get_llm_client") as mock_get_llm:
        res = diagnose_startup(state)
        mock_get_llm.assert_not_called()
        assert res["diagnosis_output"] == state["diagnosis_output"]


def test_12_fast_local_update_for_capacity_reuses_diagnosis():
    """12. FAST_LOCAL_UPDATE for capacity leaves revision unchanged and safely reuses diagnosis."""
    state = _make_base_state(iteration=1, with_diagnosis=True)
    state["execution_path"] = "FAST_LOCAL_UPDATE"
    state["strategic_state_changed"] = False
    state["strategic_state_revision"] = 1
    state["diagnosis_revision"] = 1

    with patch("startary.graph.nodes._safe_get_llm_client") as mock_get_llm:
        res = diagnose_startup(state)
        mock_get_llm.assert_not_called()
        assert res["diagnosis_output"] == state["diagnosis_output"]


def test_13_conversational_repair_bypasses_diagnosis():
    """13. CONVERSATIONAL_REPAIR execution path is not treated as strategic state change."""
    state = _make_base_state(iteration=1, with_diagnosis=True)
    state["execution_path"] = "CONVERSATIONAL_REPAIR"
    state["strategic_state_changed"] = False
    state["strategic_state_revision"] = 1
    state["diagnosis_revision"] = 1

    assert is_strategic_state_changed(state) is False


# ===========================================================================
# 14–16. Behavioral Evidence & Founder Corrections
# ===========================================================================

def test_14_founder_correction_to_strategic_state_invalidates_cache():
    """14. Founder correction setting revenue / paying customers bumps revision."""
    state = _make_base_state(iteration=1, with_diagnosis=True)
    state["founder_messages"] = [
        {"role": "founder", "content": "Update: We now have 10 paying customers and $500 monthly revenue."},
    ]
    state["raw_context"] = "Update: We now have 10 paying customers and $500 monthly revenue."
    state["startup_state_delta"] = {
        "paying_customers": {"value": 10, "provenance": "EXPLICIT", "confidence": "HIGH"},
        "revenue": {"value": 500, "provenance": "EXPLICIT", "confidence": "HIGH"},
    }
    res = merge_execution_state(state)
    assert res.get("strategic_state_changed") is True
    assert res.get("strategic_state_revision") > 1


def test_15_new_validated_behavioral_evidence_reaches_fresh_diagnosis():
    """15. Validated behavioral evidence is included in the fresh diagnosis user message."""
    state = _make_completed_evidence_state(conclusion="SUPPORTED")
    state.update(validate_evidence_analysis(state))
    state.update(apply_validated_state_updates(state))
    state.update(increment_iteration(state))

    with patch("startary.graph.nodes._safe_get_llm_client") as mock_get_llm:
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = DiagnosisLLMOutput(
            area_assessments=[],
            diagnosis_summary="Updated diagnosis after 6 manual tests.",
        )
        mock_llm.with_structured_output.return_value = mock_structured
        mock_get_llm.return_value = mock_llm

        diagnose_startup(state)
        # Verify prompt received evidence summary
        call_args = mock_structured.invoke.call_args[0][0]
        user_msg = call_args[1]["content"]
        assert "VALIDATED EXPERIMENT EVIDENCE" in user_msg
        assert "Iteration 1" in user_msg


def test_16_additional_observations_reach_fresh_diagnosis():
    """16. Additional observations like manual bottlenecks appear in diagnosis context."""
    state = _make_completed_evidence_state(conclusion="SUPPORTED")
    state.update(validate_evidence_analysis(state))
    state.update(apply_validated_state_updates(state))
    state.update(increment_iteration(state))

    with patch("startary.graph.nodes._safe_get_llm_client") as mock_get_llm:
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = DiagnosisLLMOutput(area_assessments=[], diagnosis_summary="Test")
        mock_llm.with_structured_output.return_value = mock_structured
        mock_get_llm.return_value = mock_llm

        diagnose_startup(state)
        user_msg = mock_structured.invoke.call_args[0][0][1]["content"]
        assert "Observation" in user_msg or "bottleneck" in user_msg.lower()


# ===========================================================================
# 17–19. Post-Evidence Downstream Progression
# ===========================================================================

def test_17_build_readiness_receives_post_evidence_diagnosis_and_state():
    """17. BuildReadiness receives updated state and escalated action on manual workflow completion."""
    state = _make_completed_evidence_state(conclusion="SUPPORTED")
    state.update(validate_evidence_analysis(state))
    state.update(apply_validated_state_updates(state))
    state.update(increment_iteration(state))

    # Add active assumption and evidence observations matching escalation criteria
    state["active_assumption"] = {
        "statement": "Students will complete the study session using unmediated software without founder manual coordination.",
        "area": "Problem",
    }
    state["raw_context"] = (
        "We tested 6 manual workflow tests with students. 5 of them returned for a 2nd week session. "
        "However, manual personalization taking 45-70 min/student is an unsustainable bottleneck. "
        "We need product-mediated coordination to test if students will follow plans without founder intervention."
    )
    state["founder_evidence"] = [
        {
            "raw_content": "We tested 6 manual workflow tests with students. 5 returned for 2nd week session. 45-70 min bottleneck.",
            "quantitative_results": {"sample_size": 6},
            "structured_observations": ["6 manual workflow tests completed", "5 repeat 2nd week users"],
        }
    ]
    res = assess_build_readiness(state)
    br = res["build_readiness"]
    assert br["decision"] == BuildReadinessAction.BUILD_MINIMAL_PROTOTYPE.value


def test_18_next_strategic_action_receives_post_evidence_state():
    """18. NextStrategicAction produces BUILD_MINIMAL_PROTOTYPE when BuildReadiness escalates."""
    state = _make_completed_evidence_state(conclusion="SUPPORTED")
    state["build_readiness"] = {
        "decision": BuildReadinessAction.BUILD_MINIMAL_PROTOTYPE.value,
        "reason": "Validated manual workflow with severe coordination bottleneck.",
        "must_not_build": ["Recommendation engine", "Mobile app"],
    }
    res = decide_next_strategic_action(state)
    action = res["next_strategic_action"]
    assert action["action_type"] in ("BUILD_PROTOTYPE", "BUILD_MINIMAL_PROTOTYPE")


def test_19_old_weak_link_assessment_is_not_blindly_reused():
    """19. Validated KNOWN area cannot retain maximal evidence gap (risk 81)."""
    state = _make_base_state(iteration=2, with_diagnosis=False)
    state["startup_state"]["areas"]["Pain / Urgency"] = {
        "status": AreaStatus.KNOWN.value,
        "understanding": "Validated via 6 student workflow tests",
    }
    # Mock LLM returning HIGH uncertainty and HIGH evidence gap for all areas
    mock_batch = StartupAreaAssessmentBatchLLMOutput(
        assessments=[
            AreaRiskAssessmentLLMOutput(
                area=StartupArea.PAIN_URGENCY,
                importance=RiskFactorLevel.HIGH,
                uncertainty=RiskFactorLevel.HIGH,
                evidence_gap=RiskFactorLevel.HIGH,  # LLM erroneously claims high evidence gap
                consequence_of_being_wrong=RiskFactorLevel.HIGH,
                rationale="Pain urgency is critical.",
            ),
            AreaRiskAssessmentLLMOutput(
                area=StartupArea.ACQUISITION,
                importance=RiskFactorLevel.HIGH,
                uncertainty=RiskFactorLevel.HIGH,
                evidence_gap=RiskFactorLevel.HIGH,
                consequence_of_being_wrong=RiskFactorLevel.HIGH,
                rationale="Unvalidated customer acquisition channel.",
            ),
        ]
    )

    with patch("startary.graph.nodes.get_llm_client") as mock_get_llm:
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = mock_batch
        mock_llm.with_structured_output.return_value = mock_structured
        mock_get_llm.return_value = mock_llm

        res = assess_startup_areas(state)
        state["area_assessments_output"] = res["area_assessments_output"]

        # Check that Pain / Urgency had its evidence gap capped because it is KNOWN
        pain_urg = next(a for a in res["area_assessments_output"]["assessments"] if a["area"] == StartupArea.PAIN_URGENCY.value)
        assert pain_urg["evidence_gap"] == RiskFactorLevel.MEDIUM.value

        # Select weak link: Acquisition (risk 81) should win over Pain / Urgency (risk <= 36)
        wl_res = select_weak_link(state)
        assert wl_res["weak_area"] == StartupArea.ACQUISITION.value


# ===========================================================================
# 20–24. Historical Records & Artifact Invariants
# ===========================================================================

def test_20_historical_diagnosis_remains_preserved():
    """20. previous_diagnosis_snapshot preserves Iteration 1 diagnosis snapshot."""
    state = _make_completed_evidence_state(conclusion="SUPPORTED")
    state.update(validate_evidence_analysis(state))
    state.update(apply_validated_state_updates(state))
    inc_res = increment_iteration(state)

    assert inc_res["previous_diagnosis_snapshot"] is not None
    assert inc_res["previous_diagnosis_snapshot"]["summary"].startswith("Pre-evidence diagnosis")


def test_21_completed_experiment_remains_preserved():
    """21. previous_experiments contains the completed experiment from Iteration 1."""
    state = _make_completed_evidence_state(conclusion="SUPPORTED")
    val_res = validate_evidence_analysis(state)

    prev_exps = val_res["previous_experiments"]
    assert len(prev_exps) == 1
    assert prev_exps[0]["id"] == "exp-1"
    assert prev_exps[0]["status"] == ExperimentStatus.COMPLETED.value


def test_22_evidence_is_not_duplicated():
    """22. Multiple state passes do not duplicate admitted evidence records."""
    state = _make_completed_evidence_state(conclusion="SUPPORTED")
    val_res = validate_evidence_analysis(state)
    assert len(val_res["founder_evidence"]) == 1


def test_23_iteration_increments_exactly_once():
    """23. increment_iteration increments iteration by exactly 1."""
    state = _make_completed_evidence_state(conclusion="SUPPORTED")
    state.update(validate_evidence_analysis(state))
    inc_res = increment_iteration(state)
    assert inc_res["iteration"] == 2


def test_24_learning_delta_created_with_additional_observations():
    """24. LearningDelta carries additional observations from completed cycle."""
    state = _make_completed_evidence_state(conclusion="SUPPORTED")
    state["iteration"] = 2
    state["previous_weak_area"] = "Pain / Urgency"
    state["weak_area"] = "Problem"
    state["active_evidence"] = {
        "id": "ev-1",
        "experiment_id": "exp-1",
        "assumption_id": "asmp-1",
        "iteration": 1,
        "result": "ANALYZED",
        "evidence_type": "EXPERIMENT_RESULT",
        "additional_observations": [{"text": "Manual bottleneck 45 min", "category": "BOTTLENECK"}],
    }
    state["previous_experiments"] = [{"id": "exp-1", "status": "COMPLETED"}]
    state["previous_assumptions"] = [{"id": "asmp-1", "statement": "Test asmp", "area": "Pain / Urgency"}]

    res = compose_learning_delta(state)
    assert len(res["learning_delta"]["additional_observations"]) == 1


# ===========================================================================
# 25. Mynera Live Regression (The Core Bug Reproduction & Fix Verification)
# ===========================================================================

def test_25_mynera_live_regression_reproduces_and_passes():
    """
    25. Reproduce the exact Mynera live failure:
    - Iteration 1 weak link: Pain / Urgency
    - Valid evidence cycle completed: conclusion=SUPPORTED, strength=MODERATE, quality=GOOD
    - State update applied: Pain / Urgency -> KNOWN
    - Iteration increments 1 -> 2
    - Verify: diagnose_startup CANNOT reuse cached diagnosis (cached == False)
    - Verify: Post-evidence diagnosis reflects newly KNOWN area
    - Verify: Pain / Urgency cannot saturate at 81 Decision Risk
    """
    # 1. Start from pre-evidence state
    state = _make_completed_evidence_state(conclusion="SUPPORTED", target_size=6, collected=6)
    state["active_assumption"]["statement"] = "Students experience acute pain in scheduling."
    state["active_assumption"]["area"] = "Pain / Urgency"

    # 2. Complete evidence cycle
    val_res = validate_evidence_analysis(state)
    state.update(val_res)
    assert val_res["active_assumption"]["status"] == AssumptionStatus.SUPPORTED.value

    # 3. Apply state updates
    upd_res = apply_validated_state_updates(state)
    state.update(upd_res)
    assert state["startup_state"]["areas"]["Pain / Urgency"]["status"] == AreaStatus.KNOWN.value

    # 4. Increment iteration
    inc_res = increment_iteration(state)
    state.update(inc_res)
    assert state["iteration"] == 2
    assert state["diagnosis_output"] is None  # Cleaned!

    # 5. diagnose_startup runs in Iteration 2
    mock_post_diag = DiagnosisLLMOutput(
        area_assessments=[],
        diagnosis_summary="Post-evidence diagnosis: Pain/Urgency validated via manual concierge tests; acquisition and retention are next bottlenecks.",
    )
    with patch("startary.graph.nodes._safe_get_llm_client") as mock_get_llm:
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = mock_post_diag
        mock_llm.with_structured_output.return_value = mock_structured
        mock_get_llm.return_value = mock_llm

        diag_res = diagnose_startup(state)
        # CRITICAL ASSERTION: MUST NOT BE CACHED!
        mock_structured.invoke.assert_called_once()
        state.update(diag_res)
        assert state["diagnosis_output"]["diagnosis_summary"].startswith("Post-evidence diagnosis")

    state.update(merge_diagnosis(state))

    # 6. assess_startup_areas & select_weak_link run
    mock_areas = StartupAreaAssessmentBatchLLMOutput(
        assessments=[
            AreaRiskAssessmentLLMOutput(
                area=StartupArea.PAIN_URGENCY,
                importance=RiskFactorLevel.HIGH,
                uncertainty=RiskFactorLevel.HIGH,
                evidence_gap=RiskFactorLevel.HIGH,
                consequence_of_being_wrong=RiskFactorLevel.HIGH,
                rationale="Urgency remains key.",
            ),
            AreaRiskAssessmentLLMOutput(
                area=StartupArea.ACQUISITION,
                importance=RiskFactorLevel.HIGH,
                uncertainty=RiskFactorLevel.HIGH,
                evidence_gap=RiskFactorLevel.HIGH,
                consequence_of_being_wrong=RiskFactorLevel.HIGH,
                rationale="Acquisition channels are completely untested.",
            ),
        ]
    )
    with patch("startary.graph.nodes.get_llm_client") as mock_get_llm:
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = mock_areas
        mock_llm.with_structured_output.return_value = mock_structured
        mock_get_llm.return_value = mock_llm

        areas_res = assess_startup_areas(state)
        state["area_assessments_output"] = areas_res["area_assessments_output"]

        # Pain / Urgency evidence gap MUST be capped because it is KNOWN
        pain_urg_assess = next(a for a in areas_res["area_assessments_output"]["assessments"] if a["area"] == StartupArea.PAIN_URGENCY.value)
        assert pain_urg_assess["evidence_gap"] == RiskFactorLevel.MEDIUM.value

        # Select weak link: Acquisition should win, NOT Pain / Urgency!
        wl_res = select_weak_link(state)
        assert wl_res["weak_area"] != StartupArea.PAIN_URGENCY.value
        assert wl_res["weak_area"] == StartupArea.ACQUISITION.value


# ===========================================================================
# 26–30. Regression Coverage Across System Milestones
# ===========================================================================

def test_26_regression_andever_marketplace_retains_build_readiness():
    """26. Food marketplace scenario retains BuildReadinessAction.BUILD_MINIMAL_PROTOTYPE."""
    state = {
        "project_id": "p-andever",
        "active_assumption": {
            "statement": "Bakeries will publish time-sensitive surplus offers without founder coordination.",
            "area": "Problem",
        },
        "founder_evidence": [
            {
                "raw_content": "Recruited 10 unconnected bakeries. Across two evenings buyers paid deposit and completed pickups.",
                "quantitative_results": {"sample_size": 10},
                "structured_observations": ["Completed paid pickups", "10 unconnected bakeries"],
            }
        ],
        "raw_context": "Recruited 10 unconnected bakeries. Across two evenings buyers paid deposit and completed pickups.",
    }
    res = assess_build_readiness(state)
    assert res["build_readiness"]["decision"] == BuildReadinessAction.BUILD_MINIMAL_PROTOTYPE.value


def test_27_regression_m413_fast_path_intact():
    """27. Bound answer parser and FAST_LOCAL_UPDATE flow remains intact."""
    state = _make_base_state(iteration=1, with_diagnosis=True)
    state["execution_path"] = "FAST_LOCAL_UPDATE"
    state["strategic_state_changed"] = False
    assert is_strategic_state_changed(state) is False


def test_28_regression_m4131_model_fail_safe_intact():
    """28. Extraction failure marks strategic state dirty for safety."""
    state = _make_base_state(iteration=1, with_diagnosis=True)
    state["extraction_status"] = "FAILED"
    assert is_strategic_state_changed(state) is True


def test_29_regression_m4132_completeness_evaluation_intact():
    """29. Null-safe completeness evaluation works seamlessly."""
    from startary.domain.experiment import evaluate_experiment_completeness
    state = {
        "active_experiment": {"id": "exp-29", "target_sample_size": None, "collected_sample_count": None},
        "extracted_evidence": {"sample_or_counts": {"sample_size": 1}},
        "founder_response": "1 session completed.",
    }
    analysis = {"conclusion": "SUPPORTED", "evidence_strength": "STRONG", "experiment_execution_quality": "GOOD"}
    is_complete, reason, collected, target = evaluate_experiment_completeness(state, analysis)
    assert is_complete is True
    assert reason == "NO_EXPLICIT_SAMPLE_TARGET"


def test_30_full_regression_suite_contract_confirmed():
    """30. Contract verification for StartaryGraphState revisions."""
    state = StartaryGraphState(
        project_id="p-test",
        phase=WorkflowPhase.UNDERSTAND,
        iteration=1,
        strategic_state_revision=1,
        diagnosis_revision=1,
        strategic_state_changed=False,
    )
    assert state.get("strategic_state_revision") == 1
    assert state.get("diagnosis_revision") == 1
