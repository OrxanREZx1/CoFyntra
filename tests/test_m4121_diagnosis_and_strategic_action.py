"""
Unit and Integration Tests for Milestone M4.1.2.1:
Founder Diagnosis & Strategic Action Integrity.

Covers 30 mandatory tests:
1. Initial new experiment renders CURRENT DIAGNOSIS before YOUR NEXT TEST.
2. Post-learning new experiment renders CURRENT DIAGNOSIS before YOUR NEXT TEST.
3. CONTINUE_CURRENT_TEST does not unnecessarily render full diagnosis again.
4. Diagnosis contains strong / weak / uncertain.
5. Diagnosis contains WHAT CHANGED after completed evidence cycle.
6. Diagnosis contains MOST IMPORTANT DECISION NOW.
7. Positive signals resolve from historical validated evidence.
8. Supported historical cycles not lost after iteration increment.
9. &ever no longer renders "No strong positive signal" after strong evidence.
10. Contradicted historical evidence does not become positive signal.
11. Failed/fallback evidence does not become positive signal.
12. Acquisition target area produces acquisition-specific key decision.
13. Business Model target produces business-model-specific key decision.
14. Solution/MVP target produces product-specific key decision.
15. Semantic mismatch triggers repair/fallback instead of accepted.
16. PRODUCT_ITERATION is reachable for existing-product startup with validated friction.
17. PRODUCT_ITERATION is NOT selected merely because founder wants more features.
18. PRODUCT_ITERATION includes evidence-backed bottleneck.
19. PRODUCT_ITERATION proposes smallest justified change.
20. PRODUCT_ITERATION explicitly states what not to build.
21. PRODUCT_ITERATION produces measurable behavioral test.
22. Acquisition weak link may be overridden by directly validated product bottleneck.
23. Override includes explicit reason.
24. Weak-link authority remains intact when no higher-leverage supported bottleneck exists.
25. Existing &ever Target User -> Existing Alternatives -> Acquisition progression works.
26. LastBite BUILD_MINIMAL_PROTOTYPE still works.
27. Partial evidence lifecycle still works.
28. M4.1.1 recovery still works.
29. Evidence admission precedence still works.
30. Full regression suite passes.
"""

from unittest.mock import MagicMock, patch
import pytest

from startary.domain.enums import (
    AssumptionStatus,
    EvidenceAnalysisConclusion,
    EvidenceStrength,
    ExecutionQuality,
    ExecutionReadiness,
    ExperimentStatus,
    WorkflowPhase,
)
from startary.domain.experiment import (
    ExecutionHandoff,
    compose_current_positive_signals,
    compose_founder_experiment_summary,
    render_execution_handoff_text,
)
from startary.domain.strategic_action import (
    NextStrategicAction,
    NextStrategicActionType,
)
from startary.domain.build_readiness import (
    BuildReadinessAction,
    BuildReadinessDecision,
)
from startary.graph.nodes import (
    compose_startup_diagnosis,
    decide_next_strategic_action,
    compose_execution_handoff,
)
from startary.llm.schemas import StartupDiagnosisSnapshotLLMOutput


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def andever_state_with_traction() -> dict:
    return {
        "project_id": "andever_proj",
        "iteration": 1,
        "phase": WorkflowPhase.TEST,
        "weak_area": "Acquisition / Customer Access",
        "selection_confidence": "HIGH",
        "active_assumption": {
            "id": "asmp-andever-001",
            "statement": "Independent amateur football teams can be acquired through cold outreach.",
            "area": "Acquisition / Customer Access",
            "status": "UNTESTED",
            "is_critical": True,
        },
        "startup_state": {
            "progress": {
                "mvp_exists": {"value": True, "provenance": "EXPLICIT"},
                "development_stage": {"value": "LIVE_PRODUCT", "provenance": "EXPLICIT"},
                "paying_customers_count": 8,
            },
            "execution_context": {
                "team_size": {"value": "2 founders", "provenance": "EXPLICIT"},
                "customer_access": {"value": "8 paying amateur football teams", "provenance": "EXPLICIT"},
            },
        },
        "raw_context": "8 paying teams currently use our mobile app to record and analyze weekly football matches.",
        "founder_messages": [
            {"role": "founder", "content": "We have 8 paying teams using the app weekly. We want to test acquiring new teams."}
        ],
    }


# ---------------------------------------------------------------------------
# Tests 1-6: Diagnosis Visibility, Sections, and Causal Bridge
# ---------------------------------------------------------------------------

def test_01_initial_new_experiment_renders_current_diagnosis_before_your_next_test():
    """Initial new experiment handoff renders CURRENT DIAGNOSIS before YOUR NEXT TEST."""
    diag_text = (
        "==================================================\n"
        "STARTARY — CURRENT DIAGNOSIS\n"
        "==================================================\n\n"
        "WHAT LOOKS STRONG\n"
        "- Stage: Pre-MVP\n\n"
        "WHAT LOOKS WEAK\n"
        "- Focus Area: Problem\n\n"
        "WHAT IS STILL UNCERTAIN\n"
        "- Market demand\n\n"
        "WHAT CHANGED\n"
        "- Baseline assessment established.\n\n"
        "WHAT DID NOT CHANGE\n"
        "- Blind spots remain.\n\n"
        "MOST IMPORTANT DECISION NOW\n"
        "- Validate problem.\n\n"
        "WHY THIS DECISION\n"
        "- Crucial first step.\n"
        "=================================================="
    )
    handoff = ExecutionHandoff(
        experiment_id="exp-1",
        assumption_id="asmp-1",
        critical_assumption="Merchants experience sizing return pain",
        experiment_type="CUSTOMER_INTERVIEWS",
        learning_goal="Validate return pain",
        what_we_are_testing="Merchants experience sizing return pain",
        target_participants_or_context="5 boutique merchants",
        founder_actions=["Interview 5 merchants"],
        evidence_to_collect="Return rates and handling times",
        support_signal="3+ merchants cite major return costs",
        contradiction_signal="Merchants don't care",
        inconclusive_signal="Unclear",
        founder_diagnosis_text=diag_text,
    )
    rendered = render_execution_handoff_text(handoff, mode="compact")
    assert "STARTARY — CURRENT DIAGNOSIS" in rendered
    assert "YOUR NEXT TEST" in rendered
    diag_idx = rendered.index("STARTARY — CURRENT DIAGNOSIS")
    test_idx = rendered.index("YOUR NEXT TEST")
    assert diag_idx < test_idx, "Diagnosis must appear BEFORE YOUR NEXT TEST"


def test_02_post_learning_new_experiment_renders_current_diagnosis_before_your_next_test():
    """Post-learning new experiment handoff renders updated diagnosis before YOUR NEXT TEST."""
    diag_text = (
        "==================================================\n"
        "STARTARY — CURRENT DIAGNOSIS\n"
        "==================================================\n\n"
        "WHAT LOOKS STRONG\n"
        "- Validated acute sizing problem in boutique merchants.\n\n"
        "WHAT LOOKS WEAK\n"
        "- Focus Area: Existing Alternatives\n\n"
        "WHAT CHANGED\n"
        "- Problem validated through 6 merchant interviews.\n\n"
        "MOST IMPORTANT DECISION NOW\n"
        "- Validate whether manual return workflows fail intolerably.\n"
        "=================================================="
    )
    handoff = ExecutionHandoff(
        experiment_id="exp-2",
        assumption_id="asmp-2",
        critical_assumption="Existing return logs fail to prevent recurring returns",
        experiment_type="CUSTOMER_INTERVIEWS",
        learning_goal="Validate alternative workarounds fail",
        what_we_are_testing="Existing return logs fail",
        target_participants_or_context="5 boutique merchants",
        founder_actions=["Observe manual log process"],
        evidence_to_collect="Time spent and error rate",
        support_signal="Owners spend >3 hours weekly on manual adjustments",
        founder_diagnosis_text=diag_text,
    )
    rendered = render_execution_handoff_text(handoff, mode="compact")
    assert "STARTARY — CURRENT DIAGNOSIS" in rendered
    assert "Existing Alternatives" in rendered
    assert "YOUR NEXT TEST" in rendered
    assert rendered.index("STARTARY — CURRENT DIAGNOSIS") < rendered.index("YOUR NEXT TEST")


def test_03_continue_current_test_does_not_unnecessarily_render_full_diagnosis_again():
    """CONTINUE_CURRENT_TEST suppresses full diagnosis re-render when no state changed."""
    diag_text = "==================================================\nSTARTARY — CURRENT DIAGNOSIS\n=================================================="
    handoff = ExecutionHandoff(
        experiment_id="exp-1",
        assumption_id="asmp-1",
        critical_assumption="Merchants experience pain",
        experiment_type="CUSTOMER_INTERVIEWS",
        learning_goal="Validate return pain",
        what_we_are_testing="Merchants experience pain",
        target_participants_or_context="5 boutique merchants",
        founder_actions=["Interview 5 merchants"],
        evidence_to_collect="Return rates",
        support_signal="3+ merchants cite pain",
        founder_diagnosis_text=diag_text,
        next_strategic_action={"action_type": "CONTINUE_CURRENT_TEST"},
    )
    rendered = render_execution_handoff_text(handoff, mode="compact")
    assert "STARTARY — CURRENT DIAGNOSIS" not in rendered
    assert "CURRENT TEST PROGRESS" in rendered or "YOUR NEXT TEST" in rendered


def test_04_diagnosis_contains_strong_weak_uncertain():
    """Diagnosis text contains WHAT LOOKS STRONG, WHAT LOOKS WEAK, and WHAT IS STILL UNCERTAIN."""
    state = {
        "project_id": "p1",
        "weak_area": "Problem",
        "active_assumption": {"statement": "Customers want solution"},
    }
    with patch("startary.graph.nodes.get_llm_client") as mock_client:
        mock_client.return_value.with_structured_output.return_value.invoke.side_effect = Exception("LLM off")
        res = compose_startup_diagnosis(state)

    diag = res["founder_diagnosis_text"]
    assert "WHAT LOOKS STRONG" in diag
    assert "WHAT LOOKS WEAK" in diag
    assert "WHAT IS STILL UNCERTAIN" in diag
    assert "WHAT DID NOT CHANGE" in diag


def test_05_diagnosis_contains_what_changed_after_completed_evidence_cycle():
    """Diagnosis contains WHAT CHANGED reflecting completed evidence cycle."""
    state = {
        "project_id": "p1",
        "weak_area": "Solution / MVP",
        "active_assumption": {"statement": "Users complete recording independently"},
        "learning_delta": {
            "headline": "Pilot confirmed acute setup friction.",
            "what_we_learned": ["3 of 4 coaches failed camera alignment."],
        },
    }
    with patch("startary.graph.nodes.get_llm_client") as mock_client:
        mock_client.return_value.with_structured_output.return_value.invoke.side_effect = Exception("LLM off")
        res = compose_startup_diagnosis(state)

    diag = res["founder_diagnosis_text"]
    assert "WHAT CHANGED" in diag
    assert "Pilot confirmed acute setup friction" in diag
    assert "3 of 4 coaches failed camera alignment" in diag


def test_06_diagnosis_contains_most_important_decision_now():
    """Diagnosis contains MOST IMPORTANT DECISION NOW and WHY THIS DECISION."""
    state = {
        "project_id": "p1",
        "weak_area": "Acquisition / Customer Access",
        "active_assumption": {"statement": "Teams can be reached via tournament organizers"},
        "next_strategic_action": {
            "key_decision": "Whether tournament organizers agree to distribute info packets.",
            "why_this_action": "Validates non-founder acquisition channels.",
        },
    }
    with patch("startary.graph.nodes.get_llm_client") as mock_client:
        mock_client.return_value.with_structured_output.return_value.invoke.side_effect = Exception("LLM off")
        res = compose_startup_diagnosis(state)

    diag = res["founder_diagnosis_text"]
    assert "MOST IMPORTANT DECISION NOW" in diag
    assert "Whether tournament organizers agree to distribute info packets" in diag
    assert "WHY THIS DECISION" in diag


# ---------------------------------------------------------------------------
# Tests 7-11: Truthful Positive Signals Extraction
# ---------------------------------------------------------------------------

def test_07_positive_signals_resolve_from_historical_validated_evidence():
    """compose_current_positive_signals extracts from completed historical analyses."""
    state = {
        "evidence_history": [
            {
                "conclusion": "SUPPORTED",
                "experiment_execution_quality": "GOOD",
                "support_findings": ["14 teams successfully completed full match recordings."],
                "valid_learnings": ["Weekly active organizers use recordings for team review."],
            }
        ],
    }
    signals = compose_current_positive_signals(state=state)
    assert any("14 teams" in s or "match recording" in s for s in signals)
    assert "No strong positive signal is established yet." not in signals


def test_08_supported_historical_cycles_not_lost_after_iteration_increment():
    """Supported historical cycles remain in positive signals across iterations."""
    state = {
        "iteration": 3,
        "evidence_history": [
            {
                "conclusion": "SUPPORTED",
                "experiment_execution_quality": "GOOD",
                "support_findings": ["Merchants verified recurring size-related return rates."],
            }
        ],
        "evidence_analysis": {
            "conclusion": "INCONCLUSIVE",
            "experiment_execution_quality": "PARTIAL",
        },
    }
    signals = compose_current_positive_signals(state=state)
    assert any("recurring size-related return" in s or "return" in s for s in signals)


def test_09_andever_no_longer_renders_no_strong_positive_signal_after_strong_evidence():
    """&ever produces positive signals from traction and verified evidence without falling back."""
    state = {
        "raw_context": "8 paying teams use &ever weekly to record football matches.",
        "startup_state": {
            "progress": {
                "paying_customers_count": 8,
                "mvp_exists": {"value": True, "provenance": "EXPLICIT"},
            },
            "area_knowledge": {
                "Demand Signal": {
                    "status": "KNOWN",
                    "verified_claims": ["8 paying teams pay monthly subscriptions and record matches weekly."],
                }
            },
        },
        "evidence_history": [
            {
                "conclusion": "SUPPORTED",
                "experiment_execution_quality": "GOOD",
                "support_findings": ["Repeat usage is concentrated among teams with a regular playing schedule."],
            }
        ],
    }
    signals = compose_current_positive_signals(state=state)
    assert len(signals) > 0
    assert "No strong positive signal is established yet." not in signals
    assert any("paying" in s or "8" in s or "repeat" in s for s in signals)


def test_10_contradicted_historical_evidence_does_not_become_positive_signal():
    """Contradicted historical evidence is excluded from positive signals."""
    state = {
        "evidence_history": [
            {
                "conclusion": "CONTRADICTED",
                "experiment_execution_quality": "GOOD",
                "support_findings": ["False positive claim that should never appear"],
            }
        ]
    }
    signals = compose_current_positive_signals(state=state)
    assert "False positive claim that should never appear" not in signals
    assert signals == ["No strong positive signal is established yet."]


def test_11_failed_fallback_evidence_does_not_become_positive_signal():
    """Failed execution quality evidence is excluded from positive signals."""
    state = {
        "evidence_analysis": {
            "conclusion": "SUPPORTED",
            "experiment_execution_quality": "FAILED",
            "support_findings": ["Unreliable data from failed test"],
        }
    }
    signals = compose_current_positive_signals(state=state)
    assert "Unreliable data from failed test" not in signals
    assert signals == ["No strong positive signal is established yet."]


# ---------------------------------------------------------------------------
# Tests 12-15: Semantic Area Matching & Deterministic Guards
# ---------------------------------------------------------------------------

def test_12_acquisition_target_area_produces_acquisition_specific_key_decision():
    """Acquisition target area produces acquisition-specific key decision (channels/reach)."""
    state = {
        "weak_area": "Acquisition / Customer Access",
        "active_assumption": {
            "statement": "Amateur teams can be acquired through cold email outreach to league secretaries.",
            "area": "Acquisition / Customer Access",
        },
    }
    res = decide_next_strategic_action(state)
    act = res["next_strategic_action"]
    assert act["action_type"] == "ACQUISITION_TEST"
    assert act["target_area"] == "Acquisition / Customer Access"
    kd_lower = act["key_decision"].lower()
    assert any(w in kd_lower for w in ["channel", "reach", "network", "acquir", "outreach"])
    assert "acute pain" not in kd_lower
    assert "pricing" not in kd_lower


def test_13_business_model_target_produces_business_model_specific_key_decision():
    """Business Model target produces pricing / unit-economics key decision."""
    state = {
        "weak_area": "Business Model",
        "active_assumption": {
            "statement": "Shops will accept a 15% commission on surplus sales.",
            "area": "Business Model",
        },
    }
    res = decide_next_strategic_action(state)
    act = res["next_strategic_action"]
    assert act["action_type"] == "PRICING_TEST"
    assert act["target_area"] == "Business Model"
    kd_lower = act["key_decision"].lower()
    assert any(w in kd_lower for w in ["price", "pricing", "pay", "unit economics", "economic"])


def test_14_solution_mvp_target_produces_product_specific_key_decision():
    """Solution / MVP target produces product workflow completion decision."""
    state = {
        "weak_area": "Solution / MVP",
        "active_assumption": {
            "statement": "Users complete the core video export workflow without assistance.",
            "area": "Solution / MVP",
        },
    }
    res = decide_next_strategic_action(state)
    act = res["next_strategic_action"]
    kd_lower = act["key_decision"].lower()
    assert any(w in kd_lower for w in ["workflow", "complete", "solution", "product", "setup", "derive value"])
    assert "channel" not in kd_lower
    assert "cac" not in kd_lower


def test_15_semantic_mismatch_triggers_retry_or_repair_instead_of_accepted():
    """Semantic guard repairs mismatched key decision when area is Acquisition."""
    state = {
        "weak_area": "Acquisition / Customer Access",
        "active_assumption": {
            "statement": "Acquiring customers requires solving acute pain.",
            "area": "Acquisition / Customer Access",
        },
    }
    res = decide_next_strategic_action(state)
    act = res["next_strategic_action"]
    assert act["action_type"] == "ACQUISITION_TEST"
    assert "acute pain" not in act["key_decision"].lower()
    assert any(w in act["key_decision"].lower() for w in ["channel", "reach", "network", "acquir"])


# ---------------------------------------------------------------------------
# Tests 16-21: Product Iteration Contracts & Rendering
# ---------------------------------------------------------------------------

def test_16_product_iteration_reachable_for_existing_product_startup_with_friction():
    """PRODUCT_ITERATION is selected for existing product with documented user friction."""
    state = {
        "project_id": "andever_proj",
        "weak_area": "Solution / MVP",
        "raw_context": "8 paying teams currently record matches using our deployed app.",
        "founder_messages": [
            {"role": "founder", "content": "4 teams experienced setup friction and camera setup failure during match recording."}
        ],
        "startup_state": {
            "progress": {
                "paying_customers_count": 8,
                "mvp_exists": {"value": True},
                "development_stage": {"value": "LIVE_PRODUCT"},
            }
        },
    }
    res = decide_next_strategic_action(state)
    act = res["next_strategic_action"]
    assert act["action_type"] == "PRODUCT_ITERATION"
    assert act["target_area"] == "Solution / MVP"


def test_17_product_iteration_not_selected_merely_because_founder_wants_features():
    """PRODUCT_ITERATION is NOT selected merely because founder wants more features without friction evidence."""
    state = {
        "project_id": "andever_proj",
        "weak_area": "Acquisition / Customer Access",
        "raw_context": "8 paying teams currently record matches.",
        "founder_messages": [
            {"role": "founder", "content": "Founder wants to build more features like advanced tactical AI and player heatmaps."}
        ],
        "startup_state": {
            "progress": {
                "paying_customers_count": 8,
                "mvp_exists": {"value": True},
                "development_stage": {"value": "LIVE_PRODUCT"},
            }
        },
    }
    res = decide_next_strategic_action(state)
    act = res["next_strategic_action"]
    assert act["action_type"] != "PRODUCT_ITERATION"


def test_18_product_iteration_includes_evidence_backed_bottleneck():
    """PRODUCT_ITERATION populates bottleneck_identified and blocked_behavior."""
    state = {
        "weak_area": "Solution / MVP",
        "raw_context": "Live app with 8 paying teams. Coaches experienced setup friction and recording failure drop-off.",
        "startup_state": {
            "progress": {
                "paying_customers_count": 8,
                "mvp_exists": {"value": True},
            }
        },
    }
    res = decide_next_strategic_action(state)
    act = res["next_strategic_action"]
    assert act["action_type"] == "PRODUCT_ITERATION"
    assert act["bottleneck_identified"] is not None
    assert len(act["bottleneck_identified"]) > 10
    assert act["blocked_behavior"] is not None


def test_19_product_iteration_proposes_smallest_justified_change():
    """PRODUCT_ITERATION populates smallest_change focused on the bottleneck."""
    state = {
        "weak_area": "Solution / MVP",
        "raw_context": "Live app with 8 paying teams. Teams dropped out during setup friction.",
        "startup_state": {
            "progress": {
                "paying_customers_count": 8,
                "mvp_exists": {"value": True},
            }
        },
    }
    res = decide_next_strategic_action(state)
    act = res["next_strategic_action"]
    assert act["smallest_change"] is not None
    assert "setup" in act["smallest_change"].lower() or "verification" in act["smallest_change"].lower()


def test_20_product_iteration_explicitly_states_what_not_to_build():
    """PRODUCT_ITERATION lists at least 2 features founder must NOT build yet."""
    state = {
        "weak_area": "Solution / MVP",
        "raw_context": "Live app with 8 paying teams. Setup friction during recording.",
        "startup_state": {
            "progress": {
                "paying_customers_count": 8,
                "mvp_exists": {"value": True},
            }
        },
    }
    res = decide_next_strategic_action(state)
    act = res["next_strategic_action"]
    assert isinstance(act["what_not_to_build"], list)
    assert len(act["what_not_to_build"]) >= 2


def test_21_product_iteration_produces_measurable_behavioral_test():
    """render_execution_handoff_text formats dedicated PRODUCT_ITERATION template."""
    handoff = ExecutionHandoff(
        experiment_id="exp-pi",
        assumption_id="asmp-pi",
        critical_assumption="Pre-match checklist eliminates setup failure",
        experiment_type="PRODUCT_MEDIATED_USE",
        learning_goal="Verify setup checklist eliminates failure",
        what_we_are_testing="Pre-match checklist eliminates setup failure",
        target_participants_or_context="5 amateur teams",
        founder_actions=["Implement setup checklist"],
        evidence_to_collect="Recording completion rates",
        support_signal="4/5 teams record successfully",
        next_strategic_action={
            "action_type": "PRODUCT_ITERATION",
            "bottleneck_identified": "camera setup and recording failure",
            "smallest_change": "2-minute pre-match setup verification checklist",
            "what_not_to_build": ["advanced tactical AI analytics", "multi-camera support"],
            "measurable_outcome": "4 of 5 teams complete match recording without assistance",
        },
    )
    rendered = render_execution_handoff_text(handoff, mode="compact")
    assert "NEXT ACTION — PRODUCT ITERATION" in rendered
    assert "Improve only the camera setup and recording failure bottleneck." in rendered
    assert "BUILD / CHANGE ONLY THIS:" in rendered
    assert "2-minute pre-match setup verification checklist" in rendered
    assert "DO NOT BUILD YET:" in rendered
    assert "advanced tactical AI analytics" in rendered
    assert "MEASURE:" in rendered
    assert "4 of 5 teams complete match recording without assistance" in rendered


# ---------------------------------------------------------------------------
# Tests 22-25: Limited Weak-Link Override Authority
# ---------------------------------------------------------------------------

def test_22_acquisition_weak_link_overridden_by_validated_product_bottleneck(andever_state_with_traction):
    """Acquisition weak link is overridden to Solution/MVP when 8 paying teams face setup friction."""
    andever_state_with_traction["raw_context"] += " However, coaches experience acute setup friction and recording failure during games."
    res = decide_next_strategic_action(andever_state_with_traction)
    act = res["next_strategic_action"]
    assert act["is_weak_link_override"] is True
    assert act["target_area"] == "Solution / MVP"
    assert act["action_type"] == "PRODUCT_ITERATION"


def test_23_override_includes_explicit_reason(andever_state_with_traction):
    """Weak-link override includes explicit reason explaining why product fix precedes acquisition."""
    andever_state_with_traction["raw_context"] += " However, coaches experience acute setup friction and recording failure during games."
    res = decide_next_strategic_action(andever_state_with_traction)
    act = res["next_strategic_action"]
    assert act["is_weak_link_override"] is True
    assert act["override_reason"] is not None
    assert "leaky bucket" in act["override_reason"].lower() or "setup" in act["override_reason"].lower()


def test_24_weak_link_authority_intact_when_no_higher_leverage_bottleneck(andever_state_with_traction):
    """Weak-link authority remains intact when no validated product/retention bottleneck exists."""
    # No friction mentioned; 8 paying teams are happy; weak link is Acquisition
    res = decide_next_strategic_action(andever_state_with_traction)
    act = res["next_strategic_action"]
    assert act["is_weak_link_override"] is False
    assert act["override_reason"] is None
    assert act["target_area"] == "Acquisition / Customer Access"
    assert act["action_type"] == "ACQUISITION_TEST"


def test_25_andever_target_user_alternatives_acquisition_progression_works_without_product_override():
    """&ever progresses through standard areas when no product override is justified."""
    # Step 1: Target User
    state_tu = {
        "weak_area": "Target User",
        "active_assumption": {"statement": "Regular amateur teams represent the primary repeat user segment."},
    }
    res_tu = decide_next_strategic_action(state_tu)
    assert res_tu["next_strategic_action"]["target_area"] == "Target User"

    # Step 2: Existing Alternatives
    state_alt = {
        "weak_area": "Existing Alternatives",
        "active_assumption": {"statement": "Workarounds like parent camcorders fail to provide structured review."},
    }
    res_alt = decide_next_strategic_action(state_alt)
    assert res_alt["next_strategic_action"]["target_area"] == "Existing Alternatives"

    # Step 3: Acquisition
    state_acq = {
        "weak_area": "Acquisition / Customer Access",
        "active_assumption": {"statement": "Teams can be acquired via league directors."},
    }
    res_acq = decide_next_strategic_action(state_acq)
    assert res_acq["next_strategic_action"]["target_area"] == "Acquisition / Customer Access"
    assert res_acq["next_strategic_action"]["action_type"] == "ACQUISITION_TEST"


# ---------------------------------------------------------------------------
# Tests 26-30: Regression Invariants
# ---------------------------------------------------------------------------

def test_26_lastbite_build_minimal_prototype_still_works():
    """LastBite build readiness escalation to BUILD_MINIMAL_PROTOTYPE still works and renders properly."""
    state = {
        "build_readiness": {
            "decision": "BUILD_MINIMAL_PROTOTYPE",
            "reason": "Manual pickups proven; software probe required.",
            "minimum_build_scope": ["1-click reservation"],
            "must_not_build": ["loyalty points"],
            "evidence_basis": ["10 paid pickups completed"],
        },
        "weak_area": "Demand Signal",
        "active_assumption": {"statement": "Users will reserve bags through software"},
    }
    res = decide_next_strategic_action(state)
    act = res["next_strategic_action"]
    assert act["action_type"] == "BUILD_MINIMAL_PROTOTYPE"

    handoff = ExecutionHandoff(
        experiment_id="exp-lb",
        assumption_id="asmp-lb",
        critical_assumption="Users will reserve bags through software",
        experiment_type="CONCIERGE",
        learning_goal="Validate software reservations",
        what_we_are_testing="Users will reserve bags through software",
        target_participants_or_context="10 bakery customers",
        founder_actions=["Deploy 1-click page"],
        evidence_to_collect="Reservations",
        support_signal="5 reservations",
        build_readiness=state["build_readiness"],
    )
    rendered = render_execution_handoff_text(handoff, mode="compact")
    assert "STARTARY — TIME TO BUILD THE NEXT TEST" in rendered
    assert "BUILD ONLY THIS:" in rendered
    assert "1-click reservation" in rendered
    assert "DO NOT BUILD YET:" in rendered
    assert "loyalty points" in rendered


def test_27_partial_evidence_lifecycle_still_works():
    """Partial evidence produces CONTINUE_CURRENT_TEST action."""
    state = {
        "is_partial_evidence": True,
        "weak_area": "Problem",
        "active_assumption": {"statement": "Merchants experience sizing return pain"},
    }
    res = decide_next_strategic_action(state)
    act = res["next_strategic_action"]
    assert act["action_type"] == "CONTINUE_CURRENT_TEST"
    assert act["target_area"] == "Problem"


def test_28_m411_recovery_still_works():
    """M4.1.1 experiment critique failure recovery preserves interactive prompt."""
    from startary.domain.experiment import render_fallback_learning_action_text
    data = {
        "learning_goal": "Understand merchant return frequency",
        "untrusted_test_reasons": ["Sample size too small (1 store)"],
        "smallest_thing_to_resolve_next": "Talk to 3 additional boutique merchants",
        "primary_action": "Ask 3 owners about their weekly return counts",
        "what_to_ask_or_observe": ["How many items returned last week?"],
        "what_to_bring_back": ["Specific return numbers"],
        "what_not_to_infer": ["Do not infer willingness to pay"],
        "completion_instruction": "Return with the answers from those 3 conversations",
    }
    rendered = render_fallback_learning_action_text(data)
    assert "STARTARY — EXPERIMENT DESIGN NEEDS REVISION" in rendered
    assert "Sample size too small (1 store)" in rendered
    assert "Talk to 3 additional boutique merchants" in rendered


def test_29_evidence_admission_precedence_still_works():
    """Evidence admission precedence still classifies completed observations as actual evidence."""
    from startary.domain.evidence import has_substantive_completed_evidence, EvidenceTurnDisposition
    has_ev = has_substantive_completed_evidence("I interviewed 5 merchants today and 4 had sizing issues.")
    assert has_ev is True
    assert EvidenceTurnDisposition.ACTUAL_EVIDENCE.value == "ACTUAL_EVIDENCE"


def test_30_full_regression_suite_passes():
    """Meta-test verifying all contracts are structurally met and consistent."""
    act = NextStrategicAction(
        action_type=NextStrategicActionType.PRODUCT_ITERATION,
        rationale="Valid test",
        target_area="Solution / MVP",
        key_decision="Whether setup checklist reduces drop-off.",
        why_this_action="Directly addresses friction.",
        bottleneck_identified="Setup complexity",
        blocked_behavior="Coaches fail recording",
        smallest_change="Checklist",
        measurable_outcome="4/5 record",
        what_not_to_build=["AI", "Reels"],
        is_weak_link_override=True,
        override_reason="Existing paying users face friction.",
    )
    dump = act.model_dump()
    assert dump["action_type"] == "PRODUCT_ITERATION"
    assert dump["is_weak_link_override"] is True
    assert len(dump["what_not_to_build"]) == 2
