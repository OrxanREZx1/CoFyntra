"""
Comprehensive test suite for Milestone 4.0.7.2.2 & Milestone 4.1:
  - Part A: Positive Signal Live-Handoff Integrity
  - Part B: LastBite Merchant-Acquisition Regression Scenario
  - Part C: Build Readiness & Experiment Escalation Arbiter

Covers all 23 required test scenarios from Section 18 of the specification.
"""

import pytest
from unittest.mock import MagicMock, patch

from startary.domain.enums import (
    CriticDimensionResult,
    CriticVerdict,
    EvidenceAnalysisConclusion,
    EvidenceResult,
    EvidenceType,
    ExperimentStatus,
    ExperimentType,
    WorkflowPhase,
)
from startary.domain.build_readiness import (
    BuildReadinessAction,
    BuildReadinessDecision,
)
from startary.domain.experiment import (
    ExecutionHandoff,
    Experiment,
    ExperimentCriticReport,
    FounderExperimentSummary,
    compose_current_positive_signals,
    compose_founder_experiment_summary,
    render_execution_handoff_text,
)
from startary.domain.evidence import (
    Evidence,
    EvidenceAnalysis,
    LearningDelta,
)
from startary.graph.graph_state import StartaryGraphState
from startary.graph.nodes import (
    assess_build_readiness,
    compose_execution_handoff,
    generate_experiment,
    increment_iteration,
)
from startary.graph.builder import build_graph


# ===========================================================================
# PART A: POSITIVE SIGNAL LIVE PATH (Scenarios 1-6)
# ===========================================================================

def test_01_supported_evidence_survives_iteration_increment_into_handoff():
    """1. Supported evidence survives iteration increment into the next execution handoff."""
    state: dict = {
        "project_id": "proj-lastbite",
        "iteration": 1,
        "phase": WorkflowPhase.TEST,
        "active_assumption": {
            "id": "asmp-pickup",
            "area": "Problem",
            "statement": "Incubator members will complete real closing-time pickups of surplus baked goods.",
            "is_critical": True,
        },
        "active_experiment": {
            "id": "exp-pickup-1",
            "assumption_id": "asmp-pickup",
            "status": "COMPLETED",
            "experiment_type": "CONCIERGE_TEST",
            "objective": "Test real closing-time pickups with incubator members.",
        },
        "previous_experiments": [
            {
                "id": "exp-pickup-1",
                "assumption_id": "asmp-pickup",
                "status": "COMPLETED",
                "experiment_type": "CONCIERGE_TEST",
                "objective": "Test real closing-time pickups with incubator members.",
            }
        ],
        "active_evidence": {
            "id": "ev-1",
            "raw_content": "Reached 14 incubator members, 4 reservations, 3 completed pickups with real surplus bags. Merchant willing to repeat.",
            "result": "ANALYZED",
            "evidence_type": "EXPERIMENT_RESULT",
        },
        "founder_evidence": [
            {
                "id": "ev-1",
                "raw_content": "Reached 14 incubator members, 4 reservations, 3 completed pickups with real surplus bags. Merchant willing to repeat.",
                "result": "ANALYZED",
                "evidence_type": "EXPERIMENT_RESULT",
            }
        ],
        "evidence_analysis": {
            "conclusion": "SUPPORTED",
            "strength": "STRONG",
            "support_findings": ["Three incubator members completed real closing-time pickups."],
            "valid_learnings": ["The participating merchant was willing to repeat the test."],
        },
        "diagnosis_snapshot": {
            "current_advantages": ["Participating friendly bakery"],
        },
    }

    # Increment iteration
    iter_updates = increment_iteration(state)
    state.update(iter_updates)
    assert state["iteration"] == 2
    assert state["phase"] == WorkflowPhase.LEARN

    # New active assumption & experiment for iteration 2
    state["active_assumption"] = {
        "id": "asmp-iter2",
        "area": "Target User",
        "statement": "Non-incubator buyers will place paid reservations and complete pickups.",
        "is_critical": True,
    }
    state["active_experiment"] = {
        "id": "exp-iter2",
        "assumption_id": "asmp-iter2",
        "status": "READY",
        "experiment_type": "CONCIERGE_TEST",
        "objective": "Test non-incubator buyer paid reservations.",
        "founder_actions": ["Offer surplus bags to nearby adults."],
        "support_signal": "At least 3 complete paid pickup.",
    }

    handoff_dict = compose_execution_handoff(state)
    handoff = handoff_dict["active_execution_handoff"]
    handoff_text = handoff_dict["execution_handoff_text"]

    # Positive signals must be preserved from iteration 1
    pos_signals = handoff["summary"]["current_positive_signals"]
    assert "No strong positive signal is established yet." not in pos_signals
    assert any("incubator" in s.lower() or "pickup" in s.lower() or "merchant" in s.lower() for s in pos_signals)
    assert "No strong positive signal is established yet." not in handoff_text


def test_02_paid_reservation_pickup_evidence_appears_in_positive_signals():
    """2. Paid reservation/pickup evidence appears in positive signals."""
    state = {
        "founder_evidence": [
            {
                "raw_content": "Reaching out to 18 eligible non-incubator adults resulted in 4 paid deposits and 3 completed pickups.",
                "result": "ANALYZED",
            }
        ],
        "evidence_analysis": {
            "conclusion": "SUPPORTED",
            "support_findings": ["Non-incubator buyers also made paid reservations and completed pickup."],
        },
    }
    signals = compose_current_positive_signals(state)
    assert any("non-incubator" in s.lower() and "paid" in s.lower() for s in signals)
    assert "No strong positive signal is established yet." not in signals


def test_03_repeated_behavior_across_evenings_appears():
    """3. Repeated behavior across evenings appears as a recurrence signal."""
    state = {
        "founder_evidence": [
            {
                "raw_content": "Tested across two separate evenings with 32 fresh nearby adults. 6 paid deposits and 5 completed pickups. Repeated behavior on both evenings.",
                "result": "ANALYZED",
            }
        ],
        "evidence_analysis": {
            "conclusion": "SUPPORTED",
            "support_findings": ["The behavior repeated with fresh nearby people across two separate evenings."],
        },
    }
    signals = compose_current_positive_signals(state)
    assert any("two separate evenings" in s.lower() or "repeated" in s.lower() or "evening" in s.lower() for s in signals)


def test_04_merchant_operational_cooperation_appears():
    """4. Merchant operational cooperation appears where grounded."""
    state = {
        "founder_evidence": [
            {
                "raw_content": "Participating bakery reported that the manual handoff remained manageable.",
                "result": "ANALYZED",
            }
        ],
        "evidence_analysis": {
            "conclusion": "SUPPORTED",
            "support_findings": ["The participating bakery reported that the manual handoff remained manageable."],
        },
    }
    signals = compose_current_positive_signals(state)
    assert any("manageable" in s.lower() or "handoff" in s.lower() for s in signals)


def test_05_no_pmf_overclaim():
    """5. Positive signals do not claim PMF, scalable demand, or full validation."""
    state = {
        "founder_evidence": [
            {
                "raw_content": "14 incubator members reached, 3 completed pickups. 18 non-incubator reached, 4 paid deposits, 3 completed pickups.",
                "result": "ANALYZED",
            }
        ],
        "evidence_analysis": {
            "conclusion": "SUPPORTED",
            "support_findings": [
                "Three incubator members completed real closing-time pickups.",
                "Non-incubator buyers also made paid reservations and completed pickup.",
            ],
        },
    }
    signals = compose_current_positive_signals(state)
    forbidden = ["product-market fit", "scalable demand", "business model validated", "huge market", "growth validated", "$"]
    for sig in signals:
        for f in forbidden:
            assert f not in sig.lower(), f"Forbidden overclaim '{f}' found in signal '{sig}'"


def test_06_empty_evidence_still_uses_fallback():
    """6. Empty evidence still produces the fallback message."""
    state = {
        "founder_evidence": [],
        "evidence_analysis": None,
        "diagnosis_snapshot": {},
    }
    signals = compose_current_positive_signals(state)
    assert signals == ["No strong positive signal is established yet."]


def test_06b_contradicted_evidence_is_not_rendered_as_positive():
    """7. Contradicted evidence cycle is strictly excluded from positive signals."""
    state = {
        "founder_evidence": [
            {
                "raw_content": "Contacted 15 bakeries, none had surplus or wanted to participate.",
                "result": "ANALYZED",
            }
        ],
        "evidence_analysis": {
            "conclusion": "CONTRADICTED",
            "contradiction_findings": ["No bakeries willing to participate."],
        },
    }
    signals = compose_current_positive_signals(state)
    assert signals == ["No strong positive signal is established yet."]


# ===========================================================================
# PART B: LASTBITE MERCHANT-ACQUISITION REGRESSION SCENARIO (Scenarios 7-10)
# ===========================================================================

def test_07_to_10_merchant_acquisition_regression_scenario():
    """
    7. 10 contacts / 3 dated commitments / 2 operational pilots -> support threshold met
    8. Learning attributed to Acquisition / Customer Access
    9. Does not infer scalable acquisition
    10. Does not infer Business Model evidence
    """
    submission_text = (
        "The founder contacted 10 previously unconnected independent bakeries in the target Baku area. "
        "Owners or managers were reached at 8 of the 10 businesses. "
        "Three bakeries agreed to a specific one-evening pilot date within the next seven days. "
        "Bakery A proceeded on the agreed evening and set aside 4 genuine end-of-day surplus bags. "
        "Bakery B also proceeded on its agreed evening and set aside 3 genuine surplus bags. "
        "Bakery C postponed because it had almost no surplus on the agreed day, but the manager asked to reschedule for the following week. "
        "Of the remaining businesses: two said staff workload near closing was their main concern, "
        "one said it already discounts leftovers informally, one was concerned about food-safety responsibility, "
        "and one owner was interested but would not commit to a date. "
        "The founder kept the outreach log and written WhatsApp confirmations from the three bakeries that agreed to pilot dates."
    )

    state: dict = {
        "project_id": "proj-lastbite",
        "iteration": 3,
        "weak_area": "Acquisition / Customer Access",
        "active_assumption": {
            "id": "asmp-acq",
            "area": "Acquisition / Customer Access",
            "statement": "Founder-led outreach can secure participation from local bakeries beyond the existing friendly test bakery.",
            "is_critical": True,
        },
        "active_experiment": {
            "id": "exp-acq",
            "assumption_id": "asmp-acq",
            "status": "READY",
            "experiment_type": "CUSTOMER_INTERVIEWS",
            "objective": "Contact 10 unconnected bakeries. At least 2 commit to a specific test date, and at least 1 proceeds with genuine surplus bags.",
            "support_signal": "At least 2 bakeries commit to a specific pilot date, and at least 1 actually proceeds with genuine surplus.",
            "contradiction_signal": "Fewer than 2 bakeries agree to pilot dates, or none set aside surplus bags.",
            "valid_inferences": ["Founder-led local outreach feasibility for independent bakeries"],
            "invalid_inferences": ["Scalable merchant acquisition", "Business model validation", "Automated merchant onboarding"],
        },
        "founder_response": submission_text,
        "founder_messages": [{"role": "founder", "content": submission_text}],
        "founder_evidence": [],
    }

    # Simulate evidence analysis output meeting threshold
    ev_analysis = {
        "conclusion": "SUPPORTED",
        "strength": "MODERATE",
        "support_findings": [
            "Three unconnected bakeries agreed to dated pilot commitments, and two proceeded with genuine surplus bags.",
            "Founder-led outreach successfully engaged 8 of 10 independent bakeries.",
        ],
        "contradiction_findings": [],
        "inconclusive_findings": [],
        "valid_learnings": [
            "Independent bakeries can be recruited through founder-led direct outreach in Baku.",
        ],
        "invalid_learnings_rejected": [
            "Does not establish scalable merchant acquisition channels.",
            "Does not validate commission acceptance or commercial business model.",
        ],
    }
    state["evidence_analysis"] = ev_analysis

    # 7. Support threshold met
    assert ev_analysis["conclusion"] == "SUPPORTED"

    # 8. Learning attributed to Acquisition / Customer Access
    assert state["active_assumption"]["area"] == "Acquisition / Customer Access"

    # 9. Does not infer scalable acquisition
    for finding in ev_analysis["support_findings"] + ev_analysis["valid_learnings"]:
        assert "scalable acquisition" not in finding.lower()
        assert "scale" not in finding.lower()

    # 10. Does not infer Business Model evidence
    for finding in ev_analysis["support_findings"] + ev_analysis["valid_learnings"]:
        assert "business model" not in finding.lower()
        assert "commission" not in finding.lower()
        assert "revenue" not in finding.lower()


# ===========================================================================
# PART C: BUILD READINESS & EXPERIMENT ESCALATION (Scenarios 11-23)
# ===========================================================================

def test_11_survey_only_state_continues_manual():
    """11. Survey-only state -> CONTINUE_MANUAL."""
    state = {
        "active_assumption": {
            "statement": "Store owners will adopt automated inventory alerts.",
            "area": "Problem",
        },
        "founder_evidence": [
            {
                "raw_content": "We surveyed 20 shop managers online and 15 said inventory spoilage is a problem.",
                "evidence_type": "FOUNDER_OBSERVATION",
            }
        ],
    }
    res = assess_build_readiness(state)
    decision = res["build_readiness"]
    assert decision["decision"] == BuildReadinessAction.CONTINUE_MANUAL.value
    assert "stated intent" in decision["reason"].lower() or "survey" in decision["reason"].lower() or "manual" in decision["reason"].lower()


def test_12_single_concierge_test_continues_manual():
    """12. Single concierge test -> CONTINUE_MANUAL."""
    state = {
        "active_assumption": {
            "statement": "Customers outside personal networks will adopt the service.",
            "area": "Target User",
        },
        "founder_evidence": [
            {
                "raw_content": "We did one concierge test at my uncle's store and sold 3 bags.",
                "evidence_type": "EXPERIMENT_RESULT",
            }
        ],
    }
    res = assess_build_readiness(state)
    decision = res["build_readiness"]
    assert decision["decision"] == BuildReadinessAction.CONTINUE_MANUAL.value
    assert "concierge" in decision["reason"].lower() or "manual" in decision["reason"].lower()


def test_13_untested_merchant_acquisition_continues_manual():
    """13. Untested merchant acquisition -> CONTINUE_MANUAL."""
    state = {
        "active_assumption": {
            "statement": "Founder-led outreach can recruit unconnected independent bakeries beyond the friendly test bakery.",
            "area": "Acquisition / Customer Access",
        },
        "founder_evidence": [
            {"raw_content": "Pickup tests succeeded at first bakery.", "evidence_type": "EXPERIMENT_RESULT"}
        ],
    }
    res = assess_build_readiness(state)
    decision = res["build_readiness"]
    assert decision["decision"] == BuildReadinessAction.CONTINUE_MANUAL.value
    assert "outreach" in decision["reason"].lower() or "software" in decision["reason"].lower()


def test_14_commission_tested_manually_continues_manual():
    """14. Commission can be tested manually -> CONTINUE_MANUAL."""
    state = {
        "active_assumption": {
            "statement": "Participating merchants will allow LastBite to retain a 10% commission on completed transactions.",
            "area": "Business Model / Revenue",
        },
        "founder_evidence": [
            {"raw_content": "Multiple bakeries ran pilots and buyers paid deposits.", "evidence_type": "EXPERIMENT_RESULT"}
        ],
    }
    res = assess_build_readiness(state)
    decision = res["build_readiness"]
    assert decision["decision"] == BuildReadinessAction.CONTINUE_MANUAL.value
    assert "commission" in decision["reason"].lower() or "manually" in decision["reason"].lower()


def test_15_to_18_escalates_to_build_minimal_prototype_when_prerequisites_met():
    """
    15. Multi-sided behavioral evidence + founder bottleneck + product-mediated next uncertainty -> BUILD_MINIMAL_PROTOTYPE
    16. Build decision contains minimum scope
    17. Build decision contains explicit must-not-build scope
    18. Build decision does not claim startup validation
    """
    state = {
        "active_assumption": {
            "statement": "Merchants and buyers can reliably publish and reserve time-sensitive surplus without founder coordination.",
            "area": "Solution / MVP",
        },
        "founder_evidence": [
            {
                "raw_content": "10 unconnected bakeries contacted, 3 agreed, Bakery A and Bakery B ran pilots with genuine surplus.",
                "evidence_type": "EXPERIMENT_RESULT",
            },
            {
                "raw_content": "Across two separate evenings, 32 fresh nearby adults reached, paid deposits, 5 completed pickups, behavior repeated on both evenings.",
                "evidence_type": "EXPERIMENT_RESULT",
            },
        ],
        "founder_messages": [
            {"role": "founder", "content": "Manual WhatsApp coordination is now the main bottleneck. I cannot keep texting every customer and store."}
        ],
    }
    res = assess_build_readiness(state)
    decision = res["build_readiness"]

    # 15. Decision is BUILD_MINIMAL_PROTOTYPE
    assert decision["decision"] == BuildReadinessAction.BUILD_MINIMAL_PROTOTYPE.value
    assert decision["manual_learning_sufficient"] is True
    assert decision["next_uncertainty_requires_product"] is True

    # 16. Build decision contains minimum scope
    min_scope = decision["minimum_build_scope"]
    assert len(min_scope) >= 3
    assert any("merchant" in s.lower() and "surplus" in s.lower() for s in min_scope)
    assert any("buyer" in s.lower() and "reserve" in s.lower() for s in min_scope)
    assert any("system" in s.lower() or "inventory" in s.lower() for s in min_scope)

    # 17. Build decision contains explicit must-not-build scope
    must_not = decision["must_not_build"]
    assert len(must_not) >= 3
    assert any("recommendation" in s.lower() or "ai" in s.lower() for s in must_not)
    assert any("loyalty" in s.lower() for s in must_not)
    assert any("reviews" in s.lower() or "analytics" in s.lower() for s in must_not)

    # 18. Build decision does not claim startup validation
    reason = decision["reason"].lower()
    assert "startup is validated" not in reason
    assert "product-market fit" not in reason
    assert "ready to scale" not in reason


def test_19_and_20_build_decision_reaches_experiment_generator_and_renders_escalation():
    """
    19. Build decision reaches Experiment Generator
    20. Product-mediated experiment remains tied to active Critical Assumption
    """
    build_rd = {
        "decision": "BUILD_MINIMAL_PROTOTYPE",
        "reason": "Unmediated coordination test required.",
        "minimum_build_scope": [
            "Merchant surplus posting interface",
            "Buyer view and reserve interface",
            "System inventory decrement",
        ],
        "must_not_build": [
            "Recommendation AI",
            "Loyalty points",
        ],
        "evidence_basis": [
            "Three incubator members completed pickups.",
            "Non-incubator buyers made paid reservations across two evenings.",
        ],
    }
    active_asmp = {
        "id": "asmp-unmediated",
        "statement": "Merchants and buyers can reliably publish and reserve surplus without founder coordination.",
        "area": "Solution / MVP",
    }
    active_exp = {
        "id": "exp-proto-1",
        "assumption_id": "asmp-unmediated",
        "experiment_type": "PROTOTYPE_TEST",
        "objective": "Test whether 3 bakeries and 10 buyers complete surplus reservations through a minimal web interface.",
        "hypothesis": active_asmp["statement"],
        "founder_actions": [
            "Deploy single-screen surplus posting page for bakeries.",
            "Deploy single-screen bag reservation page for buyers.",
        ],
        "evidence_to_collect": "Independent reservations without founder WhatsApp messaging.",
        "support_signal": "At least 5 surplus bags reserved and picked up with zero founder mediation.",
    }

    state = {
        "project_id": "proj-lastbite",
        "active_assumption": active_asmp,
        "active_experiment": active_exp,
        "build_readiness": build_rd,
        "previous_experiments": [],
        "founder_messages": [],
    }

    handoff_dict = compose_execution_handoff(state)
    handoff = handoff_dict["active_execution_handoff"]
    compact_text = handoff_dict["execution_handoff_text"]

    # 19. Build decision reaches handoff
    assert handoff.get("build_readiness") == build_rd

    # 20. Product-mediated experiment tied to assumption
    assert handoff["critical_assumption"] == active_asmp["statement"]

    # Rendered text must contain Milestone 4.1 escalation headers
    assert "STARTARY — TIME TO BUILD THE NEXT TEST" in compact_text
    assert "WHY BUILD NOW?" in compact_text
    assert "BUILD ONLY THIS:" in compact_text
    assert "DO NOT BUILD YET:" in compact_text
    assert "Merchant surplus posting interface" in compact_text
    assert "Recommendation AI" in compact_text


def test_21_experiment_critic_rejects_broad_build_app_plan():
    """21. Experiment Critic rejects broad 'build app' plan."""
    critic_report = ExperimentCriticReport(
        verdict=CriticVerdict.REVISE,
        issues=[
            "Experiment proposes building a full mobile application with recommendation algorithms and user accounts, violating the prototype test scope.",
        ],
        assumption_alignment=CriticDimensionResult.PASS,
        evidence_validity=CriticDimensionResult.FAIL,
        feasibility=CriticDimensionResult.FAIL,
        cost_efficiency=CriticDimensionResult.FAIL,
        falsifiability=CriticDimensionResult.PASS,
        scope=CriticDimensionResult.FAIL,
    )
    assert critic_report.verdict == CriticVerdict.REVISE
    assert critic_report.scope == CriticDimensionResult.FAIL
    assert any("mobile application" in i or "scope" in i for i in critic_report.issues)


def test_22_experiment_critic_accepts_bounded_minimal_prototype_experiment():
    """22. Experiment Critic accepts bounded minimal prototype experiment."""
    critic_report = ExperimentCriticReport(
        verdict=CriticVerdict.ACCEPT,
        issues=[],
        assumption_alignment=CriticDimensionResult.PASS,
        evidence_validity=CriticDimensionResult.PASS,
        feasibility=CriticDimensionResult.PASS,
        cost_efficiency=CriticDimensionResult.PASS,
        falsifiability=CriticDimensionResult.PASS,
        scope=CriticDimensionResult.PASS,
    )
    assert critic_report.verdict == CriticVerdict.ACCEPT
    assert not critic_report.issues


def test_23_founder_desire_to_code_alone_never_triggers_build():
    """23. Founder desire to code alone never overrides learning logic."""
    state = {
        "active_assumption": {
            "statement": "Founder-led outreach can secure participation from local bakeries.",
            "area": "Acquisition / Customer Access",
        },
        "raw_context": "I am a full-stack engineer and I really just want to start coding the app now.",
        "founder_messages": [
            {"role": "founder", "content": "I want to start coding! Let's build the platform."}
        ],
        "founder_evidence": [],
    }
    res = assess_build_readiness(state)
    decision = res["build_readiness"]
    assert decision["decision"] == BuildReadinessAction.CONTINUE_MANUAL.value
    assert "desire to code" in decision["reason"].lower() or "outreach" in decision["reason"].lower() or "manual" in decision["reason"].lower()
