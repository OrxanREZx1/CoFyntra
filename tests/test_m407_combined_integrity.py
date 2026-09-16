"""
Tests for Combined Integrity Patch:
- Milestone 4.0.7.1.4: Critical Assumption Atomicity & Single Test Scope
- Milestone 4.0.7.2: LearningDelta Cycle Integrity
- Positive Signal Regression: Domain-Agnostic Behavioral Signal Extraction
"""

import pytest
from unittest.mock import MagicMock

from startary.domain.enums import WorkflowPhase
from startary.domain.experiment import (
    Experiment,
    ExperimentStatus,
    extract_positive_signals,
)
from startary.domain.evidence import (
    Evidence,
    EvidenceType,
    EvidenceProvenance,
    EvidenceResult,
    get_most_recent_completed_evidence_cycle,
)
from startary.domain.taxonomy import StartupArea
from startary.graph.graph_state import StartaryGraphState
from startary.graph.nodes import (
    validate_critical_assumption_atomicity,
    validate_critical_assumption_area_purity,
    validate_critical_assumption_grounding,
    build_grounded_critical_assumption_fallback,
    generate_critical_assumption,
    validate_evidence_analysis,
    increment_iteration,
    can_compose_learning_delta,
    compose_learning_delta,
    answer_post_evidence_question,
)
from startary.llm.schemas import CriticalAssumptionLLMOutput


# ===========================================================================
# PART A: CRITICAL ASSUMPTION ATOMICITY (Tests 1-8)
# ===========================================================================

class TestCriticalAssumptionAtomicity:
    def test_1_business_model_multi_condition_bundling_rejected(self):
        """1. Business Model assumption combining commission acceptance + customer price tolerance + merchant economics + contribution margin is rejected."""
        ca = CriticalAssumptionLLMOutput(
            area=StartupArea.BUSINESS_MODEL,
            statement=(
                "Merchants participating in LastBite will agree to a commission-based transaction model "
                "that allows LastBite to retain revenue while maintaining acceptable customer prices, "
                "merchant economics, and positive contribution margin after transaction and operating costs."
            ),
            rationale="Broad commercial hypothesis.",
            what_falsifies_it="Merchants reject commission or contribution margin is negative.",
            grounding_points=["Founder wants to charge commission."],
            unresolved_uncertainty_addressed="Whether unit economics work.",
            introduced_test_parameters=[],
        )
        is_valid, reason = validate_critical_assumption_atomicity(ca, StartupArea.BUSINESS_MODEL)
        assert is_valid is False
        assert "combines" in reason.lower() or "multiple" in reason.lower()
        assert "commission" in reason.lower()

    def test_2_business_model_single_commission_acceptance_passes(self):
        """2. Business Model assumption only testing merchant commission acceptance passes."""
        ca = CriticalAssumptionLLMOutput(
            area=StartupArea.BUSINESS_MODEL,
            statement="Independent bakeries will accept a 10% commission on completed surplus-food transactions.",
            rationale="Testing commission take-rate.",
            what_falsifies_it="Bakeries refuse to let LastBite retain a commission.",
            grounding_points=["One bakery owner interested."],
            unresolved_uncertainty_addressed="Whether bakeries accept commission deductions.",
            introduced_test_parameters=[],
        )
        is_valid, reason = validate_critical_assumption_atomicity(ca, StartupArea.BUSINESS_MODEL)
        assert is_valid is True
        assert reason == ""

    def test_3_business_model_single_contribution_margin_passes(self):
        """3. Business Model assumption only testing positive contribution margin passes if that is the atomic uncertainty."""
        ca = CriticalAssumptionLLMOutput(
            area=StartupArea.BUSINESS_MODEL,
            statement="LastBite can achieve positive contribution margin after transaction and support costs.",
            rationale="Testing unit margins.",
            what_falsifies_it="Operating costs exceed gross revenue per transaction.",
            grounding_points=["Low payment processing overhead."],
            unresolved_uncertainty_addressed="Whether transaction contribution margin is positive.",
            introduced_test_parameters=[],
        )
        is_valid, reason = validate_critical_assumption_atomicity(ca, StartupArea.BUSINESS_MODEL)
        assert is_valid is True
        assert reason == ""

    def test_4_target_user_single_segment_role_hypothesis_passes(self):
        """4. Target User assumption containing one segment/role hypothesis passes."""
        ca = CriticalAssumptionLLMOutput(
            area=StartupArea.TARGET_USER,
            statement="Nearby students form a repeatable initial buyer segment for short-notice bakery surplus.",
            rationale="Validating student buyer density.",
            what_falsifies_it="Students do not engage with short-notice surplus notifications.",
            grounding_points=["University campus is within 2 blocks."],
            unresolved_uncertainty_addressed="Whether students represent a repeatable buyer segment.",
            introduced_test_parameters=[],
        )
        is_valid, reason = validate_critical_assumption_atomicity(ca, StartupArea.TARGET_USER)
        assert is_valid is True
        assert reason == ""

    def test_5_target_user_bundling_segment_adoption_payment_rejected(self):
        """5. Target User assumption combining segment definition + adoption + payment is rejected."""
        ca = CriticalAssumptionLLMOutput(
            area=StartupArea.TARGET_USER,
            statement="Nearby students form a repeatable segment that will adopt the app and pay a delivery fee.",
            rationale="Broad segment adoption hypothesis.",
            what_falsifies_it="Students do not adopt or pay fees.",
            grounding_points=["Local student population."],
            unresolved_uncertainty_addressed="Segment viability.",
            introduced_test_parameters=[],
        )
        is_atom_valid, atom_reason = validate_critical_assumption_atomicity(ca, StartupArea.TARGET_USER)
        is_purity_valid, purity_reason = validate_critical_assumption_area_purity(ca, StartupArea.TARGET_USER)
        assert (is_atom_valid is False) or (is_purity_valid is False)

    def test_6_atomicity_retry_receives_exact_rejection_reason(self, monkeypatch):
        """6. Atomicity retry receives the exact rejection reason on attempt 1."""
        non_atomic_output = CriticalAssumptionLLMOutput(
            area=StartupArea.BUSINESS_MODEL,
            statement=(
                "Independent bakeries will accept a commission, customers will tolerate the resulting price, "
                "and LastBite will remain profitable."
            ),
            rationale="Testing everything.",
            what_falsifies_it="Anything fails.",
            grounding_points=["Bakery context."],
            unresolved_uncertainty_addressed="Commercial viability.",
            introduced_test_parameters=[],
        )
        atomic_output = CriticalAssumptionLLMOutput(
            area=StartupArea.BUSINESS_MODEL,
            statement="The participating bakery will accept LastBite retaining a commission on completed surplus-food transactions.",
            rationale="Testing commission acceptance.",
            what_falsifies_it="Bakery refuses commission.",
            grounding_points=["Bakery context."],
            unresolved_uncertainty_addressed="Commission acceptance.",
            introduced_test_parameters=[],
        )

        call_count = 0
        captured_messages = []

        class MockStructuredLLM:
            def invoke(self, messages):
                nonlocal call_count, captured_messages
                call_count += 1
                captured_messages = list(messages)
                if call_count == 1:
                    return non_atomic_output
                return atomic_output

        class MockLLM:
            def with_structured_output(self, schema):
                return MockStructuredLLM()

        import startary.graph.nodes as nodes
        monkeypatch.setattr(nodes, "get_llm_client", lambda settings: MockLLM())

        state: StartaryGraphState = {
            "project_id": "proj-atom-retry",
            "weak_area": "Business Model / Revenue",
            "raw_context": "LastBite helps bakeries sell surplus food at discounts to nearby customers.",
            "reality_check": {
                "conclusion": "UNRESOLVED",
                "summary": "Business model has not been tested with real commissions.",
                "unresolved_uncertainty": "Whether bakeries will accept a commission on surplus sales.",
                "what_evidence_supports": ["Bakery owner willing to test surplus sales."],
                "what_evidence_does_not_support": [],
            },
            "startup_state": {"areas": {}},
        }

        res = generate_critical_assumption(state)
        assert call_count == 2
        assert res["critical_assumption_output"]["statement"] == atomic_output.statement
        retry_prompt_text = captured_messages[-1]["content"]
        assert "GROUNDING VALIDATION FAILED" in retry_prompt_text
        assert "ATOMIC" in retry_prompt_text

    def test_7_deterministic_fallbacks_pass_atomicity(self):
        """7. Deterministic fallbacks must pass atomicity."""
        state: StartaryGraphState = {
            "project_id": "proj-fb-atom",
            "weak_area": "Business Model / Revenue",
            "raw_context": "LastBite connects bakeries with local buyers for discounted surplus pastries.",
            "reality_check": {
                "conclusion": "UNRESOLVED",
                "summary": "Commission viability is unverified.",
                "unresolved_uncertainty": "Whether the participating bakery will accept a commission deduction.",
                "what_evidence_supports": ["Bakery owner interested in testing."],
                "what_evidence_does_not_support": [],
            },
        }
        fb = build_grounded_critical_assumption_fallback(
            "Business Model / Revenue",
            state["reality_check"],
            state["raw_context"],
            state,
        )
        assert fb is not None
        is_valid, reason = validate_critical_assumption_atomicity(fb, StartupArea.BUSINESS_MODEL)
        assert is_valid is True, f"Fallback failed atomicity: {reason}"

    def test_8_existing_valid_assumptions_remain_valid(self):
        """8. Existing valid assumptions from previous milestones remain valid."""
        fridgeguard_asmp = CriticalAssumptionLLMOutput(
            area=StartupArea.TARGET_USER,
            statement="There is a repeatable restaurant segment beyond personal connections in which an identifiable role is responsible for refrigeration monitoring and food safety compliance.",
            rationale="Role ownership.",
            what_falsifies_it="No restaurant role owns refrigeration.",
            grounding_points=["Commercial kitchens."],
            unresolved_uncertainty_addressed="Role responsibility.",
            introduced_test_parameters=[],
        )
        is_atom_valid, _ = validate_critical_assumption_atomicity(fridgeguard_asmp, StartupArea.TARGET_USER)
        assert is_atom_valid is True

        problem_asmp = CriticalAssumptionLLMOutput(
            area=StartupArea.PROBLEM,
            statement="Restaurants currently rely primarily on manual refrigerator monitoring that leaves overnight failures undetected.",
            rationale="Problem/alternative relationship.",
            what_falsifies_it="Overnight failures are detected by other means.",
            grounding_points=["Manual logs."],
            unresolved_uncertainty_addressed="Overnight detection.",
            introduced_test_parameters=[],
        )
        is_atom_valid, _ = validate_critical_assumption_atomicity(problem_asmp, StartupArea.PROBLEM)
        assert is_atom_valid is True


# ===========================================================================
# PART B: LEARNINGDELTA CYCLE INTEGRITY (Tests 9-17)
# ===========================================================================

class TestLearningDeltaCycleIntegrity:
    def test_9_valid_evidence_cycle_iter1_to_iter2_generates_learning_delta(self):
        """9. Valid evidence cycle iteration 1 -> increment to 2 -> LearningDelta is generated."""
        state: StartaryGraphState = {
            "project_id": "proj-delta-1",
            "iteration": 1,
            "weak_area": "Target User",
            "active_assumption": {
                "id": "asmp-001",
                "statement": "Nearby students and workers form a repeatable buyer segment for short-notice surplus.",
                "area": "Target User",
                "status": "UNTESTED",
            },
            "active_experiment": {
                "id": "exp-001",
                "assumption_id": "asmp-001",
                "status": "READY",
            },
            "previous_experiments": [
                {"id": "exp-001", "assumption_id": "asmp-001", "status": "READY"}
            ],
            "evidence_analysis": {
                "conclusion": "SUPPORTED",
                "evidence_strength": "MODERATE",
                "support_findings": ["6 unique paid pickups completed by nearby buyers."],
                "valid_learnings": ["Nearby students/workers respond to surplus notices."],
            },
            "founder_response": "We ran the test. 6 customers came and paid for surplus pastries.",
            "startup_state": {"areas": {}},
        }

        val_res = validate_evidence_analysis(state)
        state.update(val_res)

        iter_res = increment_iteration(state)
        state.update(iter_res)

        assert state["iteration"] == 2
        assert state["active_assumption"] is None
        assert state["active_experiment"] is None
        assert len(state["previous_assumptions"]) == 1

        state["weak_area"] = "Business Model / Revenue"

        assert can_compose_learning_delta(state) is True
        delta_res = compose_learning_delta(state)
        delta = delta_res.get("learning_delta")
        assert delta is not None
        assert delta["previous_weak_link"] == "Target User"
        assert delta["new_weak_link"] == "Business Model / Revenue"
        assert delta["weak_link_changed"] is True
        assert "Target User" in delta["founder_facing_summary"]
        assert "Business Model / Revenue" in delta["founder_facing_summary"]

    def test_10_valid_evidence_cycle_iter2_to_iter3_generates_learning_delta(self):
        """10. Valid evidence cycle iteration 2 -> increment to 3 -> LearningDelta is generated."""
        state: StartaryGraphState = {
            "project_id": "proj-delta-2",
            "iteration": 2,
            "previous_weak_area": "Target User",
            "weak_area": "Business Model / Revenue",
            "previous_assumptions": [
                {"id": "asmp-001", "area": "Target User", "statement": "Prior user asmp", "status": "SUPPORTED"}
            ],
            "previous_experiments": [
                {"id": "exp-001", "status": "COMPLETED"},
                {"id": "exp-002", "assumption_id": "asmp-002", "status": "READY"},
            ],
            "active_assumption": {
                "id": "asmp-002",
                "statement": "The participating bakery will accept LastBite retaining a commission on surplus sales.",
                "area": "Business Model / Revenue",
                "status": "UNTESTED",
            },
            "active_experiment": {
                "id": "exp-002",
                "assumption_id": "asmp-002",
                "status": "READY",
            },
            "evidence_analysis": {
                "conclusion": "SUPPORTED",
                "evidence_strength": "MODERATE",
                "support_findings": ["Bakery agreed to 15% commission deduction on test batch."],
                "valid_learnings": ["Commission retention is operationally acceptable to this bakery."],
            },
            "founder_response": "The bakery agreed to the 15% commission split.",
            "startup_state": {"areas": {}},
        }

        val_res = validate_evidence_analysis(state)
        state.update(val_res)
        iter_res = increment_iteration(state)
        state.update(iter_res)

        assert state["iteration"] == 3
        state["weak_area"] = "Acquisition / Customer Access"

        assert can_compose_learning_delta(state) is True
        delta_res = compose_learning_delta(state)
        delta = delta_res.get("learning_delta")
        assert delta is not None
        assert delta["previous_weak_link"] == "Business Model / Revenue"
        assert delta["new_weak_link"] == "Acquisition / Customer Access"

    def test_11_learning_delta_references_previous_completed_cycle_not_new_iteration(self):
        """11. LearningDelta references previous completed experiment/evidence rather than requiring evidence in the new iteration."""
        state: StartaryGraphState = {
            "project_id": "proj-delta-lookup",
            "iteration": 3,
            "previous_weak_area": "Problem",
            "weak_area": "Target User",
            "previous_assumptions": [
                {"id": "asmp-iter2", "area": "Problem", "statement": "Problem hypothesis", "status": "SUPPORTED"}
            ],
            "previous_experiments": [
                {"id": "exp-iter2", "assumption_id": "asmp-iter2", "status": "COMPLETED"}
            ],
            "founder_evidence": [
                {
                    "id": "ev-iter2",
                    "iteration": 2,
                    "experiment_id": "exp-iter2",
                    "assumption_id": "asmp-iter2",
                    "evidence_type": "EXPERIMENT_RESULT",
                    "result": "ANALYZED",
                    "structured_observations": ["Observed pain."],
                }
            ],
            "evidence_analysis": {
                "conclusion": "SUPPORTED",
                "evidence_strength": "STRONG",
                "support_findings": ["Strong problem evidence."],
            },
            "startup_state": {"areas": {}},
        }
        cycle = get_most_recent_completed_evidence_cycle(state)
        assert cycle is not None
        tested_asmp, comp_exp, adm_ev, ana = cycle
        assert tested_asmp["id"] == "asmp-iter2"
        assert comp_exp["id"] == "exp-iter2"
        assert adm_ev["id"] == "ev-iter2"
        assert ana["conclusion"] == "SUPPORTED"

    def test_12_tested_area_vs_new_weak_link_distinction(self):
        """12. Tested area Target User + new weak link Business Model does NOT say Business Model evidence was updated."""
        state: StartaryGraphState = {
            "project_id": "proj-delta-distinct",
            "iteration": 2,
            "previous_weak_area": "Target User",
            "weak_area": "Business Model / Revenue",
            "previous_assumptions": [
                {"id": "asmp-tu", "area": "Target User", "statement": "Target user hypothesis", "status": "SUPPORTED"}
            ],
            "previous_experiments": [
                {"id": "exp-tu", "assumption_id": "asmp-tu", "status": "COMPLETED"}
            ],
            "founder_evidence": [
                {
                    "id": "ev-tu",
                    "iteration": 1,
                    "experiment_id": "exp-tu",
                    "assumption_id": "asmp-tu",
                    "evidence_type": "EXPERIMENT_RESULT",
                    "result": "ANALYZED",
                    "structured_observations": ["6 paid pickups."],
                }
            ],
            "evidence_analysis": {
                "conclusion": "SUPPORTED",
                "evidence_strength": "MODERATE",
                "support_findings": ["6 paid pickups from nearby buyers."],
                "valid_learnings": ["Target user confirmed."],
            },
            "pending_post_evidence_question": "What did we learn and what next?",
            "startup_state": {"areas": {}},
        }
        delta_res = compose_learning_delta(state)
        state.update(delta_res)

        ans_res = answer_post_evidence_question(state)
        ans_text = ans_res["post_evidence_answer"]

        assert "Evidence base for Business Model" not in ans_text
        assert "Evidence base for Target User was updated" in ans_text or "What you learned" in ans_text
        assert "Focus on Business Model" in ans_text

    def test_13_supported_conclusion_renders_stronger_learning(self):
        """13. SUPPORTED conclusion renders stronger learning appropriately."""
        state: StartaryGraphState = {
            "project_id": "proj-supp",
            "iteration": 2,
            "previous_weak_area": "Problem",
            "weak_area": "Target User",
            "previous_assumptions": [
                {"id": "asmp-1", "area": "Problem", "statement": "Cafes face scheduling chaos.", "status": "SUPPORTED"}
            ],
            "previous_experiments": [{"id": "exp-1", "status": "COMPLETED"}],
            "founder_evidence": [{"id": "ev-1", "experiment_id": "exp-1", "result": "ANALYZED", "evidence_type": "EXPERIMENT_RESULT"}],
            "evidence_analysis": {
                "conclusion": "SUPPORTED",
                "evidence_strength": "STRONG",
                "support_findings": ["7 of 8 cafes confirmed recurring chaos."],
            },
            "startup_state": {"areas": {}},
        }
        res = compose_learning_delta(state)
        summary = res["learning_delta"]["founder_facing_summary"]
        assert "strong support for the tested Problem assumption" in summary

    def test_14_partially_supported_remains_partial_and_contextual(self):
        """14. PARTIALLY_SUPPORTED remains partial/contextual without claiming full validation."""
        state: StartaryGraphState = {
            "project_id": "proj-partial",
            "iteration": 2,
            "previous_weak_area": "Target User",
            "weak_area": "Business Model / Revenue",
            "previous_assumptions": [
                {"id": "asmp-1", "area": "Target User", "statement": "Nearby students form a buyer segment.", "status": "PARTIALLY_SUPPORTED"}
            ],
            "previous_experiments": [{"id": "exp-1", "status": "COMPLETED"}],
            "founder_evidence": [{"id": "ev-1", "experiment_id": "exp-1", "result": "ANALYZED", "evidence_type": "EXPERIMENT_RESULT"}],
            "evidence_analysis": {
                "conclusion": "PARTIALLY_SUPPORTED",
                "evidence_strength": "MODERATE",
                "support_findings": ["5 students bought surplus, but 2 were friends."],
                "inconclusive_findings": ["Broader non-student local audience unverified."],
            },
            "startup_state": {"areas": {}},
        }
        res = compose_learning_delta(state)
        summary = res["learning_delta"]["founder_facing_summary"]
        assert "directional support" in summary
        assert "contextual" in summary or "unconfirmed" in summary

    def test_15_contradicted_and_inconclusive_semantic_integrity(self):
        """15. CONTRADICTED and INCONCLUSIVE remain semantically correct."""
        state_contra: StartaryGraphState = {
            "project_id": "proj-contra",
            "iteration": 2,
            "previous_weak_area": "Demand Signal",
            "weak_area": "Demand Signal",
            "previous_assumptions": [
                {"id": "asmp-c", "area": "Demand Signal", "statement": "Managers will pre-order software.", "status": "CONTRADICTED"}
            ],
            "previous_experiments": [{"id": "exp-c", "status": "COMPLETED"}],
            "founder_evidence": [{"id": "ev-c", "experiment_id": "exp-c", "result": "ANALYZED", "evidence_type": "EXPERIMENT_RESULT"}],
            "evidence_analysis": {
                "conclusion": "CONTRADICTED",
                "evidence_strength": "STRONG",
                "contradiction_findings": ["0 of 8 managers willing to pre-order."],
            },
            "startup_state": {"areas": {}},
        }
        res_contra = compose_learning_delta(state_contra)
        assert "contradicted or significantly weakened" in res_contra["learning_delta"]["founder_facing_summary"]

        state_inconc: StartaryGraphState = {
            "project_id": "proj-inconc",
            "iteration": 2,
            "previous_weak_area": "Problem",
            "weak_area": "Problem",
            "previous_assumptions": [
                {"id": "asmp-i", "area": "Problem", "statement": "Problem statement", "status": "INCONCLUSIVE"}
            ],
            "previous_experiments": [{"id": "exp-i", "status": "COMPLETED"}],
            "founder_evidence": [{"id": "ev-i", "experiment_id": "exp-i", "result": "ANALYZED", "evidence_type": "EXPERIMENT_RESULT"}],
            "evidence_analysis": {
                "conclusion": "INCONCLUSIVE",
                "evidence_strength": "WEAK",
                "inconclusive_findings": ["Only 1 manager interviewed."],
            },
            "startup_state": {"areas": {}},
        }
        res_inconc = compose_learning_delta(state_inconc)
        assert "inconclusive" in res_inconc["learning_delta"]["founder_facing_summary"]

    def test_16_fallback_learning_input_does_not_masquerade_as_completed_evidence_cycle(self):
        """16. Fallback learning input does not masquerade as a completed evidence cycle."""
        state: StartaryGraphState = {
            "project_id": "proj-fallback-test",
            "iteration": 2,
            "previous_weak_area": "Problem",
            "weak_area": "Problem",
            "previous_assumptions": [
                {"id": "asmp-1", "area": "Problem", "statement": "Problem statement"}
            ],
            "previous_experiments": [
                {"id": "exp-failed", "status": "QUALITY_BAR_NOT_MET"}
            ],
            "founder_evidence": [
                {
                    "source": "FALLBACK_LEARNING_ACTION",
                    "content": "I talked to a bakery manager informally.",
                    "target_area": "Problem",
                }
            ],
            "evidence_analysis": None,
            "startup_state": {"areas": {}},
        }
        assert can_compose_learning_delta(state) is False
        assert get_most_recent_completed_evidence_cycle(state) is None

    def test_17_incomplete_or_rejected_experiment_does_not_create_learning_delta(self):
        """17. Incomplete/rejected experiment does not create LearningDelta."""
        state: StartaryGraphState = {
            "project_id": "proj-incomplete-exp",
            "iteration": 2,
            "previous_weak_area": "Problem",
            "weak_area": "Problem",
            "previous_assumptions": [{"id": "asmp-1", "area": "Problem", "statement": "Asmp 1"}],
            "previous_experiments": [
                {"id": "exp-abandoned", "status": "ABANDONED"}
            ],
            "founder_evidence": [],
            "evidence_analysis": {
                "conclusion": "SUPPORTED",
            },
            "startup_state": {"areas": {}},
        }
        assert can_compose_learning_delta(state) is False
        assert get_most_recent_completed_evidence_cycle(state) is None


# ===========================================================================
# PART C: POSITIVE SIGNAL EXTRACTION (Tests 18-21)
# ===========================================================================

class TestPositiveSignalExtraction:
    def test_18_five_real_paid_pickups_produce_grounded_positive_signal(self):
        """18. Five real paid pickups produce at least one grounded positive signal."""
        state = {
            "evidence_analysis": {
                "support_findings": [
                    "5 completed paid pickups occurred from real surplus.",
                ],
                "valid_learnings": [
                    "Successful buyers were generally already located near the bakery.",
                ],
            },
            "founder_messages": [
                {"role": "founder", "content": "We had 5 completed paid pickups from bakery surplus."}
            ],
        }
        signals = extract_positive_signals(state=state)
        assert len(signals) >= 1
        assert "No strong positive signal is established yet." not in signals
        assert any("paid" in s.lower() or "pickup" in s.lower() for s in signals)

    def test_19_merchant_willingness_to_repeat_surfaced_as_positive_signal(self):
        """19. Merchant willingness to repeat can be surfaced as a positive operational signal."""
        state = {
            "evidence_analysis": {
                "support_findings": [
                    "The participating merchant was willing to repeat the manual process.",
                    "The bakery reported that the manual process was manageable.",
                ]
            },
        }
        signals = extract_positive_signals(state=state)
        assert len(signals) >= 1
        assert "No strong positive signal is established yet." not in signals
        assert any("repeat" in s.lower() or "manageable" in s.lower() for s in signals)

    def test_20_positive_signals_do_not_claim_pmf_or_business_validation(self):
        """20. Positive signals remain contextual and do not claim PMF or full business validation."""
        state = {
            "evidence_analysis": {
                "support_findings": [
                    "5 completed paid pickups occurred from real surplus.",
                    "The participating merchant was willing to repeat the manual process.",
                ]
            },
            "diagnosis_snapshot": {
                "current_advantages": [
                    "Strong product-market fit established.",
                    "Bodega owner cousin gives direct channel access.",
                ]
            },
        }
        signals = extract_positive_signals(state=state)
        for s in signals:
            s_lower = s.lower()
            assert "product-market fit" not in s_lower
            assert "product market fit" not in s_lower
            assert "validated demand" not in s_lower
            assert "guaranteed success" not in s_lower

    def test_21_truly_empty_evidence_renders_no_positive_signal_fallback(self):
        """21. Truly empty/no-positive evidence still renders the no-positive fallback."""
        empty_state = {
            "evidence_analysis": {},
            "founder_evidence": [],
            "founder_messages": [],
            "startup_state": {},
        }
        signals = extract_positive_signals(state=empty_state)
        assert signals == ["No strong positive signal is established yet."]
