"""
tests/test_uncertainty_handoff.py
Tests for Milestone 4.0.2 — Uncertainty-to-Test Handoff.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from startary.domain.enums import WorkflowPhase
from startary.domain.gap_plan import (
    GapPriority,
    GapTargetDomain,
    InformationGapPlan,
    UncertaintyResolutionMode,
    can_formulate_testable_uncertainty,
)
from startary.graph.builder import (
    MAX_STRATEGIC_CLARIFICATION_TURNS,
    route_on_context_sufficiency,
    route_on_gap_plan,
    route_on_weak_link,
)
from startary.graph.nodes import plan_information_gap, assess_weak_link_discrimination
from startary.llm.schemas import InformationGapPlanLLMOutput, WeakLinkDiscriminationLLMOutput


# ---------------------------------------------------------------------------
# Test 1: UncertaintyResolutionMode enum and InformationGapPlan model fields
# ---------------------------------------------------------------------------

def test_uncertainty_resolution_mode_enum_and_model():
    assert UncertaintyResolutionMode.FOUNDER_KNOWLEDGE.value == "FOUNDER_KNOWLEDGE"
    assert UncertaintyResolutionMode.EXTERNAL_EVIDENCE.value == "EXTERNAL_EVIDENCE"
    assert UncertaintyResolutionMode.NOT_YET_FORMULATABLE.value == "NOT_YET_FORMULATABLE"

    plan = InformationGapPlan(
        should_ask=False,
        strategic_sufficient=True,
        resolution_mode=UncertaintyResolutionMode.EXTERNAL_EVIDENCE,
        can_formulate_testable_uncertainty=True,
        why_this_matters="Validating customer demand before building",
        missing_information="Whether merchants experience return losses as urgent",
        recommended_area="Problem",
        reason="External evidence required to confirm problem severity",
    )
    assert plan.resolution_mode == UncertaintyResolutionMode.EXTERNAL_EVIDENCE
    assert plan.can_formulate_testable_uncertainty is True
    assert plan.recommended_area == "Problem"
    assert plan.missing_information == "Whether merchants experience return losses as urgent"


# ---------------------------------------------------------------------------
# Test 2: Deterministic can_formulate_testable_uncertainty detector
# ---------------------------------------------------------------------------

def test_can_formulate_testable_uncertainty_detector():
    # LoopCart scenario: merchant target population and returns problem
    loopcart_state = {
        "raw_context": "B2B Shopify app for small merchants to reduce product returns. 2 technical co-founders.",
        "startup_state": {
            "area_knowledge": {
                "Target User": {
                    "understanding": "Small Shopify merchants selling fashion and home goods.",
                    "founder_claims": ["Merchants have 20-30% return rates"],
                },
                "Problem": {
                    "understanding": "High product returns destroy profit margins for online stores.",
                    "founder_claims": ["Merchants hate returns"],
                },
            }
        },
        "founder_messages": [
            {"role": "founder", "content": "We built an MVP for our cousin's store, but we haven't spoken to external merchants yet."}
        ],
    }
    assert can_formulate_testable_uncertainty(loopcart_state) is True

    # Truly vague non-idea: no target user, no problem, vague AI statement
    vague_state = {
        "raw_context": "I want to build something with AI but don't know what.",
        "startup_state": {
            "area_knowledge": {
                "Target User": {"understanding": "", "founder_claims": []},
                "Problem": {"understanding": "", "founder_claims": []},
            }
        },
        "founder_messages": [
            {"role": "founder", "content": "I want to make an AI startup. Not sure what to build yet."}
        ],
    }
    assert can_formulate_testable_uncertainty(vague_state) is False

    # Gap plan override: if gap_plan explicitly marked can_formulate_testable_uncertainty=True
    assert can_formulate_testable_uncertainty({}, {"can_formulate_testable_uncertainty": True}) is True
    assert can_formulate_testable_uncertainty({}, {"resolution_mode": "NOT_YET_FORMULATABLE"}) is False


# ---------------------------------------------------------------------------
# Test 3: FOUNDER_KNOWLEDGE gap routes to clarification
# ---------------------------------------------------------------------------

def test_founder_known_gap_routes_to_clarification():
    state = {
        "strategic_clarification_turns": 1,
        "gap_plan": {
            "should_ask": True,
            "strategic_sufficient": False,
            "resolution_mode": UncertaintyResolutionMode.FOUNDER_KNOWLEDGE.value,
            "question": "How much time can your team realistically commit each week?",
            "can_formulate_testable_uncertainty": True,
        },
    }
    next_node = route_on_gap_plan(state)
    assert next_node == "ask_and_wait_for_founder"


# ---------------------------------------------------------------------------
# Test 4: EXTERNAL_EVIDENCE gap routes directly to TEST (assess_startup_areas)
# ---------------------------------------------------------------------------

def test_external_evidence_gap_routes_to_test():
    state = {
        "strategic_clarification_turns": 1,
        "gap_plan": {
            "should_ask": False,
            "strategic_sufficient": False,
            "resolution_mode": UncertaintyResolutionMode.EXTERNAL_EVIDENCE.value,
            "can_formulate_testable_uncertainty": True,
            "missing_information": "Whether merchants will adopt a third-party return widget",
            "recommended_area": "Problem",
        },
    }
    next_node = route_on_gap_plan(state)
    # Must route directly to assess_startup_areas (the entry to TEST), NOT handle_insufficient_context_stop
    assert next_node == "assess_startup_areas"


# ---------------------------------------------------------------------------
# Test 5: "I don't know" on external issue triggers uncertainty handoff
# ---------------------------------------------------------------------------

def test_i_dont_know_on_external_issue_triggers_handoff():
    mock_output = InformationGapPlanLLMOutput(
        should_ask=False,
        strategic_sufficient=False,
        resolution_mode=UncertaintyResolutionMode.EXTERNAL_EVIDENCE.value,
        can_formulate_testable_uncertainty=True,
        missing_information="Willingness of Shopify store owners to install return prevention software",
        why_this_matters="Core problem validation requires customer response",
        recommended_area="Problem",
        reason="Founder has no merchant conversations; remaining uncertainty is external evidence.",
        question=None,
    )

    with patch("startary.graph.nodes.get_llm_client") as mock_llm_factory:
        mock_client = MagicMock()
        mock_client.with_structured_output.return_value.invoke.return_value = mock_output
        mock_llm_factory.return_value = mock_client

        state = {
            "project_id": "proj_loopcart",
            "strategic_clarification_turns": 1,
            "founder_messages": [
                {"role": "founder", "content": "I honestly don't know whether merchants will pay or install it."}
            ],
            "raw_context": "Shopify returns app for merchants.",
            "startup_state": {
                "area_knowledge": {
                    "Target User": {"understanding": "Shopify merchants", "founder_claims": []},
                    "Problem": {"understanding": "High returns hurt margins", "founder_claims": []},
                }
            },
        }

        result = plan_information_gap(state)
        assert result["is_sufficient"] is True
        assert result["gap_plan"]["resolution_mode"] == UncertaintyResolutionMode.EXTERNAL_EVIDENCE.value
        assert result["gap_plan"]["can_formulate_testable_uncertainty"] is True
        assert result["active_next_turn_decision"] is None
        assert result["current_clarifying_question"] is None


# ---------------------------------------------------------------------------
# Test 6: Absence of customer interviews does NOT route to insufficient context stop
# ---------------------------------------------------------------------------

def test_i_havent_interviewed_customers_does_not_stop():
    state = {
        "raw_context": "B2B SaaS for restaurant shift scheduling.",
        "startup_state": {
            "area_knowledge": {
                "Target User": {"understanding": "Independent restaurant and cafe managers.", "founder_claims": []},
                "Problem": {"understanding": "Last-minute cancellations cause chaotic shift rescheduling.", "founder_claims": []},
            }
        },
        "strategic_clarification_turns": 2,
        "gap_plan": {
            "should_ask": False,
            "strategic_sufficient": False,
            "resolution_mode": UncertaintyResolutionMode.EXTERNAL_EVIDENCE.value,
            "can_formulate_testable_uncertainty": True,
            "missing_information": "Manager interest in automated shift-swapping",
        },
    }
    next_node = route_on_gap_plan(state)
    assert next_node == "assess_startup_areas"
    assert next_node != "handle_insufficient_context_stop"


# ---------------------------------------------------------------------------
# Test 7: Max turns reached with testable uncertainty proceeds to TEST
# ---------------------------------------------------------------------------

def test_max_turns_with_testable_uncertainty_proceeds_to_test():
    state = {
        "raw_context": "LoopCart B2B Shopify returns tool.",
        "startup_state": {
            "area_knowledge": {
                "Target User": {"understanding": "Shopify clothing merchants.", "founder_claims": []},
                "Problem": {"understanding": "Returns cost 15% of revenue.", "founder_claims": []},
            }
        },
        "strategic_clarification_turns": MAX_STRATEGIC_CLARIFICATION_TURNS,
        "gap_plan": {
            "should_ask": True,
            "strategic_sufficient": False,
            "resolution_mode": UncertaintyResolutionMode.FOUNDER_KNOWLEDGE.value,
            "can_formulate_testable_uncertainty": True,
        },
    }
    # Even though turns are exhausted and resolution_mode was FOUNDER_KNOWLEDGE,
    # because testable uncertainty exists, it must NOT fail with handle_insufficient_context_stop!
    next_node = route_on_gap_plan(state)
    assert next_node == "assess_startup_areas"


# ---------------------------------------------------------------------------
# Test 8: Truly vague startup stops safely on max turns
# ---------------------------------------------------------------------------

def test_not_yet_formulatable_stops_on_max_turns():
    state = {
        "raw_context": "Something with AI.",
        "startup_state": {"area_knowledge": {}},
        "strategic_clarification_turns": MAX_STRATEGIC_CLARIFICATION_TURNS,
        "gap_plan": {
            "should_ask": True,
            "strategic_sufficient": False,
            "resolution_mode": UncertaintyResolutionMode.NOT_YET_FORMULATABLE.value,
            "can_formulate_testable_uncertainty": False,
        },
    }
    next_node = route_on_gap_plan(state)
    assert next_node == "handle_insufficient_context_stop"


# ---------------------------------------------------------------------------
# Test 9: Generic context compatibility question fallback is suppressed when profile is known
# ---------------------------------------------------------------------------

def test_generic_context_compatibility_fallback_suppressed():
    from startary.graph.nodes import compose_advisor_response

    mock_composer_output = MagicMock()
    mock_composer_output.question_id = "q_compat_test"
    mock_composer_output.next_question_target = "Context"
    mock_composer_output.advisor_response = "We know your team and product, but need to understand how stores handle this."
    mock_composer_output.model_dump.return_value = {
        "question_id": "q_compat_test",
        "next_question_target": "Context",
        "advisor_response": "We know your team and product, but need to understand how stores handle this.",
    }

    with patch("startary.graph.nodes.get_llm_client") as mock_llm_factory:
        mock_client = MagicMock()
        mock_client.with_structured_output.return_value.invoke.return_value = mock_composer_output
        mock_llm_factory.return_value = mock_client

        state = {
            "project_id": "proj_compat_test",
            "execution_context": {
                "team_size": {"value": 2, "provenance": "EXPLICIT"},
                "weekly_capacity_hours": {"value": 30, "provenance": "EXPLICIT"},
            },
            "startup_progress": {
                "mvp_exists": {"value": True, "provenance": "EXPLICIT"},
            },
            "gap_plan": {
                "missing_information": "how merchants currently process returns",
            },
            # No active_next_turn_decision in state
            "active_next_turn_decision": None,
            "founder_messages": [
                {"role": "founder", "content": "We have 2 developers working 30h/week and our MVP is on our cousin's store."}
            ],
        }

        # The node should run without error and not ask the generic team/stage question
        res = compose_advisor_response(state)
        assert res is not None
        assert "latest_advisor_response" in res
        advisor_res = res["latest_advisor_response"]
        next_q = advisor_res.get("next_question") or advisor_res.get("next_question_rephrased") or ""
        assert "Could you describe your team" not in next_q


# ---------------------------------------------------------------------------
# Test 10: Weak-link discrimination with EXTERNAL_EVIDENCE proceeds under LOW/MEDIUM confidence
# ---------------------------------------------------------------------------

def test_weak_link_external_evidence_gap_proceeds_under_low_confidence():
    mock_disc_output = WeakLinkDiscriminationLLMOutput(
        selection_confidence="LOW",
        competing_areas=["Problem", "Demand Signal"],
        discrimination_reason="Deciding between Problem and Demand requires live customer commitment.",
        should_clarify=True,
        resolution_mode=UncertaintyResolutionMode.EXTERNAL_EVIDENCE.value,
        recommended_information_goal="Test whether merchants will install the widget",
    )

    with patch("startary.graph.nodes.get_llm_client") as mock_llm_factory:
        mock_client = MagicMock()
        mock_client.with_structured_output.return_value.invoke.return_value = mock_disc_output
        mock_llm_factory.return_value = mock_client

        state = {
            "project_id": "proj_disc_test",
            "weak_area": "Problem",
            "area_assessments": [
                {"area": "Problem", "risk_score": 81},
                {"area": "Demand Signal", "risk_score": 81},
            ],
            "weak_link_clarification_turns": 0,
            "diagnosis": {"summary": "High risk in Problem and Demand"},
        }

        node_result = assess_weak_link_discrimination(state)
        # Verify clarify_flag was overridden to False because resolution_mode == EXTERNAL_EVIDENCE
        assert node_result["should_clarify"] is False
        assert node_result["selection_confidence"] == "LOW"

        # Verify route_on_weak_link immediately proceeds to reality_check
        state.update(node_result)
        next_route = route_on_weak_link(state)
        assert next_route == "reality_check"
