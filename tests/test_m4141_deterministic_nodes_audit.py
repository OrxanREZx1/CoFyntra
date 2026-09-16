"""
Milestone M4.1.4.1 Audit: Deterministic Nodes LLM Call Audit.

Asserts that every node claimed to be deterministic makes strictly ZERO calls to
the LLM provider (get_llm_client), validating architectural integrity and eliminating
false claims of deterministic behavior.
"""

from unittest.mock import MagicMock
import pytest

from startary.domain.enums import WorkflowPhase
from startary.graph.nodes import (
    apply_validated_state_updates,
    assess_build_readiness,
    assess_can_execute,
    assess_evidence_submission_sufficiency,
    compose_learning_delta,
    compose_startup_diagnosis,
    decide_next_strategic_action,
    handle_experiment_failure_stop,
    increment_iteration,
    merge_diagnosis,
    merge_founder_answer,
    save_final_experiment,
    select_weak_link,
    validate_evidence_analysis,
    validate_proposed_experiment,
)


@pytest.fixture
def mock_forbidden_llm(monkeypatch):
    """Ensure any call to get_llm_client raises an immediate AssertionError."""
    mock_fn = MagicMock(side_effect=AssertionError("Deterministic node must NEVER call get_llm_client!"))
    monkeypatch.setattr("startary.graph.nodes.get_llm_client", mock_fn)
    monkeypatch.setattr("startary.llm.provider.get_llm_client", mock_fn)
    return mock_fn


@pytest.fixture
def audit_base_state():
    return {
        "project_id": "proj-audit-001",
        "phase": WorkflowPhase.UNDERSTAND,
        "iteration": 1,
        "weak_area": "Problem",
        "weak_link": "Problem",
        "weak_link_details": {"confidence": "LOW", "score": 81},
        "active_assumption": {
            "id": "asm-audit-001",
            "project_id": "proj-audit-001",
            "area": "Problem",
            "statement": "Independent café managers face acute weekly scheduling disruption.",
            "is_critical": True,
            "status": "UNTESTED",
        },
        "reality_check": {
            "summary": "1 café manager confirmed weekly scheduling headaches.",
            "unresolved_uncertainty": "Whether problem is acute enough across 10+ cafés.",
            "what_evidence_supports": ["1 manager interview"],
            "what_evidence_does_not_support": ["No willingness to pay verified yet"],
        },
        "area_assessments_output": {
            "assessments": [
                {"area": "Problem", "importance": "HIGH", "uncertainty": "HIGH", "evidence_gap": "HIGH", "consequence_if_wrong": "HIGH"},
                {"area": "Value Proposition", "importance": "HIGH", "uncertainty": "MEDIUM", "evidence_gap": "HIGH", "consequence_if_wrong": "HIGH"},
            ]
        },
        "diagnosis": {
            "summary": "Early-stage concept with strong problem uncertainty.",
            "startup_stage": "IDEA",
        },
        "diagnosis_output": {
            "diagnosis_summary": "Early-stage concept with strong problem uncertainty.",
            "startup_stage": "IDEA",
            "area_assessments": [
                {"area": "Problem", "status": "UNKNOWN", "understanding": "Uncertain"},
            ],
        },
        "execution_context": {
            "team_size": {"value": "Solo founder", "provenance": "FOUNDER_STATED"},
            "monthly_budget": {"value": "$200", "provenance": "FOUNDER_STATED"},
            "weekly_hours": {"value": "15 hrs/wk", "provenance": "FOUNDER_STATED"},
            "customer_access": {"value": "Access to local cafés", "provenance": "FOUNDER_STATED"},
        },
        "startup_progress": {
            "development_stage": {"value": "idea", "provenance": "FOUNDER_STATED"},
            "mvp_exists": {"value": False, "provenance": "FOUNDER_STATED"},
        },
        "startup_state": {
            "areas": {
                "Problem": {"status": "UNKNOWN", "understanding": "High uncertainty"},
            }
        },
        "founder_messages": [
            {"role": "founder", "content": "I want to help cafés organize shifts easily."}
        ],
        "proposed_experiment_output": {
            "assumption_id": "asm-audit-001",
            "experiment_type": "CUSTOMER_INTERVIEW",
            "objective": "Verify shift disruption with 5 café managers.",
            "target_sample_size": 5,
            "estimated_cost": "$0",
            "estimated_duration_days": 7,
            "method": "Structured in-person interviews with café managers.",
            "evidence_to_collect": "Number of shift cancellations and hours spent re-scheduling.",
            "support_signal": ">= 4 managers report > 3 cancellations per week.",
            "contradiction_signal": "<= 1 manager reports cancellations as an issue.",
            "inconclusive_signal": "2-3 managers report cancellations, but without disruption.",
            "founder_actions": ["Visit local cafés", "Conduct interviews"],
            "valid_inferences": ["Managers experience friction"],
            "invalid_inferences": ["Managers will buy an app"],
        },
        "active_experiment": {
            "id": "exp-audit-001",
            "assumption_id": "asm-audit-001",
            "status": "IN_PROGRESS",
            "target_sample_size": 5,
            "collected_sample_count": 5,
        },
        "evidence_analysis": {
            "conclusion": "SUPPORTED",
            "evidence_strength": "STRONG",
            "experiment_execution_quality": "HIGH",
            "support_findings": ["5/5 managers reported shift disruptions"],
            "contradiction_findings": [],
            "inconclusive_findings": [],
            "valid_learnings": ["Shift cancellations are severe"],
            "invalid_learnings": [],
            "reasoning_summary": "All managers confirmed severe friction.",
            "recommended_state_updates": {},
        },
        "founder_response": "I spoke with 5 café managers and all 5 confirmed severe disruptions.",
        "previous_experiments": [],
        "traces": [],
    }


def test_compose_startup_diagnosis_zero_llm_calls(mock_forbidden_llm, audit_base_state):
    res = compose_startup_diagnosis(audit_base_state)
    assert mock_forbidden_llm.call_count == 0
    assert "diagnosis_snapshot" in res
    assert res["diagnosis_snapshot"]["weak_link"] == "Problem"
    assert "STARTARY — CURRENT DIAGNOSIS" in res["founder_diagnosis_text"]


def test_select_weak_link_zero_llm_calls(mock_forbidden_llm, audit_base_state):
    res = select_weak_link(audit_base_state)
    assert mock_forbidden_llm.call_count == 0
    assert res["weak_area"] == "Problem"


def test_merge_founder_answer_zero_llm_calls(mock_forbidden_llm, audit_base_state):
    state = dict(audit_base_state)
    state["founder_messages"] = [{"role": "system", "content": "Hello"}]
    state["founder_response"] = "We are 2 founders"
    res = merge_founder_answer(state)
    assert mock_forbidden_llm.call_count == 0
    assert len(res["founder_messages"]) == 2


def test_merge_diagnosis_zero_llm_calls(mock_forbidden_llm, audit_base_state):
    res = merge_diagnosis(audit_base_state)
    assert mock_forbidden_llm.call_count == 0
    assert "diagnosis" in res
    assert "startup_state" in res


def test_validate_proposed_experiment_zero_llm_calls(mock_forbidden_llm, audit_base_state):
    res = validate_proposed_experiment(audit_base_state)
    assert mock_forbidden_llm.call_count == 0
    assert "proposed_experiment_output" in res


def test_save_final_experiment_zero_llm_calls(mock_forbidden_llm, audit_base_state):
    res = save_final_experiment(audit_base_state)
    assert mock_forbidden_llm.call_count == 0
    assert "active_experiment" in res
    assert res["active_experiment"]["status"] == "READY"


def test_assess_can_execute_zero_llm_calls(mock_forbidden_llm, audit_base_state):
    res = assess_can_execute(audit_base_state)
    assert mock_forbidden_llm.call_count == 0
    assert "can_execute_result" in res
    assert res["can_execute_result"]["readiness"] == "CAN_EXECUTE"


def test_handle_experiment_failure_stop_zero_llm_calls(mock_forbidden_llm, audit_base_state):
    res = handle_experiment_failure_stop(audit_base_state)
    assert mock_forbidden_llm.call_count == 0
    assert "experiment_design_status" in res


def test_assess_evidence_submission_sufficiency_zero_llm_calls(mock_forbidden_llm, audit_base_state):
    res = assess_evidence_submission_sufficiency(audit_base_state)
    assert mock_forbidden_llm.call_count == 0
    assert "evidence_submission_sufficiency" in res


def test_validate_evidence_analysis_zero_llm_calls(mock_forbidden_llm, audit_base_state):
    state = dict(audit_base_state)
    state["active_evidence"] = {"result": "SUPPORTED", "raw_content": "5 interviews"}
    res = validate_evidence_analysis(state)
    assert mock_forbidden_llm.call_count == 0


def test_apply_validated_state_updates_zero_llm_calls(mock_forbidden_llm, audit_base_state):
    res = apply_validated_state_updates(audit_base_state)
    assert mock_forbidden_llm.call_count == 0
    assert "startup_state" in res


def test_increment_iteration_zero_llm_calls(mock_forbidden_llm, audit_base_state):
    res = increment_iteration(audit_base_state)
    assert mock_forbidden_llm.call_count == 0
    assert res["iteration"] == 2


def test_compose_learning_delta_zero_llm_calls(mock_forbidden_llm, audit_base_state):
    res = compose_learning_delta(audit_base_state)
    assert mock_forbidden_llm.call_count == 0
    assert "learning_delta" in res


def test_assess_build_readiness_zero_llm_calls(mock_forbidden_llm, audit_base_state):
    res = assess_build_readiness(audit_base_state)
    assert mock_forbidden_llm.call_count == 0
    assert "build_readiness" in res


def test_decide_next_strategic_action_zero_llm_calls(mock_forbidden_llm, audit_base_state):
    res = decide_next_strategic_action(audit_base_state)
    assert mock_forbidden_llm.call_count == 0
    assert "next_strategic_action" in res
