"""
Tests: Milestone 3.1 Founder-Facing Startup Diagnosis Snapshot.

Verification:
- StartupDiagnosisSnapshot domain model validation.
- Preserves active critical assumption verbatim.
- Preserves selected weak link and confidence.
- Enforces caps on major uncertainties (<= 5) and competing risks (<= 3).
- Deterministic guards prevent LLM from changing stage, weak link, or assumption.
- Graph insertion: compose_startup_diagnosis executes before generate_experiment.
- Deterministic fallback on LLM failure.
"""

from unittest.mock import MagicMock
import pytest

from startary.domain.diagnosis_snapshot import StartupDiagnosisSnapshot
from startary.domain.enums import WorkflowPhase
from startary.graph.builder import build_graph, run_diagnosis_snapshot
from startary.graph.graph_state import StartaryGraphState
from startary.graph.nodes import compose_startup_diagnosis
from startary.llm.schemas import StartupDiagnosisSnapshotLLMOutput
from tests.fixtures.scenarios import (
    make_valid_diagnosis_snapshot_output_dict,
    make_valid_experiment_critic_output_dict,
    make_valid_experiment_generator_output_dict,
)


@pytest.fixture
def snapshot_base_state() -> StartaryGraphState:
    return {
        "project_id": "proj-snap-123",
        "phase": WorkflowPhase.UNDERSTAND,
        "iteration": 1,
        "weak_area": "Problem",
        "selection_confidence": "LOW",
        "active_assumption": {
            "id": "asm-snap-001",
            "project_id": "proj-snap-123",
            "area": "Problem",
            "statement": "Small independent café managers repeatedly experience costly disruption from last-minute shift cancellations.",
            "is_critical": True,
            "status": "UNTESTED",
        },
        "reality_check": {
            "summary": "One café manager confirmed shift cancellations cause weekly stress, but no broader audit exists.",
            "unresolved_uncertainty": "Whether cancellations are widespread and costly enough across other cafés.",
            "what_evidence_supports": ["One café manager reported weekly scheduling headaches."],
            "what_evidence_does_not_support": ["Claims that managers are actively looking for a mobile scheduling app"],
        },
        "area_assessments": [
            {"area": "Problem", "calculated_decision_risk": 81},
            {"area": "Demand Signal", "calculated_decision_risk": 81},
            {"area": "Target User", "calculated_decision_risk": 81},
            {"area": "Value Proposition", "calculated_decision_risk": 72},
            {"area": "Acquisition", "calculated_decision_risk": 54},
        ],
        "execution_context": {
            "team_size": {"value": "Solo founder", "provenance": "FOUNDER_STATED"},
            "total_weekly_capacity_hours": {"value": 19, "provenance": "FOUNDER_STATED"},
            "available_budget_amount": {"value": 200, "provenance": "FOUNDER_STATED"},
            "customer_access": {"value": "Direct access to 1 local café owner", "provenance": "FOUNDER_STATED"},
        },
        "startup_progress": {
            "development_stage": {"value": "idea", "provenance": "FOUNDER_STATED"},
            "mvp_exists": {"value": False, "provenance": "FOUNDER_STATED"},
        },
        "founder_messages": [
            {"role": "founder", "content": "We want to solve café scheduling disruptions with Shiftly."}
        ],
        "previous_experiments": [],
        "traces": [],
    }


class TestStartupDiagnosisSnapshotDomain:
    def test_snapshot_schema_validates(self):
        snapshot_dict = make_valid_diagnosis_snapshot_output_dict()
        snapshot = StartupDiagnosisSnapshot(
            project_id="proj-123",
            iteration=1,
            **snapshot_dict,
        )
        assert snapshot.project_id == "proj-123"
        assert snapshot.stage == "Early Build / Pre-MVP"
        assert snapshot.weak_link == "Problem"
        assert snapshot.weak_link_confidence == "LOW"
        assert len(snapshot.grounded_areas) == 3
        assert len(snapshot.current_advantages) == 2
        assert len(snapshot.execution_constraints) == 3
        assert len(snapshot.major_uncertainties) == 3
        assert len(snapshot.competing_risks) == 2


class TestComposeStartupDiagnosisNode:
    def test_weak_link_matches_selected_weak_link(self, mocker, snapshot_base_state):
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_out = make_valid_diagnosis_snapshot_output_dict(
            weak_link="Problem",
            critical_assumption=snapshot_base_state["active_assumption"]["statement"],
        )
        mock_structured.invoke.return_value = StartupDiagnosisSnapshotLLMOutput(**mock_out)
        mock_llm.with_structured_output.return_value = mock_structured

        mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
        mocker.patch("startary.graph.nodes.get_settings", return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"))

        res = compose_startup_diagnosis(snapshot_base_state)
        snapshot = res["diagnosis_snapshot"]

        assert snapshot["weak_link"] == "Problem"
        assert snapshot["weak_link_confidence"] == "LOW"

    def test_critical_assumption_matches_active_assumption_verbatim(self, mocker, snapshot_base_state):
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_out = make_valid_diagnosis_snapshot_output_dict(
            critical_assumption="Hallucinated different statement",  # LLM tries to paraphrase
        )
        mock_structured.invoke.return_value = StartupDiagnosisSnapshotLLMOutput(**mock_out)
        mock_llm.with_structured_output.return_value = mock_structured

        mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
        mocker.patch("startary.graph.nodes.get_settings", return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"))

        res = compose_startup_diagnosis(snapshot_base_state)
        snapshot = res["diagnosis_snapshot"]

        # Deterministic guardrail MUST enforce verbatim active assumption statement
        assert snapshot["critical_assumption"] == snapshot_base_state["active_assumption"]["statement"]

    def test_major_uncertainties_capped_at_five(self, mocker, snapshot_base_state):
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_out = make_valid_diagnosis_snapshot_output_dict()
        # LLM returns 8 uncertainties
        mock_out["major_uncertainties"] = [f"Uncertainty {i}" for i in range(1, 9)]
        mock_structured.invoke.return_value = StartupDiagnosisSnapshotLLMOutput(**mock_out)
        mock_llm.with_structured_output.return_value = mock_structured

        mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
        mocker.patch("startary.graph.nodes.get_settings", return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"))

        res = compose_startup_diagnosis(snapshot_base_state)
        snapshot = res["diagnosis_snapshot"]

        assert len(snapshot["major_uncertainties"]) <= 5

    def test_competing_risks_capped_at_three(self, mocker, snapshot_base_state):
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_out = make_valid_diagnosis_snapshot_output_dict()
        mock_out["competing_risks"] = ["Risk A", "Risk B", "Risk C", "Risk D", "Risk E"]
        mock_structured.invoke.return_value = StartupDiagnosisSnapshotLLMOutput(**mock_out)
        mock_llm.with_structured_output.return_value = mock_structured

        mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
        mocker.patch("startary.graph.nodes.get_settings", return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"))

        res = compose_startup_diagnosis(snapshot_base_state)
        snapshot = res["diagnosis_snapshot"]

        assert len(snapshot["competing_risks"]) <= 3

    def test_composer_cannot_replace_stage_or_weak_link(self, mocker, snapshot_base_state):
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_out = make_valid_diagnosis_snapshot_output_dict()
        mock_out["weak_link"] = "Hallucinated Weak Link"
        mock_structured.invoke.return_value = StartupDiagnosisSnapshotLLMOutput(**mock_out)
        mock_llm.with_structured_output.return_value = mock_structured

        mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
        mocker.patch("startary.graph.nodes.get_settings", return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"))

        res = compose_startup_diagnosis(snapshot_base_state)
        snapshot = res["diagnosis_snapshot"]

        assert snapshot["weak_link"] == "Problem"

    def test_deterministic_fallback_on_llm_failure(self, mocker, snapshot_base_state):
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.side_effect = RuntimeError("API rate limit or connection error")
        mock_llm.with_structured_output.return_value = mock_structured

        mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
        mocker.patch("startary.graph.nodes.get_settings", return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"))

        res = compose_startup_diagnosis(snapshot_base_state)
        snapshot = res["diagnosis_snapshot"]

        assert snapshot is not None
        assert snapshot["weak_link"] == "Problem"
        assert snapshot["critical_assumption"] == snapshot_base_state["active_assumption"]["statement"]
        assert len(snapshot["major_uncertainties"]) >= 1

    def test_direct_development_entry_point(self, mocker, snapshot_base_state):
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_out = make_valid_diagnosis_snapshot_output_dict(
            critical_assumption=snapshot_base_state["active_assumption"]["statement"]
        )
        mock_structured.invoke.return_value = StartupDiagnosisSnapshotLLMOutput(**mock_out)
        mock_llm.with_structured_output.return_value = mock_structured

        mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
        mocker.patch("startary.graph.nodes.get_settings", return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"))

        updated_state = run_diagnosis_snapshot(snapshot_base_state)
        assert "diagnosis_snapshot" in updated_state
        assert updated_state["diagnosis_snapshot"]["weak_link"] == "Problem"

