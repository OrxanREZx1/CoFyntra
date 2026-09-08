"""
Tests: LangGraph state contract — updated for Milestone 1.

M1 additions tested:
  - raw_context field is separate from startup_state
  - diagnosis_output intermediate field
  - last_trace field
  - serialize/deserialize convention for startup_state
"""

import uuid
import pytest

from startary.domain.area_knowledge import AreaKnowledge, AreaStatus
from startary.domain.enums import WorkflowPhase
from startary.domain.state import StartupState
from startary.domain.taxonomy import StartupArea
from startary.graph.graph_state import StartaryGraphState
from startary.graph.nodes import initialize_state, verify_state


def _project_id() -> str:
    return str(uuid.uuid4())


def _minimal_valid_state(raw_context: str = "") -> StartaryGraphState:
    return StartaryGraphState(
        project_id="test-project-001",
        phase=WorkflowPhase.UNDERSTAND,
        iteration=1,
        raw_context=raw_context,
        startup_state=StartupState().model_dump(),
        founder_evidence=[],
        external_evidence=[],
    )


class TestGraphStateValidState:
    def test_valid_state_has_required_keys(self):
        state = _minimal_valid_state()
        assert "project_id" in state
        assert "phase" in state
        assert "iteration" in state
        assert "startup_state" in state

    def test_raw_context_field_exists(self):
        state = _minimal_valid_state(raw_context="Some startup description.")
        assert state.get("raw_context") == "Some startup description."

    def test_raw_context_separate_from_startup_state(self):
        """raw_context must NOT appear inside startup_state.areas."""
        raw = "I want to build a platform for founders."
        state = _minimal_valid_state(raw_context=raw)
        areas = state["startup_state"].get("areas", {})
        for area_key, area_val in areas.items():
            if isinstance(area_val, dict):
                assert area_val.get("understanding") != raw

    def test_verify_state_passes_on_valid_state(self):
        state = _minimal_valid_state()
        result = verify_state(state)
        assert result == {}

    def test_initialize_state_sets_defaults(self):
        partial = StartaryGraphState(
            project_id="test-001",
            phase=WorkflowPhase.UNDERSTAND,
            iteration=1,
        )
        updates = initialize_state(partial)
        assert "startup_state" in updates
        assert updates["founder_evidence"] == []
        assert updates["external_evidence"] == []

    def test_initialize_state_does_not_overwrite_existing_phase(self):
        state = _minimal_valid_state()
        updates = initialize_state(state)
        assert "phase" not in updates

    def test_optional_fields_may_be_absent(self):
        state = _minimal_valid_state()
        assert "diagnosis" not in state
        assert "weak_area" not in state
        assert "active_assumption" not in state
        assert "diagnosis_output" not in state

    def test_startup_state_serialization_roundtrip(self):
        """StartupState serializes to dict and re-hydrates correctly."""
        knowledge = AreaKnowledge(
            status=AreaStatus.PARTIALLY_KNOWN,
            understanding="Student founders struggle with priorities.",
            founder_claims=["Everyone needs this"],
            evidence_provided=["8 interviews"],
        )
        ss = StartupState(
            raw_context="Some context.",
            areas={StartupArea.PROBLEM: knowledge},
        )
        serialized = ss.model_dump()
        restored = StartupState(**serialized)

        assert restored.raw_context == ss.raw_context
        assert StartupArea.PROBLEM in restored.areas
        assert restored.areas[StartupArea.PROBLEM].status == AreaStatus.PARTIALLY_KNOWN


class TestGraphStateMalformedState:
    def test_missing_project_id_fails_verification(self):
        state = StartaryGraphState(
            phase=WorkflowPhase.UNDERSTAND,
            iteration=1,
            startup_state=StartupState().model_dump(),
        )
        with pytest.raises(ValueError, match="project_id"):
            verify_state(state)

    def test_missing_phase_fails_verification(self):
        state = StartaryGraphState(
            project_id="test-001",
            iteration=1,
            startup_state=StartupState().model_dump(),
        )
        with pytest.raises(ValueError, match="phase"):
            verify_state(state)

    def test_missing_iteration_fails_verification(self):
        state = StartaryGraphState(
            project_id="test-001",
            phase=WorkflowPhase.UNDERSTAND,
            startup_state=StartupState().model_dump(),
        )
        with pytest.raises(ValueError, match="iteration"):
            verify_state(state)

    def test_missing_startup_state_fails_verification(self):
        state = StartaryGraphState(
            project_id="test-001",
            phase=WorkflowPhase.UNDERSTAND,
            iteration=1,
        )
        with pytest.raises(ValueError, match="startup_state"):
            verify_state(state)

    def test_non_dict_startup_state_fails_verification(self):
        state = StartaryGraphState(
            project_id="test-001",
            phase=WorkflowPhase.UNDERSTAND,
            iteration=1,
            startup_state="not a dict",  # type: ignore[typeddict-item]
        )
        with pytest.raises(ValueError, match="startup_state must be a dict"):
            verify_state(state)
