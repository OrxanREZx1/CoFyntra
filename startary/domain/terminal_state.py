"""
Terminal State Rendering — Milestone 4.0.1.

Derives the true founder-facing workflow status directly from the actual final graph state
and renders consistent, factually grounded CLI banners and summaries.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from startary.domain.enums import ExecutionReadiness, ExperimentDesignStatus, ExperimentStatus, WorkflowPhase
from startary.domain.experiment import render_fallback_learning_action_text

_WIDE = "=" * 54
_DIVIDER = "-" * 54


class TerminalStatus(str, Enum):
    """Canonical workflow terminal statuses."""
    WAITING_FOR_EVIDENCE = "WAITING_FOR_EVIDENCE"
    NEEDS_HELP = "NEEDS_HELP"
    BLOCKED = "BLOCKED"
    EXPERIMENT_READY = "EXPERIMENT_READY"
    EXPERIMENT_DESIGN_NEEDS_REVISION = "EXPERIMENT_DESIGN_NEEDS_REVISION"
    WAITING_FOR_LEARNING_INPUT = "EXPERIMENT_DESIGN_NEEDS_REVISION"
    CLARIFICATION_LIMIT_REACHED = "CLARIFICATION_LIMIT_REACHED"
    INITIAL_DIAGNOSIS_COMPLETE = "INITIAL_DIAGNOSIS_COMPLETE"


def can_render_waiting_for_evidence(state: dict[str, Any]) -> bool:
    """
    Deterministic Integrity Check (Milestone 4.0.7.1.1):
    Returns True ONLY if all of the following conditions are satisfied:
      1. phase == WAITING_FOR_EVIDENCE
      2. No fallback learning action is active (no fallback_learning_action,
         learning_input_mode != 'FALLBACK', experiment_design_status != QUALITY_BAR_NOT_MET)
      3. Current experiment exists
      4. Current experiment is accepted / executable (not BLOCKED, status in READY, IN_PROGRESS, or COMPLETED)
      5. Current execution handoff exists
      6. Handoff experiment_id matches current experiment_id
    If any condition fails, returns False.
    """
    if not isinstance(state, dict):
        return False

    # 1. phase == WAITING_FOR_EVIDENCE
    phase = state.get("phase")
    if phase not in (WorkflowPhase.WAITING_FOR_EVIDENCE, WorkflowPhase.WAITING_FOR_EVIDENCE.value, "WAITING_FOR_EVIDENCE"):
        return False

    # 2. No fallback learning action is active
    if state.get("fallback_learning_action") is not None:
        return False
    if state.get("learning_input_mode") == "FALLBACK":
        return False
    exp_design_status = state.get("experiment_design_status")
    if exp_design_status in (
        ExperimentDesignStatus.QUALITY_BAR_NOT_MET,
        ExperimentDesignStatus.QUALITY_BAR_NOT_MET.value,
        "QUALITY_BAR_NOT_MET",
    ):
        return False

    # 3. Current experiment exists
    active_exp = state.get("active_experiment")
    if not active_exp or not isinstance(active_exp, dict):
        return False

    # 4. Current experiment is accepted / executable (READY or IN_PROGRESS or COMPLETED)
    exp_status = active_exp.get("status")
    if exp_status not in (
        ExperimentStatus.READY, ExperimentStatus.READY.value, "READY",
        ExperimentStatus.IN_PROGRESS, ExperimentStatus.IN_PROGRESS.value, "IN_PROGRESS",
        ExperimentStatus.COMPLETED, ExperimentStatus.COMPLETED.value, "COMPLETED",
    ):
        return False

    can_exec = state.get("can_execute_result") or {}
    readiness = can_exec.get("readiness")
    if readiness in (ExecutionReadiness.BLOCKED, ExecutionReadiness.BLOCKED.value, "BLOCKED"):
        return False

    # 5. Current execution handoff exists
    handoff = state.get("active_execution_handoff")
    if not handoff or not isinstance(handoff, dict):
        return False

    # 6. Handoff experiment_id matches current experiment_id
    exp_id = active_exp.get("id")
    handoff_exp_id = handoff.get("experiment_id")
    if not exp_id or not handoff_exp_id or str(exp_id) != str(handoff_exp_id):
        return False

    # Defensive check: if iteration is tracked, ensure no cross-iteration leak
    curr_iter = state.get("iteration")
    created_iter = active_exp.get("created_iteration")
    if curr_iter is not None and created_iter is not None and created_iter != curr_iter:
        return False

    return True


def resolve_founder_facing_status(state: dict[str, Any]) -> TerminalStatus:
    """
    Deterministically derives the current terminal status from graph state.

    Precedence order (Milestone 4.0.7.1.1):
    1. WAITING_FOR_LEARNING_INPUT / QUALITY_BAR_NOT_MET / Fallback active -> EXPERIMENT_DESIGN_NEEDS_REVISION
    2. WAITING_FOR_EVIDENCE (strictly verified by can_render_waiting_for_evidence) -> WAITING_FOR_EVIDENCE
    3. can_execute_result.readiness == NEEDS_HELP -> NEEDS_HELP
    4. can_execute_result.readiness == BLOCKED -> BLOCKED
    5. active_experiment is valid/ready (and no quality bar failure) -> EXPERIMENT_READY
    6. phase == UNDERSTAND and clarification limit was actually hit -> CLARIFICATION_LIMIT_REACHED
    7. Default -> INITIAL_DIAGNOSIS_COMPLETE
    """
    if not isinstance(state, dict):
        return TerminalStatus.INITIAL_DIAGNOSIS_COMPLETE

    phase = state.get("phase")
    can_exec = state.get("can_execute_result") or {}
    readiness = can_exec.get("readiness")
    active_exp = state.get("active_experiment")
    exp_design_status = state.get("experiment_design_status")
    fallback = state.get("fallback_learning_action")
    status_msg = (state.get("status_message") or "").lower()

    # 1. WAITING_FOR_LEARNING_INPUT / QUALITY_BAR_NOT_MET / Fallback learning action
    if (
        phase in (
            WorkflowPhase.WAITING_FOR_LEARNING_INPUT,
            WorkflowPhase.WAITING_FOR_LEARNING_INPUT.value,
            "WAITING_FOR_LEARNING_INPUT",
        )
        or fallback is not None
        or exp_design_status in (
            ExperimentDesignStatus.QUALITY_BAR_NOT_MET,
            ExperimentDesignStatus.QUALITY_BAR_NOT_MET.value,
            "QUALITY_BAR_NOT_MET",
        )
        or "unable to produce an accepted experiment" in status_msg
        or "experiment generation stopped" in status_msg
    ):
        return TerminalStatus.EXPERIMENT_DESIGN_NEEDS_REVISION

    # 2. WAITING_FOR_EVIDENCE (strictly gated by can_render_waiting_for_evidence)
    if can_render_waiting_for_evidence(state):
        return TerminalStatus.WAITING_FOR_EVIDENCE

    # 3. NEEDS_HELP
    if readiness in (ExecutionReadiness.NEEDS_HELP, ExecutionReadiness.NEEDS_HELP.value, "NEEDS_HELP"):
        return TerminalStatus.NEEDS_HELP

    # 4. BLOCKED
    if readiness in (ExecutionReadiness.BLOCKED, ExecutionReadiness.BLOCKED.value, "BLOCKED"):
        return TerminalStatus.BLOCKED

    # 5. EXPERIMENT_READY (valid accepted experiment exists)
    if (
        active_exp
        and isinstance(active_exp, dict)
        and active_exp.get("status") in ("READY", "IN_PROGRESS", "COMPLETED")
        and not fallback
        and exp_design_status not in (ExperimentDesignStatus.QUALITY_BAR_NOT_MET, "QUALITY_BAR_NOT_MET")
    ):
        return TerminalStatus.EXPERIMENT_READY

    # 6. Only treat as clarification limit if workflow actually stopped in UNDERSTAND phase
    if (phase == WorkflowPhase.UNDERSTAND or phase == "UNDERSTAND") and "maximum clarification limit" in status_msg:
        return TerminalStatus.CLARIFICATION_LIMIT_REACHED

    return TerminalStatus.INITIAL_DIAGNOSIS_COMPLETE


def render_terminal_state(state: dict[str, Any]) -> str:
    """
    Renders the formatted banner and explanation for the terminal state.
    """
    status = resolve_founder_facing_status(state)
    can_exec = state.get("can_execute_result") or {}
    reason = can_exec.get("reason") or state.get("status_message") or ""

    lines: list[str] = []

    if status == TerminalStatus.WAITING_FOR_EVIDENCE:
        if not can_render_waiting_for_evidence(state):
            # Defensive guard: never render WAITING_FOR_EVIDENCE if can_render_waiting_for_evidence fails
            if state.get("fallback_learning_action") or state.get("experiment_design_status") in (
                ExperimentDesignStatus.QUALITY_BAR_NOT_MET, "QUALITY_BAR_NOT_MET"
            ):
                status = TerminalStatus.EXPERIMENT_DESIGN_NEEDS_REVISION
            else:
                status = TerminalStatus.INITIAL_DIAGNOSIS_COMPLETE
        else:
            lines.append(f"\n{_WIDE}")
            lines.append("STARTARY — WAITING FOR EVIDENCE")
            lines.append(_WIDE)
            lines.append("\nYour experiment is ready.")
            lines.append("Run it in the real world and return with what happened.")
            lines.append(f"{_DIVIDER}\n")

    elif status == TerminalStatus.NEEDS_HELP:
        lines.append(f"\n{_WIDE}")
        lines.append("STARTARY — EXPERIMENT READY, SUPPORT NEEDED")
        lines.append(_WIDE)
        lines.append("\nThe experiment is valid, but one requirement is not yet sufficiently established:")
        lines.append(f"\n{reason}\n")
        lines.append(f"{_DIVIDER}\n")

    elif status == TerminalStatus.BLOCKED:
        lines.append(f"\n{_WIDE}")
        lines.append("STARTARY — EXPERIMENT BLOCKED")
        lines.append(_WIDE)
        lines.append(f"\nThe current experiment cannot be executed because:")
        lines.append(f"\n{reason}\n")
        lines.append(f"{_DIVIDER}\n")

    elif status == TerminalStatus.CLARIFICATION_LIMIT_REACHED:
        lines.append(f"\n{_WIDE}")
        lines.append("STARTARY — CONTEXT CLARIFICATION LIMIT REACHED")
        lines.append(_WIDE)
        lines.append(f"\n{reason}\n")
        lines.append(f"{_DIVIDER}\n")

    elif status == TerminalStatus.EXPERIMENT_DESIGN_NEEDS_REVISION:
        active_ca = (state.get("active_assumption") or {}).get("statement") or state.get("critical_assumption") or "[No critical assumption recorded]"
        critic_out = state.get("experiment_critic_output") or {}
        issues = critic_out.get("issues") or []
        if not issues and state.get("status_message"):
            issues = [state.get("status_message")]
        if not issues:
            issues = ["The proposed experiment design did not meet Startary's quality bar."]

        fallback = state.get("fallback_learning_action")

        lines.append(f"\n{_WIDE}")
        lines.append("STARTARY — EXPERIMENT DESIGN NEEDS REVISION")
        lines.append(_WIDE)
        lines.append("\nWe know what needs to be learned:\n")
        lines.append(f"{active_ca}\n")
        lines.append("The current experiment plan did not meet Startary's quality bar.\n")
        lines.append("Why:")
        for iss in issues:
            lines.append(f"- {iss}")
        lines.append("\nWhat this means:")
        lines.append("The diagnosis is still valid, but Startary has not yet produced a test reliable enough to run.")

        if fallback:
            rendered_fallback = render_fallback_learning_action_text(fallback, mode="compact")
            lines.append("\nHere is the safest next learning action:\n")
            lines.append(rendered_fallback)
            lines.append("\nThis action is for learning, not validation.")

        lines.append(f"{_DIVIDER}\n")

    elif status == TerminalStatus.EXPERIMENT_READY:
        lines.append(f"\n{_WIDE}")
        lines.append("STARTARY — EXPERIMENT READY")
        lines.append(_WIDE)
        lines.append("\nYour proposed experiment is formulated and ready for execution review.\n")
        lines.append(f"{_DIVIDER}\n")

    else:
        lines.append(f"\n{_WIDE}")
        lines.append("STARTARY — INITIAL DIAGNOSIS COMPLETE")
        lines.append(_WIDE)
        lines.append("\nInitial diagnostic evaluation is complete.\n")
        lines.append(f"{_DIVIDER}\n")

    return "\n".join(lines)

