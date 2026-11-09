"""
StartaryGraphState — the central LangGraph state contract.

Milestone 2.5 additions:
  - founder_messages: list of founder conversation turns (source context)
  - clarification_turn_count: integer tracking clarification turns (guard: <= 5)
  - context_sufficiency_output: intermediate LLM output for sufficiency assessment
  - context_sufficiency: serialized ContextSufficiencyAssessment domain model
  - current_clarifying_question: serialized ClarifyingQuestion domain model
  - clarification_history: list of all prior clarification Q&A pairs (for duplicate prevention)
  - is_sufficient: boolean flag for routing
  - status_message: explanation message when stopped or waiting

Milestone 2.6 additions:
  - execution_context: serialized ExecutionContext domain model
  - startup_progress: serialized StartupProgress domain model
  - startup_state_delta: last extracted delta from startup state extractor
  - gap_plan: last InformationGapPlan output (replaces context_sufficiency for routing)
"""

from __future__ import annotations

from typing import Annotated, Any, Optional
from typing_extensions import TypedDict


def merge_understanding_branches(current: Optional[dict[str, Any]], update: Any) -> dict[str, Any]:
    """
    Reducer for the M6.2.7 parallel understanding region: interpret_founder_turn and
    extract_startup_state write their shared-key outputs here concurrently under distinct branch
    names, so neither can overwrite the other. {"__reset__": True, ...} replaces the whole value.
    """
    if not isinstance(update, dict):
        return dict(current or {})
    if update.get("__reset__"):
        return {k: v for k, v in update.items() if k != "__reset__"}
    return {**(current or {}), **update}

from startary.domain.enums import WorkflowPhase


class StartaryGraphState(TypedDict, total=False):
    """
    Working orchestration context passed through the Startary LangGraph.

    Fields with total=False are optional — nodes update only their fields.
    project_id, phase, iteration are always present after initialization.
    """

    # --- Core identity (always present after init) ---
    project_id: str
    phase: WorkflowPhase
    iteration: int

    # --- Raw founder input & Conversation History ---
    raw_context: Optional[str]
    founder_messages: list[dict[str, str]]
    founder_response: Optional[str]
    clarification_turn_count: int               # Maintained for backward compat
    clarification_history: list[dict[str, Any]]

    # --- Milestone 2.6.1: Dedicated Turn Counters & Interrupt Routing ---
    baseline_question_turns: int                # max 5
    strategic_clarification_turns: int          # max 5
    weak_link_clarification_turns: int          # max 3
    active_interrupt_type: Optional[str]        # "baseline" | "strategic" | "weak_link"

    # --- Milestone 2.6.1: Baseline Startup Check-In ---
    baseline_sufficiency: Optional[dict[str, Any]]
    current_baseline_question: Optional[dict[str, Any]]
    baseline_history: list[dict[str, Any]]

    # --- Milestone 2.6.1 / 2.6.2: Weak-Link Discrimination ---
    weak_link_discrimination: Optional[dict[str, Any]]
    current_weak_link_question: Optional[dict[str, Any]]
    weak_link_clarification_history: list[dict[str, Any]]
    selection_confidence: Optional[str]         # "LOW" | "MEDIUM" | "HIGH"
    competing_weak_areas: list[str]
    should_clarify: Optional[bool]              # Authoritative weak-link clarification flag
    should_clarify_weak_link: Optional[bool]     # Backward-compatible alias
    is_strategic_sufficient: Optional[bool]      # Flag indicating strategic understand gate passed

    # --- Current startup understanding ---
    startup_state: dict[str, Any]

    # --- Milestone 2.6: Execution Context & Startup Progress ---
    execution_context: dict[str, Any]         # Serialized ExecutionContext
    startup_progress: dict[str, Any]          # Serialized StartupProgress
    startup_state_delta: dict[str, Any]       # Last extraction delta from extractor

    # --- Milestone 2.6: Information Gap Plan ---
    gap_plan: Optional[dict[str, Any]]        # Serialized InformationGapPlan

    # --- Diagnosis layer (M1) ---
    diagnosis_output: Optional[dict[str, Any]]
    diagnosis: Optional[dict[str, Any]]

    # --- Context Sufficiency & Clarification Gate (M2.5 — kept for backward compat) ---
    context_sufficiency_output: Optional[dict[str, Any]]
    context_sufficiency: Optional[dict[str, Any]]
    current_clarifying_question: Optional[dict[str, Any]]
    is_sufficient: Optional[bool]
    status_message: Optional[str]

    # --- Area Risk Assessment & Weak Link (M2) ---
    area_assessments_output: Optional[dict[str, Any]]
    area_assessments: Optional[list[dict[str, Any]]]
    weak_area: Optional[str]
    weak_link_details: Optional[dict[str, Any]]

    # --- Reality Check (M2) ---
    reality_check_output: Optional[dict[str, Any]]
    reality_check: Optional[dict[str, Any]]

    # --- Critical Assumption (M2) ---
    critical_assumption_output: Optional[dict[str, Any]]
    active_assumption: Optional[dict[str, Any]]

    # --- AI Traces ---
    last_trace: Optional[dict[str, Any]]
    traces: Optional[list[dict[str, Any]]]

    # --- Active experiment (Milestone 3) ---
    proposed_experiment_output: Optional[dict[str, Any]]
    experiment_critic_output: Optional[dict[str, Any]]
    critic_revision_count: int
    critic_revision_instructions: Optional[str]
    # Milestone 6.2.6 (STARTARY_EXPERIMENT_REVISION_V2 only): revision bookkeeping for the current
    # experiment cycle — rejected draft + critique summary, chosen strategy, how the active draft was
    # produced, and which fields a targeted edit changed. Reset with critic_revision_count.
    experiment_revision: Optional[dict[str, Any]]
    can_execute_result: Optional[dict[str, Any]]
    active_experiment: Optional[dict[str, Any]]
    previous_experiments: list[dict[str, Any]]
    fallback_learning_action: Optional[dict[str, Any]]
    learning_input_mode: Optional[str]
    # Milestone 6.2.5: these were written and read but undeclared, so LangGraph silently dropped them.
    # READY / REVISION_REQUIRED / QUALITY_BAR_NOT_MET (ExperimentDesignStatus); read by the terminal renderer.
    experiment_design_status: Optional[str]

    # --- Execution handoff & waiting-for-evidence loop (Milestone 4.0.3; declared in 6.2.5) ---
    active_execution_handoff: Optional[dict[str, Any]]   # composed ExecutionHandoff (compose_execution_handoff)
    execution_handoff_text: Optional[str]                # compact founder-facing handoff rendering
    waiting_response: Optional[str]                      # advisor reply shown at the next wait_for_evidence

    # --- Founder Diagnosis Snapshot (Milestone 3.1) ---
    diagnosis_snapshot: Optional[dict[str, Any]]

    # --- Founder Intent & Conversational Awareness (Milestone 3.2 & 3.2.1) ---
    latest_founder_turn_interpretation: Optional[dict[str, Any]]
    latest_advisor_response: Optional[dict[str, Any]]
    pending_question: Optional[str]
    # Milestone 6.2.3: exact founder-facing prompt, persisted by compose_founder_question BEFORE
    # the interrupt so display, resume, and transcript all use the same text (never regenerated).
    pending_founder_prompt: Optional[dict[str, Any]]
    founder_current_concern: Optional[str]
    cannot_act_now_context: Optional[bool]
    recent_corrections: list[str]

    # --- Conversational Control Integrity (Milestone 3.2.1) ---
    active_next_turn_decision: Optional[dict[str, Any]]
    question_history: list[dict[str, Any]]
    # Milestone 6.2.7 (STARTARY_PARALLEL_UNDERSTANDING only): dispatch decision for the current founder
    # turn (mode PARALLEL|SEQUENTIAL, reason, dispatch time) and the per-branch outputs of the parallel
    # region (traces / telemetry records / last_trace), merged deterministically by join_founder_turn.
    founder_turn_dispatch: Optional[dict[str, Any]]
    understanding_branches: Annotated[dict[str, Any], merge_understanding_branches]
    # Milestone 6.2.3: written by merge_founder_answer and read by telemetry (founder_turn_id);
    # undeclared keys are silently dropped by LangGraph, so attribution was previously lost.
    answered_question_id: Optional[str]
    founder_current_goal: Optional[dict[str, Any]]
    fact_conflicts: list[dict[str, Any]]
    intro_summary_shown: Optional[bool]

    # --- Evidence Admission & Routing (Milestone 4.0.3 & 4.0.7) ---
    evidence_turn_assessment: Optional[dict[str, Any]]
    pending_post_evidence_question: Optional[str]
    post_evidence_answer: Optional[str]

    # --- Evidence (Milestone 4+) ---
    founder_evidence: list[dict[str, Any]]
    external_evidence: list[dict[str, Any]]
    active_evidence: Optional[dict[str, Any]]
    extracted_evidence: Optional[dict[str, Any]]
    evidence_clarification_turns: int

    # --- Analysis result & Learning Loop (Milestone 4+) ---
    evidence_analysis: Optional[dict[str, Any]]
    learning_delta: Optional[dict[str, Any]]
    previous_diagnosis_snapshot: Optional[dict[str, Any]]
    previous_assumptions: list[dict[str, Any]]

    # --- Build Readiness & Experiment Escalation (Milestone 4.1 & 4.1.2) ---
    build_readiness: Optional[dict[str, Any]]
    next_strategic_action: Optional[dict[str, Any]]
    founder_diagnosis_text: Optional[str]
    is_partial_evidence: Optional[bool]

    # --- Routing ---
    candidate_next_action: Optional[str]
    routing_decision: Optional[str]

    # --- Runtime Latency & Reasoning Efficiency (Milestone 4.1.3 & 4.1.3.1) ---
    execution_path: Optional[str]
    is_conversational_repair: Optional[bool]
    strategic_state_changed: Optional[bool]
    strategic_state_revision: Optional[int]    # Milestone M4.1.3.3: Version counter for strategic state mutations
    diagnosis_revision: Optional[int]          # Milestone M4.1.3.3: Revision at which cached diagnosis was created
    turn_telemetry: Optional[dict[str, Any]]
    extraction_status: Optional[str]            # "SUCCESS" | "FAILED" | "BYPASSED"
    extraction_error: Optional[str]



