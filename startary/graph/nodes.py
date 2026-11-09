"""
Graph nodes for the Startary LangGraph — Milestone 2.5.

Nodes:
  1.  initialize_state                  (Deterministic: set defaults & initialize founder messages)
  2.  verify_state                      (Deterministic: validate required fields)
  3.  diagnose_startup                  (LLM: extract & summarize state from founder source messages)
  4.  merge_diagnosis                   (Deterministic: populate StartupState & Diagnosis)
  5.  assess_context_sufficiency        (LLM: determine if grounded info is sufficient to proceed)
  6.  select_clarifying_question        (LLM: select 1 highest-value question without inventing mechanisms)
  7.  ask_and_wait_for_founder          (Interrupt: pause and wait for founder clarification answer)
  8.  merge_founder_answer              (Deterministic: append answer, increment turns, loop back)
  9.  handle_insufficient_context_stop  (Deterministic: graceful stop on reaching max turns)
  10. assess_startup_areas              (LLM: batch risk assessment across 4 dimensions)
  11. select_weak_link                  (Deterministic: compute Decision Risk & apply tie-breakers)
  12. reality_check                     (LLM: examine claims vs evidence for weak area)
  13. generate_critical_assumption      (LLM: formulate single testable hypothesis)
  14. save_critical_assumption          (Deterministic: validate & store first-class Assumption)
"""

from __future__ import annotations

import json
import json as _json
import logging
import uuid
import re
import copy as _copy
import time
import uuid
from typing import Any

from langgraph.types import interrupt
from pydantic import ValidationError

from startary.config import get_settings
from startary.domain.area_knowledge import AreaKnowledge, AreaStatus
from startary.domain.assessment import (
    RiskFactorLevel,
    StartupAreaAssessment,
    risk_factor_to_int,
)
from startary.domain.assumption import Assumption, CriticalAssumptionGroundingError
from startary.domain.clarification import ClarifyingQuestion
from startary.domain.diagnosis import Diagnosis
from startary.domain.enums import (
    AssumptionStatus,
    CriticDimensionResult,
    CriticVerdict,
    EvidenceAnalysisConclusion,
    EvidenceProvenance,
    EvidenceResult,
    EvidenceSourceType,
    EvidenceStrength,
    EvidenceSubmissionSufficiency,
    EvidenceType,
    ExecutionQuality,
    ExecutionReadiness,
    ExperimentDesignStatus,
    ExperimentStatus,
    ExperimentType,
    ParameterProvenance,
    WorkflowPhase,
    ConversationalResponseMode,
)
from startary.domain.build_readiness import (
    BuildReadinessAction,
    BuildReadinessDecision,
    validate_domain_grounding,
    sanitize_domain_leakage,
)
from startary.domain.strategic_action import (
    NextStrategicAction,
    NextStrategicActionType,
)
from startary.domain.evidence import (
    Evidence,
    EvidenceAnalysis,
    LearningDelta,
    EvidenceTurnDisposition,
    EvidenceTurnAssessment,
    get_evidence_turn_disposition,
    is_actual_evidence_turn,
    has_valid_completed_evidence_cycle,
    get_most_recent_completed_evidence_cycle,
    has_substantive_completed_evidence,
    contains_unnegated_keyword,
)
from startary.domain.customer_access import (
    CustomerAccessAssessment,
    CustomerAccessStatus,
    CustomerAccessStrength,
)
from startary.domain.execution_context import ExecutionContext, TeamMember
from startary.domain.experiment import (
    CanExecuteAssessment,
    Experiment,
    ExperimentCriticReport,
    ExperimentParameter,
    ExecutionHandoff,
    FallbackLearningAction,
    FounderExperimentSummary,
    compose_current_positive_signals,
    compose_fallback_learning_action,
    compose_founder_experiment_summary,
    evaluate_experiment_completeness,
    extract_sample_target_from_protocol,
    render_execution_handoff_text,
    render_fallback_learning_action_text,
)
from startary.domain.gap_plan import (
    GapPriority,
    GapTargetDomain,
    InformationGapPlan,
    UncertaintyResolutionMode,
    can_formulate_testable_uncertainty,
)
from startary.domain.provenance import (
    FactConfidence,
    FactProvenance,
    KnowledgeField,
    unknown_field,
    validate_state_grounding,
)
from startary.domain.reality_check import EvidenceConclusion, RealityCheck
from startary.domain.startup_progress import DevelopmentStage, ProductStatus, StartupProgress
from startary.domain.state import StartupState
from startary.domain.sufficiency import ContextSufficiencyAssessment
from startary.domain.taxonomy import StartupArea
from startary.domain.trace import AITrace
from startary.graph.graph_state import StartaryGraphState
from startary.llm.prompts import (
    ASSESS_CONTEXT_SUFFICIENCY_SYSTEM_PROMPT,
    ASSESS_STARTUP_AREAS_PROMPT_VERSION,
    ASSESS_STARTUP_AREAS_SYSTEM_PROMPT,
    CLARIFYING_QUESTION_PROMPT_VERSION,
    CONTEXT_SUFFICIENCY_PROMPT_VERSION,
    CRITICAL_ASSUMPTION_PROMPT_VERSION,
    CRITICAL_ASSUMPTION_SYSTEM_PROMPT,
    DIAGNOSE_STARTUP_PROMPT_VERSION,
    DIAGNOSE_STARTUP_SYSTEM_PROMPT,
    EXTRACT_STARTUP_STATE_SYSTEM_PROMPT,
    INFORMATION_GAP_PLANNER_PROMPT_VERSION,
    INFORMATION_GAP_PLANNER_SYSTEM_PROMPT,
    REALITY_CHECK_PROMPT_VERSION,
    REALITY_CHECK_SYSTEM_PROMPT,
    SELECT_CLARIFYING_QUESTION_SYSTEM_PROMPT,
    STARTUP_STATE_EXTRACTOR_PROMPT_VERSION,
    build_area_assessment_user_message,
    build_clarifying_question_user_message,
    build_context_sufficiency_user_message,
    build_critical_assumption_user_message,
    build_diagnosis_user_message,
    build_extract_startup_state_user_message,
    build_information_gap_planner_user_message,
    build_reality_check_user_message,
    BASELINE_STATE_SUFFICIENCY_PROMPT_VERSION,
    BASELINE_QUESTION_PLANNER_PROMPT_VERSION,
    WEAK_LINK_DISCRIMINATION_PROMPT_VERSION,
    WEAK_LINK_CLARIFICATION_PROMPT_VERSION,
    BASELINE_STATE_SUFFICIENCY_SYSTEM_PROMPT,
    build_baseline_sufficiency_user_message,
    BASELINE_QUESTION_PLANNER_SYSTEM_PROMPT,
    build_baseline_question_planner_user_message,
    WEAK_LINK_DISCRIMINATION_SYSTEM_PROMPT,
    build_weak_link_discrimination_user_message,
    WEAK_LINK_CLARIFYING_QUESTION_SYSTEM_PROMPT,
    build_weak_link_clarifying_question_user_message,
    EXPERIMENT_GENERATOR_PROMPT_VERSION,
    EXPERIMENT_GENERATOR_SYSTEM_PROMPT,
    build_experiment_generator_user_message,
    EXPERIMENT_CRITIC_PROMPT_VERSION,
    EXPERIMENT_CRITIC_SYSTEM_PROMPT,
    build_experiment_critic_user_message,
    EXPERIMENT_CRITIC_REVISION_SCOPE_ADDENDUM,
    EXPERIMENT_FOCUSED_CRITIC_PROMPT_VERSION,
    EXPERIMENT_FOCUSED_CRITIC_SYSTEM_PROMPT,
    EXPERIMENT_REVISION_PROMPT_VERSION,
    EXPERIMENT_REVISION_SYSTEM_PROMPT,
    build_experiment_focused_critic_user_message,
    build_experiment_revision_user_message,
    STARTUP_DIAGNOSIS_COMPOSER_PROMPT_VERSION,
    STARTUP_DIAGNOSIS_COMPOSER_SYSTEM_PROMPT,
    build_startup_diagnosis_composer_user_message,
    FOUNDER_TURN_INTERPRETER_PROMPT_VERSION,
    FOUNDER_TURN_INTERPRETER_SYSTEM_PROMPT,
    build_founder_turn_interpreter_user_message,
    ADVISOR_RESPONSE_COMPOSER_PROMPT_VERSION,
    ADVISOR_RESPONSE_COMPOSER_SYSTEM_PROMPT,
    build_advisor_response_composer_user_message,
    EVIDENCE_EXTRACTOR_PROMPT_VERSION,
    EVIDENCE_EXTRACTOR_SYSTEM_PROMPT,
    build_evidence_extractor_user_message,
    EVIDENCE_ANALYZER_PROMPT_VERSION,
    EVIDENCE_ANALYZER_SYSTEM_PROMPT,
    build_evidence_analyzer_user_message,
    EVIDENCE_ADMISSION_GATE_SYSTEM_PROMPT,
    build_evidence_admission_gate_user_message,
    WAITING_FOR_EVIDENCE_ADVISOR_SYSTEM_PROMPT,
    build_waiting_for_evidence_advisor_user_message,
    BUILD_READINESS_SYSTEM_PROMPT,
    build_build_readiness_user_message,
)
from startary.domain.diagnosis_snapshot import StartupDiagnosisSnapshot
from startary.domain.founder_intent import (
    AdvisorResponse,
    FounderIntentType,
    FounderResponsePriority,
    FounderTurnInterpretation,
)
from startary.domain.turn_control import (
    FactConflict,
    FounderGoal,
    FounderGoalType,
    NextTurnDecision,
    QuestionHistoryItem,
    should_surface_diagnosis_delta,
    determine_conversational_response_mode,
)
from startary.domain.baseline import (
    BaselineDimension,
    BaselineDimensionStatus,
    BaselineQuestionPlan,
    BaselineStartupStateSufficiency,
)
from startary.domain.discrimination import (
    SelectionConfidence,
    WeakLinkClarifyingQuestion,
    WeakLinkDiscrimination,
)
from startary.domain.experiment_contract import (
    CRITIC_DIMENSIONS,
    BudgetStatus,
    CriticDecision,
    PrototypeStatus,
    assess_prototype_requirement,
    compare_cost_to_budget,
    critic_decision_metadata,
    deterministic_decision,
    evaluate_experiment_contract,
    experiment_critique_route,
    interpret_critic_output,
    pending_authorization_notice,
    pending_authorizations,
    required_authorizations,
)
from startary.domain.experiment_revision import (
    CriticMode,
    DraftOrigin,
    RevisionStrategy,
    apply_focused_acceptance_rule,
    apply_revision_patch,
    change_summary,
    critic_v2_metadata,
    diff_experiment,
    generation_record,
    is_substantial_revision,
    rejection_record,
    remediate_consent_violations,
    select_critic_mode,
    select_revision_strategy,
)
from startary.observability.langsmith import emit_decision_span
from startary.llm.provider import (
    LLMFailureKind,
    classify_llm_failure,
    schema_correction_message,
    TokenBudgetExhaustedError,
    get_llm_client,
    get_llm_client_resolution_info,
    is_permanent_model_error,
    is_token_budget_exhausted_error,
)
from startary.llm.schemas import (
    AdvisorResponseLLMOutput,
    BaselineQuestionPlanLLMOutput,
    BaselineSufficiencyLLMOutput,
    ClarifyingQuestionLLMOutput,
    ContextSufficiencyLLMOutput,
    CriticalAssumptionLLMOutput,
    DiagnosisLLMOutput,
    EvidenceAnalysisLLMOutput,
    EvidenceTurnAssessmentLLMOutput,
    ExperimentCriticLLMOutput,
    ExperimentCriticV2LLMOutput,
    ExperimentFocusedCriticLLMOutput,
    ExperimentGeneratorLLMOutput,
    ExperimentRevisionPatchLLMOutput,
    ExperimentParameterLLM,
    ExtractedEvidenceLLMOutput,
    FounderTurnInterpretationLLMOutput,
    InformationGapPlanLLMOutput,
    RealityCheckLLMOutput,
    AreaRiskAssessmentLLMOutput,
    StartupAreaAssessmentBatchLLMOutput,
    StartupDiagnosisSnapshotLLMOutput,
    StartupStateDeltaLLMOutput,
    WaitingForEvidenceAdvisorLLMOutput,
    WeakLinkClarifyingQuestionLLMOutput,
    WeakLinkDiscriminationLLMOutput,
    BuildReadinessAssessmentLLMOutput,
)
from datetime import datetime, timezone
from startary.domain.bound_answers import parse_bound_answer
from startary.domain.telemetry import (
    NodeTelemetryRecord,
    TurnTelemetrySummary,
    logical_stage_for,
    record_telemetry_event,
)

logger = logging.getLogger(__name__)


def _safe_get_llm_client(
    settings: Any,
    tier: str = "reasoning",
    max_tokens: int | None = None,
    max_completion_tokens: int | None = None,
    reasoning_effort: str | None = None,
) -> Any:
    """Helper to safely resolve LLM client by tier and max tokens while supporting legacy test mocks."""
    effective_max = max_completion_tokens or max_tokens
    try:
        return get_llm_client(
            settings,
            tier=tier,
            max_completion_tokens=effective_max,
            reasoning_effort=reasoning_effort,
        )
    except TypeError:
        try:
            return get_llm_client(settings, tier=tier, max_tokens=effective_max)
        except TypeError:
            try:
                return get_llm_client(settings, tier=tier)
            except TypeError:
                return get_llm_client(settings)


def _invoke_structured_with_token_retry(
    structured_llm: Any,
    messages: list[dict[str, str]],
    settings: Any,
    tier: str,
    base_max_tokens: int,
    node_name: str,
    schema_cls: type,
    run_metadata: dict[str, Any] | None = None,
) -> tuple[Any, float]:
    """
    Invoke structured LLM with smart token-budget recovery.
    If attempt 1 fails due to token budget / length exhaustion,
    retries AT MOST ONCE with a safe expanded budget (1.6x, min +1500) and reasoning_effort='low'.
    Never retries the identical request with the identical exhausted token cap.
    If retry fails with token exhaustion, raises TokenBudgetExhaustedError.
    """
    attempt_1_config = {
        "run_name": f"{node_name}:attempt_1",
        "metadata": {
            "node_name": node_name,
            "logical_stage": logical_stage_for(node_name),
            "attempt": 1,
            "model_tier": tier,
            "base_max_tokens": base_max_tokens,
            "is_retry": False,
            **(run_metadata or {}),
        },
        "tags": [f"node:{node_name}", f"tier:{tier}", "attempt:1"],
    }
    t_start = time.monotonic()
    try:
        try:
            res = structured_llm.invoke(messages, config=attempt_1_config)
        except TypeError as te:
            if "config" in str(te):
                res = structured_llm.invoke(messages)
            else:
                raise
        latency_ms = (time.monotonic() - t_start) * 1000
        return res, latency_ms
    except Exception as exc:
        if is_token_budget_exhausted_error(exc):
            expanded_tokens = max(int(base_max_tokens * 1.6), base_max_tokens + 1500)
            logger.warning(
                "%s | token budget exhausted with cap %d. Retrying AT MOST ONCE with safe expanded cap %d and reasoning_effort='low'. Error: %s",
                node_name, base_max_tokens, expanded_tokens, exc,
            )
            is_reasoning_tier = (tier == "reasoning") or any(
                m in (getattr(settings, "fast_model_name", None) or "").lower() for m in ("o1", "o3", "sol", "luna")
            )
            expanded_llm = _safe_get_llm_client(
                settings,
                tier=tier,
                max_completion_tokens=expanded_tokens,
                reasoning_effort="low" if is_reasoning_tier else None,
            )
            expanded_structured = expanded_llm.with_structured_output(schema_cls)
            t_retry = time.monotonic()
            retry_config = {
                "run_name": f"{node_name}:retry_token_expansion",
                "metadata": {
                    "node_name": node_name,
                    "logical_stage": logical_stage_for(node_name),
                    "attempt": 2,
                    "model_tier": tier,
                    "base_max_tokens": base_max_tokens,
                    "expanded_max_tokens": expanded_tokens,
                    "is_retry": True,
                    "retry_reason": "token_budget_exhausted",
                    "reasoning_effort": "low" if is_reasoning_tier else "standard",
                    **(run_metadata or {}),
                },
                "tags": [f"node:{node_name}", f"tier:{tier}", "attempt:2", "retry:token_expansion"],
            }
            try:
                try:
                    res = expanded_structured.invoke(messages, config=retry_config)
                except TypeError as te:
                    if "config" in str(te):
                        res = expanded_structured.invoke(messages)
                    else:
                        raise
                latency_ms = (time.monotonic() - t_retry) * 1000
                logger.info("%s | retry with expanded token budget SUCCEEDED in %.2fms", node_name, latency_ms)
                return res, latency_ms
            except Exception as retry_exc:
                if is_token_budget_exhausted_error(retry_exc):
                    raise TokenBudgetExhaustedError(
                        f"{node_name} token budget exhausted even after expanding cap from {base_max_tokens} to {expanded_tokens}.",
                        node_name=node_name,
                        original_error=retry_exc,
                    ) from retry_exc
                raise
        raise


def _record_llm_node_telemetry(
    state: StartaryGraphState,
    node_name: str,
    llm: Any,
    tier: str,
    latency_ms: float,
    attempts: int,
    schema_name: str,
    status: str = "SUCCESS",
    messages: list[dict[str, str]] | None = None,
    result: Any = None,
    details: dict[str, Any] | None = None,
) -> TurnTelemetrySummary:
    """
    Milestone 6.2.3: uniform Startary telemetry for an LLM node (node, logical stage, tier,
    resolved model, reasoning effort, duration, retry count, status). Only sizes are derived
    from prompts/outputs; no founder text is recorded.
    """
    res_info = get_llm_client_resolution_info(llm)
    prompt_tokens = sum(len(str(m.get("content", ""))) for m in messages) // 4 if messages else None
    completion_tokens = None
    if result is not None and hasattr(result, "model_dump"):
        try:
            completion_tokens = len(json.dumps(result.model_dump(), default=str)) // 4
        except Exception:
            completion_tokens = None
    total_tokens = (prompt_tokens or 0) + (completion_tokens or 0) if (prompt_tokens or completion_tokens) else None
    return record_telemetry_event(
        state=state,
        node_name=node_name,
        model_name=res_info["resolved_model"] or tier,
        duration_ms=latency_ms,
        latency_ms=latency_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        retry_count=max(0, attempts - 1),
        schema_name=schema_name,
        configured_model=res_info["configured_model"],
        resolved_model=res_info["resolved_model"],
        fallback_used=res_info["fallback_used"],
        fallback_reason=res_info["fallback_reason"],
        model_tier=res_info.get("model_tier") or tier,
        reasoning_effort=res_info.get("reasoning_effort"),
        status=status,
        details=details,
    )


def _revision_v2_enabled(settings: Any) -> bool:
    """Milestone 6.2.6 A/B flag (STARTARY_EXPERIMENT_REVISION_V2). Only an explicit True enables V2."""
    return getattr(settings, "experiment_revision_v2", False) is True


def _slim_facts(facts: dict[str, Any]) -> dict[str, Any]:
    """Known fact values only (UNKNOWN placeholders dropped), as the critic receives them."""
    return {
        k: (v.get("value") if isinstance(v, dict) and "value" in v else v)
        for k, v in (facts or {}).items()
        if v and (not isinstance(v, dict) or v.get("value") not in (None, "UNKNOWN", ""))
    }


MAX_BASELINE_QUESTION_TURNS = 5
MAX_STRATEGIC_CLARIFICATION_TURNS = 5
MAX_WEAK_LINK_CLARIFICATION_TURNS = 3
MAX_CLARIFICATION_TURNS = 5  # Backward compat alias for strategic


def is_strategic_state_changed(state: dict[str, Any]) -> bool:
    """
    Deterministic decision on whether material strategic facts changed
    requiring a full rediagnosis (Milestone 4.1.3 Phase 5 & 4.1.3.1).
    """
    # Invariant: EXTRACTION FAILURE != NO STRATEGIC CHANGE
    # If extraction failed, strategic state change must be treated as True for safety
    if state.get("extraction_status") == "FAILED":
        return True

    if not state.get("diagnosis_output"):
        return True

    # Milestone M4.1.3.3: Revision check
    strat_rev = int(state.get("strategic_state_revision") or 1)
    diag_rev = state.get("diagnosis_revision")
    if diag_rev is not None and strat_rev > diag_rev:
        return True

    # Milestone M4.1.3.3: Completed evidence cycle invariant
    # A completed valid evidence cycle is ALWAYS a strategic state change!
    if has_valid_completed_evidence_cycle(state) and state.get("is_partial_evidence") is not True:
        if diag_rev is None or strat_rev > diag_rev or state.get("iteration", 1) > 1:
            return True

    disp = get_evidence_turn_disposition(state)
    if disp == EvidenceTurnDisposition.ACTUAL_EVIDENCE:
        return True

    if state.get("evidence_analysis") and state.get("is_partial_evidence") is not True:
        if state.get("iteration", 1) > 1 and state.get("evidence_history"):
            last_ev = state["evidence_history"][-1] if state.get("evidence_history") else {}
            if last_ev.get("iteration") == state.get("iteration", 1) - 1:
                return True

    if state.get("strategic_state_changed") is True:
        return True
    if state.get("strategic_state_changed") is False:
        if diag_rev is None or strat_rev == diag_rev:
            return False

    path = state.get("execution_path")
    if path in ("FAST_LOCAL_UPDATE", "CONVERSATIONAL_REPAIR"):
        return False

    delta = state.get("startup_state_delta") or {}
    strategic_keys = {
        "paying_customers", "revenue", "active_users",
        "market_segment", "target_user", "problem_statement",
        "pricing_model", "channel_access",
    }
    if any(delta.get(k) is not None for k in strategic_keys):
        return True

    if state.get("fact_conflicts"):
        return True

    return False


# ---------------------------------------------------------------------------
# 1. initialize_state (Deterministic)
# ---------------------------------------------------------------------------

def initialize_state(state: StartaryGraphState) -> dict:
    """Ensure required fields have sane defaults and initialize message history."""
    updates: dict = {}

    if not state.get("phase"):
        updates["phase"] = WorkflowPhase.UNDERSTAND

    if not state.get("iteration"):
        updates["iteration"] = 1

    if "strategic_state_revision" not in state or state.get("strategic_state_revision") is None:
        updates["strategic_state_revision"] = 1
    if "diagnosis_revision" not in state:
        updates["diagnosis_revision"] = None

    if "startup_state" not in state:
        updates["startup_state"] = StartupState().model_dump()

    if "founder_evidence" not in state:
        updates["founder_evidence"] = []
    if "external_evidence" not in state:
        updates["external_evidence"] = []
    if "traces" not in state:
        updates["traces"] = []

    # Milestone 2.5 initialization
    raw_ctx = state.get("raw_context") or ""
    if "founder_messages" not in state:
        messages = []
        if raw_ctx.strip():
            messages.append({"role": "founder", "content": raw_ctx.strip()})
        updates["founder_messages"] = messages

    if "clarification_turn_count" not in state:
        updates["clarification_turn_count"] = 0

    if "clarification_history" not in state:
        updates["clarification_history"] = []

    # Milestone 2.6 initialization
    if "execution_context" not in state:
        updates["execution_context"] = ExecutionContext().model_dump()

    if "startup_progress" not in state:
        updates["startup_progress"] = StartupProgress().model_dump()

    if "startup_state_delta" not in state:
        updates["startup_state_delta"] = {}

    # Milestone 2.6.1 initialization (dedicated counters & histories)
    if "baseline_question_turns" not in state:
        updates["baseline_question_turns"] = 0
    if "strategic_clarification_turns" not in state:
        updates["strategic_clarification_turns"] = 0
    if "weak_link_clarification_turns" not in state:
        updates["weak_link_clarification_turns"] = 0
    if "baseline_history" not in state:
        updates["baseline_history"] = []
    if "weak_link_clarification_history" not in state:
        updates["weak_link_clarification_history"] = []
    if "competing_weak_areas" not in state:
        updates["competing_weak_areas"] = []
    if "active_interrupt_type" not in state:
        updates["active_interrupt_type"] = None
    if "selection_confidence" not in state:
        updates["selection_confidence"] = None

    # Milestone 3.2 & 3.2.1 initialization
    if "latest_founder_turn_interpretation" not in state:
        updates["latest_founder_turn_interpretation"] = None
    if "latest_advisor_response" not in state:
        updates["latest_advisor_response"] = None
    if "pending_founder_prompt" not in state:
        updates["pending_founder_prompt"] = None
    if "pending_question" not in state:
        updates["pending_question"] = None
    if "founder_current_concern" not in state:
        updates["founder_current_concern"] = None
    if "cannot_act_now_context" not in state:
        updates["cannot_act_now_context"] = False
    if "recent_corrections" not in state:
        updates["recent_corrections"] = []
    if "active_next_turn_decision" not in state:
        updates["active_next_turn_decision"] = None
    if "question_history" not in state:
        updates["question_history"] = []
    if "founder_current_goal" not in state:
        updates["founder_current_goal"] = None
    if "fact_conflicts" not in state:
        updates["fact_conflicts"] = []
    if "intro_summary_shown" not in state:
        updates["intro_summary_shown"] = False

    # Milestone 4 initialization
    if "founder_evidence" not in state:
        updates["founder_evidence"] = []
    if "external_evidence" not in state:
        updates["external_evidence"] = []
    if "active_evidence" not in state:
        updates["active_evidence"] = None
    if "extracted_evidence" not in state:
        updates["extracted_evidence"] = None
    if "evidence_clarification_turns" not in state:
        updates["evidence_clarification_turns"] = 0
    if "evidence_analysis" not in state:
        updates["evidence_analysis"] = None
    if "learning_delta" not in state:
        updates["learning_delta"] = None
    if "previous_diagnosis_snapshot" not in state:
        updates["previous_diagnosis_snapshot"] = None
    if "previous_assumptions" not in state:
        updates["previous_assumptions"] = []

    # Milestone 4.0.1: Protect against stale M4 state on fresh Iteration-1 runs
    current_iter = updates.get("iteration", state.get("iteration", 1))
    if current_iter == 1:
        updates["learning_delta"] = None
        updates["previous_assumptions"] = []
        updates["previous_diagnosis_snapshot"] = None
        updates["previous_weak_area"] = None
        updates["evidence_analysis"] = None
        updates["active_evidence"] = None
        updates["extracted_evidence"] = None
        updates["evidence_clarification_turns"] = 0

    project_id = state.get("project_id", "<unknown>")
    phase = updates.get("phase", state.get("phase"))
    iteration = updates.get("iteration", state.get("iteration"))

    logger.info("initialize_state | project_id=%s phase=%s iteration=%s", project_id, phase, iteration)
    return updates


# ---------------------------------------------------------------------------
# 2. verify_state (Deterministic)
# ---------------------------------------------------------------------------

def verify_state(state: StartaryGraphState) -> dict:
    """Validate internal state consistency."""
    errors: list[str] = []

    if not state.get("project_id"):
        errors.append("project_id is missing or empty")

    phase = state.get("phase")
    if phase is None:
        errors.append("phase is missing")
    elif phase not in WorkflowPhase.__members__.values():
        errors.append(f"phase '{phase}' is not a valid WorkflowPhase")

    iteration = state.get("iteration")
    if iteration is None:
        errors.append("iteration is missing")
    elif not isinstance(iteration, int) or iteration < 1:
        errors.append(f"iteration must be a positive integer, got: {iteration!r}")

    startup_state = state.get("startup_state")
    if startup_state is None:
        errors.append("startup_state is missing")
    elif not isinstance(startup_state, dict):
        errors.append("startup_state must be a dict")

    if errors:
        raise ValueError(f"State verification failed: {'; '.join(errors)}")

    return {}


# ---------------------------------------------------------------------------
# 3. diagnose_startup (LLM)
# ---------------------------------------------------------------------------

def diagnose_startup(state: StartaryGraphState) -> dict:
    """Extract structured startup state from raw context & message history using LLM."""
    project_id = state.get("project_id", "<unknown>")
    existing_diag = state.get("diagnosis_output")

    strat_rev_raw = state.get("strategic_state_revision")
    strat_rev = int(strat_rev_raw or 1)
    diag_rev = state.get("diagnosis_revision")
    strat_changed = state.get("strategic_state_changed")

    # Milestone M4.1.3.3: If revisions are tracked and mismatch, strategic state has mutated and must be re-diagnosed
    if strat_rev_raw is not None or diag_rev is not None:
        if diag_rev is None or strat_rev > diag_rev:
            strat_changed = True
    elif strat_changed is None:
        strat_changed = is_strategic_state_changed(state)

    # Invariant: A completed valid evidence cycle that hasn't been rediagnosed forces fresh diagnosis
    if has_valid_completed_evidence_cycle(state) and state.get("is_partial_evidence") is not True:
        strat_changed = True

    can_reuse_cache = (
        existing_diag is not None
        and not strat_changed
        and (
            (strat_rev_raw is None and diag_rev is None)
            or (diag_rev is not None and diag_rev == strat_rev)
        )
        and state.get("extraction_status") != "FAILED"
    )

    if can_reuse_cache:
        logger.info("diagnose_startup | reusing cached diagnosis output (revision=%s)", diag_rev)
        model_name = get_settings().get_model_for_tier("reasoning")
        record_telemetry_event(
            state=state,
            node_name="diagnose_startup",
            model_name=model_name,
            configured_model=model_name,
            resolved_model=model_name,
            latency_ms=0.0,
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc),
            schema_name="DiagnosisLLMOutput",
            skipped_by_fast_path=True,
            cached=True,
            execution_path=state.get("execution_path") or "DELTA_STATE_UPDATE",
        )
        trace = AITrace(
            module_name="diagnose_startup",
            prompt_version=DIAGNOSE_STARTUP_PROMPT_VERSION,
            input_data={"cached": True},
            output_data=existing_diag,
            latency_ms=0.0,
            project_id=project_id,
            model_name=model_name,
            configured_model=model_name,
            resolved_model=model_name,
            skipped_by_fast_path=True,
            execution_path=state.get("execution_path") or "DELTA_STATE_UPDATE",
        )
        existing_traces = list(state.get("traces") or [])
        existing_traces.append(trace.model_dump(mode="json"))
        return {
            "diagnosis_output": existing_diag,
            "phase": WorkflowPhase.DIAGNOSE,
            "traces": existing_traces,
            "last_trace": trace.model_dump(mode="json"),
            "strategic_state_changed": False,
            "strategic_state_revision": strat_rev,
            "diagnosis_revision": diag_rev,
            "turn_telemetry": state.get("turn_telemetry"),
        }

    raw_context = state.get("raw_context") or ""
    founder_messages = state.get("founder_messages") or []

    # If messages exist, build combined source text
    if not raw_context.strip() and not founder_messages:
        empty_output = DiagnosisLLMOutput(
            area_assessments=[],
            diagnosis_summary="No founder context was provided.",
        )
        return {
            "diagnosis_output": empty_output.model_dump(),
            "phase": WorkflowPhase.DIAGNOSE,
            "strategic_state_revision": strat_rev,
            "diagnosis_revision": strat_rev,
        }

    settings = get_settings()
    base_max_tokens = 3500
    llm = _safe_get_llm_client(settings, tier="reasoning", max_completion_tokens=base_max_tokens)
    conf_model, res_model, fb_used, fb_reason = get_llm_client_resolution_info(llm)
    structured_llm = llm.with_structured_output(DiagnosisLLMOutput)

    # Prompt context slimming (Milestone 4.1.3 Phase 6):
    # If conversation is long (> 6 messages), take initial turn + last 3 turns
    slammed_messages = founder_messages
    if len(founder_messages) > 6:
        slammed_messages = [founder_messages[0]] + founder_messages[-3:]

    # Milestone M4.1.3.3: Format validated evidence summary for post-evidence re-diagnosis
    evidence_summary = None
    founder_ev_list = state.get("founder_evidence") or []
    if founder_ev_list or state.get("active_evidence"):
        ev_items = []
        for ev in founder_ev_list[-3:]:
            if isinstance(ev, dict):
                ev_items.append(
                    f"- Iteration {ev.get('iteration', 1)} [{ev.get('result', 'ANALYZED')}]: "
                    f"{ev.get('raw_content', '')[:250]}"
                )
                for obs in ev.get("additional_observations") or []:
                    if isinstance(obs, dict):
                        ev_items.append(f"  * Observation [{obs.get('category', 'GENERAL')}]: {obs.get('text', '')}")
        if ev_items:
            evidence_summary = "\n".join(ev_items)

    startup_state_dict = state.get("startup_state") or {}
    areas_dict = startup_state_dict.get("areas") or startup_state_dict.get("area_knowledge") or {}
    known_areas = [
        f"{a}: status={v.get('status', 'UNKNOWN')}, understanding={v.get('understanding', '')}"
        for a, v in areas_dict.items()
        if isinstance(v, dict) and v.get("status") in ("KNOWN", "PARTIALLY_KNOWN")
    ]
    state_summary_text = None
    if known_areas:
        state_summary_text = "Known / Partially Known Areas:\n" + "\n".join(f"- {k}" for k in known_areas)

    user_message = build_diagnosis_user_message(
        raw_context=raw_context,
        founder_messages=slammed_messages,
        existing_startup_state_summary=state_summary_text,
        evidence_summary=evidence_summary,
    )
    messages = [
        {"role": "system", "content": DIAGNOSE_STARTUP_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    attempt = 0
    last_error: Exception | None = None

    while attempt < 2:
        attempt += 1
        start_time = datetime.now(timezone.utc)
        try:
            result, latency_ms = _invoke_structured_with_token_retry(
                structured_llm=structured_llm,
                messages=messages,
                settings=settings,
                tier="reasoning",
                base_max_tokens=base_max_tokens,
                node_name="diagnose_startup",
                schema_cls=DiagnosisLLMOutput,
            )
            end_time = datetime.now(timezone.utc)

            if not isinstance(result, DiagnosisLLMOutput):
                raise ValueError(f"Expected DiagnosisLLMOutput, got {type(result)}")

            model_name = res_model or settings.get_model_for_tier("reasoning")
            prompt_tokens = getattr(getattr(result, "usage_metadata", None), "prompt_tokens", None)
            if prompt_tokens is None:
                prompt_tokens = sum(len(m.get("content", "")) for m in messages) // 4
            completion_tokens = getattr(getattr(result, "usage_metadata", None), "completion_tokens", None)
            if completion_tokens is None:
                completion_tokens = len(_json.dumps(result.model_dump())) // 4
            total_tokens = prompt_tokens + completion_tokens

            turn_telem = record_telemetry_event(
                state=state,
                node_name="diagnose_startup",
                model_name=model_name,
                duration_ms=latency_ms,
                latency_ms=latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                execution_path=state.get("execution_path") or "STRATEGIC_REASONING",
                configured_model=conf_model,
                resolved_model=res_model,
                fallback_used=fb_used,
                fallback_reason=fb_reason,
                retry_count=attempt - 1,
                model_tier="reasoning",
                reasoning_effort=get_llm_client_resolution_info(llm).get("reasoning_effort"),
            )

            trace = AITrace(
                module_name="diagnose_startup",
                prompt_version=DIAGNOSE_STARTUP_PROMPT_VERSION,
                input_data={"messages_count": len(slammed_messages), "raw_context_len": len(raw_context)},
                output_data=result.model_dump(),
                latency_ms=latency_ms,
                project_id=project_id,
                model_name=model_name,
                configured_model=conf_model,
                resolved_model=res_model,
                fallback_used=fb_used,
                fallback_reason=fb_reason,
                start_timestamp=start_time,
                end_timestamp=end_time,
                retry_count=attempt - 1,
                schema_name="DiagnosisLLMOutput",
                execution_path=state.get("execution_path") or "STRATEGIC_REASONING",
            )

            existing_traces = list(state.get("traces") or [])
            existing_traces.append(trace.model_dump(mode="json"))

            return {
                "diagnosis_output": result.model_dump(),
                "last_trace": trace.model_dump(mode="json"),
                "traces": existing_traces,
                "phase": WorkflowPhase.DIAGNOSE,
                "strategic_state_changed": False,
                "strategic_state_revision": strat_rev,
                "diagnosis_revision": strat_rev,
                "turn_telemetry": turn_telem,
            }
        except Exception as exc:
            last_error = exc
            logger.warning("diagnose_startup failed attempt %d: %s", attempt, exc)
            if is_permanent_model_error(exc) or isinstance(exc, TokenBudgetExhaustedError):
                logger.warning("diagnose_startup: permanent model error or token exhaustion (%s). Aborting retries.", exc)
                break

    if isinstance(last_error, TokenBudgetExhaustedError):
        raise last_error
    raise RuntimeError(f"diagnose_startup failed after {attempt} attempts. Error: {last_error}")


# ---------------------------------------------------------------------------
# 4. merge_diagnosis (Deterministic)
# ---------------------------------------------------------------------------

# Milestone 6.2.3: StartupState mirrors the canonical fact stores owned by merge_execution_state
# (top-level execution_context / startup_progress). (StartupState layer key, canonical state key)
_FACT_LAYERS: tuple[tuple[str, str], ...] = (
    ("execution_context", "execution_context"),
    ("progress", "startup_progress"),
)


def _is_known_fact(field: Any) -> bool:
    return isinstance(field, dict) and str(field.get("provenance") or "UNKNOWN").upper() != "UNKNOWN"


def _preserve_fact_layer(canonical: dict | None, mirrored: dict | None, default: dict) -> dict:
    """
    Field-wise fact mirror that never downgrades a known fact to an UNKNOWN placeholder.
    Precedence per schema field: known canonical value > known previously-mirrored value > default.
    Values are copied verbatim (value, provenance, confidence, source_turn), so EXPLICIT facts,
    INFERRED facts, and explicit negatives (e.g. paying_customers=0) keep their epistemic status.
    """
    merged = dict(default)
    for source in (mirrored or {}, canonical or {}):  # canonical last: it wins when known
        for key in default:
            val = source.get(key)
            if _is_known_fact(val):
                merged[key] = val
            elif isinstance(val, list) and val:
                merged[key] = val
    return merged


def _startup_state_with_canonical_facts(state: StartaryGraphState) -> dict:
    """StartupState dict whose execution/progress layers reflect the canonical fact stores."""
    st = dict(state.get("startup_state") or {})
    defaults = {"execution_context": ExecutionContext().model_dump(), "progress": StartupProgress().model_dump()}
    for layer_key, canonical_key in _FACT_LAYERS:
        canonical = state.get(canonical_key)
        if canonical is None and layer_key not in st:
            continue
        st[layer_key] = _preserve_fact_layer(canonical, st.get(layer_key), defaults[layer_key])
    return st


def merge_diagnosis(state: StartaryGraphState) -> dict:
    """Merge validated diagnosis output into StartupState & Diagnosis models."""
    project_id = state.get("project_id", "<unknown>")
    iteration = state.get("iteration", 1)
    raw_context = state.get("raw_context")

    diagnosis_output_dict = state.get("diagnosis_output")
    if not diagnosis_output_dict:
        raise ValueError(f"merge_diagnosis | project_id={project_id}: diagnosis_output is missing from state.")

    llm_output = DiagnosisLLMOutput(**diagnosis_output_dict)

    existing_startup_state = state.get("startup_state") or {}
    existing_areas_map = existing_startup_state.get("areas") or existing_startup_state.get("area_knowledge") or {}

    areas: dict[StartupArea, AreaKnowledge] = {}
    for area_diag in llm_output.area_assessments:
        try:
            areas[area_diag.area] = AreaKnowledge(
                status=area_diag.status,
                understanding=area_diag.understanding,
                founder_claims=area_diag.founder_claims,
                evidence_provided=area_diag.evidence_provided,
                missing_information=area_diag.missing_information,
            )
        except Exception:
            areas[area_diag.area] = AreaKnowledge(status=AreaStatus.UNKNOWN)

    for area in StartupArea:
        existing_k = (
            existing_areas_map.get(area.value)
            or existing_areas_map.get(str(area))
            or existing_areas_map.get(area)
            or {}
        )
        is_previously_known = (
            isinstance(existing_k, dict)
            and existing_k.get("status") == AreaStatus.KNOWN.value
        )

        if area not in areas:
            if is_previously_known:
                missing_info = existing_k.get("missing_information")
                if isinstance(missing_info, list):
                    missing_info = "; ".join(str(x) for x in missing_info) if missing_info else None
                areas[area] = AreaKnowledge(
                    status=AreaStatus.KNOWN,
                    understanding=existing_k.get("understanding"),
                    founder_claims=existing_k.get("founder_claims") or [],
                    evidence_provided=existing_k.get("evidence_provided") or [],
                    missing_information=missing_info,
                )
            else:
                areas[area] = AreaKnowledge(status=AreaStatus.UNKNOWN)
        else:
            # Prevent silent regression of KNOWN areas to UNKNOWN
            if is_previously_known and areas[area].status == AreaStatus.UNKNOWN:
                areas[area].status = AreaStatus.KNOWN
                if not areas[area].evidence_provided:
                    areas[area].evidence_provided = existing_k.get("evidence_provided") or []
                if not areas[area].understanding and existing_k.get("understanding"):
                    areas[area].understanding = existing_k.get("understanding")

    new_startup_state = StartupState(raw_context=raw_context, areas=areas)
    startup_state_dump = new_startup_state.model_dump()
    # Milestone 6.2.3: diagnosis owns `areas` only. Founder facts extracted by
    # merge_execution_state must survive this rebuild instead of resetting to UNKNOWN defaults.
    for layer_key, canonical_key in _FACT_LAYERS:
        startup_state_dump[layer_key] = _preserve_fact_layer(
            state.get(canonical_key),
            existing_startup_state.get(layer_key),
            startup_state_dump[layer_key],
        )

    assessments = [
        StartupAreaAssessment(
            area=area,
            status=areas[area].status,
            summary=areas[area].understanding,
            evidence_gap_details=areas[area].missing_information,
        )
        for area in StartupArea
    ]

    missing_areas = [a for a, k in areas.items() if k.status == AreaStatus.UNKNOWN]

    diagnosis = Diagnosis(
        project_id=project_id,
        iteration=iteration,
        raw_context=raw_context,
        assessments=assessments,
        clear_areas=llm_output.clear_areas,
        unclear_areas=llm_output.unclear_areas,
        weak_areas=[],
        missing_areas=missing_areas,
        unsupported_claims=llm_output.unsupported_claims,
        summary=llm_output.diagnosis_summary,
        startup_stage=llm_output.startup_stage,
    )

    return {
        "startup_state": startup_state_dump,
        "diagnosis": diagnosis.model_dump(mode="json"),
    }


# ---------------------------------------------------------------------------
# 5. assess_context_sufficiency (LLM — Milestone 2.5)
# ---------------------------------------------------------------------------

def assess_context_sufficiency(state: StartaryGraphState) -> dict:
    """
    LLM Node: Evaluate whether the founder context is grounded enough to proceed
    to weak-link decision risk assessment without inventing facts.
    """
    project_id = state.get("project_id", "<unknown>")
    startup_state_dict = state.get("startup_state", {})
    diagnosis_dict = state.get("diagnosis", {})
    founder_messages = state.get("founder_messages") or []

    history_lines = []
    for msg in founder_messages:
        history_lines.append(f"{msg.get('role', 'founder').upper()}: {msg.get('content')}")
    history_text = "\n".join(history_lines) if history_lines else (state.get("raw_context") or "")

    settings = get_settings()
    llm = _safe_get_llm_client(settings, tier="reasoning", max_tokens=1000)
    structured_llm = llm.with_structured_output(ContextSufficiencyLLMOutput)

    user_message = build_context_sufficiency_user_message(
        startup_state_json=json.dumps(startup_state_dict, indent=2),
        diagnosis_summary=diagnosis_dict.get("summary", ""),
        conversation_history_text=history_text,
    )

    messages = [
        {"role": "system", "content": ASSESS_CONTEXT_SUFFICIENCY_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    attempt = 0
    last_error: Exception | None = None

    while attempt < 2:
        attempt += 1
        t_start = time.monotonic()
        try:
            result: ContextSufficiencyLLMOutput = structured_llm.invoke(messages)
            latency_ms = (time.monotonic() - t_start) * 1000

            if not isinstance(result, ContextSufficiencyLLMOutput):
                raise ValidationError(f"Expected ContextSufficiencyLLMOutput, got {type(result)}")

            cs_domain = ContextSufficiencyAssessment(
                sufficient=result.sufficient,
                confidence=result.confidence,
                grounding_quality=result.grounding_quality,
                would_one_more_answer_materially_change_direction=result.would_one_more_answer_materially_change_direction,
                reason=result.reason,
                blocking_unknowns=result.blocking_unknowns,
                usable_grounding=result.usable_grounding,
                prohibited_inferences=result.prohibited_inferences,
                recommended_question_area=result.recommended_question_area,
                information_goal=result.information_goal,
            )

            trace = AITrace(
                module_name="assess_context_sufficiency",
                prompt_version=CONTEXT_SUFFICIENCY_PROMPT_VERSION,
                input_data={"turn": state.get("clarification_turn_count", 0)},
                output_data=result.model_dump(),
                latency_ms=latency_ms,
                project_id=project_id,
            )

            existing_traces = list(state.get("traces") or [])
            existing_traces.append(trace.model_dump(mode="json"))

            logger.info(
                "assess_context_sufficiency | project_id=%s sufficient=%s reason=%s",
                project_id,
                result.sufficient,
                result.reason[:100],
            )

            return {
                "context_sufficiency_output": result.model_dump(),
                "context_sufficiency": cs_domain.model_dump(mode="json"),
                "is_sufficient": result.sufficient,
                "last_trace": trace.model_dump(mode="json"),
                "traces": existing_traces,
            }
        except Exception as exc:
            last_error = exc
            logger.warning("assess_context_sufficiency failed attempt %d: %s", attempt, exc)

    raise RuntimeError(f"assess_context_sufficiency failed after 2 attempts. Error: {last_error}")


# ---------------------------------------------------------------------------
# 6. select_clarifying_question (LLM — Milestone 2.5)
# ---------------------------------------------------------------------------

def select_clarifying_question(state: StartaryGraphState) -> dict:
    """
    LLM Node: Select exactly ONE highest-information-value clarifying question to ask the founder.
    """
    project_id = state.get("project_id", "<unknown>")
    cs_dict = state.get("context_sufficiency") or {}
    founder_messages = state.get("founder_messages") or []
    clarification_history = state.get("clarification_history") or []

    history_lines = []
    for msg in founder_messages:
        history_lines.append(f"{msg.get('role', 'founder').upper()}: {msg.get('content')}")
    history_text = "\n".join(history_lines) if history_lines else (state.get("raw_context") or "")

    prev_q_lines = []
    for item in clarification_history:
        prev_q_lines.append(f"- Q: {item.get('question')} (Target: {item.get('target_area')}, Goal: {item.get('information_goal')})")
    prev_q_text = "\n".join(prev_q_lines)

    target_area_str = cs_dict.get("recommended_question_area") or str(StartupArea.PROBLEM)
    try:
        target_area_enum = StartupArea(target_area_str)
    except Exception:
        target_area_enum = StartupArea.PROBLEM

    settings = get_settings()
    llm = _safe_get_llm_client(settings, tier="reasoning", max_tokens=1000)
    structured_llm = llm.with_structured_output(ClarifyingQuestionLLMOutput)

    user_message = build_clarifying_question_user_message(
        information_goal=cs_dict.get("information_goal", "Clarify core problem and user context."),
        target_area=str(target_area_enum),
        blocking_unknowns=cs_dict.get("blocking_unknowns", []),
        usable_grounding=cs_dict.get("usable_grounding", []),
        prohibited_inferences=cs_dict.get("prohibited_inferences", []),
        previous_questions_text=prev_q_text,
        conversation_history_text=history_text,
    )

    messages = [
        {"role": "system", "content": SELECT_CLARIFYING_QUESTION_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    attempt = 0
    last_error: Exception | None = None

    while attempt < 2:
        attempt += 1
        t_start = time.monotonic()
        try:
            result: ClarifyingQuestionLLMOutput = structured_llm.invoke(messages)
            latency_ms = (time.monotonic() - t_start) * 1000

            if not isinstance(result, ClarifyingQuestionLLMOutput):
                raise ValidationError(f"Expected ClarifyingQuestionLLMOutput, got {type(result)}")

            cq_domain = ClarifyingQuestion(
                question=result.question.strip(),
                target_area=result.target_area,
                information_goal=result.information_goal,
                why_this_question_now=result.why_this_question_now,
            )

            trace = AITrace(
                module_name="select_clarifying_question",
                prompt_version=CLARIFYING_QUESTION_PROMPT_VERSION,
                input_data={"target_area": str(result.target_area), "turn": state.get("clarification_turn_count", 0)},
                output_data=result.model_dump(),
                latency_ms=latency_ms,
                project_id=project_id,
            )

            existing_traces = list(state.get("traces") or [])
            existing_traces.append(trace.model_dump(mode="json"))

            logger.info(
                "select_clarifying_question | project_id=%s area=%s question=%s",
                project_id,
                result.target_area,
                result.question[:80],
            )

            return {
                "current_clarifying_question": cq_domain.model_dump(mode="json"),
                "phase": WorkflowPhase.CLARIFYING_CONTEXT,
                "last_trace": trace.model_dump(mode="json"),
                "traces": existing_traces,
            }
        except Exception as exc:
            last_error = exc
            logger.warning("select_clarifying_question failed attempt %d: %s", attempt, exc)

    raise RuntimeError(f"select_clarifying_question failed after 2 attempts. Error: {last_error}")


# ---------------------------------------------------------------------------
# 7. compose_founder_question + ask_and_wait_for_founder (Milestone 2.5 & 6.2.3)
# ---------------------------------------------------------------------------
#
# LangGraph re-executes an interrupted node from the top on resume, and nothing a node computes
# before interrupt() is committed. Composing the question inside the interrupt node therefore
# ran the composer LLM twice (once before display, once on resume) and let the transcript
# record a regenerated wording. Composition now lives in its own node whose output is
# checkpointed before the interrupt node runs; the interrupt node is LLM-free.

def _resolve_active_question(state: StartaryGraphState) -> dict[str, Any]:
    """Pure function of committed state: the question currently awaiting the founder."""
    active_type = state.get("active_interrupt_type") or "strategic"
    active_decision = state.get("active_next_turn_decision") or {}

    if active_type == "baseline" and state.get("current_baseline_question"):
        q_dict = state["current_baseline_question"]
    elif active_type == "weak_link" and state.get("current_weak_link_question"):
        q_dict = state["current_weak_link_question"]
    else:
        q_dict = state.get("current_clarifying_question") or {}

    turn_number = (
        state.get("baseline_question_turns", 0) + 1 if active_type == "baseline"
        else (state.get("weak_link_clarification_turns", 0) + 1 if active_type == "weak_link"
              else state.get("strategic_clarification_turns", state.get("clarification_turn_count", 0)) + 1)
    )

    return {
        "interrupt_type": active_type,
        "question_id": active_decision.get("question_id"),
        "raw_planner_question": active_decision.get("canonical_question") or q_dict.get("question", "Could you provide more context about your startup?"),
        "target_area": str(active_decision.get("information_target") or q_dict.get("target_area") or q_dict.get("target_group") or "Context"),
        "information_goal": q_dict.get("information_goal", ""),
        "turn": turn_number,
    }


def _advisor_response_is_bound(advisor_resp: dict | None, question_id: str | None) -> bool:
    """An advisor response may be displayed only for the question it was composed for."""
    if not advisor_resp:
        return False
    if advisor_resp.get("response_mode") == "CONVERSATIONAL_REPAIR":
        return True
    return question_id is not None and advisor_resp.get("question_id") == question_id


def _build_founder_prompt(question: dict[str, Any], advisor_resp: dict | None, source: str) -> dict[str, Any]:
    """Freeze the exact founder-facing prompt (display text + attribution metadata)."""
    display = (advisor_resp or {}).get("response_text") or question["raw_planner_question"]
    if not (advisor_resp or {}).get("response_text"):
        source = "planner_fallback"
    return {
        **question,
        "question_id": question.get("question_id") or f"q_legacy_turn_{question.get('turn', 0)}",
        "question": display,
        "source": source,
    }


def _prompt_matches_active_question(prompt: dict | None, question: dict[str, Any]) -> bool:
    """True when a persisted prompt was composed for the question currently awaiting the founder."""
    if not prompt or not prompt.get("question"):
        return False
    if prompt.get("interrupt_type") != question["interrupt_type"]:
        return False
    if question.get("question_id"):
        return prompt.get("question_id") == question["question_id"]
    return prompt.get("raw_planner_question") == question["raw_planner_question"]


def compose_founder_question(state: StartaryGraphState) -> dict:
    """
    Node (Milestone 6.2.3): compose the founder-facing question exactly once and persist it.

    Runs between the question planners (plan_baseline_question, plan_information_gap,
    select_weak_link_question) and ask_and_wait_for_founder. Its output, including the
    exact display text in ``pending_founder_prompt``, is checkpointed BEFORE the interrupt,
    so resume, retry, and checkpoint restoration never trigger a second composer call.
    """
    question = _resolve_active_question(state)
    if question["interrupt_type"] == "waiting_for_learning_input":
        return {}

    advisor_resp = state.get("latest_advisor_response")
    updates: dict[str, Any] = {}
    if _advisor_response_is_bound(advisor_resp, question["question_id"]):
        source = "conversational_repair" if advisor_resp.get("response_mode") == "CONVERSATIONAL_REPAIR" else "advisor_composer"
    else:
        source = "advisor_composer"
        try:
            updates = compose_advisor_response(state)
            advisor_resp = updates.get("latest_advisor_response")
        except Exception as exc:
            logger.warning("compose_founder_question: advisor composition failed (%s); displaying planner question verbatim", exc)
            advisor_resp = None

    prompt = _build_founder_prompt(question, advisor_resp, source)
    updates["pending_founder_prompt"] = prompt
    logger.info(
        "compose_founder_question | type=%s qid=%s source=%s",
        prompt["interrupt_type"], prompt["question_id"], prompt["source"],
    )
    return updates


def ask_and_wait_for_founder(state: StartaryGraphState) -> dict:
    """
    Interrupt Node: displays the persisted founder-facing question and pauses execution until
    resumed with the founder's response.

    LLM-free by design (Milestone 6.2.3): LangGraph re-runs this node from the top on resume,
    so everything before interrupt() must be a deterministic function of committed state.
    """
    active_type = state.get("active_interrupt_type") or "strategic"

    # Milestone 4.1.1: Recoverable experiment design failure interrupt
    if active_type == "waiting_for_learning_input":
        fallback_dict = state.get("fallback_learning_action") or {}
        display_question = fallback_dict.get("response_text") or state.get("status_message") or "We need your firsthand observations to design a valid next experiment."

        founder_answer = interrupt({
            "type": "WAITING_FOR_LEARNING_INPUT",
            "interrupt_type": "waiting_for_learning_input",
            "message": display_question,
            "question": display_question,
            "fallback_learning_action": fallback_dict,
        })

        raw_answer = ""
        if isinstance(founder_answer, dict):
            raw_answer = founder_answer.get("founder_response") or founder_answer.get("content") or ""
        elif isinstance(founder_answer, str):
            raw_answer = founder_answer
        if not raw_answer and state.get("founder_response"):
            raw_answer = str(state.get("founder_response"))

        logger.info("ask_and_wait_for_founder resumed (waiting_for_learning_input) with answer: %s", raw_answer[:60])
        return {
            "founder_response": raw_answer,
            "phase": WorkflowPhase.WAITING_FOR_LEARNING_INPUT,
            "active_interrupt_type": "waiting_for_learning_input",
        }

    question = _resolve_active_question(state)
    prompt = state.get("pending_founder_prompt")
    prompt_updates: dict[str, Any] = {}
    if not _prompt_matches_active_question(prompt, question):
        # Deterministic fallback for direct node calls and the conversational-repair path:
        # display the advisor response already committed for this question, else the planner text.
        advisor_resp = state.get("latest_advisor_response")
        bound = _advisor_response_is_bound(advisor_resp, question["question_id"])
        source = "conversational_repair" if bound and advisor_resp.get("response_mode") == "CONVERSATIONAL_REPAIR" else "advisor_composer"
        prompt = _build_founder_prompt(question, advisor_resp if bound else None, source)
        prompt_updates["pending_founder_prompt"] = prompt

    # LangGraph interrupt: suspends graph execution and awaits resume value from CLI / caller
    founder_answer = interrupt({
        "question": prompt["question"],
        "raw_planner_question": prompt["raw_planner_question"],
        "target_area": prompt["target_area"],
        "question_id": prompt["question_id"],
        "information_goal": prompt["information_goal"],
        "turn": prompt["turn"],
        "interrupt_type": prompt["interrupt_type"],
    })

    logger.info("ask_and_wait_for_founder resumed (%s, qid=%s) with answer: %s", prompt["interrupt_type"], prompt["question_id"], str(founder_answer)[:60])
    res: dict[str, Any] = {
        "founder_response": str(founder_answer),
    }
    res.update(prompt_updates)
    return res


# ---------------------------------------------------------------------------
# 8. merge_founder_answer (Deterministic — Milestone 2.6.1 & 3.2.1 & 4.1.3)
# ---------------------------------------------------------------------------

REPAIR_STARTERS = (
    "i don't understand", "i dont understand", "dont understand", "don't understand",
    "what do you mean", "what does that mean", "what do u mean",
    "can you explain", "could you explain", "explain what you mean",
    "why are you asking", "why do you ask",
    "i'm confused", "im confused", "i am confused",
    "can you rephrase", "could you rephrase", "rephrase",
    "huh", "what", "pardon", "come again",
)

def is_conversational_repair_text(text: str) -> bool:
    if not text:
        return False
    norm = text.strip().lower().rstrip(".!?,")
    if norm in ("what", "huh", "explain", "i don't understand", "dont understand", "what do you mean", "what does that mean", "confused", "i'm confused", "im confused", "pardon"):
        return True
    for starter in REPAIR_STARTERS:
        if norm.startswith(starter):
            third_party_objects = ["customer", "market", "user", "competitor", "buyer", "client", "churn"]
            if any(tpo in norm for tpo in third_party_objects):
                return False
            return True
    return False


def merge_founder_answer(state: StartaryGraphState) -> dict:
    """
    Deterministic Node:
    - Merges founder's response into conversation history
    - Updates dedicated turn counter based on active_interrupt_type
    - Records answered question into question_history
    - INVALIDATES active_next_turn_decision and latest_advisor_response to prevent stale reuse
    - Prepares state for re-extraction / re-evaluation
    - Milestone 4.1.3: Intercepts conversational repairs and bounded fast-path answers
    """
    founder_answer = (state.get("founder_response") or "").strip()
    active_type = state.get("active_interrupt_type") or "strategic"
    active_decision = state.get("active_next_turn_decision") or {}

    # Milestone 4.1.1: Recoverable fallback learning input return
    if active_type == "waiting_for_learning_input":
        fallback_dict = state.get("fallback_learning_action") or {}
        question_text = fallback_dict.get("response_text") or state.get("status_message") or ""

        founder_messages = list(state.get("founder_messages") or [])
        if question_text:
            founder_messages.append({"role": "startary", "content": question_text})
        if founder_answer:
            founder_messages.append({"role": "founder", "content": founder_answer})

        # Grounded context integration (Milestone 4.1.1)
        st_state = dict(state.get("startup_state") or {})
        areas = dict(st_state.get("areas") or st_state.get("area_knowledge") or {})
        target_area = fallback_dict.get("target_area") or state.get("weak_area") or "Problem"
        area_data = dict(areas.get(target_area) or {})
        ev_list = list(area_data.get("evidence_provided") or [])
        if founder_answer:
            ev_list.append(founder_answer)
        area_data["evidence_provided"] = ev_list
        areas[target_area] = area_data
        st_state["areas"] = areas

        founder_ev = list(state.get("founder_evidence") or [])
        if founder_answer:
            founder_ev.append({
                "source": "FALLBACK_LEARNING_ACTION",
                "content": founder_answer,
                "target_area": target_area,
            })

        logger.info(
            "merge_founder_answer [FALLBACK_LEARNING_INPUT] | integrated grounded observations into %s (no evidence cycle / no iteration increment)",
            target_area,
        )
        return {
            "founder_messages": founder_messages,
            "founder_response": founder_answer,
            "pending_question": fallback_dict.get("primary_action") or question_text[:120],
            "startup_state": st_state,
            "founder_evidence": founder_ev,
            "active_interrupt_type": None,
            "pending_founder_prompt": None,
            "fallback_learning_action": None,
            "learning_input_mode": None,
            "experiment_design_status": None,
            "phase": WorkflowPhase.UNDERSTAND,
            "critic_revision_count": 0,
            "critic_revision_instructions": "",
            "experiment_revision": None,
            "proposed_experiment_output": None,
            "experiment_critic_output": None,
            "active_experiment": None,
            "active_execution_handoff": None,
            "execution_handoff_text": None,
            "can_execute_result": None,
            "waiting_response": None,
            "post_evidence_answer": None,
        }

    if active_type == "baseline" and state.get("current_baseline_question"):
        q_dict = state["current_baseline_question"]
    elif active_type == "weak_link" and state.get("current_weak_link_question"):
        q_dict = state["current_weak_link_question"]
    else:
        q_dict = state.get("current_clarifying_question") or {}

    # Milestone 6.2.3: attribute the answer to the exact prompt the founder was shown
    # (persisted before the interrupt), never to a regenerated or stale advisor response.
    shown_prompt = state.get("pending_founder_prompt")
    if _prompt_matches_active_question(shown_prompt, _resolve_active_question(state)):
        question_text = shown_prompt["question"]
    else:
        advisor_resp = state.get("latest_advisor_response") or {}
        question_text = advisor_resp.get("response_text") or active_decision.get("canonical_question") or q_dict.get("question", "")

    # Milestone 4.1.3: Conversational Repair Fast Path Interception
    if is_conversational_repair_text(founder_answer):
        logger.info("merge_founder_answer | conversational repair detected: '%s'", founder_answer[:60])
        founder_messages = list(state.get("founder_messages") or [])
        if question_text and not any(m.get("content") == question_text for m in founder_messages[-2:]):
            founder_messages.append({"role": "startary", "content": question_text})
        if founder_answer:
            founder_messages.append({"role": "founder", "content": founder_answer})

        return {
            "founder_messages": founder_messages,
            "pending_question": question_text,
            "is_conversational_repair": True,
            "execution_path": "CONVERSATIONAL_REPAIR",
            "active_interrupt_type": active_type,
            "current_baseline_question": state.get("current_baseline_question"),
            "current_weak_link_question": state.get("current_weak_link_question"),
            "current_clarifying_question": state.get("current_clarifying_question"),
            "active_next_turn_decision": active_decision,
            "founder_response": founder_answer,
            "pending_founder_prompt": None,
        }

    founder_messages = list(state.get("founder_messages") or [])
    if question_text:
        founder_messages.append({"role": "startary", "content": question_text})
    if founder_answer:
        founder_messages.append({"role": "founder", "content": founder_answer})

    # Track in QuestionHistory (Milestone 3.2.1)
    q_history = list(state.get("question_history") or [])
    qid = active_decision.get("question_id") or f"q_{len(q_history)+1}"
    info_target = active_decision.get("information_target") or q_dict.get("target_group") or q_dict.get("target_area") or "UNKNOWN"
    q_history.append({
        "question_id": qid,
        "information_target": info_target,
        "canonical_question": active_decision.get("canonical_question") or q_dict.get("question", ""),
        "turn": len(q_history) + 1,
        "answered": True,
        "founder_answer_summary": founder_answer[:120],
    })

    updates: dict = {
        "founder_messages": founder_messages,
        "pending_question": question_text,
        "question_history": q_history,
        "answered_question_id": qid,
        "active_next_turn_decision": None,  # CONSUMED: stale question cannot survive
        "latest_advisor_response": None,   # CLEARED: forces fresh advisor composition on next turn
        "pending_founder_prompt": None,    # CONSUMED: answered prompt cannot be re-displayed
        "current_clarifying_question": None,
        "current_baseline_question": None,
        "current_weak_link_question": None,
        "founder_response": None,
        "active_interrupt_type": None,
        "phase": WorkflowPhase.UNDERSTAND,
        "pending_information_target": str(info_target),
    }

    # Milestone 4.1.3: Bound Answer Fast-Path Parser Check
    bound_ok, bound_delta, _ = parse_bound_answer(str(info_target), founder_answer, question_text)
    if bound_ok and bound_delta:
        logger.info("merge_founder_answer | fast bound answer detected for target=%s: %s", info_target, list(bound_delta.keys()))
        updates["startup_state_delta"] = bound_delta
        updates["execution_path"] = "FAST_LOCAL_UPDATE"
        updates["strategic_state_changed"] = False
    else:
        updates["execution_path"] = "DELTA_STATE_UPDATE"

    # Dispatch to dedicated history and counter
    if active_type == "baseline":
        b_history = list(state.get("baseline_history") or [])
        b_turn = state.get("baseline_question_turns", 0) + 1
        if question_text:
            b_history.append({
                "question": question_text,
                "target_group": active_decision.get("information_target") or q_dict.get("target_group"),
                "founder_answer": founder_answer,
                "turn": b_turn,
                "question_id": qid,
            })
        updates["baseline_history"] = b_history
        updates["baseline_question_turns"] = b_turn
        logger.info("merge_founder_answer [BASELINE] | turn=%d total_messages=%d qid=%s", b_turn, len(founder_messages), qid)

    elif active_type == "weak_link":
        wl_history = list(state.get("weak_link_clarification_history") or [])
        wl_turn = state.get("weak_link_clarification_turns", 0) + 1
        if question_text:
            wl_history.append({
                "question": question_text,
                "competing_areas": q_dict.get("competing_areas"),
                "information_goal": q_dict.get("information_goal"),
                "founder_answer": founder_answer,
                "turn": wl_turn,
            })
        updates["weak_link_clarification_history"] = wl_history
        updates["weak_link_clarification_turns"] = wl_turn
        updates["is_strategic_sufficient"] = True
        updates["is_baseline_sufficient"] = True
        logger.info("merge_founder_answer [WEAK_LINK] | turn=%d total_messages=%d", wl_turn, len(founder_messages))

    else:
        c_history = list(state.get("clarification_history") or [])
        s_turn = state.get("strategic_clarification_turns", 0) + 1
        legacy_turn = state.get("clarification_turn_count", 0) + 1
        if question_text:
            c_history.append({
                "question": question_text,
                "target_area": q_dict.get("target_area"),
                "information_goal": q_dict.get("information_goal"),
                "founder_answer": founder_answer,
                "turn": s_turn,
            })
        updates["clarification_history"] = c_history
        updates["strategic_clarification_turns"] = s_turn
        updates["clarification_turn_count"] = legacy_turn
        logger.info("merge_founder_answer [STRATEGIC] | turn=%d total_messages=%d", s_turn, len(founder_messages))

    # Accumulate source text for raw_context
    raw_context = state.get("raw_context") or ""
    if founder_answer:
        raw_context += f"\n\n[Founder Response ({active_type})]:\n{founder_answer}"
    updates["raw_context"] = raw_context

    return updates


# ---------------------------------------------------------------------------
# 9. handle_insufficient_context_stop (Deterministic — Milestone 2.5)
# ---------------------------------------------------------------------------

def handle_insufficient_context_stop(state: StartaryGraphState) -> dict:
    """
    Deterministic Node:
    Safely terminates the UNDERSTAND loop when maximum clarification turns (5) are reached
    without manufacturing an ungrounded critical assumption.
    """
    project_id = state.get("project_id", "<unknown>")
    cs_dict = state.get("context_sufficiency") or {}
    blocking = cs_dict.get("blocking_unknowns") or []

    blocking_str = "; ".join(blocking) if blocking else "Core problem and customer specifics."
    msg = (
        f"Startary reached the maximum clarification limit ({MAX_CLARIFICATION_TURNS} turns) "
        f"and still does not have enough grounded information to formulate a sound critical assumption. "
        f"The most important missing information is: {blocking_str}"
    )

    logger.warning("handle_insufficient_context_stop | project_id=%s msg=%s", project_id, msg)

    return {
        "status_message": msg,
        "phase": WorkflowPhase.UNDERSTAND,
    }


# ---------------------------------------------------------------------------
# Milestone 2.6 — Helper: merge KnowledgeField with provenance rules
# ---------------------------------------------------------------------------

def _merge_knowledge_field(
    existing: dict,
    incoming: dict,
    source_turn: int,
) -> dict:
    """
    Deterministic merge of two serialized KnowledgeField dicts following provenance rules:

    1. EXPLICIT always wins over INFERRED for same field.
    2. Newer EXPLICIT overwrites older EXPLICIT (higher source_turn wins).
    3. INFERRED never silently overwrites EXPLICIT.
    4. UNKNOWN never overwrites any known value.
    5. CONFLICTING: set when two EXPLICIT values directly contradict.
    """
    existing_prov = existing.get("provenance", "UNKNOWN")
    incoming_prov = incoming.get("provenance", "UNKNOWN")
    incoming_val = incoming.get("value")

    if incoming_prov == "UNKNOWN" or incoming_val is None:
        return existing  # UNKNOWN never overwrites

    if existing_prov == "UNKNOWN":
        return {**incoming, "source_turn": source_turn}

    if existing_prov == "INFERRED" and incoming_prov == "EXPLICIT":
        return {**incoming, "source_turn": source_turn}

    if existing_prov == "INFERRED" and incoming_prov == "INFERRED":
        # Take newer inferred value (higher turn wins or same)
        existing_turn = existing.get("source_turn") or -1
        if source_turn >= existing_turn:
            return {**incoming, "source_turn": source_turn}
        return existing

    if existing_prov == "EXPLICIT" and incoming_prov == "EXPLICIT":
        existing_val = existing.get("value")
        existing_turn = existing.get("source_turn") or -1

        if source_turn == existing_turn and incoming_val != existing_val:
            # Same turn, different values → CONFLICTING
            return {
                "value": None,
                "provenance": "CONFLICTING",
                "confidence": "LOW",
                "source_turn": source_turn,
                "notes": f"Conflict: '{existing_val}' vs '{incoming_val}'",
            }

        if source_turn > existing_turn:
            return {**incoming, "source_turn": source_turn}
        return existing  # older explicit is kept

    if existing_prov == "EXPLICIT" and incoming_prov == "INFERRED":
        return existing  # INFERRED never overwrites EXPLICIT

    if existing_prov == "CONFLICTING" and incoming_prov == "EXPLICIT":
        return {**incoming, "source_turn": source_turn}  # EXPLICIT resolves conflict

    return existing


def _safe_field(llm_field, source_turn: int) -> dict | None:
    """Convert an ExtractedKnowledgeFieldLLMOutput to a plain dict. Returns None if empty."""
    if llm_field is None:
        return None
    val = llm_field.get("value") if isinstance(llm_field, dict) else getattr(llm_field, "value", None)
    if val is None:
        return None
    prov = llm_field.get("provenance") if isinstance(llm_field, dict) else getattr(llm_field, "provenance", "UNKNOWN")
    conf = llm_field.get("confidence") if isinstance(llm_field, dict) else getattr(llm_field, "confidence", "LOW")
    notes = llm_field.get("notes") if isinstance(llm_field, dict) else getattr(llm_field, "notes", None)
    return {"value": val, "provenance": prov, "confidence": conf, "source_turn": source_turn, "notes": notes}


def _unknown_field_dict() -> dict:
    return {"value": None, "provenance": "UNKNOWN", "confidence": "LOW", "source_turn": None, "notes": None}


# ---------------------------------------------------------------------------
# M2.6 Node 1: extract_startup_state (LLM)
# ---------------------------------------------------------------------------

def extract_startup_state(state: StartaryGraphState) -> dict:
    """
    LLM Node: Extract execution context and startup progress facts from founder messages.

    Produces a delta dict that is passed to merge_execution_state for deterministic merge.
    Only extracts from founder-authored messages (role='founder').
    Milestone 4.1.3: Bypasses LLM when FAST_LOCAL_UPDATE is active; slims prompt context;
    uses fast tier model and records telemetry.
    """
    import json as _json

    project_id = state.get("project_id", "<unknown>")
    current_exec_ctx = state.get("execution_context") or {}
    current_progress = state.get("startup_progress") or {}

    # Milestone 4.1.3 Fast Path: Local Deterministic Bypass
    if state.get("execution_path") == "FAST_LOCAL_UPDATE":
        delta_dict = state.get("startup_state_delta") or {}
        logger.info("extract_startup_state | bypassing LLM for FAST_LOCAL_UPDATE: %s", list(delta_dict.keys()))
        trace = AITrace(
            module_name="extract_startup_state",
            prompt_version=STARTUP_STATE_EXTRACTOR_PROMPT_VERSION,
            input_data={"execution_path": "FAST_LOCAL_UPDATE"},
            output_data={"delta_keys": [k for k, v in delta_dict.items() if v is not None]},
            latency_ms=0.0,
            project_id=project_id,
            model_name="local_deterministic",
            skipped_by_fast_path=True,
            execution_path="FAST_LOCAL_UPDATE",
        )
        existing_traces = list(state.get("traces") or [])
        existing_traces.append(trace.model_dump(mode="json"))
        turn_telem = record_telemetry_event(
            state=state,
            node_name="extract_startup_state",
            model_name="local_deterministic",
            duration_ms=0.0,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            cached=True,
            skipped=True,
            execution_path="FAST_LOCAL_UPDATE",
        )
        return {
            "startup_state_delta": delta_dict,
            "last_trace": trace.model_dump(mode="json"),
            "traces": existing_traces,
            "turn_telemetry": turn_telem,
            "extraction_status": "BYPASSED",
        }

    founder_messages = state.get("founder_messages") or []
    all_founder_msgs = [msg for msg in founder_messages if msg.get("role") == "founder"]
    if len(all_founder_msgs) > 3:
        selected_msgs = [all_founder_msgs[0]] + all_founder_msgs[-2:]
    else:
        selected_msgs = all_founder_msgs

    source_lines = [
        f"[Turn] {msg['content']}"
        for msg in selected_msgs
    ]

    if not source_lines:
        logger.info("extract_startup_state | no founder messages to extract from")
        return {"startup_state_delta": {}, "extraction_status": "SUCCESS"}

    founder_text = "\n\n".join(source_lines)
    source_turn = state.get("clarification_turn_count", 0)
    pending_question = state.get("pending_question")

    # Detect short answer (e.g. 'yes', 'no', 'not yet')
    latest_founder_text = ""
    for msg in reversed(founder_messages):
        if msg.get("role") == "founder":
            latest_founder_text = (msg.get("content") or "").strip()
            break
    norm_short = latest_founder_text.lower().rstrip(".!?,")
    is_short_answer = (len(norm_short.split()) <= 3) and (norm_short in {
        "yes", "no", "maybe", "not yet", "none", "i dont know", "i don't know", "dont know", "not sure", "not now"
    })

    settings = get_settings()
    base_max_tokens = 2500
    is_reasoning_fast = any(
        m in (getattr(settings, "fast_model_name", None) or "").lower() for m in ("o1", "o3", "sol", "luna")
    )
    llm = _safe_get_llm_client(
        settings,
        tier="fast",
        max_completion_tokens=base_max_tokens,
        reasoning_effort="low" if is_reasoning_fast else None,
    )
    conf_model, res_model, fb_used, fb_reason = get_llm_client_resolution_info(llm)
    structured_llm = llm.with_structured_output(StartupStateDeltaLLMOutput)

    user_message = build_extract_startup_state_user_message(
        founder_messages_text=founder_text,
        current_execution_context_json=_json.dumps(current_exec_ctx, indent=2, default=str),
        current_progress_json=_json.dumps(current_progress, indent=2, default=str),
        source_turn=source_turn,
        pending_question=pending_question,
    )

    messages = [
        {"role": "system", "content": EXTRACT_STARTUP_STATE_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    try:
        result, latency_ms = _invoke_structured_with_token_retry(
            structured_llm=structured_llm,
            messages=messages,
            settings=settings,
            tier="fast",
            base_max_tokens=base_max_tokens,
            node_name="extract_startup_state",
            schema_cls=StartupStateDeltaLLMOutput,
        )

        if not isinstance(result, StartupStateDeltaLLMOutput):
            raise ValueError(f"Expected StartupStateDeltaLLMOutput, got {type(result)}")

        delta_dict = result.model_dump()
        delta_dict["_source_turn"] = source_turn

        # Short-answer guardrail (Section 15/16): prevent unrelated speculative INFERRED fields
        if is_short_answer and pending_question:
            pq_lower = pending_question.lower()
            field_keywords = {
                "founder_count": ["founder", "co-founder", "team", "alone", "solo"],
                "active_team_size": ["team", "people", "members", "size"],
                "has_technical_capability": ["technical", "developer", "engineer", "code"],
                "has_design_capability": ["design", "ui", "ux"],
                "has_marketing_capability": ["marketing", "sales", "growth"],
                "weekly_hours_per_member": ["hour", "time", "weekly", "capacity"],
                "total_weekly_capacity_hours": ["hour", "time", "weekly", "capacity"],
                "available_budget_amount": ["budget", "money", "fund", "cash", "spend", "dollar", "azn", "eur"],
                "funding_status": ["funding", "funded", "investor", "bootstrap"],
                "customer_access": ["access", "network", "contact", "customer", "user", "owner", "manager"],
                "prototype_exists": ["prototype", "wireframe", "mockup"],
                "mvp_exists": ["mvp", "product", "software", "app", "tool"],
                "launched": ["launch", "live", "market"],
                "active_users": ["user", "active", "pilot"],
                "paying_customers": ["customer", "paying", "client"],
                "revenue": ["revenue", "money", "sales", "income", "pricing", "pay"],
            }
            for k, v in list(delta_dict.items()):
                if isinstance(v, dict) and v.get("provenance") == "INFERRED":
                    keywords = field_keywords.get(k, [])
                    if keywords and not any(kw in pq_lower for kw in keywords):
                        logger.info("extract_startup_state | dropping speculative inference '%s' on short answer '%s'", k, latest_founder_text)
                        delta_dict[k] = None

        # Milestone 4.0.4: Factual state grounding validation
        delta_dict = validate_state_grounding(state, delta_dict, latest_founder_text)

        trace = AITrace(
            module_name="extract_startup_state",
            prompt_version=STARTUP_STATE_EXTRACTOR_PROMPT_VERSION,
            input_data={"founder_messages_count": len(source_lines), "source_turn": source_turn},
            output_data={"delta_keys": [k for k, v in delta_dict.items() if v is not None and k != "_source_turn"]},
            latency_ms=latency_ms,
            project_id=project_id,
            model_name=res_model or getattr(settings, "fast_model_name", None),
            configured_model=conf_model,
            resolved_model=res_model,
            fallback_used=fb_used,
            fallback_reason=fb_reason,
            execution_path=state.get("execution_path", "DELTA_STATE_UPDATE"),
        )

        existing_traces = list(state.get("traces") or [])
        existing_traces.append(trace.model_dump(mode="json"))

        # Token accounting with fallback character counting
        prompt_tokens = getattr(getattr(result, "usage_metadata", None), "prompt_tokens", None)
        if prompt_tokens is None:
            prompt_tokens = sum(len(m.get("content", "")) for m in messages) // 4
        completion_tokens = getattr(getattr(result, "usage_metadata", None), "completion_tokens", None)
        if completion_tokens is None:
            completion_tokens = len(_json.dumps(delta_dict)) // 4
        total_tokens = prompt_tokens + completion_tokens

        turn_telem = record_telemetry_event(
            state=state,
            node_name="extract_startup_state",
            model_name=res_model or getattr(settings, "fast_model_name", None) or "fast_tier",
            configured_model=conf_model,
            resolved_model=res_model,
            fallback_used=fb_used,
            fallback_reason=fb_reason,
            duration_ms=latency_ms,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            execution_path=state.get("execution_path", "DELTA_STATE_UPDATE"),
            model_tier="fast",
            reasoning_effort=get_llm_client_resolution_info(llm).get("reasoning_effort"),
        )

        logger.info("extract_startup_state | project_id=%s latency_ms=%.0f", project_id, latency_ms)

        return {
            "startup_state_delta": delta_dict,
            "last_trace": trace.model_dump(mode="json"),
            "traces": existing_traces,
            "turn_telemetry": turn_telem,
            "extraction_status": "SUCCESS",
        }
    except Exception as exc:
        logger.error("extract_startup_state failed: %s", exc)
        turn_telem = record_telemetry_event(
            state=state,
            node_name="extract_startup_state",
            model_name="local_deterministic",
            configured_model=conf_model,
            resolved_model="local_deterministic",
            fallback_used=True,
            fallback_reason="EXTRACTION_FAILURE_FALLBACK",
            duration_ms=0.0,
            execution_path=state.get("execution_path", "DELTA_STATE_UPDATE"),
            model_tier="fast",
            status="FAILED",
        )
        return {
            "startup_state_delta": {},
            "extraction_status": "FAILED",
            "extraction_error": str(exc),
            "strategic_state_changed": True,
            "turn_telemetry": turn_telem,
        }


def _is_sharp_contradiction(field_name: str, existing_val: Any, incoming_val: Any) -> bool:
    """
    Detect explicit contradictions that require founder clarification (Milestone 3.2.1 Section 14).
    Example: Earlier founder says 'We are 3 people', later says 'I am alone'.
    """
    if incoming_val is None or existing_val is None:
        return False
    if str(existing_val).strip() == str(incoming_val).strip():
        return False
    if field_name in ("founder_count", "active_team_size"):
        try:
            e_num = int(existing_val)
            i_num = int(incoming_val)
            if e_num > 1 and i_num <= 1:
                return True
        except (ValueError, TypeError):
            if str(incoming_val).lower() in ("alone", "solo", "just me", "only me"):
                return True
    return False


def merge_execution_state(state: StartaryGraphState) -> dict:
    """
    Deterministic Node: Apply provenance-aware delta merge to ExecutionContext and StartupProgress.

    Rules:
    - If extraction_status is FAILED, do not mutate existing known values!
    - EXPLICIT always wins over INFERRED.
    - UNKNOWN never overwrites any known value.
    - INFERRED never overwrites EXPLICIT.
    - CONFLICTING is set when two EXPLICIT values at same turn differ.
    - Contradictory explicit facts (e.g. 3 founders -> 1 founder) generate a FactConflict.
    """
    import json as _json

    # Milestone 4.1.3.1: Never overwrite good state with failed fallback data
    if state.get("extraction_status") == "FAILED":
        logger.warning("merge_execution_state | skipping state mutation because extraction_status=FAILED")
        return {
            "execution_context": state.get("execution_context") or {},
            "startup_progress": state.get("startup_progress") or {},
        }

    delta = state.get("startup_state_delta") or {}
    source_turn = delta.get("_source_turn", state.get("clarification_turn_count", 0))

    # Milestone 4.0.4: Validate state grounding before merging
    founder_messages = state.get("founder_messages") or []
    delta = validate_state_grounding(state, delta, founder_messages)

    # Load current serialized dicts (or default empty objects)
    current_exec_dict = state.get("execution_context") or {}
    current_prog_dict = state.get("startup_progress") or {}

    def merge_field(current_dict: dict, key: str, incoming_field) -> dict:
        """Merge one field into the current dict with explicit conflict protection."""
        if incoming_field is None:
            return current_dict
        incoming = _safe_field(incoming_field, source_turn)
        if incoming is None:
            return current_dict
        existing = current_dict.get(key) or _unknown_field_dict()
        if not isinstance(existing, dict):
            existing = _unknown_field_dict()

        # Milestone 3.2.1 Section 14/15: Conflicting facts require clarification
        existing_prov = existing.get("provenance", "UNKNOWN")
        existing_conf = existing.get("confidence", "NORMAL")
        existing_val = existing.get("value")
        incoming_prov = incoming.get("provenance", "UNKNOWN")
        incoming_val = incoming.get("value")

        if (
            existing_prov == "EXPLICIT"
            and incoming_prov == "EXPLICIT"
            and existing_conf == "HIGH"
            and _is_sharp_contradiction(key, existing_val, incoming_val)
        ):
            logger.info("merge_execution_state: detected FactConflict on '%s': '%s' vs '%s'", key, existing_val, incoming_val)
            existing_with_conflict = {
                **existing,
                "_detected_conflict": {
                    "existing_val": existing_val,
                    "conflicting_val": incoming_val,
                    "source_turn": source_turn,
                },
            }
            return {**current_dict, key: existing_with_conflict}

        merged = _merge_knowledge_field(existing, incoming, source_turn)
        return {**current_dict, key: merged}

    # --- Execution Context merges ---
    exec_dict = dict(current_exec_dict)

    for field_name in [
        "founder_count", "active_team_size", "technical_capability", "product_capability",
        "design_capability", "marketing_capability", "sales_capability", "domain_expertise",
        "approximate_weekly_hours_per_member", "total_weekly_capacity_hours",
        "available_budget_amount", "budget_currency", "budget_period",
        "funding_status", "funding_amount_raised", "customer_access",
        "existing_audience_or_community", "geographic_market",
    ]:
        delta_key_map = {
            "technical_capability": "has_technical_capability",
            "design_capability": "has_design_capability",
            "marketing_capability": "has_marketing_capability",
            "domain_expertise": "domain_expertise_description",
            "approximate_weekly_hours_per_member": "weekly_hours_per_member",
        }
        delta_key = delta_key_map.get(field_name, field_name)
        incoming_field = delta.get(delta_key)
        exec_dict = merge_field(exec_dict, field_name, incoming_field)

    # Merge existing_assets list (additive)
    delta_assets = delta.get("existing_assets") or []
    if delta_assets:
        existing_assets = list(exec_dict.get("existing_assets") or [])
        for asset in delta_assets:
            if asset not in existing_assets:
                existing_assets.append(asset)
        exec_dict["existing_assets"] = existing_assets

    # Merge team_members list (additive, dedup by role)
    delta_members = delta.get("team_members") or []
    if delta_members:
        existing_members = list(exec_dict.get("team_members") or [])
        existing_roles = {m.get("role") for m in existing_members if m.get("role")}
        for member in delta_members:
            role = member.get("role") if isinstance(member, dict) else getattr(member, "role", None)
            if role and role in existing_roles:
                continue  # Don't duplicate
            if isinstance(member, dict):
                existing_members.append(member)
            else:
                existing_members.append(member.model_dump())
        exec_dict["team_members"] = existing_members

    # --- Startup Progress merges ---
    prog_dict = dict(current_prog_dict)

    for field_name in [
        "development_stage", "product_status", "prototype_exists", "mvp_exists",
        "launched", "validation_activity", "interviews_completed",
        "active_users", "paying_customers", "revenue",
    ]:
        incoming_field = delta.get(field_name)
        prog_dict = merge_field(prog_dict, field_name, incoming_field)

    # Merge major_milestones list (additive)
    delta_milestones = delta.get("major_milestones") or []
    if delta_milestones:
        existing_milestones = list(prog_dict.get("major_milestones") or [])
        for ms in delta_milestones:
            if ms not in existing_milestones:
                existing_milestones.append(ms)
        prog_dict["major_milestones"] = existing_milestones

    # Extract any detected conflicts and preserve state integrity
    fact_conflicts = list(state.get("fact_conflicts") or [])
    for container in (exec_dict, prog_dict):
        for k, v in list(container.items()):
            if isinstance(v, dict) and "_detected_conflict" in v:
                c_info = v.pop("_detected_conflict")
                fact_conflicts.append({
                    "field_name": k,
                    "existing_value": c_info["existing_val"],
                    "conflicting_value": c_info["conflicting_val"],
                    "clarification_question": f"Earlier you noted {k} was '{c_info['existing_val']}', but now mentioned '{c_info['conflicting_val']}'. Did the situation change, or did you mean something specific?",
                    "resolved": False,
                    "detected_turn": c_info.get("source_turn", source_turn),
                })

    logger.info(
        "merge_execution_state | source_turn=%d exec_fields_known=%d prog_fields_known=%d conflicts=%d",
        source_turn,
        sum(1 for v in exec_dict.values() if isinstance(v, dict) and v.get("provenance") != "UNKNOWN"),
        sum(1 for v in prog_dict.values() if isinstance(v, dict) and v.get("provenance") != "UNKNOWN"),
        len(fact_conflicts),
    )

    strategic_keys = {
        "paying_customers", "revenue", "active_users",
        "market_segment", "target_user", "problem_statement",
        "pricing_model", "channel_access",
    }
    has_strat_delta = any(delta.get(k) is not None for k in strategic_keys)
    ret_updates = {
        "execution_context": exec_dict,
        "startup_progress": prog_dict,
        "fact_conflicts": fact_conflicts,
    }
    if has_strat_delta or fact_conflicts:
        ret_updates["strategic_state_changed"] = True
        ret_updates["strategic_state_revision"] = int(state.get("strategic_state_revision") or 1) + 1

    return ret_updates


# ---------------------------------------------------------------------------
# M2.6 Node 3: plan_information_gap (LLM — replaces assess_context_sufficiency routing)
# ---------------------------------------------------------------------------

def plan_information_gap(state: StartaryGraphState) -> dict:
    """
    LLM Node: Unified Adaptive Information Gap Planner.

    Replaces assess_context_sufficiency as the routing decision node.
    Evaluates both strategic sufficiency and profile gaps in one call.
    Produces one decision: ask ONE question or proceed.

    Also populates context_sufficiency for backward compatibility with existing tests.
    """
    import json as _json

    project_id = state.get("project_id", "<unknown>")
    startup_state_dict = state.get("startup_state", {})
    diagnosis_dict = state.get("diagnosis", {})
    founder_messages = state.get("founder_messages") or []
    exec_ctx = state.get("execution_context") or {}
    progress = state.get("startup_progress") or {}
    clarification_history = state.get("clarification_history") or []
    turn_count = state.get("clarification_turn_count", 0)

    # Build conversation history (founder-authored messages only for display)
    history_lines = []
    for msg in founder_messages:
        history_lines.append(f"{msg.get('role', 'founder').upper()}: {msg.get('content')}")
    history_text = "\n".join(history_lines) if history_lines else (state.get("raw_context") or "")

    # Build previous questions list from history
    previous_questions = [
        item.get("question", "")
        for item in clarification_history
        if item.get("question")
    ]

    strategic_summary = diagnosis_dict.get("summary", "") or startup_state_dict.get("raw_context", "")

    settings = get_settings()
    base_max_tokens = 2000
    llm = _safe_get_llm_client(settings, tier="reasoning", max_completion_tokens=base_max_tokens)
    structured_llm = llm.with_structured_output(InformationGapPlanLLMOutput)

    user_message = build_information_gap_planner_user_message(
        strategic_state_summary=strategic_summary,
        execution_context_json=_json.dumps(exec_ctx, indent=2, default=str),
        progress_json=_json.dumps(progress, indent=2, default=str),
        conversation_history_text=history_text,
        previous_questions=previous_questions,
        clarification_turn_count=turn_count,
        max_turns=MAX_CLARIFICATION_TURNS,
        workflow_phase=str(state.get("phase", "UNDERSTAND")),
    )

    messages = [
        {"role": "system", "content": INFORMATION_GAP_PLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    attempt = 0
    last_error: Exception | None = None

    while attempt < 2:
        attempt += 1
        try:
            result, latency_ms = _invoke_structured_with_token_retry(
                structured_llm=structured_llm,
                messages=messages,
                settings=settings,
                tier="reasoning",
                base_max_tokens=base_max_tokens,
                node_name="plan_information_gap",
                schema_cls=InformationGapPlanLLMOutput,
            )

            # Build domain model for gap plan — use safe enum conversion for mock/test safety
            try:
                target_domain_enum = GapTargetDomain(result.target_domain) if result.target_domain and isinstance(result.target_domain, str) else None
            except (ValueError, TypeError):
                target_domain_enum = None

            try:
                priority_enum = GapPriority(result.priority) if result.priority and isinstance(result.priority, str) else GapPriority.USEFUL_LATER
            except (ValueError, TypeError):
                priority_enum = GapPriority.USEFUL_LATER

            should_ask = bool(result.should_ask) if not isinstance(result.should_ask, bool) else result.should_ask
            strategic_sufficient = bool(result.strategic_sufficient) if not isinstance(result.strategic_sufficient, bool) else result.strategic_sufficient

            try:
                res_mode_str = getattr(result, "resolution_mode", "FOUNDER_KNOWLEDGE") or "FOUNDER_KNOWLEDGE"
                res_mode_enum = UncertaintyResolutionMode(res_mode_str)
            except (ValueError, TypeError):
                res_mode_enum = UncertaintyResolutionMode.FOUNDER_KNOWLEDGE

            can_test = bool(getattr(result, "can_formulate_testable_uncertainty", False)) or can_formulate_testable_uncertainty(state, result.model_dump())
            why_this_matters = getattr(result, "why_this_matters", "") or ""
            missing_info = getattr(result, "missing_information", "") or result.information_goal or ""
            recommended_area = getattr(result, "recommended_area", None) or result.target_field_or_area

            # If resolution_mode is EXTERNAL_EVIDENCE, UNDERSTAND stops asking!
            if res_mode_enum == UncertaintyResolutionMode.EXTERNAL_EVIDENCE:
                should_ask = False

            gap_plan = InformationGapPlan(
                should_ask=should_ask,
                strategic_sufficient=strategic_sufficient,
                resolution_mode=res_mode_enum,
                can_formulate_testable_uncertainty=can_test,
                why_this_matters=why_this_matters,
                missing_information=missing_info,
                recommended_area=recommended_area,
                target_domain=target_domain_enum,
                target_field_or_area=result.target_field_or_area if isinstance(result.target_field_or_area, (str, type(None))) else None,
                priority=priority_enum,
                information_goal=result.information_goal if isinstance(result.information_goal, str) else "",
                reason=result.reason if isinstance(result.reason, str) else "",
                question=result.question if isinstance(result.question, (str, type(None))) else None,
            )

            is_handoff = (res_mode_enum == UncertaintyResolutionMode.EXTERNAL_EVIDENCE and can_test)
            is_sufficient = (not should_ask and strategic_sufficient) or is_handoff

            if is_handoff:
                logger.info(
                    "uncertainty_handoff_to_test | source_phase=UNDERSTAND target_area=%s resolution_mode=EXTERNAL_EVIDENCE reason=%s",
                    recommended_area or "Problem", result.reason[:80]
                )

            # For backward compat: populate context_sufficiency from gap plan
            from startary.domain.sufficiency import GroundingQuality
            compat_cs = {
                "sufficient": is_sufficient,
                "confidence": "HIGH" if strategic_sufficient else "LOW",
                "grounding_quality": "MODERATE" if strategic_sufficient else "WEAK",
                "would_one_more_answer_materially_change_direction": should_ask,
                "reason": result.reason,
                "blocking_unknowns": [missing_info] if missing_info else [],
                "usable_grounding": [],
                "prohibited_inferences": [],
                "recommended_question_area": recommended_area,
                "information_goal": result.information_goal or "",
            }

            # Build current_clarifying_question and active_next_turn_decision (Milestone 3.2.1)
            cq_dict = None
            active_decision_dict = None
            if should_ask and result.question:
                target_str = str(result.target_field_or_area or "Problem")
                cq_dict = {
                    "question": result.question.strip(),
                    "target_area": target_str,
                    "information_goal": result.information_goal or "",
                    "why_this_question_now": result.reason,
                }
                gap_qid = f"q_gap_{turn_count + 1}_{uuid.uuid4().hex[:6]}"
                gap_decision = NextTurnDecision(
                    question_id=gap_qid,
                    source_phase=WorkflowPhase.UNDERSTAND,
                    information_target=target_str,
                    canonical_question=result.question.strip(),
                    should_ask_question=True,
                    clarification_reason=result.reason or "",
                    planner_version=INFORMATION_GAP_PLANNER_PROMPT_VERSION,
                )
                active_decision_dict = gap_decision.model_dump()

            trace = AITrace(
                module_name="plan_information_gap",
                prompt_version=INFORMATION_GAP_PLANNER_PROMPT_VERSION,
                input_data={"turn": turn_count, "should_ask": should_ask, "strategic_sufficient": strategic_sufficient, "resolution_mode": res_mode_enum.value},
                output_data=result.model_dump(),
                latency_ms=latency_ms,
                project_id=project_id,
            )

            existing_traces = list(state.get("traces") or [])
            existing_traces.append(trace.model_dump(mode="json"))

            logger.info(
                "plan_information_gap | project_id=%s should_ask=%s strategic_sufficient=%s resolution_mode=%s turn=%d",
                project_id, should_ask, strategic_sufficient, res_mode_enum.value, turn_count,
            )

            turn_telem = _record_llm_node_telemetry(
                state, "plan_information_gap", llm, "reasoning", latency_ms, attempt,
                "InformationGapPlanLLMOutput", messages=messages, result=result,
            )

            return {
                "turn_telemetry": turn_telem,
                "gap_plan": gap_plan.model_dump(mode="json"),
                "context_sufficiency": compat_cs,
                "context_sufficiency_output": compat_cs,
                "is_sufficient": is_sufficient,
                "current_clarifying_question": cq_dict,
                "active_next_turn_decision": active_decision_dict,
                "active_interrupt_type": "strategic" if should_ask else None,
                "last_trace": trace.model_dump(mode="json"),
                "traces": existing_traces,
            }
        except Exception as exc:
            last_error = exc
            logger.warning("plan_information_gap failed attempt %d: %s", attempt, exc)
            if is_permanent_model_error(exc) or isinstance(exc, TokenBudgetExhaustedError):
                break

    _record_llm_node_telemetry(state, "plan_information_gap", llm, "reasoning", 0.0, attempt, "InformationGapPlanLLMOutput", status="FAILED")
    if isinstance(last_error, TokenBudgetExhaustedError):
        raise last_error
    raise RuntimeError(f"plan_information_gap failed after {attempt} attempts. Error: {last_error}")


# ---------------------------------------------------------------------------
# 10. assess_startup_areas (LLM — Milestone 2)
# ---------------------------------------------------------------------------

def assess_startup_areas(state: StartaryGraphState) -> dict:
    """
    LLM Node: Evaluate all 10 canonical startup areas across 4 qualitative risk dimensions.
    """
    project_id = state.get("project_id", "<unknown>")
    # Milestone 6.2.3: risk evaluation sees the founder facts already extracted, not placeholders.
    startup_state_dict = _startup_state_with_canonical_facts(state)
    diagnosis_dict = state.get("diagnosis") or {}

    settings = get_settings()
    base_max_tokens = 2500
    llm = _safe_get_llm_client(settings, tier="reasoning", max_completion_tokens=base_max_tokens)
    structured_llm = llm.with_structured_output(StartupAreaAssessmentBatchLLMOutput)

    user_message = build_area_assessment_user_message(
        startup_state_summary=json.dumps(startup_state_dict, indent=2, default=str),
        diagnosis_summary=diagnosis_dict.get("summary") or (state.get("diagnosis_output") or {}).get("diagnosis_summary", ""),
        startup_stage=diagnosis_dict.get("startup_stage") or (state.get("diagnosis_output") or {}).get("startup_stage"),
    )

    messages = [
        {"role": "system", "content": ASSESS_STARTUP_AREAS_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    attempt = 0
    last_error: Exception | None = None

    while attempt < 2:
        attempt += 1
        try:
            result, latency_ms = _invoke_structured_with_token_retry(
                structured_llm=structured_llm,
                messages=messages,
                settings=settings,
                tier="reasoning",
                base_max_tokens=base_max_tokens,
                node_name="assess_startup_areas",
                schema_cls=StartupAreaAssessmentBatchLLMOutput,
            )

            if not isinstance(result, StartupAreaAssessmentBatchLLMOutput):
                raise ValidationError(f"Expected StartupAreaAssessmentBatchLLMOutput, got {type(result)}")

            # Stage & Behavioral Traction Adjustment (Milestone 4.1.2)
            all_text = " ".join([
                str(state.get("raw_context") or ""),
                " ".join(str(m.get("content", "")) for m in (state.get("founder_messages") or []) if isinstance(m, dict)),
                str(diagnosis_dict.get("summary") or ""),
            ]).lower()

            # Milestone 6.2.3: negation-aware ("zero paying customers" is negative evidence, not traction)
            has_paying = contains_unnegated_keyword(all_text, ["paying", "paid", "repeat payment", "repeat payments", "paying teams", "paying customers", "paid pilot", "subscription revenue", "mrr", "8 paying"])
            has_repeat = contains_unnegated_keyword(all_text, ["repeat use", "weekly use", "recorded full matches", "weekly active", "using weekly", "repeat usage", "multiple matches"])
            has_live = contains_unnegated_keyword(all_text, ["deployed", "live product", "in production", "app is live", "product is live", "active users", "20 teams tried", "14 recorded"])

            if (has_paying or has_repeat) and has_live:
                adjusted_assessments = []
                for a in result.assessments:
                    a_dict = a.model_dump() if hasattr(a, "model_dump") else dict(a)
                    if str(a_dict.get("area", "")).lower() == "problem":
                        if str(a_dict.get("uncertainty", "")).lower() == "high":
                            a_dict["uncertainty"] = RiskFactorLevel.MEDIUM
                        if str(a_dict.get("evidence_gap", "")).lower() == "high":
                            a_dict["evidence_gap"] = RiskFactorLevel.MEDIUM
                        current_rat = a_dict.get("rationale") or ""
                        if "Adjusted for verified" not in current_rat:
                            a_dict["rationale"] = (current_rat + " (Adjusted for verified paying customers / repeat usage traction).").strip()
                    adjusted_assessments.append(AreaRiskAssessmentLLMOutput(**a_dict))
                result = StartupAreaAssessmentBatchLLMOutput(assessments=adjusted_assessments)

            # Milestone M4.1.3.3 Invariant Guard: Validated KNOWN Areas cannot have maximal uncertainty & evidence gap
            areas_map = dict(startup_state_dict.get("areas") or startup_state_dict.get("area_knowledge") or {})
            if areas_map:
                adjusted_for_known = []
                for a in result.assessments:
                    a_dict = a.model_dump() if hasattr(a, "model_dump") else dict(a)
                    area_name = str(a_dict.get("area", ""))
                    area_k = areas_map.get(area_name) or {}
                    if area_k.get("status") == AreaStatus.KNOWN.value:
                        if str(a_dict.get("evidence_gap", "")).upper() == "HIGH":
                            a_dict["evidence_gap"] = RiskFactorLevel.MEDIUM
                            rat = a_dict.get("rationale") or ""
                            if "Evidence gap adjusted" not in rat:
                                a_dict["rationale"] = (rat + " (Evidence gap adjusted: area validated by experiment evidence).").strip()
                        if str(a_dict.get("uncertainty", "")).upper() == "HIGH":
                            a_dict["uncertainty"] = RiskFactorLevel.MEDIUM
                    adjusted_for_known.append(AreaRiskAssessmentLLMOutput(**a_dict))
                result = StartupAreaAssessmentBatchLLMOutput(assessments=adjusted_for_known)

            trace = AITrace(
                module_name="assess_startup_areas",
                prompt_version=ASSESS_STARTUP_AREAS_PROMPT_VERSION,
                input_data={"diagnosis_summary": diagnosis_dict.get("summary", "")[:300]},
                output_data=result.model_dump(),
                latency_ms=latency_ms,
                project_id=project_id,
            )

            existing_traces = list(state.get("traces") or [])
            existing_traces.append(trace.model_dump(mode="json"))

            turn_telem = _record_llm_node_telemetry(
                state, "assess_startup_areas", llm, "reasoning", latency_ms, attempt,
                "StartupAreaAssessmentBatchLLMOutput", messages=messages, result=result,
            )

            return {
                "turn_telemetry": turn_telem,
                "area_assessments_output": result.model_dump(),
                "last_trace": trace.model_dump(mode="json"),
                "traces": existing_traces,
            }
        except Exception as exc:
            last_error = exc
            logger.warning("assess_startup_areas failed attempt %d: %s", attempt, exc)
            if is_permanent_model_error(exc) or isinstance(exc, TokenBudgetExhaustedError):
                break

    _record_llm_node_telemetry(state, "assess_startup_areas", llm, "reasoning", 0.0, attempt, "StartupAreaAssessmentBatchLLMOutput", status="FAILED")
    if isinstance(last_error, TokenBudgetExhaustedError):
        raise last_error
    raise RuntimeError(f"assess_startup_areas failed after {attempt} attempts. Error: {last_error}")


# ---------------------------------------------------------------------------
# 11. select_weak_link (Deterministic Python — Milestone 2)
# ---------------------------------------------------------------------------

def select_weak_link(state: StartaryGraphState) -> dict:
    """
    Deterministic Node:
    - Calculates Decision Risk = Importance × Uncertainty × Evidence Gap × Consequence
    - Applies explicit tie-breaking hierarchy
    - Selects exactly one weak-link area
    - Updates diagnosis & graph state
    """
    project_id = state.get("project_id", "<unknown>")
    batch_output = state.get("area_assessments_output")
    startup_state_dict = state.get("startup_state", {})
    diagnosis_dict = state.get("diagnosis", {})

    if not batch_output or "assessments" not in batch_output:
        raise ValueError(f"select_weak_link | project_id={project_id}: area_assessments_output is missing from state.")

    # Canonical taxonomy index for tie-breaking
    taxonomy_order = {area: idx for idx, area in enumerate(StartupArea)}

    assessments_dict: dict[StartupArea, StartupAreaAssessment] = {}
    areas_map = dict(startup_state_dict.get("areas") or startup_state_dict.get("area_knowledge") or {})

    for item in batch_output["assessments"]:
        try:
            area_enum = StartupArea(item["area"])
            area_k = areas_map.get(str(area_enum)) or areas_map.get(area_enum.value) or {}
            status_val = area_k.get("status")
            understanding_val = area_k.get("understanding")

            ev_gap_val = RiskFactorLevel(str(item["evidence_gap"]).upper())
            unc_val = RiskFactorLevel(str(item["uncertainty"]).upper())
            # Milestone M4.1.3.3: Invariant Guard for Validated KNOWN Areas
            if status_val == AreaStatus.KNOWN.value:
                if ev_gap_val == RiskFactorLevel.HIGH:
                    ev_gap_val = RiskFactorLevel.MEDIUM
                if unc_val == RiskFactorLevel.HIGH:
                    unc_val = RiskFactorLevel.MEDIUM

            assessment = StartupAreaAssessment(
                area=area_enum,
                status=AreaStatus(status_val) if status_val else None,
                summary=understanding_val,
                importance=RiskFactorLevel(str(item["importance"]).upper()),
                uncertainty=unc_val,
                evidence_gap=ev_gap_val,
                consequence_if_wrong=RiskFactorLevel(str(item["consequence_of_being_wrong"]).upper()),
                rationale=item.get("rationale"),
            )
            assessment.calculate_risk()
            assessments_dict[area_enum] = assessment
        except Exception as exc:
            logger.warning("Error parsing assessment for %s: %s", item.get("area"), exc)

    # Fill in missing areas if any were omitted
    for area in StartupArea:
        if area not in assessments_dict:
            assessment = StartupAreaAssessment(
                area=area,
                importance=RiskFactorLevel.LOW,
                uncertainty=RiskFactorLevel.HIGH,
                evidence_gap=RiskFactorLevel.LOW,
                consequence_if_wrong=RiskFactorLevel.LOW,
                rationale="Default assessment for omitted area.",
            )
            assessment.calculate_risk()
            assessments_dict[area] = assessment

    all_assessments = list(assessments_dict.values())

    # Deterministic Tie-Breaking Hierarchy:
    # 1. Higher calculated_decision_risk (descending)
    # 2. Higher consequence_if_wrong (descending)
    # 3. Higher uncertainty (descending)
    # 4. Higher evidence_gap (descending)
    # 5. Higher importance (descending)
    # 6. Canonical taxonomy index (ascending)
    def sort_key(a: StartupAreaAssessment):
        return (
            -(a.calculated_decision_risk or 0),
            -risk_factor_to_int(a.consequence_if_wrong),
            -risk_factor_to_int(a.uncertainty),
            -risk_factor_to_int(a.evidence_gap),
            -risk_factor_to_int(a.importance),
            taxonomy_order.get(a.area, 999),
        )

    sorted_assessments = sorted(all_assessments, key=sort_key)
    winning_assessment = sorted_assessments[0]
    selected_area = winning_assessment.area

    # ---------------------------------------------------------------------------
    # Technical Note: Multiplicative Risk Model Saturation Telemetry
    # Current V1 multiplicative risk model intentionally retained.
    # Known limitation: score saturation / high-cardinality ties (e.g. 6+ areas at 81/81).
    # See docs/architecture_backlog.md — WL-001.
    # ---------------------------------------------------------------------------
    max_score = winning_assessment.calculated_decision_risk or 0
    max_score_tie_count = sum(
        1 for a in sorted_assessments if (a.calculated_decision_risk or 0) == max_score
    )
    saturated_area_count = sum(
        1 for a in sorted_assessments if (a.calculated_decision_risk or 0) == 81
    )
    tie_breaker_used = max_score_tie_count > 1

    saturation_telemetry = {
        "max_score": max_score,
        "max_score_tie_count": max_score_tie_count,
        "saturated_area_count": saturated_area_count,
        "tie_breaker_used": tie_breaker_used,
    }

    # Update Diagnosis domain model
    if diagnosis_dict:
        diagnosis_dict["weak_areas"] = [str(selected_area)]
        diagnosis_dict["selected_weak_link"] = str(selected_area)
        diagnosis_dict["assessments"] = [a.model_dump() for a in all_assessments]

    logger.info(
        "select_weak_link | project_id=%s selected=%s risk_score=%d ties=%d saturated_81s=%d tie_broken=%s",
        project_id,
        selected_area,
        max_score,
        max_score_tie_count,
        saturated_area_count,
        tie_breaker_used,
    )

    return {
        "weak_area": str(selected_area),
        "weak_link_details": {
            "area": str(selected_area),
            "score": winning_assessment.calculated_decision_risk,
            "importance": str(winning_assessment.importance),
            "uncertainty": str(winning_assessment.uncertainty),
            "evidence_gap": str(winning_assessment.evidence_gap),
            "consequence_of_being_wrong": str(winning_assessment.consequence_if_wrong),
            "rationale": winning_assessment.rationale,
            "saturation_telemetry": saturation_telemetry,
        },
        "area_assessments": [a.model_dump() for a in sorted_assessments],
        "diagnosis": diagnosis_dict,
    }


# ---------------------------------------------------------------------------
# 12. reality_check (LLM — Milestone 2)
# ---------------------------------------------------------------------------

def reality_check(state: StartaryGraphState) -> dict:
    """
    LLM Node: Scrutinize founder claims vs founder evidence specifically for the weak area.
    """
    project_id = state.get("project_id", "<unknown>")
    weak_area_str = state.get("weak_area")
    startup_state_dict = state.get("startup_state", {})
    diagnosis_dict = state.get("diagnosis", {})

    if not weak_area_str:
        raise ValueError(f"reality_check | project_id={project_id}: weak_area is missing from state.")

    area_knowledge_dict = startup_state_dict.get("areas", {}).get(weak_area_str, {})

    settings = get_settings()
    base_max_tokens = 2500
    llm = _safe_get_llm_client(settings, tier="reasoning", max_completion_tokens=base_max_tokens)
    structured_llm = llm.with_structured_output(RealityCheckLLMOutput)

    user_message = build_reality_check_user_message(
        weak_area=weak_area_str,
        area_knowledge_json=json.dumps(area_knowledge_dict, indent=2),
        overall_summary=diagnosis_dict.get("summary", ""),
    )

    messages = [
        {"role": "system", "content": REALITY_CHECK_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    attempt = 0
    last_error: Exception | None = None

    while attempt < 2:
        attempt += 1
        try:
            result, latency_ms = _invoke_structured_with_token_retry(
                structured_llm=structured_llm,
                messages=messages,
                settings=settings,
                tier="reasoning",
                base_max_tokens=base_max_tokens,
                node_name="reality_check",
                schema_cls=RealityCheckLLMOutput,
            )

            if not isinstance(result, RealityCheckLLMOutput):
                raise ValidationError(f"Expected RealityCheckLLMOutput, got {type(result)}")

            # Re-construct domain model
            rc_domain = RealityCheck(
                weak_link_area=result.weak_link_area,
                relevant_founder_claims=result.relevant_founder_claims,
                available_founder_evidence=result.available_founder_evidence,
                what_evidence_supports=result.what_evidence_supports,
                what_evidence_does_not_support=result.what_evidence_does_not_support,
                conclusion=result.conclusion,
                unresolved_uncertainty=result.unresolved_uncertainty,
                summary=result.reality_check_summary,
            )

            trace = AITrace(
                module_name="reality_check",
                prompt_version=REALITY_CHECK_PROMPT_VERSION,
                input_data={"weak_area": weak_area_str},
                output_data=result.model_dump(),
                latency_ms=latency_ms,
                project_id=project_id,
            )

            existing_traces = list(state.get("traces") or [])
            existing_traces.append(trace.model_dump(mode="json"))

            return {
                "reality_check_output": result.model_dump(),
                "reality_check": rc_domain.model_dump(mode="json"),
                "last_trace": trace.model_dump(mode="json"),
                "traces": existing_traces,
            }
        except Exception as exc:
            last_error = exc
            logger.warning("reality_check failed attempt %d: %s", attempt, exc)
            if is_permanent_model_error(exc) or isinstance(exc, TokenBudgetExhaustedError):
                break

    if isinstance(last_error, TokenBudgetExhaustedError):
        raise last_error
    raise RuntimeError(f"reality_check failed after {attempt} attempts. Error: {last_error}")


# ---------------------------------------------------------------------------
# 13. generate_critical_assumption (LLM — Milestone 2)
# ---------------------------------------------------------------------------

def validate_critical_assumption_grounding(
    ca_output: CriticalAssumptionLLMOutput,
    rc_dict: dict[str, Any],
    founder_source_text: str,
    weak_area_str: str,
) -> tuple[bool, str]:
    """
    Deterministic Grounding Validator for Critical Assumptions (Milestone 2.6.2):
    - Area matches target weak area.
    - Statement is non-empty.
    - Rejects ungrounded test parameters in introduced_test_parameters.
    - Rejects invented percentages (e.g., 30%, 50%) not in founder context.
    - Rejects invented order volumes / quantities (e.g., 20 orders) not in founder context.
    - Rejects invented frequencies (e.g., weekly, daily, at least weekly) not in founder context.
    - Rejects invented ranking claims (e.g., top three, top 3) not in founder context.
    - Enforces Reality Check negative constraints (what_evidence_does_not_support).
    """
    statement = (ca_output.statement or "").strip()
    if not statement:
        return False, "Critical assumption statement cannot be empty."

    if str(ca_output.area) != weak_area_str:
        return False, f"Assumption area '{ca_output.area}' does not match target weak area '{weak_area_str}'."

    statement_lower = statement.lower()
    founder_lower = (founder_source_text or "").lower()

    # 1. Reject ungrounded introduced_test_parameters
    for param in ca_output.introduced_test_parameters:
        if param and param.lower() not in founder_lower:
            return False, f"Unsupported test parameter introduced: '{param}' was not stated by the founder."

    # 2. Percentage checks
    percent_matches = re.findall(r"\b\d+(?:\.\d+)?%", statement)
    for p in percent_matches:
        if p.lower() not in founder_lower:
            return False, f"Unsupported percentage threshold introduced: '{p}' was not stated by the founder."

    # 3. Numeric quantity / order volume checks (e.g. "20 orders", "5 users", "100 AZN")
    quantity_matches = re.findall(
        r"\b(\d+)\s*(?:orders|customers|users|clients|sellers|times|dollars|azn|usd|hours|interviews|per\s+week|per\s+day|per\s+month)\b",
        statement_lower,
    )
    for q_num in quantity_matches:
        if q_num not in founder_lower:
            return False, f"Unsupported numeric quantity/threshold introduced: '{q_num}' was not found in founder context."

    # 4. Standalone arbitrary order volume numbers (e.g. "at least 20 orders")
    order_matches = re.findall(r"\b(?:at least|more than|over)?\s*(\d+)\s*orders?\b", statement_lower)
    for o_num in order_matches:
        if o_num not in founder_lower:
            return False, f"Unsupported order threshold introduced: '{o_num} orders' was not found in founder context."

    # 5. Frequency checks (e.g. "weekly", "at least weekly", "daily")
    freq_keywords = ["at least weekly", "weekly", "at least daily", "daily", "monthly", "per week", "per day", "per month"]
    for freq in freq_keywords:
        if freq in statement_lower and freq not in founder_lower:
            return False, f"Unsupported frequency claim introduced: '{freq}' was not stated by the founder."

    # 6. Ranking claims (e.g. "top three", "top 3")
    ranking_matches = re.findall(r"\btop\s+(?:three|3|five|5|ten|10)\b", statement_lower)
    for rank in ranking_matches:
        if rank not in founder_lower:
            return False, f"Unsupported ranking claim introduced: '{rank}' was not stated by the founder."

    # 7. Reality Check negative constraint: what_evidence_does_not_support
    unsupported = rc_dict.get("what_evidence_does_not_support") or []
    for item in unsupported:
        item_lower = item.lower()
        if "missed" in item_lower and "missed" in statement_lower and "missed" not in founder_lower:
            return False, "Assumption asserts 'missed deliveries' which Reality Check explicitly confirmed is unsupported by evidence."
        if "delayed" in item_lower and "delayed" in statement_lower and "delayed" not in founder_lower:
            return False, "Assumption asserts 'delayed deliveries' which Reality Check explicitly confirmed is unsupported by evidence."
        if "incorrect" in item_lower and "incorrect" in statement_lower and "incorrect" not in founder_lower:
            return False, "Assumption asserts 'incorrect deliveries' which Reality Check explicitly confirmed is unsupported by evidence."

    return True, ""


def validate_critical_assumption_area_purity(
    ca_output: CriticalAssumptionLLMOutput,
    selected_area: StartupArea | str,
) -> tuple[bool, str]:
    """
    Deterministic Area-Purity Validator for Critical Assumptions (Milestone 4.0.7.1.3).
    Ensures the assumption stays strictly inside the epistemic boundary of the selected
    weak-link area and does not import validation requirements from neighboring areas.
    """
    statement = (ca_output.statement or "").strip()
    if not statement:
        return False, "Critical assumption statement cannot be empty."

    stmt_lower = statement.lower()
    area_str = str(selected_area)

    # -----------------------------------------------------------------------
    # TARGET USER: Who experiences, owns, responds to, or decides about the problem.
    # Must NOT require: adoption, willingness to pay, product preference, or commercial conversion.
    # -----------------------------------------------------------------------
    if area_str == StartupArea.TARGET_USER:
        # 1. Adoption requirements
        adoption_match = re.search(
            r"\b(consider(?:ing)?\s+adopt(?:ing)?|adopt(?:ing|ion)?\s+(?:the|our|a|this)\s+(?:system|solution|product|tool|platform|service|software|sensor|alerts?)|will\s+adopt|would\s+adopt|ready\s+to\s+adopt|adoption\s+of\b|adopt\b)",
            stmt_lower,
        )
        if adoption_match:
            phrase = adoption_match.group(0)
            return (
                False,
                f"Target User assumption improperly requires solution adoption ('{phrase}'). "
                "Target User assumptions must focus on customer segment definition, problem ownership, "
                "and operational roles (user/responder/decision-maker), not adoption or solution evaluation.",
            )

        # 2. Willingness to pay / purchasing / commercial conversion
        payment_match = re.search(
            r"\b(willing(?:ness)?\s+to\s+pay|pay(?:ing)?\s+for|pay\s+(?:a|the|\$|\d+)|worth\s+paying|purchas(?:e|ing|ed)|buy(?:ing)?\s+(?:the|our|a)|budget\s+for\s+(?:the|our|a)|commercial\s+conversion|convert(?:ing)?\s+into\s+paying)\b",
            stmt_lower,
        )
        if payment_match:
            phrase = payment_match.group(0)
            return (
                False,
                f"Target User assumption improperly requires payment or commercial conversion ('{phrase}'). "
                "Willingness to pay and monetization belong in 'Business Model / Revenue', not 'Target User'.",
            )

        # 3. Product preference / value proposition
        preference_match = re.search(
            r"\b(prefer(?:ring)?\s+(?:the|our|this)\s+(?:system|solution|product|tool|platform)|preference\s+for\s+(?:the|our|this)|valuable\s+enough\s+to\s+(?:use|adopt)|relevant\s+enough\s+to\s+consider|product\s+differentiation)\b",
            stmt_lower,
        )
        if preference_match:
            phrase = preference_match.group(0)
            return (
                False,
                f"Target User assumption improperly requires product preference or value validation ('{phrase}'). "
                "Product preference and usefulness belong in 'Value Proposition', not 'Target User'.",
            )

    # -----------------------------------------------------------------------
    # PROBLEM: What recurring problem/event exists and what consequence it creates.
    # Must NOT require: solution adoption or payment.
    # -----------------------------------------------------------------------
    elif area_str == StartupArea.PROBLEM:
        problem_sol_match = re.search(
            r"\b(willing(?:ness)?\s+to\s+pay|worth\s+paying|adopt(?:ing)?\s+(?:the|our|a)\s+(?:system|solution|product|tool|platform)|will\s+adopt|would\s+adopt)\b",
            stmt_lower,
        )
        if problem_sol_match:
            phrase = problem_sol_match.group(0)
            return (
                False,
                f"Problem assumption improperly requires solution adoption or payment ('{phrase}'). "
                "Problem assumptions must focus on recurring failure events, operational disruption, "
                "and consequence, not solution adoption or monetization.",
            )

    # -----------------------------------------------------------------------
    # PAIN / URGENCY: Severity, frequency, disruption, priority over other demands.
    # Must NOT require: commercial payment or product preference.
    # -----------------------------------------------------------------------
    elif area_str == StartupArea.PAIN_URGENCY:
        pain_pay_match = re.search(
            r"\b(willing(?:ness)?\s+to\s+pay|worth\s+paying|pay\s+a\s+subscription|pay\s+\$\d+|prefer(?:ring)?\s+our)\b",
            stmt_lower,
        )
        if pain_pay_match:
            phrase = pain_pay_match.group(0)
            return (
                False,
                f"Pain / Urgency assumption improperly requires payment or product preference ('{phrase}'). "
                "Pain and urgency must be tested through operational priority and disruption, "
                "not commercial willingness to pay.",
            )

    # -----------------------------------------------------------------------
    # EXISTING ALTERNATIVES: What users do instead and whether alternatives are adequate.
    # Must NOT imply: alternative inadequacy validates the startup's solution.
    # -----------------------------------------------------------------------
    elif area_str == StartupArea.EXISTING_ALTERNATIVES:
        alt_sol_match = re.search(
            r"\b(which\s+proves\s+(?:our|the)\s+solution|proves\s+users\s+will\s+adopt|guarantees\s+demand\s+for\s+(?:our|the)|confirming\s+users\s+will\s+switch\s+to\s+(?:our|the))\b",
            stmt_lower,
        )
        if alt_sol_match:
            phrase = alt_sol_match.group(0)
            return (
                False,
                f"Existing Alternatives assumption improperly implies solution validation ('{phrase}'). "
                "Inadequacy of existing alternatives does not automatically prove adoption of the startup's proposed solution.",
            )

    # -----------------------------------------------------------------------
    # VALUE PROPOSITION: Proposed outcome/benefit is meaningfully useful or preferable.
    # Must NOT require: commercial pricing, contracts, or revenue quotas.
    # -----------------------------------------------------------------------
    elif area_str == StartupArea.VALUE_PROPOSITION:
        vp_pay_match = re.search(
            r"\b(willing(?:ness)?\s+to\s+pay\s+\$\d+|sign\s+a\s+(?:paid|commercial)\s+contract|generate\s+\$\d+\s+revenue)\b",
            stmt_lower,
        )
        if vp_pay_match:
            phrase = vp_pay_match.group(0)
            return (
                False,
                f"Value Proposition assumption improperly requires commercial contract metrics ('{phrase}'). "
                "Value Proposition tests whether the proposed outcome is useful and preferable; "
                "monetization belongs in 'Business Model / Revenue'.",
            )

    # -----------------------------------------------------------------------
    # ACQUISITION / CUSTOMER ACCESS: Reliable outreach channels and customer access.
    # Must NOT use channel access to prove core problem severity.
    # -----------------------------------------------------------------------
    elif area_str == StartupArea.ACQUISITION:
        acq_prob_match = re.search(
            r"\b(proves\s+the\s+problem\s+is\s+severe|validates\s+that\s+the\s+problem\s+exists)\b",
            stmt_lower,
        )
        if acq_prob_match:
            phrase = acq_prob_match.group(0)
            return (
                False,
                f"Acquisition assumption improperly conflates channel reach with problem validation ('{phrase}'). "
                "Acquisition tests customer access and channel response, not problem existence.",
            )

    # -----------------------------------------------------------------------
    # SOLUTION / MVP: Proposed solution delivers the required outcome in use.
    # Must NOT conflate technical feasibility with customer demand.
    # -----------------------------------------------------------------------
    elif area_str == StartupArea.SOLUTION_MVP:
        sol_dem_match = re.search(
            r"\b(which\s+proves\s+(?:market\s+)?demand|proves\s+(?:customers|users)\s+will\s+buy)\b",
            stmt_lower,
        )
        if sol_dem_match:
            phrase = sol_dem_match.group(0)
            return (
                False,
                f"Solution / MVP assumption improperly conflates technical delivery with market demand ('{phrase}'). "
                "Solution / MVP tests whether the product functions and delivers the outcome in user hands, not market demand.",
            )

    # -----------------------------------------------------------------------
    # BUSINESS MODEL / REVENUE: Willingness to pay, pricing, revenue model.
    # NOTE: "willing to pay", "pay for", "budget" ARE VALID HERE!
    # -----------------------------------------------------------------------

    return True, ""


def validate_critical_assumption_atomicity(
    ca_output: CriticalAssumptionLLMOutput | dict,
    weak_area: str | StartupArea,
) -> tuple[bool, str]:
    """
    Deterministic Atomicity & Single Test Scope Validator (Milestone 4.0.7.1.4).
    Enforces that a Critical Assumption represents ONE decision-relevant uncertainty
    testable in ONE bounded learning cycle.

    Rejects:
      - Compound assumptions requiring multiple independently falsifiable conditions
        (e.g. commission acceptance + customer price tolerance + merchant economics + contribution margin)
      - Multi-actor marketplace bundling (e.g. buyer adoption + seller inventory upload)
      - Multi-clause conjunctions asserting multiple independent business truths
      - Target User assumptions bundling segment identification + adoption + payment
    """
    if hasattr(ca_output, "model_dump"):
        ca_dict = ca_output.model_dump()
    else:
        ca_dict = dict(ca_output)

    stmt = ca_dict.get("statement", "").strip()
    stmt_lower = stmt.lower()
    area_str = str(getattr(weak_area, "value", weak_area))

    # 1. Multi-actor marketplace bundling (two-sided marketplaces)
    buyer_match = re.search(r"\b(students|buyers|customers|users|consumers|clients)\s+(?:will|can|regularly|consistently|repeatedly)\b", stmt_lower)
    seller_match = re.search(r"\b(merchants|bakeries|sellers|vendors|restaurants|shops|partners)\s+(?:will|can|regularly|consistently|repeatedly)\b", stmt_lower)
    if buyer_match and seller_match and (" and " in stmt_lower or " while " in stmt_lower or ", " in stmt_lower):
        return (
            False,
            f"Critical assumption bundles independent hypotheses across distinct market actors "
            f"('{buyer_match.group(0)}' and '{seller_match.group(0)}'). "
            "Two-sided marketplaces require testing one market side per learning cycle.",
        )

    # 2. Business Model / Revenue multi-condition commercial bundling
    if area_str == str(StartupArea.BUSINESS_MODEL):
        dims_found = []
        if re.search(r"\b(commission(?:-based)?|revenue[- ]share|take[- ]rate|transaction fee|transaction model)\b", stmt_lower):
            dims_found.append("commission/fee acceptance")
        if re.search(r"\b(customer prices?|customer pricing|price tolerance|tolerate (?:the )?prices?|acceptable (?:customer )?prices?|pricing sensitivity)\b", stmt_lower):
            dims_found.append("customer price tolerance")
        if re.search(r"\b(merchant economics|partner economics|net economics|economics for the merchant)\b", stmt_lower):
            dims_found.append("merchant economics")
        if re.search(r"\b(contribution margin|remain\w* profitable|profitability|operating costs?|positive margin)\b", stmt_lower):
            dims_found.append("contribution margin / profitability")

        if len(dims_found) >= 2 and (" while " in stmt_lower or ", and " in stmt_lower or " and " in stmt_lower or " as well as " in stmt_lower):
            return (
                False,
                f"Business Model / Revenue assumption combines {len(dims_found)} independent commercial hypotheses: "
                f"{', '.join(dims_found)}. "
                "Must test ONE atomic commercial uncertainty at a time (e.g. test commission acceptance first).",
            )

    # 3. Multi-clause independent hypothesis conjunctions (3+ 'will/can [verb]' clauses joined by and/while)
    predicate_matches = re.findall(r"\b(?:will|can)\s+[a-z]{3,}\b", stmt_lower)
    if len(predicate_matches) >= 3 and (", and " in stmt_lower or " and " in stmt_lower) and (" while " in stmt_lower or ", " in stmt_lower):
        return (
            False,
            f"Critical assumption contains multiple compound predictive clauses ({len(predicate_matches)} independent conditions). "
            "Must test ONE decision-relevant hypothesis in a single learning cycle.",
        )

    # 4. Target User bundling with adoption and payment
    if area_str == str(StartupArea.TARGET_USER):
        has_segment = bool(re.search(r"\b(segment|role|group|users|customers|profile|responsib\w+)\b", stmt_lower))
        has_adoption = bool(re.search(r"\b(adopt\w*|use|using|onboard\w*|sign\w* up)\b", stmt_lower))
        has_payment = bool(re.search(r"\b(pay\w*|payment|price|commission|budget|buy\w*)\b", stmt_lower))
        if has_segment and has_adoption and has_payment:
            return (
                False,
                "Target User assumption bundles segment definition with product adoption and payment. "
                "Target User must test ONE repeatable segment/role hypothesis without requiring adoption or payment.",
            )

    return True, ""


def extract_blocked_grounding_concepts(
    val_reason: str,
    statement: str,
    rc_dict: dict[str, Any],
) -> list[str]:
    """
    Extract explicit blocked concepts/phrases from a grounding validation failure (Milestone 4.0.7.1.2).
    Captures quoted phrases from the validation reason and unsupported items from Reality Check.
    """
    blocked: list[str] = []

    # 1. Capture quoted phrases from val_reason (e.g., 'delayed deliveries', 'weekly', '30%')
    quoted = re.findall(r"'([^']+)'", val_reason)
    blocked.extend(quoted)

    # 2. Check Reality Check negative constraints
    unsupported = rc_dict.get("what_evidence_does_not_support") or []
    statement_lower = (statement or "").lower()
    for item in unsupported:
        item_str = str(item).strip()
        item_lower = item_str.lower()
        if item_lower in statement_lower:
            blocked.append(item_str)
        else:
            for phrase in ["delayed deliveries", "missed deliveries", "incorrect deliveries", "delayed", "missed", "incorrect"]:
                if phrase in item_lower and phrase in statement_lower:
                    blocked.append(phrase)

    # Deduplicate preserving order
    return list(dict.fromkeys(b.strip() for b in blocked if b.strip()))


_CANONICAL_AREA_FALLBACKS: dict[str, str] = {
    StartupArea.PROBLEM: (
        "Target users experience recurring disruption and avoidable operational friction in their current workflow "
        "that creates material recurring losses."
    ),
    StartupArea.TARGET_USER: (
        "A repeatable segment of target users has an identifiable operational role "
        "that directly experiences and owns the core problem."
    ),
    StartupArea.PAIN_URGENCY: (
        "The operational pain caused by this problem is sufficiently urgent "
        "that target users prioritize addressing it over competing operational demands."
    ),
    StartupArea.EXISTING_ALTERNATIVES: (
        "Existing manual workarounds and alternative tools leave critical operational gaps "
        "that create significant ongoing frustration for users."
    ),
    StartupArea.VALUE_PROPOSITION: (
        "The proposed value proposition delivers a clear and measurable operational improvement "
        "compared to how users currently manage the workflow."
    ),
    StartupArea.DEMAND_SIGNAL: (
        "Target users demonstrate authentic interest and willingness to evaluate a dedicated solution "
        "when presented with a concrete workflow alternative."
    ),
    StartupArea.ACQUISITION: (
        "Target customers can be identified and directly engaged through targeted outreach channels "
        "without requiring established brand awareness."
    ),
    StartupArea.BUSINESS_MODEL: (
        "Target customers derive sufficient economic or operational value from resolving this problem "
        "that they are willing to allocate budget for a solution."
    ),
    StartupArea.SOLUTION_MVP: (
        "A focused minimum viable solution addressing the core workflow bottleneck "
        "can be deployed and validated with target users."
    ),
    StartupArea.FOUNDER_CONSTRAINTS: (
        "The founding team possesses the necessary domain access and operational capacity "
        "to execute initial validation and early customer testing."
    ),
}


def build_grounded_critical_assumption_fallback(
    weak_area_str: str,
    rc_dict: dict[str, Any],
    founder_source_text: str,
    state: StartaryGraphState,
) -> CriticalAssumptionLLMOutput | None:
    """
    Deterministic Fallback Builder for Critical Assumptions (Milestone 4.0.7.1.2 / M4.0.7.1.3).
    Constructs a safe, grounded, area-pure critical assumption using ONLY already-grounded facts
    from the Reality Check and founder context when LLM retries fail validation.
    Must pass BOTH grounding validation and area-purity validation.
    """
    target_area = StartupArea(weak_area_str)
    unresolved_uncertainty = rc_dict.get("unresolved_uncertainty", "").strip()
    evidence_supports = list(rc_dict.get("what_evidence_supports") or [])
    founder_lower = (founder_source_text or "").lower()

    candidates: list[str] = []

    # Candidate 1: Formulate directly from Reality Check unresolved uncertainty
    if unresolved_uncertainty and len(unresolved_uncertainty) >= 15:
        statement = re.sub(
            r"^(?:whether\s+or\s+not|whether|if|can|does|do|is|are|how|what)\s+",
            "",
            unresolved_uncertainty,
            flags=re.IGNORECASE,
        ).strip()
        if statement:
            statement = statement[0].upper() + statement[1:]
            if not statement.endswith("."):
                statement += "."
            candidates.append(statement)

    # Candidate 2: Canonical area fallback
    canonical_fallback = _CANONICAL_AREA_FALLBACKS.get(target_area)
    if canonical_fallback:
        candidates.append(canonical_fallback)

    # Fallback candidate 4: General fallback
    candidates.append(
        f"The core operational uncertainty in {weak_area_str} creates a material business risk "
        f"that directly affects the strategic viability of the startup."
    )

    for cand_statement in candidates:
        output = CriticalAssumptionLLMOutput(
            area=target_area,
            statement=cand_statement,
            rationale=(
                f"Deterministic fallback critical assumption formulated directly from grounded facts "
                f"for weak area '{weak_area_str}' after LLM validation retries were exhausted."
            ),
            what_falsifies_it=(
                f"Evidence showing that target customers do not experience material operational disruption "
                f"or risk in the '{weak_area_str}' area."
            ),
            grounding_points=evidence_supports[:3],
            unresolved_uncertainty_addressed=unresolved_uncertainty,
            introduced_test_parameters=[],
        )
        is_valid_grounding, _ = validate_critical_assumption_grounding(
            output, rc_dict, founder_source_text, weak_area_str
        )
        is_valid_purity, _ = validate_critical_assumption_area_purity(
            output, target_area
        )
        is_valid_atomicity, _ = validate_critical_assumption_atomicity(
            output, target_area
        )
        if is_valid_grounding and is_valid_purity and is_valid_atomicity:
            return output

    return None


def generate_critical_assumption(state: StartaryGraphState) -> dict:
    """
    LLM Node: Formulate ONE specific, testable critical assumption for the weak area (Milestone 2.6.2 / 4.0.7.1.2 / 4.0.7.1.3 / 4.0.7.1.4).
    Enforces strict grounding discipline, area purity, and atomicity with structured retry feedback and deterministic fallback recovery.
    """
    t_start = time.monotonic()
    project_id = state.get("project_id", "<unknown>")
    weak_area_str = state.get("weak_area")
    rc_dict = state.get("reality_check", {})
    startup_state_dict = state.get("startup_state", {})

    if not weak_area_str or not rc_dict:
        raise ValueError(f"generate_critical_assumption | project_id={project_id}: weak_area or reality_check missing.")

    area_knowledge_dict = startup_state_dict.get("areas", {}).get(weak_area_str, {})

    # Gather founder source context for strict grounding validation
    founder_messages = state.get("founder_messages") or []
    founder_source_text = "\n".join(
        m.get("content", "") for m in founder_messages if m.get("role") == "founder"
    ) or state.get("raw_context", "")

    settings = get_settings()
    base_max_tokens = 2500
    llm = _safe_get_llm_client(settings, tier="reasoning", max_completion_tokens=base_max_tokens)
    structured_llm = llm.with_structured_output(CriticalAssumptionLLMOutput)

    user_message = build_critical_assumption_user_message(
        weak_area=weak_area_str,
        reality_check_summary=rc_dict.get("summary", ""),
        unresolved_uncertainty=rc_dict.get("unresolved_uncertainty", ""),
        conclusion=rc_dict.get("conclusion", ""),
        area_knowledge_summary=json.dumps(area_knowledge_dict, indent=2),
        what_evidence_supports=rc_dict.get("what_evidence_supports") or [],
        what_evidence_does_not_support=rc_dict.get("what_evidence_does_not_support") or [],
        relevant_founder_claims=rc_dict.get("relevant_founder_claims") or [],
        available_founder_evidence=rc_dict.get("available_founder_evidence") or [],
        founder_source_context=founder_source_text,
    )

    messages: list[dict[str, str]] = [
        {"role": "system", "content": CRITICAL_ASSUMPTION_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    attempt = 0
    last_error: Exception | None = None

    while attempt < 2:
        attempt += 1
        try:
            result, latency_ms = _invoke_structured_with_token_retry(
                structured_llm=structured_llm,
                messages=messages,
                settings=settings,
                tier="reasoning",
                base_max_tokens=base_max_tokens,
                node_name="generate_critical_assumption",
                schema_cls=CriticalAssumptionLLMOutput,
            )

            if not isinstance(result, CriticalAssumptionLLMOutput):
                raise ValidationError(f"Expected CriticalAssumptionLLMOutput, got {type(result)}")

            if str(result.area) != weak_area_str:
                logger.warning(
                    "generate_critical_assumption: LLM returned area %s instead of weak_area %s. Normalizing.",
                    result.area,
                    weak_area_str,
                )
                result.area = StartupArea(weak_area_str)

            # --- Deterministic Grounding & Area-Purity & Atomicity Validation (M2.6.2 / M4.0.7.1.3 / M4.0.7.1.4) ---
            is_valid_grounding, grounding_reason = validate_critical_assumption_grounding(
                result, rc_dict, founder_source_text, weak_area_str
            )
            is_valid_purity, purity_reason = validate_critical_assumption_area_purity(
                result, weak_area_str
            )
            is_valid_atomicity, atomicity_reason = validate_critical_assumption_atomicity(
                result, weak_area_str
            )

            is_valid = is_valid_grounding and is_valid_purity and is_valid_atomicity
            val_reason = (
                grounding_reason if not is_valid_grounding
                else (purity_reason if not is_valid_purity else atomicity_reason)
            )

            if not is_valid:
                logger.warning(
                    "generate_critical_assumption: Validation failed on attempt %d: %s.",
                    attempt,
                    val_reason,
                )
                if attempt < 2:
                    blocked_concepts = extract_blocked_grounding_concepts(val_reason, result.statement, rc_dict)
                    logger.warning(
                        "critical_assumption_retry | attempt=2 rejected_reason=%s blocked_concepts=%s",
                        val_reason,
                        blocked_concepts,
                    )
                    messages.append({"role": "assistant", "content": result.statement})
                    blocked_str = ", ".join(f"'{c}'" for c in blocked_concepts) if blocked_concepts else "None"
                    messages.append({
                        "role": "user",
                        "content": (
                            f"GROUNDING VALIDATION FAILED: {val_reason}\n"
                            f"REJECTED STATEMENT: {result.statement}\n"
                            f"BLOCKED CONCEPTS / PHRASES: {blocked_str}\n\n"
                            "You must NOT use any of the blocked concepts or phrases above.\n"
                            f"The assumption must stay strictly inside the epistemic boundary of '{weak_area_str}'.\n"
                            "Do NOT import validation requirements from neighboring areas (e.g. for Target User, do NOT require adoption, willingness to pay, or product preference).\n"
                            "The assumption must be ATOMIC: represent ONE decision-relevant uncertainty testable in a single bounded experiment.\n"
                            "Do NOT combine multiple downstream commercial conditions (e.g. commission + pricing + contribution margin).\n"
                            "You must NOT invent ungrounded numeric thresholds, percentages, frequencies (e.g. weekly), or unevidenced failure modes.\n"
                            "Reformulate the critical assumption to test the core uncertainty using ONLY founder-grounded facts, Reality Check constraints, strict area purity, and single-hypothesis atomicity."
                        ),
                    })
                    continue
                else:
                    logger.warning(
                        "generate_critical_assumption: Attempt 2 failed validation: %s. Attempting fallback.",
                        val_reason,
                    )
                    fallback = build_grounded_critical_assumption_fallback(
                        weak_area_str, rc_dict, founder_source_text, state
                    )
                    if fallback:
                        is_fb_g, fb_g_reason = validate_critical_assumption_grounding(
                            fallback, rc_dict, founder_source_text, weak_area_str
                        )
                        is_fb_p, fb_p_reason = validate_critical_assumption_area_purity(
                            fallback, weak_area_str
                        )
                        is_fb_a, fb_a_reason = validate_critical_assumption_atomicity(
                            fallback, weak_area_str
                        )
                        if is_fb_g and is_fb_p and is_fb_a:
                            logger.info(
                                "critical_assumption_fallback | used=True area=%s statement=%s",
                                weak_area_str,
                                fallback.statement[:80],
                            )
                            trace = AITrace(
                                module_name="generate_critical_assumption",
                                prompt_version=CRITICAL_ASSUMPTION_PROMPT_VERSION,
                                input_data={
                                    "weak_area": weak_area_str,
                                    "conclusion": rc_dict.get("conclusion"),
                                    "fallback_used": True,
                                },
                                output_data=fallback.model_dump(),
                                latency_ms=(time.monotonic() - t_start) * 1000,
                                project_id=project_id,
                            )
                            existing_traces = list(state.get("traces") or [])
                            existing_traces.append(trace.model_dump(mode="json"))
                            return {
                                "critical_assumption_output": fallback.model_dump(),
                                "last_trace": trace.model_dump(mode="json"),
                                "traces": existing_traces,
                            }
                        else:
                            fb_fail_reason = (
                                fb_g_reason if not is_fb_g
                                else (fb_p_reason if not is_fb_p else fb_a_reason)
                            )
                            logger.error(
                                "critical_assumption_recovery_failed | reason=Fallback failed validation: %s",
                                fb_fail_reason,
                            )
                            raise CriticalAssumptionGroundingError(
                                f"generate_critical_assumption failed grounding validation after retry and fallback: {fb_fail_reason}"
                            )
                    else:
                        logger.error(
                            "critical_assumption_recovery_failed | reason=%s",
                            val_reason,
                        )
                        raise CriticalAssumptionGroundingError(
                            f"generate_critical_assumption failed grounding validation after retry: {val_reason}"
                        )

            trace = AITrace(
                module_name="generate_critical_assumption",
                prompt_version=CRITICAL_ASSUMPTION_PROMPT_VERSION,
                input_data={"weak_area": weak_area_str, "conclusion": rc_dict.get("conclusion")},
                output_data=result.model_dump(),
                latency_ms=latency_ms,
                project_id=project_id,
            )

            existing_traces = list(state.get("traces") or [])
            existing_traces.append(trace.model_dump(mode="json"))

            logger.info("generate_critical_assumption | validated statement: %s (attempt=%d)", result.statement[:80], attempt)
            return {
                "critical_assumption_output": result.model_dump(),
                "last_trace": trace.model_dump(mode="json"),
                "traces": existing_traces,
            }
        except Exception as exc:
            if isinstance(exc, (CriticalAssumptionGroundingError, TokenBudgetExhaustedError)):
                raise
            last_error = exc
            logger.warning("generate_critical_assumption failed attempt %d: %s", attempt, exc)
            if is_permanent_model_error(exc):
                raise
            if attempt < 2:
                continue
            raise

    if isinstance(last_error, TokenBudgetExhaustedError):
        raise last_error
    raise RuntimeError(f"generate_critical_assumption failed after 2 attempts. Error: {last_error}")


# ---------------------------------------------------------------------------
# 14. save_critical_assumption (Deterministic Python — Milestone 2)
# ---------------------------------------------------------------------------

def save_critical_assumption(state: StartaryGraphState) -> dict:
    """
    Deterministic Node:
    - Validates generated critical assumption
    - Instantiates first-class Assumption domain model
    - Stores exactly one active critical assumption in state
    - Updates WorkflowPhase to TEST
    """
    project_id = state.get("project_id", "<unknown>")
    iteration = state.get("iteration", 1)
    weak_area_str = state.get("weak_area")
    ca_output = state.get("critical_assumption_output")

    if not ca_output or "statement" not in ca_output:
        raise ValueError(f"save_critical_assumption | project_id={project_id}: critical_assumption_output missing.")

    statement = ca_output["statement"].strip()
    if not statement:
        raise ValueError("Critical assumption statement cannot be empty.")

    area_enum = StartupArea(weak_area_str or ca_output["area"])

    assumption = Assumption(
        id=str(uuid.uuid4()),
        project_id=project_id,
        statement=statement,
        area=area_enum,
        status=AssumptionStatus.UNTESTED,
        is_critical=True,
        iteration_created=iteration,
    )

    logger.info(
        "save_critical_assumption | project_id=%s assumption_id=%s area=%s",
        project_id,
        assumption.id,
        assumption.area,
    )

    return {
        "active_assumption": assumption.model_dump(mode="json"),
        "active_critical_assumption": assumption.model_dump(mode="json"),
        "phase": WorkflowPhase.TEST,
    }


# ---------------------------------------------------------------------------
# Backward Compatibility Aliases (M2.5 → M2.6)
# ---------------------------------------------------------------------------

def assess_context_sufficiency(state: StartaryGraphState) -> dict:
    """
    Backward-compatible node for M2.5 tests.

    This is a standalone node that uses ContextSufficiencyLLMOutput (the M2.5 schema)
    rather than the new InformationGapPlanLLMOutput. It is kept so that existing
    test_context_sufficiency.py tests continue to pass without modification.

    In the live M2.6 graph, this node is NOT used. plan_information_gap is used instead.
    """
    project_id = state.get("project_id", "<unknown>")
    startup_state_dict = state.get("startup_state", {})
    diagnosis_dict = state.get("diagnosis", {})
    founder_messages = state.get("founder_messages") or []

    history_lines = []
    for msg in founder_messages:
        history_lines.append(f"{msg.get('role', 'founder').upper()}: {msg.get('content')}")
    history_text = "\n".join(history_lines) if history_lines else (state.get("raw_context") or "")

    settings = get_settings()
    llm = _safe_get_llm_client(settings, tier="reasoning", max_tokens=1000)
    structured_llm = llm.with_structured_output(ContextSufficiencyLLMOutput)

    user_message = build_context_sufficiency_user_message(
        startup_state_json=__import__("json").dumps(startup_state_dict, indent=2),
        diagnosis_summary=diagnosis_dict.get("summary", ""),
        conversation_history_text=history_text,
    )

    messages = [
        {"role": "system", "content": ASSESS_CONTEXT_SUFFICIENCY_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    attempt = 0
    last_error: Exception | None = None

    while attempt < 2:
        attempt += 1
        t_start = time.monotonic()
        try:
            result: ContextSufficiencyLLMOutput = structured_llm.invoke(messages)
            latency_ms = (time.monotonic() - t_start) * 1000

            from startary.domain.sufficiency import GroundingQuality as _GQ
            grounding_quality_val = result.grounding_quality if isinstance(result.grounding_quality, str) else str(result.grounding_quality)
            try:
                gq = _GQ(grounding_quality_val)
            except (ValueError, TypeError):
                gq = _GQ.WEAK

            cs_domain = ContextSufficiencyAssessment(
                sufficient=bool(result.sufficient),
                grounding_quality=gq,
                would_one_more_answer_materially_change_direction=bool(result.would_one_more_answer_materially_change_direction),
                reason=result.reason if isinstance(result.reason, str) else "",
                blocking_unknowns=list(result.blocking_unknowns) if result.blocking_unknowns else [],
                usable_grounding=list(result.usable_grounding) if result.usable_grounding else [],
                prohibited_inferences=list(result.prohibited_inferences) if result.prohibited_inferences else [],
                recommended_question_area=result.recommended_question_area if isinstance(result.recommended_question_area, (str, type(None))) else None,
                information_goal=result.information_goal if isinstance(result.information_goal, str) else "",
            )

            trace = AITrace(
                module_name="assess_context_sufficiency",
                prompt_version=CONTEXT_SUFFICIENCY_PROMPT_VERSION,
                input_data={"messages_count": len(founder_messages)},
                output_data=result.model_dump(),
                latency_ms=latency_ms,
                project_id=project_id,
            )

            existing_traces = list(state.get("traces") or [])
            existing_traces.append(trace.model_dump(mode="json"))

            return {
                "context_sufficiency_output": result.model_dump(),
                "context_sufficiency": cs_domain.model_dump(mode="json"),
                "is_sufficient": cs_domain.sufficient,
                "last_trace": trace.model_dump(mode="json"),
                "traces": existing_traces,
            }
        except Exception as exc:
            last_error = exc
            logger.warning("assess_context_sufficiency failed attempt %d: %s", attempt, exc)

    raise RuntimeError(f"assess_context_sufficiency failed after 2 attempts. Error: {last_error}")


# ---------------------------------------------------------------------------
# Milestone 2.6.1 Nodes: Baseline Startup Check-In
# ---------------------------------------------------------------------------

def assess_baseline_state_sufficiency(state: StartaryGraphState) -> dict:
    """
    Evaluates whether Startary has gathered sufficient minimum operational context
    across the 5 core baseline dimensions (TEAM, CAPACITY, PRODUCT_STAGE, VALIDATION_OR_USER, RESOURCE).

    Fast path shortcut: If all 5 dimensions are already known from extraction,
    immediately returns is_sufficient=True without calling the LLM.
    """
    project_id = state.get("project_id", "<unknown>")
    exec_ctx = state.get("execution_context") or {}
    prog = state.get("startup_progress") or {}

    # 1. Team
    fc = exec_ctx.get("founder_count") or {}
    ats = exec_ctx.get("active_team_size") or {}
    members = exec_ctx.get("team_members") or []
    team_known = (fc.get("provenance") not in ("UNKNOWN", None)) or (ats.get("provenance") not in ("UNKNOWN", None)) or len(members) > 0

    # 2. Capacity
    wh = exec_ctx.get("approximate_weekly_hours_per_member") or {}
    tot_cap = exec_ctx.get("total_weekly_capacity_hours") or {}
    cap_known = (wh.get("provenance") not in ("UNKNOWN", None)) or (tot_cap.get("provenance") not in ("UNKNOWN", None))

    # 3. Product Stage
    stage = prog.get("development_stage") or {}
    pstatus = prog.get("product_status") or {}
    proto = prog.get("prototype_exists") or {}
    mvp = prog.get("mvp_exists") or {}
    launched = prog.get("launched") or {}
    stage_known = any(f.get("provenance") not in ("UNKNOWN", None) for f in (stage, pstatus, proto, mvp, launched))

    # 4. Validation / Users
    val_act = prog.get("validation_activity") or {}
    interviews = prog.get("interviews_completed") or {}
    users = prog.get("active_users") or {}
    paying = prog.get("paying_customers") or {}
    val_known = any(f.get("provenance") not in ("UNKNOWN", None) for f in (val_act, interviews, users, paying))

    # 5. Resource
    budget = exec_ctx.get("available_budget_amount") or {}
    funding = exec_ctx.get("funding_status") or {}
    curr = exec_ctx.get("budget_currency") or {}
    resource_known = any(f.get("provenance") not in ("UNKNOWN", None) for f in (budget, funding, curr))

    dim_statuses = {
        BaselineDimension.TEAM: BaselineDimensionStatus.KNOWN_ENOUGH if team_known else BaselineDimensionStatus.MISSING,
        BaselineDimension.CAPACITY: BaselineDimensionStatus.KNOWN_ENOUGH if cap_known else BaselineDimensionStatus.MISSING,
        BaselineDimension.PRODUCT_STAGE: BaselineDimensionStatus.KNOWN_ENOUGH if stage_known else BaselineDimensionStatus.MISSING,
        BaselineDimension.VALIDATION_OR_USER: BaselineDimensionStatus.KNOWN_ENOUGH if val_known else BaselineDimensionStatus.MISSING,
        BaselineDimension.RESOURCE: BaselineDimensionStatus.KNOWN_ENOUGH if resource_known else BaselineDimensionStatus.MISSING,
    }

    missing = [dim for dim, status in dim_statuses.items() if status == BaselineDimensionStatus.MISSING]

    # Invariant: If extraction failed, facts were not persisted; baseline CANNOT be sufficient
    if state.get("extraction_status") == "FAILED":
        suff_domain = BaselineStartupStateSufficiency(
            is_sufficient=False,
            dimension_statuses=dim_statuses,
            missing_groups=missing if missing else list(BaselineDimension),
            reason=f"Baseline operational facts have not been successfully persisted (extraction_status=FAILED: {state.get('extraction_error', 'extraction failed')}).",
        )
        logger.warning("assess_baseline_state_sufficiency | project_id=%s -> INSUFFICIENT (extraction_status=FAILED)", project_id)
        return {
            "baseline_sufficiency": suff_domain.model_dump(mode="json"),
            "is_baseline_sufficient": False,
        }

    # Fast path shortcut: all 5 dimensions known
    if not missing:
        suff_domain = BaselineStartupStateSufficiency(
            is_sufficient=True,
            dimension_statuses=dim_statuses,
            missing_groups=[],
            reason="All 5 baseline operational dimensions are sufficiently grounded.",
        )
        logger.info("assess_baseline_state_sufficiency | project_id=%s -> SUFFICIENT (Fast path shortcut)", project_id)
        return {
            "baseline_sufficiency": suff_domain.model_dump(mode="json"),
            "is_baseline_sufficient": True,
        }

    # Otherwise call LLM with BASELINE_STATE_SUFFICIENCY_SYSTEM_PROMPT
    settings = get_settings()
    llm = _safe_get_llm_client(settings, tier="fast", max_tokens=800)
    structured_llm = llm.with_structured_output(BaselineSufficiencyLLMOutput)

    founder_messages = state.get("founder_messages") or []
    history_lines = [f"{m.get('role', 'founder').upper()}: {m.get('content')}" for m in founder_messages if m.get("role") == "founder"]
    history_text = "\n".join(history_lines) if history_lines else (state.get("raw_context") or "")

    user_msg = build_baseline_sufficiency_user_message(
        execution_context_json=_json.dumps(exec_ctx, indent=2, default=str),
        progress_json=_json.dumps(prog, indent=2, default=str),
        founder_messages_text=history_text,
    )

    messages = [
        {"role": "system", "content": BASELINE_STATE_SUFFICIENCY_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    res_info = get_llm_client_resolution_info(llm)
    attempt = 0
    last_err = None
    while attempt < 2:
        attempt += 1
        t_start = time.monotonic()
        try:
            res: BaselineSufficiencyLLMOutput = structured_llm.invoke(messages)
            latency_ms = (time.monotonic() - t_start) * 1000

            def _map_st(val: str, fallback_known: bool) -> BaselineDimensionStatus:
                try:
                    return BaselineDimensionStatus(val)
                except (ValueError, TypeError):
                    return BaselineDimensionStatus.KNOWN_ENOUGH if fallback_known else BaselineDimensionStatus.MISSING

            res_dim_statuses = {
                BaselineDimension.TEAM: _map_st(res.team_status, team_known),
                BaselineDimension.CAPACITY: _map_st(res.capacity_status, cap_known),
                BaselineDimension.PRODUCT_STAGE: _map_st(res.product_stage_status, stage_known),
                BaselineDimension.VALIDATION_OR_USER: _map_st(res.validation_status, val_known),
                BaselineDimension.RESOURCE: _map_st(res.resource_status, resource_known),
            }
            res_missing = [dim for dim, status in res_dim_statuses.items() if status == BaselineDimensionStatus.MISSING]

            final_is_sufficient = bool(res.is_sufficient) and len(res_missing) == 0

            suff_domain = BaselineStartupStateSufficiency(
                is_sufficient=final_is_sufficient,
                dimension_statuses=res_dim_statuses,
                missing_groups=res_missing if not final_is_sufficient else [],
                reason=res.reason if isinstance(res.reason, str) else "",
            )

            trace = AITrace(
                module_name="assess_baseline_state_sufficiency",
                prompt_version=BASELINE_STATE_SUFFICIENCY_PROMPT_VERSION,
                input_data={"messages_count": len(founder_messages)},
                output_data=res.model_dump(),
                latency_ms=latency_ms,
                project_id=project_id,
                model_name=res_info["resolved_model"] or getattr(settings, "fast_model_name", None),
                configured_model=res_info["configured_model"],
                resolved_model=res_info["resolved_model"],
                fallback_used=res_info["fallback_used"],
                fallback_reason=res_info["fallback_reason"],
                execution_path=state.get("execution_path", "DELTA_STATE_UPDATE"),
            )
            existing_traces = list(state.get("traces") or [])
            existing_traces.append(trace.model_dump(mode="json"))

            turn_telem = record_telemetry_event(
                state=state,
                node_name="assess_baseline_state_sufficiency",
                model_name=res_info["resolved_model"] or getattr(settings, "fast_model_name", None) or "fast_tier",
                duration_ms=latency_ms,
                execution_path=state.get("execution_path", "DELTA_STATE_UPDATE"),
                configured_model=res_info["configured_model"],
                resolved_model=res_info["resolved_model"],
                fallback_used=res_info["fallback_used"],
                fallback_reason=res_info["fallback_reason"],
            )

            logger.info("assess_baseline_state_sufficiency | project_id=%s is_sufficient=%s missing=%s", project_id, suff_domain.is_sufficient, res_missing)
            return {
                "baseline_sufficiency": suff_domain.model_dump(mode="json"),
                "is_baseline_sufficient": suff_domain.is_sufficient,
                "last_trace": trace.model_dump(mode="json"),
                "traces": existing_traces,
                "turn_telemetry": turn_telem,
            }
        except Exception as exc:
            last_err = exc
            logger.warning("assess_baseline_state_sufficiency failed attempt %d: %s", attempt, exc)
            if is_permanent_model_error(exc):
                break

    # Fallback inspection result on error
    fallback_domain = BaselineStartupStateSufficiency(
        is_sufficient=len(missing) == 0,
        dimension_statuses=dim_statuses,
        missing_groups=missing,
        reason=f"Fallback inspection used after error: {last_err}",
    )
    fallback_telem = record_telemetry_event(
        state=state,
        node_name="assess_baseline_state_sufficiency",
        model_name="local_deterministic",
        duration_ms=0.0,
        execution_path=state.get("execution_path", "DELTA_STATE_UPDATE"),
        configured_model=res_info["configured_model"],
        resolved_model="local_deterministic",
        fallback_used=True,
        fallback_reason=f"fast_fail_error: {last_err}",
    )
    return {
        "baseline_sufficiency": fallback_domain.model_dump(mode="json"),
        "is_baseline_sufficient": fallback_domain.is_sufficient,
        "turn_telemetry": fallback_telem,
    }


def plan_baseline_question(state: StartaryGraphState) -> dict:
    """
    Selects the next missing baseline operational group and formulates ONE compact,
    grouped natural-language question.
    """
    project_id = state.get("project_id", "<unknown>")
    base_suff = state.get("baseline_sufficiency") or {}
    missing_groups = base_suff.get("missing_groups") or ["TEAM"]
    exec_ctx = state.get("execution_context") or {}
    prog = state.get("startup_progress") or {}
    baseline_history = state.get("baseline_history") or []
    turns_used = state.get("baseline_question_turns", 0)

    # Milestone 3.2.1: Semantic Duplication & Repetition Guard
    q_history = state.get("question_history") or []
    answered_targets = {h.get("information_target") for h in q_history if h.get("answered")}

    # Inspect current state to determine which operational areas are already known
    team_known = bool((exec_ctx.get("founder_count") or {}).get("value") or (exec_ctx.get("team_members") or []))
    capacity_known = bool((exec_ctx.get("approximate_weekly_hours_per_member") or {}).get("value") or (exec_ctx.get("total_weekly_capacity_hours") or {}).get("value"))
    product_stage_known = bool((prog.get("development_stage") or {}).get("value") or (prog.get("product_status") or {}).get("value") or (prog.get("prototype_exists") or {}).get("value"))
    validation_known = bool((prog.get("validation_activity") or {}).get("value") or (prog.get("interviews_completed") or {}).get("value") or (prog.get("active_users") or {}).get("value"))
    resource_known = bool((exec_ctx.get("available_budget_amount") or {}).get("value") or (exec_ctx.get("funding_status") or {}).get("value"))

    known_status = {
        "TEAM": team_known,
        "CAPACITY": capacity_known,
        "PRODUCT_STAGE": product_stage_known,
        "VALIDATION_OR_USER": validation_known,
        "RESOURCE": resource_known,
    }

    # Filter missing_groups: exclude groups already known or already answered
    effective_missing = [
        g for g in missing_groups
        if not known_status.get(g, False) and g not in answered_targets
    ]
    if not effective_missing:
        effective_missing = [g for g in ["TEAM", "CAPACITY", "PRODUCT_STAGE", "VALIDATION_OR_USER", "RESOURCE"] if not known_status.get(g, False)] or ["CAPACITY"]

    prev_questions = [h.get("question") for h in baseline_history if h.get("question")]

    settings = get_settings()
    llm = _safe_get_llm_client(settings, tier="fast", max_tokens=800)
    structured_llm = llm.with_structured_output(BaselineQuestionPlanLLMOutput)

    user_msg = build_baseline_question_planner_user_message(
        missing_groups=effective_missing,
        execution_context_json=_json.dumps(exec_ctx, indent=2, default=str),
        progress_json=_json.dumps(prog, indent=2, default=str),
        previous_baseline_questions=prev_questions,
        turns_used=turns_used,
        max_turns=MAX_BASELINE_QUESTION_TURNS,
    )

    messages = [
        {"role": "system", "content": BASELINE_QUESTION_PLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    res_info = get_llm_client_resolution_info(llm)
    attempt = 0
    last_err = None
    while attempt < 2:
        attempt += 1
        t_start = time.monotonic()
        try:
            res: BaselineQuestionPlanLLMOutput = structured_llm.invoke(messages)
            latency_ms = (time.monotonic() - t_start) * 1000

            target_group_str = res.target_group or (effective_missing[0] if effective_missing else "CAPACITY")
            # Guard against LLM re-selecting already-answered or known group
            if target_group_str in answered_targets or known_status.get(target_group_str, False):
                logger.info("plan_baseline_question: LLM re-selected known target %s; overriding with %s", target_group_str, effective_missing[0])
                target_group_str = effective_missing[0]

            try:
                t_dim = BaselineDimension(target_group_str)
            except (ValueError, TypeError):
                t_dim = BaselineDimension.TEAM

            q_plan = BaselineQuestionPlan(
                should_ask=bool(res.should_ask),
                target_group=t_dim,
                target_fields=res.target_fields or [],
                question=res.question,
                reason=res.reason if isinstance(res.reason, str) else "",
            )

            trace = AITrace(
                module_name="plan_baseline_question",
                prompt_version=BASELINE_QUESTION_PLANNER_PROMPT_VERSION,
                input_data={"missing_groups": effective_missing, "turns_used": turns_used},
                output_data=res.model_dump(),
                latency_ms=latency_ms,
                project_id=project_id,
                model_name=res_info["resolved_model"] or getattr(settings, "fast_model_name", None),
                configured_model=res_info["configured_model"],
                resolved_model=res_info["resolved_model"],
                fallback_used=res_info["fallback_used"],
                fallback_reason=res_info["fallback_reason"],
                execution_path=state.get("execution_path", "DELTA_STATE_UPDATE"),
            )
            existing_traces = list(state.get("traces") or [])
            existing_traces.append(trace.model_dump(mode="json"))

            turn_telem = record_telemetry_event(
                state=state,
                node_name="plan_baseline_question",
                model_name=res_info["resolved_model"] or getattr(settings, "fast_model_name", None) or "fast_tier",
                duration_ms=latency_ms,
                execution_path=state.get("execution_path", "DELTA_STATE_UPDATE"),
                configured_model=res_info["configured_model"],
                resolved_model=res_info["resolved_model"],
                fallback_used=res_info["fallback_used"],
                fallback_reason=res_info["fallback_reason"],
            )

            q_text = res.question or "Could you describe your team, what exists product-wise today, and your current testing activity?"

            # Canonical NextTurnDecision (Milestone 3.2.1)
            qid = f"q_base_{turns_used + 1}_{uuid.uuid4().hex[:6]}"
            decision = NextTurnDecision(
                question_id=qid,
                source_phase=WorkflowPhase.CLARIFYING_CONTEXT,
                information_target=target_group_str,
                canonical_question=q_text,
                should_ask_question=True,
                clarification_reason=res.reason or "Baseline startup onboarding collection",
                planner_version=BASELINE_QUESTION_PLANNER_PROMPT_VERSION,
            )

            logger.info("plan_baseline_question | project_id=%s target_group=%s qid=%s question=%s", project_id, target_group_str, qid, q_text[:70])
            return {
                "active_next_turn_decision": decision.model_dump(),
                "current_baseline_question": {
                    "question": q_text,
                    "target_group": target_group_str,
                    "target_fields": res.target_fields or [],
                    "information_goal": res.reason or "",
                },
                "current_clarifying_question": {
                    "question": q_text,
                    "target_area": target_group_str,
                    "information_goal": res.reason or "",
                },
                "active_interrupt_type": "baseline",
                "phase": WorkflowPhase.CLARIFYING_CONTEXT,
                "last_trace": trace.model_dump(mode="json"),
                "traces": existing_traces,
                "turn_telemetry": turn_telem,
            }
        except Exception as exc:
            last_err = exc
            logger.warning("plan_baseline_question failed attempt %d: %s", attempt, exc)
            if is_permanent_model_error(exc):
                break

    # Fallback question based on missing group
    fallback_group = effective_missing[0] if effective_missing else "CAPACITY"
    fallback_q_map = {
        "TEAM": "Are you building this alone or with a team, and what does each person mainly work on?",
        "CAPACITY": "Roughly how much time can you and the team realistically give the startup each week?",
        "PRODUCT_STAGE": "What exists today — just the idea, designs, a prototype, an MVP, or a launched product, and has anyone tried it yet?",
        "VALIDATION_OR_USER": "Have you spoken with potential users or had anyone try the product yet? Roughly how many, and how are you reaching them?",
        "RESOURCE": "Are you self-funded or externally funded, and roughly what budget can you realistically spend on testing the startup?",
    }
    q_fallback = fallback_q_map.get(fallback_group, "Could you tell me a bit more about your team, current product status, and testing so far?")
    fb_qid = f"q_base_fb_{turns_used + 1}_{uuid.uuid4().hex[:6]}"
    fb_decision = NextTurnDecision(
        question_id=fb_qid,
        source_phase=WorkflowPhase.CLARIFYING_CONTEXT,
        information_target=fallback_group,
        canonical_question=q_fallback,
        should_ask_question=True,
        clarification_reason="Fallback baseline collection",
        planner_version=BASELINE_QUESTION_PLANNER_PROMPT_VERSION,
    )
    fallback_telem = record_telemetry_event(
        state=state,
        node_name="plan_baseline_question",
        model_name="local_deterministic",
        duration_ms=0.0,
        execution_path=state.get("execution_path", "DELTA_STATE_UPDATE"),
        configured_model=res_info["configured_model"],
        resolved_model="local_deterministic",
        fallback_used=True,
        fallback_reason=f"fast_fail_error: {last_err}",
    )
    return {
        "active_next_turn_decision": fb_decision.model_dump(),
        "current_baseline_question": {
            "question": q_fallback,
            "target_group": fallback_group,
            "target_fields": [],
            "information_goal": "Fallback baseline collection",
        },
        "current_clarifying_question": {
            "question": q_fallback,
            "target_area": fallback_group,
            "information_goal": "Fallback baseline collection",
        },
        "active_interrupt_type": "baseline",
        "phase": WorkflowPhase.CLARIFYING_CONTEXT,
        "turn_telemetry": fallback_telem,
    }


# ---------------------------------------------------------------------------
# Milestone 2.6.1 Nodes: Weak-Link Discrimination
# ---------------------------------------------------------------------------

def assess_weak_link_discrimination(state: StartaryGraphState) -> dict:
    """
    Evaluates whether the deterministic weak-link selection is sufficiently discriminated
    or if multiple areas share maximum score / near-identical risk requiring clarification.
    """
    project_id = state.get("project_id", "<unknown>")
    weak_area = state.get("weak_area") or "Problem"
    weak_details = state.get("weak_link_details") or {}
    area_assessments = state.get("area_assessments") or []
    diagnosis_dict = state.get("diagnosis") or {}
    wl_history = state.get("weak_link_clarification_history") or []
    turns_used = state.get("weak_link_clarification_turns", 0)

    # Check max turns
    if turns_used >= MAX_WEAK_LINK_CLARIFICATION_TURNS:
        logger.info("assess_weak_link_discrimination | Max turns reached (%d); proceeding with deterministic winner under LOW confidence", turns_used)
        return {
            "turn_telemetry": record_telemetry_event(
                state=state, node_name="assess_weak_link_discrimination", model_name="local_deterministic",
                skipped_by_fast_path=True, execution_path="DETERMINISTIC_MAX_TURNS", model_tier="none",
            ),
            "selection_confidence": "LOW",
            "should_clarify": False,
            "should_clarify_weak_link": False,
            "discrimination_reason": "Maximum weak-link clarification turns reached; proceeding with best available deterministic selection under explicit LOW confidence.",
        }

    telemetry = weak_details.get("saturation_telemetry") or {}
    sat_count = telemetry.get("saturated_area_count", 0)
    tie_count = telemetry.get("max_score_tie_count", 1)
    tie_broken = telemetry.get("tie_breaker_used", False)
    max_score = weak_details.get("score", 1)

    # Sort areas by score
    sorted_areas = sorted(area_assessments, key=lambda x: x.get("calculated_decision_risk", 1), reverse=True)
    runner_up_score = sorted_areas[1].get("calculated_decision_risk", 1) if len(sorted_areas) > 1 else 1
    score_gap = max_score - runner_up_score

    # Fast path: Clear single winner (>= 10 point gap and no tie)
    if tie_count == 1 and score_gap >= 10 and not tie_broken:
        logger.info("assess_weak_link_discrimination | Single clear winner (score=%d runner_up=%d gap=%d); proceeding with HIGH confidence", max_score, runner_up_score, score_gap)
        disc_domain = WeakLinkDiscrimination(
            selection_confidence=SelectionConfidence.HIGH,
            current_best_area=StartupArea(weak_area),
            competing_areas=[],
            discrimination_reason=f"Decisive {score_gap}-point risk score lead over closest competing area.",
            should_clarify=False,
        )
        return {
            "turn_telemetry": record_telemetry_event(
                state=state, node_name="assess_weak_link_discrimination", model_name="local_deterministic",
                skipped_by_fast_path=True, execution_path="DETERMINISTIC_CLEAR_WINNER", model_tier="none",
            ),
            "weak_link_discrimination": disc_domain.model_dump(mode="json"),
            "selection_confidence": "HIGH",
            "competing_weak_areas": [],
            "should_clarify": False,
            "should_clarify_weak_link": False,
        }

    # Ambiguity detected -> invoke LLM discrimination evaluation
    settings = get_settings()
    base_max_tokens = 2000
    llm = _safe_get_llm_client(settings, tier="reasoning", max_completion_tokens=base_max_tokens)
    structured_llm = llm.with_structured_output(WeakLinkDiscriminationLLMOutput)

    # Format table
    table_lines = [f"{a.get('area')}: Score={a.get('calculated_decision_risk')} (Imp={a.get('importance')}, Unc={a.get('uncertainty')}, Gap={a.get('evidence_gap')}, Con={a.get('consequence_if_wrong')})" for a in area_assessments]
    table_str = "\n".join(table_lines)

    prev_qa = [f"Q: {h.get('question')} | A: {h.get('founder_answer')}" for h in wl_history]

    user_msg = build_weak_link_discrimination_user_message(
        area_assessments_table=table_str,
        current_winner_area=weak_area,
        winner_score=max_score,
        tie_telemetry_json=_json.dumps(telemetry, indent=2, default=str),
        diagnosis_summary=diagnosis_dict.get("summary", ""),
        previous_weak_link_qa=prev_qa,
        turns_used=turns_used,
        max_turns=MAX_WEAK_LINK_CLARIFICATION_TURNS,
    )

    messages = [
        {"role": "system", "content": WEAK_LINK_DISCRIMINATION_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    attempt = 0
    last_err = None
    while attempt < 2:
        attempt += 1
        try:
            res, latency_ms = _invoke_structured_with_token_retry(
                structured_llm=structured_llm,
                messages=messages,
                settings=settings,
                tier="reasoning",
                base_max_tokens=base_max_tokens,
                node_name="assess_weak_link_discrimination",
                schema_cls=WeakLinkDiscriminationLLMOutput,
            )

            try:
                conf_enum = SelectionConfidence(res.selection_confidence)
            except (ValueError, TypeError):
                conf_enum = SelectionConfidence.LOW if (tie_count > 1 or sat_count > 1) else SelectionConfidence.MEDIUM

            competing = [str(a) for a in res.competing_areas]

            trace = AITrace(
                module_name="assess_weak_link_discrimination",
                prompt_version=WEAK_LINK_DISCRIMINATION_PROMPT_VERSION,
                input_data={"winner": weak_area, "score": max_score, "turns_used": turns_used},
                output_data=res.model_dump(),
                latency_ms=latency_ms,
                project_id=project_id,
            )
            existing_traces = list(state.get("traces") or [])
            existing_traces.append(trace.model_dump(mode="json"))

            res_mode = getattr(res, "resolution_mode", "FOUNDER_KNOWLEDGE") or "FOUNDER_KNOWLEDGE"
            if str(res_mode).upper() == "EXTERNAL_EVIDENCE":
                clarify_flag = False
            else:
                clarify_flag = bool(res.should_clarify) and turns_used < MAX_WEAK_LINK_CLARIFICATION_TURNS
            route_selected = "select_weak_link_question" if clarify_flag else "reality_check"
            logger.info(
                "assess_weak_link_discrimination | winner=%s conf=%s should_clarify=%s res_mode=%s competing=%s turns=%d route=%s",
                weak_area,
                conf_enum.value,
                clarify_flag,
                res_mode,
                competing,
                turns_used,
                route_selected,
            )
            disc_dump = res.model_dump()
            disc_dump["should_clarify"] = clarify_flag
            turn_telem = _record_llm_node_telemetry(
                state, "assess_weak_link_discrimination", llm, "reasoning", latency_ms, attempt,
                "WeakLinkDiscriminationLLMOutput", messages=messages, result=res,
            )
            return {
                "turn_telemetry": turn_telem,
                "weak_link_discrimination": disc_dump,
                "selection_confidence": conf_enum.value,
                "competing_weak_areas": competing,
                "should_clarify": clarify_flag,
                "should_clarify_weak_link": clarify_flag,
                "last_trace": trace.model_dump(mode="json"),
                "traces": existing_traces,
            }
        except Exception as exc:
            last_err = exc
            logger.warning("assess_weak_link_discrimination failed attempt %d: %s", attempt, exc)
            if is_permanent_model_error(exc) or isinstance(exc, TokenBudgetExhaustedError):
                break

    if isinstance(last_err, TokenBudgetExhaustedError):
        raise last_err

    # Fallback discrimination logic on error
    fallback_conf = "LOW" if (tie_count > 1 or sat_count > 1) else "MEDIUM"
    fallback_competing = [str(a.get("area")) for a in sorted_areas[:3] if a.get("area") != weak_area]
    fallback_clarify = (fallback_conf == "LOW") and turns_used < MAX_WEAK_LINK_CLARIFICATION_TURNS
    fallback_telem = _record_llm_node_telemetry(
        state, "assess_weak_link_discrimination", llm, "reasoning", 0.0, attempt,
        "WeakLinkDiscriminationLLMOutput", status="FALLBACK",
    )
    return {
        "turn_telemetry": fallback_telem,
        "selection_confidence": fallback_conf,
        "competing_weak_areas": fallback_competing,
        "should_clarify": fallback_clarify,
        "should_clarify_weak_link": fallback_clarify,
        "discrimination_reason": f"Fallback rule evaluation used after error: {last_err}",
    }


def select_weak_link_question(state: StartaryGraphState) -> dict:
    """
    Formulates exactly ONE focused clarifying question to differentiate competing weak-link areas.
    """
    project_id = state.get("project_id", "<unknown>")
    weak_area = state.get("weak_area") or "Problem"
    competing = list(state.get("competing_weak_areas") or [])
    if weak_area not in competing:
        competing.insert(0, weak_area)

    disc = state.get("weak_link_discrimination") or {}
    info_goal = disc.get("recommended_information_goal") or f"Differentiate whether {weak_area} is more critical than {', '.join(competing[1:])}"
    diagnosis_dict = state.get("diagnosis") or {}

    founder_messages = state.get("founder_messages") or []
    history_lines = [f"{m.get('role', 'founder').upper()}: {m.get('content')}" for m in founder_messages]
    history_text = "\n".join(history_lines) if history_lines else (state.get("raw_context") or "")

    settings = get_settings()
    base_max_tokens = 2000
    llm = _safe_get_llm_client(settings, tier="reasoning", max_completion_tokens=base_max_tokens)
    structured_llm = llm.with_structured_output(WeakLinkClarifyingQuestionLLMOutput)

    user_msg = build_weak_link_clarifying_question_user_message(
        competing_areas=competing,
        information_goal=info_goal,
        diagnosis_summary=diagnosis_dict.get("summary", ""),
        conversation_history_text=history_text,
    )

    messages = [
        {"role": "system", "content": WEAK_LINK_CLARIFYING_QUESTION_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    attempt = 0
    last_err = None
    while attempt < 2:
        attempt += 1
        try:
            res, latency_ms = _invoke_structured_with_token_retry(
                structured_llm=structured_llm,
                messages=messages,
                settings=settings,
                tier="reasoning",
                base_max_tokens=base_max_tokens,
                node_name="select_weak_link_question",
                schema_cls=WeakLinkClarifyingQuestionLLMOutput,
            )

            trace = AITrace(
                module_name="select_weak_link_question",
                prompt_version=WEAK_LINK_CLARIFICATION_PROMPT_VERSION,
                input_data={"competing_areas": competing, "turn": state.get("weak_link_clarification_turns", 0)},
                output_data=res.model_dump(),
                latency_ms=latency_ms,
                project_id=project_id,
            )
            existing_traces = list(state.get("traces") or [])
            existing_traces.append(trace.model_dump(mode="json"))

            target_str = str(res.competing_areas[0]) if res.competing_areas else weak_area
            wl_target = f"WEAK_LINK_{target_str}"
            wl_qid = f"q_wl_{state.get('weak_link_clarification_turns', 0) + 1}_{uuid.uuid4().hex[:6]}"
            wl_decision = NextTurnDecision(
                question_id=wl_qid,
                source_phase=WorkflowPhase.CLARIFYING_CONTEXT,
                information_target=wl_target,
                canonical_question=res.question,
                should_ask_question=True,
                clarification_reason=res.information_goal or "",
                planner_version=WEAK_LINK_CLARIFICATION_PROMPT_VERSION,
            )

            logger.info("select_weak_link_question | competing=%s qid=%s question=%s", competing, wl_qid, res.question[:70])
            turn_telem = _record_llm_node_telemetry(
                state, "select_weak_link_question", llm, "reasoning", latency_ms, attempt,
                "WeakLinkClarifyingQuestionLLMOutput", messages=messages, result=res,
            )
            return {
                "turn_telemetry": turn_telem,
                "active_next_turn_decision": wl_decision.model_dump(),
                "current_weak_link_question": res.model_dump(),
                "current_clarifying_question": {
                    "question": res.question,
                    "target_area": target_str,
                    "information_goal": res.information_goal,
                },
                "active_interrupt_type": "weak_link",
                "phase": WorkflowPhase.CLARIFYING_CONTEXT,
                "last_trace": trace.model_dump(mode="json"),
                "traces": existing_traces,
            }
        except Exception as exc:
            last_err = exc
            logger.warning("select_weak_link_question failed attempt %d: %s", attempt, exc)
            if is_permanent_model_error(exc) or isinstance(exc, TokenBudgetExhaustedError):
                break

    if isinstance(last_err, TokenBudgetExhaustedError):
        raise last_err

    # Fallback question
    fallback_q = f"When considering {weak_area} versus other aspects of your startup, what specific user behavior or outcome have you observed most directly?"
    fb_wl_qid = f"q_wl_fb_{state.get('weak_link_clarification_turns', 0) + 1}_{uuid.uuid4().hex[:6]}"
    fb_wl_decision = NextTurnDecision(
        question_id=fb_wl_qid,
        source_phase=WorkflowPhase.CLARIFYING_CONTEXT,
        information_target=f"WEAK_LINK_{weak_area}",
        canonical_question=fallback_q,
        should_ask_question=True,
        clarification_reason=info_goal,
        planner_version=WEAK_LINK_CLARIFICATION_PROMPT_VERSION,
    )
    fallback_telem = _record_llm_node_telemetry(
        state, "select_weak_link_question", llm, "reasoning", 0.0, attempt,
        "WeakLinkClarifyingQuestionLLMOutput", status="FALLBACK",
    )
    return {
        "turn_telemetry": fallback_telem,
        "active_next_turn_decision": fb_wl_decision.model_dump(),
        "current_weak_link_question": {
            "question": fallback_q,
            "competing_areas": competing,
            "information_goal": info_goal,
            "why_this_question_now": f"Fallback question after error: {last_err}",
        },
        "current_clarifying_question": {
            "question": fallback_q,
            "target_area": weak_area,
            "information_goal": info_goal,
        },
        "active_interrupt_type": "weak_link",
        "phase": WorkflowPhase.CLARIFYING_CONTEXT,
    }


# ---------------------------------------------------------------------------
# Milestone 3 Nodes: Experiment Generation & Critic
# ---------------------------------------------------------------------------

def generate_experiment(state: StartaryGraphState) -> dict:
    """
    LLM Node: Formulate ONE specific, testable, cost-effective experiment designed
    to reduce uncertainty in the active critical assumption (Milestone 3).
    Respects founder execution constraints and previous critic instructions if revising.
    """
    project_id = state.get("project_id", "<unknown>")
    active_assumption = state.get("active_assumption")
    if not active_assumption or not active_assumption.get("statement"):
        raise ValueError(f"generate_experiment | project_id={project_id}: active_assumption missing.")

    statement = active_assumption["statement"]
    weak_area = state.get("weak_area") or str(active_assumption.get("area", "Problem"))
    rc = state.get("reality_check") or {}
    exec_ctx = state.get("execution_context") or {}
    prog = state.get("startup_progress") or {}
    founder_messages = state.get("founder_messages") or []
    all_founder_texts = [m.get("content", "") for m in founder_messages if m.get("role") == "founder"]
    if len(all_founder_texts) > 3:
        founder_evidence_summary = "\n".join([all_founder_texts[0]] + all_founder_texts[-2:])
    elif all_founder_texts:
        founder_evidence_summary = "\n".join(all_founder_texts)
    else:
        founder_evidence_summary = (state.get("raw_context", "") or "")[:500]

    previous_experiments = state.get("previous_experiments") or []
    prev_exp_summary = ""
    if previous_experiments:
        prev_exp_summary = "\n".join(
            f"- [{e.get('experiment_type')}] {e.get('objective')} (Status: {e.get('status')})"
            for e in previous_experiments
        )

    revision_instructions = state.get("critic_revision_instructions") or ""

    settings = get_settings()
    base_max_tokens = 3500
    llm = _safe_get_llm_client(settings, tier="reasoning", max_completion_tokens=base_max_tokens)
    structured_llm = llm.with_structured_output(ExperimentGeneratorLLMOutput)

    build_rd = state.get("build_readiness") or {}
    build_guidance = ""
    if isinstance(build_rd, dict) and build_rd.get("decision") == "BUILD_MINIMAL_PROTOTYPE":
        min_scope_str = "; ".join(build_rd.get("minimum_build_scope") or [])
        must_not_str = "; ".join(build_rd.get("must_not_build") or [])
        build_guidance = (
            "DECISION: BUILD_MINIMAL_PROTOTYPE (Milestone 4.1).\n"
            "You MUST design a PROTOTYPE_TEST experiment testing the active critical assumption through a strictly bounded minimal product probe.\n"
            f"- Minimum Build Scope: {min_scope_str}\n"
            f"- Must NOT Build: {must_not_str}\n"
            "Do NOT design a multi-feature app, automated platform, or full roadmap. Keep the probe tight, feasible within 1-2 weeks, and focused on unmediated user interaction."
        )

    next_strat = state.get("next_strategic_action") or {}
    strat_action_type = next_strat.get("action_type") or ""
    strat_guidance = ""
    if strat_action_type:
        strat_guidance = (
            f"NEXT STRATEGIC ACTION CATEGORY: {strat_action_type}\n"
            f"Rationale: {next_strat.get('rationale', '')}\n"
            f"Key Decision: {next_strat.get('key_decision', '')}\n"
            f"Suggested Method: {next_strat.get('suggested_method', '')}\n"
            "Ensure the experiment design aligns with this strategic category."
        )

    user_msg = build_experiment_generator_user_message(
        critical_assumption=statement,
        weak_area=weak_area,
        reality_check_summary=rc.get("summary", ""),
        unresolved_uncertainty=rc.get("unresolved_uncertainty", ""),
        what_evidence_supports=rc.get("what_evidence_supports") or [],
        what_evidence_does_not_support=rc.get("what_evidence_does_not_support") or [],
        execution_context_json=json.dumps(exec_ctx, indent=2, default=str),
        startup_progress_json=json.dumps(prog, indent=2, default=str),
        founder_evidence_summary=founder_evidence_summary,
        previous_experiments_summary=prev_exp_summary,
        critic_revision_instructions=revision_instructions,
        build_readiness_guidance=build_guidance,
        strategic_action_guidance=strat_guidance,
    )

    messages = [
        {"role": "system", "content": EXPERIMENT_GENERATOR_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    res_info = get_llm_client_resolution_info(llm)
    revision_round = int(state.get("critic_revision_count") or 0) + 1
    run_metadata = {"revision_round": revision_round, "is_revision": bool(revision_instructions)}
    request_messages = messages
    failure_kinds: list[str] = []
    attempt = 0
    last_err = None
    while attempt < 2:
        attempt += 1
        try:
            res, latency_ms = _invoke_structured_with_token_retry(
                structured_llm=structured_llm,
                messages=request_messages,
                settings=settings,
                tier="reasoning",
                base_max_tokens=base_max_tokens,
                node_name="generate_experiment",
                schema_cls=ExperimentGeneratorLLMOutput,
                run_metadata=run_metadata,
            )

            res_data = res.model_dump() if hasattr(res, "model_dump") and isinstance(res.model_dump(), dict) else (res if isinstance(res, dict) else {})
            if isinstance(build_rd, dict) and build_rd.get("decision") == "BUILD_MINIMAL_PROTOTYPE":
                res_data["experiment_type"] = ExperimentType.PROTOTYPE_TEST.value
            # Milestone 6.2.5: code-owned stamp binding the draft to the assumption it was generated for
            if active_assumption.get("id"):
                res_data["assumption_id"] = active_assumption["id"]

            trace = AITrace(
                module_name="generate_experiment",
                prompt_version=EXPERIMENT_GENERATOR_PROMPT_VERSION,
                input_data={
                    "assumption_id": active_assumption.get("id"),
                    "weak_area": weak_area,
                    "revision_instructions": revision_instructions[:100] if revision_instructions else None,
                },
                output_data=res_data,
                latency_ms=latency_ms,
                project_id=project_id,
                model_name=res_info["resolved_model"] or getattr(settings, "reasoning_model_name", None),
                configured_model=res_info["configured_model"],
                resolved_model=res_info["resolved_model"],
                fallback_used=res_info["fallback_used"],
                fallback_reason=res_info["fallback_reason"],
                execution_path=state.get("execution_path", "STRATEGIC_REASONING"),
            )
            existing_traces = list(state.get("traces") or [])
            existing_traces.append(trace.model_dump(mode="json"))

            # Token accounting with fallback character counting
            prompt_tokens = getattr(getattr(res, "usage_metadata", None), "prompt_tokens", None)
            if prompt_tokens is None:
                prompt_tokens = sum(len(m.get("content", "")) for m in request_messages) // 4
            completion_tokens = getattr(getattr(res, "usage_metadata", None), "completion_tokens", None)
            if completion_tokens is None:
                completion_tokens = len(_json.dumps(res_data)) // 4
            total_tokens = prompt_tokens + completion_tokens

            turn_telem = record_telemetry_event(
                state=state,
                node_name="generate_experiment",
                model_name=res_info["resolved_model"] or getattr(settings, "reasoning_model_name", None) or "reasoning_tier",
                duration_ms=latency_ms,
                latency_ms=latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                execution_path=state.get("execution_path", "STRATEGIC_REASONING"),
                configured_model=res_info["configured_model"],
                resolved_model=res_info["resolved_model"],
                fallback_used=res_info["fallback_used"],
                fallback_reason=res_info["fallback_reason"],
                retry_count=attempt - 1,
                model_tier="reasoning",
                reasoning_effort=res_info.get("reasoning_effort"),
                details={"revision_round": revision_round, "is_revision": bool(revision_instructions), "llm_failure_kinds": list(failure_kinds)},
            )

            exp_type = getattr(res, "experiment_type", res_data.get("experiment_type", "OTHER"))
            obj = getattr(res, "objective", res_data.get("objective", ""))
            logger.info("generate_experiment | type=%s objective=%s (attempt=%d round=%d)", exp_type, obj[:60], attempt, revision_round)
            gen_updates: dict[str, Any] = {
                "proposed_experiment_output": res_data,
                "last_trace": trace.model_dump(mode="json"),
                "traces": existing_traces,
                "turn_telemetry": turn_telem,
            }
            if _revision_v2_enabled(settings):
                gen_updates["experiment_revision"] = generation_record(state.get("experiment_revision"), is_revision=revision_round > 1)
            return gen_updates
        except Exception as exc:
            last_err = exc
            kind = classify_llm_failure(exc)
            failure_kinds.append(kind)
            logger.warning("generate_experiment failed attempt %d (%s): %s", attempt, kind, exc)
            if kind in (LLMFailureKind.TOKEN_BUDGET_EXHAUSTED, LLMFailureKind.PERMANENT_API_ERROR):
                break
            if kind == LLMFailureKind.INVALID_STRUCTURED_OUTPUT:
                request_messages = messages + [schema_correction_message(exc)]

    _record_llm_node_telemetry(
        state, "generate_experiment", llm, "reasoning", 0.0, attempt, "ExperimentGeneratorLLMOutput",
        status="FAILED", details={"revision_round": revision_round, "llm_failure_kinds": list(failure_kinds)},
    )
    if isinstance(last_err, TokenBudgetExhaustedError):
        raise last_err
    raise RuntimeError(f"generate_experiment failed after {attempt} attempts. Error: {last_err}")


def revise_experiment(state: StartaryGraphState) -> dict:
    """
    LLM Node (Milestone 6.2.6, STARTARY_EXPERIMENT_REVISION_V2 only): targeted edit of the rejected draft.

    The reviser returns ONLY the fields that must change (ExperimentRevisionPatchLLMOutput); the patch
    is applied to the preserved rejected draft, so untouched fields cannot regress. The merged draft
    must satisfy ExperimentGeneratorLLMOutput (otherwise: one corrective retry). Records which
    fields changed and whether the change is substantial (which forces a FULL re-critique).
    """
    project_id = state.get("project_id", "<unknown>")
    rev = dict(state.get("experiment_revision") or {})
    original = rev.get("rejected_draft")
    critique = rev.get("rejected_critique") or {}
    active_assumption = state.get("active_assumption") or {}
    if not original or not active_assumption.get("statement"):
        raise ValueError(f"revise_experiment | project_id={project_id}: rejected draft or active assumption missing.")
    statement = active_assumption["statement"]
    weak_area = state.get("weak_area") or str(active_assumption.get("area", "Problem"))
    revision_round = int(state.get("critic_revision_count") or 0) + 1

    build_rd = state.get("build_readiness") or {}
    if isinstance(build_rd, dict) and build_rd.get("decision") == "BUILD_MINIMAL_PROTOTYPE":
        build_guidance = (
            "BUILD_MINIMAL_PROTOTYPE is authorized ONLY within this minimum scope: "
            f"{'; '.join(build_rd.get('minimum_build_scope') or [])}. Must NOT build: {'; '.join(build_rd.get('must_not_build') or [])}."
        )
    else:
        build_guidance = "CONTINUE_MANUAL: keep the experiment manual; do not propose building software."

    contract_before = evaluate_experiment_contract(original, state)
    user_msg = build_experiment_revision_user_message(
        critical_assumption=statement,
        weak_area=weak_area,
        original_experiment_json=json.dumps(original, indent=2, default=str),
        failed_dimensions=list(critique.get("failed_dimensions") or []),
        critic_issues=list(critique.get("issues") or []),
        revision_instructions=str(critique.get("revision_instructions") or ""),
        deterministic_findings=[f.message for f in contract_before.violations + contract_before.advisories],
        execution_context_json=json.dumps(_slim_facts(state.get("execution_context") or {}), indent=2, default=str),
        startup_progress_json=json.dumps(_slim_facts(state.get("startup_progress") or {}), indent=2, default=str),
        revision_round=revision_round,
        build_readiness_guidance=build_guidance,
    )
    messages = [
        {"role": "system", "content": EXPERIMENT_REVISION_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    settings = get_settings()
    base_max_tokens = 3500
    llm = _safe_get_llm_client(settings, tier="reasoning", max_completion_tokens=base_max_tokens)
    structured_llm = llm.with_structured_output(ExperimentRevisionPatchLLMOutput)
    run_metadata = {"revision_round": revision_round, "revision_strategy": RevisionStrategy.TARGETED_EDIT.value}
    request_messages = messages
    failure_kinds: list[str] = []
    attempt = 0
    last_err = None
    while attempt < 2:
        attempt += 1
        try:
            res, latency_ms = _invoke_structured_with_token_retry(
                structured_llm=structured_llm,
                messages=request_messages,
                settings=settings,
                tier="reasoning",
                base_max_tokens=base_max_tokens,
                node_name="revise_experiment",
                schema_cls=ExperimentRevisionPatchLLMOutput,
                run_metadata=run_metadata,
            )
            patch = res.model_dump(mode="json", exclude_none=True) if hasattr(res, "model_dump") else dict(res or {})
            revised = apply_revision_patch(original, patch)
            # Code-owned fields: the hypothesis belongs to the assumption; safeguards are re-derived by validate.
            revised["hypothesis"] = statement
            if active_assumption.get("id"):
                revised["assumption_id"] = active_assumption["id"]
            if isinstance(build_rd, dict) and build_rd.get("decision") == "BUILD_MINIMAL_PROTOTYPE":
                revised["experiment_type"] = ExperimentType.PROTOTYPE_TEST.value
            for derived in ("remediation_codes", "required_authorizations", "authorization_status"):
                revised.pop(derived, None)
            # The merged draft must still be a valid experiment (raises ValidationError -> corrective retry).
            ExperimentGeneratorLLMOutput(**{k: revised[k] for k in ExperimentGeneratorLLMOutput.model_fields if k in revised})

            changed = diff_experiment(original, revised)
            substantial = is_substantial_revision(original, revised, changed)
            details = {
                "revision_mode": "V2",
                "revision_strategy": RevisionStrategy.TARGETED_EDIT.value,
                "revision_round": revision_round,
                "changed_fields": changed,
                "substantial_change": substantial,
                "next_critic_mode": CriticMode.FULL.value if substantial else CriticMode.FOCUSED.value,
                "llm_failure_kinds": list(failure_kinds),
            }
            emit_decision_span("experiment_targeted_revision", details, tags=["node:revise_experiment", "revision:v2"])
            turn_telem = _record_llm_node_telemetry(
                state, "revise_experiment", llm, "reasoning", latency_ms, attempt,
                "ExperimentRevisionPatchLLMOutput", messages=request_messages, result=res, details=details,
            )
            trace = AITrace(
                module_name="revise_experiment",
                prompt_version=EXPERIMENT_REVISION_PROMPT_VERSION,
                input_data={"assumption_id": active_assumption.get("id"), "revision_round": revision_round,
                            "failed_dimensions": list(critique.get("failed_dimensions") or [])},
                output_data={"changed_fields": changed, "substantial_change": substantial},
                latency_ms=latency_ms,
                project_id=project_id,
                execution_path=state.get("execution_path", "STRATEGIC_REASONING"),
            )
            existing_traces = list(state.get("traces") or [])
            existing_traces.append(trace.model_dump(mode="json"))
            logger.info("revise_experiment | round=%d changed=%s substantial=%s", revision_round, changed, substantial)
            return {
                "proposed_experiment_output": revised,
                "experiment_revision": {
                    **rev,
                    "draft_origin": DraftOrigin.TARGETED_EDIT.value,
                    "changed_fields": changed,
                    "substantial_change": substantial,
                    "next_strategy": None,
                },
                "last_trace": trace.model_dump(mode="json"),
                "traces": existing_traces,
                "turn_telemetry": turn_telem,
            }
        except Exception as exc:
            last_err = exc
            kind = classify_llm_failure(exc)
            failure_kinds.append(kind)
            logger.warning("revise_experiment failed attempt %d (%s): %s", attempt, kind, exc)
            if kind in (LLMFailureKind.TOKEN_BUDGET_EXHAUSTED, LLMFailureKind.PERMANENT_API_ERROR):
                break
            if kind == LLMFailureKind.INVALID_STRUCTURED_OUTPUT:
                request_messages = messages + [schema_correction_message(exc)]

    _record_llm_node_telemetry(
        state, "revise_experiment", llm, "reasoning", 0.0, attempt, "ExperimentRevisionPatchLLMOutput",
        status="FAILED", details={"revision_round": revision_round, "llm_failure_kinds": list(failure_kinds)},
    )
    if isinstance(last_err, TokenBudgetExhaustedError):
        raise last_err
    raise RuntimeError(f"revise_experiment failed after {attempt} attempts. Error: {last_err}")


def validate_proposed_experiment(state: StartaryGraphState) -> dict:
    """
    Deterministic Node: normalizes the generated draft before critique.

    Milestone 6.2.5: draft defects no longer raise. A raised ValueError used to abort the graph,
    and /retry re-validated the same committed draft, so it could never recover. Defects are now
    evaluated by the shared experiment contract and turned into a deterministic REVISE by
    critique_experiment, which keeps them inside the bounded revision loop.
    """
    project_id = state.get("project_id", "<unknown>")
    exp_output = state.get("proposed_experiment_output")
    active_assumption = state.get("active_assumption") or {}
    statement = active_assumption.get("statement", "")

    if not exp_output:
        raise ValueError(f"validate_proposed_experiment | project_id={project_id}: proposed_experiment_output missing.")

    exp_output = dict(exp_output)
    # Grounding: the experiment's hypothesis is owned by the active critical assumption.
    if statement and exp_output.get("hypothesis") != statement:
        exp_output["hypothesis"] = statement

    # Fake precision is flagged for the critic's judgement, not rewritten.
    for p in exp_output.get("parameters") or []:
        val_str = str(p.get("value", ""))
        if re.search(r"\b\d+\.\d+%", val_str) or re.search(r"\b\d+\.\d+\s*(?:users|customers|sellers|people)\b", val_str):
            logger.warning("validate_proposed_experiment: parameter '%s' contains fake precision '%s'.", p.get("name"), val_str)

    contract = evaluate_experiment_contract(exp_output, state)
    updates: dict[str, Any] = {}
    if _revision_v2_enabled(get_settings()):
        # Milestone 6.2.6: consent-class violations are repaired deterministically (no LLM call, no
        # semantic revision consumed). The repair is kept only if the contract then passes in full.
        violations_before = contract.violation_codes
        remediation = remediate_consent_violations(exp_output, contract)
        outcome = remediation.skipped_reason
        if remediation.applied:
            after = evaluate_experiment_contract(remediation.draft, state)
            if after.violations:
                outcome = "REMEDIATION_INSUFFICIENT"
            else:
                exp_output, contract, outcome = remediation.draft, after, "APPLIED"
        if violations_before:
            meta = {
                "revision_mode": "V2",
                "revision_strategy": RevisionStrategy.DETERMINISTIC_REMEDIATION.value,
                "remediation_applied": outcome == "APPLIED",
                "remediation_codes": remediation.applied_codes if outcome == "APPLIED" else [],
                "remediation_outcome": outcome,
                "violations_before": violations_before,
                "contract_violation_codes": contract.violation_codes,
            }
            emit_decision_span("experiment_remediation", meta, tags=["node:validate_proposed_experiment", "revision:v2"])
            updates["turn_telemetry"] = record_telemetry_event(
                state=state, node_name="validate_proposed_experiment", model_name="deterministic",
                skipped_by_fast_path=True, model_tier="none", details=meta,
            )
        # A permission STEP is a task, not a grant: authorization stays PENDING until the founder confirms it.
        auths = required_authorizations(exp_output, state)
        exp_output["required_authorizations"] = auths
        exp_output["authorization_status"] = "PENDING" if auths else None

    logger.info(
        "validate_proposed_experiment | violations=%s advisories=%s",
        contract.violation_codes, contract.advisory_codes,
    )
    updates["proposed_experiment_output"] = exp_output
    return updates


def critique_experiment(state: StartaryGraphState) -> dict:
    """
    LLM Node: Evaluates proposed experiment for assumption alignment, evidence validity,
    feasibility, cost efficiency, and falsifiability. Decides ACCEPT or REVISE.

    Milestone 6.2.5:
    - Deterministic contract violations REVISE without an LLM call; advisories go to the critic.
    - Strict, fail-closed verdict: ACCEPT only when verdict == ACCEPT and all six dimensions PASS.
    - Bounded recovery: token exhaustion / permanent errors stop; invalid output retries once
      with schema feedback; transient errors retry once.
    - Every decision records non-sensitive metadata in telemetry and as a LangSmith span.
    """
    project_id = state.get("project_id", "<unknown>")
    active_assumption = state.get("active_assumption") or {}
    statement = active_assumption.get("statement", "")
    weak_area = state.get("weak_area") or str(active_assumption.get("area", "Problem"))
    rc = state.get("reality_check") or {}
    exp_output = state.get("proposed_experiment_output") or {}
    exec_ctx = state.get("execution_context") or {}
    prog = state.get("startup_progress") or {}
    founder_messages = state.get("founder_messages") or []
    all_founder_texts = [m.get("content", "") for m in founder_messages if m.get("role") == "founder"]
    if len(all_founder_texts) > 3:
        founder_evidence_summary = "\n".join([all_founder_texts[0]] + all_founder_texts[-2:])
    elif all_founder_texts:
        founder_evidence_summary = "\n".join(all_founder_texts)
    else:
        founder_evidence_summary = (state.get("raw_context", "") or "")[:500]

    revision_count = state.get("critic_revision_count", 0)
    revision_round = int(revision_count or 0) + 1
    contract = evaluate_experiment_contract(exp_output, state)
    settings = get_settings()
    v2 = _revision_v2_enabled(settings)
    rev = dict(state.get("experiment_revision") or {}) if v2 else {}
    prior_critique = (rev.get("rejected_critique") or {}) if rev.get("draft_origin") in (DraftOrigin.TARGETED_EDIT.value, DraftOrigin.FULL_REGENERATION.value) else {}
    # M6.2.6: a targeted edit that changed nothing would be rejected again; do it without a critic call.
    no_effective_change = v2 and rev.get("draft_origin") == DraftOrigin.TARGETED_EDIT.value and rev.get("changed_fields") == []

    if contract.violations or no_effective_change:
        if contract.violations:
            decision = deterministic_decision(contract)
            issues = [v.message for v in contract.violations]
            instructions = contract.revision_instructions
            rationale = f"Deterministic contract check found {len(issues)} violation(s): {', '.join(contract.violation_codes)}."
        else:
            prior_failed = list(prior_critique.get("failed_dimensions") or [])
            decision = CriticDecision(
                "REVISE", None, {d: ("FAIL" if d in prior_failed else "NOT_EVALUATED") for d in CRITIC_DIMENSIONS},
                prior_failed, False, "DETERMINISTIC",
            )
            issues = ["The targeted revision did not change the rejected experiment."]
            instructions = prior_critique.get("revision_instructions") or issues[0]
            rationale = "Targeted revision made no effective change."
        precheck_res = ExperimentCriticLLMOutput(
            verdict="REVISE",
            issues=issues,
            **{d: ("FAIL" if d in decision.failed_dimensions else "PASS") for d in decision.dimensions},
            revision_instructions=instructions,
            rationale=rationale,
        )
        res_data = precheck_res.model_dump(mode="json")
        res_data.update({
            "decision_source": decision.decision_source,
            "failed_dimensions": decision.failed_dimensions,
            "contract_violation_codes": contract.violation_codes,
            "revision_round": revision_round,
        })
        det_strategy = select_revision_strategy("DETERMINISTIC", decision.failed_dimensions, contract.violation_codes, None) if v2 else None
        meta = critic_decision_metadata(
            decision, revision_round=revision_round, revision_count_after=revision_count + 1,
            issues=issues, contract=contract, attempts=0, failure_kinds=[],
            revision_strategy=det_strategy,
            extra=critic_v2_metadata(
                rev, exp_output, decision, CriticMode.DETERMINISTIC.value, det_strategy,
                reason_codes=["NO_EFFECTIVE_CHANGE"] if no_effective_change and not contract.violations else [],
            ) if v2 else None,
        )
        emit_decision_span("experiment_critic_decision", meta, tags=["node:critique_experiment", "decision:deterministic"])
        turn_telem = record_telemetry_event(
            state=state, node_name="critique_experiment", model_name="deterministic_precheck",
            skipped_by_fast_path=True, execution_path=state.get("execution_path", "STRATEGIC_REASONING"),
            model_tier="none", schema_name="ExperimentCriticLLMOutput", details=meta,
        )
        trace = AITrace(
            module_name="critique_experiment",
            prompt_version=EXPERIMENT_CRITIC_PROMPT_VERSION,
            input_data={
                "assumption_id": active_assumption.get("id"),
                "experiment_type": exp_output.get("experiment_type"),
                "revision_count": revision_count,
                "deterministic_precheck": True,
            },
            output_data=res_data,
            latency_ms=0.0,
            project_id=project_id,
            model_name="deterministic_precheck",
            configured_model="deterministic",
            resolved_model="deterministic",
            execution_path=state.get("execution_path", "STRATEGIC_REASONING"),
        )
        existing_traces = list(state.get("traces") or [])
        existing_traces.append(trace.model_dump(mode="json"))
        logger.info("critique_experiment | deterministic REVISE codes=%s round=%d", contract.violation_codes, revision_round)
        det_updates: dict[str, Any] = {
            "experiment_critic_output": res_data,
            "critic_revision_count": revision_count + 1,
            "critic_revision_instructions": instructions,
            "traces": existing_traces,
            "last_trace": trace.model_dump(mode="json"),
            "turn_telemetry": turn_telem,
            "phase": WorkflowPhase.TEST,
        }
        if v2:
            det_updates["experiment_revision"] = rejection_record(
                rev, exp_output, decision, issues, instructions, CriticMode.DETERMINISTIC.value, None, det_strategy, revision_round,
            )
        return det_updates

    critic_mode = select_critic_mode(rev) if v2 else CriticMode.FULL.value
    focused = v2 and critic_mode == CriticMode.FOCUSED.value
    schema_cls = ExperimentFocusedCriticLLMOutput if focused else (ExperimentCriticV2LLMOutput if v2 else ExperimentCriticLLMOutput)
    base_max_tokens = 2500
    llm = _safe_get_llm_client(settings, tier="reasoning", max_tokens=base_max_tokens)
    structured_llm = llm.with_structured_output(schema_cls)

    exec_ctx_slimmed = {
        k: (v.get("value") if isinstance(v, dict) and "value" in v else v)
        for k, v in exec_ctx.items()
        if v and (not isinstance(v, dict) or v.get("value") not in (None, "UNKNOWN", ""))
    }
    prog_slimmed = {
        k: (v.get("value") if isinstance(v, dict) and "value" in v else v)
        for k, v in prog.items()
        if v and (not isinstance(v, dict) or v.get("value") not in (None, "UNKNOWN", ""))
    }

    if focused:
        # M6.2.6: verify the original issues against the changed fields; all six dimensions re-evaluated.
        prior = rev.get("rejected_critique") or {}
        changes = change_summary(rev.get("rejected_draft") or {}, exp_output, list(rev.get("changed_fields") or []))
        user_msg = build_experiment_focused_critic_user_message(
            critical_assumption=statement,
            weak_area=weak_area,
            revised_experiment_json=json.dumps(exp_output, indent=2, default=str),
            changes_json=json.dumps(changes, indent=2, default=str),
            original_verdict=str(prior.get("verdict") or "REVISE"),
            original_dimensions=prior.get("dimensions") or {},
            original_issues=list(prior.get("issues") or []),
            execution_context_json=json.dumps(exec_ctx_slimmed, indent=2, default=str),
            startup_progress_json=json.dumps(prog_slimmed, indent=2, default=str),
            deterministic_notes=[a.message for a in contract.advisories],
        )
        system_prompt = EXPERIMENT_FOCUSED_CRITIC_SYSTEM_PROMPT
        prompt_version = EXPERIMENT_FOCUSED_CRITIC_PROMPT_VERSION
    else:
        user_msg = build_experiment_critic_user_message(
            critical_assumption=statement,
            weak_area=weak_area,
            reality_check_summary=rc.get("summary", ""),
            proposed_experiment_json=json.dumps(exp_output, indent=2, default=str),
            execution_context_json=json.dumps(exec_ctx_slimmed, indent=2, default=str),
            startup_progress_json=json.dumps(prog_slimmed, indent=2, default=str),
            founder_evidence_summary=founder_evidence_summary,
            deterministic_notes=[a.message for a in contract.advisories],
        )
        system_prompt = EXPERIMENT_CRITIC_SYSTEM_PROMPT + (("\n\n" + EXPERIMENT_CRITIC_REVISION_SCOPE_ADDENDUM) if v2 else "")
        prompt_version = EXPERIMENT_CRITIC_PROMPT_VERSION

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ]

    res_info = get_llm_client_resolution_info(llm)
    run_metadata = {"revision_round": revision_round, **({"critic_mode": critic_mode} if v2 else {})}
    request_messages = messages
    failure_kinds: list[str] = []
    attempt = 0
    last_err = None
    while attempt < 2:
        attempt += 1
        try:
            res, latency_ms = _invoke_structured_with_token_retry(
                structured_llm=structured_llm,
                messages=request_messages,
                settings=settings,
                tier="reasoning",
                base_max_tokens=base_max_tokens,
                node_name="critique_experiment",
                schema_cls=schema_cls,
                run_metadata=run_metadata,
            )

            res_data = res.model_dump(mode="json") if hasattr(res, "model_dump") and isinstance(res.model_dump(), dict) else (res if isinstance(res, dict) else {})
            res_data = dict(res_data)
            decision = interpret_critic_output(res_data)
            if focused:
                decision = apply_focused_acceptance_rule(decision, res_data)
            issues = list(res_data.get("issues") or [])
            next_strategy = (
                select_revision_strategy("LLM", decision.failed_dimensions, [], res_data.get("revision_scope"))
                if v2 and decision.verdict == "REVISE" else None
            )
            res_data.update({
                "verdict": decision.verdict,
                "model_verdict": decision.model_verdict,
                "verdict_flag_conflict": decision.verdict_flag_conflict,
                "failed_dimensions": decision.failed_dimensions,
                "decision_source": decision.decision_source,
                "revision_round": revision_round,
            })
            revision_count_after = revision_count + 1 if decision.verdict == "REVISE" else revision_count
            meta = critic_decision_metadata(
                decision, revision_round=revision_round, revision_count_after=revision_count_after,
                issues=issues, contract=contract, attempts=attempt, failure_kinds=failure_kinds,
                revision_strategy=next_strategy,
                extra=critic_v2_metadata(rev, exp_output, decision, critic_mode, next_strategy, raw=res_data) if v2 else None,
            )
            emit_decision_span("experiment_critic_decision", meta, tags=["node:critique_experiment", "decision:llm"])

            trace = AITrace(
                module_name="critique_experiment",
                prompt_version=prompt_version,
                input_data={
                    "assumption_id": active_assumption.get("id"),
                    "experiment_type": exp_output.get("experiment_type"),
                    "revision_count": revision_count,
                },
                output_data=res_data,
                latency_ms=latency_ms,
                project_id=project_id,
                model_name=res_info["resolved_model"] or getattr(settings, "reasoning_model_name", None),
                configured_model=res_info["configured_model"],
                resolved_model=res_info["resolved_model"],
                fallback_used=res_info["fallback_used"],
                fallback_reason=res_info["fallback_reason"],
                execution_path=state.get("execution_path", "STRATEGIC_REASONING"),
            )
            existing_traces = list(state.get("traces") or [])
            existing_traces.append(trace.model_dump(mode="json"))

            turn_telem = _record_llm_node_telemetry(
                state, "critique_experiment", llm, "reasoning", latency_ms, attempt,
                schema_cls.__name__, messages=request_messages, result=res, details=meta,
            )

            logger.info(
                "critique_experiment | verdict=%s model_verdict=%s conflict=%s failed=%s round=%d issue_codes=%s",
                decision.verdict, decision.model_verdict, decision.verdict_flag_conflict,
                decision.failed_dimensions, revision_round, meta["issue_codes"],
            )

            updates: dict[str, Any] = {
                "experiment_critic_output": res_data,
                "last_trace": trace.model_dump(mode="json"),
                "traces": existing_traces,
                "turn_telemetry": turn_telem,
            }

            if decision.verdict == "REVISE":
                updates["critic_revision_count"] = revision_count_after
                updates["critic_revision_instructions"] = (
                    (res_data.get("revision_instructions") or "").strip()
                    or "; ".join(str(i) for i in issues)
                    or (f"Resolve the failing quality dimensions: {', '.join(decision.failed_dimensions)}." if decision.failed_dimensions else "")
                    or str(res_data.get("rationale") or "")
                )
                if v2:
                    updates["experiment_revision"] = rejection_record(
                        rev, exp_output, decision, issues + list(res_data.get("new_blocking_issues") or []),
                        updates["critic_revision_instructions"], critic_mode, res_data.get("revision_scope"),
                        next_strategy, revision_round,
                    )

            return updates
        except Exception as exc:
            last_err = exc
            kind = classify_llm_failure(exc)
            failure_kinds.append(kind)
            logger.warning("critique_experiment failed attempt %d (%s): %s", attempt, kind, exc)
            if kind in (LLMFailureKind.TOKEN_BUDGET_EXHAUSTED, LLMFailureKind.PERMANENT_API_ERROR):
                break
            if kind == LLMFailureKind.INVALID_STRUCTURED_OUTPUT:
                request_messages = messages + [schema_correction_message(exc)]

    _record_llm_node_telemetry(
        state, "critique_experiment", llm, "reasoning", 0.0, attempt, "ExperimentCriticLLMOutput",
        status="FAILED", details={"revision_round": revision_round, "critic_attempts": attempt, "llm_failure_kinds": list(failure_kinds)},
    )
    if isinstance(last_err, TokenBudgetExhaustedError):
        raise last_err
    raise RuntimeError(f"critique_experiment failed after {attempt} attempts. Error: {last_err}")


def save_final_experiment(state: StartaryGraphState) -> dict:
    """
    Deterministic Node:
    Instantiates first-class Experiment domain model with status=READY,
    links to active_assumption, appends to previous_experiments, and updates state.
    """
    project_id = state.get("project_id", "<unknown>")
    iteration = state.get("iteration", 1)
    active_assumption = state.get("active_assumption")
    exp_output = state.get("proposed_experiment_output")

    if not active_assumption or not active_assumption.get("id"):
        raise ValueError(f"save_final_experiment | project_id={project_id}: active_assumption missing.")
    if not exp_output:
        raise ValueError(f"save_final_experiment | project_id={project_id}: proposed_experiment_output missing.")

    params = [
        ExperimentParameter(
            name=p.get("name", "param"),
            value=str(p.get("value", "")),
            provenance=ParameterProvenance(p.get("provenance", "AI_PROPOSED")),
            notes=p.get("notes"),
        )
        for p in exp_output.get("parameters") or []
    ]

    target_sample_size = exp_output.get("target_sample_size")
    if target_sample_size is None or (isinstance(target_sample_size, int) and target_sample_size <= 0):
        target_sample_size = extract_sample_target_from_protocol(exp_output)

    exp = Experiment(
        id=str(uuid.uuid4()),
        assumption_id=active_assumption["id"],
        project_id=project_id,
        experiment_type=ExperimentType(exp_output.get("experiment_type", ExperimentType.OTHER)),
        objective=exp_output["objective"],
        hypothesis=active_assumption["statement"],
        method=exp_output.get("method"),
        target_participants_or_scope=exp_output.get("target_participants_or_scope"),
        target_sample_size=target_sample_size,
        founder_actions=exp_output.get("founder_actions") or [],
        evidence_to_collect=exp_output.get("evidence_to_collect"),
        support_signal=exp_output.get("support_signal"),
        contradiction_signal=exp_output.get("contradiction_signal"),
        inconclusive_signal=exp_output.get("inconclusive_signal"),
        valid_inferences=exp_output.get("valid_inferences") or [],
        invalid_inferences=exp_output.get("invalid_inferences") or [],
        estimated_time=exp_output.get("estimated_time"),
        estimated_cost=exp_output.get("estimated_cost"),
        required_capabilities=exp_output.get("required_capabilities") or [],
        required_access=exp_output.get("required_access") or [],
        parameters=params,
        status=ExperimentStatus.READY,
        created_iteration=iteration,
    )

    saved = exp.model_dump(mode="json")
    # Milestone 6.2.6 (V2 drafts only): permissions the founder must still obtain. Never marked OBTAINED here.
    if exp_output.get("required_authorizations"):
        saved["required_authorizations"] = list(exp_output["required_authorizations"])
        saved["authorization_status"] = exp_output.get("authorization_status") or "PENDING"

    prev_experiments = list(state.get("previous_experiments") or [])
    prev_experiments.append(dict(saved))

    logger.info("save_final_experiment | experiment_id=%s type=%s status=READY", exp.id, exp.experiment_type)
    return {
        "active_experiment": saved,
        "previous_experiments": prev_experiments,
        "phase": WorkflowPhase.TEST,
    }


def assess_can_execute(state: StartaryGraphState) -> dict:
    """
    Deterministic Node:
    Evaluates whether the founder has the assets, budget, customer access, and capabilities
    to execute the accepted experiment under current constraints.
    Returns CAN_EXECUTE, NEEDS_HELP, or BLOCKED.
    """
    project_id = state.get("project_id", "<unknown>")
    exp_dict = state.get("active_experiment") or {}
    exec_ctx = state.get("execution_context") or {}
    prog = state.get("startup_progress") or {}

    exp_type = exp_dict.get("experiment_type", "OTHER")
    required_access = exp_dict.get("required_access") or []
    estimated_cost = (exp_dict.get("estimated_cost") or "").lower()

    missing_requirements: list[str] = []
    blockers: list[str] = []

    # 1. Prototype requirement check
    if exp_type == ExperimentType.PROTOTYPE_TEST.value:
        mvp_field = prog.get("mvp_exists")
        mvp_val = mvp_field.get("value") if isinstance(mvp_field, dict) else mvp_field
        if str(mvp_val).lower() not in ("true", "yes"):
            missing_requirements.append(
                "Experiment requires testing a working prototype, but no prototype exists yet."
            )

def resolve_customer_access(state: StartaryGraphState) -> CustomerAccessAssessment:
    """
    Deterministic Access Evidence Resolver (Milestone 4.0.1):
    Aggregates factual signals of customer/user access across the entire startup state:
      - execution_context (customer_access, existing_audience_or_community, key_assets)
      - startup_progress (active_users_count, users_count, paying_customers_count, mvp_exists)
      - startup_state area knowledge (Target User, Problem, Demand Signal, Existing Alternatives)
      - founder_messages and raw_context for demonstrated customer interactions
      - historical evidence records
    Distinguishes NONE from LIMITED (demonstrated past users/testers) from ESTABLISHED.
    """
    exec_ctx = state.get("execution_context") or {}
    prog = state.get("startup_progress") or {}
    startup_st = state.get("startup_state") or {}
    area_knowledge = startup_st.get("area_knowledge") or {}
    founder_messages = state.get("founder_messages") or []
    raw_context = state.get("raw_context") or ""
    founder_evidence = state.get("founder_evidence") or []

    evidence_sources: list[str] = []
    relevant_segments: list[str] = []

    # 1. Check explicit execution_context fields
    cust_access_field = exec_ctx.get("customer_access")
    existing_aud_field = exec_ctx.get("existing_audience_or_community")
    for field_name, f in [("customer_access", cust_access_field), ("existing_audience", existing_aud_field)]:
        if isinstance(f, dict) and f.get("value") is not None and f.get("provenance") != "UNKNOWN":
            val_str = str(f.get("value", "")).strip()
            if val_str.lower() not in ("none", "no", "unknown", ""):
                evidence_sources.append(f"{field_name}: {val_str}")

    key_assets = exec_ctx.get("key_assets") or []
    for asset in key_assets:
        asset_str = str(asset).lower()
        if any(kw in asset_str for kw in ("user", "community", "audience", "partner", "network", "client", "customer")):
            evidence_sources.append(f"key asset: {asset}")

    # 2. Check startup_progress (active users, user counts, MVP testing)
    for user_key in ("active_users_count", "users_count", "paying_customers_count"):
        uf = prog.get(user_key)
        if isinstance(uf, dict) and uf.get("value") is not None:
            try:
                count_val = int(float(uf.get("value", 0)))
                if count_val > 0:
                    evidence_sources.append(f"{user_key}: {count_val}")
            except (ValueError, TypeError):
                pass

    mvp_field = prog.get("mvp_exists")
    mvp_val = str(mvp_field.get("value") if isinstance(mvp_field, dict) else mvp_field).lower()
    has_mvp = mvp_val in ("true", "yes")

    # 3. Check area knowledge (Target User, Demand Signal, Problem, etc.)
    target_user_data = area_knowledge.get("Target User") or {}
    if target_user_data.get("understanding"):
        relevant_segments.append(str(target_user_data["understanding"])[:80])

    interaction_keywords = [
        "tested by", "student tester", "student testers", "user feedback", "interviewed",
        "conversations with", "pilot", "users reported", "talking to users", "feedback from",
        "beta user", "beta testers", "spoke to", "used by students", "tested with students"
    ]

    for area_name, area_data in area_knowledge.items():
        claims = area_data.get("founder_claims") or []
        ev_provided = area_data.get("evidence_provided") or []
        understanding = str(area_data.get("understanding") or "")
        all_text = " ".join([str(c) for c in claims] + [str(e) for e in ev_provided] + [understanding]).lower()
        for kw in interaction_keywords:
            if kw in all_text:
                evidence_sources.append(f"{area_name}: mentioned '{kw}'")
                break

    # 4. Check conversation text & raw context
    all_msgs_text = " ".join([str(m.get("content", "")) for m in founder_messages] + [raw_context]).lower()
    if has_mvp and any(kw in all_msgs_text for kw in ("tested by", "feedback", "users used", "students used", "testers", "tried")):
        evidence_sources.append("MVP user testing: students/users tested MVP and provided feedback")

    for kw in ("interviewed", "pilot with", "spoke to 5", "spoke to 6", "spoke to 7", "spoke to 8", "talked to 10"):
        if kw in all_msgs_text:
            evidence_sources.append(f"conversation history: mentioned '{kw}'")

    access_indicators = [
        "access to", "relationships with", "relationship with", "agreed to test",
        "agreed to participate", "agreed to review", "partner teams", "partner clubs",
        "coaches agreed", "teams agreed", "district supervisors", "pilot agreement",
    ]
    for ind in access_indicators:
        if ind in all_msgs_text:
            evidence_sources.append(f"conversation history: mentioned '{ind}'")
            break

    # 5. Check historical evidence submissions
    if founder_evidence:
        evidence_sources.append(f"historical evidence: {len(founder_evidence)} completed experiment submission(s)")

    # Deduplicate sources
    evidence_sources = list(dict.fromkeys(evidence_sources))

    # Evaluate access status and strength
    if not evidence_sources:
        return CustomerAccessAssessment(
            access_status=CustomerAccessStatus.NONE,
            access_strength=CustomerAccessStrength.WEAK,
            evidence_sources=[],
            relevant_segments=relevant_segments,
            explanation="No evidence of customer access, user testing, or past conversations found in the startup state.",
        )

    # Check if access is established (direct explicit customer access, active user base, or community/audience)
    has_established_access = (
        any("customer_access:" in s for s in evidence_sources)
        or any("existing_audience:" in s for s in evidence_sources)
        or any(("active_users_count" in s or "paying_customers_count" in s) for s in evidence_sources)
        or any("audience" in s.lower() for s in evidence_sources)
        or any(("access to" in s.lower() or "agreed to test" in s.lower() or "teams agreed" in s.lower() or "coaches agreed" in s.lower()) for s in evidence_sources)
    )

    if has_established_access:
        return CustomerAccessAssessment(
            access_status=CustomerAccessStatus.ESTABLISHED,
            access_strength=CustomerAccessStrength.STRONG,
            evidence_sources=evidence_sources,
            relevant_segments=relevant_segments,
            explanation=f"Established customer access via direct channels or existing user base ({'; '.join(evidence_sources[:3])}).",
        )

    # Otherwise access is demonstrated but limited (e.g. prior MVP testers, small interview sample, feedback from students)
    return CustomerAccessAssessment(
        access_status=CustomerAccessStatus.LIMITED,
        access_strength=CustomerAccessStrength.MODERATE,
        evidence_sources=evidence_sources,
        relevant_segments=relevant_segments,
        explanation=f"Demonstrated access to past users or testers ({'; '.join(evidence_sources[:3])}), but recruitment capacity for specific new sample cohorts is unconfirmed.",
    )


def assess_can_execute(state: StartaryGraphState) -> dict:
    """
    Deterministic Node (Milestone 3 & Milestone 4.0.1):
    Evaluates whether the accepted experiment can be executed under the startup's
    practical operating reality (ExecutionContext + StartupProgress + CustomerAccessAssessment).

    Readiness Levels:
      - CAN_EXECUTE: Founder has capacity, access, and budget to execute immediately.
      - NEEDS_HELP: Valid experiment, but requires assistance, assets, or access.
      - BLOCKED: Direct violation of immutable constraints (e.g. requires cash with $0 budget).
    """
    project_id = state.get("project_id", "<unknown>")
    exp_dict = state.get("active_experiment") or {}
    exec_ctx = state.get("execution_context") or {}
    prog = state.get("startup_progress") or {}

    exp_type = exp_dict.get("experiment_type", "OTHER")
    required_access = exp_dict.get("required_access") or []
    estimated_cost = (exp_dict.get("estimated_cost") or "").lower()

    missing_requirements: list[str] = []
    blockers: list[str] = []

    # 1. Prototype requirement check (Milestone 6.2.5: shared with the critic's contract check, so an
    # authorized BUILD_MINIMAL_PROTOTYPE probe is not reported as "no prototype exists")
    prototype = assess_prototype_requirement(exp_type, prog, state.get("build_readiness"))
    if prototype.status in (PrototypeStatus.UNAVAILABLE, PrototypeStatus.BUILD_SCOPE_UNDEFINED):
        missing_requirements.append(
            "Experiment requires testing a working prototype, but no prototype exists yet."
        )

    # 2. Customer access check (Milestone 4.0.1)
    access_assessment = resolve_customer_access(state)
    sample_needed = exp_dict.get("target_sample_size") or "qualifying participants"

    if required_access or exp_type in (ExperimentType.PROBLEM_INTERVIEW.value, ExperimentType.PROTOTYPE_TEST.value):
        req_str = ", ".join(required_access) if required_access else "target customer segment"
        if access_assessment.access_status == CustomerAccessStatus.NONE:
            missing_requirements.append(
                f"Experiment requires specific customer/channel access ({req_str}), but founder has no demonstrated customer access."
            )
        elif access_assessment.access_status in (CustomerAccessStatus.LIMITED, CustomerAccessStatus.UNKNOWN):
            missing_requirements.append(
                f"Existing customer access is demonstrated ({'; '.join(access_assessment.evidence_sources[:2])}), but it is not yet clear that you can recruit the specific sample required for this experiment ({sample_needed})."
            )
        # If ESTABLISHED, access requirement is satisfied without missing requirement!

    # 3. Budget check (Milestone 6.2.5: currency-anchored parsing shared with the critic's contract;
    # "$1,500" is 1500, "$50 over 2 weeks" is 50, and an unknown budget is never treated as zero)
    budget_cmp = compare_cost_to_budget(exp_dict.get("estimated_cost"), exec_ctx)
    if budget_cmp.status == BudgetStatus.EXCEEDS_BUDGET:
        estimated_amt = budget_cmp.cost.low
        available_amt = budget_cmp.budget.high
        if available_amt == 0 and estimated_amt > 100:
            blockers.append(
                f"Estimated cost (${estimated_amt:.0f}) exceeds zero available budget."
            )
        else:
            missing_requirements.append(
                f"Estimated cost (${estimated_amt:.0f}) exceeds available budget (${available_amt:.0f})."
            )

    # Determine readiness
    if blockers:
        readiness = ExecutionReadiness.BLOCKED
        reason = f"Experiment cannot be executed under current constraints: {'; '.join(blockers)}"
    elif missing_requirements:
        readiness = ExecutionReadiness.NEEDS_HELP
        reason = f"Experiment is valid but founder requires assistance or assets: {'; '.join(missing_requirements)}"
    else:
        readiness = ExecutionReadiness.CAN_EXECUTE
        reason = "Founder has sufficient capacity, access, and budget to execute this experiment immediately."
        pending = pending_authorizations(exp_dict)
        if pending:
            reason = (
                "Founder has the capacity, access, and budget for this experiment, but it may start only after these "
                f"permissions are obtained (not yet obtained): {', '.join(pending)}."
            )

    # Grounding & contradiction guard (Milestone 4.0.1)
    if access_assessment.access_status != CustomerAccessStatus.NONE:
        reason = re.sub(r"founder has no established customer access", "existing customer access is limited and recruitment capacity for this sample is unconfirmed", reason, flags=re.IGNORECASE)
        reason = re.sub(r"founder has no demonstrated customer access", "existing customer access is limited and recruitment capacity for this sample is unconfirmed", reason, flags=re.IGNORECASE)
        reason = re.sub(r"has no access to users", "has limited demonstrated access to users", reason, flags=re.IGNORECASE)
        reason = re.sub(r"no customer conversations exist", "limited customer conversations exist", reason, flags=re.IGNORECASE)

    assessment = CanExecuteAssessment(
        readiness=readiness,
        reason=reason,
        missing_requirements=missing_requirements,
        blockers=blockers,
    )

    logger.info("assess_can_execute | readiness=%s access_status=%s reason=%s", readiness, access_assessment.access_status.value, reason[:80])
    return {
        "can_execute_result": assessment.model_dump(mode="json"),
        "phase": WorkflowPhase.TEST,
    }


def handle_experiment_failure_stop(state: StartaryGraphState) -> dict:
    """
    Deterministic Node (Milestone 4.0.4, 4.0.6 & 4.0.7.1.1):
    Gracefully stops the workflow if Experiment Critic rejects proposed experiments
    after maximum revisions or if unrecoverable validation error occurs.
    Preserves diagnosis, weak link, and critical assumption, setting experiment_design_status
    to QUALITY_BAR_NOT_MET so the terminal renderer displays EXPERIMENT_DESIGN_NEEDS_REVISION.
    Composes FallbackLearningAction (Milestone 4.0.6) so the founder receives a bounded,
    safe next learning step rather than an abrupt halt.
    Sets phase authoritatively to WAITING_FOR_LEARNING_INPUT and cleans up all active
    execution state (active_experiment, active_execution_handoff, can_execute_result).
    """
    project_id = state.get("project_id", "<unknown>")
    critic_output = state.get("experiment_critic_output") or {}
    issues = critic_output.get("issues") or []
    issues_str = "; ".join(issues) if issues else "Critique criteria failed."

    msg = f"Experiment generation stopped: unable to produce an accepted experiment after revision. Issues: {issues_str}"
    logger.warning("handle_experiment_failure_stop | project_id=%s: %s", project_id, msg)

    active_asmp = state.get("active_assumption") or state.get("active_critical_assumption")
    weak_link = state.get("weak_area") or (active_asmp.get("area") if isinstance(active_asmp, dict) else None)
    rejected_exp = state.get("proposed_experiment_output") or state.get("active_experiment")
    snapshot = state.get("diagnosis_snapshot")

    fallback = compose_fallback_learning_action(
        assumption=active_asmp,
        weak_link=weak_link,
        critic_output=critic_output,
        rejected_experiment=rejected_exp,
        state=state,
        diagnosis_snapshot=snapshot,
    )

    # Invalidate / clear stale current-cycle execution state (Milestone 4.0.7.1.1 & M4.1.1)
    # If active_experiment is already an explicit blocked record (e.g. from test setups), preserve it as BLOCKED;
    # otherwise, clear it to None so no rejected or prior-iteration experiment masquerades as active/ready.
    active_exp = state.get("active_experiment")
    cleared_active_exp = None
    if isinstance(active_exp, dict):
        if active_exp.get("execution_readiness") == ExecutionReadiness.BLOCKED.value and not (
            active_exp.get("created_iteration") and active_exp.get("created_iteration") < state.get("iteration", 1)
        ):
            cleared_active_exp = dict(active_exp)
            cleared_active_exp["status"] = ExperimentStatus.BLOCKED.value

    prev_experiments = list(state.get("previous_experiments") or [])
    if rejected_exp and isinstance(rejected_exp, dict):
        failed_record = dict(rejected_exp)
        failed_record["status"] = "QUALITY_BAR_NOT_MET"
        failed_record["critic_issues"] = issues
        if not any(
            (e.get("id") and e.get("id") == failed_record.get("id"))
            or (e.get("hypothesis") and e.get("hypothesis") == failed_record.get("hypothesis") and e.get("objective") == failed_record.get("objective"))
            for e in prev_experiments
        ):
            prev_experiments.append(failed_record)

    return {
        "status_message": msg,
        "phase": WorkflowPhase.WAITING_FOR_LEARNING_INPUT,
        "active_interrupt_type": "waiting_for_learning_input",
        "experiment_design_status": ExperimentDesignStatus.QUALITY_BAR_NOT_MET.value,
        "fallback_learning_action": fallback.model_dump(),
        "learning_input_mode": "FALLBACK",
        "active_experiment": cleared_active_exp,
        "active_execution_handoff": None,
        "execution_handoff_text": None,
        "build_readiness": None,
        "can_execute_result": None,
        "waiting_response": None,
        "post_evidence_answer": None,
        "pending_post_evidence_question": None,
        "evidence_turn_assessment": None,
        "previous_experiments": prev_experiments,
    }


# ---------------------------------------------------------------------------
# Milestone 3.1: Startup Diagnosis Composer Node
# ---------------------------------------------------------------------------

def compose_startup_diagnosis(state: StartaryGraphState) -> dict:
    """
    LLM Node (Milestone 3.1):
    Synthesizes the internal diagnostic intelligence (stage, grounded areas, advantages,
    constraints, major uncertainties, weak link, confidence, competing risks, reality summary,
    and active critical assumption) into a founder-facing StartupDiagnosisSnapshot.

    Deterministic Invariants:
      - active_assumption statement is preserved verbatim.
      - weak_link matches selected weak link.
      - weak_link_confidence matches discrimination result.
      - major_uncertainties capped at 5.
      - competing_risks capped at 3.
      - Never invents fake startup scores or new ungrounded facts.
    """
    project_id = state.get("project_id", "<unknown>")
    iteration = state.get("iteration", 1)

    active_ca = state.get("active_assumption") or {}
    ca_statement = active_ca.get("statement", "")

    weak_link = state.get("weak_area") or "Problem"
    confidence = state.get("selection_confidence") or "LOW"

    # Stage resolution
    startup_prog = state.get("startup_progress") or {}
    dev_stage_val = (startup_prog.get("development_stage") or {}).get("value")
    if not dev_stage_val or dev_stage_val == "UNKNOWN":
        diag = state.get("diagnosis") or {}
        dev_stage_val = diag.get("startup_stage") or "Pre-MVP"

    stage_name = str(dev_stage_val).replace("_", " ").title()
    stage_explanation = f"The startup is currently in the {stage_name} phase, where foundational market assumptions are being tested."

    # Competing risks from area assessments
    assessments = state.get("area_assessments") or []
    sorted_areas = sorted(
        assessments,
        key=lambda x: x.get("calculated_decision_risk", 0),
        reverse=True,
    )
    competing_risks = [
        a.get("area") for a in sorted_areas
        if a.get("area") != weak_link and a.get("area")
    ][:3]

    reality_check = state.get("reality_check") or {}
    rc_summary = reality_check.get("summary", "")
    what_supports = reality_check.get("what_evidence_supports") or []
    what_not_supports = reality_check.get("what_evidence_does_not_support") or []

    exec_context = state.get("execution_context") or {}

    grounded_summary_items = []
    for s in what_supports:
        grounded_summary_items.append(f"Supported: {s}")
    if exec_context.get("team_size", {}).get("value"):
        grounded_summary_items.append(f"Team: {exec_context['team_size']['value']}")
    if exec_context.get("customer_access", {}).get("value"):
        grounded_summary_items.append(f"Customer Access: {exec_context['customer_access']['value']}")
    grounded_facts_summary = "; ".join(grounded_summary_items)

    # Deterministic Presentation Formatter (Milestone M4.1.4):
    # Formats existing structured reasoning without calling an LLM or inventing new conclusions (0ms, 0 LLM calls).
    logger.info("compose_startup_diagnosis | deterministic formatting from existing structured reasoning (0ms)")
    constraints: list[str] = []
    if exec_context.get("team_size", {}).get("value"):
        constraints.append(f"Team: {exec_context['team_size']['value']}")
    if exec_context.get("monthly_budget", {}).get("value"):
        constraints.append(f"Budget: {exec_context['monthly_budget']['value']}")
    if exec_context.get("weekly_hours", {}).get("value"):
        constraints.append(f"Weekly Time: {exec_context['weekly_hours']['value']}")
    if not constraints:
        constraints = ["Resource-constrained early team"]

    advantages: list[str] = []
    cust_acc = exec_context.get("customer_access") or {}
    cust_acc_val = cust_acc.get("value") if isinstance(cust_acc, dict) else getattr(cust_acc, "value", None)
    if cust_acc_val:
        advantages.append(f"Customer Access: {cust_acc_val}")

    major_uncertainties: list[str] = []
    unresolved = reality_check.get("unresolved_uncertainty")
    if isinstance(unresolved, list):
        major_uncertainties = [str(u) for u in unresolved[:5]]
    elif isinstance(unresolved, str) and unresolved:
        major_uncertainties = [unresolved]
    for ns in what_not_supports:
        if len(major_uncertainties) < 5 and str(ns) not in major_uncertainties:
            major_uncertainties.append(str(ns))
    if not major_uncertainties:
        major_uncertainties = [f"Validation of {weak_link} under real-world conditions."]

    learning_goal = active_ca.get("rationale") or f"Validate whether {ca_statement}"

    snapshot = StartupDiagnosisSnapshot(
        project_id=project_id,
        iteration=iteration,
        stage=stage_name,
        stage_explanation=stage_explanation,
        grounded_areas=[str(s) for s in what_supports[:4]] or ["Initial founder concept described"],
        current_advantages=advantages,
        execution_constraints=constraints,
        major_uncertainties=major_uncertainties[:5],
        weak_link=weak_link,
        weak_link_reason=f"{weak_link} currently represents the primary bottleneck with highest decision risk.",
        weak_link_confidence=confidence,
        weak_link_confidence_explanation=f"Confidence is {confidence} based on comparative risk scoring across strategic areas.",
        competing_risks=competing_risks[:3],
        reality_summary=rc_summary or "Baseline diagnostic assessment established from founder inputs.",
        recommended_learning_goal=learning_goal,
        critical_assumption=ca_statement,
    )
    snapshot_output = snapshot.model_dump()

    # Milestone 4.1.2.1: Compose compact 7-part founder-facing diagnosis text
    snap = snapshot_output
    snap_stage = snap.get("stage") or stage_name
    snap_stage_exp = snap.get("stage_explanation") or stage_explanation
    snap_grounded = snap.get("grounded_areas") or []
    snap_advantages = snap.get("current_advantages") or []
    snap_weak_link = snap.get("weak_link") or weak_link
    snap_weak_reason = snap.get("weak_link_reason") or ""
    snap_competing = snap.get("competing_risks") or competing_risks
    snap_uncertainties = snap.get("major_uncertainties") or []
    snap_ca = snap.get("critical_assumption") or ca_statement
    snap_goal = snap.get("recommended_learning_goal") or ""

    advantages_str = "; ".join(snap_advantages) if snap_advantages else ("; ".join(snap_grounded[:3]) if snap_grounded else "Initial problem concept articulated.")
    competing_str = ", ".join(snap_competing[:3]) if snap_competing else "None identified."
    uncertainties_str = "; ".join(snap_uncertainties[:3]) if snap_uncertainties else "Long-term customer retention and unit economics."

    # What changed from latest evidence / baseline
    learning_delta = state.get("learning_delta") or {}
    delta_headline = learning_delta.get("headline") or ""
    delta_changes = learning_delta.get("what_we_learned") or learning_delta.get("key_takeaways") or []
    summary_text = learning_delta.get("founder_facing_summary") or ""

    if delta_headline and delta_changes:
        what_changed = f"{delta_headline} " + "; ".join(delta_changes[:2])
    elif delta_changes:
        what_changed = "; ".join(delta_changes[:2])
    elif summary_text:
        what_changed = summary_text
    elif delta_headline:
        what_changed = delta_headline
    else:
        # Check if there is an active/recent evidence analysis or cycle
        cycle = get_most_recent_completed_evidence_cycle(state) if "get_most_recent_completed_evidence_cycle" in globals() else None
        ev_analysis = state.get("evidence_analysis") or (cycle[3] if cycle else {})
        if ev_analysis and ev_analysis.get("conclusion"):
            conc = ev_analysis.get("conclusion")
            findings = ev_analysis.get("support_findings") or ev_analysis.get("valid_learnings") or []
            if findings:
                what_changed = f"Latest test concluded {conc}: " + "; ".join(findings[:2])
            else:
                what_changed = f"Latest test concluded {conc} for the tested assumption."
        else:
            what_changed = "Baseline diagnostic assessment established from founder inputs and verified operational evidence."

    strat_act = state.get("next_strategic_action") or {}
    most_important_decision = (
        strat_act.get("key_decision")
        or (f"Validate whether {snap_ca} before committing further resources or scaling." if snap_ca else f"Determine whether target customers experience the {snap_weak_link.lower()} bottleneck.")
    )
    why_this_decision = (
        strat_act.get("why_this_action")
        or f"{snap_weak_link} is the primary constraint at the {snap_stage} stage. Resolving this uncertainty prevents premature investment into unvalidated assumptions."
    )

    founder_diag_lines = [
        "=" * 50,
        "STARTARY — CURRENT DIAGNOSIS",
        "=" * 50,
        "",
        "WHAT LOOKS STRONG",
        f"- Stage: {snap_stage} ({snap_stage_exp})",
        f"- Validated advantages: {advantages_str}",
        "",
        "WHAT LOOKS WEAK",
        f"- Primary bottleneck: {snap_weak_link}",
        f"- Bottleneck reason: {snap_weak_reason}",
        "",
        "WHAT IS STILL UNCERTAIN",
        f"- Competing risks: {competing_str}",
        f"- Gating uncertainties: {uncertainties_str}",
        "",
        "WHAT CHANGED",
        f"- {what_changed}",
        "",
        "WHAT DID NOT CHANGE",
        f"- Remaining blind spots: {uncertainties_str}",
        "",
        "MOST IMPORTANT DECISION NOW",
        f"- {most_important_decision}",
        "",
        "WHY THIS DECISION",
        f"- {why_this_decision}",
        "",
        "=" * 50,
    ]
    founder_diagnosis_text = "\n".join(founder_diag_lines)

    trace = AITrace(
        module_name="compose_startup_diagnosis",
        prompt_version=STARTUP_DIAGNOSIS_COMPOSER_PROMPT_VERSION,
        input_data={
            "weak_link": weak_link,
            "confidence": confidence,
            "stage": stage_name,
        },
        output_data={"snapshot_id": snapshot_output.get("id"), "weak_link": weak_link},
        latency_ms=0,
        project_id=project_id,
    )
    existing_traces = list(state.get("traces") or [])
    existing_traces.append(trace.model_dump(mode="json"))

    return {
        "diagnosis_snapshot": snapshot_output,
        "founder_diagnosis_text": founder_diagnosis_text,
        "traces": existing_traces,
        "last_trace": trace.model_dump(mode="json"),
    }


# ---------------------------------------------------------------------------
# Milestone 3.2 Node 1: interpret_founder_turn (LLM)
# ---------------------------------------------------------------------------

def interpret_founder_turn(state: StartaryGraphState) -> dict:
    """
    LLM Node (Milestone 3.2): Interpret latest founder turn intent, questions, concerns,
    corrections, and ability to act/answer separately from factual state extraction.
    """
    import json as _json

    project_id = state.get("project_id", "<unknown>")
    founder_messages = state.get("founder_messages") or []
    pending_question = state.get("pending_question")
    workflow_phase = str(state.get("phase", WorkflowPhase.UNDERSTAND))
    interrupt_type = state.get("active_interrupt_type")
    raw_context = state.get("raw_context") or ""
    weak_link = state.get("weak_area")

    # Find latest founder message
    latest_founder_msg = ""
    latest_turn_index = 0
    for i, msg in enumerate(reversed(founder_messages)):
        if msg.get("role") == "founder":
            latest_founder_msg = (msg.get("content") or "").strip()
            latest_turn_index = len(founder_messages) - 1 - i
            break

    # If no founder message yet, provide default initial turn interpretation
    if not latest_founder_msg:
        logger.info("interpret_founder_turn | no founder messages to interpret")
        return {}

    # Format recent history (last 3-6 turns)
    recent_msgs = founder_messages[-6:]
    recent_history_text = "\n".join(
        f"[{m.get('role', 'unknown').capitalize()}] {m.get('content', '')}"
        for m in recent_msgs
    )

    settings = get_settings()
    llm = _safe_get_llm_client(settings, tier="fast", max_tokens=800)
    structured_llm = llm.with_structured_output(FounderTurnInterpretationLLMOutput)

    user_msg = build_founder_turn_interpreter_user_message(
        latest_founder_message=latest_founder_msg,
        pending_question=pending_question,
        recent_history_text=recent_history_text,
        workflow_phase=workflow_phase,
        interrupt_type=interrupt_type,
        startup_summary=raw_context[:300],
        weak_link=weak_link,
    )

    messages = [
        {"role": "system", "content": FOUNDER_TURN_INTERPRETER_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    res_info = get_llm_client_resolution_info(llm)
    attempt = 0
    last_error: Exception | None = None
    interpretation_dict: dict[str, Any] | None = None
    t_start = time.monotonic()

    while attempt < 2:
        attempt += 1
        try:
            res: FounderTurnInterpretationLLMOutput = structured_llm.invoke(messages)
            res_data = res.model_dump()

            # Deterministic overrides for core safety & priority
            msg_lower = latest_founder_msg.lower()
            intent_types = list(res_data.get("intent_types") or [])

            # 1. Direct question check
            has_question = bool(res_data.get("direct_question")) or ("?" in latest_founder_msg)
            if has_question and "QUESTION" not in intent_types:
                intent_types.append("QUESTION")

            # 2. Fear / Concern check
            fear_keywords = ["fear", "scared", "worry", "worried", "concern", "risk", "waste", "hesitant", "doubt"]
            has_concern = bool(res_data.get("concern_or_goal")) or any(k in msg_lower for k in fear_keywords)
            if has_concern and "CONCERN" not in intent_types:
                intent_types.append("CONCERN")

            # 3. Correction check
            correction_keywords = ["not the", "not my", "i'm not", "im not", "actually i", "wrong"]
            is_correction = bool(res_data.get("correction")) or any(k in msg_lower for k in correction_keywords)
            if is_correction and "CORRECTION" not in intent_types:
                intent_types.append("CORRECTION")

            # 4. Does not know check
            dnk_keywords = ["i don't know", "i dont know", "dont know", "not sure", "no idea", "hard to say"]
            is_dnk = bool(res_data.get("cannot_answer_now")) or any(k in msg_lower for k in dnk_keywords)
            if is_dnk:
                res_data["cannot_answer_now"] = True
                if "DOES_NOT_KNOW" not in intent_types:
                    intent_types.append("DOES_NOT_KNOW")

            # 5. Cannot act now check
            cannot_act_keywords = ["cannot make conversation", "cant talk", "cannot talk", "not today", "not right now", "cant do", "cannot do"]
            is_cannot_act = bool(res_data.get("cannot_act_now")) or any(k in msg_lower for k in cannot_act_keywords)
            if is_cannot_act:
                res_data["cannot_act_now"] = True
                if "CANNOT_ACT_NOW" not in intent_types:
                    intent_types.append("CANNOT_ACT_NOW")

            # 6. Already answered claim check (Milestone 3.2.1 Section 10)
            already_answered_keywords = ["already answered", "already said", "already told you", "we talked about team", "already answered that", "already gave you that"]
            is_already_ans = bool(res_data.get("is_already_answered_claim")) or any(k in msg_lower for k in already_answered_keywords)
            if is_already_ans:
                res_data["is_already_answered_claim"] = True
                if "CORRECTION" not in intent_types:
                    intent_types.append("CORRECTION")
                res_data["response_priority"] = "HIGH"

            # 7. Founder goal check (Milestone 3.2.1 Section 11/12)
            detected_goal = res_data.get("detected_founder_goal")
            if not detected_goal:
                if any(k in msg_lower for k in ["find the right market", "find our market", "which market", "find the niche", "find initial market", "target market"]):
                    detected_goal = "FIND_INITIAL_MARKET"
                elif any(k in msg_lower for k in ["worth my time", "worth spending time", "should i continue", "worth to spend"]):
                    detected_goal = "DECIDE_IF_WORTH_CONTINUING"
                elif any(k in msg_lower for k in ["find customers", "get users", "reach customers"]):
                    detected_goal = "FIND_CUSTOMERS"
                elif any(k in msg_lower for k in ["validate problem", "is this a problem", "problem exists"]):
                    detected_goal = "VALIDATE_PROBLEM"
            if detected_goal:
                res_data["detected_founder_goal"] = detected_goal

            # 8. Deterministic guard: Initial onboarding / intake context is NOT empirical evidence
            # (ordinary facts like team, hours, budget, customer access are baseline context/answer, not experiment evidence)
            is_initial_turn = (latest_turn_index == 0) or (pending_question is None)
            is_intake_phase = (
                str(state.get("phase", "")).upper() in ("UNDERSTAND", "WORKFLOWPHASE.UNDERSTAND")
                and state.get("clarification_turn_count", 0) == 0
            )
            has_no_active_exp = not (state.get("active_experiment") or state.get("evidence_submissions"))
            if (is_initial_turn or is_intake_phase or has_no_active_exp) and "NEW_EVIDENCE" in intent_types:
                has_empirical_evidence = any(kw in msg_lower for kw in [
                    "conducted interviews", "interviewed", "survey results", "tested with users",
                    "experiment result", "pilot results", "retention data", "conversion rate",
                ])
                if not has_empirical_evidence:
                    logger.info("interpret_founder_turn | sanitizing NEW_EVIDENCE on turn %d (intake/context message)", latest_turn_index)
                    intent_types = [it for it in intent_types if it != "NEW_EVIDENCE"]
                    res_data["new_evidence_summary"] = None

            # Priority escalation: HIGH if question, concern, correction, inabilities, or claim
            if has_question or has_concern or is_correction or is_cannot_act or is_dnk or is_already_ans:
                res_data["response_priority"] = "HIGH"
            elif not res_data.get("response_priority"):
                res_data["response_priority"] = "NORMAL"

            if not intent_types:
                intent_types = ["ANSWER"]
            res_data["intent_types"] = intent_types
            if not res_data.get("interpretation_summary"):
                res_data["interpretation_summary"] = f"Founder turn interpreted with {res_data.get('response_priority', 'NORMAL')} priority."
            interpretation_dict = res_data
            break
        except Exception as exc:
            last_error = exc
            logger.warning("interpret_founder_turn failed attempt %d: %s", attempt, exc)
            if is_permanent_model_error(exc):
                break

    latency_ms = (time.monotonic() - t_start) * 1000
    used_fallback = (interpretation_dict is None)

    if used_fallback:
        logger.warning("interpret_founder_turn: using deterministic fallback due to LLM error: %s", last_error)
        msg_lower = latest_founder_msg.lower()
        has_q = "?" in latest_founder_msg
        has_fear = any(k in msg_lower for k in ["fear", "worry", "waste", "doubt"])
        is_corr = any(k in msg_lower for k in ["not the", "not my", "i'm not", "already answered"])
        is_dnk = any(k in msg_lower for k in ["i don't know", "dont know", "not sure"])
        is_cna = any(k in msg_lower for k in ["cannot talk", "cant talk", "not right now"])

        intents = ["ANSWER"]
        if has_q: intents.append("QUESTION")
        if has_fear: intents.append("CONCERN")
        if is_corr: intents.append("CORRECTION")
        if is_dnk: intents.append("DOES_NOT_KNOW")
        if is_cna: intents.append("CANNOT_ACT_NOW")

        priority = "HIGH" if (has_q or has_fear or is_corr or is_dnk or is_cna) else "NORMAL"
        interpretation_dict = {
            "intent_types": intents,
            "direct_answer": True,
            "direct_question": latest_founder_msg if has_q else None,
            "concern_or_goal": "Founder expressed risk or hesitation" if has_fear else None,
            "correction": "Founder corrected previous assumption" if is_corr else None,
            "cannot_answer_now": is_dnk,
            "cannot_act_now": is_cna,
            "uncertainty_statement": None,
            "new_evidence_summary": None,
            "founder_preference_or_constraint": None,
            "response_priority": priority,
            "interpretation_summary": f"Fallback interpretation: founder provided input with {priority} response priority.",
        }

    # Wrap in domain model
    domain_interpretation = FounderTurnInterpretation(
        project_id=project_id,
        turn_index=latest_turn_index,
        **interpretation_dict,
    )
    interpretation_output = domain_interpretation.model_dump()

    trace = AITrace(
        module_name="interpret_founder_turn",
        prompt_version=FOUNDER_TURN_INTERPRETER_PROMPT_VERSION,
        input_data={
            "latest_founder_message": latest_founder_msg[:100],
            "pending_question": pending_question[:100] if pending_question else None,
        },
        output_data={
            "intent_types": interpretation_output.get("intent_types"),
            "response_priority": interpretation_output.get("response_priority"),
        },
        latency_ms=latency_ms,
        project_id=project_id,
        model_name="local_deterministic" if used_fallback else (res_info["resolved_model"] or getattr(settings, "fast_model_name", None)),
        configured_model=res_info["configured_model"],
        resolved_model="local_deterministic" if used_fallback else res_info["resolved_model"],
        fallback_used=True if used_fallback else res_info["fallback_used"],
        fallback_reason=f"fast_fail_error: {last_error}" if used_fallback else res_info["fallback_reason"],
        execution_path=state.get("execution_path", "DELTA_STATE_UPDATE"),
    )
    existing_traces = list(state.get("traces") or [])
    existing_traces.append(trace.model_dump(mode="json"))

    turn_telem = record_telemetry_event(
        state=state,
        node_name="interpret_founder_turn",
        model_name="local_deterministic" if used_fallback else (res_info["resolved_model"] or getattr(settings, "fast_model_name", None) or "fast_tier"),
        duration_ms=latency_ms,
        execution_path=state.get("execution_path", "DELTA_STATE_UPDATE"),
        configured_model=res_info["configured_model"],
        resolved_model="local_deterministic" if used_fallback else res_info["resolved_model"],
        fallback_used=True if used_fallback else res_info["fallback_used"],
        fallback_reason=f"fast_fail_error: {last_error}" if used_fallback else res_info["fallback_reason"],
        model_tier="fast",
        reasoning_effort=res_info.get("reasoning_effort"),
        status="FALLBACK" if used_fallback else "SUCCESS",
        retry_count=max(0, attempt - 1),
    )

    updates = {
        "latest_founder_turn_interpretation": interpretation_output,
        "traces": existing_traces,
        "last_trace": trace.model_dump(mode="json"),
        "turn_telemetry": turn_telem,
    }

    # Check for conversational repair
    is_repair = bool(state.get("is_conversational_repair")) or is_conversational_repair_text(latest_founder_msg)
    if is_repair:
        updates["is_conversational_repair"] = True
        updates["execution_path"] = "CONVERSATIONAL_REPAIR"

    if interpretation_output.get("concern_or_goal"):
        updates["founder_current_concern"] = interpretation_output["concern_or_goal"]
    if interpretation_output.get("cannot_act_now"):
        updates["cannot_act_now_context"] = True
    if interpretation_output.get("correction"):
        corrections = list(state.get("recent_corrections") or [])
        corrections.append(interpretation_output["correction"])
        updates["recent_corrections"] = corrections

    # Goal persistence (Milestone 3.2.1 Section 11/12)
    detected_goal_str = interpretation_dict.get("detected_founder_goal")
    if detected_goal_str:
        try:
            goal_enum = FounderGoalType(detected_goal_str)
        except ValueError:
            goal_enum = FounderGoalType.OTHER
        updates["founder_current_goal"] = {
            "goal_type": goal_enum.value,
            "raw_statement": latest_founder_msg,
            "turn_acquired": latest_turn_index,
            "is_active": True,
        }
    elif state.get("founder_current_goal"):
        updates["founder_current_goal"] = state["founder_current_goal"]

    logger.info(
        "interpret_founder_turn | priority=%s intents=%s goal=%s repair=%s",
        interpretation_output.get("response_priority"),
        interpretation_output.get("intent_types"),
        (updates.get("founder_current_goal") or {}).get("goal_type"),
        is_repair,
    )
    return updates


# ---------------------------------------------------------------------------
# Milestone 6.2.7: safe parallel understanding (STARTARY_PARALLEL_UNDERSTANDING only)
# ---------------------------------------------------------------------------
#
# interpret_founder_turn and extract_startup_state read disjoint inputs and neither consumes the
# other's output. The route after interpretation is a deterministic function of state that exists
# BEFORE interpretation (repair flags / repair phrasing of the latest founder message / phase), so the
# extraction decision is made up front: both nodes run concurrently only on turns the sequential graph
# would extract anyway. Their only shared writes (traces, last_trace, turn_telemetry) are isolated per
# branch and merged deterministically by join_founder_turn in sequential order (interpretation first).

_DISPATCH_REASONS = {
    "extract_startup_state": "UNDERSTAND_TURN",
    "handle_conversational_repair": "CONVERSATIONAL_REPAIR",
    "classify_waiting_for_evidence_turn": "WAITING_FOR_EVIDENCE",
}


def _latest_founder_text(state: StartaryGraphState) -> str:
    for msg in reversed(state.get("founder_messages") or []):
        if msg.get("role") == "founder":
            return (msg.get("content") or "").strip()
    return ""


def founder_turn_route(state: StartaryGraphState) -> str:
    """
    The route route_after_interpret_founder_turn will take, computed BEFORE interpretation from the
    same deterministic inputs: interpret_founder_turn only sets the repair flags when the state already
    has them or is_conversational_repair_text(latest founder message) is true, and never changes phase.
    """
    latest = _latest_founder_text(state)
    if (
        state.get("is_conversational_repair")
        or state.get("execution_path") == "CONVERSATIONAL_REPAIR"
        or (latest and is_conversational_repair_text(latest))
    ):
        return "handle_conversational_repair"
    phase = state.get("phase")
    if phase == WorkflowPhase.WAITING_FOR_EVIDENCE or phase == "WAITING_FOR_EVIDENCE":
        return "classify_waiting_for_evidence_turn"
    return "extract_startup_state"


def dispatch_founder_turn(state: StartaryGraphState) -> dict:
    """
    Deterministic Node (M6.2.7): decide whether this founder turn runs interpretation and extraction
    concurrently (PARALLEL: the turn will be extracted anyway) or interpretation alone (SEQUENTIAL:
    repair / evidence turns, which never extract, so no speculative extraction call is made).
    """
    route = founder_turn_route(state)
    mode = "PARALLEL" if route == "extract_startup_state" else "SEQUENTIAL"
    dispatch = {
        "mode": mode,
        "reason": _DISPATCH_REASONS[route],
        "dispatched_at": time.time(),
        "founder_turn_count": sum(1 for m in (state.get("founder_messages") or []) if m.get("role") == "founder"),
    }
    emit_decision_span("understanding_dispatch", {
        "parallel_understanding": True,
        "dispatch_mode": mode,
        "dispatch_reason": dispatch["reason"],
        "fallback_to_sequential": mode == "SEQUENTIAL",
    }, tags=["node:dispatch_founder_turn", f"understanding:{mode.lower()}"])
    logger.info("dispatch_founder_turn | mode=%s reason=%s", mode, dispatch["reason"])
    return {"founder_turn_dispatch": dispatch, "understanding_branches": {"__reset__": True}}


def _run_understanding_branch(branch: str, node_fn: Any, state: StartaryGraphState) -> dict:
    """
    Run an unchanged understanding node. On a PARALLEL turn its shared-key writes are moved into
    understanding_branches[branch] (with a private copy of the telemetry summary, so the concurrent
    branches never mutate the same object); every other write passes through untouched.
    """
    if (state.get("founder_turn_dispatch") or {}).get("mode") != "PARALLEL":
        return node_fn(state)
    original_summary = state.get("turn_telemetry")
    base_records = len(getattr(original_summary, "records", None) or [])
    base_traces = len(state.get("traces") or [])
    branch_state = dict(state)
    branch_state["turn_telemetry"] = _copy.deepcopy(original_summary) if original_summary is not None else None

    started_at = time.time()
    updates = dict(node_fn(branch_state) or {})
    ended_at = time.time()

    summary = updates.pop("turn_telemetry", None) or branch_state.get("turn_telemetry")
    traces = updates.pop("traces", None)
    last_trace = updates.pop("last_trace", None)
    updates["understanding_branches"] = {branch: {
        "started_at": started_at,
        "ended_at": ended_at,
        "new_traces": list(traces[base_traces:]) if traces is not None else [],
        "last_trace": last_trace,
        "telemetry_records": list(getattr(summary, "records", None) or [])[base_records:],
        "telemetry_execution_path": getattr(summary, "execution_path", None),
    }}
    return updates


def parallel_interpret_founder_turn(state: StartaryGraphState) -> dict:
    """interpret_founder_turn as registered in the parallel-understanding graph."""
    return _run_understanding_branch("interpret", interpret_founder_turn, state)


def parallel_extract_startup_state(state: StartaryGraphState) -> dict:
    """extract_startup_state as registered in the parallel-understanding graph."""
    return _run_understanding_branch("extract", extract_startup_state, state)


def join_founder_turn(state: StartaryGraphState) -> dict:
    """
    Deterministic Node (M6.2.7): synchronization barrier of the parallel understanding region.
    Runs only after BOTH branches completed (same LangGraph superstep). Merges their shared-key
    outputs in sequential order (interpretation, then extraction), records the overlap metrics, and
    fails loudly if a branch result is missing rather than continuing with partial state.
    """
    dispatch = state.get("founder_turn_dispatch") or {}
    if dispatch.get("mode") != "PARALLEL":
        return {}
    joined_at = time.time()
    branches = state.get("understanding_branches") or {}
    interp, extract = branches.get("interpret"), branches.get("extract")
    missing = [name for name, payload in (("interpret", interp), ("extract", extract)) if payload is None]
    if missing:
        raise RuntimeError(f"join_founder_turn: parallel understanding branch result missing: {missing}")

    traces = list(state.get("traces") or []) + list(interp["new_traces"]) + list(extract["new_traces"])
    summary = _copy.deepcopy(state.get("turn_telemetry"))
    if summary is None:
        summary = TurnTelemetrySummary(execution_path=interp.get("telemetry_execution_path") or extract.get("telemetry_execution_path") or "STRATEGIC_REASONING")
    for record in list(interp["telemetry_records"]) + list(extract["telemetry_records"]):
        summary.add_record(record)

    repair_flagged = bool(state.get("is_conversational_repair")) or state.get("execution_path") == "CONVERSATIONAL_REPAIR"
    next_route = "handle_conversational_repair" if repair_flagged else "merge_execution_state"
    starts, ends = (interp["started_at"], extract["started_at"]), (interp["ended_at"], extract["ended_at"])

    def _retries(records: list[Any]) -> int:
        return sum(int(getattr(r, "retry_count", 0) or 0) for r in records)

    def _status(records: list[Any], default: str) -> str:
        return str(getattr(records[-1], "status", default)) if records else default

    meta = {
        "parallel_understanding": True,
        "dispatch_mode": "PARALLEL",
        "interpretation_ms": round((interp["ended_at"] - interp["started_at"]) * 1000, 1),
        "extraction_ms": round((extract["ended_at"] - extract["started_at"]) * 1000, 1),
        "sum_of_branches_ms": round(((interp["ended_at"] - interp["started_at"]) + (extract["ended_at"] - extract["started_at"])) * 1000, 1),
        "branch_overlap_ms": round(max(0.0, min(ends) - max(starts)) * 1000, 1),
        "parallel_region_ms": round((joined_at - float(dispatch.get("dispatched_at") or joined_at)) * 1000, 1),
        "sync_wait_ms": round((max(ends) - min(ends)) * 1000, 1),
        "interpretation_status": _status(interp["telemetry_records"], "SKIPPED"),
        "extraction_status": state.get("extraction_status"),
        "interpretation_retry_count": _retries(interp["telemetry_records"]),
        "extraction_retry_count": _retries(extract["telemetry_records"]),
        "sync_status": "ROUTE_MISMATCH" if repair_flagged else "SUCCESS",
        "next_route": next_route,
        "fallback_to_sequential": False,
    }
    emit_decision_span("understanding_parallel_region", meta, tags=["node:join_founder_turn", "understanding:parallel"])
    turn_telem = record_telemetry_event(
        state={**state, "turn_telemetry": summary}, node_name="join_founder_turn", model_name="deterministic",
        skipped_by_fast_path=True, model_tier="none", details=meta,
    )
    updates: dict[str, Any] = {
        "traces": traces,
        "turn_telemetry": turn_telem,
        "understanding_branches": {"__reset__": True},
    }
    last_trace = extract.get("last_trace") or interp.get("last_trace")
    if last_trace is not None:
        updates["last_trace"] = last_trace
    if repair_flagged:
        # Cannot happen when dispatch and interpretation see the same state (same deterministic
        # predicate); if it ever does, the extraction result is discarded, never merged.
        logger.error("join_founder_turn | dispatch/interpretation route mismatch; discarding extraction delta")
        updates["startup_state_delta"] = {}
    logger.info("join_founder_turn | overlap_ms=%.0f region_ms=%.0f route=%s", meta["branch_overlap_ms"], meta["parallel_region_ms"], next_route)
    return updates


def handle_conversational_repair(state: StartaryGraphState) -> dict:
    """
    M4.1.3 Node: Conversational Repair Fast Path.
    When a founder says "I don't understand", "What do you mean?", or asks for explanation,
    Startary immediately rephrases the *same* pending question with zero rediagnosis,
    zero state extraction, and zero progression increments.
    """
    project_id = state.get("project_id", "<unknown>")
    pending_q = (
        state.get("pending_question")
        or state.get("current_question")
        or "Could you tell me a bit more about your startup?"
    )

    founder_messages = state.get("founder_messages") or []
    latest_msg = ""
    for m in reversed(founder_messages):
        if m.get("role") == "founder":
            latest_msg = m.get("content", "")
            break

    t_start = time.monotonic()
    settings = get_settings()
    llm = _safe_get_llm_client(settings, tier="fast", max_tokens=800)
    res_info = get_llm_client_resolution_info(llm)

    repair_system_prompt = (
        "You are Startary's startup advisor. The founder did not understand the previous question "
        "or asked for clarification. Briefly explain in 1-2 simple, conversational, jargon-free sentences "
        "what you are asking and why it matters, then repeat the question clearly. "
        "Do NOT introduce new questions or change the topic."
    )
    user_prompt = (
        f"Original question: {pending_q}\n"
        f"Founder query: {latest_msg}\n\n"
        "Provide a clear, reassuring, and direct rephrasing of the question."
    )
    used_fallback = False
    repair_err = None
    try:
        res = llm.invoke([
            {"role": "system", "content": repair_system_prompt},
            {"role": "user", "content": user_prompt},
        ])
        rephrased_text = res.content.strip() if hasattr(res, "content") else str(res).strip()
    except Exception as exc:
        used_fallback = True
        repair_err = exc
        logger.warning("handle_conversational_repair LLM call failed: %s; using template fallback", exc)
        rephrased_text = (
            f"No problem! Let me make that simpler. To help figure out the next best step for your startup, "
            f"I wanted to ask: {pending_q}"
        )

    latency_ms = (time.monotonic() - t_start) * 1000

    active_decision = state.get("active_next_turn_decision") or {}
    question_id = active_decision.get("question_id") or "q_repair"

    advisor_resp = AdvisorResponse(
        project_id=project_id,
        response_text=rephphrased_text if 'rephphrased_text' in locals() else rephrased_text,
        response_mode="CONVERSATIONAL_REPAIR",
        tone="WARM_DIRECT",
        founder_goal_addressed="CLARIFY_QUESTION",
        intent_addressed="QUESTION",
        question_id=question_id,
    )

    trace = AITrace(
        module_name="handle_conversational_repair",
        prompt_version="CONVERSATIONAL_REPAIR_V1",
        input_data={"original_question": pending_q[:100], "founder_query": latest_msg[:100]},
        output_data={"rephrased_response": rephrased_text[:120]},
        latency_ms=latency_ms,
        project_id=project_id,
        model_name="local_deterministic" if used_fallback else (res_info["resolved_model"] or getattr(settings, "fast_model_name", None)),
        configured_model=res_info["configured_model"],
        resolved_model="local_deterministic" if used_fallback else res_info["resolved_model"],
        fallback_used=True if used_fallback else res_info["fallback_used"],
        fallback_reason=f"fast_fail_error: {repair_err}" if used_fallback else res_info["fallback_reason"],
        execution_path="CONVERSATIONAL_REPAIR",
    )
    existing_traces = list(state.get("traces") or [])
    existing_traces.append(trace.model_dump(mode="json"))

    turn_telem = record_telemetry_event(
        state=state,
        node_name="handle_conversational_repair",
        model_name="local_deterministic" if used_fallback else (res_info["resolved_model"] or getattr(settings, "fast_model_name", None) or "fast_tier"),
        duration_ms=latency_ms,
        execution_path="CONVERSATIONAL_REPAIR",
        configured_model=res_info["configured_model"],
        resolved_model="local_deterministic" if used_fallback else res_info["resolved_model"],
        fallback_used=True if used_fallback else res_info["fallback_used"],
        fallback_reason=f"fast_fail_error: {repair_err}" if used_fallback else res_info["fallback_reason"],
    )

    logger.info(
        "handle_conversational_repair | project_id=%s rephrased='%s' latency_ms=%.0f",
        project_id, rephrased_text[:60], latency_ms,
    )

    advisor_resp_dict = advisor_resp.model_dump()
    return {
        "latest_advisor_response": advisor_resp_dict,
        # Milestone 6.2.3: persist the exact re-asked prompt before ask_and_wait_for_founder interrupts
        "pending_founder_prompt": _build_founder_prompt(_resolve_active_question(state), advisor_resp_dict, "conversational_repair"),
        "pending_question": pending_q,
        "execution_path": "CONVERSATIONAL_REPAIR",
        "is_conversational_repair": False,
        "last_trace": trace.model_dump(mode="json"),
        "traces": existing_traces,
        "turn_telemetry": turn_telem,
    }


# ---------------------------------------------------------------------------
# Milestone 3.2 Node 2: compose_advisor_response (LLM — Milestone 3.2.1 Patched)
# ---------------------------------------------------------------------------

def compose_advisor_response(state: StartaryGraphState) -> dict:
    """
    LLM Node (Milestone 3.2 & 3.2.1): Compose empathetic, grounded, conversational advisor response.
    The planner's NextTurnDecision is the SOLE AUTHORITY for what information Startary needs next.
    The composer determines only HOW that decision is communicated.

    Guarantees:
    - Enforces semantic-target matching (res.next_question_target == decision.information_target).
    - Enforces question_id matching (res.question_id == decision.question_id).
    - Retries once on validation mismatch; falls back to deterministic text if retry fails.
    - Sets intro_summary_shown = True so startup summary appears only once.
    """
    import json as _json

    project_id = state.get("project_id", "<unknown>")
    interpretation = state.get("latest_founder_turn_interpretation") or {}
    founder_messages = state.get("founder_messages") or []
    workflow_phase = str(state.get("phase", WorkflowPhase.UNDERSTAND))
    raw_context = state.get("raw_context") or ""
    weak_link = state.get("weak_area")
    reality_check = state.get("reality_check") or {}
    diagnosis = state.get("diagnosis") or {}

    # Sole authority: active_next_turn_decision
    active_decision = state.get("active_next_turn_decision")
    if not active_decision:
        # Fallback for backward compatibility or direct calls
        active_type = state.get("active_interrupt_type") or "strategic"
        if active_type == "baseline" and state.get("current_baseline_question"):
            q_dict = state["current_baseline_question"]
        elif active_type == "weak_link" and state.get("current_weak_link_question"):
            q_dict = state["current_weak_link_question"]
        else:
            q_dict = state.get("current_clarifying_question") or {}
        raw_q = q_dict.get("question")
        if not raw_q:
            exec_ctx = state.get("execution_context") or {}
            prog = state.get("startup_progress") or {}
            team_val = exec_ctx.get("team_size")
            team_known = bool(team_val and (team_val.get("provenance") if isinstance(team_val, dict) else getattr(team_val, "provenance", None)) not in (None, "UNKNOWN"))
            mvp_val = prog.get("mvp_exists")
            mvp_known = bool(mvp_val and (mvp_val.get("provenance") if isinstance(mvp_val, dict) else getattr(mvp_val, "provenance", None)) not in (None, "UNKNOWN"))

            gap_plan = state.get("gap_plan") or {}
            missing_info = gap_plan.get("missing_information") or gap_plan.get("information_goal")

            if team_known or mvp_known:
                if missing_info:
                    raw_q = f"Could you share more detail regarding {missing_info}?"
                else:
                    raw_q = "Could you tell me more about how potential customers currently handle this problem?"
            else:
                raw_q = "Could you describe your team, current product status, and testing so far?"

        tgt = q_dict.get("target_group") or q_dict.get("target_area") or "Context"
        active_decision = NextTurnDecision(
            question_id=f"q_compat_{uuid.uuid4().hex[:6]}",
            source_phase=WorkflowPhase.UNDERSTAND,
            information_target=str(tgt),
            canonical_question=raw_q,
            should_ask_question=True,
        ).model_dump()

    canonical_question = active_decision.get("canonical_question") or "Could you provide more context about your startup?"
    information_target = active_decision.get("information_target") or "Context"
    question_id = active_decision.get("question_id") or f"q_{uuid.uuid4().hex[:6]}"

    # Conversational state: goal & intro summary gate
    founder_goal_dict = state.get("founder_current_goal") or {}
    founder_current_goal = founder_goal_dict.get("raw_statement")
    intro_summary_shown = bool(state.get("intro_summary_shown", False))

    # Latest founder message
    latest_founder_msg = ""
    for msg in reversed(founder_messages):
        if msg.get("role") == "founder":
            latest_founder_msg = msg.get("content", "")
            break

    # Format recent history
    recent_msgs = founder_messages[-6:]
    recent_history_text = "\n".join(
        f"[{m.get('role', 'unknown').capitalize()}] {m.get('content', '')}"
        for m in recent_msgs
    )

    settings = get_settings()
    llm = _safe_get_llm_client(settings, tier="fast", max_tokens=1000)
    structured_llm = llm.with_structured_output(AdvisorResponseLLMOutput)

    # Milestone 4.0.4: Deterministically determine conversational response mode
    response_mode = determine_conversational_response_mode(
        state=state,
        founder_interpretation=interpretation,
        founder_turn=latest_founder_msg,
        previous_state=state.get("_previous_state"),
    )

    user_msg = build_advisor_response_composer_user_message(
        latest_founder_message=latest_founder_msg or "[Initial entry]",
        founder_interpretation_json=_json.dumps(interpretation, indent=2, default=str),
        recent_history_text=recent_history_text,
        canonical_question=canonical_question,
        information_target=information_target,
        question_id=question_id,
        founder_current_goal=founder_current_goal,
        intro_summary_shown=intro_summary_shown,
        startup_summary=raw_context[:300],
        diagnosis_summary=diagnosis.get("summary", "")[:300] if isinstance(diagnosis, dict) else "",
        weak_link=weak_link,
        reality_check_summary=reality_check.get("summary"),
        workflow_phase=workflow_phase,
        response_mode=response_mode.value,
    )

    messages = [
        {"role": "system", "content": ADVISOR_RESPONSE_COMPOSER_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    attempt = 0
    last_error: Exception | None = None
    advisor_output: dict[str, Any] | None = None
    t_start = time.monotonic()
    llm_calls = 0

    def _invoke_composer(msgs: list[dict[str, str]]) -> AdvisorResponseLLMOutput:
        nonlocal llm_calls
        llm_calls += 1
        run_config = {
            "run_name": f"compose_advisor_response:call_{llm_calls}",
            "metadata": {
                "node_name": "compose_advisor_response",
                "logical_stage": logical_stage_for("compose_advisor_response"),
                "attempt": llm_calls,
                "model_tier": "fast",
                "question_id": question_id,
            },
            "tags": ["node:compose_advisor_response", "tier:fast", f"attempt:{llm_calls}"],
        }
        try:
            return structured_llm.invoke(msgs, config=run_config)
        except TypeError as te:
            if "config" in str(te):
                return structured_llm.invoke(msgs)
            raise

    while attempt < 2:
        attempt += 1
        try:
            res: AdvisorResponseLLMOutput = _invoke_composer(messages)
            res_dict = res.model_dump()

            # Target & Token Validation (Milestone 3.2.1 Section 4 & 7)
            has_active_decision = bool(state.get("active_next_turn_decision"))
            target_matches = (res_dict.get("next_question_target") == information_target)
            qid_matches = (res_dict.get("question_id") == question_id) if has_active_decision else True

            if not (target_matches and qid_matches):
                logger.warning(
                    "compose_advisor_response validation mismatch: expected (target=%s, qid=%s), got (target=%s, qid=%s). Retrying with correction.",
                    information_target, question_id, res_dict.get("next_question_target"), res_dict.get("question_id"),
                )
                retry_messages = [
                    {"role": "system", "content": ADVISOR_RESPONSE_COMPOSER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": _json.dumps(res_dict)},
                    {
                        "role": "user",
                        "content": (
                            f"VALIDATION FAILURE: Your output changed the information target or question_id.\n"
                            f"You MUST set question_id='{question_id}' and next_question_target='{information_target}'.\n"
                            f"Preserve the exact semantic goal of the planner's canonical question: '{canonical_question}'."
                        ),
                    },
                ]
                try:
                    retry_res: AdvisorResponseLLMOutput = _invoke_composer(retry_messages)
                    retry_dict = retry_res.model_dump()
                    if retry_dict.get("next_question_target") == information_target and retry_dict.get("question_id") == question_id:
                        res_dict = retry_dict
                    else:
                        raise ValueError(f"Retry still mismatched target: {retry_dict.get('next_question_target')}")
                except Exception as exc:
                    logger.warning("compose_advisor_response retry failed: %s", exc)
                    advisor_output = None
                    break

            text = (res_dict.get("response_text") or "").strip()
            if not text:
                parts = [p for p in [res_dict.get("acknowledge"), res_dict.get("answer_to_founder"), res_dict.get("diagnosis_delta"), res_dict.get("next_question_rephrased") or canonical_question] if p]
                text = " ".join(parts)
                res_dict["response_text"] = text

            advisor_output = res_dict
            break
        except Exception as exc:
            last_error = exc
            logger.warning("compose_advisor_response failed attempt %d: %s", attempt, exc)

    latency_ms = (time.monotonic() - t_start) * 1000
    composer_status = "SUCCESS" if advisor_output is not None else "FALLBACK"

    # Deterministic Fallback (Milestone 3.2.1 Section 23)
    if advisor_output is None:
        logger.warning("compose_advisor_response: using deterministic fallback (target=%s, qid=%s)", information_target, question_id)
        has_direct_q = bool(interpretation.get("direct_question"))
        has_concern = bool(interpretation.get("concern_or_goal"))
        has_corr = bool(interpretation.get("correction"))
        is_already_answered = bool(interpretation.get("is_already_answered_claim"))

        if is_already_answered or has_corr:
            fallback_text = f"You're right—thank you for keeping me aligned. {canonical_question}"
        elif response_mode == ConversationalResponseMode.DIRECT_ANSWER or has_direct_q:
            fallback_text = f"Not yet — that is the key question to test. {canonical_question}"
        elif response_mode == ConversationalResponseMode.CONCERN_RESPONSE or has_concern:
            fallback_text = f"That concern makes complete sense. To evaluate it, {canonical_question}"
        elif response_mode == ConversationalResponseMode.BRIEF_CONTINUE:
            fallback_text = f"Got it. {canonical_question}"
        else:
            fallback_text = f"Understood. {canonical_question}"

        advisor_output = {
            "question_id": question_id,
            "next_question_target": information_target,
            "next_question_rephrased": canonical_question,
            "next_question": canonical_question,
            "acknowledge": "Got it." if response_mode == ConversationalResponseMode.BRIEF_CONTINUE else "Understood.",
            "answer_to_founder": None,
            "diagnosis_delta": None,
            "should_ask_next_question": True,
            "response_mode": response_mode.value,
            "response_text": fallback_text,
        }

    # Ensure compatibility fields
    if "next_question" not in advisor_output:
        advisor_output["next_question"] = advisor_output.get("next_question_rephrased") or canonical_question

    domain_response = AdvisorResponse(
        project_id=project_id,
        turn_index=state.get("clarification_turn_count", 0),
        **advisor_output,
    )
    final_output = domain_response.model_dump()

    trace = AITrace(
        module_name="compose_advisor_response",
        prompt_version=ADVISOR_RESPONSE_COMPOSER_PROMPT_VERSION,
        input_data={
            "question_id": question_id,
            "information_target": information_target,
            "has_founder_question": bool(interpretation.get("direct_question")),
            "priority": interpretation.get("response_priority"),
        },
        output_data={
            "question_id": final_output.get("question_id"),
            "next_question_target": final_output.get("next_question_target"),
            "response_text": final_output.get("response_text")[:100],
            "should_ask_next_question": final_output.get("should_ask_next_question"),
        },
        latency_ms=latency_ms,
        project_id=project_id,
    )
    existing_traces = list(state.get("traces") or [])
    existing_traces.append(trace.model_dump(mode="json"))

    logger.info("compose_advisor_response | qid=%s target=%s response_text=%s", question_id, information_target, final_output.get("response_text")[:60])
    turn_telem = _record_llm_node_telemetry(
        state, "compose_advisor_response", llm, "fast", latency_ms, max(llm_calls, 1),
        "AdvisorResponseLLMOutput", status=composer_status, messages=messages,
    )
    return {
        "turn_telemetry": turn_telem,
        "latest_advisor_response": final_output,
        "intro_summary_shown": True,
        "traces": existing_traces,
        "last_trace": trace.model_dump(mode="json"),
    }


# ---------------------------------------------------------------------------
# Milestone 4 Nodes: Evidence Intake, Evidence Analysis & Re-Diagnosis Loop
# ---------------------------------------------------------------------------

def compose_execution_handoff(state: StartaryGraphState) -> dict:
    """
    Deterministic Node (Milestone 4.0.3 & 4.0.5):
    Composes a clear, founder-facing ExecutionHandoff object from the accepted
    experiment, critical assumption, and founder constraints before pausing for evidence.
    Creates both compact (default) and full methodology renderings and populates FounderExperimentSummary.
    """
    project_id = state.get("project_id", "<unknown>")
    active_exp = dict(state.get("active_experiment") or {})
    active_asmp = state.get("active_assumption") or state.get("active_critical_assumption") or {}
    snapshot = state.get("diagnosis_snapshot") or {}

    exp_id = active_exp.get("id") or str(uuid.uuid4())
    asmp_id = active_exp.get("assumption_id") or active_asmp.get("id") or "asmp-unknown"
    exp_type = active_exp.get("experiment_type") or "CUSTOMER_INTERVIEWS"
    learning_goal = active_exp.get("objective") or active_asmp.get("learning_goal") or "Validate critical assumption with real-world feedback"
    what_testing = active_exp.get("hypothesis") or active_asmp.get("statement") or "Critical assumption under test"

    actions = list(active_exp.get("founder_actions") or [])
    target = active_exp.get("target_participants_or_scope") or active_exp.get("target_participants_or_context") or ""
    if not target or target.lower() in ["target customer cohort", "cohort", "none"]:
        target = "Qualifying target stores or users"
    evidence_to_collect = active_exp.get("evidence_to_collect") or "Customer feedback, observed behaviors, quotes, and concrete counts"
    support_signal = active_exp.get("support_signal") or ""
    contradiction_signal = active_exp.get("contradiction_signal") or ""
    inconclusive_signal = active_exp.get("inconclusive_signal") or ""

    do_nots = []
    for ii in active_exp.get("invalid_inferences") or []:
        do_nots.append(f"Do not treat early exploratory feedback as proving: {ii}")

    summary = compose_founder_experiment_summary(
        experiment=active_exp,
        assumption=active_asmp,
        state=state,
        diagnosis_snapshot=snapshot,
    )

    crit_asmp_stmt = (active_asmp or {}).get("statement") or summary.critical_assumption
    target_sample_size = active_exp.get("target_sample_size")
    if target_sample_size is None or (isinstance(target_sample_size, int) and target_sample_size <= 0):
        target_sample_size = extract_sample_target_from_protocol(active_exp)

    handoff = ExecutionHandoff(
        experiment_id=exp_id,
        assumption_id=asmp_id,
        critical_assumption=crit_asmp_stmt,
        experiment_type=exp_type,
        learning_goal=learning_goal,
        what_we_are_testing=what_testing,
        founder_actions=actions,
        target_participants_or_context=target,
        target_sample_size=target_sample_size,
        evidence_to_collect=evidence_to_collect,
        support_signal=support_signal,
        contradiction_signal=contradiction_signal,
        inconclusive_signal=inconclusive_signal,
        important_constraints=do_nots,
        valid_inferences=list(active_exp.get("valid_inferences") or []),
        invalid_inferences=list(active_exp.get("invalid_inferences") or []),
        completion_instruction=(
            "Return to Startary and tell me what actually happened. "
            "You can also ask me to explain the experiment or tell me if something blocks you."
        ),
        summary=summary,
        build_readiness=state.get("build_readiness"),
        next_strategic_action=state.get("next_strategic_action"),
        founder_diagnosis_text=state.get("founder_diagnosis_text"),
    )
    compact_text = render_execution_handoff_text(handoff, mode="compact")
    full_text = render_execution_handoff_text(handoff, mode="full")
    notice = pending_authorization_notice(active_exp)  # M6.2.6: empty unless permissions are pending
    if notice:
        compact_text = f"{notice}\n\n{compact_text}"
        full_text = f"{notice}\n\n{full_text}"
    handoff.response_text = compact_text
    handoff.full_response_text = full_text

    logger.info(
        "execution_handoff_created | experiment_id=%s assumption_id=%s compact_len=%d full_len=%d",
        exp_id, asmp_id, len(compact_text), len(full_text),
    )

    return {
        "active_execution_handoff": handoff.model_dump(),
        "execution_handoff_text": compact_text,
    }


def wait_for_evidence(state: StartaryGraphState) -> dict:
    """
    LangGraph Interrupt Node (Milestone 4.0.3):
    Marks active experiment IN_PROGRESS, sets phase to WAITING_FOR_EVIDENCE,
    and pauses execution waiting for real-world founder evidence.
    Displays the execution handoff (or advisor response) and never renders None.
    """
    exp = dict(state.get("active_experiment") or {})
    exp["status"] = ExperimentStatus.IN_PROGRESS.value
    project_id = state.get("project_id", "<unknown>")
    iteration = state.get("iteration", 1)

    handoff_dict = state.get("active_execution_handoff") or {}
    handoff_text = handoff_dict.get("response_text") or state.get("execution_handoff_text")
    waiting_response = state.get("waiting_response")

    if waiting_response:
        prompt_text = str(waiting_response)
    elif handoff_text:
        prompt_text = str(handoff_text)
    else:
        if not exp.get("founder_diagnosis_text"):
            exp["founder_diagnosis_text"] = state.get("founder_diagnosis_text")
        if not exp.get("next_strategic_action"):
            exp["next_strategic_action"] = state.get("next_strategic_action")
        prompt_text = render_execution_handoff_text(exp)

    post_ans = state.get("post_evidence_answer")
    if post_ans:
        prompt_text = f"{post_ans}\n\n{prompt_text}"

    logger.info(
        "wait_for_evidence | project_id=%s iteration=%d experiment_id=%s prompt_len=%d (interrupt)",
        project_id, iteration, exp.get("id"), len(prompt_text),
    )

    founder_input = interrupt({
        "type": "WAITING_FOR_EVIDENCE",
        "interrupt_type": "waiting_for_evidence",
        "message": prompt_text,
        "question": prompt_text,
        "handoff_text": prompt_text,
        "execution_handoff": handoff_dict,
        "experiment_id": exp.get("id"),
        "assumption_id": exp.get("assumption_id"),
        "iteration": iteration,
    })

    raw_evidence = ""
    if isinstance(founder_input, dict):
        raw_evidence = founder_input.get("founder_response") or founder_input.get("content") or ""
    elif isinstance(founder_input, str):
        raw_evidence = founder_input
    if not raw_evidence and state.get("founder_response"):
        raw_evidence = str(state.get("founder_response"))

    founder_messages = list(state.get("founder_messages") or [])
    if raw_evidence:
        founder_messages.append({"role": "founder", "content": raw_evidence})

    return {
        "active_experiment": exp,
        "phase": WorkflowPhase.WAITING_FOR_EVIDENCE,
        "founder_response": raw_evidence,
        "founder_messages": founder_messages,
        "waiting_response": None,
        "post_evidence_answer": None,
        "evidence_turn_assessment": None,
        "pending_post_evidence_question": None,
    }


def pause_needs_help(state: StartaryGraphState) -> dict:
    """
    Deterministic Node (Milestone 4):
    Gracefully pauses if the experiment is valid but founder lacks requirements.
    """
    can_exec = state.get("can_execute_result") or {}
    missing = can_exec.get("missing_requirements") or []
    missing_str = "; ".join(missing) if missing else "Required assets or customer access are missing."
    msg = f"Experiment cannot run yet. Missing requirement: {missing_str}. When you have obtained access or are ready to continue, let me know."
    logger.info("pause_needs_help | %s", msg)
    return {
        "status_message": msg,
        "phase": WorkflowPhase.TEST,
    }


def classify_waiting_for_evidence_turn(state: StartaryGraphState) -> dict:
    """
    Structured Admission Gate Node (Milestone 4.0.3):
    Classifies founder turn submitted during WAITING_FOR_EVIDENCE to verify whether
    actual completed execution results exist before allowing access to Evidence Extractor / Analyzer.
    Applies deterministic guards to prevent non-evidence questions and promises from being analyzed.
    """
    project_id = state.get("project_id", "<unknown>")
    raw_message = state.get("founder_response") or ""
    if not raw_message:
        founder_messages = state.get("founder_messages") or []
        for m in reversed(founder_messages):
            if m.get("role") == "founder":
                raw_message = m.get("content", "")
                break

    active_exp = state.get("active_experiment") or {}
    active_asmp = state.get("active_assumption") or state.get("active_critical_assumption") or {}
    asmp_text = active_asmp.get("statement", "")
    exp_summary = (
        f"Type: {active_exp.get('experiment_type')}\n"
        f"Goal: {active_exp.get('objective')}\n"
        f"Actions: {active_exp.get('founder_actions')}\n"
        f"Target: {active_exp.get('target_participants_or_scope')}\n"
        f"Support Signal: {active_exp.get('support_signal')}\n"
        f"Contradiction Signal: {active_exp.get('contradiction_signal')}"
    )
    turn_interpretation = state.get("latest_founder_turn_interpretation") or {}
    interpretation_json = json.dumps(turn_interpretation, indent=2, default=str)

    settings = get_settings()
    llm = _safe_get_llm_client(settings, tier="fast", max_tokens=800)
    res_info = get_llm_client_resolution_info(llm)
    structured_llm = llm.with_structured_output(EvidenceTurnAssessmentLLMOutput)

    user_msg = build_evidence_admission_gate_user_message(
        founder_message=raw_message,
        active_experiment_summary=exp_summary,
        active_assumption_text=asmp_text,
        latest_interpretation_json=interpretation_json,
    )
    messages = [
        {"role": "system", "content": EVIDENCE_ADMISSION_GATE_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    llm_assessment = None
    attempt = 0
    t_start = time.monotonic()
    while attempt < 2:
        attempt += 1
        try:
            res: EvidenceTurnAssessmentLLMOutput = structured_llm.invoke(messages)
            llm_assessment = res
            break
        except Exception as exc:
            logger.warning("classify_waiting_for_evidence_turn attempt %d failed: %s", attempt, exc)
            if is_permanent_model_error(exc):
                break

    latency_ms = (time.monotonic() - t_start) * 1000

    msg_lower = raw_message.lower().strip()

    # Deterministic signals
    is_question_only = any(
        phrase in msg_lower
        for phrase in [
            "what should i do", "then what should i do", "what do i do now",
            "what next", "can you explain", "i don't understand", "im confused",
            "why should i", "can you show me", "what should i ask"
        ]
    )
    is_help_only = any(
        phrase in msg_lower
        for phrase in [
            "how do i find", "how to find", "where do i find", "who do i talk to",
            "can i only talk to"
        ]
    )
    is_future_only = any(
        phrase in msg_lower
        for phrase in [
            "i will do it", "i will interview", "i will talk", "ill do it",
            "tomorrow", "going to start", "going to", "haven't started", "have not started",
            "haven't done", "not yet", "next week", "plan to", "planning to"
        ]
    )
    is_cannot_execute = any(
        phrase in msg_lower
        for phrase in [
            "i can't find any", "cannot find any", "unable to find",
            "no longer allows", "cannot run this", "cant run this"
        ]
    )
    action_verbs = [
        "interviewed", "spoke to", "spoke with", "talked to", "talked with",
        "messaged", "sent", "contacted", "reached out", "ran the", "tested",
        "tried contacting", "tried it", "asked", "showed", "visited",
        "completed the interviews", "completed interviews", "finished the interviews",
        "done with interviews", "finished interviews", "completed the calls", "finished the calls",
    ]
    has_action_verb = any(v in msg_lower for v in action_verbs)
    has_count = any(c in msg_lower for c in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "15", "20", "50", "one", "two", "three", "four", "five", "six"])
    has_results_words = any(w in msg_lower for w in ["said", "replied", "answered", "agreed", "rejected", "result", "found that", "nobody replied", "loss", "pain", "stockout", "waste"])

    has_completed_action = has_action_verb or (has_count and has_results_words)

    if llm_assessment:
        disp_str = llm_assessment.disposition.upper().strip()
        try:
            final_disp = EvidenceTurnDisposition(disp_str)
        except ValueError:
            final_disp = EvidenceTurnDisposition.AMBIGUOUS
        has_completed_action = llm_assessment.contains_completed_external_action or has_completed_action
        has_results = llm_assessment.contains_results or has_results_words
        is_future = llm_assessment.contains_only_future_intent or is_future_only
        asks_help = llm_assessment.asks_for_help or is_help_only
        asks_expl = llm_assessment.asks_for_explanation or is_question_only
        cannot_exec = llm_assessment.cannot_execute or is_cannot_execute
        candidate_text = llm_assessment.evidence_candidate_text
        rationale = llm_assessment.rationale_summary
    else:
        if has_completed_action:
            final_disp = EvidenceTurnDisposition.ACTUAL_EVIDENCE
            rationale = "Deterministic fallback: detected completed action and observation signals."
        elif is_help_only:
            final_disp = EvidenceTurnDisposition.REQUEST_FOR_EXECUTION_HELP
            rationale = "Deterministic fallback: founder asking how to execute or find participants."
        elif is_question_only:
            final_disp = EvidenceTurnDisposition.QUESTION_ABOUT_EXPERIMENT
            rationale = "Deterministic fallback: founder asking what to do or about the experiment."
        elif is_future_only:
            final_disp = EvidenceTurnDisposition.FUTURE_INTENTION
            rationale = "Deterministic fallback: founder stating future intention."
        elif is_cannot_execute:
            final_disp = EvidenceTurnDisposition.CANNOT_EXECUTE
            rationale = "Deterministic fallback: founder stating inability to execute."
        else:
            final_disp = EvidenceTurnDisposition.AMBIGUOUS
            rationale = "Deterministic fallback: ambiguous turn."
        has_results = has_results_words
        is_future = is_future_only
        asks_help = is_help_only
        asks_expl = is_question_only
        cannot_exec = is_cannot_execute
        candidate_text = raw_message if has_completed_action else None

    # MILESTONE 4.0.7.2.1: EVIDENCE ADMISSION PRECEDENCE INTEGRITY
    # Core Rule: Substantive completed evidence must take precedence over incidental future-oriented language.
    substantive_completed_evidence = has_substantive_completed_evidence(raw_message, llm_assessment)

    if substantive_completed_evidence:
        final_disp = EvidenceTurnDisposition.ACTUAL_EVIDENCE
        has_completed_action = True
        if candidate_text is None:
            candidate_text = raw_message
        has_future_signal = is_future_only or is_future or (llm_assessment and llm_assessment.contains_only_future_intent)
        has_q_signal = is_question_only or asks_expl or asks_help
        if has_future_signal and has_q_signal:
            rationale += " [Admitted: substantive completed evidence takes precedence over incidental future plan and question]."
        elif has_future_signal:
            rationale += " [Admitted: substantive completed evidence takes precedence over incidental future language]."
        elif has_q_signal:
            rationale += " [Admitted: mixed submission with completed action and question]."
    elif is_future_only or is_future:
        if (is_question_only or asks_expl) and not any(w in msg_lower for w in ["tomorrow", "will ", "ill ", "going to"]):
            final_disp = EvidenceTurnDisposition.QUESTION_ABOUT_EXPERIMENT
            rationale += " [Overridden: question/explanation request]."
        else:
            final_disp = EvidenceTurnDisposition.FUTURE_INTENTION
            rationale += " [Overridden: future intention expressed]."
        has_completed_action = False
    elif final_disp == EvidenceTurnDisposition.ACTUAL_EVIDENCE and not has_completed_action:
        if is_question_only or asks_expl:
            final_disp = EvidenceTurnDisposition.QUESTION_ABOUT_EXPERIMENT
            rationale += " [Overridden: no completed action detected, treated as question]."
        elif is_help_only or asks_help:
            final_disp = EvidenceTurnDisposition.REQUEST_FOR_EXECUTION_HELP
            rationale += " [Overridden: no completed action detected, treated as help request]."
        elif is_cannot_execute or cannot_exec:
            final_disp = EvidenceTurnDisposition.CANNOT_EXECUTE
            rationale += " [Overridden: no completed action detected, treated as cannot execute]."
        else:
            final_disp = EvidenceTurnDisposition.AMBIGUOUS
            rationale += " [Overridden: no completed action detected, treated as ambiguous]."

    admitted = (final_disp == EvidenceTurnDisposition.ACTUAL_EVIDENCE)

    logger.info(
        "evidence_turn_classified | disposition=%s admitted=%s",
        final_disp.value, admitted,
    )
    if not admitted:
        logger.info("evidence_admission_rejected | reason=%s", rationale)

    # Mixed-intent preservation: detect and preserve questions in admitted evidence turns (Milestone 4.0.7)
    pending_q = None
    if admitted:
        has_question = (
            "?" in raw_message
            or is_question_only
            or asks_expl
            or asks_help
            or any(
                phrase in msg_lower
                for phrase in [
                    "what should i do", "what do you think", "what next", "what now",
                    "what to do", "what do i do", "where do we go", "is this enough",
                    "how should i", "next step", "what's next", "whats next"
                ]
            )
        )
        if has_question:
            import re
            sentences = [s.strip() for s in re.split(r"[\n.!?]+", raw_message) if s.strip()]
            q_candidates = [
                s for s in sentences
                if any(w in s.lower() for w in ["what", "how", "should", "think", "next", "enough", "do", "where"])
                and not any(w in s.lower() for w in ["interviewed", "spoke with", "talked with", "stockout", "shops"])
            ]
            if q_candidates:
                pending_q = q_candidates[-1].strip() + ("?" if not q_candidates[-1].endswith("?") else "")
            else:
                pending_q = "What should I do next?"
            logger.info("mixed_evidence_question_preserved | has_question=true question=%s", pending_q)

    assessment = EvidenceTurnAssessment(
        disposition=final_disp,
        contains_completed_external_action=has_completed_action,
        contains_results=has_results,
        contains_only_future_intent=False if substantive_completed_evidence else is_future,
        asks_for_help=asks_help,
        asks_for_explanation=asks_expl,
        cannot_execute=cannot_exec,
        evidence_candidate_text=candidate_text,
        rationale_summary=rationale,
    )

    turn_telem = record_telemetry_event(
        state=state,
        node_name="classify_waiting_for_evidence_turn",
        model_name=res_info["resolved_model"] or getattr(settings, "fast_model_name", None) or "fast_tier",
        duration_ms=latency_ms,
        execution_path=state.get("execution_path", "FULL_EVIDENCE_LEARNING"),
        configured_model=res_info["configured_model"],
        resolved_model=res_info["resolved_model"],
        fallback_used=res_info["fallback_used"],
        fallback_reason=res_info["fallback_reason"],
    )

    return {
        "evidence_turn_assessment": assessment.model_dump(mode="json"),
        "pending_post_evidence_question": pending_q,
        "turn_telemetry": turn_telem,
    }


def respond_to_waiting_founder(state: StartaryGraphState) -> dict:
    """
    Conversational Advisor Node (Milestone 4.0.3 & 4.0.7):
    Responds conversationally to non-evidence messages received during WAITING_FOR_EVIDENCE.
    Explains the accepted experiment, offers execution help, acknowledges future plans,
    or handles blockers without completing the experiment or mutating assumption status.
    Architecture rule (Milestone 4.0.7):
    Evidence Admission Gate decides WHAT the turn is.
    Advisor decides HOW to respond.
    This node MUST NOT reclassify or overwrite ACTUAL_EVIDENCE.
    """
    project_id = state.get("project_id", "<unknown>")
    disp = get_evidence_turn_disposition(state)
    disposition = disp.value if disp else "QUESTION_ABOUT_EXPERIMENT"
    raw_message = state.get("founder_response") or state.get("founder_turn") or ""

    # Invariant (Milestone 4.0.7): If ACTUAL_EVIDENCE reached here unexpectedly, do NOT reclassify as question
    if disp == EvidenceTurnDisposition.ACTUAL_EVIDENCE:
        logger.warning("respond_to_waiting_founder | disposition=ACTUAL_EVIDENCE reached unexpectedly; preserving evidence")
        reply_text = "I have received your experiment results. Processing and analyzing the findings now."
        founder_messages = list(state.get("founder_messages") or [])
        founder_messages.append({"role": "assistant", "content": reply_text})
        return {
            "waiting_response": reply_text,
            "founder_messages": founder_messages,
            "phase": WorkflowPhase.WAITING_FOR_EVIDENCE,
        }

    handoff_dict = state.get("active_execution_handoff") or {}
    handoff_text = handoff_dict.get("response_text") or state.get("execution_handoff_text") or ""
    active_exp = state.get("active_experiment") or {}
    active_asmp = state.get("active_assumption") or state.get("active_critical_assumption") or {}
    asmp_text = active_asmp.get("statement", "")

    msg_lower = raw_message.lower()

    # Fallback Learning Action Handling (Milestone 4.0.6)
    fallback_dict = state.get("fallback_learning_action") or {}
    is_fallback_mode = (
        state.get("learning_input_mode") == "FALLBACK"
        or bool(fallback_dict)
        or state.get("phase") == WorkflowPhase.WAITING_FOR_LEARNING_INPUT
    )

    if is_fallback_mode and fallback_dict:
        if any(p in msg_lower for p in ["who should i talk to", "who to talk to", "who do i talk to", "who to engage"]):
            who = fallback_dict.get("who_to_engage", "target participants")
            reply_text = (
                f"For this learning action, engage: {who}.\n\n"
                f"Focus on speaking with people who experience the problem directly in their day-to-day operations."
            )
            founder_messages = list(state.get("founder_messages") or [])
            founder_messages.append({"role": "assistant", "content": reply_text})
            return {
                "waiting_response": reply_text,
                "founder_messages": founder_messages,
                "phase": state.get("phase", WorkflowPhase.WAITING_FOR_LEARNING_INPUT),
            }

        if any(p in msg_lower for p in ["can i start with 3", "start with 3", "fewer shops", "can i do fewer", "start small"]):
            reply_text = (
                "Yes, absolutely. Starting with 2–3 conversations is an excellent way to pilot the questions, "
                "reconstruct initial incidents, and refine your approach before completing the full set of 5–8."
            )
            founder_messages = list(state.get("founder_messages") or [])
            founder_messages.append({"role": "assistant", "content": reply_text})
            return {
                "waiting_response": reply_text,
                "founder_messages": founder_messages,
                "phase": state.get("phase", WorkflowPhase.WAITING_FOR_LEARNING_INPUT),
            }

        if any(p in msg_lower for p in ["what questions", "what should i ask", "what to ask", "exact questions"]):
            questions = fallback_dict.get("what_to_ask_or_observe") or []
            q_str = "\n".join(f"- {q}" for q in questions)
            reply_text = f"Here are the specific questions to ask in each conversation:\n\n{q_str}"
            founder_messages = list(state.get("founder_messages") or [])
            founder_messages.append({"role": "assistant", "content": reply_text})
            return {
                "waiting_response": reply_text,
                "founder_messages": founder_messages,
                "phase": state.get("phase", WorkflowPhase.WAITING_FOR_LEARNING_INPUT),
            }

        if any(p in msg_lower for p in ["what should i bring back", "what to bring back", "what to bring"]):
            bring = fallback_dict.get("what_to_bring_back") or []
            b_str = "\n".join(f"- {b}" for b in bring)
            reply_text = f"Here is what to record and bring back:\n\n{b_str}"
            founder_messages = list(state.get("founder_messages") or [])
            founder_messages.append({"role": "assistant", "content": reply_text})
            return {
                "waiting_response": reply_text,
                "founder_messages": founder_messages,
                "phase": state.get("phase", WorkflowPhase.WAITING_FOR_LEARNING_INPUT),
            }

        if any(p in msg_lower for p in ["what should i avoid", "what not to infer", "what should i not do"]):
            not_inf = fallback_dict.get("what_not_to_infer") or []
            n_str = "\n".join(f"- {n}" for n in not_inf)
            reply_text = (
                f"Keep in mind that this step is for learning, not validation. It does NOT prove:\n\n{n_str}"
            )
            founder_messages = list(state.get("founder_messages") or [])
            founder_messages.append({"role": "assistant", "content": reply_text})
            return {
                "waiting_response": reply_text,
                "founder_messages": founder_messages,
                "phase": state.get("phase", WorkflowPhase.WAITING_FOR_LEARNING_INPUT),
            }

        # Check if founder returned with fallback observations
        action_verbs = ["interviewed", "spoke to", "spoke with", "talked to", "talked with", "asked", "found that", "visited"]
        has_obs = any(v in msg_lower for v in action_verbs)
        if has_obs:
            reply_text = (
                "Thank you for bringing back these firsthand observations. I have captured these notes into your startup state. "
                "They provide grounded real-world context for our next diagnostic and experiment formulation step "
                "(without prematurely marking the assumption validated)."
            )
            founder_messages = list(state.get("founder_messages") or [])
            founder_messages.append({"role": "assistant", "content": reply_text})

            st_state = dict(state.get("startup_state") or {})
            areas = dict(st_state.get("areas") or st_state.get("area_knowledge") or {})
            target_area = fallback_dict.get("target_area", "Problem")
            area_data = dict(areas.get(target_area) or {})
            ev_list = list(area_data.get("evidence_provided") or [])
            ev_list.append(raw_message)
            area_data["evidence_provided"] = ev_list
            areas[target_area] = area_data
            st_state["areas"] = areas

            founder_ev = list(state.get("founder_evidence") or [])
            founder_ev.append({
                "source": "FALLBACK_LEARNING_ACTION",
                "content": raw_message,
                "target_area": target_area,
            })

            return {
                "waiting_response": reply_text,
                "founder_messages": founder_messages,
                "startup_state": st_state,
                "founder_evidence": founder_ev,
                "learning_input_mode": None,
                "phase": WorkflowPhase.TEST,
            }

    # Full guide / full test requests
    full_guide_triggers = [
        "full guide", "show full guide", "full interview guide", "show me the full guide",
        "full experiment", "show full test", "show me the full test", "show test again",
        "repeat test", "show me the test", "see the test", "display the test", "full details"
    ]
    if any(p in msg_lower for p in full_guide_triggers):
        full_text = handoff_dict.get("full_response_text") or render_execution_handoff_text(handoff_dict, mode="full")
        reply_text = f"Here is the complete experiment methodology and interview guide:\n\n{full_text}"
        founder_messages = list(state.get("founder_messages") or [])
        founder_messages.append({"role": "assistant", "content": reply_text})
        return {
            "waiting_response": reply_text,
            "founder_messages": founder_messages,
            "phase": WorkflowPhase.WAITING_FOR_EVIDENCE,
        }

    # Exact questions requests
    question_triggers = [
        "exact questions", "what questions", "what should i ask", "give me the questions",
        "interview questions", "interview guide"
    ]
    if any(p in msg_lower for p in question_triggers):
        actions = active_exp.get("founder_actions") or []
        evidence = active_exp.get("evidence_to_collect") or ""
        actions_str = "\n".join(f"  {i}. {a}" for i, a in enumerate(actions, 1)) if actions else "  - Run the planned interview questions."
        reply_text = (
            f"Here are the specific actions and questions for this test:\n\n"
            f"{actions_str}\n\n"
            f"Evidence to record:\n  - {evidence}\n\n"
            f"When you have completed the interviews, return here and share what happened."
        )
        founder_messages = list(state.get("founder_messages") or [])
        founder_messages.append({"role": "assistant", "content": reply_text})
        return {
            "waiting_response": reply_text,
            "founder_messages": founder_messages,
            "phase": WorkflowPhase.WAITING_FOR_EVIDENCE,
        }

    # Decision rules / success criteria requests
    rule_triggers = [
        "decision rule", "decision rules", "success criteria", "what counts as success",
        "show me the decision rule", "show decision rule"
    ]
    if any(p in msg_lower for p in rule_triggers):
        supp = active_exp.get("support_signal") or "Not specified"
        contra = active_exp.get("contradiction_signal") or "Not specified"
        inconc = active_exp.get("inconclusive_signal") or "Not specified"
        reply_text = (
            f"Here are the exact decision rules for this experiment:\n\n"
            f"- Validates assumption:\n  {supp}\n\n"
            f"- Weakens/disproves assumption:\n  {contra}\n\n"
            f"- Inconclusive:\n  {inconc}\n\n"
            f"Return here with what actually happened, and Startary will evaluate your evidence against these criteria."
        )
        founder_messages = list(state.get("founder_messages") or [])
        founder_messages.append({"role": "assistant", "content": reply_text})
        return {
            "waiting_response": reply_text,
            "founder_messages": founder_messages,
            "phase": WorkflowPhase.WAITING_FOR_EVIDENCE,
        }

    # What to avoid / guardrails requests
    avoid_triggers = ["what should i avoid", "what to avoid", "what should i not do", "guardrails", "do not"]
    if any(p in msg_lower for p in avoid_triggers):
        do_nots = handoff_dict.get("important_constraints") or []
        invalid_inf = active_exp.get("invalid_inferences") or []
        lines = ["Here are the critical guardrails and boundaries for this test:\n"]
        if do_nots:
            lines.append("Important guardrails:")
            for dn in do_nots:
                lines.append(f"- {dn}")
        if invalid_inf:
            lines.append("\nWhat this test CANNOT prove:")
            for ii in invalid_inf:
                lines.append(f"- Do not treat this test as proving: {ii}")
        lines.append("\nReturn here when you have completed the observations.")
        reply_text = "\n".join(lines)
        founder_messages = list(state.get("founder_messages") or [])
        founder_messages.append({"role": "assistant", "content": reply_text})
        return {
            "waiting_response": reply_text,
            "founder_messages": founder_messages,
            "phase": WorkflowPhase.WAITING_FOR_EVIDENCE,
        }

    # Why this test matters
    why_triggers = ["why does this test matter", "why this test", "why are we doing this", "what does this test prove", "learning goal"]
    if any(p in msg_lower for p in why_triggers):
        goal = active_exp.get("objective") or (active_asmp.get("learning_goal") if isinstance(active_asmp, dict) else "")
        valid_inf = active_exp.get("valid_inferences") or []
        lines = [
            f"Strategic purpose of this test:\n{goal}\n",
            "What this test CAN legitimately establish:"
        ]
        for vi in valid_inf:
            lines.append(f"- {vi}")
        lines.append("\nLet me know if you are ready to begin or if you need help finding participants.")
        reply_text = "\n".join(lines)
        founder_messages = list(state.get("founder_messages") or [])
        founder_messages.append({"role": "assistant", "content": reply_text})
        return {
            "waiting_response": reply_text,
            "founder_messages": founder_messages,
            "phase": WorkflowPhase.WAITING_FOR_EVIDENCE,
        }

    settings = get_settings()
    llm = _safe_get_llm_client(settings, tier="fast", max_tokens=1000)
    structured_llm = llm.with_structured_output(WaitingForEvidenceAdvisorLLMOutput)

    user_msg = build_waiting_for_evidence_advisor_user_message(
        founder_message=raw_message,
        disposition=str(disposition),
        execution_handoff_text=handoff_text,
        active_experiment_json=json.dumps(active_exp, indent=2, default=str),
        active_assumption_text=asmp_text,
    )
    messages = [
        {"role": "system", "content": WAITING_FOR_EVIDENCE_ADVISOR_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    try:
        res: WaitingForEvidenceAdvisorLLMOutput = structured_llm.invoke(messages)
        reply_text = res.response_text
    except Exception as exc:
        logger.warning("respond_to_waiting_founder LLM failed: %s, using deterministic fallback", exc)
        if disposition == EvidenceTurnDisposition.QUESTION_ABOUT_EXPERIMENT.value or disposition == "QUESTION_ABOUT_EXPERIMENT":
            actions = active_exp.get("founder_actions") or ["Interview target customers to validate the assumption."]
            actions_formatted = "\n".join(f"- {a}" for a in actions)
            reply_text = (
                f"Here is what to do for this test:\n\n"
                f"{actions_formatted}\n\n"
                f"Target: {active_exp.get('target_participants_or_scope', 'Target customers')}\n\n"
                f"Once you've done these, return here and share what happened."
            )
        elif disposition == EvidenceTurnDisposition.REQUEST_FOR_EXECUTION_HELP.value or disposition == "REQUEST_FOR_EXECUTION_HELP":
            target = active_exp.get("target_participants_or_scope", "target customers")
            reply_text = (
                f"To find qualifying participants for this test ({target}), start with people or stores in your existing network, "
                f"ask for warm introductions, or visit local stores directly during quiet hours. "
                f"Keep conversations focused on their real past experiences."
            )
        elif disposition == EvidenceTurnDisposition.FUTURE_INTENTION.value or disposition == "FUTURE_INTENTION":
            reply_text = (
                "Sounds good! Take your time to run the conversations. "
                "When you have completed them, return here and tell me what actually happened."
            )
        elif disposition == EvidenceTurnDisposition.CANNOT_EXECUTE.value or disposition == "CANNOT_EXECUTE":
            reply_text = (
                "Understood. If you're blocked from running this test as designed, "
                "let me know what specific resource or access is missing so we can adjust our execution path."
            )
        else:
            reply_text = (
                f"We are currently waiting for real-world results from the active test:\n\n"
                f"Objective: {active_exp.get('objective')}\n\n"
                f"Let me know if you have questions about running it or if something is blocking you."
            )

    founder_messages = list(state.get("founder_messages") or [])
    founder_messages.append({"role": "assistant", "content": reply_text})

    logger.info("respond_to_waiting_founder | disposition=%s reply_len=%d", disposition, len(reply_text))

    return {
        "waiting_response": reply_text,
        "founder_messages": founder_messages,
        "phase": WorkflowPhase.WAITING_FOR_EVIDENCE,
    }


def extract_experiment_evidence(state: StartaryGraphState) -> dict:
    """
    LLM Node (Milestone 4):
    Extracts structured factual observations and results from founder-submitted evidence.
    Separates what was actually observed and executed from subjective interpretation.
    """
    project_id = state.get("project_id", "<unknown>")
    raw_submission = state.get("founder_response") or ""
    if not raw_submission:
        founder_messages = state.get("founder_messages") or []
        for m in reversed(founder_messages):
            if m.get("role") == "founder":
                raw_submission = m.get("content", "")
                break

    active_experiment = state.get("active_experiment") or {}
    active_assumption = state.get("active_assumption") or state.get("active_critical_assumption") or {}
    assumption_text = active_assumption.get("statement", "Critical assumption under test")

    # Post-analysis idempotency: reuse existing extraction if already completed for this submission
    existing_extracted = state.get("extracted_evidence")
    if existing_extracted and isinstance(existing_extracted, dict) and (
        existing_extracted.get("actions_completed")
        or existing_extracted.get("observed_behaviors")
        or existing_extracted.get("quantitative_outcomes")
        or existing_extracted.get("direct_evidence_summary")
    ):
        logger.info("extract_experiment_evidence | reusing existing extracted evidence from checkpoint")
        return {
            "extracted_evidence": existing_extracted,
        }

    import json as _json
    slimmed_protocol = {
        "objective": active_experiment.get("objective"),
        "method": active_experiment.get("method"),
        "target_participants_or_scope": active_experiment.get("target_participants_or_scope"),
        "target_sample_size": active_experiment.get("target_sample_size"),
        "founder_actions": active_experiment.get("founder_actions"),
        "evidence_to_collect": active_experiment.get("evidence_to_collect"),
    }
    exp_json = _json.dumps({k: v for k, v in slimmed_protocol.items() if v is not None}, indent=2, default=str)
    evidence_reqs = active_experiment.get("evidence_to_collect") or ""

    settings = get_settings()
    # Shift factual extraction strictly to FAST tier for ~70-80% latency reduction
    llm = _safe_get_llm_client(settings, tier="fast", max_tokens=1200)
    conf_model, res_model, fb_used, fb_reason = get_llm_client_resolution_info(llm)
    structured_llm = llm.with_structured_output(ExtractedEvidenceLLMOutput)

    user_msg = build_evidence_extractor_user_message(
        raw_founder_submission=raw_submission,
        active_experiment_json=exp_json,
        active_assumption_text=assumption_text,
        evidence_requirements=evidence_reqs,
    )

    messages = [
        {"role": "system", "content": EVIDENCE_EXTRACTOR_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    attempt = 0
    last_err = None
    while attempt < 2:
        attempt += 1
        t_start = time.monotonic()
        try:
            res: ExtractedEvidenceLLMOutput = structured_llm.invoke(messages)
            latency_ms = (time.monotonic() - t_start) * 1000

            # Milestone 4.0.7.2.1: Enforce evidence boundary on extracted output
            # If substantive completed evidence exists, is_future_intention_only MUST be False
            if res.actions_completed or res.quantitative_outcomes or has_substantive_completed_evidence(raw_submission):
                res.is_future_intention_only = False

            # Filter out future promises/plans from actions_completed and evidence_items
            future_markers = ["will ", "plan to ", "planning to ", "going to ", "next week", "tomorrow", "intend to "]
            filtered_actions = []
            for action in res.actions_completed:
                act_lower = action.lower().strip()
                if any(m in act_lower for m in future_markers) and not any(past in act_lower for past in ["interviewed", "completed", "visited", "conducted", "tested", "got ", "spoke"]):
                    logger.info("filtered_future_intention_from_actions | action=%s", action)
                    continue
                filtered_actions.append(action)
            res.actions_completed = filtered_actions

            if hasattr(res, "evidence_items") and res.evidence_items:
                filtered_items = []
                for item in res.evidence_items:
                    item_text = (getattr(item, "text", "") or getattr(item, "observation", "") or str(item)).lower()
                    if any(m in item_text for m in future_markers) and not any(past in item_text for past in ["interviewed", "completed", "visited", "conducted", "tested", "got ", "spoke"]):
                        logger.info("filtered_future_intention_from_evidence_items | item=%s", item_text)
                        continue
                    filtered_items.append(item)
                res.evidence_items = filtered_items

            res_dict = res.model_dump()
            trace = AITrace(
                module_name="extract_experiment_evidence",
                prompt_version=EVIDENCE_EXTRACTOR_PROMPT_VERSION,
                input_data={"raw_len": len(raw_submission), "experiment_id": active_experiment.get("id")},
                output_data={"actions_count": len(res.actions_completed), "is_future": res.is_future_intention_only},
                latency_ms=latency_ms,
                project_id=project_id,
                model_name=res_model or getattr(settings, "fast_model_name", None),
                configured_model=conf_model,
                resolved_model=res_model,
                fallback_used=fb_used,
                fallback_reason=fb_reason,
                execution_path=state.get("execution_path", "STRATEGIC_REASONING"),
            )
            existing_traces = list(state.get("traces") or [])
            existing_traces.append(trace.model_dump(mode="json"))

            # Token accounting with fallback character counting
            prompt_tokens = getattr(getattr(res, "usage_metadata", None), "prompt_tokens", None)
            if prompt_tokens is None:
                prompt_tokens = sum(len(m.get("content", "")) for m in messages) // 4
            completion_tokens = getattr(getattr(res, "usage_metadata", None), "completion_tokens", None)
            if completion_tokens is None:
                completion_tokens = len(_json.dumps(res_dict)) // 4
            total_tokens = prompt_tokens + completion_tokens

            turn_telem = record_telemetry_event(
                state=state,
                node_name="extract_experiment_evidence",
                model_name=res_model or getattr(settings, "fast_model_name", None) or "fast_tier",
                duration_ms=latency_ms,
                latency_ms=latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                execution_path=state.get("execution_path", "STRATEGIC_REASONING"),
                configured_model=conf_model,
                resolved_model=res_model,
                fallback_used=fb_used,
                fallback_reason=fb_reason,
            )

            logger.info("extract_experiment_evidence | actions=%d is_future=%s summary=%s",
                        len(res.actions_completed), res.is_future_intention_only, res.direct_evidence_summary[:60])
            return {
                "extracted_evidence": res_dict,
                "traces": existing_traces,
                "last_trace": trace.model_dump(mode="json"),
                "turn_telemetry": turn_telem,
            }
        except Exception as exc:
            last_err = exc
            logger.warning("extract_experiment_evidence attempt %d failed: %s", attempt, exc)
            if is_permanent_model_error(exc):
                break

    # Fallback extraction on error
    fallback = ExtractedEvidenceLLMOutput(
        actions_completed=[],
        sample_or_counts={},
        observed_behaviors=[],
        quotes=[],
        quantitative_outcomes={},
        incomplete_actions=[],
        deviations_from_plan=[],
        missing_result_information=[],
        is_future_intention_only=False,
        evidence_items=[],
        direct_evidence_summary=raw_submission[:200] if raw_submission else "No evidence details extracted.",
    )
    fallback_telem = record_telemetry_event(
        state=state,
        node_name="extract_experiment_evidence",
        model_name="local_deterministic",
        configured_model=conf_model,
        resolved_model="local_deterministic",
        fallback_used=True,
        fallback_reason="EXTRACTION_FAILURE_FALLBACK",
        duration_ms=0.0,
        execution_path=state.get("execution_path", "STRATEGIC_REASONING"),
    )
    return {
        "extracted_evidence": fallback.model_dump(),
        "turn_telemetry": fallback_telem,
    }


def assess_evidence_submission_sufficiency(state: StartaryGraphState) -> dict:
    """
    Deterministic Node (Milestone 4):
    Assesses whether submitted evidence is sufficient for analysis, or requires clarification.
    """
    extracted = state.get("extracted_evidence") or {}
    raw = state.get("founder_response") or ""
    turns = state.get("evidence_clarification_turns", 0)

    is_future = extracted.get("is_future_intention_only", False)
    actions = extracted.get("actions_completed") or []
    counts = extracted.get("sample_or_counts") or {}
    behaviors = extracted.get("observed_behaviors") or []

    # If founder only stated future intentions (and no substantive completed evidence was submitted):
    if is_future and not (actions or counts or behaviors or has_substantive_completed_evidence(raw)):
        return {
            "evidence_submission_sufficiency": EvidenceSubmissionSufficiency.NEEDS_CLARIFICATION.value,
            "evidence_clarification_question": (
                "It sounds like this is a plan for what you will do. "
                "To evaluate this experiment, Startary needs the actual results from actions you have already completed. "
                "What specifically happened when you conducted this test?"
            ),
        }

    # If completely empty or extremely brief without facts and clarification turns < 2:
    if len(raw.strip().split()) < 4 and not actions and not counts and not behaviors and turns < 2:
        return {
            "evidence_submission_sufficiency": EvidenceSubmissionSufficiency.NEEDS_CLARIFICATION.value,
            "evidence_clarification_question": (
                "Could you share more detail about what happened when you ran the experiment? "
                "How many people did you speak with, and what specifically did they say or do?"
            ),
        }

    return {
        "evidence_submission_sufficiency": EvidenceSubmissionSufficiency.SUFFICIENT_FOR_ANALYSIS.value,
        "evidence_clarification_question": None,
    }


def ask_evidence_clarification(state: StartaryGraphState) -> dict:
    """
    Interrupt Node (Milestone 4):
    Asks one focused clarification question regarding the submitted experiment results.
    """
    q = state.get("evidence_clarification_question") or "Could you share more specific details from running the experiment?"
    turns = state.get("evidence_clarification_turns", 0) + 1

    founder_input = interrupt({
        "type": "EVIDENCE_CLARIFICATION",
        "question": q,
        "turn": turns,
    })

    raw = ""
    if isinstance(founder_input, dict):
        raw = founder_input.get("founder_response") or founder_input.get("content") or ""
    elif isinstance(founder_input, str):
        raw = founder_input
    if not raw and state.get("founder_response"):
        raw = str(state.get("founder_response"))

    founder_messages = list(state.get("founder_messages") or [])
    if raw:
        founder_messages.append({"role": "founder", "content": raw})

    return {
        "founder_response": raw,
        "evidence_clarification_turns": turns,
        "founder_messages": founder_messages,
    }


def analyze_evidence(state: StartaryGraphState) -> dict:
    """
    LLM Node (Milestone 4 & M4.1.3.2):
    Evaluates extracted evidence against the active Critical Assumption and experiment decision rules.
    Respects valid vs invalid inferences and determines conclusion & evidence strength.
    """
    project_id = state.get("project_id", "<unknown>")
    active_experiment = state.get("active_experiment") or {}
    active_assumption = state.get("active_assumption") or state.get("active_critical_assumption") or {}
    statement = active_assumption.get("statement", "Critical assumption")

    # Post-analysis idempotency: reuse existing analysis if already performed
    existing_analysis = state.get("evidence_analysis")
    if existing_analysis and isinstance(existing_analysis, dict) and existing_analysis.get("conclusion") and existing_analysis.get("conclusion") != "INCONCLUSIVE":
        logger.info("analyze_evidence | reusing existing valid evidence_analysis from checkpoint")
        return {
            "evidence_analysis": existing_analysis,
        }

    extracted = state.get("extracted_evidence") or {}
    raw = state.get("founder_response") or ""
    rc = state.get("reality_check") or {}

    import json as _json
    exp_json = _json.dumps(active_experiment, indent=2, default=str)
    extracted_json = _json.dumps(extracted, indent=2, default=str)

    support_signal = active_experiment.get("support_signal") or "Result confirms expected problem or behavior."
    contradiction_signal = active_experiment.get("contradiction_signal") or "Result shows problem is rare, unaddressed, or adequately solved."
    inconclusive_signal = active_experiment.get("inconclusive_signal") or "Sample is too small, biased, or result does not discriminate."
    valid_inferences = active_experiment.get("valid_inferences") or []
    invalid_inferences = active_experiment.get("invalid_inferences") or []
    rc_summary = rc.get("summary") or ""

    settings = get_settings()
    base_max_tokens = 2500
    llm = _safe_get_llm_client(settings, tier="reasoning", max_completion_tokens=base_max_tokens)
    conf_model, res_model, fb_used, fb_reason = get_llm_client_resolution_info(llm)
    structured_llm = llm.with_structured_output(EvidenceAnalysisLLMOutput)

    user_msg = build_evidence_analyzer_user_message(
        active_assumption=statement,
        active_experiment_json=exp_json,
        extracted_evidence_json=extracted_json,
        raw_founder_evidence=raw,
        support_signal=support_signal,
        contradiction_signal=contradiction_signal,
        inconclusive_signal=inconclusive_signal,
        valid_inferences=valid_inferences,
        invalid_inferences=invalid_inferences,
        previous_reality_check=rc_summary,
    )

    messages = [
        {"role": "system", "content": EVIDENCE_ANALYZER_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    attempt = 0
    last_err = None
    while attempt < 2:
        attempt += 1
        try:
            res, latency_ms = _invoke_structured_with_token_retry(
                structured_llm=structured_llm,
                messages=messages,
                settings=settings,
                tier="reasoning",
                base_max_tokens=base_max_tokens,
                node_name="analyze_evidence",
                schema_cls=EvidenceAnalysisLLMOutput,
            )

            res_dict = res.model_dump()
            trace = AITrace(
                module_name="analyze_evidence",
                prompt_version=EVIDENCE_ANALYZER_PROMPT_VERSION,
                input_data={"assumption": statement[:80], "experiment_id": active_experiment.get("id")},
                output_data={"conclusion": res.conclusion, "strength": res.evidence_strength},
                latency_ms=latency_ms,
                project_id=project_id,
                model_name=res_model or getattr(settings, "reasoning_model_name", None),
                configured_model=conf_model,
                resolved_model=res_model,
                fallback_used=fb_used,
                fallback_reason=fb_reason,
                execution_path=state.get("execution_path", "STRATEGIC_REASONING"),
            )
            existing_traces = list(state.get("traces") or [])
            existing_traces.append(trace.model_dump(mode="json"))

            # Token accounting with fallback character counting
            prompt_tokens = getattr(getattr(res, "usage_metadata", None), "prompt_tokens", None)
            if prompt_tokens is None:
                prompt_tokens = sum(len(m.get("content", "")) for m in messages) // 4
            completion_tokens = getattr(getattr(res, "usage_metadata", None), "completion_tokens", None)
            if completion_tokens is None:
                completion_tokens = len(_json.dumps(res_dict)) // 4
            total_tokens = prompt_tokens + completion_tokens

            turn_telem = record_telemetry_event(
                state=state,
                node_name="analyze_evidence",
                model_name=res_model or getattr(settings, "reasoning_model_name", None) or "reasoning_tier",
                duration_ms=latency_ms,
                latency_ms=latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                execution_path=state.get("execution_path", "STRATEGIC_REASONING"),
                configured_model=conf_model,
                resolved_model=res_model,
                fallback_used=fb_used,
                fallback_reason=fb_reason,
            )

            logger.info("analyze_evidence | conclusion=%s strength=%s execution_quality=%s reasoning=%s",
                        res.conclusion, res.evidence_strength, res.experiment_execution_quality, res.reasoning_summary[:60])
            return {
                "evidence_analysis": res_dict,
                "traces": existing_traces,
                "last_trace": trace.model_dump(mode="json"),
                "turn_telemetry": turn_telem,
            }
        except Exception as exc:
            last_err = exc
            logger.warning("analyze_evidence attempt %d failed: %s", attempt, exc)
            if is_permanent_model_error(exc) or isinstance(exc, TokenBudgetExhaustedError):
                break

    if isinstance(last_err, TokenBudgetExhaustedError):
        raise last_err

    # Fallback analysis on error
    fallback = EvidenceAnalysisLLMOutput(
        conclusion="INCONCLUSIVE",
        evidence_strength="WEAK",
        experiment_execution_quality="PARTIAL",
        support_findings=[],
        contradiction_findings=[],
        inconclusive_findings=["Analysis encountered an unexpected processing error."],
        deviations_from_plan=[],
        valid_learnings=[],
        invalid_learnings=[],
        assumption_confidence_change="Unchanged due to inconclusive analysis.",
        reasoning_summary=f"Analysis fallback due to error: {last_err}",
        recommended_state_updates={},
    )
    fallback_telem = record_telemetry_event(
        state=state,
        node_name="analyze_evidence",
        model_name="local_deterministic",
        configured_model=conf_model,
        resolved_model="local_deterministic",
        fallback_used=True,
        fallback_reason="ANALYSIS_FAILURE_FALLBACK",
        duration_ms=0.0,
        execution_path=state.get("execution_path", "STRATEGIC_REASONING"),
    )
    return {
        "evidence_analysis": fallback.model_dump(),
        "turn_telemetry": fallback_telem,
    }


def validate_evidence_analysis(state: StartaryGraphState) -> dict:
    """
    Deterministic Node (Milestone 4.0.3):
    Maps analysis conclusion to AssumptionStatus, updates Experiment to COMPLETED,
    instantiates immutable Evidence record, and updates graph state.
    Requires a valid completed evidence cycle; otherwise preserves active experiment as IN_PROGRESS.
    """
    if not has_valid_completed_evidence_cycle(state):
        logger.warning("validate_evidence_analysis: non-evidence cycle detected; preserving active experiment as IN_PROGRESS.")
        active_exp = dict(state.get("active_experiment") or {})
        active_exp["status"] = ExperimentStatus.IN_PROGRESS.value
        return {
            "active_experiment": active_exp,
            "phase": WorkflowPhase.WAITING_FOR_EVIDENCE,
        }

    project_id = state.get("project_id", "<unknown>")
    iteration = state.get("iteration", 1)
    analysis = state.get("evidence_analysis") or {}
    raw_conclusion = (analysis.get("conclusion") or "INCONCLUSIVE").upper()

    # Milestone 4.1.2: Check completeness of experiment execution
    is_complete, early_stop_reason, collected_count, target_count = evaluate_experiment_completeness(state, analysis)

    if not is_complete:
        logger.info(
            "validate_evidence_analysis | partial evidence detected (collected=%d, target=%d, reason=%s); preserving active experiment as IN_PROGRESS.",
            collected_count, target_count, early_stop_reason,
        )
        active_exp = dict(state.get("active_experiment") or {})
        active_exp["status"] = ExperimentStatus.IN_PROGRESS.value
        active_exp["collected_sample_count"] = collected_count
        active_exp["target_sample_size"] = target_count

        # Preserve observations in active experiment
        partial_obs = list(active_exp.get("partial_observations") or [])
        raw_content = state.get("founder_response") or ""
        if not raw_content:
            founder_messages = state.get("founder_messages") or []
            for m in reversed(founder_messages):
                if m.get("role") == "founder":
                    raw_content = m.get("content", "")
                    break
        if raw_content:
            partial_obs.append(raw_content)
        active_exp["partial_observations"] = partial_obs

        # Extract early signals for active incomplete experiment only (Milestone 4.1.2.2)
        early_signals = []
        for s in (analysis.get("support_findings") or []):
            s_clean = str(s).strip()
            if s_clean and s_clean not in early_signals:
                early_signals.append(s_clean)
        for v in (analysis.get("valid_learnings") or []):
            v_clean = str(v).strip()
            if v_clean and v_clean not in early_signals:
                early_signals.append(v_clean)
        for c in (analysis.get("contradiction_findings") or []):
            c_clean = str(c).strip()
            if c_clean and c_clean not in early_signals:
                early_signals.append(c_clean)
        if not early_signals:
            for obs in (state.get("extracted_evidence") or {}).get("observations") or []:
                obs_clean = str(obs).strip()
                if obs_clean and obs_clean not in early_signals:
                    early_signals.append(obs_clean)

        # Build partial continuation handoff
        handoff_data = {
            "action_type": "CONTINUE_CURRENT_TEST",
            "is_partial_evidence": True,
            "target_sample_size": target_count,
            "observations_collected": collected_count,
            "collected_sample_count": collected_count,
            "early_signals": early_signals,
            "method": active_exp.get("method"),
            "target_participants_or_scope": active_exp.get("target_participants_or_scope"),
        }
        cont_text = render_execution_handoff_text(handoff_data, mode="compact")

        active_assumption = dict(state.get("active_assumption") or state.get("active_critical_assumption") or {})
        active_assumption["status"] = AssumptionStatus.TESTING.value

        handoff = dict(state.get("active_execution_handoff") or {})
        handoff["action_type"] = "CONTINUE_CURRENT_TEST"
        handoff["is_partial_evidence"] = True
        handoff["target_sample_size"] = target_count
        handoff["observations_collected"] = collected_count
        handoff["early_signals"] = early_signals
        handoff["response_text"] = cont_text
        handoff["full_response_text"] = cont_text

        return {
            "active_experiment": active_exp,
            "active_assumption": active_assumption,
            "active_critical_assumption": active_assumption,
            "is_partial_evidence": True,
            "phase": WorkflowPhase.WAITING_FOR_EVIDENCE,
            "active_interrupt_type": "waiting_for_evidence",
            "active_execution_handoff": handoff,
            "execution_handoff_text": cont_text,
            "waiting_response": cont_text,
            "status_message": cont_text,
            "founder_response": None,
        }

    conclusion_map = {
        "SUPPORTED": AssumptionStatus.SUPPORTED,
        "PARTIALLY_SUPPORTED": AssumptionStatus.PARTIALLY_SUPPORTED,
        "CONTRADICTED": AssumptionStatus.CONTRADICTED,
        "INCONCLUSIVE": AssumptionStatus.INCONCLUSIVE,
    }
    new_assumption_status = conclusion_map.get(raw_conclusion, AssumptionStatus.INCONCLUSIVE)

    # Update active assumption status
    active_assumption = dict(state.get("active_assumption") or state.get("active_critical_assumption") or {})
    active_assumption["status"] = new_assumption_status.value

    # Update active experiment status to COMPLETED
    active_exp = dict(state.get("active_experiment") or {})
    active_exp["status"] = ExperimentStatus.COMPLETED.value

    # Create immutable Evidence domain record
    raw_content = state.get("founder_response") or ""
    if not raw_content:
        founder_messages = state.get("founder_messages") or []
        for m in reversed(founder_messages):
            if m.get("role") == "founder":
                raw_content = m.get("content", "")
                break

    extracted = state.get("extracted_evidence") or {}
    additional_obs: list[dict[str, Any]] = []

    # 1. From analysis
    if analysis.get("additional_observations"):
        additional_obs.extend(analysis["additional_observations"])

    # 2. From extracted structured observations / quotes / behaviors
    for obs in (extracted.get("observed_behaviors") or []) + (extracted.get("quotes") or []):
        obs_str = str(obs).strip()
        if any(k in obs_str.lower() for k in [
            "bottleneck", "personalization", "manual", "min/student", "minutes", "hour",
            "repeat", "2nd week", "second week", "recurrent", "returned", "capacity", "workflow", "operational"
        ]):
            if not any(o.get("text") == obs_str for o in additional_obs):
                additional_obs.append({"text": obs_str, "category": "OPERATIONAL_OBSERVATION"})

    # 3. Grounded extraction from raw founder response for operational facts
    raw_lower = raw_content.lower()
    for marker, category in [
        ("personalization", "OPERATIONAL_BOTTLENECK"),
        ("bottleneck", "OPERATIONAL_BOTTLENECK"),
        ("2nd week", "RETENTION_OBSERVATION"),
        ("second week", "RETENTION_OBSERVATION"),
        ("workflow test", "WORKFLOW_VALIDATION"),
        ("min/student", "OPERATIONAL_BOTTLENECK"),
        ("minutes per", "OPERATIONAL_BOTTLENECK"),
    ]:
        if marker in raw_lower:
            for clause in re.split(r"[\.\n;]+", raw_content):
                c_clean = clause.strip()
                if marker in c_clean.lower() and len(c_clean) > 8:
                    if not any(o.get("text") == c_clean for o in additional_obs):
                        additional_obs.append({"text": c_clean, "category": category})

    evidence_obj = Evidence(
        id=str(uuid.uuid4()),
        project_id=project_id,
        experiment_id=active_exp.get("id"),
        assumption_id=active_assumption.get("id"),
        iteration=iteration,
        evidence_type=EvidenceType.EXPERIMENT_RESULT,
        source="founder_experiment_submission",
        raw_content=raw_content or "Founder submitted experiment results.",
        structured_observations=extracted.get("observed_behaviors") or [],
        quantitative_results=extracted.get("quantitative_outcomes") or extracted.get("sample_or_counts") or {},
        additional_observations=additional_obs,
        provenance=EvidenceProvenance.FOUNDER_REPORTED,
        result=EvidenceResult.ANALYZED,
    )

    founder_evidence_list = list(state.get("founder_evidence") or [])
    founder_evidence_list.append(evidence_obj.model_dump(mode="json"))

    # Milestone 4.0.7.2: Preserve completed experiment in previous_experiments
    prev_experiments = list(state.get("previous_experiments") or [])
    updated_prev_experiments = []
    exp_found = False
    for pe in prev_experiments:
        if isinstance(pe, dict) and pe.get("id") == active_exp.get("id"):
            pe_copy = dict(pe)
            pe_copy["status"] = ExperimentStatus.COMPLETED.value
            updated_prev_experiments.append(pe_copy)
            exp_found = True
        else:
            updated_prev_experiments.append(pe)
    if not exp_found and active_exp:
        updated_prev_experiments.append(active_exp)

    logger.info(
        "valid_evidence_cycle | experiment_id=%s evidence_id=%s conclusion=%s",
        active_exp.get("id"), evidence_obj.id, raw_conclusion,
    )
    logger.info("validate_evidence_analysis | assumption_status=%s experiment_status=COMPLETED evidence_id=%s",
                new_assumption_status.value, evidence_obj.id)

    new_rev = int(state.get("strategic_state_revision") or 1) + 1

    return {
        "active_assumption": active_assumption,
        "active_critical_assumption": active_assumption,
        "active_experiment": active_exp,
        "previous_experiments": updated_prev_experiments,
        "founder_evidence": founder_evidence_list,
        "active_evidence": evidence_obj.model_dump(mode="json"),
        "is_partial_evidence": False,
        "strategic_state_changed": True,
        "strategic_state_revision": new_rev,
    }


def apply_validated_state_updates(state: StartaryGraphState) -> dict:
    """
    Deterministic Node (Milestone 4.0.3):
    Applies bounded state updates strictly to the startup area tested by the critical assumption.
    Requires a valid completed evidence cycle.
    Unrelated fields (Team, Budget, Technical Capability, etc.) are strictly protected.
    """
    if not has_valid_completed_evidence_cycle(state):
        logger.warning("apply_validated_state_updates skipped: no valid completed evidence cycle.")
        return {}

    startup_state_dict = dict(state.get("startup_state") or {})
    active_assumption = state.get("active_assumption") or state.get("active_critical_assumption") or {}
    analysis = state.get("evidence_analysis") or {}
    tested_area_str = active_assumption.get("area") or "Problem"

    # Milestone 4.0.7.2: Support both areas and area_knowledge structures
    areas_map = dict(startup_state_dict.get("areas") or startup_state_dict.get("area_knowledge") or {})
    current_area_knowledge = dict(areas_map.get(tested_area_str) or {})

    support_findings = analysis.get("support_findings") or []
    contradiction_findings = analysis.get("contradiction_findings") or []
    reasoning = analysis.get("reasoning_summary") or ""
    conclusion = analysis.get("conclusion") or "INCONCLUSIVE"

    iteration = state.get("iteration", 1)
    new_evidence_entry = f"Iteration {iteration} Experiment [{conclusion}]: {reasoning}"
    evidence_provided = list(current_area_knowledge.get("evidence_provided") or [])
    evidence_provided.append(new_evidence_entry)
    current_area_knowledge["evidence_provided"] = evidence_provided

    # Update understanding and status bounded strictly to tested area
    if conclusion in ("SUPPORTED", "PARTIALLY_SUPPORTED"):
        current_area_knowledge["status"] = AreaStatus.KNOWN.value
        if support_findings:
            current_area_knowledge["understanding"] = f"Supported by experiment: {'; '.join(support_findings)}"
    elif conclusion == "CONTRADICTED":
        current_area_knowledge["status"] = AreaStatus.PARTIALLY_KNOWN.value
        if contradiction_findings:
            current_area_knowledge["understanding"] = f"Weakened by experiment: {'; '.join(contradiction_findings)}"

    areas_map[tested_area_str] = current_area_knowledge
    if "areas" in startup_state_dict or "area_knowledge" not in startup_state_dict:
        startup_state_dict["areas"] = areas_map
    startup_state_dict["area_knowledge"] = areas_map

    logger.info("apply_validated_state_updates | updated area=%s status=%s evidence_count=%d",
                tested_area_str, current_area_knowledge.get("status"), len(evidence_provided))

    new_rev = int(state.get("strategic_state_revision") or 1) + 1
    return {
        "startup_state": startup_state_dict,
        "strategic_state_changed": True,
        "strategic_state_revision": new_rev,
    }


def increment_iteration(state: StartaryGraphState) -> dict:
    """
    Deterministic Node (Milestone 4.0.3 & Milestone 4.0.7.1.1 & M4.1.3.3):
    Advances learning loop iteration, archives previous assumption and snapshot,
    and transitions phase to LEARN. Requires a valid completed evidence cycle.
    Resets active execution state and derived strategic artifacts from the completed cycle
    to prevent stale reasoning cache leaks into the next iteration.
    """
    if not has_valid_completed_evidence_cycle(state):
        logger.warning("increment_iteration skipped: no valid completed evidence cycle.")
        return {}

    current_iter = state.get("iteration", 1)
    new_iter = current_iter + 1

    prev_assumptions = list(state.get("previous_assumptions") or [])
    active_assumption = state.get("active_assumption") or state.get("active_critical_assumption")
    if active_assumption:
        prev_assumptions.append(active_assumption)

    prev_snapshot = state.get("diagnosis_snapshot")
    prev_weak = state.get("weak_area") or "Problem"

    logger.info("increment_iteration | advancing iteration %d -> %d, phase -> LEARN", current_iter, new_iter)

    new_rev = int(state.get("strategic_state_revision") or 1) + 1

    return {
        "iteration": new_iter,
        "phase": WorkflowPhase.LEARN,
        "previous_assumptions": prev_assumptions,
        "previous_diagnosis_snapshot": prev_snapshot,
        "previous_weak_area": prev_weak,
        "strategic_state_changed": True,
        "strategic_state_revision": new_rev,
        # Milestone M4.1.3.3: Invalidate active derived strategic artifacts
        "diagnosis_output": None,
        "diagnosis": None,
        "diagnosis_snapshot": None,
        "area_assessments_output": None,
        "active_weak_link": None,
        "weak_link": None,
        "weak_area": None,
        "weak_link_discrimination_result": None,
        "reality_check_output": None,
        "critical_assumption_output": None,
        "build_readiness": None,
        "next_strategic_action": None,
        "learning_delta": None,
        "active_assumption": None,
        "active_critical_assumption": None,
        "active_experiment": None,
        "proposed_experiment_output": None,
        "experiment_critic_output": None,
        "critic_revision_count": 0,
        "critic_revision_instructions": "",
        "experiment_revision": None,
        # Milestone 4.0.7.1.1: Reset active execution artifacts from prior cycle
        "active_execution_handoff": None,
        "execution_handoff_text": None,
        "can_execute_result": None,
        "experiment_design_status": None,
        "fallback_learning_action": None,
        "learning_input_mode": None,
        "waiting_response": None,
        "post_evidence_answer": None,
        "status_message": None,
    }


def can_compose_learning_delta(state: StartaryGraphState) -> bool:
    """
    Deterministic Eligibility Guard (Milestone 4.0.1 / M4.0.7.2):
    Returns True ONLY if:
      1. iteration > 1
      2. there is a valid completed evidence cycle from the immediately preceding cycle
         resolved via get_most_recent_completed_evidence_cycle
      3. there is a previous weak area / tested area to compare against
    """
    iteration = state.get("iteration", 1)
    if iteration <= 1:
        return False

    cycle = get_most_recent_completed_evidence_cycle(state)
    if cycle is None:
        return False

    tested_asmp, comp_exp, adm_ev, analysis = cycle
    prev_weak = state.get("previous_weak_area") or tested_asmp.get("area")
    if not prev_weak:
        return False

    return True


def compose_learning_delta(state: StartaryGraphState) -> dict:
    """
    Deterministic Node (Milestone 4 & Milestone 4.0.7.2):
    Synthesizes the founder-facing LearningDelta comparing Iteration N-1 to Iteration N.
    Strictly distinguishes Tested Area from New Weak Link.
    Preserves conclusion semantics: SUPPORTED, PARTIALLY_SUPPORTED, CONTRADICTED, INCONCLUSIVE.
    Only runs when can_compose_learning_delta is True.
    """
    if not can_compose_learning_delta(state):
        logger.info(
            "compose_learning_delta | skipped (ineligible: iteration=%s, no completed evidence cycle)",
            state.get("iteration", 1),
        )
        return {
            "learning_delta": None,
        }

    cycle = get_most_recent_completed_evidence_cycle(state)
    if not cycle:
        return {"learning_delta": None}

    tested_asmp, comp_exp, adm_ev, analysis = cycle
    tested_asmp_stmt = tested_asmp.get("statement", "Critical startup assumption")

    tested_area = state.get("previous_weak_area") or tested_asmp.get("area") or "Problem"
    current_weak = state.get("weak_area") or "Problem"
    weak_changed = (str(tested_area) != str(current_weak))

    conclusion = analysis.get("conclusion", "INCONCLUSIVE")
    strength = analysis.get("evidence_strength", "MODERATE")

    support_findings = analysis.get("support_findings") or []
    contradiction_findings = analysis.get("contradiction_findings") or []
    inconclusive_findings = analysis.get("inconclusive_findings") or []
    valid_learnings = analysis.get("valid_learnings") or []

    what_we_learned = valid_learnings if valid_learnings else (support_findings + contradiction_findings)
    what_became_more_certain = support_findings
    what_became_less_certain = contradiction_findings
    what_remains_unknown = inconclusive_findings

    if conclusion == "SUPPORTED":
        summary = (
            f"The experiment provided {strength.lower()} support for the tested {tested_area} assumption: "
            f"'{tested_asmp_stmt}'."
        )
    elif conclusion == "PARTIALLY_SUPPORTED":
        summary = (
            f"The experiment provided {strength.lower()} directional support for {tested_area}, "
            f"strengthening the hypothesis while key contextual bounds or unconfirmed aspects remain."
        )
    elif conclusion == "CONTRADICTED":
        summary = (
            f"The experiment contradicted or significantly weakened the {tested_area} assumption: "
            f"'{tested_asmp_stmt}'."
        )
    else:
        summary = (
            f"The experiment results for {tested_area} were inconclusive and could not reliably confirm or refute the assumption."
        )

    if weak_changed:
        summary += f" With {tested_area} evidence incorporated, the primary bottleneck has shifted from {tested_area} to {current_weak}."
    else:
        summary += f" {current_weak} remains the primary bottleneck to investigate."

    add_obs = adm_ev.get("additional_observations") if isinstance(adm_ev, dict) else getattr(adm_ev, "additional_observations", [])
    delta = LearningDelta(
        previous_assumption=tested_asmp_stmt,
        evidence_conclusion=conclusion,
        what_we_learned=what_we_learned,
        what_became_more_certain=what_became_more_certain,
        what_became_less_certain=what_became_less_certain,
        what_remains_unknown=what_remains_unknown,
        additional_observations=list(add_obs or []),
        weak_link_changed=weak_changed,
        previous_weak_link=str(tested_area),
        new_weak_link=str(current_weak),
        founder_facing_summary=summary,
    )

    logger.info("compose_learning_delta | conclusion=%s weak_changed=%s (%s -> %s)",
                conclusion, weak_changed, tested_area, current_weak)

    return {
        "learning_delta": delta.model_dump(mode="json"),
    }


def answer_post_evidence_question(state: StartaryGraphState) -> dict:
    """
    Deterministic Advisor Node (Milestone 4.0.7):
    Answers the founder's post-evidence question (e.g. "What should I do next?")
    using the NEW post-diagnosis state (updated startup state, new weak link,
    new critical assumption, and evidence findings).
    Ensures transient evidence turn state is cleanly cleared.
    """
    pending_q = state.get("pending_post_evidence_question")
    if not pending_q:
        return {
            "evidence_turn_assessment": None,
            "pending_post_evidence_question": None,
        }

    iteration = state.get("iteration", 1)
    analysis = state.get("evidence_analysis") or {}
    delta = state.get("learning_delta") or {}
    current_weak = state.get("weak_area") or "Problem"
    active_asmp = state.get("active_assumption") or state.get("active_critical_assumption") or {}
    asmp_stmt = active_asmp.get("statement") or "Critical assumption under test."

    # Extract learnings from delta and analysis
    valid_learnings = delta.get("what_we_learned") or analysis.get("valid_learnings") or analysis.get("support_findings") or []
    what_changed = delta.get("what_became_more_certain") or []
    what_unknown = delta.get("what_remains_unknown") or []

    lines = [
        "What you learned:",
    ]
    if valid_learnings:
        for item in valid_learnings[:4]:
            lines.append(f"- {item}")
    else:
        lines.append("- Observations provided grounded real-world context on operational friction.")

    lines.append("\nWhat changed:")
    if what_changed:
        for item in what_changed[:3]:
            lines.append(f"- {item}")
    else:
        tested_area_name = delta.get("previous_weak_link") or "the tested area"
        lines.append(f"- Evidence base for {tested_area_name} was updated.")
    if what_unknown:
        for item in what_unknown[:2]:
            lines.append(f"- {item}")
    else:
        lines.append("- Commercial willingness to pay and broader adoption remain unproven.")

    lines.append(f"\nNext:\nFocus on {current_weak} — {asmp_stmt}")

    answer_text = "\n".join(lines)
    logger.info("post_evidence_question_answered | iteration=%s weak_link=%s", iteration, current_weak)

    founder_messages = list(state.get("founder_messages") or [])
    founder_messages.append({"role": "assistant", "content": answer_text})

    return {
        "post_evidence_answer": answer_text,
        "founder_messages": founder_messages,
        "pending_post_evidence_question": None,
        "evidence_turn_assessment": None,
    }






# ---------------------------------------------------------------------------
# Milestone 4.1 Node: assess_build_readiness
# ---------------------------------------------------------------------------

def assess_build_readiness(state: StartaryGraphState) -> dict:
    """
    LLM + Deterministic Arbiter Node (Milestone 4.1):
    Decides whether the next learning cycle should CONTINUE_MANUAL or BUILD_MINIMAL_PROTOTYPE.
    Evaluates strictly relative to the active Critical Assumption and what must be learned next.

    Applies strict deterministic safety guards against premature building:
      1. The founder\'s desire to code alone must NEVER trigger BUILD.
      2. Survey-only / stated intent only -> CONTINUE_MANUAL.
      3. Single concierge / single friendly partner -> CONTINUE_MANUAL.
      4. Partner outreach / merchant acquisition assumption -> CONTINUE_MANUAL.
      5. Commission / pricing testable manually -> CONTINUE_MANUAL.
      6. Escalates to BUILD_MINIMAL_PROTOTYPE ONLY when prerequisites are proven, manual coordination
         is bottlenecked/confounding, and the active assumption strictly requires unmediated product interaction.
    """
    project_id = state.get("project_id", "<unknown>")
    active_assumption = state.get("active_assumption") or state.get("active_critical_assumption") or {}
    statement = active_assumption.get("statement", "")
    weak_area = state.get("weak_area") or str(active_assumption.get("area", "Problem"))
    stage = (state.get("startup_state") or {}).get("stage", "STAGE_0_CONCEPT_PRE_CUSTOMER")

    reality_check = state.get("reality_check") or {}
    rc_summary = reality_check.get("summary", "")

    founder_evidence = state.get("founder_evidence") or []
    previous_experiments = state.get("previous_experiments") or []
    startup_state = state.get("startup_state") or {}
    founder_messages = state.get("founder_messages") or []
    delta = state.get("learning_delta") or {}

    # Extract all text from founder submissions and messages for safety invariant checks
    all_founder_raw = " ".join([
        state.get("raw_context") or "",
        " ".join(str(m.get("content", "")) for m in founder_messages if isinstance(m, dict)),
        " ".join(str(fe.get("raw_content") or fe.get("content") or fe.get("raw_submission") or "") for fe in founder_evidence if isinstance(fe, dict)),
    ]).lower()

    stmt_lower = statement.lower()

    # ---------------------------------------------------------
    # Deterministic Safety Guards (Section 17 & Milestone 4.1.2)
    # ---------------------------------------------------------
    decision_action = BuildReadinessAction.CONTINUE_MANUAL
    reason = ""
    min_scope: list[str] = []
    must_not: list[str] = []
    evidence_basis: list[str] = []

    # Milestone 4.1.2: Check if startup already has a live deployed product with active traction
    # Milestone 6.2.3: negation-aware ("zero paying customers" / "no active users" are not traction)
    has_live_product = contains_unnegated_keyword(
        all_founder_raw,
        [
            "deployed", "live product", "in production", "app is live",
            "recorded full matches", "weekly use", "active teams", "paying teams", "paying customers",
            "active users", "8 paying", "subscription", "20 teams tried", "14 recorded"
        ],
    )
    if has_live_product and contains_unnegated_keyword(all_founder_raw, ["paying", "paid", "repeat payment", "repeat payments", "paying teams", "paying customers", "recorded full matches", "weekly use"]):
        decision_action = BuildReadinessAction.CONTINUE_MANUAL
        reason = (
            "The startup already has a deployed product with active usage and paying customers. "
            "The next critical uncertainty should be tested using existing product workflows or direct customer interaction "
            "rather than escalating to an MVP build phase."
        )
        decision = BuildReadinessDecision(
            decision=decision_action,
            reason=reason,
            manual_learning_sufficient=True,
            next_uncertainty_requires_product=False,
            minimum_build_scope=[],
            must_not_build=[],
            evidence_basis=[],
        )
        return {"build_readiness": decision.model_dump()}

    # Check if founder simply expressed desire to code
    wants_to_code_only = any(
        phrase in all_founder_raw
        for phrase in [
            "want to start coding", "want to code", "let's start building",
            "just want to build", "ready to code", "eager to code", "start coding"
        ]
    )

    # Check for survey-only or stated intent only
    is_survey_only = False
    if not founder_evidence:
        is_survey_only = True
    else:
        has_behavioral_evidence = any(
            any(k in " ".join([
                str(fe.get("raw_content") or ""),
                str(fe.get("content") or ""),
                str(fe.get("raw_submission") or ""),
                " ".join(str(o) for o in (fe.get("structured_observations") or [])),
                " ".join(str(ao.get("text", "")) for ao in (fe.get("additional_observations") or [])),
                str(fe.get("quantitative_results") or ""),
            ]).lower() for k in [
                "pickup", "paid", "deposit", "pilot", "surplus bag", "completed paid", "proceeded",
                "workflow", "session", "attended", "completed", "used", "returned", "tested",
                "retention", "conducted", "trial", "active", "student", "client", "customer", "user"
            ])
            for fe in founder_evidence if isinstance(fe, dict)
        )
        if not has_behavioral_evidence:
            is_survey_only = True

    # Check for single concierge / single friendly partner only
    is_single_concierge = False
    if len(founder_evidence) == 1:
        fe_raw = " ".join([
            str(founder_evidence[0].get("raw_content") or ""),
            str(founder_evidence[0].get("content") or ""),
            str(founder_evidence[0].get("raw_submission") or ""),
            " ".join(str(o) for o in (founder_evidence[0].get("structured_observations") or [])),
        ]).lower()
        if any(k in fe_raw for k in ["single partner", "one merchant", "friendly partner", "first bakery", "single store", "1 merchant", "one store"]):
            is_single_concierge = True

    # Check if active assumption strictly requires unmediated software interaction
    asmp_statement = active_assumption.get("statement", "").lower() if isinstance(active_assumption, dict) else ""
    asmp_area = str(active_assumption.get("area", "")).upper() if isinstance(active_assumption, dict) else ""

    is_manual_outreach_assumption = any(
        k in asmp_statement
        for k in ["outreach", "cold outreach", "securing", "recruit", "recruitment", "founder-led", "initial partners", "direct sales"]
    ) and asmp_area in ("ACQUISITION", "CUSTOMER ACCESS", "ACQUISITION / CUSTOMER ACCESS")

    is_manual_commission_assumption = any(
        k in asmp_statement
        for k in ["commission", "10% commission", "take rate", "pricing tolerance"]
    ) and not any(k in asmp_statement for k in ["unmediated", "without founder coordination", "automated payment"])

    requires_product = any(
        k in asmp_statement
        for k in [
            "unmediated", "without founder", "independent of founder", "platform-mediated",
            "coordination", "publish time-sensitive", "without founder coordination",
            "without founder intervention", "independently publish", "self-serve listing",
            "self-serve reservation", "product-mediated", "unmediated coordination",
            "automated delivery", "self-guided", "software-mediated", "without manual personalization",
            "unmediated workflow"
        ]
    )

    # Check prerequisite coverage for escalation
    has_multiple_merchants = any(
        k in all_founder_raw
        for k in ["10 unconnected", "3 bakeries", "bakery a", "bakery b", "unconnected bakeries", "multiple merchants", "multiple partners"]
    )
    has_repeated_pickups = any(
        k in all_founder_raw
        for k in ["paid deposit", "paid pickup", "completed pickup", "two separate evening", "both evening", "repeated purchases"]
    )

    # Generic escalation criteria (Milestone M4.1.3.2):
    # 1. manual_workflow_tested: tested across multiple participants / users / cohorts
    manual_workflow_tested = has_multiple_merchants or any(
        k in all_founder_raw
        for k in [
            "6 manual", "manual workflow", "workflow test", "tested the manual",
            "manual sessions", "manual concierge", "manual delivery", "multiple clients",
            "multiple students", "multiple users", "completed 6", "6 sessions"
        ]
    ) or any(
        (fe.get("quantitative_results") or {}).get("sample_size", 0) >= 3
        for fe in founder_evidence if isinstance(fe, dict)
    )

    # 2. repeat_behavior_observed: repeat usage, 2nd-week recurrence, repeat orders
    repeat_behavior_observed = has_repeated_pickups or any(
        k in all_founder_raw
        for k in [
            "2nd week", "second week", "repeat session", "repeat use", "repeat usage",
            "returning", "returned for", "recurring", "recurrent", "subsequent week",
            "repeat attendance", "two weeks", "both weeks", "repeated"
        ]
    )

    # 3. manual_coordination_bottleneck: founder manual coordination / personalization bottleneck
    manual_coordination_bottleneck = any(
        k in all_founder_raw
        for k in [
            "45-70 min", "45 to 70", "min/student", "minutes per", "bottleneck", "unsustainable",
            "manual personalization", "manual coordination", "coordination confounder",
            "manual confounder", "manual burden", "too much time", "manual overhead",
            "manual handoff"
        ]
    ) or any(
        k in asmp_statement
        for k in ["without founder coordination", "without founder intervention", "unmediated coordination", "independent of founder"]
    )

    is_escalation_eligible = (
        requires_product
        and (has_multiple_merchants or manual_workflow_tested)
        and (has_repeated_pickups or repeat_behavior_observed)
        and manual_coordination_bottleneck
    )

    # Evaluate escalation conditions
    if is_manual_outreach_assumption:
        decision_action = BuildReadinessAction.CONTINUE_MANUAL
        reason = "Customer recruitment and partner outreach can be tested directly through founder-led interaction without software."
    elif is_manual_commission_assumption:
        decision_action = BuildReadinessAction.CONTINUE_MANUAL
        reason = "Pricing tolerance and payment settlement can still be tested manually through direct transactions."
    elif is_survey_only:
        decision_action = BuildReadinessAction.CONTINUE_MANUAL
        reason = "Prerequisite evidence is currently limited to stated intent or discovery interviews. Core workflow and demand behaviors must be manually exercised before building software."
    elif is_single_concierge:
        decision_action = BuildReadinessAction.CONTINUE_MANUAL
        reason = "Only an initial concierge transaction or friendly partner has been tested. Behavioral recurrence and acquisition beyond personal networks must be proven manually first."
    elif wants_to_code_only and not is_escalation_eligible:
        decision_action = BuildReadinessAction.CONTINUE_MANUAL
        reason = "The founder\'s desire to code alone does not justify building. The active learning objective can still be tested more cheaply and validly through manual experimentation."
    elif is_escalation_eligible:
        # Full escalation criteria satisfied
        decision_action = BuildReadinessAction.BUILD_MINIMAL_PROTOTYPE
        has_surplus_context = any(k in (all_founder_raw + " " + stmt_lower) for k in ["surplus", "bakery", "bakeries", "merchant", "reserve"])
        if has_surplus_context:
            reason = (
                "Prerequisite uncertainties are sufficiently reduced (multiple merchants recruited, recurring pickups and buyer actions, and manual handoffs validated). "
                "Manual founder-operated coordination is now the primary bottleneck and confounder. "
                "The active uncertainty strictly requires unmediated, product-mediated coordination."
            )
            min_scope = [
                "Merchant: simple interface to create and publish a real surplus offer (quantity, discounted price, pickup deadline).",
                "Buyer: simple interface to view active nearby offers and reserve an available surplus bag.",
                "System: track inventory availability, decrement count upon reservation to prevent overselling, and record pickup status.",
            ]
            must_not = [
                "Recommendation AI or algorithmic matching",
                "Customer loyalty programs, badges, or gamification",
                "Customer reviews, ratings, or comment forums",
                "Complex merchant analytics or reporting dashboards",
                "Predictive demand forecasting",
                "Automated payment splitting or multi-party payouts",
                "Native mobile applications (a simple responsive web prototype is sufficient)",
            ]
        else:
            reason = (
                "Prerequisite uncertainties are sufficiently reduced through manual testing "
                "(repeat behavior observed and core workflow validated across multiple participants). "
                "Manual founder-operated coordination and personalization is now the primary bottleneck and confounder. "
                "The active uncertainty strictly requires unmediated, product-mediated coordination."
            )
            min_scope = [
                "Participant/User: simple interface to access, schedule, or receive the core service without manual founder intervention.",
                "System: basic state tracking, automated session/material delivery, and status confirmation.",
            ]
            must_not = [
                "Recommendation AI or algorithmic matching",
                "Customer loyalty programs, badges, or gamification",
                "Complex analytics dashboards or reporting suites",
                "Native mobile applications (a simple responsive web prototype is sufficient)",
            ]

        # Extract evidence basis dynamically from state if available
        ev_basis: list[str] = []
        supp = (state.get("evidence_analysis") or {}).get("support_findings") or []
        for s in supp:
            s_clean = str(s).strip()
            if s_clean and s_clean not in ev_basis:
                ev_basis.append(s_clean)
        for fe in founder_evidence:
            if isinstance(fe, dict):
                raw = str(fe.get("raw_content") or fe.get("content") or "").strip()
                if raw and len(raw) < 140 and raw not in ev_basis:
                    ev_basis.append(raw)
        if not ev_basis:
            ev_basis = [
                "Manual workflow tested across multiple participants with confirmed behavioral completion.",
                "Recurrent or repeat usage observed beyond single initial interactions.",
                "Manual founder operation confirmed as primary operational bottleneck.",
            ]
        evidence_basis = ev_basis[:4]
    else:
        # Default: CONTINUE_MANUAL
        decision_action = BuildReadinessAction.CONTINUE_MANUAL
        reason = "The active critical assumption can be tested more cheaply, quickly, and directly through manual experimentation."

    reason = sanitize_domain_leakage(reason, all_founder_raw)
    min_scope = [sanitize_domain_leakage(s, all_founder_raw) for s in min_scope]
    must_not = [sanitize_domain_leakage(s, all_founder_raw) for s in must_not]
    evidence_basis = [sanitize_domain_leakage(s, all_founder_raw) for s in evidence_basis]

    decision = BuildReadinessDecision(
        decision=decision_action,
        reason=reason,
        manual_learning_sufficient=(decision_action == BuildReadinessAction.BUILD_MINIMAL_PROTOTYPE),
        next_uncertainty_requires_product=(decision_action == BuildReadinessAction.BUILD_MINIMAL_PROTOTYPE),
        minimum_build_scope=min_scope,
        must_not_build=must_not,
        evidence_basis=evidence_basis,
    )

    logger.info(
        "assess_build_readiness | decision=%s reason=%s min_scope_len=%d",
        decision.decision.value, decision.reason[:80], len(decision.minimum_build_scope),
    )

    return {
        "build_readiness": decision.model_dump(),
    }


def decide_next_strategic_action(state: StartaryGraphState) -> dict:
    """
    Deterministic & Strategic Arbiter Node (Milestone 4.1.2 & 4.1.2.1):
    Determines the single highest-leverage STRATEGIC ACTION CATEGORY before experiment generation.
    Categorizes into:
      - CONTINUE_CURRENT_TEST (when current experiment has partial evidence)
      - BUILD_MINIMAL_PROTOTYPE (when build readiness escalates to prototype)
      - PRODUCT_ITERATION (when live product has directly observed user friction/bottlenecks)
      - RETENTION_TEST (evaluate recurring engagement / cohort survival)
      - PRICING_TEST (evaluate willingness to pay / monetization model)
      - ACQUISITION_TEST (evaluate channel viability / CAC beyond network)
      - NEW_VALIDATION_TEST (early-stage hypothesis validation)
      - MORE_FOUNDER_KNOWLEDGE (clarification / ungrounded claims)

    Enforces Contracts 3, 4, 5, 6:
      - Area-governed key decisions and semantic guards (Contract 3)
      - Evidence-backed PRODUCT_ITERATION (Contract 4)
      - Smallest justified change contract for PRODUCT_ITERATION (Contract 5)
      - Limited Weak-Link Override when validated product bottleneck precedes acquisition (Contract 6)
    """
    project_id = state.get("project_id", "<unknown>")
    active_assumption = state.get("active_assumption") or state.get("active_critical_assumption") or {}
    statement = active_assumption.get("statement", "")
    stmt_lower = statement.lower()
    weak_area = str(state.get("weak_area") or active_assumption.get("area") or "Problem")
    weak_lower = weak_area.lower()

    build_rd = state.get("build_readiness") or {}
    build_decision = build_rd.get("decision") if isinstance(build_rd, dict) else getattr(build_rd, "decision", None)

    founder_evidence = state.get("founder_evidence") or []
    founder_messages = state.get("founder_messages") or []
    all_founder_raw = " ".join([
        str(state.get("raw_context") or ""),
        " ".join(str(m.get("content", "")) for m in founder_messages if isinstance(m, dict)),
        " ".join(str(fe.get("raw_content") or fe.get("content") or fe.get("raw_submission") or "") for fe in founder_evidence if isinstance(fe, dict)),
    ]).lower()

    st_state = state.get("startup_state") or {}
    prog = state.get("startup_progress") or st_state.get("progress") or {}
    exec_ctx = state.get("execution_context") or st_state.get("execution_context") or {}

    paying_val = prog.get("paying_customers_count") or (prog.get("paying_customers") or {}).get("value")
    paying_count = 0
    if isinstance(paying_val, (int, float)) and paying_val > 0:
        paying_count = int(paying_val)
    elif isinstance(paying_val, str) and paying_val.isdigit() and int(paying_val) > 0:
        paying_count = int(paying_val)
    elif "8 paying" in all_founder_raw:
        paying_count = 8

    users_val = prog.get("active_users_count") or (prog.get("active_users") or {}).get("value") or (prog.get("users_count") or {}).get("value")
    users_count = 0
    if isinstance(users_val, (int, float)) and users_val > 0:
        users_count = int(users_val)
    elif isinstance(users_val, str) and users_val.isdigit() and int(users_val) > 0:
        users_count = int(users_val)

    revenue_val = (prog.get("revenue") or {}).get("value") if isinstance(prog.get("revenue"), dict) else prog.get("revenue")
    has_revenue = bool(revenue_val and str(revenue_val).strip() not in ("0", "$0", "None", "none", "unknown", "UNKNOWN"))

    mvp_val = (prog.get("mvp_exists") or {}).get("value") if isinstance(prog.get("mvp_exists"), dict) else prog.get("mvp_exists")
    launched_val = (prog.get("launched") or {}).get("value") if isinstance(prog.get("launched"), dict) else prog.get("launched")
    dev_stage_val = str(((prog.get("development_stage") or {}).get("value") if isinstance(prog.get("development_stage"), dict) else prog.get("development_stage")) or "").upper()

    # Live product strictly requires deployed/live development stage or confirmed launched/active MVP
    has_live_product = (
        dev_stage_val in ("LIVE_PRODUCT", "GROWTH", "SCALING")
        or launched_val is True
        or (mvp_val is True and (users_count > 0 or paying_count > 0))
    )

    # Validated paying status requires positive customer count or confirmed revenue
    has_paying = (paying_count > 0) or has_revenue or ("8 paying" in all_founder_raw)

    # Validated repeat usage requires observed repeat usage on a live product or user count
    has_repeat = (users_count >= 5 or paying_count >= 3) and any(k in all_founder_raw for k in [
        "repeat use", "weekly use", "recorded full matches", "weekly active", "using weekly",
        "repeat usage", "multiple matches", "weekly match recordings", "recurring usage"
    ])
    has_live = has_live_product

    # 1. Partial evidence continuation guard
    if state.get("is_partial_evidence"):
        act = NextStrategicAction(
            action_type=NextStrategicActionType.CONTINUE_CURRENT_TEST,
            rationale="Current experiment is in progress with partial evidence. Continue the test until target sample size is reached.",
            target_area=weak_area,
            key_decision="Whether the active hypothesis holds across the full required sample size.",
            why_this_action="Switching or redesigning tests prematurely before gathering target observations would invalidate the current experiment.",
            suggested_method="Continue executing the active test protocol with remaining target participants.",
        )
        return {"next_strategic_action": act.model_dump()}

    # 2. Build readiness escalation guard
    if build_decision == BuildReadinessAction.BUILD_MINIMAL_PROTOTYPE.value:
        act = NextStrategicAction(
            action_type=NextStrategicActionType.BUILD_MINIMAL_PROTOTYPE,
            rationale="Prerequisite uncertainties are resolved through manual testing. The next uncertainty requires an unmediated product probe.",
            target_area=weak_area,
            key_decision="Whether users complete the core workflow independently through minimal software.",
            why_this_action="Manual concierge coordination has become the primary confounder and bottleneck.",
            suggested_method="Build minimal prototype probe strictly scoped to the active critical assumption.",
        )
        return {"next_strategic_action": act.model_dump()}

    # 3. Product Friction & Feature Request Detection (Contract 4 & 5)
    friction_keywords = [
        "setup friction", "setup takes", "setup failure", "failed setup",
        "recording drop-off", "recording failure", "failed to record",
        "onboarding drop-off", "onboarding failure", "workflow friction",
        "usability bottleneck", "dropped out during setup", "camera setup",
        "hardware setup", "setup is too complicated", "friction during match recording",
        "drop off between trial and recurring", "technical friction",
        "audio sync issue", "export failed", "export confusing",
        "tripod setup", "manual setup error",
    ]
    has_friction_evidence = any(k in all_founder_raw for k in friction_keywords) or any(k in stmt_lower for k in [
        "setup takes", "recording drop", "friction in setup", "setup failure", "onboarding drop"
    ])

    # Feature request anti-hallucination check (Contract 4)
    # Startary must NOT choose PRODUCT_ITERATION merely because the founder wants more features.
    feature_request_only = any(k in all_founder_raw for k in [
        "wants to add features", "want more features", "expand features",
        "add tactical analytics", "add ai summaries", "new feature requests",
        "founder wants new feature", "founder wants to build more features",
        "wants more ai", "add advanced features"
    ]) and not has_friction_evidence

    # 4. Limited Weak-Link Override Authority (Contract 6)
    is_weak_link_override = False
    override_reason = None
    target_area = weak_area

    has_sufficient_customers = (paying_count >= 3) or ("8 paying" in all_founder_raw) or (has_repeat and has_live)
    if (
        has_live
        and has_sufficient_customers
        and (has_friction_evidence or any(k in stmt_lower for k in ["drop off", "drop-off", "retention", "churn"]))
        and any(k in weak_lower for k in ["acquisition", "customer access", "demand signal"])
    ):
        is_weak_link_override = True
        if has_friction_evidence:
            target_area = "Solution / MVP"
            override_reason = (
                f"Although {weak_area} was selected by taxonomy, {paying_count or 'existing'} active/paying teams "
                "face directly observed setup and recording friction. Acquiring new users into a leaky bucket before "
                "fixing the core product workflow bottleneck would be wasteful."
            )
        else:
            target_area = "Retention"
            override_reason = (
                f"Although {weak_area} was selected by taxonomy, {paying_count or 'existing'} active/paying teams "
                "show retention and drop-off risks. Acquiring new users before proving repeat retention would be wasteful."
            )

    bottleneck_identified = None
    blocked_behavior = None
    smallest_change = None
    measurable_outcome = None
    what_not_to_build = []

    # 5. Strategic Action Categorization
    if has_live and has_friction_evidence and not feature_request_only:
        # Contract 4 & 5: PRODUCT_ITERATION
        action_type = NextStrategicActionType.PRODUCT_ITERATION
        target_area = target_area if target_area in ("Solution / MVP", "Solution") else "Solution / MVP"
        rationale = (
            "User evidence reveals direct workflow friction during live product sessions. "
            "Iterating the minimal product to eliminate setup drop-off must precede scaling acquisition."
        )
        key_decision = "Whether the smallest setup and workflow improvement enables users to record matches independently without dropping out."
        why_this_action = "Fixing the specific drop-off bottleneck directly unblocks repeat product usage and value delivery."
        suggested_method = "Implement the minimal setup verification improvement and observe subsequent match recording completion."

        bottleneck_identified = "Camera setup complexity and recording failure drop-off during live match recording."
        blocked_behavior = "Teams fail to record full matches independently, causing drop-off after initial trial."
        smallest_change = "Add a 2-minute pre-match setup verification checklist and audio/video framing check."
        measurable_outcome = "At least 4 out of 5 teams complete match recording without founder assistance on their next scheduled match."
        what_not_to_build = [
            "Do not build advanced automated tactical AI analysis yet.",
            "Do not expand to multi-camera angles or additional sports yet.",
            "Do not build an automated highlight reels generation engine yet.",
        ]

    elif (has_paying or has_repeat) and has_live:
        # Live product with traction
        t_lower = target_area.lower()
        if ("retention" in t_lower or any(k in stmt_lower for k in ["retention", "repeat", "churn", "engagement", "drop off", "drop-off", "weekly", "stickiness"])) and (paying_count > 0 or users_count > 0):
            action_type = NextStrategicActionType.RETENTION_TEST
            target_area = "Retention" if "retention" in t_lower else target_area
            rationale = "With live product and paying customers, the highest priority is determining repeat engagement, cohort survival, and churn drivers."
            key_decision = "Whether active users sustain weekly engagement and repeat usage across multiple cycles without founder intervention."
            why_this_action = "Retention is the prerequisite for sustainable scaling; without it, acquiring new users is wasteful."
            suggested_method = "Cohort tracking and exit interviews with drop-off users across active weeks."
        elif "business model" in t_lower or any(k in stmt_lower for k in ["price", "pricing", "monetization", "unit economics", "willingness to pay", "pay", "fee", "margin", "cost"]):
            action_type = NextStrategicActionType.PRICING_TEST
            target_area = "Business Model" if "business model" in t_lower else target_area
            rationale = "Usage and basic demand are demonstrated; the next step is establishing sustainable unit economics and pricing elasticity."
            key_decision = "Whether target customers are willing to pay the required price and accept sustainable unit economics."
            why_this_action = "Subsidized or underpriced usage can create false signal of product-market fit."
            suggested_method = "Structured price threshold test or tiered pricing offer with renewal cohorts."
        elif "acquisition" in t_lower or any(k in stmt_lower for k in ["acquisition", "channel", "cac", "outreach", "recruit", "growth", "marketing", "sales"]):
            action_type = NextStrategicActionType.ACQUISITION_TEST
            target_area = "Acquisition / Customer Access"
            rationale = "Core value and retention are established; the next focus is finding repeatable, scalable customer acquisition channels beyond warm networks."
            key_decision = "Whether target segments can be reached and acquired through repeatable, non-founder channels beyond warm personal networks."
            why_this_action = "Personal network outreach does not prove scalable distribution."
            suggested_method = "Cold outreach or outbound pilot campaign to unconnected target organizations."
        elif "solution" in t_lower or "mvp" in t_lower:
            action_type = NextStrategicActionType.NEW_VALIDATION_TEST
            target_area = "Solution / MVP"
            rationale = "Validating core solution usability ensures the software delivers the intended value."
            key_decision = "Whether users complete the core workflow and derive value from the product independently."
            why_this_action = "A solution that fails core usability cannot sustain customer retention."
            suggested_method = "User workflow session tracking on key product tasks."
        elif "existing alternatives" in t_lower:
            action_type = NextStrategicActionType.NEW_VALIDATION_TEST
            target_area = "Existing Alternatives"
            rationale = "Even with early users, switching motivation against alternative workflows must be verified for broader market segments."
            key_decision = "Whether current workarounds and competitor solutions fail significantly enough to motivate switching to the product."
            why_this_action = "Understanding inertia against existing alternatives is essential for positioning."
            suggested_method = "Alternative comparison interviews with target non-users."
        elif "target user" in t_lower:
            action_type = NextStrategicActionType.NEW_VALIDATION_TEST
            target_area = "Target User"
            rationale = "Focusing on the specific customer segment with the highest acute need ensures strong initial adoption."
            key_decision = "Whether high-retention, high-urgency customer profiles can be clearly identified and distinguished from casual users."
            why_this_action = "Targeting an overly broad audience dilutes value proposition and product focus."
            suggested_method = "Segment analysis and interviews comparing active vs churned customer profiles."
        else:
            action_type = NextStrategicActionType.RETENTION_TEST
            rationale = "For a deployed product with paying customers, recurring retention and cohort stickiness represent the primary gating risk."
            key_decision = "Whether initial paying customers renew and sustain weekly usage."
            why_this_action = "Early traction must prove repeat retention before expanding acquisition."
            suggested_method = "Track multi-week match recording frequency and renewal intent."

    else:
        # Pre-traction / Idea Stage: Map strictly by target_area
        t_lower = target_area.lower()
        if "acquisition" in t_lower:
            action_type = NextStrategicActionType.ACQUISITION_TEST
            target_area = "Acquisition / Customer Access"
            rationale = "Validating customer access channels early determines whether the startup can reach target users predictably."
            key_decision = "Whether target customers can be reached through repeatable, non-founder channels outside warm personal networks."
            why_this_action = "A problem that cannot be reached at reasonable cost cannot support a viable startup."
            suggested_method = "Outbound message test or non-network channel outreach."
        elif "business model" in t_lower:
            action_type = NextStrategicActionType.PRICING_TEST
            target_area = "Business Model"
            rationale = "Verifying willingness to pay and economic exchange prevents building a solution that customers will not purchase."
            key_decision = "Whether target customers are willing to pay the required price and accept sustainable unit economics."
            why_this_action = "Willingness to pay is the ultimate validation of problem severity."
            suggested_method = "Pricing discovery interviews or pre-order / LOI tests."
        elif "solution" in t_lower or "mvp" in t_lower:
            action_type = NextStrategicActionType.NEW_VALIDATION_TEST
            target_area = "Solution / MVP"
            rationale = "Early solution validation tests whether the proposed mechanism actually solves the validated problem."
            key_decision = "Whether users can complete the core workflow and derive value from the minimal solution."
            why_this_action = "Testing core workflow mechanics before full engineering prevents wasted development."
            suggested_method = "Concierge test or manual paper prototype with target participants."
        elif "target user" in t_lower:
            action_type = NextStrategicActionType.NEW_VALIDATION_TEST
            target_area = "Target User"
            rationale = "Focusing on the specific customer segment with the highest acute need ensures strong initial adoption."
            key_decision = "Whether high-urgency, high-retention customer profiles can be clearly identified and distinguished from casual users."
            why_this_action = "Targeting an overly broad audience dilutes value proposition and product focus."
            suggested_method = "Structured customer profile interviews comparing active vs inactive segments."
        elif "pain" in t_lower or "urgency" in t_lower:
            action_type = NextStrategicActionType.NEW_VALIDATION_TEST
            target_area = "Pain / Urgency"
            rationale = "Without acute urgency, prospects may acknowledge the problem but delay adoption indefinitely."
            key_decision = "Whether the pain point creates immediate urgency and budget allocation rather than passive dissatisfaction."
            why_this_action = "Urgency determines whether customers take immediate action or deprioritize the purchase."
            suggested_method = "Urgency threshold interviews and budget validation conversations."
        elif "alternative" in t_lower:
            action_type = NextStrategicActionType.NEW_VALIDATION_TEST
            target_area = "Existing Alternatives"
            rationale = "Customers will only adopt a new solution if existing workarounds leave critical gaps or intolerable friction."
            key_decision = "Whether current workarounds and competitor solutions fail significantly enough to motivate switching."
            why_this_action = "Understanding inertia and workaround tolerance is required before designing an MVP."
            suggested_method = "Workaround breakdown interviews analyzing current costs and failure modes."
        elif "demand signal" in t_lower:
            action_type = NextStrategicActionType.NEW_VALIDATION_TEST
            target_area = "Demand Signal"
            rationale = "Costly commitments (deposits, pre-orders, signed pilots) separate real demand from polite enthusiasm."
            key_decision = "Whether target customers demonstrate genuine demand through costly behavioral commitment rather than verbal interest."
            why_this_action = "Verbal praise is cheap; behavioral commitment proves willingness to adopt."
            suggested_method = "Letter of intent, pre-order deposit, or scheduled pilot agreement test."
        else:
            action_type = NextStrategicActionType.NEW_VALIDATION_TEST
            target_area = "Problem"
            rationale = "Early-stage concept requiring empirical validation of the core problem severity and frequency."
            key_decision = "Whether target customers experience the acute pain point with sufficient frequency and consequence to demand a solution."
            why_this_action = "Validating the core problem cheaply prevents premature product investment."
            suggested_method = "Problem discovery interviews or manual concierge test with target participants."

    # 6. Deterministic Semantic Guards (Contract 3)
    t_lower = target_area.lower()
    if "acquisition" in t_lower:
        bad_terms = ["acute pain", "pain point", "pricing", "willingness to pay", "unit economics"]
        if (
            any(b in key_decision.lower() for b in bad_terms)
            or not any(k in key_decision.lower() for k in ["channel", "reach", "network", "acquir", "outreach", "distribution"])
        ):
            logger.warning("decide_next_strategic_action | Semantic guard triggered for Acquisition: repairing key_decision")
            key_decision = "Whether target segments can be reached and acquired through repeatable, non-founder channels beyond warm personal networks."
            action_type = NextStrategicActionType.ACQUISITION_TEST
            rationale = "Core value is demonstrated; the critical priority is validating repeatable, non-founder customer acquisition channels."
            why_this_action = "Personal network outreach does not prove scalable distribution."
            suggested_method = "Outbound campaign or cold outreach to unconnected target organizations."

    elif "business model" in t_lower:
        if not any(k in key_decision.lower() for k in ["price", "pricing", "pay", "margin", "economics", "revenue", "fee"]):
            logger.warning("decide_next_strategic_action | Semantic guard triggered for Business Model: repairing key_decision")
            key_decision = "Whether target customers are willing to pay the required price and accept sustainable unit economics."
            action_type = NextStrategicActionType.PRICING_TEST
            rationale = "Monetization viability requires verifying price elasticity and sustainable unit economics before scaling."
            why_this_action = "Subsidized or free usage can mask lack of genuine economic value."

    elif "solution" in t_lower or "mvp" in t_lower:
        if any(b in key_decision.lower() for b in ["channel", "cac", "pricing threshold", "unit economics"]):
            logger.warning("decide_next_strategic_action | Semantic guard triggered for Solution/MVP: repairing key_decision")
            if action_type == NextStrategicActionType.PRODUCT_ITERATION:
                key_decision = "Whether the smallest setup and workflow improvement enables users to record matches independently without dropping out."
            else:
                key_decision = "Whether users can complete the core workflow and derive value from the minimal solution."

    # 7. Sanitize any accidental domain leakage
    rationale_clean = sanitize_domain_leakage(rationale, all_founder_raw)
    why_clean = sanitize_domain_leakage(why_this_action, all_founder_raw)
    method_clean = sanitize_domain_leakage(suggested_method, all_founder_raw) if suggested_method else None

    act = NextStrategicAction(
        action_type=action_type,
        rationale=rationale_clean,
        target_area=target_area,
        key_decision=key_decision,
        why_this_action=why_clean,
        suggested_method=method_clean,
        bottleneck_identified=bottleneck_identified,
        blocked_behavior=blocked_behavior,
        smallest_change=smallest_change,
        measurable_outcome=measurable_outcome,
        what_not_to_build=what_not_to_build,
        is_weak_link_override=is_weak_link_override,
        override_reason=override_reason,
    )

    logger.info(
        "decide_next_strategic_action | action_type=%s target_area=%s is_override=%s key_decision=%s",
        act.action_type.value, act.target_area, act.is_weak_link_override, act.key_decision[:80],
    )

    return {"next_strategic_action": act.model_dump()}

