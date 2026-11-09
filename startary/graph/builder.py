"""
LangGraph graph builder for the Startary AI system — Milestone 2.6.1.

M2.6.1 graph flow (Three Distinct Information Loops):
  initialize_state
       ↓
  verify_state
       ↓
  extract_startup_state (Passive Delta Extraction)
       ↓
  merge_execution_state (Deterministic Provenance Merge)
       ↓
  assess_baseline_state_sufficiency (M2.6.1 Baseline Gate)
       ↓
  [route_on_baseline]
       ├─(incomplete & turns < 5)──> plan_baseline_question ──> ask_and_wait_for_founder (interrupt)
       │                                                                ↓
       │                                                        merge_founder_answer
       │                                                                ↓
       │                                                        extract_startup_state ──┐
       │                                                                                │
       └─(sufficient or turns >= 5)──> diagnose_startup                                 │
                                            ↓                                           │
                                       merge_diagnosis                                  │
                                            ↓                                           │
                                       plan_information_gap (Strategic Sufficiency)     │
                                            ↓                                           │
                                       [route_on_gap_plan]                              │
                                            ├─(should_ask & turns < 5)──> ask_and_wait──┤
                                            │                                           │
                                            ├─(should_ask & turns >= 5)──> safe stop    │
                                            │                                           │
                                            └─(sufficient)──> assess_startup_areas      │
                                                                   ↓                    │
                                                              select_weak_link          │
                                                                   ↓                    │
                                                              assess_weak_link_discrimination
                                                                   ↓
                                                              [route_on_weak_link]
                                                                   ├─(clarify & turns < 3)──> select_weak_link_question ──> ask_and_wait
                                                                   │
                                                                   └─(proceed or turns >= 3)──> reality_check
                                                                                                    ↓
                                                                                               generate_critical_assumption
                                                                                                    ↓
                                                                                               save_critical_assumption
                                                                                                    ↓
                                                                                                   END

M6.2.3 question delivery (every planner question):
  plan_baseline_question / plan_information_gap / select_weak_link_question
       ↓
  compose_founder_question   (single composer LLM call; exact prompt persisted in checkpoint)
       ↓
  ask_and_wait_for_founder   (LLM-free interrupt; re-executes safely on resume)
       ↓
  merge_founder_answer       (transcript records the persisted prompt verbatim)
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from startary.domain.experiment_contract import experiment_critique_route
from startary.domain.enums import (
    EvidenceSubmissionSufficiency,
    ExecutionReadiness,
    WorkflowPhase,
)
from startary.domain.evidence import (
    EvidenceTurnDisposition,
    get_evidence_turn_disposition,
    has_substantive_completed_evidence,
)
from startary.domain.gap_plan import (
    UncertaintyResolutionMode,
    can_formulate_testable_uncertainty,
)
from startary.config import get_settings
from startary.graph.graph_state import StartaryGraphState
from startary.graph.nodes import (
    MAX_BASELINE_QUESTION_TURNS,
    MAX_CLARIFICATION_TURNS,
    MAX_STRATEGIC_CLARIFICATION_TURNS,
    MAX_WEAK_LINK_CLARIFICATION_TURNS,
    analyze_evidence,
    answer_post_evidence_question,
    apply_validated_state_updates,
    assess_build_readiness,
    decide_next_strategic_action,
    ask_and_wait_for_founder,
    ask_evidence_clarification,
    assess_baseline_state_sufficiency,
    assess_can_execute,
    assess_evidence_submission_sufficiency,
    assess_startup_areas,
    assess_weak_link_discrimination,
    classify_waiting_for_evidence_turn,
    compose_advisor_response,
    compose_execution_handoff,
    compose_founder_question,
    compose_learning_delta,
    compose_startup_diagnosis,
    critique_experiment,
    revise_experiment,
    diagnose_startup,
    extract_experiment_evidence,
    dispatch_founder_turn,
    extract_startup_state,
    generate_critical_assumption,
    generate_experiment,
    handle_conversational_repair,
    handle_experiment_failure_stop,
    handle_insufficient_context_stop,
    increment_iteration,
    initialize_state,
    interpret_founder_turn,
    join_founder_turn,
    merge_diagnosis,
    merge_execution_state,
    merge_founder_answer,
    parallel_extract_startup_state,
    parallel_interpret_founder_turn,
    pause_needs_help,
    plan_baseline_question,
    plan_information_gap,
    reality_check,
    respond_to_waiting_founder,
    save_critical_assumption,
    save_final_experiment,
    select_weak_link,
    select_weak_link_question,
    validate_evidence_analysis,
    validate_proposed_experiment,
    verify_state,
    wait_for_evidence,
)

logger = logging.getLogger(__name__)


def route_on_baseline(state: StartaryGraphState) -> str:
    """
    Route after assessing baseline state sufficiency:
    - If baseline is sufficient or baseline turns >= MAX: proceed to diagnose_startup.
    - If baseline is incomplete and turns < MAX: plan next grouped baseline question.
    """
    base_suff = state.get("baseline_sufficiency") or {}
    is_sufficient = state.get("is_baseline_sufficient", base_suff.get("is_sufficient", False))
    turn_count = state.get("baseline_question_turns", 0)
    wl_turns = state.get("weak_link_clarification_turns", 0)

    if is_sufficient or turn_count >= MAX_BASELINE_QUESTION_TURNS or wl_turns > 0:
        logger.info("route_on_baseline -> diagnose_startup (is_sufficient=%s, turns=%d, wl_turns=%d)", is_sufficient, turn_count, wl_turns)
        return "diagnose_startup"

    logger.info("route_on_baseline -> plan_baseline_question (incomplete, turn=%d)", turn_count)
    return "plan_baseline_question"


def route_after_merge_diagnosis(state: StartaryGraphState) -> str:
    """
    Route after merge_diagnosis (Milestone 2.6.2):
    - If in a weak-link clarification loop (weak_link_clarification_turns > 0) or strategic
      sufficiency was already achieved, skip strategic question planning and go directly to
      re-assessing startup areas.
    - Otherwise, proceed to plan_information_gap for strategic understand evaluation.
    """
    if state.get("weak_link_clarification_turns", 0) > 0 or state.get("is_strategic_sufficient"):
        logger.info("route_after_merge_diagnosis -> assess_startup_areas (weak-link loop re-evaluation / strategic sufficient)")
        return "assess_startup_areas"

    logger.info("route_after_merge_diagnosis -> plan_information_gap (strategic evaluate)")
    return "plan_information_gap"


def route_on_gap_plan(state: StartaryGraphState) -> str:
    """
    Deterministic Strategic Routing Decision (Milestone 2.6 & Milestone 4.0.2):
    - If resolution_mode == EXTERNAL_EVIDENCE and can_formulate_testable_uncertainty:
        Proceed to TEST (assess_startup_areas) without asking more UNDERSTAND questions.
    - If should_ask=False:
        If strategic_sufficient or testable uncertainty exists -> assess_startup_areas.
        Otherwise -> handle_insufficient_context_stop.
    - If should_ask=True:
        If turns < MAX -> ask_and_wait_for_founder.
        If turns >= MAX:
            If can_formulate_testable_uncertainty and not NOT_YET_FORMULATABLE:
                proceed to assess_startup_areas (do not drop understandable startups!).
            Else:
                handle_insufficient_context_stop.
    """
    gap_plan = state.get("gap_plan") or {}
    turn_count = state.get("strategic_clarification_turns", state.get("clarification_turn_count", 0))

    if not gap_plan:
        is_sufficient = state.get("is_sufficient", False)
        if is_sufficient:
            logger.info("route_on_gap_plan -> assess_startup_areas (SUFFICIENT via legacy)")
            return "assess_startup_areas"
        if turn_count >= MAX_STRATEGIC_CLARIFICATION_TURNS:
            can_test = can_formulate_testable_uncertainty(state)
            if can_test:
                logger.info("route_on_gap_plan -> assess_startup_areas (MAX_TURNS legacy, but testable)")
                return "assess_startup_areas"
            logger.info("route_on_gap_plan -> handle_insufficient_context_stop (MAX_TURNS via legacy)")
            return "handle_insufficient_context_stop"
        logger.info("route_on_gap_plan -> ask_and_wait_for_founder (INSUFFICIENT via legacy, turn=%d)", turn_count)
        return "ask_and_wait_for_founder"

    should_ask = gap_plan.get("should_ask", True)
    strategic_sufficient = gap_plan.get("strategic_sufficient", False)
    res_mode = gap_plan.get("resolution_mode") or "FOUNDER_KNOWLEDGE"
    can_test = gap_plan.get("can_formulate_testable_uncertainty")
    if can_test is None:
        can_test = can_formulate_testable_uncertainty(state, gap_plan)

    # 1. External Evidence Handoff: Immediate route to TEST
    if str(res_mode).upper() == UncertaintyResolutionMode.EXTERNAL_EVIDENCE.value and can_test:
        logger.info("route_on_gap_plan -> assess_startup_areas (EXTERNAL_EVIDENCE handoff to TEST, turn=%d)", turn_count)
        return "assess_startup_areas"

    # 2. Strategic sufficiency / no question needed
    if not should_ask:
        if strategic_sufficient or can_test:
            logger.info("route_on_gap_plan -> assess_startup_areas (SUFFICIENT / testable, strategic_sufficient=%s, can_test=%s)", strategic_sufficient, can_test)
            return "assess_startup_areas"
        logger.info("route_on_gap_plan -> handle_insufficient_context_stop (should_ask=False, unformulatable)")
        return "handle_insufficient_context_stop"

    # 3. Turns exhausted: Check if testable uncertainty exists before stopping!
    if turn_count >= MAX_STRATEGIC_CLARIFICATION_TURNS:
        if can_test and str(res_mode).upper() != UncertaintyResolutionMode.NOT_YET_FORMULATABLE.value:
            logger.info("route_on_gap_plan -> assess_startup_areas (MAX_TURNS reached, but testable uncertainty exists -> handoff to TEST, turn=%d)", turn_count)
            return "assess_startup_areas"
        logger.info("route_on_gap_plan -> handle_insufficient_context_stop (MAX_TURNS_REACHED, unformulatable, turn=%d)", turn_count)
        return "handle_insufficient_context_stop"

    logger.info("route_on_gap_plan -> ask_and_wait_for_founder (should_ask=True, res_mode=%s, turn=%d)", res_mode, turn_count)
    return "ask_and_wait_for_founder"


def route_on_context_sufficiency(state: StartaryGraphState) -> str:
    """
    Backward-compatible alias for route_on_gap_plan.
    M2.5 tests import this function by name from builder; it delegates to the strategic router.
    """
    gap_plan = state.get("gap_plan") or {}
    cs_dict = state.get("context_sufficiency") or {}
    is_sufficient = gap_plan.get("strategic_sufficient", cs_dict.get("sufficient", False))
    res_mode = gap_plan.get("resolution_mode") or "FOUNDER_KNOWLEDGE"
    can_test = can_formulate_testable_uncertainty(state, gap_plan)
    turn_count = state.get("strategic_clarification_turns", state.get("clarification_turn_count", 0))

    if is_sufficient or (str(res_mode).upper() == UncertaintyResolutionMode.EXTERNAL_EVIDENCE.value and can_test):
        return "assess_startup_areas"
    if turn_count >= MAX_STRATEGIC_CLARIFICATION_TURNS:
        if can_test and str(res_mode).upper() != UncertaintyResolutionMode.NOT_YET_FORMULATABLE.value:
            return "assess_startup_areas"
        return "handle_insufficient_context_stop"
    return "select_clarifying_question"


def route_on_weak_link(state: StartaryGraphState) -> str:
    """
    Deterministic Weak-Link Discrimination Routing (Milestone 2.6.2 & Milestone 4.0.2):
    - The structured field `should_clarify` is authoritative.
    - If resolution_mode == EXTERNAL_EVIDENCE: proceed under current confidence to reality_check.
    - If should_clarify=True and turns < MAX: ask discriminating question.
    - Otherwise (single clear winner, or max turns reached): proceed to reality_check.
    - Low/Medium selection confidence is preserved if max turns reached without tie breaking.
    """
    disc = state.get("weak_link_discrimination") or {}
    should_clarify = state.get("should_clarify")
    if should_clarify is None:
        should_clarify = disc.get("should_clarify")
    if should_clarify is None:
        should_clarify = state.get("should_clarify_weak_link", False)

    res_mode = disc.get("resolution_mode") or "FOUNDER_KNOWLEDGE"
    if str(res_mode).upper() == UncertaintyResolutionMode.EXTERNAL_EVIDENCE.value:
        should_clarify = False

    turns_used = state.get("weak_link_clarification_turns", 0)

    if bool(should_clarify) and turns_used < MAX_WEAK_LINK_CLARIFICATION_TURNS:
        logger.info("route_on_weak_link -> select_weak_link_question (should_clarify=True, turn=%d)", turns_used)
        return "select_weak_link_question"

    logger.info("route_on_weak_link -> reality_check (proceeding, should_clarify=%s, conf=%s, turns=%d)", should_clarify, state.get("selection_confidence"), turns_used)
    return "reality_check"


def route_on_experiment_critique(state: StartaryGraphState) -> str:
    """
    Routes based on Experiment Critic verdict (Milestone 3):
    - If ACCEPT -> save_final_experiment
    - If REVISE and revision_count <= 1 -> generate_experiment (triggers 1 internal revision loop)
    - If REVISE and revision_count > 1 -> handle_experiment_failure_stop
    """
    critic_output = state.get("experiment_critic_output") or {}
    revision_count = state.get("critic_revision_count", 0)
    # Milestone 6.2.5: exact-token verdict (no substring matching); rule shared with critic telemetry.
    strategy = (state.get("experiment_revision") or {}).get("next_strategy")  # M6.2.6 V2 only
    route = experiment_critique_route(critic_output.get("verdict"), revision_count, strategy)
    log = logger.warning if route == "handle_experiment_failure_stop" else logger.info
    log(
        "route_on_experiment_critique -> %s (verdict=%s source=%s failed=%s revision_count=%d)",
        route, critic_output.get("verdict"), critic_output.get("decision_source"),
        critic_output.get("failed_dimensions"), revision_count,
    )
    return route


def route_after_can_execute(state: StartaryGraphState) -> str:
    """
    Routes based on Execution Readiness (Milestone 4.0.3):
    - CAN_EXECUTE -> compose_execution_handoff (presents accepted experiment before pausing)
    - NEEDS_HELP -> pause_needs_help (pauses with missing requirement explanation)
    - BLOCKED -> handle_experiment_failure_stop
    """
    can_exec = state.get("can_execute_result") or {}
    readiness = can_exec.get("readiness")
    if readiness == ExecutionReadiness.CAN_EXECUTE.value or readiness == ExecutionReadiness.CAN_EXECUTE:
        return "compose_execution_handoff"
    if readiness == ExecutionReadiness.NEEDS_HELP.value or readiness == ExecutionReadiness.NEEDS_HELP:
        return "pause_needs_help"
    return "handle_experiment_failure_stop"


def route_after_interpret_founder_turn(state: StartaryGraphState) -> str:
    """
    Routes after founder turn interpretation (Milestone 4.0.3 & 4.1.3):
    - If conversational repair -> handle_conversational_repair
    - If in WAITING_FOR_EVIDENCE phase -> classify_waiting_for_evidence_turn (Evidence Admission Gate)
    - Otherwise (UNDERSTAND phase) -> extract_startup_state
    """
    if state.get("is_conversational_repair") or state.get("execution_path") == "CONVERSATIONAL_REPAIR":
        logger.info("route_after_interpret_founder_turn -> handle_conversational_repair (CONVERSATIONAL_REPAIR)")
        return "handle_conversational_repair"

    phase = state.get("phase")
    if phase == WorkflowPhase.WAITING_FOR_EVIDENCE or phase == "WAITING_FOR_EVIDENCE":
        return "classify_waiting_for_evidence_turn"
    return "extract_startup_state"


# --- Milestone 6.2.7: parallel understanding routes (only wired when STARTARY_PARALLEL_UNDERSTANDING=true) ---

def _dispatch_is_parallel(state: StartaryGraphState) -> bool:
    return (state.get("founder_turn_dispatch") or {}).get("mode") == "PARALLEL"


def route_on_founder_turn_dispatch(state: StartaryGraphState) -> Any:
    """PARALLEL -> fan out to both understanding nodes (same superstep); SEQUENTIAL -> interpretation only."""
    if _dispatch_is_parallel(state):
        return ["interpret_founder_turn", "extract_startup_state"]
    return "interpret_founder_turn"


def route_after_interpret_parallel_aware(state: StartaryGraphState) -> str:
    """In the parallel region interpretation always goes to the barrier; otherwise the unchanged route."""
    if _dispatch_is_parallel(state):
        return "join_founder_turn"
    return route_after_interpret_founder_turn(state)


def route_after_join_founder_turn(state: StartaryGraphState) -> str:
    """After the barrier: the same repair check route_after_interpret_founder_turn applies, else merge."""
    if state.get("is_conversational_repair") or state.get("execution_path") == "CONVERSATIONAL_REPAIR":
        return "handle_conversational_repair"
    return "merge_execution_state"


def parallel_understanding_enabled() -> bool:
    """STARTARY_PARALLEL_UNDERSTANDING; only an explicit boolean True enables it."""
    return getattr(get_settings(), "parallel_understanding", False) is True


def route_waiting_for_evidence_turn(state: StartaryGraphState) -> str:
    """
    Evidence Admission Gate Routing (Milestone 4.0.3 & 4.0.7):
    - ACTUAL_EVIDENCE -> extract_experiment_evidence (enters M4 evidence extraction/analysis pipeline)
    - All other dispositions (questions, requests for help, future intentions, cannot execute) -> respond_to_waiting_founder
    - Router fail-safe: if evidence_turn_assessment is missing, logs evidence_routing_integrity_error and applies safe repair.
    """
    disp = get_evidence_turn_disposition(state)

    if disp == EvidenceTurnDisposition.ACTUAL_EVIDENCE:
        logger.info("evidence_turn_routed | disposition=ACTUAL_EVIDENCE route=extract_experiment_evidence")
        return "extract_experiment_evidence"

    if disp is None:
        # Router Fail-Safe (Milestone 4.0.7)
        logger.error(
            "evidence_routing_integrity_error | reason=missing_evidence_turn_assessment | project_id=%s",
            state.get("project_id", "<unknown>"),
        )
        # Attempt fail-safe recovery from founder turn text
        raw_msg = state.get("founder_response") or ""
        if not raw_msg:
            founder_messages = state.get("founder_messages") or []
            for m in reversed(founder_messages):
                if m.get("role") == "founder":
                    raw_msg = m.get("content", "")
                    break
        if has_substantive_completed_evidence(raw_msg):
            logger.warning("evidence_routing_fail_safe_recovered | disposition=ACTUAL_EVIDENCE route=extract_experiment_evidence")
            logger.info("evidence_turn_routed | disposition=ACTUAL_EVIDENCE route=extract_experiment_evidence")
            return "extract_experiment_evidence"

        logger.info("evidence_turn_routed | disposition=None route=respond_to_waiting_founder")
        return "respond_to_waiting_founder"

    logger.info("evidence_turn_routed | disposition=%s route=respond_to_waiting_founder", disp.value)
    return "respond_to_waiting_founder"


def route_on_evidence_sufficiency(state: StartaryGraphState) -> str:
    """
    Routes based on Evidence Submission Sufficiency (Milestone 4):
    - NEEDS_CLARIFICATION and turns < 2 -> ask_evidence_clarification
    - SUFFICIENT_FOR_ANALYSIS or max turns -> analyze_evidence
    """
    suff = state.get("evidence_submission_sufficiency")
    turns = state.get("evidence_clarification_turns", 0)
    if suff == EvidenceSubmissionSufficiency.NEEDS_CLARIFICATION.value and turns < 2:
        return "ask_evidence_clarification"
    return "analyze_evidence"


def route_after_validate_evidence(state: StartaryGraphState) -> str:
    """
    Milestone 4.1.2: If partial evidence was submitted, pause and loop back to wait_for_evidence.
    Otherwise proceed to apply_validated_state_updates.
    """
    if state.get("is_partial_evidence"):
        return "wait_for_evidence"
    return "apply_validated_state_updates"


def build_graph(checkpointer: Optional[Any] = None, parallel_understanding: Optional[bool] = None):
    """
    Compile and return the Startary LangGraph (Milestone 4 version).

    Supports human-in-the-loop interrupt/resume across:
      1. Baseline Startup Check-In
      2. Strategic Understand Clarification
      3. Weak-Link Discrimination
      4. Execution Handoff (wait_for_evidence)
      5. Evidence Clarification (if needed)
    Followed by Evidence Analysis, Validated State Updates, Iteration Increment,
    Re-Diagnosis, and Learning Delta.

    parallel_understanding (Milestone 6.2.7): None reads STARTARY_PARALLEL_UNDERSTANDING. False builds
    the unchanged sequential graph. True inserts dispatch_founder_turn / join_founder_turn around
    interpret_founder_turn and extract_startup_state so they run concurrently on extraction turns.
    """
    if parallel_understanding is None:
        parallel_understanding = parallel_understanding_enabled()
    parallel_understanding = parallel_understanding is True
    graph = StateGraph(StartaryGraphState)

    # --- Nodes ---
    graph.add_node("initialize_state", initialize_state)
    graph.add_node("verify_state", verify_state)
    graph.add_node("extract_startup_state", parallel_extract_startup_state if parallel_understanding else extract_startup_state)
    graph.add_node("merge_execution_state", merge_execution_state)

    # Baseline Nodes (M2.6.1)
    graph.add_node("assess_baseline_state_sufficiency", assess_baseline_state_sufficiency)
    graph.add_node("plan_baseline_question", plan_baseline_question)

    # Common Interrupt & Answer Merge Nodes
    # M6.2.3: composition is checkpointed before the interrupt so resume never re-composes.
    graph.add_node("compose_founder_question", compose_founder_question)
    graph.add_node("ask_and_wait_for_founder", ask_and_wait_for_founder)
    graph.add_node("merge_founder_answer", merge_founder_answer)

    # Strategic Diagnosis & Gap Planning Nodes
    graph.add_node("diagnose_startup", diagnose_startup)
    graph.add_node("merge_diagnosis", merge_diagnosis)
    graph.add_node("plan_information_gap", plan_information_gap)
    graph.add_node("handle_insufficient_context_stop", handle_insufficient_context_stop)

    # Downstream Assessment & Weak Link Nodes
    graph.add_node("assess_startup_areas", assess_startup_areas)
    graph.add_node("select_weak_link", select_weak_link)

    # Weak-Link Discrimination Nodes (M2.6.1)
    graph.add_node("assess_weak_link_discrimination", assess_weak_link_discrimination)
    graph.add_node("select_weak_link_question", select_weak_link_question)

    # Reality Check & Critical Assumption Nodes (M2)
    graph.add_node("reality_check", reality_check)
    graph.add_node("generate_critical_assumption", generate_critical_assumption)
    graph.add_node("save_critical_assumption", save_critical_assumption)

    # Milestone 3.1: Founder Diagnosis Snapshot
    graph.add_node("compose_startup_diagnosis", compose_startup_diagnosis)

    # Milestone 3.2: Founder Intent & Conversational Awareness Nodes
    graph.add_node("interpret_founder_turn", parallel_interpret_founder_turn if parallel_understanding else interpret_founder_turn)
    graph.add_node("compose_advisor_response", compose_advisor_response)

    # Milestone 3 Nodes: Experiment Generation & Critic
    graph.add_node("generate_experiment", generate_experiment)
    graph.add_node("validate_proposed_experiment", validate_proposed_experiment)
    graph.add_node("revise_experiment", revise_experiment)  # M6.2.6 V2: targeted edit (unreachable with the flag off)
    graph.add_node("critique_experiment", critique_experiment)
    graph.add_node("save_final_experiment", save_final_experiment)
    graph.add_node("assess_can_execute", assess_can_execute)
    graph.add_node("handle_experiment_failure_stop", handle_experiment_failure_stop)

    # Milestone 4 Nodes: Evidence Intake, Analysis, State Update, Learning Loop
    graph.add_node("wait_for_evidence", wait_for_evidence)
    graph.add_node("pause_needs_help", pause_needs_help)
    graph.add_node("extract_experiment_evidence", extract_experiment_evidence)
    graph.add_node("assess_evidence_submission_sufficiency", assess_evidence_submission_sufficiency)
    graph.add_node("ask_evidence_clarification", ask_evidence_clarification)
    graph.add_node("analyze_evidence", analyze_evidence)
    graph.add_node("validate_evidence_analysis", validate_evidence_analysis)
    graph.add_node("apply_validated_state_updates", apply_validated_state_updates)
    graph.add_node("increment_iteration", increment_iteration)
    graph.add_node("compose_learning_delta", compose_learning_delta)
    graph.add_node("answer_post_evidence_question", answer_post_evidence_question)
    graph.add_node("assess_build_readiness", assess_build_readiness)
    graph.add_node("decide_next_strategic_action", decide_next_strategic_action)

    # Milestone 4.0.3 Nodes: Execution Handoff & Evidence Admission Gate
    graph.add_node("compose_execution_handoff", compose_execution_handoff)
    graph.add_node("classify_waiting_for_evidence_turn", classify_waiting_for_evidence_turn)
    graph.add_node("respond_to_waiting_founder", respond_to_waiting_founder)

    # Milestone 4.1.3 Node: Conversational Repair Fast Path
    graph.add_node("handle_conversational_repair", handle_conversational_repair)

    # Milestone 6.2.7: every founder turn enters understanding here (interpret_founder_turn when the
    # flag is off, exactly as before; dispatch_founder_turn when it is on).
    understanding_entry = "interpret_founder_turn"
    if parallel_understanding:
        understanding_entry = "dispatch_founder_turn"
        graph.add_node("dispatch_founder_turn", dispatch_founder_turn)
        graph.add_node("join_founder_turn", join_founder_turn)

    # --- Flow Topology ---
    graph.set_entry_point("initialize_state")
    graph.add_edge("initialize_state", "verify_state")
    graph.add_edge("verify_state", understanding_entry)
    if not parallel_understanding:
        graph.add_conditional_edges(
            "interpret_founder_turn",
            route_after_interpret_founder_turn,
            {
                "extract_startup_state": "extract_startup_state",
                "classify_waiting_for_evidence_turn": "classify_waiting_for_evidence_turn",
                "handle_conversational_repair": "handle_conversational_repair",
            },
        )
        graph.add_edge("extract_startup_state", "merge_execution_state")
    else:
        # Fan-out: both branches run in one superstep; LangGraph completes the superstep (both
        # branches) before join_founder_turn runs, and join verifies both branch results exist.
        graph.add_conditional_edges(
            "dispatch_founder_turn",
            route_on_founder_turn_dispatch,
            ["interpret_founder_turn", "extract_startup_state"],
        )
        graph.add_conditional_edges(
            "interpret_founder_turn",
            route_after_interpret_parallel_aware,
            {
                "join_founder_turn": "join_founder_turn",
                "extract_startup_state": "extract_startup_state",
                "classify_waiting_for_evidence_turn": "classify_waiting_for_evidence_turn",
                "handle_conversational_repair": "handle_conversational_repair",
            },
        )
        graph.add_edge("extract_startup_state", "join_founder_turn")
        graph.add_conditional_edges(
            "join_founder_turn",
            route_after_join_founder_turn,
            {
                "merge_execution_state": "merge_execution_state",
                "handle_conversational_repair": "handle_conversational_repair",
            },
        )
    graph.add_edge("handle_conversational_repair", "ask_and_wait_for_founder")
    graph.add_edge("merge_execution_state", "assess_baseline_state_sufficiency")

    # --- Conditional Edge: Baseline Routing ---
    graph.add_conditional_edges(
        "assess_baseline_state_sufficiency",
        route_on_baseline,
        {
            "diagnose_startup": "diagnose_startup",
            "plan_baseline_question": "plan_baseline_question",
        },
    )
    graph.add_edge("plan_baseline_question", "compose_founder_question")

    # --- Strategic Diagnosis Flow ---
    graph.add_edge("diagnose_startup", "merge_diagnosis")
    graph.add_conditional_edges(
        "merge_diagnosis",
        route_after_merge_diagnosis,
        {
            "assess_startup_areas": "assess_startup_areas",
            "plan_information_gap": "plan_information_gap",
        },
    )

    # --- Conditional Edge: Strategic Gap Plan Routing ---
    graph.add_conditional_edges(
        "plan_information_gap",
        route_on_gap_plan,
        {
            "assess_startup_areas": "assess_startup_areas",
            # Route key kept stable; the founder question is composed before the interrupt.
            "ask_and_wait_for_founder": "compose_founder_question",
            "handle_insufficient_context_stop": "handle_insufficient_context_stop",
        },
    )

    # --- Answer Loop (interprets turn and re-extracts delta after founder responds) ---
    graph.add_edge("compose_founder_question", "ask_and_wait_for_founder")
    graph.add_edge("ask_and_wait_for_founder", "merge_founder_answer")
    graph.add_edge("merge_founder_answer", understanding_entry)

    # --- Insufficient Context Safe Stop ---
    graph.add_edge("handle_insufficient_context_stop", END)

    # --- Area Risk Assessment & Weak Link ---
    graph.add_edge("assess_startup_areas", "select_weak_link")
    graph.add_edge("select_weak_link", "assess_weak_link_discrimination")

    # --- Conditional Edge: Weak-Link Discrimination Routing ---
    graph.add_conditional_edges(
        "assess_weak_link_discrimination",
        route_on_weak_link,
        {
            "select_weak_link_question": "select_weak_link_question",
            "reality_check": "reality_check",
        },
    )
    graph.add_edge("select_weak_link_question", "compose_founder_question")

    # --- Downstream Pipeline ---
    graph.add_edge("reality_check", "generate_critical_assumption")
    graph.add_edge("generate_critical_assumption", "save_critical_assumption")
    graph.add_edge("save_critical_assumption", "compose_learning_delta")
    graph.add_edge("compose_learning_delta", "compose_startup_diagnosis")
    graph.add_edge("compose_startup_diagnosis", "answer_post_evidence_question")
    graph.add_edge("answer_post_evidence_question", "assess_build_readiness")
    graph.add_edge("assess_build_readiness", "decide_next_strategic_action")
    graph.add_edge("decide_next_strategic_action", "generate_experiment")

    # --- Milestone 3 Experiment Pipeline ---
    graph.add_edge("generate_experiment", "validate_proposed_experiment")
    graph.add_edge("validate_proposed_experiment", "critique_experiment")

    graph.add_conditional_edges(
        "critique_experiment",
        route_on_experiment_critique,
        {
            "save_final_experiment": "save_final_experiment",
            "generate_experiment": "generate_experiment",
            "revise_experiment": "revise_experiment",
            "handle_experiment_failure_stop": "handle_experiment_failure_stop",
        },
    )
    graph.add_edge("revise_experiment", "validate_proposed_experiment")

    graph.add_edge("save_final_experiment", "assess_can_execute")
    graph.add_conditional_edges(
        "assess_can_execute",
        route_after_can_execute,
        {
            "compose_execution_handoff": "compose_execution_handoff",
            "pause_needs_help": "pause_needs_help",
            "handle_experiment_failure_stop": "handle_experiment_failure_stop",
        },
    )
    graph.add_edge("compose_execution_handoff", "wait_for_evidence")
    graph.add_edge("pause_needs_help", END)
    # Milestone 4.1.1: Recoverable experiment failure enters interrupt loop instead of terminating
    graph.add_edge("handle_experiment_failure_stop", "ask_and_wait_for_founder")

    # --- Milestone 4 & 4.0.3 Evidence Admission & Learning Loop ---
    graph.add_edge("wait_for_evidence", understanding_entry)
    graph.add_conditional_edges(
        "classify_waiting_for_evidence_turn",
        route_waiting_for_evidence_turn,
        {
            "extract_experiment_evidence": "extract_experiment_evidence",
            "respond_to_waiting_founder": "respond_to_waiting_founder",
        },
    )
    graph.add_edge("respond_to_waiting_founder", "wait_for_evidence")

    graph.add_edge("extract_experiment_evidence", "assess_evidence_submission_sufficiency")

    graph.add_conditional_edges(
        "assess_evidence_submission_sufficiency",
        route_on_evidence_sufficiency,
        {
            "ask_evidence_clarification": "ask_evidence_clarification",
            "analyze_evidence": "analyze_evidence",
        },
    )
    graph.add_edge("ask_evidence_clarification", "extract_experiment_evidence")

    graph.add_edge("analyze_evidence", "validate_evidence_analysis")
    graph.add_conditional_edges(
        "validate_evidence_analysis",
        route_after_validate_evidence,
        {
            "wait_for_evidence": "wait_for_evidence",
            "apply_validated_state_updates": "apply_validated_state_updates",
        },
    )
    graph.add_edge("apply_validated_state_updates", "increment_iteration")

    # Re-diagnosis loop after evidence incorporated
    graph.add_edge("increment_iteration", "diagnose_startup")

    compiled = graph.compile(checkpointer=checkpointer)
    logger.debug("LangGraph compiled successfully (Milestone 4 with checkpointer, parallel_understanding=%s)", parallel_understanding)
    return compiled


def build_experiment_graph(checkpointer: Optional[Any] = None):
    """
    Compile and return a standalone Experiment Generation & Critic subgraph.
    Enables development and testing directly from WorkflowPhase.TEST.
    """
    graph = StateGraph(StartaryGraphState)
    graph.add_node("generate_experiment", generate_experiment)
    graph.add_node("validate_proposed_experiment", validate_proposed_experiment)
    graph.add_node("revise_experiment", revise_experiment)  # M6.2.6 V2: targeted edit (unreachable with the flag off)
    graph.add_node("critique_experiment", critique_experiment)
    graph.add_node("save_final_experiment", save_final_experiment)
    graph.add_node("assess_can_execute", assess_can_execute)
    graph.add_node("handle_experiment_failure_stop", handle_experiment_failure_stop)

    graph.set_entry_point("generate_experiment")
    graph.add_edge("generate_experiment", "validate_proposed_experiment")
    graph.add_edge("validate_proposed_experiment", "critique_experiment")
    graph.add_conditional_edges(
        "critique_experiment",
        route_on_experiment_critique,
        {
            "save_final_experiment": "save_final_experiment",
            "generate_experiment": "generate_experiment",
            "revise_experiment": "revise_experiment",
            "handle_experiment_failure_stop": "handle_experiment_failure_stop",
        },
    )
    graph.add_edge("revise_experiment", "validate_proposed_experiment")
    graph.add_edge("save_final_experiment", "assess_can_execute")
    graph.add_edge("assess_can_execute", END)
    graph.add_edge("handle_experiment_failure_stop", END)

    return graph.compile(checkpointer=checkpointer)


def run_experiment_pipeline(state: StartaryGraphState, checkpointer: Optional[Any] = None) -> StartaryGraphState:
    """
    Development/testing entry point:
    Executes experiment generation starting directly from an existing valid TEST-phase state.
    """
    compiled = build_experiment_graph(checkpointer=checkpointer)
    thread_id = state.get("project_id", "dev-test-experiment")
    cfg = {"configurable": {"thread_id": thread_id}}
    return compiled.invoke(state, cfg)


def run_diagnosis_snapshot(state: StartaryGraphState) -> StartaryGraphState:
    """
    Direct development/testing entry point (Milestone 3.1):
    Executes compose_startup_diagnosis directly on the given state without
    rerunning conversational onboarding or experiment generation.
    """
    res = compose_startup_diagnosis(state)
    updated = dict(state)
    updated.update(res)
    return updated


def run_founder_turn_interpretation(state: StartaryGraphState) -> StartaryGraphState:
    """
    Direct development/testing entry point (Milestone 3.2):
    Executes interpret_founder_turn directly on the given state.
    """
    updates = interpret_founder_turn(state)
    updated = dict(state)
    updated.update(updates)
    return updated


def run_advisor_response(state: StartaryGraphState) -> StartaryGraphState:
    """
    Direct development/testing entry point (Milestone 3.2):
    Executes compose_advisor_response directly on the given state.
    """
    updates = compose_advisor_response(state)
    updated = dict(state)
    updated.update(updates)
    return updated


def run_planner_turn_decision(state: StartaryGraphState) -> StartaryGraphState:
    """
    Direct development/testing entry point (Milestone 3.2.1):
    Executes plan_baseline_question directly on state and returns updated state with active_next_turn_decision.
    """
    updates = plan_baseline_question(state)
    updated = dict(state)
    updated.update(updates)
    return updated


def run_evidence_learning_pipeline(
    state: StartaryGraphState,
    founder_evidence_text: str,
) -> StartaryGraphState:
    """
    Direct development & testing entry point (Milestone 4.0.3):
    Runs turn interpretation, admission gate, evidence extraction, analysis, state update,
    iteration increment, re-diagnosis, and learning delta synthesis starting directly from an active experiment.
    If the turn is non-evidence, routes to respond_to_waiting_founder and returns without analyzing.
    """
    current_state = dict(state)
    current_state["founder_response"] = founder_evidence_text
    founder_messages = list(current_state.get("founder_messages") or [])
    founder_messages.append({"role": "founder", "content": founder_evidence_text})
    current_state["founder_messages"] = founder_messages

    # 0. Interpret turn & classify admission (Milestone 4.0.3)
    interp_updates = interpret_founder_turn(current_state)
    current_state.update(interp_updates)
    class_updates = classify_waiting_for_evidence_turn(current_state)
    current_state.update(class_updates)

    if route_waiting_for_evidence_turn(current_state) != "extract_experiment_evidence":
        resp_updates = respond_to_waiting_founder(current_state)
        current_state.update(resp_updates)
        return current_state

    # 1. Extract evidence
    ext_updates = extract_experiment_evidence(current_state)
    current_state.update(ext_updates)

    # 2. Assess sufficiency
    suff_updates = assess_evidence_submission_sufficiency(current_state)
    current_state.update(suff_updates)

    # 3. Analyze evidence
    ana_updates = analyze_evidence(current_state)
    current_state.update(ana_updates)

    # 4. Validate analysis & update assumption / experiment
    val_updates = validate_evidence_analysis(current_state)
    current_state.update(val_updates)

    # Milestone 4.1.2: If partial evidence was submitted, do not increment iteration or re-diagnose; return immediately
    if current_state.get("is_partial_evidence"):
        return current_state

    # 5. Apply bounded state updates
    st_updates = apply_validated_state_updates(current_state)
    current_state.update(st_updates)

    # 6. Increment iteration
    iter_updates = increment_iteration(current_state)
    current_state.update(iter_updates)

    # 7. Re-diagnose
    diag_updates = diagnose_startup(current_state)
    current_state.update(diag_updates)
    merge_updates = merge_diagnosis(current_state)
    current_state.update(merge_updates)
    areas_updates = assess_startup_areas(current_state)
    current_state.update(areas_updates)
    wl_updates = select_weak_link(current_state)
    current_state.update(wl_updates)

    # 8. Reality check & Critical assumption
    rc_updates = reality_check(current_state)
    current_state.update(rc_updates)
    ca_updates = generate_critical_assumption(current_state)
    current_state.update(ca_updates)
    save_ca_updates = save_critical_assumption(current_state)
    current_state.update(save_ca_updates)

    # 9. Learning delta & Startup diagnosis snapshot
    delta_updates = compose_learning_delta(current_state)
    current_state.update(delta_updates)
    snap_updates = compose_startup_diagnosis(current_state)
    current_state.update(snap_updates)
    ans_updates = answer_post_evidence_question(current_state)
    current_state.update(ans_updates)
    br_updates = assess_build_readiness(current_state)
    current_state.update(br_updates)
    strat_updates = decide_next_strategic_action(current_state)
    current_state.update(strat_updates)

    return current_state


def resume_post_evidence_diagnosis(state: StartaryGraphState) -> StartaryGraphState:
    """
    Direct development & testing entry point (Milestone 4.1.2.2):
    Resumes reasoning after a post-evidence transient failure (e.g. diagnose_startup timeout).
    Idempotently continues from diagnose_startup without:
    - re-admitting evidence
    - duplicating evidence_history
    - re-running state updates
    - incrementing iteration a second time
    - duplicating LearningDelta
    """
    current_state = dict(state)
    diag_updates = diagnose_startup(current_state)
    current_state.update(diag_updates)
    merge_updates = merge_diagnosis(current_state)
    current_state.update(merge_updates)
    areas_updates = assess_startup_areas(current_state)
    current_state.update(areas_updates)
    wl_updates = select_weak_link(current_state)
    current_state.update(wl_updates)
    rc_updates = reality_check(current_state)
    current_state.update(rc_updates)
    ca_updates = generate_critical_assumption(current_state)
    current_state.update(ca_updates)
    save_ca_updates = save_critical_assumption(current_state)
    current_state.update(save_ca_updates)
    delta_updates = compose_learning_delta(current_state)
    current_state.update(delta_updates)
    snap_updates = compose_startup_diagnosis(current_state)
    current_state.update(snap_updates)
    ans_updates = answer_post_evidence_question(current_state)
    current_state.update(ans_updates)
    br_updates = assess_build_readiness(current_state)
    current_state.update(br_updates)
    strat_updates = decide_next_strategic_action(current_state)
    current_state.update(strat_updates)
    return current_state






