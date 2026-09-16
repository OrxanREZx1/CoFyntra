"""
Startary local development entry point — Milestone 2.5.

M2.5 Conversational Understanding Loop:
  1. Founder enters initial startup context.
  2. LangGraph executes:
     - diagnose_startup + merge_diagnosis
     - assess_context_sufficiency
  3. If context is insufficient:
     - select_clarifying_question
     - interrupt & ask founder
     - founder answers -> merge -> re-diagnose -> re-evaluate sufficiency (up to 5 turns).
  4. Once sufficient:
     - assess_startup_areas + select_weak_link
     - reality_check
     - generate_critical_assumption + save_critical_assumption
  5. Full structured results displayed.
"""

from __future__ import annotations

from enum import Enum, StrEnum
import sys

from langgraph.types import Command
from pydantic import ValidationError

from startary.config import configure_logging, get_settings
from startary.domain.assumption import CriticalAssumptionGroundingError
from startary.domain.enums import ExperimentDesignStatus, WorkflowPhase
from startary.domain.project import StartupProject
from startary.domain.taxonomy import StartupArea
from startary.domain.terminal_state import (
    TerminalStatus,
    can_render_waiting_for_evidence,
    render_terminal_state,
    resolve_founder_facing_status,
)
from startary.graph.builder import build_graph
from startary.graph.graph_state import StartaryGraphState
from startary.llm.provider import is_token_budget_exhausted_error


class ErrorCategory(str, Enum):
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"
    PROVIDER_ERROR = "PROVIDER_ERROR"
    REASONING_ERROR = "REASONING_ERROR"
    TOKEN_BUDGET_EXHAUSTED = "TOKEN_BUDGET_EXHAUSTED"
    UNEXPECTED_ERROR = "UNEXPECTED_ERROR"
    TIMEOUT = "TIMEOUT"
    RATE_LIMIT = "RATE_LIMIT"
    INSUFFICIENT_QUOTA = "INSUFFICIENT_QUOTA"
    AUTHENTICATION = "AUTHENTICATION"
    NETWORK = "NETWORK"

    def __eq__(self, other):
        other_val = getattr(other, "value", str(other))
        if other_val == "PROVIDER_ERROR":
            return self.value in ("TIMEOUT", "RATE_LIMIT", "INSUFFICIENT_QUOTA", "AUTHENTICATION", "NETWORK", "TOKEN_BUDGET_EXHAUSTED", "PROVIDER_ERROR")
        return self.value == other_val

    def __hash__(self):
        return hash(self.value)


def classify_cli_error(exc: Exception, state: dict[str, Any] | None = None) -> tuple[ErrorCategory, str, str]:
    """
    Classify a CLI/graph execution exception into actionable operational categories (Milestone 4.0.7.1.2, 4.1.2.2 & 4.1.3).

    Returns:
        (category, title, guidance)
    """
    exc_str = str(exc)
    exc_type_name = type(exc).__name__
    exc_module = getattr(type(exc), "__module__", "")

    # 0. TOKEN_BUDGET_EXHAUSTED (Milestone 4.1.4.1)
    if is_token_budget_exhausted_error(exc):
        return (
            ErrorCategory.TOKEN_BUDGET_EXHAUSTED,
            "AI Provider Token Budget Exhausted",
            "The model's completion token budget (reasoning + output) was exhausted before completing the response. "
            "The system attempted recovery with an expanded budget. "
            "Please retry with /retry or submit a more concise update.",
        )

    # 1. REASONING_ERROR: Internal grounding, validation, invariant, or domain constraint failures
    if (
        isinstance(exc, CriticalAssumptionGroundingError)
        or "grounding validation" in exc_str.lower()
        or "failed grounding" in exc_str.lower()
        or "quality bar" in exc_str.lower()
    ):
        return (
            ErrorCategory.REASONING_ERROR,
            "Reasoning & Grounding Validation Failure",
            "An internal reasoning or grounding validation constraint was not met. "
            "This is an internal reasoning failure, not an OpenAI API key or configuration issue.",
        )

    # 2. CONFIGURATION_ERROR: Missing or placeholder API key, invalid local env
    if (
        ("openai_api_key" in exc_str.lower() and "not configured" in exc_str.lower())
        or "paste_your_openai_api_key_here" in exc_str
        or "api key is not configured" in exc_str.lower()
        or ("api_key" in exc_str.lower() and "missing" in exc_str.lower())
    ):
        return (
            ErrorCategory.CONFIGURATION_ERROR,
            "Configuration Error",
            "Ensure OPENAI_API_KEY is configured in .env.",
        )

    # 3. Specific Provider Error Classifications (Milestone 4.1.2.2 & 4.1.3)
    # A. TIMEOUT (Stage-Aware Messaging — Milestone 4.1.3)
    is_conn_error = (
        isinstance(exc, ConnectionError)
        or exc_type_name in ("APIConnectionError", "ConnectionError", "ConnectError")
    )
    is_timeout = not is_conn_error and (
        isinstance(exc, TimeoutError)
        or exc_type_name in ("APITimeoutError", "TimeoutError", "TimeoutException", "ReadTimeout", "ConnectTimeout")
        or "request timed out" in exc_str.lower()
        or "timed out" in exc_str.lower()
        or ("timeout" in exc_str.lower() and "connect" not in exc_str.lower())
    )
    if is_timeout:
        exc_str_lower = exc_str.lower()
        current_node = (state.get("current_node") or "") if state else ""
        has_evidence = False
        if state:
            has_evidence = bool(state.get("evidence_history")) or bool(state.get("founder_evidence")) or bool(state.get("evidence_analysis"))

        if "generate_experiment" in exc_str_lower or current_node == "generate_experiment":
            timeout_title = "STARTARY — EXPERIMENT GENERATION TEMPORARILY UNAVAILABLE"
            timeout_guidance = (
                "==================================================\n"
                "STARTARY — EXPERIMENT GENERATION TEMPORARILY UNAVAILABLE\n"
                "==================================================\n\n"
                "The AI request timed out while designing your next experiment.\n"
                "Your startup state and diagnosis are safe.\n"
                "No new experiment was created. No experiment has been created yet.\n\n"
                "Retry experiment generation to continue.\n"
                "Type /retry to resume.\n"
                "=================================================="
            )
        elif "extract_startup_state" in exc_str_lower or current_node == "extract_startup_state":
            timeout_title = "STARTARY — COULD NOT PROCESS THIS UPDATE"
            timeout_guidance = (
                "==================================================\n"
                "STARTARY — COULD NOT PROCESS THIS UPDATE\n"
                "==================================================\n\n"
                "The AI request timed out while interpreting your latest answer.\n"
                "Your previous startup state is safe.\n"
                "Your previous state is safe.\n\n"
                "Retry processing this answer to continue.\n"
                "Type /retry to resume.\n"
                "=================================================="
            )
        elif has_evidence or (state and state.get("iteration", 1) > 1):
            timeout_title = "STARTARY — DIAGNOSIS TEMPORARILY UNAVAILABLE"
            timeout_guidance = (
                "==================================================\n"
                "STARTARY — DIAGNOSIS TEMPORARILY UNAVAILABLE\n"
                "==================================================\n\n"
                "The next diagnosis timed out.\n"
                "Your completed evidence has been saved. Your evidence was already saved and applied.\n"
                "You do not need to repeat or resubmit anything.\n\n"
                "Retry diagnosis to continue from the current startup state.\n"
                "Type /retry to resume.\n"
                "=================================================="
            )
        else:
            timeout_title = "STARTARY — DIAGNOSIS TEMPORARILY UNAVAILABLE"
            timeout_guidance = (
                "==================================================\n"
                "STARTARY — DIAGNOSIS TEMPORARILY UNAVAILABLE\n"
                "==================================================\n\n"
                "The AI request timed out while analyzing your startup context.\n"
                "Your startup state is safe.\n\n"
                "Retry diagnosis to continue.\n"
                "Type /retry to resume.\n"
                "=================================================="
            )
        return (
            ErrorCategory.TIMEOUT,
            timeout_title,
            timeout_guidance,
        )

    # B. INSUFFICIENT_QUOTA
    is_quota = (
        "insufficient_quota" in exc_str.lower()
        or "quota exceeded" in exc_str.lower()
        or "exceeded your current quota" in exc_str.lower()
    )
    if is_quota:
        return (
            ErrorCategory.INSUFFICIENT_QUOTA,
            "AI Provider Quota Exceeded",
            "You have exceeded your current OpenAI API quota or billing limit. "
            "Please check your billing and quota settings on platform.openai.com.",
        )

    # C. RATE_LIMIT
    is_rate_limit = (
        exc_type_name == "RateLimitError"
        or "rate limit" in exc_str.lower()
        or "429" in exc_str
    )
    if is_rate_limit:
        return (
            ErrorCategory.RATE_LIMIT,
            "AI Provider Rate Limit Exceeded",
            "OpenAI API rate limit reached (HTTP 429). Please wait a moment before retrying.",
        )

    # D. AUTHENTICATION
    is_auth = (
        exc_type_name == "AuthenticationError"
        or "401" in exc_str
        or "authentication" in exc_str.lower()
        or "incorrect api key" in exc_str.lower()
        or "invalid api key" in exc_str.lower()
    )
    if is_auth:
        return (
            ErrorCategory.AUTHENTICATION,
            "AI Provider Authentication Error",
            "OpenAI rejected the API key or authentication failed (HTTP 401). "
            "Check that your OPENAI_API_KEY in .env is active and valid.",
        )

    # E. NETWORK
    is_network = (
        exc_type_name in ("APIConnectionError", "ConnectionError", "ConnectError")
        or "connection refused" in exc_str.lower()
        or "connection error" in exc_str.lower()
        or "network" in exc_str.lower()
    )
    if is_network:
        return (
            ErrorCategory.NETWORK,
            "AI Provider Network Connection Error",
            "Network connection to OpenAI API failed or was interrupted. "
            "Check your internet connection and OpenAI service status.",
        )

    # F. Generic PROVIDER_ERROR
    is_provider_module = exc_module.startswith("openai") or exc_module.startswith("langchain_openai")
    is_provider_name = exc_type_name in {
        "APIError",
        "InternalServerError",
        "ServiceUnavailableError",
        "BadRequestError",
    }
    if is_provider_module or is_provider_name or "openai" in exc_str.lower():
        return (
            ErrorCategory.PROVIDER_ERROR,
            "AI Provider Service Error",
            "OpenAI API encountered an unexpected service error. "
            "Check internet connection and OpenAI service status.",
        )

    # 4. Other validation / assertion errors are internal reasoning errors
    if isinstance(exc, (AssertionError, ValidationError)):
        return (
            ErrorCategory.REASONING_ERROR,
            "Data Validation Invariant Error",
            "An internal schema or domain invariant failed during graph execution.",
        )

    # 5. UNEXPECTED_ERROR: Fallback for all other exceptions
    return (
        ErrorCategory.UNEXPECTED_ERROR,
        "Unexpected System Error",
        "An unexpected error occurred during execution. Inspect logs for trace details.",
    )


_BANNER = """
+------------------------------------------------------+
|        STARTARY AI SYSTEM -- DEVELOPMENT MODE        |
|                    Milestone 2.5                     |
+------------------------------------------------------+
"""

_WIDE = "=" * 54
_THIN = "-" * 54


def _print_area_section(area_name: str, knowledge_dict: dict) -> None:
    """Print one startup area section from a serialized AreaKnowledge dict."""
    status = knowledge_dict.get("status", "UNKNOWN")
    understanding = knowledge_dict.get("understanding")
    claims = knowledge_dict.get("founder_claims", [])
    evidence = knowledge_dict.get("evidence_provided", [])
    missing = knowledge_dict.get("missing_information")

    print(f"\n{area_name.upper()}")
    print(f"Status: {status}")

    if understanding:
        print(f"Understanding:\n  {understanding}")

    if claims:
        print("Founder claims (not yet evidenced):")
        for claim in claims:
            print(f"  - {claim}")

    if evidence:
        print("Evidence:")
        for ev in evidence:
            print(f"  - {ev}")

    if missing:
        print(f"Missing:\n  {missing}")

    if status == "UNKNOWN":
        print("  [No information provided for this area]")

    print(_THIN)


def _print_diagnosis_output(final_state: StartaryGraphState) -> None:
    """Print the full structured diagnosis from final graph state."""
    diagnosis_dict = final_state.get("diagnosis")
    startup_state_dict = final_state.get("startup_state", {})
    areas_dict = startup_state_dict.get("areas", {})

    print(f"\n{_WIDE}")
    print("STARTARY -- STRUCTURED STARTUP STATE")
    print(_WIDE)

    if not areas_dict:
        print("\n  No startup state could be built.")
        return

    for area in StartupArea:
        area_key = str(area)
        knowledge = areas_dict.get(area_key, {"status": "UNKNOWN"})
        _print_area_section(str(area), knowledge)

    if diagnosis_dict:
        print(f"\n{_WIDE}")
        print("DIAGNOSIS SUMMARY")
        print(_WIDE)

        summary = diagnosis_dict.get("summary")
        if summary:
            print(f"\n{summary}")

        stage = diagnosis_dict.get("startup_stage")
        if stage:
            print(f"\nInferred stage: {stage}")

        clear = diagnosis_dict.get("clear_areas", [])
        if clear:
            print(f"\nRelatively clear areas: {', '.join(clear)}")

        unclear = diagnosis_dict.get("unclear_areas", [])
        if unclear:
            print(f"Unclear / ambiguous areas: {', '.join(unclear)}")

        missing = diagnosis_dict.get("missing_areas", [])
        if missing:
            print(f"No information provided: {', '.join(missing)}")

        claims = diagnosis_dict.get("unsupported_claims", [])
        if claims:
            print("\nUnsupported founder claims:")
            for c in claims:
                print(f"  - {c}")


def _print_weak_link_assessment(final_state: StartaryGraphState) -> None:
    """Print the Weak-Link assessment table and winning weak area."""
    area_assessments = final_state.get("area_assessments", [])
    weak_details = final_state.get("weak_link_details", {})
    weak_area = final_state.get("weak_area")

    print(f"\n{_WIDE}")
    print("WEAK-LINK RISK ASSESSMENT")
    print(_WIDE)

    if not area_assessments:
        print("No area assessments available.")
        return

    print(f"{'Area':<30} | {'Imp':<5} | {'Unc':<5} | {'Gap':<5} | {'Con':<5} | {'Score':<5}")
    print("-" * 65)

    for item in area_assessments:
        area = item.get("area", "")
        imp = item.get("importance", "LOW")[:3]
        unc = item.get("uncertainty", "LOW")[:3]
        gap = item.get("evidence_gap", "LOW")[:3]
        con = item.get("consequence_if_wrong", "LOW")[:3]
        score = item.get("calculated_decision_risk", 1)

        prefix = "-> " if area == weak_area else "   "
        print(f"{prefix}{area:<27} | {imp:<5} | {unc:<5} | {gap:<5} | {con:<5} | {score:<5}")

    print("-" * 65)

    if weak_area:
        print(f"\nSELECTED WEAK LINK: {weak_area.upper()}")
        print(f"Decision Risk Score : {weak_details.get('score')} / 81")
        print(f"Selection Confidence: {final_state.get('selection_confidence', 'UNKNOWN')}")
        competing = final_state.get("competing_weak_areas", [])
        if competing:
            print(f"Competing Risks     : {', '.join(competing)}")
        print(f"Decision Rationale  : {weak_details.get('rationale')}")


def _print_reality_check(final_state: StartaryGraphState) -> None:
    """Print the Reality Check analysis for the weak link."""
    rc = final_state.get("reality_check")
    if not rc:
        return

    print(f"\n{_WIDE}")
    print("REALITY CHECK")
    print(_WIDE)

    print(f"Area       : {rc.get('weak_link_area')}")
    print(f"Conclusion : {rc.get('conclusion')}")

    claims = rc.get("relevant_founder_claims", [])
    if claims:
        print("\nFounder Beliefs / Claims:")
        for c in claims:
            print(f"  - {c}")

    evidence = rc.get("available_founder_evidence", [])
    if evidence:
        print("\nAvailable Founder Evidence:")
        for e in evidence:
            print(f"  - {e}")

    supports = rc.get("what_evidence_supports", [])
    if supports:
        print("\nEvidence SUPPORTS:")
        for s in supports:
            print(f"  - {s}")

    not_supports = rc.get("what_evidence_does_not_support", [])
    if not_supports:
        print("\nEvidence DOES NOT Support:")
        for ns in not_supports:
            print(f"  - {ns}")

    print(f"\nUnresolved Uncertainty:\n  {rc.get('unresolved_uncertainty')}")
    print(f"\nReality Check Summary:\n  {rc.get('summary')}")


def _print_critical_assumption(final_state: StartaryGraphState) -> None:
    """Print the generated Critical Assumption."""
    assumption = final_state.get("active_assumption")
    if not assumption:
        return

    print(f"\n{_WIDE}")
    print("CRITICAL ASSUMPTION")
    print(_WIDE)

    print(f"Category   : {assumption.get('area')}")
    print(f"Assumption : {assumption.get('statement')}")
    print(f"Status     : {assumption.get('status')}")
    print(f"Critical   : {'YES' if assumption.get('is_critical') else 'NO'}")
    print(f"Phase      : {final_state.get('phase')}")
    print(_WIDE)


def _print_startup_profile(final_state: StartaryGraphState) -> None:
    """Print the concise Startup Profile (Milestone 2.6)."""
    exec_ctx = final_state.get("execution_context") or {}
    prog = final_state.get("startup_progress") or {}

    print(f"\n{_WIDE}")
    print("STARTUP PROFILE")
    print(_WIDE)

    def _fmt(field: Any, default_label: str = "Unknown") -> str:
        if not isinstance(field, dict):
            return default_label
        val = field.get("value")
        prov = field.get("provenance", "UNKNOWN")
        if val is None or prov == "UNKNOWN":
            return default_label
        return f"{val} / {prov.lower()}"

    # Development stage
    stage_str = _fmt(prog.get("development_stage"))

    # MVP
    mvp_field = prog.get("mvp_exists")
    if isinstance(mvp_field, dict) and mvp_field.get("value") is not None and mvp_field.get("provenance") != "UNKNOWN":
        mvp_val = "Yes" if str(mvp_field.get("value")).lower() == "true" else "No"
        mvp_str = f"{mvp_val} / {mvp_field.get('provenance', 'UNKNOWN').lower()}"
    else:
        mvp_str = "Unknown"

    # Team
    team_field = exec_ctx.get("active_team_size") or exec_ctx.get("founder_count")
    if isinstance(team_field, dict) and team_field.get("value") is not None and team_field.get("provenance") != "UNKNOWN":
        team_str = f"{team_field.get('value')} active members / {team_field.get('provenance', 'UNKNOWN').lower()}"
    else:
        team_str = "Unknown"

    # Capacity
    cap_field = exec_ctx.get("total_weekly_capacity_hours")
    if isinstance(cap_field, dict) and cap_field.get("value") is not None and cap_field.get("provenance") != "UNKNOWN":
        cap_str = f"~{cap_field.get('value')} h/week / {cap_field.get('provenance', 'UNKNOWN').lower()}"
    else:
        avail_field = exec_ctx.get("approximate_weekly_hours_per_member")
        if isinstance(avail_field, dict) and avail_field.get("value") is not None and avail_field.get("provenance") != "UNKNOWN":
            cap_str = f"~{avail_field.get('value')} h/member/week / {avail_field.get('provenance', 'UNKNOWN').lower()}"
        else:
            cap_str = "Unknown"

    # Budget
    budget_amt = exec_ctx.get("available_budget_amount")
    if isinstance(budget_amt, dict) and budget_amt.get("value") is not None and budget_amt.get("provenance") != "UNKNOWN":
        curr = (exec_ctx.get("budget_currency") or {}).get("value") or ""
        period = (exec_ctx.get("budget_period") or {}).get("value") or ""
        amt = budget_amt.get("value")
        suffix = f"/{period}" if period else ""
        prefix_curr = f" {curr}" if curr else ""
        budget_str = f"~{amt}{prefix_curr}{suffix} / {budget_amt.get('provenance', 'UNKNOWN').lower()}"
    else:
        budget_str = "Unknown"

    # Funding
    funding_str = _fmt(exec_ctx.get("funding_status"))

    # Users
    users_str = _fmt(prog.get("active_users"))

    # Revenue
    rev_str = _fmt(prog.get("revenue"))

    # Customer access
    cust_access = exec_ctx.get("customer_access") or exec_ctx.get("existing_audience_or_community")
    access_str = _fmt(cust_access)

    print(f"Development Stage : {stage_str}")
    print(f"MVP               : {mvp_str}")
    print(f"Team              : {team_str}")
    print(f"Capacity          : {cap_str}")
    print(f"Budget            : {budget_str}")
    print(f"Funding           : {funding_str}")
    print(f"Users             : {users_str}")
    print(f"Revenue           : {rev_str}")
    print(f"Customer Access   : {access_str}")


def _print_diagnosis_snapshot(final_state: StartaryGraphState) -> None:
    """Print the concise, founder-facing Startup Diagnosis Snapshot (Milestone 3.1)."""
    snapshot = final_state.get("diagnosis_snapshot")
    if not snapshot:
        return

    print(f"\n{_WIDE}")
    print("STARTUP DIAGNOSIS")
    print(_WIDE)

    print("\nCURRENT STAGE")
    print(snapshot.get("stage", "Unknown"))
    if snapshot.get("stage_explanation"):
        print(f"\n{snapshot.get('stage_explanation')}")

    grounded = snapshot.get("grounded_areas") or []
    if grounded:
        print(f"\n{_THIN}")
        print("\nWHAT WE KNOW\n")
        for g in grounded:
            print(f"[x] {g}")

    advantages = snapshot.get("current_advantages") or []
    print(f"\n{_THIN}")
    print("\nCURRENT ADVANTAGES\n")
    if advantages:
        for a in advantages:
            print(f"- {a}")
    else:
        print("- None identified at this early stage.")

    constraints = snapshot.get("execution_constraints") or []
    if constraints:
        print(f"\n{_THIN}")
        print("\nEXECUTION CONSTRAINTS\n")
        for c in constraints:
            print(f"- {c}")

    uncertainties = snapshot.get("major_uncertainties") or []
    if uncertainties:
        print(f"\n{_THIN}")
        print("\nMAJOR UNCERTAINTIES\n")
        for u in uncertainties:
            print(f"- {u}")

    print(f"\n{_THIN}")
    print("\nCURRENT WEAK LINK")
    print(snapshot.get("weak_link", "Problem"))
    if snapshot.get("weak_link_reason"):
        print(f"\n{snapshot.get('weak_link_reason')}")

    print(f"\nConfidence:\n{snapshot.get('weak_link_confidence', 'LOW')}")
    if snapshot.get("weak_link_confidence_explanation"):
        print(f"\n{snapshot.get('weak_link_confidence_explanation')}")

    competing = snapshot.get("competing_risks") or []
    if competing:
        print("\nOther Important Risks:")
        for r in competing:
            print(f"- {r}")

    if snapshot.get("reality_summary"):
        print(f"\n{_THIN}")
        print("\nREALITY CHECK\n")
        print(snapshot.get("reality_summary"))

    if snapshot.get("recommended_learning_goal"):
        print(f"\n{_THIN}")
        print("\nWHAT WE NEED TO LEARN NEXT\n")
        print(snapshot.get("recommended_learning_goal"))

    if snapshot.get("critical_assumption"):
        print(f"\n{_THIN}")
        print("\nCRITICAL ASSUMPTION\n")
        print(snapshot.get("critical_assumption"))

    print(f"\n{_WIDE}")


def _print_experiment(final_state: StartaryGraphState) -> None:
    """Print the final Experiment, Critic report, and Execution Readiness (Milestone 3)."""
    exp = final_state.get("active_experiment")
    if not exp:
        return

    print(f"\n{_WIDE}")
    print("PROPOSED EXPERIMENT")
    print(_WIDE)

    print(f"Assumption         : {exp.get('hypothesis') or (final_state.get('active_assumption') or {}).get('statement')}")
    print(f"Experiment Type    : {exp.get('experiment_type')}")
    print(f"Objective          : {exp.get('objective')}")
    print(f"Method             : {exp.get('method')}")
    scope = exp.get("target_participants_or_scope")
    if scope:
        print(f"Target / Scope     : {scope}")

    actions = exp.get("founder_actions") or []
    if actions:
        print("\nFounder Actions:")
        for idx, act in enumerate(actions, 1):
            print(f"  {idx}. {act}")

    print(f"\nEvidence to Collect:\n  {exp.get('evidence_to_collect')}")

    valid_inf = exp.get("valid_inferences") or []
    if valid_inf:
        print("\nWhat This CAN Tell Us:")
        for vi in valid_inf:
            print(f"  - {vi}")

    invalid_inf = exp.get("invalid_inferences") or []
    if invalid_inf:
        print("\nWhat This CANNOT Tell Us:")
        for ii in invalid_inf:
            print(f"  - {ii}")

    print(f"\nSupport Signal     :\n  {exp.get('support_signal')}")
    print(f"\nContradiction Signal:\n  {exp.get('contradiction_signal')}")
    print(f"\nInconclusive Signal:\n  {exp.get('inconclusive_signal')}")

    print(f"\nEstimated Time     : {exp.get('estimated_time') or exp.get('time_constraint')}")
    print(f"Estimated Cost     : {exp.get('estimated_cost') or exp.get('cost_constraint')}")

    # Parameters if any
    params = exp.get("parameters") or []
    if params:
        print("\nExperiment Parameters:")
        for p in params:
            prov = p.get("provenance", "AI_PROPOSED")
            print(f"  - {p.get('name')}: {p.get('value')} ({prov})")

    # Critic report
    critic = final_state.get("experiment_critic_output")
    if critic:
        print(f"\n{_WIDE}")
        print("EXPERIMENT CRITIC")
        print(_WIDE)
        print(f"Verdict            : {critic.get('verdict')}")
        print(f"Alignment          : {critic.get('assumption_alignment')}")
        print(f"Evidence Validity  : {critic.get('evidence_validity')}")
        print(f"Feasibility        : {critic.get('feasibility')}")
        print(f"Cost Efficiency    : {critic.get('cost_efficiency')}")
        print(f"Falsifiability     : {critic.get('falsifiability')}")
        print(f"Scope              : {critic.get('scope')}")
        issues = critic.get("issues") or []
        if issues:
            print("\nIssues Identified:")
            for issue in issues:
                print(f"  - {issue}")
        if critic.get("rationale"):
            print(f"\nCritic Rationale   : {critic.get('rationale')}")

    # Execution Readiness
    can_exec = final_state.get("can_execute_result")
    if can_exec:
        print(f"\n{_WIDE}")
        print("EXECUTION READINESS")
        print(_WIDE)
        print(f"Status : {can_exec.get('readiness')}")
        print(f"Reason : {can_exec.get('reason')}")
        missing = can_exec.get("missing_requirements") or []
        if missing:
            print("\nMissing Requirements / Support Needed:")
            for m in missing:
                print(f"  - {m}")
        blockers = can_exec.get("blockers") or []
        if blockers:
            print("\nBlockers:")
            for b in blockers:
                print(f"  - {b}")
        print(_WIDE)


def read_founder_input(prompt: str = "> ") -> str:
    """
    Reads founder input, supporting normal single-line input or multiline paste mode.
    If the user types '/paste', enters multiline mode and collects lines until 'END'.
    """
    try:
        first_line = input(prompt)
    except (EOFError, KeyboardInterrupt):
        return ""

    if first_line.strip().lower() == "/paste":
        print("Entering multiline mode. Paste your text, then type END on a new line:")
        lines = []
        while True:
            try:
                line = input()
            except (EOFError, KeyboardInterrupt):
                break
            if line.strip() == "END":
                break
            lines.append(line)
        return "\n".join(lines).strip()

    return first_line.strip()


def check_models_diagnostic(settings: Any) -> None:
    """
    Developer diagnostic CLI utility (Milestone M4.1.3.1).
    Checks access for FAST and REASONING configured and fallback models,
    printing a clean diagnostic table without exposing API keys.
    """
    from startary.llm.provider import check_model_access, is_valid_api_key

    api_key = getattr(settings, "openai_api_key", None) or ""
    if not is_valid_api_key(api_key):
        print("ERROR: OpenAI API key is not configured.")
        print("  Open the project .env file and configure OPENAI_API_KEY.\n")
        sys.exit(1)

    fast_cfg = getattr(settings, "fast_model_name", None) or "gpt-4o-mini"
    fast_fb = getattr(settings, "fast_model_fallback", None) or "gpt-4o-mini"
    reasoning_cfg = getattr(settings, "reasoning_model_name", None) or getattr(settings, "model_name", "gpt-4o")
    reasoning_fb = getattr(settings, "reasoning_model_fallback", None) or getattr(settings, "model_name", "gpt-4o")

    rows = [
        ("FAST", "configured", fast_cfg),
        ("FAST", "fallback", fast_fb),
        ("REASONING", "configured", reasoning_cfg),
        ("REASONING", "fallback", reasoning_fb),
    ]

    print(f"\n{'Tier':<10} {'Role':<11} {'Model Name':<15} {'Accessible':<11} {'Status/Error'}")
    print("-" * 75)

    for tier, role, model in rows:
        avail, reason = check_model_access(api_key, model)
        accessible_str = "YES" if avail else "NO"
        status_str = "OK" if avail else (reason or "Unavailable")
        print(f"{tier:<10} {role:<11} {model:<15} {accessible_str:<11} {status_str}")
    print()


def main() -> None:
    """Run the Startary local development prototype (Milestone 2.6)."""
    settings = get_settings()
    configure_logging(settings.log_level)

    if "--check-models" in sys.argv:
        check_models_diagnostic(settings)
        sys.exit(0)

    print(_BANNER)

    from startary.llm.provider import is_valid_api_key, run_model_preflight

    if not is_valid_api_key(settings.openai_api_key):
        print("ERROR: OpenAI API key is not configured.")
        print("  Open the project .env file and replace PASTE_YOUR_OPENAI_API_KEY_HERE with your API key.\n")
        sys.exit(1)

    # Preflight model access check (Milestone M4.1.3.1)
    preflight = run_model_preflight(settings, probe=True)
    if not preflight.is_ready:
        failed_model = preflight.fast_configured if preflight.fast_status != "AVAILABLE" and not preflight.fast_resolved else preflight.reasoning_configured
        print("\n==================================================")
        print("STARTARY — MODEL CONFIGURATION ERROR")
        print("==================================================")
        print(f"The configured model '{failed_model}' is not accessible with this API key/project and no usable fallback succeeded.")
        print("Please verify model access or update FAST_MODEL_NAME / REASONING_MODEL_NAME in .env.\n")
        sys.exit(1)

    # ── Project creation ───────────────────────────────────────────────────
    print("Enter your startup name (or press Enter for 'My Startup'):")
    try:
        raw_title = input("> ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nAborted.")
        sys.exit(0)

    title = raw_title if raw_title else "My Startup"
    project = StartupProject(title=title, phase=WorkflowPhase.UNDERSTAND, iteration=1)

    print(f"\nProject created.")
    print(f"  Project ID : {project.id}")
    print(f"  Phase      : {project.phase}")
    print(f"  Iteration  : {project.iteration}")
    print()

    # ── Raw startup context ────────────────────────────────────────────────
    print("Describe the startup, problem, founder situation, and anything you already know.")
    print("(Be as detailed as you like. Press Enter when done.)")
    print("Type /paste for multiline input.\n")

    print("Enter startup context:")
    raw_context = read_founder_input("> ")

    if not raw_context:
        print("\nNo context provided. Exiting.")
        sys.exit(0)

    print()
    print("Analyzing startup context...")
    print()
    from langgraph.checkpoint.memory import MemorySaver
    graph = build_graph(checkpointer=MemorySaver())
    thread_config = {"configurable": {"thread_id": project.id}}

    initial_state: StartaryGraphState = {
        "project_id": project.id,
        "phase": project.phase,
        "iteration": project.iteration,
        "raw_context": raw_context,
        "founder_messages": [{"role": "founder", "content": raw_context}],
        "clarification_turn_count": 0,
        "strategic_clarification_turns": 0,
        "baseline_question_turns": 0,
        "weak_link_clarification_turns": 0,
        "clarification_history": [],
        "baseline_history": [],
        "weak_link_clarification_history": [],
    }

    try:
        graph.invoke(initial_state, thread_config)

        last_interrupt_type = None

        # Handle Conversational Interrupt Loop across the 3 distinct activities
        while True:
            state_snapshot = graph.get_state(thread_config)

            # Check if execution paused at an interrupt
            if state_snapshot.tasks and state_snapshot.tasks[0].interrupts:
                interrupt_val = state_snapshot.tasks[0].interrupts[0].value
                question_text = None
                interrupt_type = "strategic"
                show_full_experiment = "--full-experiment" in sys.argv
                if isinstance(interrupt_val, dict):
                    if show_full_experiment and interrupt_val.get("execution_handoff"):
                        question_text = (
                            interrupt_val["execution_handoff"].get("full_response_text")
                            or interrupt_val.get("handoff_text")
                            or interrupt_val.get("question")
                        )
                    else:
                        question_text = (
                            interrupt_val.get("handoff_text")
                            or interrupt_val.get("question")
                            or interrupt_val.get("message")
                        )
                    interrupt_type = interrupt_val.get("interrupt_type") or interrupt_val.get("type", "strategic")
                elif interrupt_val:
                    question_text = str(interrupt_val)

                if not question_text or question_text == "None":
                    state_values = state_snapshot.values or {}
                    handoff = state_values.get("active_execution_handoff") or {}
                    if show_full_experiment:
                        question_text = handoff.get("full_response_text") or handoff.get("response_text") or state_values.get("status_message") or "Ready for your input."
                    else:
                        question_text = handoff.get("response_text") or state_values.get("status_message") or "Ready for your input."

                if interrupt_type != last_interrupt_type:
                    if interrupt_type == "baseline":
                        print(f"\nBefore we go deeper, I need a quick picture of where {title} is today.")
                    elif interrupt_type == "strategic":
                        print("\nThanks. Now I want to understand the core problem and customer situation.")
                    elif interrupt_type == "weak_link":
                        print("\nLet's look closer at the biggest risks to prioritize the right test.")
                    last_interrupt_type = interrupt_type

                if str(interrupt_type).lower() in ("waiting_for_evidence", "waiting_for_learning_input"):
                    print(f"\n{question_text}\n")
                else:
                    print(f"\nStartary:\n{question_text}\n")

                print("Type /paste for multiline input.")
                founder_reply = read_founder_input("> ")

                if not founder_reply:
                    print("\nNo response provided. Exiting conversation.")
                    sys.exit(0)

                print("\nIntegrating your answer and re-evaluating...")
                try:
                    graph.invoke(Command(resume=founder_reply), thread_config)
                except Exception as invoke_exc:
                    cur_state = None
                    try:
                        cur_state = graph.get_state(thread_config).values
                    except Exception:
                        pass
                    category, title, guidance = classify_cli_error(invoke_exc, state=cur_state)
                    if category == ErrorCategory.TIMEOUT:
                        print(f"\n[{category}] {title}: {invoke_exc}")
                        print(f"\n{guidance}\n")
                        while True:
                            print("Type /retry to resume from current startup state, or press Enter to exit:")
                            retry_cmd = read_founder_input("> ")
                            if not retry_cmd:
                                sys.exit(1)
                            if retry_cmd.strip().lower() == "/retry":
                                print("\nRetrying operation from current state...")
                                try:
                                    graph.invoke(None, thread_config)
                                    break
                                except Exception as retry_exc:
                                    r_cat, r_title, r_guidance = classify_cli_error(retry_exc, state=cur_state)
                                    print(f"\n[{r_cat}] {r_title}: {retry_exc}")
                                    print(f"\n{r_guidance}\n")
                                    if r_cat != ErrorCategory.TIMEOUT:
                                        sys.exit(1)
                            else:
                                print(f"Unrecognized command: '{retry_cmd}'. Type /retry to resume or Enter to exit.")
                    else:
                        print(f"\n[{category}] {title}: {invoke_exc}")
                        print(f"\n{guidance}\n")
                        sys.exit(1)
            else:
                # Graph finished execution
                break

        final_state: StartaryGraphState = graph.get_state(thread_config).values

    except Exception as exc:
        category, title, guidance = classify_cli_error(exc)
        print(f"\n[{category}] {title}: {exc}")
        print(f"\n{guidance}\n")
        sys.exit(1)

    # ── Display results (Milestone 4.0.1 Centralized Rendering) ───────────
    terminal_output = render_terminal_state(final_state)
    print(terminal_output)

    terminal_status = resolve_founder_facing_status(final_state)
    if terminal_status == TerminalStatus.CLARIFICATION_LIMIT_REACHED:
        sys.exit(0)

    # Milestone 3.1: Print Founder-Facing Startup Diagnosis Snapshot if available
    if final_state.get("diagnosis_snapshot"):
        _print_diagnosis_snapshot(final_state)
    else:
        _print_startup_profile(final_state)
        _print_diagnosis_output(final_state)
        _print_weak_link_assessment(final_state)
        _print_reality_check(final_state)
        _print_critical_assumption(final_state)

    if (
        final_state.get("active_experiment")
        and not final_state.get("fallback_learning_action")
        and final_state.get("experiment_design_status") != ExperimentDesignStatus.QUALITY_BAR_NOT_MET
    ):
        _print_experiment(final_state)

    # Milestone 4: Print Learning Delta if present
    if final_state.get("learning_delta"):
        delta = final_state["learning_delta"]
        print(f"\n{_WIDE}")
        print("STARTARY — LEARNING DELTA")
        print(_WIDE)
        print(f"Outcome: {delta.get('evidence_conclusion')}")
        print(f"Summary: {delta.get('founder_facing_summary')}\n")

    # Milestone 4.0.7: Print Post-Evidence Advisor Answer if present
    if final_state.get("post_evidence_answer"):
        print(f"\n{_WIDE}")
        print("STARTARY — ADVISOR RESPONSE")
        print(_WIDE)
        print(f"\n{final_state['post_evidence_answer']}\n")

    print("\nStartary evaluation complete.\n")


if __name__ == "__main__":
    main()
