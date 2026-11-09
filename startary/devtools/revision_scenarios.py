"""
Seeded scenarios for the M6.2.6.1 revision probe (fictional startups; no real founder data).

Each scenario provides the minimum graph state the production experiment subgraph needs to start
right after generate_experiment, plus a deliberately SEMANTICALLY flawed draft that satisfies
ExperimentGeneratorLLMOutput and trips no deterministic contract check. Whether the flaw is real
enough to reject is left entirely to the live critic.
"""

from __future__ import annotations

import copy
from typing import Any

from startary.domain.execution_context import ExecutionContext
from startary.domain.startup_progress import StartupProgress

SHIFTRELAY_ASSUMPTION = {
    "id": "asmp-shiftrelay-absence-pain",
    "area": "Pain / Urgency",
    "statement": (
        "In independent cafés and small restaurants in South Korea employing around 8–25 workers, last-minute "
        "employee absences recur and create material operational disruption for managers."
    ),
}

SHIFTRELAY_FOUNDER_MESSAGE = (
    "We are ShiftRelay (fictional test scenario). We want to help independent cafés and small restaurants in South "
    "Korea cope with last-minute staff absences. We are two founders with about 8 hours each per week, a total budget "
    "of $350, and at most two weeks for this validation. We can talk to three café/restaurant managers and four "
    "part-time workers about real past incidents. We have no MVP or prototype, no paying customers and no revenue. "
    "We have no access to private schedules, payroll databases or confidential KakaoTalk messages."
)


def _fact(value: Any, provenance: str = "EXPLICIT", confidence: str = "HIGH") -> dict[str, Any]:
    return {"value": value, "provenance": provenance, "confidence": confidence, "source_turn": 0, "notes": None}


def _shiftrelay_facts() -> tuple[dict[str, Any], dict[str, Any]]:
    exec_ctx = ExecutionContext().model_dump()
    exec_ctx.update({
        "founder_count": _fact("2"),
        "active_team_size": _fact("2"),
        "approximate_weekly_hours_per_member": _fact("8"),
        "total_weekly_capacity_hours": _fact("16", provenance="INFERRED", confidence="MEDIUM"),
        "available_budget_amount": _fact("350"),
        "budget_currency": _fact("USD"),
        "customer_access": _fact("Three independent café/restaurant managers and four part-time workers willing to discuss past absence incidents"),
        "geographic_market": _fact("South Korea"),
    })
    progress = StartupProgress().model_dump()
    progress.update({
        "development_stage": _fact("IDEA", provenance="INFERRED", confidence="MEDIUM"),
        "prototype_exists": _fact("false"),
        "mvp_exists": _fact("false"),
        "launched": _fact("false"),
        "validation_activity": _fact("No structured interviews yet; interviews about real past incidents are possible"),
        "paying_customers": _fact("0"),
        "revenue": _fact("0"),
    })
    return exec_ctx, progress


_SHIFTRELAY_METHOD = (
    "Run 30–45 minute retrospective interviews with the three accessible managers. With each manager, reconstruct "
    "the real last-minute absences of the previous eight weeks one incident at a time: what happened, how a "
    "replacement was found, how much coordination time it took, and what operational consequences followed "
    "(service delays, overtime, closed sections). Keep anonymized written notes only; do not request schedules, "
    "payroll data or private messages."
)

# Primary fixture: sound interview method, but decision criteria measure hypothetical interest in an
# AI assistant instead of recurring, materially consequential absence pain.
SHIFTRELAY_DRAFT: dict[str, Any] = {
    "experiment_type": "PROBLEM_INTERVIEW",
    "objective": "Learn whether last-minute absences are a recurring and costly problem for small café and restaurant managers.",
    "hypothesis": SHIFTRELAY_ASSUMPTION["statement"],
    "method": _SHIFTRELAY_METHOD,
    "target_participants_or_scope": "The three accessible managers of independent cafés/small restaurants in South Korea (8–25 staff each); the four accessible part-time workers as an optional second perspective.",
    "target_sample_size": 3,
    "founder_actions": [
        "Schedule a 30–45 minute conversation with each of the three managers during the first week.",
        "Ask each manager to walk through the most recent last-minute absences of the past eight weeks, one incident at a time.",
        "For each incident, note how the gap was filled, how long coordination took, and what operational impact followed.",
        "At the end of each interview, describe ShiftRelay's AI scheduling assistant and ask whether it sounds useful.",
        "Optionally ask the four part-time workers how replacement requests usually reach them.",
        "Write anonymized incident notes after each interview; do not request schedules, payroll data or private messages.",
    ],
    "evidence_to_collect": "Anonymized notes per incident (how the replacement was found, coordination time, operational consequence) and each manager's answer on whether an AI scheduling assistant sounds useful.",
    "support_signal": "At least two of the three managers say an AI scheduling assistant sounds useful.",
    "contradiction_signal": "None of the three managers expresses interest in an AI scheduling assistant.",
    "inconclusive_signal": "Only one manager expresses interest in an AI scheduling assistant.",
    "valid_inferences": [
        "How these three managers currently handle last-minute absences.",
        "Which replacement-coordination steps managers describe as time-consuming.",
    ],
    "invalid_inferences": [
        "Willingness to pay for ShiftRelay.",
        "How common last-minute absences are across all South Korean cafés and restaurants.",
    ],
    "estimated_time": "1–2 weeks",
    "estimated_cost": "$30 (coffee for interviewees)",
    "required_capabilities": ["Structured retrospective interviewing"],
    "required_access": ["Three accessible café/restaurant managers", "Four accessible part-time workers (optional)"],
    "parameters": [
        {"name": "sample_size", "value": "3 managers", "provenance": "AI_PROPOSED", "notes": "All accessible managers."},
        {"name": "support_threshold", "value": "at least 2 of 3 managers say the assistant sounds useful", "provenance": "AI_PROPOSED", "notes": "Provisional decision rule."},
    ],
}

# Second predefined fixture (use only if the critic ACCEPTS the primary one): same method, but the
# criteria cannot meaningfully fail and never measure recurrence or material disruption.
SHIFTRELAY_B_DRAFT: dict[str, Any] = {
    **copy.deepcopy(SHIFTRELAY_DRAFT),
    "founder_actions": [a for a in SHIFTRELAY_DRAFT["founder_actions"] if "AI scheduling assistant" not in a],
    "evidence_to_collect": "Anonymized notes per incident: how the replacement was found, coordination time, and operational consequence.",
    "support_signal": "At least one manager recalls at least one last-minute absence in the past eight weeks.",
    "contradiction_signal": "No manager has ever experienced a last-minute absence.",
    "inconclusive_signal": "Managers recall absences but cannot say exactly when they happened.",
    "parameters": [
        {"name": "sample_size", "value": "3 managers", "provenance": "AI_PROPOSED", "notes": "All accessible managers."},
        {"name": "support_threshold", "value": "any recalled absence", "provenance": "AI_PROPOSED", "notes": "Provisional decision rule."},
    ],
}

# Third fixture (M6.2.6.2): a high-quality experiment with ONE narrow decision-rule defect. The
# contradiction rule ("fewer than 2") and the inconclusive rule ("exactly 1") overlap when exactly one
# manager qualifies. Method, participants, evidence source, budget and assumption are all appropriate,
# so the ideal repair edits only the contradiction/inconclusive signals.
SHIFTRELAY_C_QUALIFYING_PATTERN = (
    "2 or more last-minute absences in the past eight weeks, each with a concrete operational consequence "
    "(30+ minutes of replacement coordination, a short-staffed shift, delayed service, or unplanned overtime)"
)
SHIFTRELAY_C_DRAFT: dict[str, Any] = {
    "experiment_type": "PROBLEM_INTERVIEW",
    "objective": "Determine whether last-minute absences recur and cause concrete operational disruption for these café and restaurant managers.",
    "hypothesis": SHIFTRELAY_ASSUMPTION["statement"],
    "method": _SHIFTRELAY_METHOD,
    "target_participants_or_scope": "The three accessible managers of independent cafés/small restaurants in South Korea (8–25 staff each); the four accessible part-time workers as an optional second perspective.",
    "target_sample_size": 3,
    "founder_actions": [
        "Schedule a 30–45 minute conversation with each of the three managers within the first week.",
        "Ask each manager to walk through every last-minute absence of the past eight weeks, one incident at a time.",
        "For each incident, note how the absence was communicated, how the gap was filled, the coordination time, and the operational consequence.",
        "After each interview, decide whether that manager meets the qualifying pattern defined in the parameters.",
        "Optionally ask the four part-time workers how replacement requests usually reach them.",
        "Write anonymized incident notes after each interview; do not request schedules, payroll data or private messages.",
    ],
    "evidence_to_collect": "Anonymized notes per incident (week, how the absence was communicated, how the replacement was found, coordination minutes, operational consequence) and, per manager, whether the qualifying pattern is met.",
    "support_signal": f"At least 2 of the 3 managers meet the qualifying pattern: {SHIFTRELAY_C_QUALIFYING_PATTERN}.",
    "contradiction_signal": "Fewer than 2 of the 3 managers meet the qualifying pattern.",
    "inconclusive_signal": "Exactly 1 manager meets the qualifying pattern, or one or more of the three interviews cannot be completed.",
    "valid_inferences": [
        "Whether last-minute absences recur and cause concrete operational disruption for these three managers.",
        "Which consequences (coordination time, short-staffed shifts, service delays, overtime) are most common.",
    ],
    "invalid_inferences": [
        "How common last-minute absences are across all South Korean cafés and restaurants.",
        "Whether managers would adopt or pay for a scheduling product.",
    ],
    "estimated_time": "1–2 weeks",
    "estimated_cost": "$30 (coffee for interviewees)",
    "required_capabilities": ["Structured retrospective interviewing"],
    "required_access": ["Three accessible café/restaurant managers", "Four accessible part-time workers (optional)"],
    "parameters": [
        {"name": "qualifying_pattern", "value": SHIFTRELAY_C_QUALIFYING_PATTERN, "provenance": "AI_PROPOSED", "notes": "Applied per manager from reconstructed incidents."},
        {"name": "support_threshold", "value": "at least 2 of 3 managers meet the qualifying pattern", "provenance": "AI_PROPOSED", "notes": "Provisional decision rule."},
    ],
}

SCENARIOS: dict[str, dict[str, Any]] = {
    "shiftrelay": {
        "draft": SHIFTRELAY_DRAFT,
        "flaw": "Decision criteria measure hypothetical interest in an AI assistant, not recurring, material absence pain.",
    },
    "shiftrelay-b": {
        "draft": SHIFTRELAY_B_DRAFT,
        "flaw": "Criteria cannot meaningfully fail (any single recalled absence supports) and ignore recurrence and disruption.",
    },
    "shiftrelay-c": {
        "draft": SHIFTRELAY_C_DRAFT,
        "flaw": "Contradiction ('fewer than 2 qualify') and inconclusive ('exactly 1 qualifies') overlap when exactly one manager qualifies.",
    },
}


def scenario_state(name: str) -> dict[str, Any]:
    """Graph state as it would be immediately after generate_experiment produced the seeded draft."""
    scenario = SCENARIOS[name]
    exec_ctx, progress = _shiftrelay_facts()
    draft = copy.deepcopy(scenario["draft"])
    draft["assumption_id"] = SHIFTRELAY_ASSUMPTION["id"]  # code-owned stamp generate_experiment would add
    return {
        "project_id": f"revision-probe-{name}",
        "iteration": 1,
        "phase": "TEST",
        "active_assumption": dict(SHIFTRELAY_ASSUMPTION),
        "weak_area": SHIFTRELAY_ASSUMPTION["area"],
        "reality_check": {
            "summary": "No direct evidence yet on how often absences occur or how much disruption they cause.",
            "unresolved_uncertainty": "Whether absences recur and cause material disruption, not whether managers like a tool.",
            "what_evidence_supports": [],
            "what_evidence_does_not_support": ["No interviews completed", "No usage or payment evidence"],
        },
        "build_readiness": {"decision": "CONTINUE_MANUAL", "minimum_build_scope": [], "must_not_build": []},
        "execution_context": exec_ctx,
        "startup_progress": progress,
        "founder_messages": [{"role": "founder", "content": SHIFTRELAY_FOUNDER_MESSAGE}],
        "raw_context": SHIFTRELAY_FOUNDER_MESSAGE,
        "previous_experiments": [],
        "critic_revision_count": 0,
        "critic_revision_instructions": "",
        "proposed_experiment_output": draft,
        "experiment_revision": {"draft_origin": "INITIAL"},
    }
