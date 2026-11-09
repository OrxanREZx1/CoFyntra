"""
PitchVision AI first-turn scenario for offline graph tests (Milestone 6.2.3).

Scripts every first-turn LLM stage so the real graph reaches a weak-link clarification
interrupt: two areas tie at the maximum decision-risk score, discrimination asks to clarify,
select_weak_link_question plans the question, and the advisor composer phrases it.
"""

from __future__ import annotations

from typing import Any

from startary.domain.taxonomy import StartupArea
from tests.fixtures.scripted_llm import ScriptedLLM, composer_echo

PITCHVISION_FIRST_MESSAGE = (
    "We are PitchVision AI. We use computer vision to analyze amateur football matches and provide tactical insights. "
    "We are two co-founders, spending about 15 hours each per week. We have a budget of $1,500. "
    "We have direct access to two amateur football teams and three coaches who agreed to test our prototype once ready. "
    "We have not built the MVP yet, we have zero paying customers, zero revenue, and haven't run structured customer interviews yet."
)

PITCHVISION_ANSWER_1 = "Honestly the coaches. They already film matches on phones but never rewatch them because it takes hours."


def _f(value: str, provenance: str = "EXPLICIT", confidence: str = "HIGH", notes: str | None = None) -> dict[str, Any]:
    return {"value": value, "provenance": provenance, "confidence": confidence, "notes": notes}


# What a correct extractor returns for the first message (values are strings, as the schema requires).
PITCHVISION_DELTA: dict[str, Any] = {
    "founder_count": _f("2"),
    "active_team_size": _f("2"),
    "weekly_hours_per_member": _f("15"),
    "total_weekly_capacity_hours": _f("30", provenance="INFERRED", confidence="MEDIUM", notes="2 founders x 15h"),
    "available_budget_amount": _f("1500"),
    "budget_currency": _f("USD"),
    "customer_access": _f("Direct access to two amateur football teams and three coaches who agreed to test"),
    "development_stage": _f("IDEA", provenance="INFERRED", confidence="MEDIUM"),
    "mvp_exists": _f("false"),
    "validation_activity": _f("No structured customer interviews yet"),
    "interviews_completed": _f("0"),
    "paying_customers": _f("0"),
    "revenue": _f("0"),
}

# Facts the founder never stated: these must stay UNKNOWN everywhere.
PITCHVISION_NEVER_STATED = {
    "execution_context": ["funding_status", "funding_amount_raised", "geographic_market", "existing_audience_or_community"],
    "progress": ["active_users", "launched", "prototype_exists", "product_status"],
}


def pitchvision_diagnosis() -> dict[str, Any]:
    return {
        "area_assessments": [
            {
                "area": StartupArea.PROBLEM,
                "status": "PARTIALLY_KNOWN",
                "understanding": "Amateur coaches lack affordable post-match tactical analysis.",
                "founder_claims": ["Coaches want tactical insights from match video"],
                "evidence_provided": [],
                "missing_information": "Whether coaches feel this pain strongly enough to change behaviour.",
            },
            {
                "area": StartupArea.TARGET_USER,
                "status": "PARTIALLY_KNOWN",
                "understanding": "Amateur football teams and their coaches.",
                "founder_claims": ["Coaches are the primary users"],
                "evidence_provided": ["Two amateur teams and three coaches agreed to test a prototype"],
                "missing_information": "Who decides on and pays for analysis tools.",
            },
            {
                "area": StartupArea.FOUNDER_CONSTRAINTS,
                "status": "KNOWN",
                "understanding": "Two founders, about 15 hours each per week, $1,500 budget.",
                "founder_claims": [],
                "evidence_provided": [],
                "missing_information": None,
            },
        ],
        "clear_areas": [StartupArea.FOUNDER_CONSTRAINTS],
        "unclear_areas": [StartupArea.PROBLEM],
        "missing_areas": [StartupArea.DEMAND_SIGNAL, StartupArea.BUSINESS_MODEL],
        "unsupported_claims": ["Coaches want tactical insights from match video"],
        "diagnosis_summary": "Idea-stage computer-vision tactics tool for amateur football; no MVP, no revenue, no interviews.",
        "startup_stage": "idea",
    }


def tied_area_assessments() -> dict[str, Any]:
    """Problem and Target User tie at the maximum score (81) so discrimination is required."""
    top = {StartupArea.PROBLEM, StartupArea.TARGET_USER}
    return {
        "assessments": [
            {
                "area": area,
                "importance": "HIGH" if area in top else "LOW",
                "uncertainty": "HIGH" if area in top else "LOW",
                "evidence_gap": "HIGH" if area in top else "LOW",
                "consequence_of_being_wrong": "HIGH" if area in top else "LOW",
                "rationale": f"Risk assessment for {area}.",
            }
            for area in StartupArea
        ]
    }


def make_pitchvision_llm(turn_deltas: list[dict[str, Any]] | None = None, **overrides: Any) -> ScriptedLLM:
    """
    Build a ScriptedLLM for the PitchVision flow. ``turn_deltas[i]`` is the extraction delta for
    founder turn i (default: full delta on turn 0, empty delta afterwards). ``overrides`` replace
    individual schema responders.
    """
    deltas = turn_deltas if turn_deltas is not None else [PITCHVISION_DELTA]

    def extract(messages: list[dict[str, Any]], rec: ScriptedLLM) -> dict[str, Any]:
        turn = rec.count("StartupStateDeltaLLMOutput") - 1
        return deltas[turn] if turn < len(deltas) else {}

    def discrimination(messages: list[dict[str, Any]], rec: ScriptedLLM) -> dict[str, Any]:
        return {
            "selection_confidence": "LOW",
            "competing_areas": [StartupArea.PROBLEM, StartupArea.TARGET_USER],
            "discrimination_reason": "Problem and Target User share the maximum decision risk.",
            "missing_discriminating_information": "Whether coaches or players feel the pain most directly.",
            "should_clarify": True,
            "recommended_information_goal": "Separate problem-severity risk from target-user risk.",
        }

    def weak_link_question(messages: list[dict[str, Any]], rec: ScriptedLLM) -> dict[str, Any]:
        n = rec.count("WeakLinkClarifyingQuestionLLMOutput")
        return {
            "question": f"(Q{n}) Who struggles more today with reviewing matches: the coaches or the players?",
            "competing_areas": [StartupArea.PROBLEM, StartupArea.TARGET_USER],
            "information_goal": "Identify whose pain is most direct.",
            "why_this_question_now": "Separates problem severity from target-user ambiguity.",
        }

    responders = {
        "FounderTurnInterpretationLLMOutput": lambda m, r: {
            "intent_types": ["ANSWER"],
            "direct_answer": True,
            "direct_question": None,
            "concern_or_goal": None,
            "correction": None,
            "cannot_answer_now": False,
            "cannot_act_now": False,
            "uncertainty_statement": None,
            "new_evidence_summary": None,
            "founder_preference_or_constraint": None,
            "response_priority": "NORMAL",
            "interpretation_summary": "Founder shared startup context.",
        },
        "StartupStateDeltaLLMOutput": extract,
        "BaselineSufficiencyLLMOutput": lambda m, r: {
            "is_sufficient": True,
            "team_status": "KNOWN_ENOUGH",
            "capacity_status": "KNOWN_ENOUGH",
            "product_stage_status": "KNOWN_ENOUGH",
            "validation_status": "KNOWN_ENOUGH",
            "resource_status": "KNOWN_ENOUGH",
            "missing_groups": [],
            "reason": "All baseline dimensions known.",
        },
        "DiagnosisLLMOutput": lambda m, r: pitchvision_diagnosis(),
        "InformationGapPlanLLMOutput": lambda m, r: {
            "should_ask": False,
            "strategic_sufficient": True,
            "target_domain": None,
            "target_field_or_area": None,
            "priority": "USEFUL_LATER",
            "information_goal": "",
            "reason": "Context is sufficiently grounded to evaluate decision risk.",
            "question": None,
        },
        "StartupAreaAssessmentBatchLLMOutput": lambda m, r: tied_area_assessments(),
        "WeakLinkDiscriminationLLMOutput": discrimination,
        "WeakLinkClarifyingQuestionLLMOutput": weak_link_question,
        "AdvisorResponseLLMOutput": composer_echo,
        "TEXT": lambda m, r: "No problem. I am asking whose pain is most direct, because that decides what we test first. Who struggles more today with reviewing matches: the coaches or the players?",
    }
    responders.update(overrides)
    return ScriptedLLM(responders)
