"""
InformationGapPlan — Startary Milestone 2.6.

The output of the Adaptive Information Gap Planner.

This planner combines:
  - Strategic context sufficiency evaluation (replaces assess_context_sufficiency)
  - Execution context gap assessment
  - Startup progress gap assessment

One unified LLM call produces one structured decision:
  - should_ask (bool): True if one question should be asked right now
  - target_domain: where the gap lies (STRATEGIC / EXECUTION / PROGRESS)
  - priority: REQUIRED_NOW / USEFUL_LATER / IRRELEVANT_NOW
  - question: the one short, natural question to ask (if should_ask=true)

Design constraints:
  - ONE question at a time (never a list)
  - Asks only when missing information materially improves the current decision
  - Unknown fields alone do NOT trigger questions
  - Strategic gaps take priority over profile gaps when turns are limited
  - Does NOT ask for information already present in state
"""

from __future__ import annotations

from enum import StrEnum
from typing import Optional

from pydantic import BaseModel, Field


class GapPriority(StrEnum):
    """Relevance of a missing information gap to the current workflow step."""

    REQUIRED_NOW = "REQUIRED_NOW"     # Missing info materially blocks or distorts the current decision
    USEFUL_LATER = "USEFUL_LATER"     # Would help eventually (e.g. budget for experiment design)
    IRRELEVANT_NOW = "IRRELEVANT_NOW" # Not needed for any current or near-future decision


class GapTargetDomain(StrEnum):
    """Which layer of the startup model the gap belongs to."""

    STRATEGIC = "STRATEGIC"   # Problem / users / demand / alternatives / value prop etc.
    EXECUTION = "EXECUTION"   # Team / budget / capabilities / customer access etc.
    PROGRESS = "PROGRESS"     # MVP / product / users / revenue / validation activity etc.


class UncertaintyResolutionMode(StrEnum):
    """
    How a strategic information gap or uncertainty should be resolved (Milestone 4.0.2).
    """
    FOUNDER_KNOWLEDGE = "FOUNDER_KNOWLEDGE"       # Founder may already know the answer. Ask clarification.
    EXTERNAL_EVIDENCE = "EXTERNAL_EVIDENCE"       # Answer requires real-world customer/market evidence not currently available. Route toward TEST.
    NOT_YET_FORMULATABLE = "NOT_YET_FORMULATABLE" # Context is genuinely too incomplete even to formulate a testable assumption.


class InformationGapPlan(BaseModel):
    """
    The unified output of the Adaptive Information Gap Planner.

    This replaces the separate ContextSufficiencyAssessment for routing decisions.
    The planner evaluates strategic sufficiency, profile gaps, and uncertainty resolution mode.
    """

    should_ask: bool = Field(
        description=(
            "True if exactly one clarifying question should be asked right now. "
            "False if context is sufficient to proceed OR if no question would materially improve the decision."
        )
    )

    strategic_sufficient: bool = Field(
        default=False,
        description=(
            "True if founder context is grounded enough to identify the next testable uncertainty. "
            "This is the strategic sufficiency signal."
        ),
    )

    resolution_mode: UncertaintyResolutionMode = Field(
        default=UncertaintyResolutionMode.FOUNDER_KNOWLEDGE,
        description="FOUNDER_KNOWLEDGE | EXTERNAL_EVIDENCE | NOT_YET_FORMULATABLE",
    )

    can_formulate_testable_uncertainty: bool = Field(
        default=False,
        description="True if context is sufficient to formulate a testable hypothesis/assumption for reality to answer.",
    )

    why_this_matters: str = Field(
        default="",
        description="Why this uncertainty or information gap matters for the next step.",
    )

    missing_information: str = Field(
        default="",
        description="Specific missing information or uncertainty identified.",
    )

    recommended_area: Optional[str] = Field(
        default=None,
        description="Startup area where the critical uncertainty lies (e.g. 'Problem', 'Target User').",
    )

    target_domain: Optional[GapTargetDomain] = Field(
        default=None,
        description="The domain where the highest-value gap exists. None if should_ask=False.",
    )

    target_field_or_area: Optional[str] = Field(
        default=None,
        description="The specific field or strategic area that needs clarification. None if should_ask=False.",
    )

    priority: GapPriority = Field(
        default=GapPriority.USEFUL_LATER,
        description="Priority of the identified gap for the current reasoning step.",
    )

    information_goal: str = Field(
        default="",
        description="The specific missing information that would be resolved by the question.",
    )

    reason: str = Field(
        description="Short analytical explanation of why this question was selected (or why no question is needed).",
    )

    question: Optional[str] = Field(
        default=None,
        description=(
            "The single, short, natural-language question to ask the founder. "
            "None if should_ask=False. "
            "Must NOT be a list or multi-part questionnaire."
        ),
    )


def can_formulate_testable_uncertainty(state: dict, gap_plan: dict | None = None) -> bool:
    """
    Deterministic helper (Milestone 4.0.2):
    Evaluates whether the current startup context is sufficient to formulate
    a testable critical assumption and experiment, even if founder possesses 0 customer evidence.

    Returns True when enough exists to express:
      - a recognizable target population / customer context
      - a recognizable problem or value proposition
      - an addressable uncertainty
    Returns False only for content-free/vague non-ideas (e.g. 'I want to build something with AI').
    """
    if gap_plan and isinstance(gap_plan, dict):
        if gap_plan.get("can_formulate_testable_uncertainty") is True:
            return True
        if gap_plan.get("resolution_mode") == UncertaintyResolutionMode.NOT_YET_FORMULATABLE.value:
            return False

    startup_st = state.get("startup_state") or {}
    # Milestone 6.2.3: merge_diagnosis persists areas under "areas"; "area_knowledge" is only
    # written after an evidence cycle. Areas explicitly diagnosed UNKNOWN carry placeholder
    # narration, not recognizable founder content, so they never count as grounding.
    raw_areas = startup_st.get("areas") or startup_st.get("area_knowledge") or {}
    area_knowledge = {
        str(name): info
        for name, info in raw_areas.items()
        if isinstance(info, dict) and str(info.get("status") or "").upper() != "UNKNOWN"
    }
    diagnosis = state.get("diagnosis") or {}
    raw_ctx = (state.get("raw_context") or "").lower().strip()
    founder_messages = state.get("founder_messages") or []
    all_msgs_text = " ".join([str(m.get("content", "")) for m in founder_messages] + [raw_ctx]).lower()

    # Vague content-free non-ideas
    vague_markers = [
        "something with ai", "an app with ai", "make an ai startup", "don't know what",
        "dont know what", "not sure what to build", "want to build something with ai"
    ]
    if any(marker in all_msgs_text for marker in vague_markers) and len(all_msgs_text.split()) < 20:
        return False

    # Check 1: Recognizable target population
    target_user_info = area_knowledge.get("Target User") or {}
    target_user_text = (
        str(target_user_info.get("understanding") or "")
        + " " + " ".join(str(c) for c in (target_user_info.get("founder_claims") or []))
        + " " + str(diagnosis.get("target_users") or "")
    ).strip()

    has_target_population = len(target_user_text.split()) >= 3 or any(
        kw in all_msgs_text for kw in (
            "merchant", "merchants", "store", "café", "cafe", "manager", "managers",
            "student", "students", "driver", "drivers", "freight", "b2b", "consumer",
            "client", "customer", "shoppers", "retailer", "patients", "doctors"
        )
    )

    # Check 2: Recognizable problem or value proposition
    problem_info = area_knowledge.get("Problem") or {}
    val_prop_info = area_knowledge.get("Value Proposition") or {}
    problem_text = (
        str(problem_info.get("understanding") or "")
        + " " + " ".join(str(c) for c in (problem_info.get("founder_claims") or []))
        + " " + str(val_prop_info.get("understanding") or "")
    ).strip()

    has_problem_or_value = len(problem_text.split()) >= 3 or any(
        kw in all_msgs_text for kw in (
            "returns", "return", "cancellation", "cancellations", "scheduling",
            "career", "hiring", "delivery", "shipping", "waste", "cost", "tracking",
            "inventory", "automate", "reduce", "platform", "tool", "software"
        )
    )

    return bool(has_target_population and has_problem_or_value)
