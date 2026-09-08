"""
Milestone 3.2.1: Domain models for Conversational Control Integrity.

Defines:
- NextTurnDecision: Authoritative planner decision for next information need.
- QuestionHistoryItem: History item tracking asked/answered questions to prevent semantic loops.
- FactConflict: Tracks explicit fact contradictions that require founder clarification.
- FounderGoal: Lightweight, transient founder goal for conversational alignment.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
import uuid

from pydantic import BaseModel, ConfigDict, Field

from startary.domain.enums import WorkflowPhase


class FounderGoalType(str, Enum):
    FIND_INITIAL_MARKET = "FIND_INITIAL_MARKET"
    DECIDE_IF_WORTH_CONTINUING = "DECIDE_IF_WORTH_CONTINUING"
    VALIDATE_PROBLEM = "VALIDATE_PROBLEM"
    FIND_CUSTOMERS = "FIND_CUSTOMERS"
    UNDERSTAND_WHAT_TO_BUILD = "UNDERSTAND_WHAT_TO_BUILD"
    OTHER = "OTHER"


class FounderGoal(BaseModel):
    model_config = ConfigDict(frozen=True)

    goal_type: FounderGoalType = Field(
        default=FounderGoalType.OTHER,
        description="Categorized founder goal for response framing",
    )
    raw_statement: str = Field(
        description="The founder's explicit statement of what they are trying to achieve",
    )
    turn_acquired: int = Field(
        default=0,
        description="Conversation turn when this goal was expressed",
    )
    is_active: bool = Field(
        default=True,
        description="Whether this goal currently influences conversational framing",
    )


class NextTurnDecision(BaseModel):
    """
    Authoritative decision produced by the planner/routing layer specifying
    WHAT information Startary needs next. The Advisor Composer must preserve
    this target and cannot independently choose a different information need.
    """
    model_config = ConfigDict(frozen=True)

    question_id: str = Field(
        default_factory=lambda: f"q_{uuid.uuid4().hex[:10]}",
        description="Unique turn token binding the planner decision to the advisor response",
    )
    source_phase: WorkflowPhase = Field(
        default=WorkflowPhase.UNDERSTAND,
        description="Workflow phase where this decision was created",
    )
    information_target: str = Field(
        description="Semantic target (e.g. TEAM, CAPACITY, PRODUCT_STAGE, VALIDATION_OR_USER, RESOURCE, PROBLEM_DETAIL, etc.)",
    )
    canonical_question: str = Field(
        description="Authoritative, unembellished question formulated by the planner",
    )
    should_ask_question: bool = Field(
        default=True,
        description="True if a question should be asked to the founder this turn",
    )
    clarification_reason: str = Field(
        default="",
        description="Why this specific information is needed right now",
    )
    planner_version: str = Field(
        default="",
        description="Version string of the planner that generated this decision",
    )
    founder_question_requires_answer: bool = Field(
        default=False,
        description="True if the founder asked a direct question requiring an answer first",
    )
    founder_concern_requires_response: bool = Field(
        default=False,
        description="True if the founder expressed a fear/concern requiring validation",
    )
    correction_requires_acknowledgement: bool = Field(
        default=False,
        description="True if the founder corrected Startary requiring explicit acknowledgement",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )


class QuestionHistoryItem(BaseModel):
    """Tracks asked questions to prevent semantic loops and repetition."""
    model_config = ConfigDict(frozen=True)

    question_id: str = Field(description="Unique question token")
    information_target: str = Field(description="Semantic target of the question")
    canonical_question: str = Field(description="The core question asked")
    turn: int = Field(description="Conversation turn index when asked")
    answered: bool = Field(default=False, description="Whether the founder answered this question")
    founder_answer_summary: Optional[str] = Field(default=None, description="Brief summary of founder response")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class FactConflict(BaseModel):
    """Tracks contradictions between explicit facts that require founder clarification."""
    model_config = ConfigDict(frozen=True)

    field_name: str = Field(description="State field where conflict occurred (e.g. founder_count)")
    existing_value: Any = Field(description="Confirmed existing high-confidence value")
    conflicting_value: Any = Field(description="New incoming contradictory value")
    clarification_question: str = Field(description="Clarification question to resolve the conflict")
    resolved: bool = Field(default=False, description="Whether the founder has clarified the conflict")
    detected_turn: int = Field(default=0, description="Turn when conflict was detected")


# ---------------------------------------------------------------------------
# Milestone 4.0.4: Conversational Naturalness & Response Modes
# ---------------------------------------------------------------------------

import re as _re
from startary.domain.enums import ConversationalResponseMode


def should_surface_diagnosis_delta(
    previous_state: dict[str, Any] | None,
    updated_state: dict[str, Any] | None,
    founder_turn: str | dict[str, Any] | None = None,
) -> bool:
    """
    Materiality gate for diagnosis delta commentary (Milestone 4.0.4 Part C).

    Returns True ONLY if the founder's answer materially changes:
    - startup stage
    - weak-link reasoning or selection
    - risk ranking or competing risks
    - target user or problem understanding
    - evidence strength (e.g. validated evidence submitted)
    - execution feasibility
    - active founder concern / strategic goal

    Routine operational answers (e.g. '15 hours each', '$500 budget', 'USD')
    MUST return False to avoid repetitive, formulaic advisor commentary.
    """
    if not updated_state:
        return False

    prev = previous_state or {}

    # 1. Stage change
    prev_stage = prev.get("startup_stage") or prev.get("stage")
    curr_stage = updated_state.get("startup_stage") or updated_state.get("stage")
    if prev_stage and curr_stage and prev_stage != curr_stage:
        return True

    # 2. Weak-link change
    prev_wl = prev.get("weak_area") or prev.get("weak_link")
    curr_wl = updated_state.get("weak_area") or updated_state.get("weak_link")
    if prev_wl and curr_wl and prev_wl != curr_wl:
        return True

    # 3. Evidence learning cycle completed
    if updated_state.get("evidence_learning_delta") or updated_state.get("latest_evidence_analysis"):
        return True

    # 4. Critical assumption change
    prev_ca = (prev.get("active_assumption") or {}).get("statement")
    curr_ca = (updated_state.get("active_assumption") or {}).get("statement")
    if prev_ca and curr_ca and prev_ca != curr_ca:
        return True

    # 5. Founder expressed direct concern or strategic goal
    if founder_turn:
        text = founder_turn if isinstance(founder_turn, str) else str(founder_turn.get("content", ""))
        text_lower = text.lower()
        strategic_concern_patterns = [
            r"\b(worth\s+(to\s+)?(ship|pursu|build|continu))\b",
            r"\b(should\s+we\s+(build|pivot|give\s+up|stop))\b",
            r"\b(is\s+this\s+a\s+real\s+problem)\b",
            r"\b(waste\s+of\s+time)\b",
        ]
        if any(_re.search(pat, text_lower) for pat in strategic_concern_patterns):
            return True

    # Otherwise, routine operational updates (hours, team, budget, tools) do NOT warrant commentary
    return False


def determine_conversational_response_mode(
    state: dict[str, Any],
    founder_interpretation: dict[str, Any] | None = None,
    founder_turn: str | dict[str, Any] | None = None,
    previous_state: dict[str, Any] | None = None,
) -> ConversationalResponseMode:
    """
    Deterministically resolves the appropriate ConversationalResponseMode for advisor cadence (Milestone 4.0.4).

    Precedence:
    1. CORRECTION_RESPONSE: founder corrected Startary
    2. DIRECT_ANSWER: founder asked a direct question or asked for advice/guidance
    3. CONCERN_RESPONSE: founder expressed fear, doubt, or hesitation
    4. HANDOFF_TO_TEST: transitioning to experiment execution / external evidence
    5. MATERIAL_UPDATE: material diagnostic or strategic shift occurred
    6. BRIEF_CONTINUE: normal routine baseline or clarification turn
    """
    interp = founder_interpretation or {}
    decision = state.get("active_next_turn_decision") or {}

    # Extract founder text
    text = ""
    if isinstance(founder_turn, str):
        text = founder_turn
    elif isinstance(founder_turn, dict):
        text = str(founder_turn.get("content") or "")
    elif not text and state.get("founder_messages"):
        msgs = state.get("founder_messages") or []
        for m in reversed(msgs):
            if isinstance(m, dict) and m.get("role") == "founder":
                text = str(m.get("content") or "")
                break
    text_lower = text.lower()

    # 1. Correction
    if (
        interp.get("is_correction")
        or decision.get("correction_requires_acknowledgement")
        or any(phrase in text_lower for phrase in ["i already said", "i already answered", "that's not what i said", "you misunderstood"])
    ):
        return ConversationalResponseMode.CORRECTION_RESPONSE

    # 2. Direct Answer (Question / Advice request)
    is_question = (
        interp.get("is_question")
        or interp.get("asks_for_guidance")
        or decision.get("founder_question_requires_answer")
        or "?" in text
        or any(q in text_lower for q in [
            "worth to ship", "worth shipping", "worth pursuing", "should i", "should we",
            "what should i do", "what do you think", "how do we know", "is this problem worth"
        ])
    )
    if is_question:
        return ConversationalResponseMode.DIRECT_ANSWER

    # 3. Concern / Fear
    is_concern = (
        interp.get("expresses_concern")
        or interp.get("expresses_fear")
        or decision.get("founder_concern_requires_response")
        or any(w in text_lower for w in ["worried", "afraid", "doubt", "hesitant", "scared", "not sure if anyone wants"])
    )
    if is_concern:
        return ConversationalResponseMode.CONCERN_RESPONSE

    # 4. Handoff to Test
    phase = state.get("phase")
    uncertainty_mode = state.get("uncertainty_resolution_mode")
    if (
        phase == WorkflowPhase.TEST
        or phase == "TEST"
        or uncertainty_mode == "EXTERNAL_EVIDENCE"
    ):
        return ConversationalResponseMode.HANDOFF_TO_TEST

    # 5. Material Update
    if should_surface_diagnosis_delta(previous_state, state, text):
        return ConversationalResponseMode.MATERIAL_UPDATE

    # 6. Default: Brief Continue
    return ConversationalResponseMode.BRIEF_CONTINUE


