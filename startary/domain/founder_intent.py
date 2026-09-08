"""
Founder Intent & Conversational Awareness Domain Models (Milestone 3.2).

Enables two-level message understanding:
1. Startup factual extraction (what does this say about the startup?)
2. Founder conversational intent (what is the founder trying to communicate/ask right now?)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class FounderIntentType(str, Enum):
    """Controlled taxonomy of founder conversational intents."""
    ANSWER = "ANSWER"
    QUESTION = "QUESTION"
    CONCERN = "CONCERN"
    CORRECTION = "CORRECTION"
    DOES_NOT_KNOW = "DOES_NOT_KNOW"
    CANNOT_ANSWER_NOW = "CANNOT_ANSWER_NOW"
    CANNOT_ACT_NOW = "CANNOT_ACT_NOW"
    NEW_EVIDENCE = "NEW_EVIDENCE"
    PREFERENCE = "PREFERENCE"
    REQUEST_FOR_ADVICE = "REQUEST_FOR_ADVICE"
    OTHER = "OTHER"


class FounderResponsePriority(str, Enum):
    """Response priority level indicating whether founder needs an immediate answer."""
    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"


class FounderTurnInterpretation(BaseModel):
    """
    Structured interpretation of the latest founder turn.
    Classifies the founder's communicative intent separately from factual state extraction.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    project_id: str = Field(description="Project identifier.")
    turn_index: int = Field(ge=0, default=0, description="Turn index of the interpreted message.")

    intent_types: list[FounderIntentType] = Field(
        default_factory=list,
        description="One or more intent categories that describe this message.",
    )
    direct_answer: bool = Field(
        default=False,
        description="Whether this message answers the currently pending Startary question.",
    )
    direct_question: Optional[str] = Field(
        default=None,
        description="Specific question the founder asked Startary, if any.",
    )
    concern_or_goal: Optional[str] = Field(
        default=None,
        description="Core emotional, financial, or strategic fear/concern expressed.",
    )
    correction: Optional[str] = Field(
        default=None,
        description="Specific misunderstanding or false assumption the founder corrected.",
    )
    cannot_answer_now: bool = Field(
        default=False,
        description="True if founder indicates they do not know or cannot answer this information gap.",
    )
    cannot_act_now: bool = Field(
        default=False,
        description="True if founder indicates they cannot perform external tasks (interviews, outreach) right now.",
    )
    uncertainty_statement: Optional[str] = Field(
        default=None,
        description="Specific uncertainty the founder explicitly pointed out.",
    )
    new_evidence_summary: Optional[str] = Field(
        default=None,
        description="Summary of any new customer/operational evidence introduced in this turn.",
    )
    founder_preference_or_constraint: Optional[str] = Field(
        default=None,
        description="Personal preference, working style, or hard boundary expressed.",
    )
    is_already_answered_claim: bool = Field(
        default=False,
        description="True if founder stated they already answered this question or Startary already asked it.",
    )
    response_priority: FounderResponsePriority = Field(
        default=FounderResponsePriority.NORMAL,
        description="Priority: HIGH if founder asked a question, expressed fear, corrected Startary, or cannot act.",
    )
    interpretation_summary: str = Field(
        default="Founder turn interpreted.",
        description="Concise 1-2 sentence summary of what the founder communicated in this turn.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )


class AdvisorResponse(BaseModel):
    """
    Structured advisor response presented to the founder.
    Integrates direct answering, acknowledgement, diagnosis delta, and context-connected next question.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    project_id: str = Field(description="Project identifier.")
    turn_index: int = Field(ge=0, default=0)

    acknowledge: Optional[str] = Field(
        default=None,
        description="Brief, specific acknowledgement of founder input, concern, or correction.",
    )
    answer_to_founder: Optional[str] = Field(
        default=None,
        description="Substantive, grounded answer to founder's direct question or concern (if applicable).",
    )
    diagnosis_delta: Optional[str] = Field(
        default=None,
        description="Explanation of how new evidence materially strengthened or changed Startary's assessment.",
    )
    next_question: Optional[str] = Field(
        default=None,
        description="Legacy field for the next information-gathering question.",
    )
    question_id: Optional[str] = Field(
        default=None,
        description="Unique token matching the active NextTurnDecision.",
    )
    next_question_target: Optional[str] = Field(
        default=None,
        description="Semantic information target matching NextTurnDecision.",
    )
    next_question_rephrased: Optional[str] = Field(
        default=None,
        description="Natural rephrasing of the canonical question.",
    )
    should_ask_next_question: bool = Field(
        default=True,
        description="Whether a question should be asked now, or if answering the founder is sufficient.",
    )
    response_mode: Optional[str] = Field(
        default="BRIEF_CONTINUE",
        description="Conversational response mode (Milestone 4.0.4).",
    )
    response_text: str = Field(
        description="The final integrated founder-facing message (1-4 sentences total).",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
