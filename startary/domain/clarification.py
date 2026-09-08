"""
Clarifying Question schema for Startary Milestone 2.5.

When context is insufficient, Startary asks exactly ONE highest-information-value
question at a time to resolve the primary blocking uncertainty without inventing
unsupported product mechanisms.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from startary.domain.taxonomy import StartupArea


class ClarifyingQuestion(BaseModel):
    """
    A single structured clarification question to ask the founder.
    """

    question: str = Field(
        min_length=10,
        description="The natural-language question to present directly to the founder.",
    )

    target_area: StartupArea = Field(
        description="The startup area targeted by this clarifying question.",
    )

    information_goal: str = Field(
        description="The specific missing context this question seeks to obtain.",
    )

    why_this_question_now: str = Field(
        description="Analytical rationale explaining why this question maximizes decision-relevant grounding.",
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp of when this question was generated.",
    )

