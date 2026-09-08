"""
Domain models for Weak-Link Discrimination — Milestone 2.6.1.

Tracks selection confidence, competing high-risk areas, and discriminating questions
before Startary enters Reality Check and Critical Assumption generation.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field

from startary.domain.taxonomy import StartupArea


class SelectionConfidence(str, Enum):
    """
    Confidence that the selected weak-link area deserves priority over competing candidates.
    This is NOT validation confidence or area risk; it measures decision discrimination.
    """
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class WeakLinkDiscrimination(BaseModel):
    """
    Evaluation of whether the deterministic weak-link winner is sufficiently discriminated
    from other high-risk or tied areas.
    """
    selection_confidence: SelectionConfidence = Field(
        description="Confidence level in the weak-link priority selection (LOW, MEDIUM, HIGH)."
    )
    current_best_area: StartupArea = Field(
        description="The candidate area currently leading according to deterministic scoring."
    )
    competing_areas: list[StartupArea] = Field(
        default_factory=list,
        description="Other areas tied or within near-identical decision risk of the current winner.",
    )
    discrimination_reason: str = Field(
        description="Analytical explanation of why the winner is distinct or why ambiguity exists."
    )
    missing_discriminating_information: str = Field(
        default="",
        description="Specific real-world evidence or founder context that would separate the candidates.",
    )
    should_clarify: bool = Field(
        default=False,
        description="True if one discriminating question should be asked to separate the candidates.",
    )
    recommended_information_goal: str = Field(
        default="",
        description="Goal for the discriminating clarification question.",
    )


class WeakLinkClarifyingQuestion(BaseModel):
    """
    Single question crafted specifically to differentiate tied or saturated weak-link candidates.
    """
    question: str = Field(
        min_length=10,
        description="The natural-language question targeted at candidate differentiation.",
    )
    competing_areas: list[StartupArea] = Field(
        default_factory=list,
        description="The areas this question is designed to differentiate.",
    )
    information_goal: str = Field(
        description="What missing differentiator this question aims to uncover.",
    )
    why_this_question_now: str = Field(
        description="Why this question is optimal for resolving weak-link ambiguity.",
    )

