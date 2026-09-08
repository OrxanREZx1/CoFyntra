"""
Context Sufficiency Assessment schema for Startary Milestone 2.5.1.

The Context Sufficiency Gate evaluates whether there is enough grounded information
from the founder to identify a meaningful decision-critical uncertainty WITHOUT
guessing or inventing missing context (subject, behavior, mechanism, evidence, or decision context).

Epistemic Levels:
  Level 1 — CONCEPTUAL UNDERSTANDING (founder hypothesis / generalized story) -> WEAK grounding.
  Level 2 — GROUNDED UNDERSTANDING (direct personal experience / customer conversations / observed actions) -> MODERATE grounding.
  Level 3 — VALIDATED (systematic evidence) -> STRONG grounding (not required to proceed).

Rules:
  - Sufficiency is NOT completeness: not all 10 startup areas need to be known.
  - Sufficiency is NOT validation: statistical proof is not required before proceeding.
  - Sufficiency requires GROUNDING: generalized hypothetical stories without direct observation or experience
    must remain INSUFFICIENT.
  - If would_one_more_answer_materially_change_direction is True, context remains INSUFFICIENT.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Optional

from pydantic import BaseModel, Field

from startary.domain.assessment import RiskFactorLevel
from startary.domain.taxonomy import StartupArea


class GroundingQuality(StrEnum):
    """
    Evaluates how grounded the current understanding is in founder-supplied reality
    (distinct from validation strength).
    """
    WEAK = "WEAK"          # Mostly generalized founder beliefs, hypothetical examples, abstractions
    MODERATE = "MODERATE"  # At least one real observation, founder personal experience, or customer conversation
    STRONG = "STRONG"      # Multiple concrete observations or qualitative/quantitative evidence


class ContextSufficiencyAssessment(BaseModel):
    """
    Structured assessment of whether the current founder context is sufficient
    to proceed to Weak-Link risk evaluation and Critical Assumption generation.
    """

    sufficient: bool = Field(
        description="True if grounded context is sufficient to reason about critical uncertainty without inventing facts."
    )

    confidence: RiskFactorLevel = Field(
        default=RiskFactorLevel.MEDIUM,
        description="Confidence level in this sufficiency judgment (LOW, MEDIUM, HIGH).",
    )

    grounding_quality: GroundingQuality = Field(
        default=GroundingQuality.WEAK,
        description=(
            "Degree of founder-supplied grounding: "
            "WEAK (generalized beliefs/hypotheticals), "
            "MODERATE (real observation/direct experience/conversations), "
            "STRONG (multiple concrete observations/evidence)."
        ),
    )

    would_one_more_answer_materially_change_direction: bool = Field(
        default=False,
        description=(
            "True if one additional clarifying answer has a realistic chance of materially "
            "changing the selected weak link, problem interpretation, or critical assumption."
        ),
    )

    reason: str = Field(
        description="Short analytical explanation of why the context is or is not sufficient."
    )

    blocking_unknowns: list[str] = Field(
        default_factory=list,
        description="Specific unknowns that actively prevent formulating a sound, grounded critical assumption.",
    )

    usable_grounding: list[str] = Field(
        default_factory=list,
        description="Concrete pieces of founder-provided information that are already grounded and usable.",
    )

    prohibited_inferences: list[str] = Field(
        default_factory=list,
        description="Details or mechanisms Startary would have to fabricate if it proceeded right now.",
    )

    recommended_question_area: Optional[StartupArea] = Field(
        default=None,
        description="The primary startup area that a clarifying question should target, if insufficient.",
    )

    information_goal: str = Field(
        default="",
        description="The specific missing information goal that must be resolved next.",
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp of when this assessment was generated.",
    )
