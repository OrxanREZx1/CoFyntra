"""
Strategic Action Domain Models — Milestone 4.1.2: Decision Loop Integrity & Progression.

Provides explicit strategic categorization of the next required action before
experiment generation, preventing the system from mechanically defaulting to
generic experiments.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class NextStrategicActionType(str, Enum):
    """
    Bounded taxonomy of strategic action types.
    Represents the high-level strategic decision required for the startup's current stage and evidence.
    """
    CONTINUE_CURRENT_TEST = "CONTINUE_CURRENT_TEST"
    NEW_VALIDATION_TEST = "NEW_VALIDATION_TEST"
    RETENTION_TEST = "RETENTION_TEST"
    PRICING_TEST = "PRICING_TEST"
    ACQUISITION_TEST = "ACQUISITION_TEST"
    PRODUCT_ITERATION = "PRODUCT_ITERATION"
    BUILD_MINIMAL_PROTOTYPE = "BUILD_MINIMAL_PROTOTYPE"
    MORE_FOUNDER_KNOWLEDGE = "MORE_FOUNDER_KNOWLEDGE"


class NextStrategicAction(BaseModel):
    """
    Structured strategic decision produced before experiment generation.
    """
    action_type: NextStrategicActionType = Field(
        ...,
        description="The category of strategic action determined to be highest leverage.",
    )
    rationale: str = Field(
        ...,
        description="Strategic justification based on startup stage, traction, and evidence.",
    )
    target_area: str = Field(
        ...,
        description="Canonical startup area this strategic action addresses.",
    )
    key_decision: str = Field(
        ...,
        description="The core business or product decision this action will inform.",
    )
    why_this_action: str = Field(
        ...,
        description="Why this specific action type is superior to alternative actions at this stage.",
    )
    suggested_method: Optional[str] = Field(
        default=None,
        description="Suggested operational or experimental method for this action.",
    )
    # Milestone 4.1.2.1: Product Iteration & Weak-Link Override Fields
    bottleneck_identified: Optional[str] = Field(
        default=None,
        description="Concrete evidence-backed bottleneck.",
    )
    blocked_behavior: Optional[str] = Field(
        default=None,
        description="Specific user behavior blocked by this bottleneck.",
    )
    smallest_change: Optional[str] = Field(
        default=None,
        description="Smallest product or workflow change capable of testing the causal hypothesis.",
    )
    measurable_outcome: Optional[str] = Field(
        default=None,
        description="Measurable behavioral outcome to track.",
    )
    what_not_to_build: list[str] = Field(
        default_factory=list,
        description="Explicit scope boundary on what NOT to build yet.",
    )
    is_weak_link_override: bool = Field(
        default=False,
        description="Whether this strategic action temporarily overrides the taxonomy weak link.",
    )
    override_reason: Optional[str] = Field(
        default=None,
        description="Explicit strategic justification if overriding the taxonomy weak link.",
    )

