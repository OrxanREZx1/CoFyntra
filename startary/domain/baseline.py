"""
Domain models for Baseline Startup Check-In — Milestone 2.6.1.

Separates operational baseline collection from strategic and weak-link clarification.

Dimensions:
  - TEAM: solo vs team, active team size, basic roles/capabilities
  - CAPACITY: weekly availability, total execution capacity
  - PRODUCT_STAGE: idea, designs, prototype, MVP, launched, testing
  - VALIDATION_OR_USER: interviews, active users, access to target community
  - RESOURCE: available budget, funding status, resource constraints
"""

from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class BaselineDimension(str, Enum):
    """The five core operational baseline dimensions."""
    TEAM = "TEAM"
    CAPACITY = "CAPACITY"
    PRODUCT_STAGE = "PRODUCT_STAGE"
    VALIDATION_OR_USER = "VALIDATION_OR_USER"
    RESOURCE = "RESOURCE"


class BaselineDimensionStatus(str, Enum):
    """Knowledge status of a baseline dimension."""
    KNOWN_ENOUGH = "KNOWN_ENOUGH"
    MISSING = "MISSING"
    NOT_RELEVANT = "NOT_RELEVANT"


class BaselineStartupStateSufficiency(BaseModel):
    """
    Evaluation of whether minimum operational picture exists before entering strategic reasoning.
    """
    is_sufficient: bool = Field(
        description="True if Startary knows enough minimum operational facts to proceed to strategic diagnosis."
    )
    dimension_statuses: dict[BaselineDimension, BaselineDimensionStatus] = Field(
        default_factory=dict,
        description="Status for each of the 5 baseline dimensions.",
    )
    missing_groups: list[BaselineDimension] = Field(
        default_factory=list,
        description="List of dimensions that are still MISSING.",
    )
    reason: str = Field(
        default="",
        description="Brief analytical justification for the baseline sufficiency judgment.",
    )


class BaselineQuestionPlan(BaseModel):
    """
    Action plan for the next grouped baseline prompt.
    """
    should_ask: bool = Field(
        description="True if another baseline question should be asked right now."
    )
    target_group: Optional[BaselineDimension] = Field(
        default=None,
        description="The primary baseline dimension being addressed.",
    )
    target_fields: list[str] = Field(
        default_factory=list,
        description="Specific fields this question aims to populate (e.g. ['active_team_size', 'member_roles']).",
    )
    question: Optional[str] = Field(
        default=None,
        description="Compact, conversational natural-language question combining related fields.",
    )
    reason: str = Field(
        default="",
        description="Why this question group was prioritized.",
    )

