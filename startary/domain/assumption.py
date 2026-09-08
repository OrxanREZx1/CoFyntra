"""
Assumption schema.

Assumptions are first-class domain objects in Startary.  They are NOT
buried inside chat text or stored as unstructured strings in graph state.

An assumption captures a specific belief about the startup that must be
tested.  Its status is updated by evidence analysis modules (Milestone 3+).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from startary.domain.enums import AssumptionStatus
from startary.domain.taxonomy import StartupArea


class Assumption(BaseModel):
    """
    A specific, testable belief held by the founder or inferred by Startary.

    Status lifecycle:
      UNTESTED → [experiment runs] → SUPPORTED | WEAKENED | CONTRADICTED | INCONCLUSIVE
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique assumption identifier.",
    )
    project_id: str = Field(description="The project this assumption belongs to.")

    statement: str = Field(
        ...,
        min_length=5,
        description="The specific belief being held. Must be a complete sentence.",
    )
    area: StartupArea = Field(
        description="Which startup area this assumption belongs to.",
    )
    status: AssumptionStatus = Field(
        default=AssumptionStatus.UNTESTED,
        description="Current evidential status of this assumption.",
    )
    is_critical: bool = Field(
        default=False,
        description=(
            "True if this assumption is currently the critical one being actively tested. "
            "Only one assumption per iteration should be critical at a time."
        ),
    )
    iteration_created: int = Field(
        ge=1,
        description="Learning-loop iteration in which this assumption was first identified.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )


class CriticalAssumptionGroundingError(ValueError):
    """
    Raised when critical assumption grounding validation fails after all LLM retries
    and deterministic fallback recovery attempts have been exhausted (Milestone 4.0.7.1.2).
    Subclasses ValueError for backward compatibility with existing exception handlers.
    """
    pass


