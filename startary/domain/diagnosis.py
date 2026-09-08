"""
Diagnosis schema.

In Milestone 1, Diagnosis is populated by the diagnose_startup + merge_diagnosis
graph nodes.

Changes from M0:
  - raw_context: the verbatim founder input that was diagnosed
  - startup_stage: conservative stage inference from the LLM
  - unsupported_claims: cross-area list of founder claims without evidence
  - clear_areas/unclear_areas added for CLI display clarity

Design note:
  weak_areas in M0 was described as "weakest or most uncertain areas".
  In M1 we DO populate missing_areas (areas with no information).
  We do NOT populate weak_areas — that is Milestone 2's Weak Link Selector.
  The distinction matters: missing != weak.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from startary.domain.assessment import StartupAreaAssessment
from startary.domain.taxonomy import StartupArea


class Diagnosis(BaseModel):
    """
    A structured diagnosis of a startup's current state.

    Produced by the M1 diagnose_startup → merge_diagnosis node pair.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique diagnosis identifier.",
    )
    project_id: str = Field(description="The project this diagnosis belongs to.")
    iteration: int = Field(ge=1, description="Learning-loop iteration of this diagnosis.")

    raw_context: str | None = Field(
        default=None,
        description="The verbatim founder input that was diagnosed.",
    )

    assessments: list[StartupAreaAssessment] = Field(
        default_factory=list,
        description="Per-area assessments produced by the diagnosis module.",
    )

    clear_areas: list[StartupArea] = Field(
        default_factory=list,
        description="Areas that are relatively well-understood.",
    )

    unclear_areas: list[StartupArea] = Field(
        default_factory=list,
        description="Areas that were mentioned but remain ambiguous or contradictory.",
    )

    weak_areas: list[StartupArea] = Field(
        default_factory=list,
        description=(
            "Areas identified as top strategic risks by Milestone 2 Weak Link Selector."
        ),
    )

    selected_weak_link: StartupArea | None = Field(
        default=None,
        description="The single highest-priority weak-link startup area selected for testing.",
    )

    missing_areas: list[StartupArea] = Field(
        default_factory=list,
        description="Areas with no information at all in the founder context.",
    )

    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="Cross-area founder claims that lack supporting evidence.",
    )

    summary: str | None = Field(
        default=None,
        description="High-level analytical diagnosis summary (AI-generated).",
    )

    startup_stage: str | None = Field(
        default=None,
        description=(
            "Conservative stage inference. One of: raw_idea, problem_exploration, "
            "validation, mvp_planning, early_build. Null if insufficient evidence."
        ),
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
