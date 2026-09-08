"""
Startup Diagnosis Snapshot Domain Model (Milestone 3.1).

A lightweight, founder-facing presentation and composition layer that translates
internal diagnostic intelligence (stage, grounded areas, advantages, constraints,
major uncertainties, weak link, confidence, competing risks, reality summary,
and critical assumption) into a concise, understandable snapshot.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pydantic import BaseModel, Field


class StartupDiagnosisSnapshot(BaseModel):
    """
    A concise, founder-facing snapshot of the startup's current diagnostic reality.

    Distinguishes:
      - Grounded / Clear (what Startary understands)
      - Current Advantages (practical execution/learning leverage)
      - Execution Constraints (capacity, budget, skills)
      - Major Uncertainties (top 3-5 decision-critical unknowns)
      - Current Weak Link & Confidence
      - Competing Risks (max 3)
      - Reality Summary
      - Recommended Learning Goal
      - Critical Assumption (exact statement moving into TEST)
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique snapshot identifier.",
    )
    project_id: str = Field(description="The project this snapshot belongs to.")
    iteration: int = Field(ge=1, default=1, description="Learning-loop iteration of this snapshot.")

    # Current Stage
    stage: str = Field(description="Current startup stage (e.g. Early Build / Pre-MVP, Idea, Solution Exploration).")
    stage_explanation: str = Field(description="1-2 sentence founder-friendly explanation of why the startup is at this stage.")

    # Grounding, Advantages, and Constraints
    grounded_areas: list[str] = Field(
        default_factory=list,
        description="Things Startary understands with reasonable grounding (neutral facts).",
    )
    current_advantages: list[str] = Field(
        default_factory=list,
        description="Genuine practical execution or learning assets (e.g. customer access, existing prototype, relevant skills).",
    )
    execution_constraints: list[str] = Field(
        default_factory=list,
        description="Factual limitations on team, budget, time, or capabilities.",
    )

    # Uncertainties
    major_uncertainties: list[str] = Field(
        default_factory=list,
        max_length=5,
        description="Top 3 to 5 uncertainties that materially matter to the current stage and next decision.",
    )

    # Weak Link & Confidence
    weak_link: str = Field(description="The primary weak-link area (e.g. Problem, Demand Signal).")
    weak_link_reason: str = Field(description="Empathetic, clear explanation of why this area represents the highest immediate risk.")
    weak_link_confidence: str = Field(description="Selection confidence level: LOW, MEDIUM, or HIGH.")
    weak_link_confidence_explanation: str = Field(description="Founder-friendly explanation of what the confidence level means.")
    competing_risks: list[str] = Field(
        default_factory=list,
        max_length=3,
        description="Up to 3 most relevant competing risk areas.",
    )

    # Reality Check Summary & Learning Goal
    reality_summary: str = Field(description="Concise 2-4 sentence founder-facing summary of the Reality Check.")
    recommended_learning_goal: str = Field(description="The strategic learning objective (what we need to learn next) before the experiment.")

    # Critical Assumption
    critical_assumption: str = Field(description="The verbatim statement of the active Critical Assumption.")

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when the snapshot was generated.",
    )

