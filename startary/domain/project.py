"""
StartupProject schema.

Represents one founder's startup workspace.  This is NOT the LangGraph
state — it is the durable domain object that will eventually live in
the database.  In Milestone 0 it is created in-memory.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from startary.domain.enums import WorkflowPhase


class StartupProject(BaseModel):
    """Identity and current phase tracking for a single founder's workspace."""

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique project identifier (UUID string).",
    )
    title: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Human-readable project / startup name.",
    )
    phase: WorkflowPhase = Field(
        default=WorkflowPhase.UNDERSTAND,
        description="Current workflow phase.",
    )
    iteration: int = Field(
        default=1,
        ge=1,
        description="Current learning-loop iteration (starts at 1).",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of project creation.",
    )

    model_config = {"frozen": True}

