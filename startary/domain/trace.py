"""
AITrace schema.

Lightweight trace contract for future AI module debuggability.

Each time a future AI node runs, it should emit an AITrace capturing
what went in, what came out, and how long it took.  This enables
debugging without a full observability platform.

No AI calls happen in Milestone 0 — this schema is defined only.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class AITrace(BaseModel):
    """
    Execution trace for one AI module invocation.

    Kept deliberately small — not an observability platform.
    Future milestones may add trace storage, but the contract is stable.
    """

    module_name: str = Field(
        ...,
        min_length=1,
        description="Name of the LangGraph node or AI module that was executed.",
    )
    prompt_version: str = Field(
        ...,
        min_length=1,
        description="Version identifier for the prompt template used (e.g. 'v1.0').",
    )
    input_data: dict = Field(
        default_factory=dict,
        description="Structured input passed to the module.",
    )
    output_data: dict = Field(
        default_factory=dict,
        description="Structured output returned by the module.",
    )
    latency_ms: float = Field(
        ge=0.0,
        description="Wall-clock execution time in milliseconds.",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of when the module was invoked.",
    )
    project_id: str = Field(
        description="The project context in which this trace was captured.",
    )

