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
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


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
    start_timestamp: Optional[datetime] = Field(
        default=None,
        description="Timestamp when call started.",
    )
    end_timestamp: Optional[datetime] = Field(
        default=None,
        description="Timestamp when call completed.",
    )
    project_id: str = Field(
        description="The project context in which this trace was captured.",
    )
    model_name: Optional[str] = Field(
        default=None,
        description="Model identifier used for this call.",
    )
    prompt_tokens: Optional[int] = Field(
        default=None,
        description="Number of prompt/input tokens consumed.",
    )
    completion_tokens: Optional[int] = Field(
        default=None,
        description="Number of completion/output tokens generated.",
    )
    total_tokens: Optional[int] = Field(
        default=None,
        description="Total tokens consumed.",
    )
    retry_count: int = Field(
        default=0,
        description="Number of retries attempted.",
    )
    timeout_status: bool = Field(
        default=False,
        description="Whether this call timed out.",
    )
    schema_name: Optional[str] = Field(
        default=None,
        description="Structured output schema name.",
    )
    turn_id: Optional[str] = Field(
        default=None,
        description="Founder turn ID or question ID.",
    )
    iteration: Optional[int] = Field(
        default=None,
        description="Learning cycle iteration.",
    )
    execution_path: Optional[str] = Field(
        default=None,
        description="Classification category: FAST_LOCAL_UPDATE | CONVERSATIONAL_REPAIR | DELTA_STATE_UPDATE | STRATEGIC_REASONING | FULL_EVIDENCE.",
    )
    skipped_by_fast_path: bool = Field(
        default=False,
        description="Whether this reasoning step was skipped by a fast path.",
    )

    configured_model: Optional[str] = Field(
        default=None,
        description="Configured model identifier requested.",
    )
    resolved_model: Optional[str] = Field(
        default=None,
        description="Actual resolved model identifier used.",
    )
    fallback_used: bool = Field(
        default=False,
        description="Whether a fallback model or deterministic path was used.",
    )
    fallback_reason: Optional[str] = Field(
        default=None,
        description="Reason why fallback was selected.",
    )

    @field_validator("model_name", "configured_model", "resolved_model", mode="before")
    @classmethod
    def coerce_model_name(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        if isinstance(v, str):
            return v
        if hasattr(v, "_mock_name") or "Mock" in type(v).__name__:
            return "mock_model"
        return str(v)

    @field_validator("execution_path", "schema_name", "turn_id", "fallback_reason", mode="before")
    @classmethod
    def coerce_str_fields(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        if isinstance(v, str):
            return v
        if hasattr(v, "_mock_name") or "Mock" in type(v).__name__:
            return "mock_str"
        return str(v)



