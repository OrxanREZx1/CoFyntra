"""
Customer Access Domain Model — Milestone 4.0.1.

Evaluates evidence of customer access from across the entire startup state:
historical MVP testers, user conversations, active users, community audience,
and explicit founder context.
"""

from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, Field


class CustomerAccessStatus(str, Enum):
    """Overall status of customer access."""
    NONE = "NONE"
    LIMITED = "LIMITED"
    ESTABLISHED = "ESTABLISHED"
    UNKNOWN = "UNKNOWN"


class CustomerAccessStrength(str, Enum):
    """Strength of customer access evidence."""
    WEAK = "WEAK"
    MODERATE = "MODERATE"
    STRONG = "STRONG"


class CustomerAccessAssessment(BaseModel):
    """
    Structured assessment of the founder's demonstrated customer access.
    Distinguishes complete lack of access from demonstrated past access
    that may still be insufficient for specific new experiment sample requirements.
    """
    access_status: CustomerAccessStatus = Field(
        ...,
        description="NONE | LIMITED | ESTABLISHED | UNKNOWN",
    )
    access_strength: CustomerAccessStrength = Field(
        ...,
        description="WEAK | MODERATE | STRONG",
    )
    evidence_sources: list[str] = Field(
        default_factory=list,
        description="Specific state sources that demonstrate customer access (e.g. 'previous MVP testers', 'student interviews').",
    )
    relevant_segments: list[str] = Field(
        default_factory=list,
        description="Target customer segments with demonstrated contact.",
    )
    explanation: str = Field(
        ...,
        description="Factually grounded summary of demonstrated access.",
    )

