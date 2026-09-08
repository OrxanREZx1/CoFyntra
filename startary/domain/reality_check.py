"""
Reality Check schema for Startary Milestone 2.

Reality Check analyzes what the founder currently believes in a specific
startup area against what available founder-provided evidence actually justifies.

Rules:
  - Scoped strictly to the selected weak-link area.
  - Relies solely on founder-provided evidence already present in StartupState.
  - Distinguishes between what evidence supports vs. what it does NOT support.
  - Preserves inconclusive or no-evidence states without forcing binary validation.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, Field

from startary.domain.taxonomy import StartupArea


class EvidenceConclusion(StrEnum):
    """
    Epistemic conclusion of a reality check.

    SUPPORTED           — Evidence directly and adequately supports the founder's claims.
    PARTIALLY_SUPPORTED — Evidence supports part of the claim or an initial subset, but gaps remain.
    UNSUPPORTED         — Founder makes claims but no valid supporting evidence was provided.
    CONTRADICTED        — Collected evidence directly opposes the founder's claims.
    INCONCLUSIVE        — Available evidence is mixed, ambiguous, or insufficient to decide.
    NO_EVIDENCE         — No observations, interviews, or data exist for this area.
    """

    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    INCONCLUSIVE = "INCONCLUSIVE"
    NO_EVIDENCE = "NO_EVIDENCE"


class RealityCheck(BaseModel):
    """
    A structured Reality Check evaluation for a single weak-link startup area.
    """

    weak_link_area: StartupArea = Field(
        description="The startup area evaluated by this Reality Check."
    )

    relevant_founder_claims: list[str] = Field(
        default_factory=list,
        description="Founder beliefs or assertions regarding this area.",
    )

    available_founder_evidence: list[str] = Field(
        default_factory=list,
        description="Observations, interviews, or data points actually provided by founder.",
    )

    what_evidence_supports: list[str] = Field(
        default_factory=list,
        description="Specific facts or sub-claims justified by the available evidence.",
    )

    what_evidence_does_not_support: list[str] = Field(
        default_factory=list,
        description="Assumptions or extrapolations that the evidence does NOT validate.",
    )

    conclusion: EvidenceConclusion = Field(
        description="Overall epistemic conclusion regarding the founder claims in this area."
    )

    unresolved_uncertainty: str = Field(
        description="The primary remaining question or unknown in this area."
    )

    summary: str = Field(
        description="Analytical synthesis of claims vs evidence for this area."
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of when this Reality Check was produced.",
    )

