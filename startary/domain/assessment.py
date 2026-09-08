"""
StartupAreaAssessment schema — updated for Milestone 2.

In Milestone 2, each startup area is assessed across 4 qualitative risk dimensions:
  1. Importance: How important is this area to the current startup direction?
  2. Uncertainty: How uncertain is the current understanding of this area?
  3. Evidence Gap: How large is the gap between current claims and supporting evidence?
  4. Consequence of Being Wrong: What is the downside risk if this area is misunderstood?

Factors use a controlled ordinal scale:
  LOW = 1, MEDIUM = 2, HIGH = 3

Decision Risk is calculated deterministically in Python:
  Decision Risk = Importance × Uncertainty × Evidence Gap × Consequence of Being Wrong
  (Score ranges from 1 × 1 × 1 × 1 = 1 to 3 × 3 × 3 × 3 = 81)
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from startary.domain.area_knowledge import AreaStatus
from startary.domain.taxonomy import StartupArea


class RiskFactorLevel(StrEnum):
    """Controlled qualitative ordinal scale for risk factor evaluation."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


def risk_factor_to_int(level: RiskFactorLevel | str | None) -> int:
    """Convert an ordinal risk factor level to its integer equivalent (1, 2, 3)."""
    if level is None:
        return 1
    if isinstance(level, str):
        val = level.upper()
        if val == "LOW":
            return 1
        elif val == "MEDIUM":
            return 2
        elif val == "HIGH":
            return 3
    raise ValueError(f"Invalid risk factor level: {level!r}. Expected LOW, MEDIUM, or HIGH.")


class StartupAreaAssessment(BaseModel):
    """
    A structured risk assessment of one startup area.

    The LLM assesses the qualitative factors (LOW, MEDIUM, HIGH) and rationale.
    Deterministic Python calculates `calculated_decision_risk`.
    """

    area: StartupArea = Field(description="The startup area being assessed.")

    status: AreaStatus | None = Field(
        default=None,
        description="Knowledge status inferred by the diagnosis module.",
    )

    summary: str | None = Field(
        default=None,
        description="What is currently understood about this area.",
    )

    importance: RiskFactorLevel | None = Field(
        default=None,
        description="Contextual importance to the current startup direction (LOW/MEDIUM/HIGH).",
    )

    uncertainty: RiskFactorLevel | None = Field(
        default=None,
        description="Level of uncertainty in current understanding (LOW/MEDIUM/HIGH).",
    )

    evidence_gap: RiskFactorLevel | None = Field(
        default=None,
        description="Disparity between founder claims and available evidence (LOW/MEDIUM/HIGH).",
    )

    consequence_if_wrong: RiskFactorLevel | None = Field(
        default=None,
        description="Potential severity if this area is misunderstood (LOW/MEDIUM/HIGH).",
    )

    calculated_decision_risk: int | None = Field(
        default=None,
        ge=1,
        le=81,
        description="Deterministic product: Importance × Uncertainty × Evidence Gap × Consequence.",
    )

    rationale: str | None = Field(
        default=None,
        description="Brief reasoning explaining the assigned factor levels.",
    )

    evidence_gap_details: str | None = Field(
        default=None,
        description="What specific evidence is missing to reduce uncertainty.",
    )

    def calculate_risk(self) -> int:
        """
        Calculate and update the deterministic decision-risk score.

        Returns:
            The calculated decision-risk integer (1–81).
        """
        imp = risk_factor_to_int(self.importance)
        unc = risk_factor_to_int(self.uncertainty)
        gap = risk_factor_to_int(self.evidence_gap)
        con = risk_factor_to_int(self.consequence_if_wrong)

        score = imp * unc * gap * con
        self.calculated_decision_risk = score
        return score
