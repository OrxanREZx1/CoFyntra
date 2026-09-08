"""
AreaKnowledge schema.

Represents Startary's structured understanding of one startup area.

M1 design decisions:

1. STATUS — We use a simple qualitative enum (UNKNOWN / PARTIALLY_KNOWN / KNOWN / UNCLEAR)
   rather than a numerical confidence score.  Numerical scores would create false precision
   at this stage: the LLM cannot reliably calibrate a 0.73 confidence.  An ordinal
   status is honest about what we actually know.

2. FACT / BELIEF / EVIDENCE SEPARATION — Three distinct lists:
   - understanding:       what can be stated as direct context from the founder
   - founder_claims:      things the founder asserts about users/market without evidence
   - evidence_provided:   concrete observations/data the founder has actually collected

   This distinction is the foundation for Milestone 2's Reality Check, which will
   challenge unsupported claims.  It must be established here, not retrofitted later.

3. MISSING INFORMATION — Explicitly represented as a string, not just an absent key.
   When the LLM identifies what would be needed to clarify an area, that belongs
   in the domain object so future nodes can act on it.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class AreaStatus(StrEnum):
    """
    Qualitative status of knowledge about one startup area.

    UNKNOWN         — no information provided by founder
    PARTIALLY_KNOWN — some information present but significant gaps remain
    KNOWN           — sufficient information to form a working picture
    UNCLEAR         — information was provided but it is ambiguous or contradictory
    """

    UNKNOWN = "UNKNOWN"
    PARTIALLY_KNOWN = "PARTIALLY_KNOWN"
    KNOWN = "KNOWN"
    UNCLEAR = "UNCLEAR"


class AreaKnowledge(BaseModel):
    """
    Startary's structured understanding of one startup area after diagnosis.

    This is NOT a raw founder quote.  It is the system's interpreted
    representation of what has been established, claimed, evidenced, and
    what remains unknown.

    Rules:
    - understanding:    only what is directly established by founder context.
                        Do not invent information.
    - founder_claims:   beliefs/assertions the founder made without evidence.
                        Presence here does NOT mean they are validated.
    - evidence_provided: concrete things the founder actually observed or collected.
                        An informal interview counts; a market assumption does not.
    - missing_information: what would materially improve understanding of this area.
    """

    status: AreaStatus = Field(
        default=AreaStatus.UNKNOWN,
        description="Qualitative knowledge status for this area.",
    )

    understanding: str | None = Field(
        default=None,
        description=(
            "What is directly established about this area from founder context. "
            "None if nothing is known."
        ),
    )

    founder_claims: list[str] = Field(
        default_factory=list,
        description=(
            "Claims or beliefs the founder stated about this area "
            "that are not yet supported by evidence."
        ),
    )

    evidence_provided: list[str] = Field(
        default_factory=list,
        description=(
            "Concrete observations, interviews, or data points the founder "
            "has actually collected. Empty list = no evidence."
        ),
    )

    missing_information: str | None = Field(
        default=None,
        description=(
            "What specific information would reduce uncertainty about this area. "
            "None if the area is KNOWN and gaps are minor."
        ),
    )
