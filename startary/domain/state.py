"""
StartupState schema — Milestone 2.6.

Milestone 2.6 adds two new conceptual layers to the startup state:

  B. EXECUTION CONTEXT (team, budget, capabilities, customer access, geography, assets)
  C. STARTUP PROGRESS  (development stage, MVP, users, revenue, validation)

These join the existing:

  A. STRATEGIC STATE (10-area diagnosis: problem, users, pain, etc.)
  D. LEARNING STATE  (assumptions, experiments, evidence — future milestones)

Architecture contract:
  - `areas` maps StartupArea → AreaKnowledge (strategic diagnosis, existing M1–M2.5)
  - `execution_context` holds raw operational facts (M2.6)
  - `progress` holds startup maturity facts (M2.6)
  - raw_context stores verbatim founder input (separate from interpreted state)
  - LLM reads founder messages → returns structured deltas
  - Deterministic Python merges deltas → updates state
  - LLM NEVER directly mutates durable state

Backward compatibility:
  - `areas` and `raw_context` are unchanged.
  - Existing code that only touches `areas` continues to work.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from startary.domain.area_knowledge import AreaKnowledge, AreaStatus
from startary.domain.execution_context import ExecutionContext
from startary.domain.startup_progress import StartupProgress
from startary.domain.taxonomy import StartupArea


class StartupState(BaseModel):
    """
    Structured snapshot of what Startary currently knows about a startup.

    Four conceptual layers (Milestone 2.6):
      A. STRATEGIC STATE:   `areas` — 10-area diagnosis
      B. EXECUTION CONTEXT: `execution_context` — team, budget, capabilities
      C. STARTUP PROGRESS:  `progress` — product, users, revenue, validation
      D. LEARNING STATE:    assumptions/experiments in graph state (M3+)

    ``raw_context`` holds the verbatim founder input, separate from interpreted state.
    """

    raw_context: str | None = Field(
        default=None,
        description=(
            "Verbatim founder-provided startup description. "
            "This is the input to diagnosis, NOT a startup area value."
        ),
    )

    areas: dict[StartupArea, AreaKnowledge] = Field(
        default_factory=dict,
        description=(
            "Map of startup area → structured knowledge. "
            "Absent keys = not yet diagnosed. "
            "All values are AreaKnowledge objects with explicit status."
        ),
    )

    execution_context: ExecutionContext = Field(
        default_factory=ExecutionContext,
        description=(
            "Practical operating reality: team, budget, capabilities, customer access. "
            "All fields default to UNKNOWN provenance until extracted from founder messages."
        ),
    )

    progress: StartupProgress = Field(
        default_factory=StartupProgress,
        description=(
            "Startup maturity and validation progress: "
            "product status, MVP, users, revenue, validation activity. "
            "All fields default to UNKNOWN provenance."
        ),
    )

    # ── Strategic state helpers (existing) ───────────────────────────

    def known_areas(self) -> list[StartupArea]:
        """Return areas with status KNOWN or PARTIALLY_KNOWN."""
        return [
            a for a, k in self.areas.items()
            if k.status in (AreaStatus.KNOWN, AreaStatus.PARTIALLY_KNOWN)
        ]

    def missing_areas(self) -> list[StartupArea]:
        """Return areas not yet diagnosed."""
        return [a for a in StartupArea if a not in self.areas]

    def unknown_areas(self) -> list[StartupArea]:
        """Return areas that were diagnosed but remain UNKNOWN."""
        return [
            a for a, k in self.areas.items()
            if k.status == AreaStatus.UNKNOWN
        ]

    def unclear_areas(self) -> list[StartupArea]:
        """Return areas diagnosed as UNCLEAR (ambiguous/contradictory)."""
        return [
            a for a, k in self.areas.items()
            if k.status == AreaStatus.UNCLEAR
        ]

    def is_empty(self) -> bool:
        """True if no area has been diagnosed yet."""
        return len(self.areas) == 0

    def get_area(self, area: StartupArea) -> AreaKnowledge:
        """
        Return the AreaKnowledge for a given area.
        If the area has not been diagnosed, returns a default UNKNOWN knowledge object.
        """
        return self.areas.get(area, AreaKnowledge(status=AreaStatus.UNKNOWN))
