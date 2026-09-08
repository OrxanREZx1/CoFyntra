"""
StartupProgress — Startary Milestone 2.6.

Represents the actual maturity and validation progress of a startup.

Design principles:
  - active_users / paying_customers / revenue are semantically SEPARATE.
    Never transform "50 users" → "50 paying customers".
    Never transform "no revenue" → "no users".
  - mvp_exists = false does NOT imply development_stage = PROBLEM_EXPLORATION.
    The system must distinguish safe facts from interpretation.
  - Keep revenue as a string to avoid false numeric precision.
  - launched = true does NOT imply paying_customers > 0.
  - All fields default to UNKNOWN provenance.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from startary.domain.provenance import KnowledgeField, unknown_field


class DevelopmentStage(StrEnum):
    """
    Controlled vocabulary for a startup's development stage.

    Provenance notes:
      - Founder says "we only have an idea" → RAW_IDEA, EXPLICIT
      - Founder says "we have Figma screens but no working product" →
          stage may be PROTOTYPE, provenance INFERRED
      - Absent stage info → UNKNOWN (do not guess)
    """

    RAW_IDEA = "RAW_IDEA"
    PROBLEM_EXPLORATION = "PROBLEM_EXPLORATION"
    SOLUTION_EXPLORATION = "SOLUTION_EXPLORATION"
    PROTOTYPE = "PROTOTYPE"
    PRE_MVP = "PRE_MVP"
    MVP = "MVP"
    EARLY_USAGE = "EARLY_USAGE"
    EARLY_REVENUE = "EARLY_REVENUE"
    GROWTH = "GROWTH"
    UNKNOWN = "UNKNOWN"


class ProductStatus(StrEnum):
    """
    Current state of the product artifact.

    NOTHING   — No product exists yet
    CONCEPT   — Documented idea, wireframes, or design doc
    PROTOTYPE — Clickable or functional prototype (may not be user-ready)
    MVP_BUILT — Minimum viable product built and ready for user testing
    LAUNCHED  — Product actively available to real users
    UNKNOWN   — Insufficient information
    """

    NOTHING = "NOTHING"
    CONCEPT = "CONCEPT"
    PROTOTYPE = "PROTOTYPE"
    MVP_BUILT = "MVP_BUILT"
    LAUNCHED = "LAUNCHED"
    UNKNOWN = "UNKNOWN"


class StartupProgress(BaseModel):
    """
    What has actually been built, shipped, validated, and achieved.

    Semantic constraints (enforced by prompts, not Pydantic):
      - active_users != paying_customers (do NOT equate them)
      - revenue = None does NOT imply users = 0
      - mvp_exists = false does NOT imply development_stage = PROBLEM_EXPLORATION
      - launched = True does NOT imply any revenue
    """

    # ── Product state ────────────────────────────────────────────────
    development_stage: KnowledgeField = Field(
        default_factory=unknown_field,
        description="DevelopmentStage value (str). EXPLICIT if stated; INFERRED if derived.",
    )
    product_status: KnowledgeField = Field(
        default_factory=unknown_field,
        description="ProductStatus value (str). What product artifact currently exists.",
    )
    prototype_exists: KnowledgeField = Field(
        default_factory=unknown_field,
        description="bool — does a prototype exist? EXPLICIT only.",
    )
    mvp_exists: KnowledgeField = Field(
        default_factory=unknown_field,
        description="bool — does an MVP exist? EXPLICIT only.",
    )
    launched: KnowledgeField = Field(
        default_factory=unknown_field,
        description="bool — has the product been launched to real users? EXPLICIT only.",
    )

    # ── Validation activity ──────────────────────────────────────────
    validation_activity: KnowledgeField = Field(
        default_factory=unknown_field,
        description="Summary of validation activities performed (str).",
    )
    interviews_completed: KnowledgeField = Field(
        default_factory=unknown_field,
        description="Number of user/customer interviews completed (int). EXPLICIT only.",
    )

    # ── Usage & revenue ──────────────────────────────────────────────
    # IMPORTANT: These three fields are semantically separate.
    active_users: KnowledgeField = Field(
        default_factory=unknown_field,
        description="Number of active users (int). Do NOT infer from revenue.",
    )
    paying_customers: KnowledgeField = Field(
        default_factory=unknown_field,
        description="Number of paying customers (int). Separate from active_users.",
    )
    revenue: KnowledgeField = Field(
        default_factory=unknown_field,
        description=(
            "Revenue description (str, NOT numeric). "
            "E.g. '$500/month', 'zero revenue', '2 paying pilots'. "
            "Keep as string to avoid false precision."
        ),
    )

    # ── Milestones ───────────────────────────────────────────────────
    major_milestones: list[str] = Field(
        default_factory=list,
        description="Major milestones already achieved (e.g. 'first paying customer', 'accelerator acceptance').",
    )

    def is_empty(self) -> bool:
        """True if no progress information has been extracted."""
        fields_with_values = [
            self.development_stage.is_known(),
            self.product_status.is_known(),
            self.prototype_exists.is_known(),
            self.mvp_exists.is_known(),
            self.launched.is_known(),
            self.validation_activity.is_known(),
            self.interviews_completed.is_known(),
            self.active_users.is_known(),
            self.paying_customers.is_known(),
            self.revenue.is_known(),
            bool(self.major_milestones),
        ]
        return not any(fields_with_values)
