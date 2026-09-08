"""
ExecutionContext — Startary Milestone 2.6.

Represents the practical operating reality of a startup:
team, capabilities, resources, budget, customer access, geography, assets.

Design principles:
  - All fields are optional (KnowledgeField defaults to UNKNOWN).
  - Raw facts are stored here; strategic risk assessment belongs in diagnosis.
  - funding_status / funding_amount_raised are SEPARATE from available_budget.
  - users / active_users / paying_customers / revenue are NOT stored here — they belong
    in StartupProgress.
  - Geography is only recorded if explicitly mentioned by the founder.
  - Do NOT infer founder nationality or location from market geography.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from startary.domain.provenance import KnowledgeField, unknown_field


class TeamMember(BaseModel):
    """
    A single team member's known attributes.
    All fields optional — never invent what the founder did not provide.
    """

    name: Optional[str] = Field(
        default=None,
        description="Name or identifier if supplied.",
    )
    role: Optional[str] = Field(
        default=None,
        description="Role or title (e.g. 'developer', 'designer', 'CEO').",
    )
    capabilities: list[str] = Field(
        default_factory=list,
        description="Specific capabilities mentioned (e.g. 'AI', 'backend', 'design').",
    )
    weekly_hours: Optional[float] = Field(
        default=None,
        description="Approximate weekly hours this member can contribute.",
    )


class ExecutionContext(BaseModel):
    """
    Practical startup operating reality extracted from founder messages.

    Separation principle:
      RAW EXECUTION FACTS (stored here) ≠ STRATEGIC RISK ASSESSMENT (in Diagnosis).

    All KnowledgeField instances default to UNKNOWN.
    Lists default to empty (no assets, no team members, etc.).

    Important constraints:
      - budget_amount and funding_amount_raised are SEPARATE.
      - An unknown field must remain UNKNOWN; do not zero-fill.
      - total_weekly_capacity_hours should be INFERRED (not EXPLICIT) unless stated directly.
    """

    # ── Team ────────────────────────────────────────────────────────
    founder_count: KnowledgeField = Field(
        default_factory=unknown_field,
        description="Number of co-founders (int). EXPLICIT only if stated directly.",
    )
    active_team_size: KnowledgeField = Field(
        default_factory=unknown_field,
        description="Total number of actively working team members (int).",
    )
    team_members: list[TeamMember] = Field(
        default_factory=list,
        description="Structured team member entries if individual details were provided.",
    )

    # ── Capabilities ─────────────────────────────────────────────────
    technical_capability: KnowledgeField = Field(
        default_factory=unknown_field,
        description="Team has technical / engineering capability (bool).",
    )
    product_capability: KnowledgeField = Field(
        default_factory=unknown_field,
        description="Team has product management capability (bool).",
    )
    design_capability: KnowledgeField = Field(
        default_factory=unknown_field,
        description="Team has UI/UX design capability (bool).",
    )
    marketing_capability: KnowledgeField = Field(
        default_factory=unknown_field,
        description="Team has marketing / GTM capability (bool).",
    )
    sales_capability: KnowledgeField = Field(
        default_factory=unknown_field,
        description="Team has direct sales capability (bool).",
    )
    domain_expertise: KnowledgeField = Field(
        default_factory=unknown_field,
        description="Team has relevant domain expertise (description string).",
    )

    # ── Time capacity ────────────────────────────────────────────────
    approximate_weekly_hours_per_member: KnowledgeField = Field(
        default_factory=unknown_field,
        description="Approximate weekly hours per team member (float). EXPLICIT if stated.",
    )
    total_weekly_capacity_hours: KnowledgeField = Field(
        default_factory=unknown_field,
        description=(
            "Total team weekly capacity in hours (float). "
            "Should be INFERRED (team_size × avg_hours) unless stated directly."
        ),
    )

    # ── Budget (separate from funding) ───────────────────────────────
    available_budget_amount: KnowledgeField = Field(
        default_factory=unknown_field,
        description="Approximate experiment / operational budget available (float).",
    )
    budget_currency: KnowledgeField = Field(
        default_factory=unknown_field,
        description="Currency code/name for budget (str). Do not assume USD.",
    )
    budget_period: KnowledgeField = Field(
        default_factory=unknown_field,
        description="Budget period (str): 'month', 'week', 'total', etc.",
    )

    # ── Funding (separate from budget) ───────────────────────────────
    funding_status: KnowledgeField = Field(
        default_factory=unknown_field,
        description="Funding status (str): 'bootstrapped', 'pre-seed', 'seed', etc.",
    )
    funding_amount_raised: KnowledgeField = Field(
        default_factory=unknown_field,
        description="Amount raised if explicitly stated (float).",
    )

    # ── Customer access ──────────────────────────────────────────────
    customer_access: KnowledgeField = Field(
        default_factory=unknown_field,
        description="Description of access to potential customers (str).",
    )
    existing_audience_or_community: KnowledgeField = Field(
        default_factory=unknown_field,
        description="Existing audience, community, email list, or social following (str).",
    )
    partnerships: list[str] = Field(
        default_factory=list,
        description="Named partnerships or institutional relationships.",
    )
    existing_assets: list[str] = Field(
        default_factory=list,
        description=(
            "Existing assets that could support experiment design: "
            "landing page, waitlist, email list, dataset, codebase, community, etc."
        ),
    )

    # ── Geography ────────────────────────────────────────────────────
    geographic_market: KnowledgeField = Field(
        default_factory=unknown_field,
        description=(
            "Target geographic market (str). "
            "Only record if founder explicitly mentioned a geography. "
            "Do NOT infer founder location from target market."
        ),
    )

    def is_empty(self) -> bool:
        """True if no execution context information has been extracted."""
        fields_with_values = [
            self.founder_count.is_known(),
            self.active_team_size.is_known(),
            bool(self.team_members),
            self.technical_capability.is_known(),
            self.total_weekly_capacity_hours.is_known(),
            self.available_budget_amount.is_known(),
            self.funding_status.is_known(),
            self.customer_access.is_known(),
            self.geographic_market.is_known(),
            bool(self.existing_assets),
        ]
        return not any(fields_with_values)
