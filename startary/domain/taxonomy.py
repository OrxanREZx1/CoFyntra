"""
Canonical startup area taxonomy.

This is the SINGLE authoritative definition of startup areas used
across the entire Startary system.  All domain schemas, diagnosis
modules, and routing logic must import from here — never re-define
these strings elsewhere.

Adding a new area: extend the enum below.  No other file needs to change
to accommodate a new area, provided downstream code iterates over
StartupArea members rather than hard-coding strings.
"""

from dataclasses import dataclass
from enum import StrEnum


class StartupArea(StrEnum):
    """The ten canonical areas Startary evaluates for a startup."""

    PROBLEM = "Problem"
    TARGET_USER = "Target User"
    PAIN_URGENCY = "Pain / Urgency"
    EXISTING_ALTERNATIVES = "Existing Alternatives"
    VALUE_PROPOSITION = "Value Proposition"
    DEMAND_SIGNAL = "Demand Signal"
    ACQUISITION = "Acquisition / Customer Access"
    BUSINESS_MODEL = "Business Model / Revenue"
    SOLUTION_MVP = "Solution / MVP"
    FOUNDER_CONSTRAINTS = "Founder Constraints"


@dataclass(frozen=True)
class AreaScopeGuidance:
    """Explicit epistemic boundaries for critical assumptions in a canonical startup area (Milestone 4.0.7.1.3)."""
    area: StartupArea
    focus: str
    allowed_concepts: list[str]
    forbidden_concepts: list[str]


AREA_SCOPE_GUIDANCE: dict[StartupArea, AreaScopeGuidance] = {
    StartupArea.PROBLEM: AreaScopeGuidance(
        area=StartupArea.PROBLEM,
        focus="What recurring problem/event exists and what consequence it creates.",
        allowed_concepts=[
            "recurring failure events",
            "workflow friction",
            "operational breakdowns",
            "avoidable loss or waste",
            "consequence of the problem",
        ],
        forbidden_concepts=[
            "solution adoption",
            "willingness to pay",
            "purchasing",
            "product evaluation",
            "solution preference",
        ],
    ),
    StartupArea.TARGET_USER: AreaScopeGuidance(
        area=StartupArea.TARGET_USER,
        focus="Who consistently experiences, owns, responds to, or decides about the problem.",
        allowed_concepts=[
            "repeatable customer segment",
            "problem ownership",
            "user roles",
            "responder roles",
            "decision-maker roles",
            "operational profiles",
            "segment boundaries",
        ],
        forbidden_concepts=[
            "adoption",
            "willingness to pay",
            "product preference",
            "differentiation",
            "purchase intent",
            "commercial conversion",
        ],
    ),
    StartupArea.PAIN_URGENCY: AreaScopeGuidance(
        area=StartupArea.PAIN_URGENCY,
        focus="Whether the problem is serious, frequent, costly, disruptive, or prioritized enough to act on.",
        allowed_concepts=[
            "operational disruption",
            "frequency of pain",
            "cost of inaction",
            "priority over competing demands",
            "urgency to resolve",
        ],
        forbidden_concepts=[
            "willingness to pay",
            "commercial pricing",
            "product preference",
        ],
    ),
    StartupArea.EXISTING_ALTERNATIVES: AreaScopeGuidance(
        area=StartupArea.EXISTING_ALTERNATIVES,
        focus="What users currently do instead and whether those approaches adequately address the need.",
        allowed_concepts=[
            "manual workarounds",
            "competitor tools",
            "status quo practices",
            "gaps and limitations in existing approaches",
        ],
        forbidden_concepts=[
            "implying alternative inadequacy validates our proposed solution",
            "requiring product adoption",
        ],
    ),
    StartupArea.VALUE_PROPOSITION: AreaScopeGuidance(
        area=StartupArea.VALUE_PROPOSITION,
        focus="Whether the proposed outcome/benefit is meaningfully useful or preferable.",
        allowed_concepts=[
            "operational improvement over status quo",
            "meaningful utility of proposed outcome",
            "user preference for the improved outcome",
        ],
        forbidden_concepts=[
            "mandatory willingness to pay",
            "commercial pricing conversion",
            "contract commitment",
        ],
    ),
    StartupArea.DEMAND_SIGNAL: AreaScopeGuidance(
        area=StartupArea.DEMAND_SIGNAL,
        focus="Whether users demonstrate credible behavioral commitment or intent beyond polite interest.",
        allowed_concepts=[
            "trial intent",
            "evaluation actions",
            "behavioral commitment",
            "active engagement",
            "pilot participation",
        ],
        forbidden_concepts=[
            "redefining the customer segment",
            "pricing model viability",
        ],
    ),
    StartupArea.ACQUISITION: AreaScopeGuidance(
        area=StartupArea.ACQUISITION,
        focus="Whether the startup can reliably reach the relevant customer group through realistic channels.",
        allowed_concepts=[
            "identifiable outreach channels",
            "contactability",
            "response rates from cold or organic access",
            "customer discovery channels",
        ],
        forbidden_concepts=[
            "proving problem severity",
            "proving product value",
        ],
    ),
    StartupArea.BUSINESS_MODEL: AreaScopeGuidance(
        area=StartupArea.BUSINESS_MODEL,
        focus="Whether value can translate into payment/revenue under a plausible economic model.",
        allowed_concepts=[
            "willingness to pay",
            "pricing tiers",
            "budget allocation",
            "unit economics",
            "economic return for customer",
        ],
        forbidden_concepts=[
            "redefining basic problem existence without economic viability",
        ],
    ),
    StartupArea.SOLUTION_MVP: AreaScopeGuidance(
        area=StartupArea.SOLUTION_MVP,
        focus="Whether the proposed solution can deliver the required outcome in actual use.",
        allowed_concepts=[
            "technical feasibility",
            "workflow execution",
            "delivering core promise in user hands",
            "functional reliability",
        ],
        forbidden_concepts=[
            "assuming technical delivery proves customer demand",
        ],
    ),
    StartupArea.FOUNDER_CONSTRAINTS: AreaScopeGuidance(
        area=StartupArea.FOUNDER_CONSTRAINTS,
        focus="Whether the founder/team can realistically execute the next stage under time, skills, budget, access, and resource constraints.",
        allowed_concepts=[
            "domain access",
            "execution bandwidth",
            "technical capability",
            "runway",
            "team capacity",
        ],
        forbidden_concepts=[
            "inferring customer demand from founder capability",
        ],
    ),
}

