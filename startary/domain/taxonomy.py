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

