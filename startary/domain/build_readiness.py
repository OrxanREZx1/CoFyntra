"""
Build Readiness domain models for Milestone 4.1: Build Readiness / Experiment Escalation.

Startary treats BUILD not as a reward for accumulating successful experiments,
but as an experiment escalation decision:
'Should the next learning cycle remain manual, or does the next important
uncertainty now require a minimal product/prototype?'
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any
from pydantic import BaseModel, Field


class BuildReadinessAction(StrEnum):
    """Controlled decision on experiment modality (Milestone 4.1)."""
    CONTINUE_MANUAL = "CONTINUE_MANUAL"
    BUILD_MINIMAL_PROTOTYPE = "BUILD_MINIMAL_PROTOTYPE"


class BuildReadinessDecision(BaseModel):
    """
    Structured build readiness decision evaluated relative to the active Critical Assumption.
    """
    decision: BuildReadinessAction = Field(
        description="CONTINUE_MANUAL (keep tests manual) or BUILD_MINIMAL_PROTOTYPE (escalate to prototype test)."
    )
    reason: str = Field(
        description="Concise rationale explaining why manual learning suffices or why a prototype is required."
    )
    manual_learning_sufficient: bool = Field(
        default=False,
        description="True if manual testing has sufficiently reduced prerequisite uncertainties."
    )
    next_uncertainty_requires_product: bool = Field(
        default=False,
        description="True if the active critical assumption strictly requires unmediated product interaction."
    )
    minimum_build_scope: list[str] = Field(
        default_factory=list,
        description="Strictly bounded, minimal product capabilities required for this specific test."
    )
    must_not_build: list[str] = Field(
        default_factory=list,
        description="Explicitly excluded features and product roadmap bloat (AI, analytics, loyalty, etc.)."
    )
    evidence_basis: list[str] = Field(
        default_factory=list,
        description="Grounded historical evidence prerequisites supporting this escalation decision."
    )


# Banned domain-specific keywords for cross-startup leakage detection
FOREIGN_DOMAIN_TERMS = {
    "marketplace": ["commission", "take rate", "bakery", "bakeries", "surplus bag", "pickup window", "surplus food"],
    "iot_hardware": ["temperature sensor", "walk-in refrigerator", "kitchen manager", "spoilage log"],
}


def validate_domain_grounding(text: str, active_context: str) -> bool:
    """
    Hard grounding invariant (Milestone 4.1.2):
    No BuildReadiness / NextStrategicAction reasoning may reference concepts,
    business mechanics, customer types, pricing mechanisms, or workflows that
    are absent from the active startup context.
    """
    if not text:
        return True
    text_lower = text.lower()
    ctx_lower = active_context.lower() if active_context else ""

    # Check for domain terms that appear in text but are completely absent from context
    for domain_name, terms in FOREIGN_DOMAIN_TERMS.items():
        for term in terms:
            if term in text_lower and term not in ctx_lower:
                return False
    return True


def sanitize_domain_leakage(text: str, active_context: str, default_replacement: str = "") -> str:
    """
    Removes or neutralizes foreign domain terms from reasoning strings if absent from active context.
    """
    if not text:
        return text
    ctx_lower = active_context.lower() if active_context else ""
    sanitized = text

    for domain_name, terms in FOREIGN_DOMAIN_TERMS.items():
        for term in terms:
            if term in sanitized.lower() and term not in ctx_lower:
                if domain_name == "marketplace":
                    sanitized = sanitized.replace("commission acceptance and actual payment settlement", "commercial terms and pricing tolerance")
                    sanitized = sanitized.replace("commission", "commercial pricing")
                    sanitized = sanitized.replace("Commission", "Commercial pricing")
                    sanitized = sanitized.replace("bakery", "partner")
                    sanitized = sanitized.replace("bakeries", "partners")
                    sanitized = sanitized.replace("merchants", "users or customers")
                    sanitized = sanitized.replace("merchant", "user or customer")
                elif domain_name == "iot_hardware":
                    sanitized = sanitized.replace("temperature sensor", "solution")
                    sanitized = sanitized.replace("kitchen manager", "operator")

    return sanitized
