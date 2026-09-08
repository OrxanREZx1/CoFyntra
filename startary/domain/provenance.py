"""
Epistemic provenance metadata for Startary's Rich Startup State — Milestone 2.6.

These types attach to factual fields (execution context, progress) to preserve
the epistemological status of every extracted value:

  EXPLICIT   — founder directly stated this fact
  INFERRED   — safely derived from founder statements (e.g. total capacity = size × hours)
  UNKNOWN    — no usable information available
  CONFLICTING — founder statements contradict each other; resolution needed

Architecture contract:
  - EXPLICIT always wins over INFERRED for the same field.
  - Newer EXPLICIT overwrites older EXPLICIT.
  - INFERRED NEVER silently overwrites an EXPLICIT value.
  - UNKNOWN never overwrites any known value.
  - CONFLICTING is set when two EXPLICIT values directly contradict.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Optional

from pydantic import BaseModel, Field


class FactProvenance(StrEnum):
    """How a factual value became known."""

    EXPLICIT = "EXPLICIT"         # Founder directly stated this
    INFERRED = "INFERRED"         # Safely derived/interpreted from founder statements
    UNKNOWN = "UNKNOWN"           # No usable information
    CONFLICTING = "CONFLICTING"   # Founder statements contradict; unresolved


class FactConfidence(StrEnum):
    """Qualitative confidence in an extracted or inferred value."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class KnowledgeField(BaseModel):
    """
    A single factual field with epistemic metadata.

    value:
        The extracted or inferred value. None when provenance is UNKNOWN.
    provenance:
        How the value was obtained (EXPLICIT / INFERRED / UNKNOWN / CONFLICTING).
    confidence:
        Qualitative confidence in the value.
    source_turn:
        Clarification turn index when this value was last updated (0 = initial message).
        None if not tracked.
    notes:
        Optional freeform note about the extraction context or conflict.
    """

    value: Any = Field(
        default=None,
        description="The factual value. None when provenance is UNKNOWN.",
    )

    provenance: FactProvenance = Field(
        default=FactProvenance.UNKNOWN,
        description="How this value was determined.",
    )

    confidence: FactConfidence = Field(
        default=FactConfidence.LOW,
        description="Qualitative confidence level.",
    )

    source_turn: Optional[int] = Field(
        default=None,
        description="Conversation turn index when this value was last updated.",
    )

    notes: Optional[str] = Field(
        default=None,
        description="Optional context note about the extraction or conflict.",
    )

    # ------------------------------------------------------------------
    # Merge helpers
    # ------------------------------------------------------------------

    def is_known(self) -> bool:
        """True if this field contains usable information (not UNKNOWN)."""
        return self.provenance not in (FactProvenance.UNKNOWN,)

    def can_be_overwritten_by(self, other: "KnowledgeField") -> bool:
        """
        Whether `other` should replace this field under provenance merge rules:

        - EXPLICIT always wins over INFERRED.
        - Newer EXPLICIT overwrites older EXPLICIT (higher turn index or equal).
        - INFERRED never overwrites EXPLICIT.
        - UNKNOWN never overwrites any known value.
        - CONFLICTING is handled specially by the merge function.
        """
        if self.provenance == FactProvenance.UNKNOWN:
            # Any known value replaces UNKNOWN
            return other.provenance != FactProvenance.UNKNOWN

        if self.provenance == FactProvenance.INFERRED:
            # EXPLICIT wins over INFERRED; equal provenance also allows update
            return other.provenance in (FactProvenance.EXPLICIT, FactProvenance.INFERRED)

        if self.provenance == FactProvenance.EXPLICIT:
            # Only another EXPLICIT can overwrite (newer turn wins or same turn)
            if other.provenance == FactProvenance.EXPLICIT:
                # Allow overwrite by newer or same-turn explicit values
                self_turn = self.source_turn if self.source_turn is not None else -1
                other_turn = other.source_turn if other.source_turn is not None else -1
                return other_turn >= self_turn
            return False

        if self.provenance == FactProvenance.CONFLICTING:
            # Only EXPLICIT can resolve a conflict
            return other.provenance == FactProvenance.EXPLICIT

        return False


def unknown_field() -> KnowledgeField:
    """Return a default UNKNOWN KnowledgeField."""
    return KnowledgeField(value=None, provenance=FactProvenance.UNKNOWN, confidence=FactConfidence.LOW)


def explicit_field(value: Any, confidence: FactConfidence = FactConfidence.HIGH, source_turn: int = 0) -> KnowledgeField:
    """Return an EXPLICIT KnowledgeField with the given value."""
    return KnowledgeField(value=value, provenance=FactProvenance.EXPLICIT, confidence=confidence, source_turn=source_turn)


def inferred_field(value: Any, confidence: FactConfidence = FactConfidence.MEDIUM, source_turn: int = 0, notes: str | None = None) -> KnowledgeField:
    """Return an INFERRED KnowledgeField with the given value."""
    return KnowledgeField(value=value, provenance=FactProvenance.INFERRED, confidence=confidence, source_turn=source_turn, notes=notes)


# ---------------------------------------------------------------------------
# Milestone 4.0.4: Factual State Grounding & Provenance Validation
# ---------------------------------------------------------------------------

import logging as _logging
import re as _re

_logger = _logging.getLogger(__name__)


def _is_negative_candidate_value(val: Any) -> bool:
    """Return True if value represents absence, zero, or False."""
    if val is None:
        return False
    if val is False:
        return True
    s = str(val).strip().lower()
    return s in ("false", "0", "0.0", "none", "no", "zero", "null")


def _extract_founder_text(
    previous_state: dict[str, Any] | None,
    founder_turn: str | dict[str, Any] | list[Any] | None,
) -> str:
    """Extract raw founder-authored text and context from turn input or state history."""
    parts: list[str] = []
    if isinstance(founder_turn, str):
        parts.append(founder_turn)
    elif isinstance(founder_turn, dict):
        parts.append(str(founder_turn.get("content") or ""))
    elif isinstance(founder_turn, list):
        for m in founder_turn:
            if isinstance(m, dict):
                parts.append(str(m.get("content") or ""))
            else:
                parts.append(str(m))

    if previous_state and isinstance(previous_state, dict):
        msgs = previous_state.get("founder_messages") or []
        for m in msgs:
            if isinstance(m, dict) and m.get("role") == "founder":
                parts.append(str(m.get("content") or ""))
        if previous_state.get("raw_context"):
            parts.append(str(previous_state["raw_context"]))
        if previous_state.get("pending_question"):
            parts.append(str(previous_state["pending_question"]))

    return "\n".join(parts)


def validate_state_grounding(
    previous_state: dict[str, Any],
    candidate_updates: dict[str, Any],
    founder_turn: str | dict[str, Any] | list[Any] | None = None,
) -> dict[str, Any]:
    """
    Deterministic validation of candidate factual updates against founder source grounding (Milestone 4.0.4).

    Rules:
    1. ABSENCE OF INFORMATION != NEGATIVE FACT:
       Mentioning coding/technical capability NEVER sets design, marketing, or sales to False.
       Missing capabilities remain UNKNOWN.
    2. NEGATIVE FACT REQUIREMENT:
       Negative claims (no design, no marketing, no sales, no budget, 0 users, 0 revenue, not launched, no prototype)
       strictly require explicit founder statements (FOUNDER_EXPLICIT).
       AI_INFERRED is forbidden from creating negative factual claims.
    3. INFERENCE RESTRICTION:
       AI_INFERRED is only allowed for valid derived fields:
       - total_weekly_capacity_hours (team size × hours)
       - development_stage (derived from prototype/mvp/launch facts)
       Other fields with provenance INFERRED require explicit mention in founder_turn or are dropped.
    4. MULTI-INTENT MESSAGE ISOLATION:
       Factual updates in multi-intent messages must be grounded.
       Unrelated speculative fields are dropped.
    """
    if not candidate_updates or not isinstance(candidate_updates, dict):
        return {}

    validated: dict[str, Any] = dict(candidate_updates)
    founder_text = _extract_founder_text(previous_state, founder_turn)
    text_lower = founder_text.lower()

    # Define capability negative patterns that justify explicit absence
    capability_neg_patterns = {
        "has_design_capability": [
            r"\b(no|not|neither|none|lack|without|can't|cannot|don't know|dont know|never)\b.{0,35}\b(design|ui|ux|designer|figma)\b",
            r"\b(design|ui|ux|designer|figma)\b.{0,35}\b(no|not|none|lack|zero|can't|cannot|don't know|dont know)\b",
        ],
        "design_capability": [
            r"\b(no|not|neither|none|lack|without|can't|cannot|don't know|dont know|never)\b.{0,35}\b(design|ui|ux|designer|figma)\b",
            r"\b(design|ui|ux|designer|figma)\b.{0,35}\b(no|not|none|lack|zero|can't|cannot|don't know|dont know)\b",
        ],
        "has_marketing_capability": [
            r"\b(no|not|neither|none|lack|without|can't|cannot|don't know|dont know|never)\b.{0,35}\b(marketing|sales|gtm|growth|marketer|selling)\b",
            r"\b(marketing|sales|gtm|growth|marketer|selling)\b.{0,35}\b(no|not|none|lack|zero|can't|cannot|don't know|dont know)\b",
        ],
        "marketing_capability": [
            r"\b(no|not|neither|none|lack|without|can't|cannot|don't know|dont know|never)\b.{0,35}\b(marketing|sales|gtm|growth|marketer|selling)\b",
            r"\b(marketing|sales|gtm|growth|marketer|selling)\b.{0,35}\b(no|not|none|lack|zero|can't|cannot|don't know|dont know)\b",
        ],
        "has_sales_capability": [
            r"\b(no|not|neither|none|lack|without|can't|cannot|don't know|dont know|never)\b.{0,35}\b(sales|selling|salesperson)\b",
            r"\b(sales|selling|salesperson)\b.{0,35}\b(no|not|none|lack|zero|can't|cannot|don't know|dont know)\b",
        ],
        "sales_capability": [
            r"\b(no|not|neither|none|lack|without|can't|cannot|don't know|dont know|never)\b.{0,35}\b(sales|selling|salesperson)\b",
            r"\b(sales|selling|salesperson)\b.{0,35}\b(no|not|none|lack|zero|can't|cannot|don't know|dont know)\b",
        ],
        "has_technical_capability": [
            r"\b(no|not|neither|none|lack|without|can't|cannot|don't know|dont know|never|non[- ]technical)\b.{0,35}\b(code|coding|technical|developer|engineer|software)\b",
            r"\b(code|coding|technical|developer|engineer|software)\b.{0,35}\b(no|not|none|lack|zero|can't|cannot|don't know|dont know)\b",
        ],
        "technical_capability": [
            r"\b(no|not|neither|none|lack|without|can't|cannot|don't know|dont know|never|non[- ]technical)\b.{0,35}\b(code|coding|technical|developer|engineer|software)\b",
            r"\b(code|coding|technical|developer|engineer|software)\b.{0,35}\b(no|not|none|lack|zero|can't|cannot|don't know|dont know)\b",
        ],
    }

    # Keyword presence requirements for any claim about that field
    field_mention_keywords = {
        "has_design_capability": ["design", "ui", "ux", "designer", "figma", "wireframe"],
        "design_capability": ["design", "ui", "ux", "designer", "figma", "wireframe"],
        "has_marketing_capability": ["marketing", "sales", "gtm", "growth", "marketer", "selling"],
        "marketing_capability": ["marketing", "sales", "gtm", "growth", "marketer", "selling"],
        "has_sales_capability": ["sales", "selling", "salesperson"],
        "sales_capability": ["sales", "selling", "salesperson"],
        "available_budget_amount": ["budget", "money", "fund", "funds", "cash", "dollar", "usd", "azn", "eur", "capital"],
        "active_users": ["active user", "active users", "daily active", "monthly active", "users"],
        "paying_customers": ["paying", "paying customer", "paying customers", "paid user", "paid client"],
        "revenue": ["revenue", "mrr", "arr", "income", "sales", "money made"],
        "launched": ["launch", "launched", "live", "production"],
        "mvp_exists": ["mvp", "product"],
        "prototype_exists": ["prototype", "mockup", "wireframe", "clickable"],
        "customer_access": ["access", "talked to", "spoke to", "reached out", "network", "contact", "community", "audience"],
    }

    for key, candidate in list(validated.items()):
        if candidate is None or key.startswith("_"):
            continue

        # Lists like existing_assets, team_members, major_milestones are preserved as-is
        if isinstance(candidate, list) or key in ("existing_assets", "team_members", "major_milestones"):
            continue

        val = candidate.get("value") if isinstance(candidate, dict) else getattr(candidate, "value", None)
        prov = str(candidate.get("provenance") if isinstance(candidate, dict) else getattr(candidate, "provenance", "UNKNOWN")).upper()

        if val is None or prov == "UNKNOWN":
            validated[key] = None
            continue

        is_neg = _is_negative_candidate_value(val)

        # 1. Capability checks
        if key in capability_neg_patterns:
            if is_neg:
                # Negative capability claim strictly requires explicit founder mention of absence
                patterns = capability_neg_patterns[key]
                has_explicit_negation = any(_re.search(pat, text_lower) for pat in patterns)
                if not has_explicit_negation or prov == "INFERRED":
                    _logger.info(
                        "validate_state_grounding | dropping ungrounded negative capability '%s' (val=%s, prov=%s)",
                        key, val, prov,
                    )
                    validated[key] = None
                    continue
            else:
                # Positive capability claim requires mention of relevant capability keywords
                keywords = field_mention_keywords.get(key, [])
                if keywords and not any(kw in text_lower for kw in keywords):
                    _logger.info(
                        "validate_state_grounding | dropping unmentioned positive capability '%s' (val=%s)",
                        key, val,
                    )
                    validated[key] = None
                    continue

        # 2. Negative progress / metric checks (0 users, 0 revenue, not launched, etc.)
        if key in ("active_users", "paying_customers", "revenue", "launched", "available_budget_amount"):
            if is_neg:
                if prov == "INFERRED":
                    _logger.info(
                        "validate_state_grounding | dropping speculative inferred negative metric '%s' (val=%s)",
                        key, val,
                    )
                    validated[key] = None
                    continue
                keywords = field_mention_keywords.get(key, [])
                if keywords and not any(kw in text_lower for kw in keywords):
                    _logger.info(
                        "validate_state_grounding | dropping ungrounded negative metric '%s' (not mentioned in founder text)",
                        key,
                    )
                    validated[key] = None
                    continue

        # 3. Speculative INFERRED fields across unrelated areas
        allowed_inferred_fields = {"total_weekly_capacity_hours", "development_stage"}
        if prov == "INFERRED" and key not in allowed_inferred_fields:
            keywords = field_mention_keywords.get(key, [])
            if keywords and not any(kw in text_lower for kw in keywords):
                _logger.info(
                    "validate_state_grounding | dropping speculative inferred field '%s' (no keywords in founder text)",
                    key,
                )
                validated[key] = None
    return validated


