"""
Bound Answer Fast-Path Parser (Milestone 4.1.3 Phase 2).

Deterministically extracts bounded execution context answers (TEAM, CAPACITY,
RESOURCE/BUDGET, PRODUCT_STAGE, VALIDATION_USERS_ACCESS) without requiring full
LLM startup state extraction or rediagnosis.
"""

from __future__ import annotations

import re
from typing import Any, Optional


def _normalize_target(target: str | None) -> str:
    if not target:
        return ""
    t = target.upper().strip()
    if "CAPACITY" in t or "TIME" in t or "HOURS" in t:
        return "CAPACITY"
    if "TEAM" in t or "FOUNDER" in t:
        return "TEAM"
    if "RESOURCE" in t or "BUDGET" in t or "FUND" in t:
        return "RESOURCE"
    if "STAGE" in t or "PROTOTYPE" in t or "MVP" in t or "PRODUCT" in t:
        return "PRODUCT_STAGE"
    if "ACCESS" in t or "USER" in t or "CUSTOMER_ACCESS" in t:
        return "VALIDATION_USERS_ACCESS"
    return t


def _is_multi_topic_or_strategic(text: str) -> bool:
    """Check if the founder message introduces multi-topic or strategic claims."""
    t_lower = text.lower()
    strategic_keywords = [
        "paying customer", "revenue", "churn", "pricing", "problem is",
        "competitor", "alternative", "interviewed", "feedback was",
        "freshmen", "students", "cohort", "tested", "experiment",
    ]
    # If the message is long and contains strategic keywords, fall back
    if len(text.split()) > 25 and any(k in t_lower for k in strategic_keywords):
        return True
    return False


def parse_capacity_answer(text: str) -> tuple[bool, dict[str, Any], str]:
    """Parse weekly capacity/hours from text."""
    t_lower = text.lower().strip()
    # Check for corrections like 'actually not 10, more like 20 hours'
    # Use the LAST occurrence of hours
    corr_match = re.findall(r"(?:not\s+\d+[\w\s,]*?)?(?:more\s+like|actually|now|rather)\s+(\d+)\s*(?:hours?|hrs?|h\b)", t_lower)
    if corr_match:
        val = int(corr_match[-1])
        return True, {
            "total_weekly_capacity_hours": {"value": str(val), "provenance": "EXPLICIT", "confidence": "HIGH"},
            "weekly_hours_per_member": {"value": str(val), "provenance": "EXPLICIT", "confidence": "HIGH"},
        }, "HIGH"

    # Range like '10-15 hours' or '10 to 15 hours' -> take upper or midpoint
    range_match = re.findall(r"(\d+)\s*(?:-|to)\s*(\d+)\s*(?:hours?|hrs?|h\b)", t_lower)
    if range_match:
        val = int(range_match[-1][1])  # conservative upper
        return True, {
            "total_weekly_capacity_hours": {"value": str(val), "provenance": "EXPLICIT", "confidence": "HIGH"},
            "weekly_hours_per_member": {"value": str(val), "provenance": "EXPLICIT", "confidence": "HIGH"},
        }, "HIGH"

    # 'more than 10 hours' / 'at least 20 hours' / 'about 15 hours' / '10 hours weekly'
    single_match = re.findall(r"(?:more\s+than|at\s+least|about|roughly|around|approx(?:\.|\s+)?)?\s*(\d+)\s*(?:hours?|hrs?|h\b)", t_lower)
    if single_match:
        val = int(single_match[-1])
        return True, {
            "total_weekly_capacity_hours": {"value": str(val), "provenance": "EXPLICIT", "confidence": "HIGH"},
            "weekly_hours_per_member": {"value": str(val), "provenance": "EXPLICIT", "confidence": "HIGH"},
        }, "HIGH"

    if any(k in t_lower for k in ["full time", "full-time", "fulltime"]):
        return True, {
            "total_weekly_capacity_hours": {"value": "40", "provenance": "EXPLICIT", "confidence": "HIGH"},
            "weekly_hours_per_member": {"value": "40", "provenance": "EXPLICIT", "confidence": "HIGH"},
        }, "HIGH"

    if any(k in t_lower for k in ["part time", "part-time", "parttime"]):
        return True, {
            "total_weekly_capacity_hours": {"value": "20", "provenance": "EXPLICIT", "confidence": "HIGH"},
            "weekly_hours_per_member": {"value": "20", "provenance": "EXPLICIT", "confidence": "HIGH"},
        }, "HIGH"

    # Digits only if short: e.g. "10", "20"
    if t_lower.isdigit() and len(t_lower) <= 3:
        val = int(t_lower)
        return True, {
            "total_weekly_capacity_hours": {"value": str(val), "provenance": "EXPLICIT", "confidence": "HIGH"},
            "weekly_hours_per_member": {"value": str(val), "provenance": "EXPLICIT", "confidence": "HIGH"},
        }, "HIGH"

    return False, {}, "LOW"


def parse_team_answer(text: str) -> tuple[bool, dict[str, Any], str]:
    """Parse team size / solo status from text."""
    t_lower = text.lower().strip()

    # Solo indicators
    solo_keywords = ["alone", "solo", "just me", "only me", "single founder", "myself", "nobody else", "one person"]
    if any(re.search(rf"\b{re.escape(k)}\b", t_lower) for k in solo_keywords):
        return True, {
            "founder_count": {"value": "1", "provenance": "EXPLICIT", "confidence": "HIGH"},
            "active_team_size": {"value": "1", "provenance": "EXPLICIT", "confidence": "HIGH"},
        }, "HIGH"

    # Correction like 'actually not alone, we are 2'
    corr = re.findall(r"(?:not\s+(?:alone|solo|1)[\w\s,]*?)?(?:more\s+like|actually|now|rather|we\s+are)\s+(\d+)", t_lower)
    if corr:
        val = int(corr[-1])
        return True, {
            "founder_count": {"value": str(val), "provenance": "EXPLICIT", "confidence": "HIGH"},
            "active_team_size": {"value": str(val), "provenance": "EXPLICIT", "confidence": "HIGH"},
        }, "HIGH"

    # '2 cofounders', 'we are 2', '2 of us', 'team of 3'
    count_match = re.findall(r"\b(\d+)\s*(?:co-?founders?|founders?|people|members?|of\s+us)\b", t_lower)
    if count_match:
        val = int(count_match[-1])
        return True, {
            "founder_count": {"value": str(val), "provenance": "EXPLICIT", "confidence": "HIGH"},
            "active_team_size": {"value": str(val), "provenance": "EXPLICIT", "confidence": "HIGH"},
        }, "HIGH"

    team_of = re.findall(r"\bteam\s+of\s+(\d+)\b", t_lower)
    if team_of:
        val = int(team_of[-1])
        return True, {
            "founder_count": {"value": str(val), "provenance": "EXPLICIT", "confidence": "HIGH"},
            "active_team_size": {"value": str(val), "provenance": "EXPLICIT", "confidence": "HIGH"},
        }, "HIGH"

    # 'we are two' / 'two cofounders'
    word_map = {"two": 2, "three": 3, "four": 4, "five": 5}
    for w, n in word_map.items():
        if re.search(rf"\b{w}\s+(?:co-?founders?|founders?|of\s+us|people)\b", t_lower) or re.search(rf"\bwe\s+are\s+{w}\b", t_lower):
            return True, {
                "founder_count": {"value": str(n), "provenance": "EXPLICIT", "confidence": "HIGH"},
                "active_team_size": {"value": str(n), "provenance": "EXPLICIT", "confidence": "HIGH"},
            }, "HIGH"

    if any(k in t_lower for k in ["my cofounder handles", "have a cofounder", "have a co-founder"]):
        return True, {
            "founder_count": {"value": "2", "provenance": "EXPLICIT", "confidence": "HIGH"},
            "active_team_size": {"value": "2", "provenance": "EXPLICIT", "confidence": "HIGH"},
        }, "HIGH"

    return False, {}, "LOW"


def parse_resource_answer(text: str) -> tuple[bool, dict[str, Any], str]:
    """Parse budget/resource answer from text."""
    t_lower = text.lower().strip()

    # Zero / no budget
    if any(k in t_lower for k in ["no budget", "zero budget", "$0", "0$", "0 azn", "0 eur", "bootstrapped with no funds", "no money"]):
        return True, {
            "available_budget_amount": {"value": "0", "provenance": "EXPLICIT", "confidence": "HIGH"},
            "funding_status": {"value": "BOOTSTRAPPED", "provenance": "INFERRED", "confidence": "HIGH"},
        }, "HIGH"

    # Amount like '$500', '500 USD', '5000 azn', '10k'
    money_match = re.findall(r"(?:\$|usd\s*|azn\s*|eur\s*|€\s*)?(\d+(?:,\d+)?(?:\.\d+)?)\s*(?:k\b|\$|usd|azn|eur|€|dollars?|bucks?)?", t_lower)
    clean_k = re.search(r"\b(\d+)\s*k\b", t_lower)
    if clean_k:
        val = int(clean_k.group(1)) * 1000
        return True, {
            "available_budget_amount": {"value": str(val), "provenance": "EXPLICIT", "confidence": "HIGH"},
        }, "HIGH"

    amt_match = re.search(r"(?:\$|usd\s*|azn\s*|eur\s*|€\s*)(\d[\d,]*)", t_lower) or re.search(r"(\d[\d,]*)\s*(?:\$|usd|azn|eur|€|dollars?)", t_lower)
    if amt_match:
        val_str = amt_match.group(1).replace(",", "")
        return True, {
            "available_budget_amount": {"value": val_str, "provenance": "EXPLICIT", "confidence": "HIGH"},
        }, "HIGH"

    return False, {}, "LOW"


def parse_product_stage_answer(text: str) -> tuple[bool, dict[str, Any], str]:
    """Parse product development stage from text."""
    t_lower = text.lower().strip()

    if any(k in t_lower for k in ["no product", "just an idea", "raw idea", "idea stage", "haven't built", "havent built"]):
        return True, {
            "prototype_exists": {"value": "false", "provenance": "EXPLICIT", "confidence": "HIGH"},
            "mvp_exists": {"value": "false", "provenance": "EXPLICIT", "confidence": "HIGH"},
            "launched": {"value": "false", "provenance": "EXPLICIT", "confidence": "HIGH"},
        }, "HIGH"

    if any(k in t_lower for k in ["prototype", "mockup", "wireframe", "clickable prototype"]):
        return True, {
            "prototype_exists": {"value": "true", "provenance": "EXPLICIT", "confidence": "HIGH"},
            "mvp_exists": {"value": "false", "provenance": "INFERRED", "confidence": "NORMAL"},
            "launched": {"value": "false", "provenance": "INFERRED", "confidence": "NORMAL"},
        }, "HIGH"

    if any(k in t_lower for k in ["working mvp", "built an mvp", "mvp exists", "have an mvp"]):
        return True, {
            "prototype_exists": {"value": "true", "provenance": "INFERRED", "confidence": "HIGH"},
            "mvp_exists": {"value": "true", "provenance": "EXPLICIT", "confidence": "HIGH"},
        }, "HIGH"

    if any(k in t_lower for k in ["live", "launched", "in production", "available on"]):
        return True, {
            "prototype_exists": {"value": "true", "provenance": "INFERRED", "confidence": "HIGH"},
            "mvp_exists": {"value": "true", "provenance": "INFERRED", "confidence": "HIGH"},
            "launched": {"value": "true", "provenance": "EXPLICIT", "confidence": "HIGH"},
        }, "HIGH"

    return False, {}, "LOW"


def parse_bound_answer(
    information_target: str | None,
    founder_message: str,
    pending_question: str | None = None,
) -> tuple[bool, dict[str, Any], str]:
    """
    Main dispatch for bounded answers.

    Returns:
        (success: bool, delta_dict: dict, confidence: str)
    """
    if not founder_message or _is_multi_topic_or_strategic(founder_message):
        return False, {}, "LOW"

    target_norm = _normalize_target(information_target)

    # If target not known from active question, try inferring from question text
    if not target_norm and pending_question:
        target_norm = _normalize_target(pending_question)

    if target_norm == "CAPACITY":
        return parse_capacity_answer(founder_message)
    if target_norm == "TEAM":
        return parse_team_answer(founder_message)
    if target_norm == "RESOURCE":
        return parse_resource_answer(founder_message)
    if target_norm == "PRODUCT_STAGE":
        return parse_product_stage_answer(founder_message)

    # Try heuristic matching if target was generic
    cap_ok, cap_delta, cap_conf = parse_capacity_answer(founder_message)
    if cap_ok and ("hour" in founder_message.lower() or "hr" in founder_message.lower()):
        return True, cap_delta, cap_conf

    team_ok, team_delta, team_conf = parse_team_answer(founder_message)
    if team_ok and ("alone" in founder_message.lower() or "cofounder" in founder_message.lower() or "team" in founder_message.lower()):
        return True, team_delta, team_conf

    return False, {}, "LOW"

