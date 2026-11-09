"""
Experiment revision architecture V2 (Milestone 6.2.6, behind STARTARY_EXPERIMENT_REVISION_V2).

Pure, deterministic rules for:
  * choosing TARGETED_EDIT vs FULL_REGENERATION after a REVISE,
  * safely remediating consent-class contract violations without an LLM call,
  * applying a targeted patch and measuring how much it changed,
  * choosing a FOCUSED vs FULL re-critique,
  * the hard bounds of the revision state machine.

All contract judgements reuse startary.domain.experiment_contract (single source of truth).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Optional

from startary.domain.experiment_contract import (
    MAX_EXPERIMENT_REVISIONS,
    ContractCode,
    CriticDecision,
    ExperimentContractResult,
    _CONSENT,
    _GUARDIAN,
    _MINOR_GROUP,
    _PERSONAL_DATA,
    _RECORDING,
    _experiment_text,
    classify_issue_codes,
)


class RevisionStrategy(StrEnum):
    TARGETED_EDIT = "TARGETED_EDIT"
    FULL_REGENERATION = "FULL_REGENERATION"
    DETERMINISTIC_REMEDIATION = "DETERMINISTIC_REMEDIATION"


class DraftOrigin(StrEnum):
    INITIAL = "INITIAL"
    TARGETED_EDIT = "TARGETED_EDIT"
    FULL_REGENERATION = "FULL_REGENERATION"


class CriticMode(StrEnum):
    FULL = "FULL"
    FOCUSED = "FOCUSED"
    DETERMINISTIC = "DETERMINISTIC"


# ---------------------------------------------------------------------------
# Bounds (documented in docs/M6_2_6_EFFICIENT_EXPERIMENT_REVISION.md)
# ---------------------------------------------------------------------------
MAX_SEMANTIC_REVISIONS = MAX_EXPERIMENT_REVISIONS   # 1 revision (targeted edit OR full regeneration) per cycle
MAX_REMEDIATIONS_PER_DRAFT = 1                       # deterministic, free; does not consume the semantic allowance
MAX_LLM_ATTEMPTS_PER_CALL = 2                        # transient / invalid-output retries per LLM call (+1 token expansion)
MAX_FAILED_DIMENSIONS_FOR_TARGETED_EDIT = 3          # more than half of the six failing => redesign


# ---------------------------------------------------------------------------
# Strategy selection
# ---------------------------------------------------------------------------

# Deterministic violations a local edit can fix vs. ones that require a different design.
TARGETED_FIXABLE_VIOLATIONS = frozenset({
    ContractCode.COST_EXCEEDS_BUDGET.value,
    ContractCode.REQUIRED_FIELD_EMPTY.value,
    ContractCode.INVALID_SAMPLE_SIZE.value,
})
REDESIGN_VIOLATIONS = frozenset({
    ContractCode.PROTOTYPE_NOT_AVAILABLE.value,      # method depends on an artifact that does not exist
    ContractCode.BUILD_SCOPE_UNDEFINED.value,
    ContractCode.ASSUMPTION_MISMATCH.value,          # stale draft for another assumption
    ContractCode.CONSENT_STEP_MISSING.value,         # only reaches here when remediation was unsafe (e.g. covert recording)
    ContractCode.GUARDIAN_CONSENT_MISSING.value,
})


def select_revision_strategy(
    decision_source: str,
    failed_dimensions: list[str],
    violation_codes: list[str],
    critic_revision_scope: Optional[str],
) -> str:
    """Strategy chosen by the rules in select_revision_strategy_with_reason (first match wins)."""
    return select_revision_strategy_with_reason(decision_source, failed_dimensions, violation_codes, critic_revision_scope)[0]


def select_revision_strategy_with_reason(
    decision_source: str,
    failed_dimensions: list[str],
    violation_codes: list[str],
    critic_revision_scope: Optional[str],
) -> tuple[str, str]:
    """
    Explicit rules (first match wins):
      Deterministic REVISE:
        1. any violation in REDESIGN_VIOLATIONS                    -> FULL_REGENERATION
        2. all violations in TARGETED_FIXABLE_VIOLATIONS             -> TARGETED_EDIT
        3. otherwise                                                 -> FULL_REGENERATION
      LLM REVISE:
        4. assumption_alignment FAIL (tests the wrong thing)        -> FULL_REGENERATION
        5. critic revision_scope is not exactly TARGETED            -> FULL_REGENERATION
           (REDESIGN = new method/evidence source needed; missing = conservative)
        6. more than 3 of the six dimensions FAIL                    -> FULL_REGENERATION
        7. otherwise                                                 -> TARGETED_EDIT
    """
    full, targeted = RevisionStrategy.FULL_REGENERATION.value, RevisionStrategy.TARGETED_EDIT.value
    if decision_source == "DETERMINISTIC":
        codes = set(violation_codes or [])
        if codes & REDESIGN_VIOLATIONS:
            return full, "REDESIGN_VIOLATION"
        if codes and codes <= TARGETED_FIXABLE_VIOLATIONS:
            return targeted, "LOCALLY_FIXABLE_VIOLATIONS"
        return full, "OTHER_DETERMINISTIC_VIOLATION"
    failed = set(failed_dimensions or [])
    if "assumption_alignment" in failed:
        return full, "ASSUMPTION_ALIGNMENT_FAILED"
    if str(critic_revision_scope or "").strip().upper() != "TARGETED":
        return full, "CRITIC_SCOPE_NOT_TARGETED"
    if len(failed) > MAX_FAILED_DIMENSIONS_FOR_TARGETED_EDIT:
        return full, "TOO_MANY_FAILED_DIMENSIONS"
    return targeted, "CRITIC_SCOPE_TARGETED"


# ---------------------------------------------------------------------------
# Deterministic consent remediation
# ---------------------------------------------------------------------------

REMEDIABLE_VIOLATIONS = frozenset({ContractCode.CONSENT_STEP_MISSING.value, ContractCode.GUARDIAN_CONSENT_MISSING.value})

REMEDIATION_ACTIONS: dict[str, str] = {
    ContractCode.CONSENT_STEP_MISSING.value: (
        "Before any recording or data collection, explain the purpose to everyone involved and obtain their informed "
        "permission; record nothing without it. Keep recordings private to the founding team and delete them once the "
        "analysis is complete."
    ),
    ContractCode.GUARDIAN_CONSENT_MISSING.value: (
        "Before recording, interviewing, or collecting information about any participant under 18, obtain written "
        "parent/guardian authorization and the club's permission; involve no minor without both."
    ),
}

# A permission step cannot make these methods acceptable: they would have to change, not be annotated.
_UNREMEDIABLE_METHOD = re.compile(
    r"\b(?:secret(?:ly)?|covert(?:ly)?|hidden\s+cameras?|undercover|scrap(?:e|es|ed|ing))\b"
    r"|\bwithout\s+(?:their\s+|the\s+|anyone'?s\s+)?(?:knowledge|knowing|telling|consent|permission|noticing)\b",
    re.IGNORECASE,
)
_SENSITIVE_ACTION = re.compile(
    _RECORDING.pattern + "|" + _PERSONAL_DATA.pattern + "|" + _MINOR_GROUP.pattern
    + r"|\b(?:interview|survey|observe|test)\w*\b",
    re.IGNORECASE,
)


@dataclass
class RemediationResult:
    draft: dict[str, Any]
    applied_codes: list[str] = field(default_factory=list)
    skipped_reason: Optional[str] = None

    @property
    def applied(self) -> bool:
        return bool(self.applied_codes)


def remediate_consent_violations(exp: dict[str, Any], contract: ExperimentContractResult) -> RemediationResult:
    """
    Insert a code-authored authorization step BEFORE the first sensitive founder action, for
    CONSENT_STEP_MISSING / GUARDIAN_CONSENT_MISSING only. Idempotent (an identical step is never
    added twice), preserves existing actions and their order, never changes the method, and refuses
    when other violations exist or the method itself is unremediable (covert recording, scraping).
    Inserting the step does NOT grant permission: callers must keep authorization PENDING.
    """
    codes = contract.violation_codes
    if not codes:
        return RemediationResult(exp, skipped_reason="NO_VIOLATIONS")
    if not set(codes) <= REMEDIABLE_VIOLATIONS:
        return RemediationResult(exp, skipped_reason="NON_REMEDIABLE_VIOLATION_PRESENT")
    if _UNREMEDIABLE_METHOD.search(_experiment_text(exp)):
        return RemediationResult(exp, skipped_reason="UNREMEDIABLE_METHOD")

    actions = [str(a) for a in (exp.get("founder_actions") or [])]
    to_insert = [REMEDIATION_ACTIONS[c] for c in (ContractCode.CONSENT_STEP_MISSING.value, ContractCode.GUARDIAN_CONSENT_MISSING.value)
                 if c in codes and REMEDIATION_ACTIONS[c] not in actions]
    if not to_insert:
        return RemediationResult(exp, skipped_reason="ALREADY_REMEDIATED")
    position = next((i for i, a in enumerate(actions) if _SENSITIVE_ACTION.search(a)), 0)
    draft = dict(exp)
    draft["founder_actions"] = actions[:position] + to_insert + actions[position:]
    applied = [c for c in codes if REMEDIATION_ACTIONS[c] in to_insert]
    draft["remediation_codes"] = sorted(set((exp.get("remediation_codes") or [])) | set(applied))
    return RemediationResult(draft, applied_codes=applied)


def consent_language_present(exp: dict[str, Any]) -> bool:
    text = _experiment_text(exp)
    return bool(_CONSENT.search(text) or _GUARDIAN.search(text))


# ---------------------------------------------------------------------------
# Targeted edits: patch application, diff, substantiality
# ---------------------------------------------------------------------------

EDITABLE_FIELDS = (
    "experiment_type", "objective", "method", "target_participants_or_scope", "target_sample_size",
    "founder_actions", "evidence_to_collect", "support_signal", "contradiction_signal", "inconclusive_signal",
    "valid_inferences", "invalid_inferences", "estimated_time", "estimated_cost",
    "required_capabilities", "required_access", "parameters",
)
# Fields whose change means a different methodology, population, or evidence source.
SUBSTANTIAL_FIELDS = ("method", "target_participants_or_scope", "evidence_to_collect")
SUBSTANTIAL_SIMILARITY_THRESHOLD = 0.5
_WORD = re.compile(r"[a-z0-9]+")


def apply_revision_patch(original: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """Overwrite only editable fields the patch provides (non-null). Code-owned keys are preserved."""
    revised = dict(original)
    for name in EDITABLE_FIELDS:
        value = patch.get(name)
        if value is not None:
            revised[name] = value
    return revised


def _norm(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [_norm(v) for v in value]
    if isinstance(value, dict):
        return {k: _norm(v) for k, v in sorted(value.items())}
    if isinstance(value, str):
        return " ".join(value.split())
    return getattr(value, "value", value)


def diff_experiment(original: dict[str, Any], revised: dict[str, Any]) -> list[str]:
    """Names of editable fields whose content actually changed (whitespace-insensitive)."""
    return [f for f in EDITABLE_FIELDS if _norm(original.get(f)) != _norm(revised.get(f))]


def _similarity(a: Any, b: Any) -> float:
    wa, wb = set(_WORD.findall(str(a or "").lower())), set(_WORD.findall(str(b or "").lower()))
    if not wa and not wb:
        return 1.0
    return len(wa & wb) / max(1, len(wa | wb))


def is_substantial_revision(original: dict[str, Any], revised: dict[str, Any], changed: list[str]) -> bool:
    """
    A targeted edit is substantial (=> FULL re-critique) when it changes the experiment type, or
    rewrites the method, participant population, or evidence source beyond a light edit
    (word-set similarity below 0.5).
    """
    if "experiment_type" in changed:
        return True
    return any(
        f in changed and _similarity(original.get(f), revised.get(f)) < SUBSTANTIAL_SIMILARITY_THRESHOLD
        for f in SUBSTANTIAL_FIELDS
    )


def change_summary(original: dict[str, Any], revised: dict[str, Any], changed: list[str]) -> list[dict[str, Any]]:
    """Structured before/after comparison of changed fields (for the focused critic's prompt only)."""
    return [{"field": f, "before": original.get(f), "after": revised.get(f)} for f in changed]


def generation_record(revision: dict[str, Any] | None, is_revision: bool) -> dict[str, Any]:
    """Revision bookkeeping after generate_experiment: a fresh cycle, or a full regeneration of a rejected draft."""
    if not is_revision:
        return {"draft_origin": DraftOrigin.INITIAL.value}
    return {**(revision or {}), "draft_origin": DraftOrigin.FULL_REGENERATION.value,
            "changed_fields": None, "substantial_change": None, "next_strategy": None}


def rejection_record(
    revision: dict[str, Any] | None,
    rejected_draft: dict[str, Any],
    decision: CriticDecision,
    issues: list[Any],
    instructions: str,
    critic_mode: str,
    revision_scope: Optional[str],
    next_strategy: Optional[str],
    revision_round: int,
) -> dict[str, Any]:
    """Preserve the rejected draft and its critique until the next revision completes."""
    return {
        **(revision or {}),
        "rejected_draft": dict(rejected_draft),
        "rejected_critique": {
            "verdict": decision.verdict,
            "dimensions": dict(decision.dimensions),
            "failed_dimensions": list(decision.failed_dimensions),
            "issues": [str(i) for i in issues or []],
            "revision_instructions": instructions,
            "critic_mode": critic_mode,
            "revision_scope": revision_scope,
        },
        "next_strategy": next_strategy,
        "rejected_round": revision_round,
    }


def critic_v2_metadata(
    revision: dict[str, Any] | None,
    draft: dict[str, Any],
    decision: CriticDecision,
    critic_mode: str,
    next_strategy: Optional[str],
    raw: dict[str, Any] | None = None,
    reason_codes: Optional[list[str]] = None,
) -> dict[str, Any]:
    """V2 additions to the critic decision record: codes, enums, counts and field NAMES only."""
    rev = revision or {}
    raw = raw or {}
    origin = rev.get("draft_origin") or DraftOrigin.INITIAL.value
    prior_failed = list((rev.get("rejected_critique") or {}).get("failed_dimensions") or []) if origin != DraftOrigin.INITIAL.value else []
    new_blocking = list(raw.get("new_blocking_issues") or [])
    targeted = origin == DraftOrigin.TARGETED_EDIT.value
    return {
        "revision_mode": "V2",
        "critic_mode": critic_mode,
        "draft_origin": origin,
        "revision_strategy": next_strategy,
        "original_failed_dimensions": prior_failed,
        "remaining_failed_dimensions": [d for d in prior_failed if d in decision.failed_dimensions],
        "changed_fields": list(rev.get("changed_fields") or []) if targeted else [],
        "substantial_change": bool(rev.get("substantial_change")) if targeted else None,
        "remediation_applied": bool(draft.get("remediation_codes")),
        "remediation_codes": list(draft.get("remediation_codes") or []),
        "required_authorizations": list(draft.get("required_authorizations") or []),
        "new_blocking_issue_count": len(new_blocking),
        "new_blocking_issue_codes": classify_issue_codes(new_blocking),
        "original_issues_resolved": raw.get("original_issues_resolved") if critic_mode == CriticMode.FOCUSED.value else None,
        # M6.2.6.2: the critic's own scope judgement (an enum), so a TARGETED scope overridden by the
        # deterministic rules (e.g. assumption_alignment FAIL) is visible in traces.
        "critic_revision_scope": strict_scope(raw.get("revision_scope")),
        "revision_reason_codes": list(reason_codes or []),
    }


def strict_scope(value: Any) -> Optional[str]:
    token = str(getattr(value, "value", value) or "").strip().upper()
    return token if token in ("NONE", "TARGETED", "REDESIGN") else None


def apply_focused_acceptance_rule(decision: CriticDecision, raw: dict[str, Any]) -> CriticDecision:
    """
    Focused re-critique is fail-closed like the full critic (exact ACCEPT + all six PASS) and
    additionally requires every original blocking issue resolved and no new blocking issue.
    """
    if decision.verdict != "ACCEPT":
        return decision
    if raw.get("original_issues_resolved") is True and not (raw.get("new_blocking_issues") or []):
        return decision
    return CriticDecision("REVISE", decision.model_verdict, decision.dimensions, decision.failed_dimensions, True, decision.decision_source)


def select_critic_mode(revision: dict[str, Any] | None) -> str:
    """FOCUSED only for a non-substantial targeted edit with a recorded prior critique; FULL otherwise."""
    rev = revision or {}
    if (
        rev.get("draft_origin") == DraftOrigin.TARGETED_EDIT.value
        and not rev.get("substantial_change")
        and rev.get("rejected_draft")
        and rev.get("rejected_critique")
    ):
        return CriticMode.FOCUSED.value
    return CriticMode.FULL.value
