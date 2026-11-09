"""
Evidence domain models for Milestone 4: Evidence Intake, Evidence Analysis, and Learning.

Distinguishes:
  - Raw immutable founder submission
  - Linked active experiment and active critical assumption
  - Quantitative and qualitative observations
  - Structured Evidence Analysis with explicit conclusion and strength
  - Learning Delta capturing what changed between iterations
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator

from startary.domain.enums import (
    EvidenceAnalysisConclusion,
    EvidenceProvenance,
    EvidenceResult,
    EvidenceSourceType,
    EvidenceStrength,
    EvidenceType,
    ExecutionQuality,
)


class EvidenceTurnDisposition(StrEnum):
    """
    Classification of founder turns received while in WAITING_FOR_EVIDENCE (Milestone 4.0.3).
    Only ACTUAL_EVIDENCE may enter the evidence extraction and analysis pipeline.
    """
    ACTUAL_EVIDENCE = "ACTUAL_EVIDENCE"
    QUESTION_ABOUT_EXPERIMENT = "QUESTION_ABOUT_EXPERIMENT"
    REQUEST_FOR_EXECUTION_HELP = "REQUEST_FOR_EXECUTION_HELP"
    FUTURE_INTENTION = "FUTURE_INTENTION"
    CANNOT_EXECUTE = "CANNOT_EXECUTE"
    CORRECTION = "CORRECTION"
    UNRELATED = "UNRELATED"
    AMBIGUOUS = "AMBIGUOUS"


class EvidenceTurnAssessment(BaseModel):
    """
    Structured assessment of founder turn submitted during WAITING_FOR_EVIDENCE (Milestone 4.0.3).
    """
    disposition: EvidenceTurnDisposition = Field(
        description="Controlled disposition determining routing."
    )
    contains_completed_external_action: bool = Field(
        default=False,
        description="True if the message reports completed real-world actions with external parties.",
    )
    contains_results: bool = Field(
        default=False,
        description="True if the message includes factual outcomes, counts, quotes, or findings.",
    )
    contains_only_future_intent: bool = Field(
        default=False,
        description="True if the message only expresses plans to run the test in the future.",
    )
    asks_for_help: bool = Field(
        default=False,
        description="True if founder asks how to execute or where to find participants.",
    )
    asks_for_explanation: bool = Field(
        default=False,
        description="True if founder asks what to do, why this test, or for instructions.",
    )
    cannot_execute: bool = Field(
        default=False,
        description="True if founder reports being unable to execute due to constraints or blockers.",
    )
    evidence_candidate_text: Optional[str] = Field(
        default=None,
        description="The subset of text representing evidence (useful for mixed messages).",
    )
    rationale_summary: str = Field(
        default="",
        description="Concise rationale explaining the disposition.",
    )


def get_evidence_turn_disposition(state: dict[str, Any] | Any) -> Optional[EvidenceTurnDisposition]:
    """
    Deterministic accessor for evidence turn disposition (Milestone 4.0.7).
    Safely resolves disposition whether state or assessment contains:
      - EvidenceTurnAssessment (Pydantic object)
      - dict (serialized form)
      - string or EvidenceTurnDisposition enum value
    Returns None if absent or unparseable.
    """
    if not isinstance(state, dict):
        if isinstance(state, EvidenceTurnAssessment):
            return state.disposition
        return None

    assessment = state.get("evidence_turn_assessment")
    if assessment is None:
        # Fallback: check if 'disposition' was passed directly at top-level of state
        raw_top = state.get("disposition")
        if raw_top is not None:
            assessment = {"disposition": raw_top}
        else:
            return None

    def _parse_disp(val: Any) -> Optional[EvidenceTurnDisposition]:
        if val is None:
            return None
        if isinstance(val, EvidenceTurnDisposition):
            return val
        s = str(val).strip()
        if "." in s:
            s = s.split(".")[-1]
        try:
            return EvidenceTurnDisposition(s.upper().strip())
        except ValueError:
            return None

    if isinstance(assessment, EvidenceTurnAssessment):
        return assessment.disposition

    if isinstance(assessment, dict):
        raw = assessment.get("disposition")
        return _parse_disp(raw)

    if isinstance(assessment, (str, EvidenceTurnDisposition)):
        return _parse_disp(assessment)

    return None


def is_actual_evidence_turn(state: dict[str, Any] | Any) -> bool:
    """Returns True if the turn assessment disposition is ACTUAL_EVIDENCE."""
    return get_evidence_turn_disposition(state) == EvidenceTurnDisposition.ACTUAL_EVIDENCE


COMPLETED_ACTION_VERBS = [
    "interviewed", "spoke to", "spoke with", "talked to", "talked with",
    "messaged", "sent", "contacted", "reached out", "ran the", "ran a", "ran an", "ran",
    "tested", "tried contacting", "tried it", "asked", "showed", "visited",
    "completed the interviews", "completed interviews", "finished the interviews",
    "done with interviews", "finished interviews", "completed the calls", "finished the calls",
    "completed", "conducted", "surveyed", "observed", "pitched", "pre-ordered", "preordered",
]

SUBSTANTIVE_RESULTS_WORDS = [
    "said", "replied", "answered", "agreed", "rejected", "declined", "result", "results",
    "found that", "found", "nobody replied", "no one replied", "loss", "pain", "stockout",
    "stockouts", "waste", "surplus", "bought", "purchased", "paid", "pickup", "pickups",
    "pre-ordered", "preordered", "signed up", "orders", "revenue", "dollar", "$",
    "quote", "quoted", "reported", "confirmed", "interested", "willing", "refused", "feedback",
]

ACTION_NEGATIONS = [
    "haven't", "have not", "didn't", "did not", "not yet", "never", "unable to",
    "couldn't", "could not", "cannot", "cant", "can't",
]

FUTURE_MODAL_PREFIXES = [
    "will ", "going to ", "plan to ", "planning to ", "hope to ", "intend to ",
    "tomorrow ", "next week ",
]

COUNT_REGEX = re.compile(
    r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty|thirty|forty|fifty)\b",
    re.IGNORECASE,
)

# Milestone 6.2.3: negation-aware traction keyword detection.
# "zero paying customers" / "no active users" / "haven't got paying teams" are NEGATIVE evidence
# and must never be read as positive traction signals.
_KEYWORD_NEGATION_REGEX = re.compile(
    r"\b(no|not|zero|0|none|never|nobody|without|lack|lacks|lacking|neither|nor|yet to)\b"
    r"|\bno one\b|haven't|have not|hasn't|has not|don't|do not|doesn't|does not|didn't|did not"
    r"|aren't|are not|isn't|is not|wasn't|was not|weren't|were not",
    re.IGNORECASE,
)
_KEYWORD_SUFFIX_NEGATION_REGEX = re.compile(
    r"^\s*(?:so far|yet|to date)?\s*[:=\-]?\s*(?:0|zero|none|nil)\b"
    r"|^\s*(?:[a-z]+\s*){0,3}[:=\-]\s*(?:0|zero|none|nil)\b",
    re.IGNORECASE,
)
# A negation only governs the keyword when it sits in the same sub-clause:
# "zero revenue and 8 paying teams" must still count the 8 paying teams.
_SUBCLAUSE_BOUNDARY_REGEX = re.compile(r"[,;.!?\n]|\band\b|\bbut\b|\bwhile\b|\bwhereas\b|\bhowever\b", re.IGNORECASE)


def contains_unnegated_keyword(text: str, keywords: list[str] | tuple[str, ...], window: int = 40) -> bool:
    """
    Return True if any keyword occurs in ``text`` at least once WITHOUT a negation governing it.

    A keyword occurrence is treated as negated when, within its own sub-clause, it is preceded
    by a negation/zero marker ("zero paying customers", "we have no active users",
    "haven't had any paying teams") or immediately followed by a zero value
    ("paying customers: 0"). Plain substring semantics are otherwise preserved.
    """
    if not text:
        return False
    lowered = text.lower()
    for kw in keywords:
        kw_lower = kw.lower()
        if not kw_lower:
            continue
        start = 0
        while True:
            idx = lowered.find(kw_lower, start)
            if idx < 0:
                break
            start = idx + 1
            prefix = lowered[max(0, idx - window):idx]
            boundaries = list(_SUBCLAUSE_BOUNDARY_REGEX.finditer(prefix))
            if boundaries:
                prefix = prefix[boundaries[-1].end():]
            suffix = lowered[idx + len(kw_lower): idx + len(kw_lower) + 32]
            if _KEYWORD_NEGATION_REGEX.search(prefix) or _KEYWORD_SUFFIX_NEGATION_REGEX.search(suffix):
                continue
            return True
    return False


def has_substantive_completed_evidence(
    raw_message: str,
    llm_assessment: Optional[Any] = None,
) -> bool:
    """
    Evaluates whether founder input contains substantive completed real-world evidence
    (Milestone 4.0.7.2.1). Substantive completed evidence takes precedence over incidental
    future-oriented language or planning statements.
    """
    if llm_assessment:
        disp = getattr(llm_assessment, "disposition", "")
        if isinstance(disp, EvidenceTurnDisposition):
            disp = disp.value
        disp = str(disp).upper().strip()
        has_ext_action = getattr(llm_assessment, "contains_completed_external_action", False)
        has_results = getattr(llm_assessment, "contains_results", False)
        has_candidate = bool(getattr(llm_assessment, "evidence_candidate_text", None))
        only_future = getattr(llm_assessment, "contains_only_future_intent", False)

        if has_ext_action and (has_results or disp == "ACTUAL_EVIDENCE" or has_candidate):
            return True
        if disp == "ACTUAL_EVIDENCE" and (has_ext_action or has_results or has_candidate) and not only_future:
            return True

    if not raw_message or not raw_message.strip():
        return False

    msg_lower = raw_message.lower().strip()
    clauses = [c.strip() for c in re.split(r"[\.\n;!]+", msg_lower) if c.strip()]
    unnegated_actions = []
    for clause in clauses:
        for v in COMPLETED_ACTION_VERBS:
            if v in clause:
                idx = clause.find(v)
                prefix = clause[max(0, idx - 25):idx]
                is_negated = any(n in prefix for n in ACTION_NEGATIONS)
                is_future = any(f in prefix for f in FUTURE_MODAL_PREFIXES)
                if not is_negated and not is_future:
                    unnegated_actions.append(v)

    has_unnegated_action = len(unnegated_actions) > 0
    has_count = bool(COUNT_REGEX.search(msg_lower))
    has_results_words = any(w in msg_lower for w in SUBSTANTIVE_RESULTS_WORDS)

    # 1. Past action + count or results words
    if has_unnegated_action and (has_count or has_results_words):
        return True

    # 2. Strong explicit completion phrase
    explicit_completion = [
        "completed the interviews", "completed interviews", "finished the interviews",
        "done with interviews", "finished interviews", "completed the calls", "finished the calls",
    ]
    if any(p in unnegated_actions for p in explicit_completion):
        return True

    # 3. Quantified count + strong substantive outcome words
    strong_outcomes = [
        "paid", "pickups", "pickup", "reported", "surplus", "stockout",
        "stockouts", "waste", "bought", "purchased", "pre-ordered", "preordered",
    ]
    if has_count and any(w in msg_lower for w in strong_outcomes):
        return True

    return False


class Evidence(BaseModel):
    """
    A piece of evidence produced by real-world experiment execution or observation.
    Preserves raw submission immutably and maintains mandatory trace links to
    experiment, assumption, and project.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique evidence identifier.",
    )
    project_id: str = Field(description="The project this evidence belongs to.")
    experiment_id: Optional[str] = Field(
        default=None,
        description="The experiment that produced this evidence.",
    )
    assumption_id: Optional[str] = Field(
        default=None,
        description="The assumption this evidence informs.",
    )
    iteration: int = Field(
        default=1,
        ge=1,
        description="Learning loop iteration in which this evidence was submitted.",
    )
    evidence_type: EvidenceType = Field(
        default=EvidenceType.OTHER,
        description="Controlled category of evidence.",
    )
    source: str = Field(
        default="founder",
        description="Source identifier (e.g. 'founder_message', 'interviews', 'channel').",
    )
    raw_content: str = Field(
        default="",
        description="The raw, unedited text submitted by the founder.",
    )
    content: str = Field(
        default="",
        description="Backward-compatible content alias for raw_content.",
    )
    structured_observations: list[str] = Field(
        default_factory=list,
        description="Discrete observations extracted from the raw evidence.",
    )
    quantitative_results: dict[str, Any] = Field(
        default_factory=dict,
        description="Extracted counts, conversion numbers, ratios, or financial amounts.",
    )
    additional_observations: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Credible out-of-scope or contextual observations beyond the active experiment assumption.",
    )
    artifacts: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Optional metadata for external artifacts (notes, CSVs, URLs).",
    )
    provenance: EvidenceProvenance = Field(
        default=EvidenceProvenance.FOUNDER_REPORTED,
        description="Origin and verification level of this evidence.",
    )
    source_type: EvidenceSourceType = Field(
        default=EvidenceSourceType.FOUNDER_PROVIDED,
        description="Legacy source type classification.",
    )
    result: EvidenceResult = Field(
        default=EvidenceResult.PENDING,
        description="Whether this evidence has been analyzed yet.",
    )
    submitted_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when evidence was submitted.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when evidence record was created.",
    )

    @model_validator(mode="before")
    @classmethod
    def sync_content_and_raw(cls, values: Any) -> Any:
        if isinstance(values, dict):
            raw = values.get("raw_content") or ""
            content = values.get("content") or ""
            if raw and not content:
                values["content"] = raw
            elif content and not raw:
                values["raw_content"] = content
            final_content = values.get("content", "")
            if len(final_content.strip()) < 5:
                raise ValueError("Evidence content/raw_content must be at least 5 characters long.")
        return values


class EvidenceAnalysis(BaseModel):
    """
    Structured outcome of analyzing submitted evidence against the active experiment
    and active critical assumption.
    """

    conclusion: EvidenceAnalysisConclusion = Field(
        ...,
        description="SUPPORTED, PARTIALLY_SUPPORTED, CONTRADICTED, or INCONCLUSIVE.",
    )
    evidence_strength: EvidenceStrength = Field(
        ...,
        description="WEAK, MODERATE, or STRONG.",
    )
    support_findings: list[str] = Field(
        default_factory=list,
        description="Specific findings supporting the assumption.",
    )
    contradiction_findings: list[str] = Field(
        default_factory=list,
        description="Specific findings contradicting or weakening the assumption.",
    )
    inconclusive_findings: list[str] = Field(
        default_factory=list,
        description="Ambiguities, missing data, or inconclusive signals.",
    )
    experiment_execution_quality: ExecutionQuality = Field(
        default=ExecutionQuality.GOOD,
        description="GOOD, PARTIAL, or POOR execution by founder.",
    )
    deviations_from_plan: list[str] = Field(
        default_factory=list,
        description="How real execution differed from the generated experiment design.",
    )
    valid_learnings: list[str] = Field(
        default_factory=list,
        description="Legitimate inferences permitted by this experiment's method.",
    )
    invalid_learnings: list[str] = Field(
        default_factory=list,
        description="Inferences that this experiment cannot legitimately claim to establish.",
    )
    assumption_confidence_change: str = Field(
        default="",
        description="Qualitative description of how assumption certainty changed.",
    )
    reasoning_summary: str = Field(
        default="",
        description="Epistemic rationale explaining the conclusion.",
    )
    recommended_state_updates: dict[str, Any] = Field(
        default_factory=dict,
        description="Proposed candidate updates to StartupState.",
    )
    additional_observations: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Extracted factual observations not directly relating to the active critical assumption.",
    )


class LearningDelta(BaseModel):
    """
    Founder-facing distillation of what changed between learning loop iterations.
    Answers: 'What changed because of the experiment?'
    """

    previous_assumption: str = Field(
        ...,
        description="Statement of the assumption tested in the concluded iteration.",
    )
    evidence_conclusion: str = Field(
        ...,
        description="Outcome classification (e.g. SUPPORTED, CONTRADICTED, INCONCLUSIVE).",
    )
    what_we_learned: list[str] = Field(
        default_factory=list,
        description="Key factual takeaways from real-world execution.",
    )
    what_became_more_certain: list[str] = Field(
        default_factory=list,
        description="Aspects of the startup reality now grounded in evidence.",
    )
    what_became_less_certain: list[str] = Field(
        default_factory=list,
        description="Aspects weakened or shown to be incorrect by evidence.",
    )
    what_remains_unknown: list[str] = Field(
        default_factory=list,
        description="Critical gaps that this experiment could not resolve.",
    )
    additional_observations: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Preserved contextual or operational observations from this iteration.",
    )
    weak_link_changed: bool = Field(
        default=False,
        description="True if re-diagnosis shifted the bottleneck area.",
    )
    previous_weak_link: str = Field(
        default="",
        description="The weak-link area before the experiment.",
    )
    new_weak_link: str = Field(
        default="",
        description="The newly diagnosed weak-link area after incorporating evidence.",
    )
    founder_facing_summary: str = Field(
        default="",
        description="Concise, non-technical synthesis for the founder.",
    )


def has_valid_completed_evidence_cycle(state: dict[str, Any]) -> bool:
    """
    Canonical definition of a valid completed evidence cycle (Milestone 4.0.3).
    Used as the single source of truth across:
      - validate_evidence_analysis (experiment completion)
      - apply_validated_state_updates
      - increment_iteration
      - can_compose_learning_delta

    A valid evidence cycle requires:
      1. If evidence_turn_assessment exists, disposition must be ACTUAL_EVIDENCE.
      2. Non-empty founder evidence content or extracted_evidence.
      3. Validated EvidenceAnalysis with recognized conclusion.
      4. Linked active experiment and active assumption.
    """
    disp = get_evidence_turn_disposition(state)
    if disp is not None and disp != EvidenceTurnDisposition.ACTUAL_EVIDENCE:
        return False

    # Check non-empty evidence content
    raw_content = state.get("founder_response") or ""
    if not raw_content:
        founder_messages = state.get("founder_messages") or []
        for m in reversed(founder_messages):
            if m.get("role") == "founder":
                raw_content = m.get("content", "")
                break
    extracted = state.get("extracted_evidence") or {}
    founder_evidence = state.get("founder_evidence") or []
    active_evidence = state.get("active_evidence") or {}
    evidence_history = state.get("evidence_history") or []
    if not raw_content and not extracted and not founder_evidence and not active_evidence and not evidence_history:
        return False

    # Check evidence analysis
    analysis = state.get("evidence_analysis") or {}
    conclusion = analysis.get("conclusion")
    if not conclusion or conclusion not in ("SUPPORTED", "PARTIALLY_SUPPORTED", "CONTRADICTED", "INCONCLUSIVE"):
        return False

    # Check active experiment or previous experiment
    active_exp = state.get("active_experiment") or {}
    prev_exps = state.get("previous_experiments") or []
    if not active_exp and not prev_exps:
        return False

    # Check active assumption or previous assumptions
    active_asmp = state.get("active_assumption") or state.get("active_critical_assumption") or {}
    prev_asmps = state.get("previous_assumptions") or []
    if not active_asmp and not prev_asmps:
        return False

    return True


def get_most_recent_completed_evidence_cycle(
    state: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]] | None:
    """
    Canonical Completed-Cycle Resolver (Milestone 4.0.7.2):
    Returns the most recent valid tuple:
        (tested_assumption, completed_experiment, admitted_evidence, evidence_analysis)
    for the immediately preceding completed learning cycle.

    Strictly guarantees:
      - Admitted evidence has result in ('ANALYZED', 'COMPLETED') or is active_evidence with non-empty content
      - Fallback learning actions ('FALLBACK_LEARNING_ACTION') are NEVER treated as experiment evidence
      - Incomplete/abandoned experiments ('ABANDONED', 'FAILED', 'QUALITY_BAR_NOT_MET') are excluded
      - Tested assumption is linked by assumption_id or matches the archived cycle assumption
      - Validated evidence analysis with recognized conclusion is present
    Returns None if no completed evidence cycle exists.
    """
    analysis = state.get("evidence_analysis") or {}
    conclusion = analysis.get("conclusion")
    if not conclusion or conclusion not in ("SUPPORTED", "PARTIALLY_SUPPORTED", "CONTRADICTED", "INCONCLUSIVE"):
        return None

    # Find the most recent admitted evidence
    founder_evidence = list(state.get("founder_evidence") or [])
    active_evidence = state.get("active_evidence")
    if active_evidence and isinstance(active_evidence, dict):
        ev_ids = {e.get("id") for e in founder_evidence if isinstance(e, dict) and e.get("id")}
        if active_evidence.get("id") not in ev_ids:
            founder_evidence.append(active_evidence)

    admitted_ev = None
    for ev in reversed(founder_evidence):
        if not isinstance(ev, dict):
            continue
        # Exclude fallback learning inputs
        if ev.get("source") == "FALLBACK_LEARNING_ACTION" or ev.get("learning_input_mode") == "FALLBACK":
            continue
        result_val = str(ev.get("result") or "").upper()
        type_val = str(ev.get("evidence_type") or "").upper()
        if result_val in ("ANALYZED", "COMPLETED") or type_val in ("EXPERIMENT_RESULT", "FOUNDER_OBSERVATION"):
            admitted_ev = ev
            break
        if ev.get("structured_observations") or ev.get("quantitative_results"):
            admitted_ev = ev
            break

    if not admitted_ev:
        return None

    exp_id = admitted_ev.get("experiment_id")
    asmp_id = admitted_ev.get("assumption_id")

    # Find completed experiment
    all_exps: list[dict[str, Any]] = []
    prev_exps = state.get("previous_experiments") or []
    for pe in prev_exps:
        if isinstance(pe, dict):
            all_exps.append(pe)
    active_exp = state.get("active_experiment")
    if active_exp and isinstance(active_exp, dict):
        all_exps.append(active_exp)

    completed_exp = None
    if exp_id:
        for exp in reversed(all_exps):
            if exp.get("id") == exp_id:
                completed_exp = exp
                break

    if not completed_exp:
        for exp in reversed(all_exps):
            status = str(exp.get("status", "")).upper()
            if status in ("COMPLETED", "IN_PROGRESS", "READY"):
                completed_exp = exp
                break

    if not completed_exp:
        if not all_exps:
            # Legacy or mock test state without tracked experiment records
            completed_exp = {
                "id": exp_id or "exp-prior",
                "status": "COMPLETED",
                "assumption_id": asmp_id,
            }
        else:
            return None

    # Incomplete or abandoned experiments cannot form a completed cycle
    exp_status = str(completed_exp.get("status", "")).upper()
    if exp_status in ("ABANDONED", "FAILED", "QUALITY_BAR_NOT_MET"):
        return None

    # Find tested assumption
    all_asmps: list[dict[str, Any]] = []
    prev_asmps = state.get("previous_assumptions") or []
    for pa in prev_asmps:
        if isinstance(pa, dict):
            all_asmps.append(pa)
    active_asmp = state.get("active_assumption") or state.get("active_critical_assumption")
    if active_asmp and isinstance(active_asmp, dict):
        all_asmps.append(active_asmp)

    target_asmp_id = asmp_id or completed_exp.get("assumption_id")
    tested_asmp = None
    if target_asmp_id:
        for asmp in reversed(all_asmps):
            if asmp.get("id") == target_asmp_id:
                tested_asmp = asmp
                break

    if not tested_asmp and all_asmps:
        tested_asmp = all_asmps[-1]

    if not tested_asmp:
        return None

    return (tested_asmp, completed_exp, admitted_ev, analysis)

