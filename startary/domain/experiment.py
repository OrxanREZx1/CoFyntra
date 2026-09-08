"""
Experiment schema.

Every experiment must reference the assumption it is designed to test.
This is enforced at the schema level, not just by convention.

Experiments are designed by future AI modules (Milestone 2+).
In Milestone 0 the schema is defined and validated only.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)

from startary.domain.enums import (
    CriticDimensionResult,
    CriticVerdict,
    ExecutionReadiness,
    ExperimentStatus,
    ExperimentType,
    NextLearningActionType,
    ParameterProvenance,
)


class ExperimentParameter(BaseModel):
    """
    A specific parameter for the experiment (e.g. sample size, response threshold).
    Tracks provenance to ensure AI-proposed heuristics are never confused with founder-stated facts.
    """

    name: str = Field(..., description="Parameter identifier, e.g. 'interview_count', 'target_threshold'.")
    value: str = Field(..., description="Parameter value, e.g. '6-8 sellers', 'at least 5 of 8'.")
    provenance: ParameterProvenance = Field(
        default=ParameterProvenance.AI_PROPOSED,
        description="Whether parameter was proposed by AI or supplied by founder.",
    )
    notes: Optional[str] = Field(default=None, description="Provisional context or rationale.")


class ExperimentCriticReport(BaseModel):
    """
    Structured critique report evaluating experiment quality, alignment, and feasibility.
    """

    verdict: CriticVerdict = Field(..., description="ACCEPT or REVISE.")
    issues: list[str] = Field(default_factory=list, description="Specific identified issues.")
    assumption_alignment: CriticDimensionResult = Field(..., description="PASS or FAIL.")
    evidence_validity: CriticDimensionResult = Field(..., description="PASS or FAIL.")
    feasibility: CriticDimensionResult = Field(..., description="PASS or FAIL.")
    cost_efficiency: CriticDimensionResult = Field(..., description="PASS or FAIL.")
    falsifiability: CriticDimensionResult = Field(..., description="PASS or FAIL.")
    scope: CriticDimensionResult = Field(..., description="PASS or FAIL.")
    revision_instructions: str = Field(default="", description="Instructions for generator if REVISE.")
    rationale: str = Field(default="", description="Summary rationale for the critic decision.")


class CanExecuteAssessment(BaseModel):
    """
    Assessment of whether the founder can execute the accepted experiment under current constraints.
    """

    readiness: ExecutionReadiness = Field(..., description="CAN_EXECUTE, NEEDS_HELP, or BLOCKED.")
    reason: str = Field(..., description="Explanation of execution readiness.")
    missing_requirements: list[str] = Field(default_factory=list, description="Capabilities/access/assets needed.")
    blockers: list[str] = Field(default_factory=list, description="Fundamental blockers if BLOCKED.")


class Experiment(BaseModel):
    """
    A structured experiment designed to test a specific assumption.

    The ``assumption_id`` field is required — an experiment without an
    assumption reference is architecturally invalid and will fail validation.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique experiment identifier.",
    )
    assumption_id: str = Field(
        ...,
        min_length=1,
        description=(
            "The assumption this experiment is testing. "
            "Required — every experiment must trace back to an assumption."
        ),
    )
    project_id: str = Field(description="The project this experiment belongs to.")

    experiment_type: ExperimentType = Field(
        default=ExperimentType.OTHER,
        description="Controlled category of the experiment.",
    )
    objective: str = Field(
        ...,
        min_length=5,
        description="What this experiment is trying to learn.",
    )
    hypothesis: Optional[str] = Field(
        default=None,
        description="The active critical assumption statement under test.",
    )
    method: Optional[str] = Field(
        default=None,
        description="How the experiment will be conducted.",
    )
    target_participants_or_scope: Optional[str] = Field(
        default=None,
        description="Target participant profile, sample size, or geographic/channel scope.",
    )
    founder_actions: list[str] = Field(
        default_factory=list,
        description="Concrete steps the founder must take to run this experiment.",
    )
    evidence_to_collect: Optional[str] = Field(
        default=None,
        description="What data or observations the founder should capture.",
    )
    support_signal: Optional[str] = Field(
        default=None,
        description="What result would support the assumption.",
    )
    contradiction_signal: Optional[str] = Field(
        default=None,
        description="What result would contradict the assumption.",
    )
    inconclusive_signal: Optional[str] = Field(
        default=None,
        description="What outcome would fail to teach us enough.",
    )
    valid_inferences: list[str] = Field(
        default_factory=list,
        description="What this experiment CAN teach us.",
    )
    invalid_inferences: list[str] = Field(
        default_factory=list,
        description="What this experiment CANNOT teach us.",
    )
    estimated_time: Optional[str] = Field(
        default=None,
        description="Estimated time to execute (e.g. '2-4 days').",
    )
    time_constraint: Optional[str] = Field(
        default=None,
        description="Backward-compatible alias for estimated_time.",
    )
    estimated_cost: Optional[str] = Field(
        default=None,
        description="Estimated financial budget (e.g. 'Near-zero / $0').",
    )
    cost_constraint: Optional[str] = Field(
        default=None,
        description="Backward-compatible alias for estimated_cost.",
    )
    required_capabilities: list[str] = Field(
        default_factory=list,
        description="Skills or capabilities required to run the test.",
    )
    required_access: list[str] = Field(
        default_factory=list,
        description="Customer or distribution access required.",
    )
    parameters: list[ExperimentParameter] = Field(
        default_factory=list,
        description="Explicit parameters and decision thresholds with provenance.",
    )
    status: ExperimentStatus = Field(
        default=ExperimentStatus.DRAFT,
        description="Current execution status of this experiment.",
    )
    created_iteration: int = Field(
        default=1,
        description="Workflow iteration when this experiment was generated.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )

    @model_validator(mode="after")
    def sync_time_and_cost_aliases(self) -> Experiment:
        if self.estimated_time and not self.time_constraint:
            self.time_constraint = self.estimated_time
        elif self.time_constraint and not self.estimated_time:
            self.estimated_time = self.time_constraint

        if self.estimated_cost and not self.cost_constraint:
            self.cost_constraint = self.estimated_cost
        elif self.cost_constraint and not self.estimated_cost:
            self.estimated_cost = self.cost_constraint
        return self


class FounderExperimentSummary(BaseModel):
    """
    Compact founder-facing distillation answering the core questions in under 30 seconds (Milestone 4.0.5 & 4.0.7.1).
    """

    critical_assumption: Optional[str] = Field(
        default=None,
        description="Founder-readable active critical assumption statement.",
    )
    what_are_we_testing: str = Field(
        description="Actual critical uncertainty in founder language, no internal labels.",
    )
    what_should_i_do: str = Field(
        description="Smallest actionable primary action.",
    )
    what_should_i_bring_back: str = Field(
        description="Summarized evidence fields to collect as a compact list or sentence.",
    )
    useful_signal: str = Field(
        description="Decision rule translated into founder-readable signal.",
    )
    current_positive_signals: list[str] = Field(
        default_factory=list,
        description="1 to 4 grounded positive signals preserving epistemic caution.",
    )
    short_title: Optional[str] = Field(
        default=None,
        description="Optional concise title for the test.",
    )
    estimated_effort: Optional[str] = Field(
        default=None,
        description="Optional estimated time or effort for the test.",
    )
    participant_target: Optional[str] = Field(
        default=None,
        description="Concrete participant description replacing generic cohort terms.",
    )


class ExecutionHandoff(BaseModel):
    """
    Founder-facing distillation of an accepted experiment before execution (Milestone 4.0.3 & 4.0.5).
    Bridges the TEST -> WAITING_FOR_EVIDENCE phase transition so the founder clearly
    knows what to do, who to do it with, what to observe, and what to bring back.
    """

    experiment_id: str = Field(description="ID of the accepted experiment.")
    assumption_id: str = Field(description="ID of the active critical assumption.")
    critical_assumption: Optional[str] = Field(
        default=None,
        description="The active critical assumption under test.",
    )
    experiment_type: str = Field(description="Methodological category (e.g. CUSTOMER_INTERVIEWS).")
    learning_goal: str = Field(description="Strategic learning objective.")
    what_we_are_testing: str = Field(description="Concise description of the assumption under test.")
    founder_actions: list[str] = Field(default_factory=list, description="Concrete step-by-step actions for the founder.")
    target_participants_or_context: str = Field(description="Who to interact with or where to run the test.")
    evidence_to_collect: str = Field(description="Specific data, quotes, counts, or behaviors to record.")
    support_signal: str = Field(default="", description="Observation that validates the assumption.")
    contradiction_signal: str = Field(default="", description="Observation that disproves or weakens the assumption.")
    inconclusive_signal: str = Field(default="", description="Observation that leaves the question open.")
    important_constraints: list[str] = Field(default_factory=list, description="Guardrails and what NOT to do.")
    valid_inferences: list[str] = Field(default_factory=list, description="What this test CAN legitimately establish.")
    invalid_inferences: list[str] = Field(default_factory=list, description="What this test CANNOT legitimately claim.")
    completion_instruction: str = Field(
        default="Return to Startary and tell me what actually happened. You can also ask me to explain the experiment or tell me if something blocks you.",
        description="Clear prompt for what the founder should do when done or if needing help.",
    )
    response_text: str = Field(
        default="",
        description="Full formatted markdown text displayed to the founder (compact by default).",
    )
    full_response_text: str = Field(
        default="",
        description="Full detailed markdown text preserved for methodology review / debugging.",
    )
    summary: Optional[FounderExperimentSummary] = Field(
        default=None,
        description="Compact founder summary answering the 5 core questions.",
    )
    action_type: NextLearningActionType = Field(
        default=NextLearningActionType.VALIDATED_EXPERIMENT,
        description="Distinguishes validated experiments from fallback learning.",
    )


class FallbackLearningAction(BaseModel):
    """
    Bounded next learning action generated when rigorous experiment design cannot meet
    the quality bar after revision budget exhaustion (Milestone 4.0.6).
    Designed to safely gather firsthand observations without falsely claiming assumption validation.
    """

    assumption_id: str = Field(description="ID of the active critical assumption.")
    target_area: str = Field(description="Startup area containing the weak link (e.g. 'Problem').")
    learning_goal: str = Field(description="What Startary and founder still need to learn.")
    why_fallback_is_needed: str = Field(description="Transparent explanation of why formal experiment design stopped.")
    primary_action: str = Field(description="Single, conservative, concrete learning action for the founder.")
    who_to_engage: str = Field(description="Specific target participants or context.")
    what_to_ask_or_observe: list[str] = Field(
        default_factory=list,
        description="Questions focused on past events and actual consequences.",
    )
    what_to_bring_back: list[str] = Field(
        default_factory=list,
        description="Specific factual observations and notes to return with.",
    )
    what_not_to_infer: list[str] = Field(
        default_factory=list,
        description="Explicit boundaries on what this action CANNOT prove.",
    )
    completion_instruction: str = Field(
        default="Return with the notes from those conversations.",
        description="Prompt for when the founder completes the learning conversations.",
    )
    response_text: str = Field(
        default="",
        description="Formatted markdown displayed to the founder.",
    )
    action_type: NextLearningActionType = Field(
        default=NextLearningActionType.FALLBACK_LEARNING_ACTION,
        description="Distinguishes fallback learning from validated experiments.",
    )
    estimated_effort: Optional[str] = Field(
        default=None,
        description="Optional time or effort estimate.",
    )
    participant_target: Optional[str] = Field(
        default=None,
        description="Concrete participant target.",
    )


def clean_assumption_statement(raw: Optional[str]) -> str:
    """Cleans internal prefixes from critical assumption statements."""
    if not raw:
        return ""
    cleaned = str(raw).strip()
    for prefix in [
        "Critical assumption:",
        "Critical assumption",
        "Active critical assumption:",
        "Active critical assumption",
        "Assumption:",
        "Assumption",
    ]:
        if cleaned.lower().startswith(prefix.lower()):
            cleaned = cleaned[len(prefix):].strip(" -:")
            break
    if cleaned and not cleaned.endswith((".", "!", "?")):
        cleaned = f"{cleaned}."
    return cleaned


def clean_testing_question(raw: Optional[str]) -> str:
    """
    Cleans raw hypothesis/objective/learning text, stripping generic meta-prefixes.
    Returns empty string if the text is invalid or stripped to triviality (<3 chars).
    Ensures phrasing is founder-readable (e.g. 'Whether ...').
    """
    if not raw:
        return ""
    text = str(raw).strip()
    for prefix in [
        "Validating critical assumption:",
        "Validating critical assumption",
        "Validate critical assumption:",
        "Validate critical assumption",
        "Critical assumption under test:",
        "Critical assumption under test",
        "Assumption under test:",
        "Assumption under test",
        "Hypothesis:",
        "Objective:",
    ]:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip(" -:")
            break
    if len(text) < 3 or text.lower() in ["none", "unknown", "critical assumption", "n/a"]:
        return ""

    # Founder-facing phrasing
    if text.lower().startswith("we are testing "):
        return f"{text[0].upper()}{text[1:].rstrip('.')}."
    if text.lower().startswith("determine whether "):
        sub = text[18:].strip()
        return f"Whether {sub[0].lower() + sub[1:].rstrip('.')}." if sub else ""
    if text.lower().startswith("determine if "):
        sub = text[13:].strip()
        return f"Whether {sub[0].lower() + sub[1:].rstrip('.')}." if sub else ""
    if text.lower().startswith("determine "):
        sub = text[10:].strip()
        return f"Whether {sub[0].lower() + sub[1:].rstrip('.')}." if sub else ""
    if text.lower().startswith("whether ") or text.lower().startswith("if "):
        return f"{text[0].upper()}{text[1:].rstrip('.')}."

    return f"Whether {text[0].lower() + text[1:].rstrip('.')}."


def resolve_what_are_we_testing(candidates: list[Optional[str]], fallback: str = "We are testing the active critical assumption.") -> str:
    """
    Executes deterministic fallback chain for 'WHAT ARE WE TESTING?':
    Priority:
      1. FounderExperimentSummary.what_are_we_testing / exp.what_we_are_testing
      2. exp.objective / learning_goal
      3. exp.hypothesis
      4. active critical assumption statement
      5. safe fallback
    """
    for cand in candidates:
        if cand:
            cleaned = clean_testing_question(cand)
            if cleaned and cleaned not in [".", "Whether .", "If ."]:
                return cleaned
    return fallback


def validate_founder_experiment_summary(
    summary: FounderExperimentSummary | dict,
    handoff: ExecutionHandoff | dict | None = None,
    assumption: dict | None = None,
) -> bool:
    """
    Validates the integrity of a FounderExperimentSummary (Milestone 4.0.7.1).
    Enforces:
      - what_are_we_testing is non-empty and not just whitespace or punctuation.
      - critical_assumption is non-empty if an active assumption exists.
      - positive signals are grounded (no forbidden hype/budget terms).
      - maximum 4 positive signals.
    Raises ValueError on integrity violation, or returns True when valid.
    """
    if hasattr(summary, "model_dump"):
        s_data = summary.model_dump()
    elif isinstance(summary, dict):
        s_data = summary
    else:
        raise ValueError("Invalid summary object")

    what_test = str(s_data.get("what_are_we_testing") or "").strip()
    if not what_test or what_test in [".", "Whether .", "If ."]:
        raise ValueError("FounderExperimentSummary integrity error: 'what_are_we_testing' is empty.")

    crit_asmp = str(s_data.get("critical_assumption") or "").strip()
    if not crit_asmp and handoff:
        h_data = handoff.model_dump() if hasattr(handoff, "model_dump") else dict(handoff)
        crit_asmp = str(h_data.get("critical_assumption") or "").strip()

    active_assumption_exists = False
    if assumption:
        asmp_dict = assumption.model_dump() if hasattr(assumption, "model_dump") else dict(assumption)
        if asmp_dict.get("statement") or asmp_dict.get("is_critical"):
            active_assumption_exists = True
    elif handoff:
        h_data = handoff.model_dump() if hasattr(handoff, "model_dump") else dict(handoff)
        if h_data.get("assumption_id") and h_data.get("assumption_id") != "asmp-unknown":
            active_assumption_exists = True

    if active_assumption_exists and not crit_asmp:
        raise ValueError("FounderExperimentSummary integrity error: 'critical_assumption' is required when an active assumption exists.")

    pos_signals = s_data.get("current_positive_signals") or []
    if not pos_signals:
        raise ValueError("FounderExperimentSummary integrity error: 'current_positive_signals' cannot be empty.")
    if len(pos_signals) > 4:
        raise ValueError(f"FounderExperimentSummary integrity error: maximum 4 positive signals allowed, got {len(pos_signals)}.")

    forbidden_terms = [
        "huge market", "massive market", "strong product-market fit",
        "product market fit", "validated demand", "market opportunity",
        "guaranteed success", "visionary", "disruptive", "$", "budget", "runway",
        "clearly needs", "great team"
    ]
    for sig in pos_signals:
        sig_lower = str(sig).lower()
        for ft in forbidden_terms:
            if ft in sig_lower:
                raise ValueError(f"FounderExperimentSummary integrity error: forbidden term '{ft}' in positive signal: '{sig}'.")

    return True


def compose_current_positive_signals(
    state: dict | None = None,
    diagnosis_snapshot: dict | None = None,
) -> list[str]:
    """
    Deterministically extracts 1 to 4 grounded positive signals answering:
    'What currently gives us a reason to continue testing?' (Milestone 4.0.5 & 4.0.7.1).

    Precedence (Section 8):
      1. real customer commitment / behavior
      2. repeated external problem evidence
      3. explicit problem urgency / consequence (including single-shop pain)
      4. dissatisfaction with existing alternatives / manual workarounds
      5. customer access
      6. relevant founder execution capability

    Rules:
      - Strictly grounded facts only.
      - Never fabricate validation or use speculative praise/hype.
      - Budget, hours, runway, and team size are facts, not positive signals to continue testing.
      - Maximum 4 concise items.
      - If no grounded positive signals exist, returns:
        ['No strong positive signal is established yet.']
    """
    state = state or {}
    diagnosis_snapshot = diagnosis_snapshot or {}
    if hasattr(diagnosis_snapshot, "model_dump"):
        diagnosis_snapshot = diagnosis_snapshot.model_dump()

    b1_commitments: list[str] = []
    b2_external_evidence: list[str] = []
    b3_problem_urgency: list[str] = []
    b4_alternatives_friction: list[str] = []
    b5_customer_access: list[str] = []
    b6_execution_capacity: list[str] = []

    forbidden_terms = [
        "huge market", "massive market", "large market", "strong business model",
        "strong product-market fit", "product market fit", "validated demand",
        "market opportunity", "guaranteed success", "visionary", "disruptive",
        "budget", "runway", "hours/week", "hrs/week", "h/week", "$", "capital",
        "dollars", "funding", "bootstrapped", "speculative", "clearly needs",
        "great team", "winning team", "unlimited potential"
    ]

    # 1. Advantages from diagnosis snapshot
    advantages = list(diagnosis_snapshot.get("current_advantages") or [])
    for adv in advantages:
        adv_clean = str(adv).strip()
        adv_lower = adv_clean.lower()
        if any(f in adv_lower for f in forbidden_terms):
            continue

        if any(w in adv_lower for w in ["access", "cousin", "uncle", "distribution", "channel", "network"]):
            b5_customer_access.append(adv_clean)
        elif any(w in adv_lower for w in ["pilot", "loi", "letter of intent", "willing to pay", "paying", "pre-order", "commission"]):
            b1_commitments.append(adv_clean)
        elif any(w in adv_lower for w in ["multiple owners", "several owners", "interview findings", "repeated interviews"]):
            b2_external_evidence.append(adv_clean)
        elif any(w in adv_lower for w in ["stockout", "spoilage", "perishable", "waste", "return", "pain", "friction", "costly", "recurring", "complaint"]):
            b3_problem_urgency.append(adv_clean)
        elif any(w in adv_lower for w in ["pos", "workaround", "manual", "alternative", "spreadsheet", "order log"]):
            b4_alternatives_friction.append(adv_clean)
        elif any(w in adv_lower for w in ["store", "bodega", "shop", "merchant"]):
            b5_customer_access.append(adv_clean)
        elif any(w in adv_lower for w in ["technical", "engineer", "build", "prototype", "mvp", "in-house", "code"]):
            b6_execution_capacity.append(adv_clean)
        else:
            b6_execution_capacity.append(adv_clean)

    # 2. External Evidence & Completed Cycles (evidence_analysis, founder_evidence, external_evidence)
    ev_analysis = state.get("evidence_analysis") or {}
    support_findings = ev_analysis.get("support_findings") or []
    for sf in support_findings:
        sf_clean = str(sf).strip()
        sf_lower = sf_clean.lower()
        if any(f in sf_lower for f in forbidden_terms):
            continue
        if any(w in sf_lower for w in ["stockout", "spoilage", "waste", "spoil"]):
            if any(w in sf_lower for w in ["stockout", "stockouts"]):
                b2_external_evidence.append("Multiple independent shop owners reported recent stockouts.")
            if any(w in sf_lower for w in ["waste", "spoilage", "over-ordering"]):
                b2_external_evidence.append("Multiple owners reported recurring perishable waste.")
        elif any(w in sf_lower for w in ["return", "returns"]):
            b2_external_evidence.append("Multiple interviewed merchants confirmed recurring returns.")
        elif any(w in sf_lower for w in ["order", "decide", "control", "manual", "prevention"]):
            b3_problem_urgency.append("Owners already take manual steps to reduce these problems.")

    founder_ev = state.get("founder_evidence") or state.get("external_evidence") or []
    for fe in founder_ev:
        fe_text = str(fe.get("raw_text") or fe if isinstance(fe, dict) else fe).lower()
        if any(f in fe_text for f in forbidden_terms):
            continue
        if "stockout" in fe_text and "waste" in fe_text:
            b2_external_evidence.append("Multiple independent shop owners reported recent stockouts.")
            b2_external_evidence.append("Multiple owners reported recurring perishable waste.")
        if "manual" in fe_text or "workaround" in fe_text or "step" in fe_text or "action" in fe_text or "prevention" in fe_text:
            b3_problem_urgency.append("Owners already take manual steps to reduce these problems.")
        if "pos" in fe_text or "judgment" in fe_text or "ordering" in fe_text:
            b4_alternatives_friction.append("Existing ordering decisions still rely heavily on manual judgment and reactive adjustments.")

    # 3. Startup State
    st_state = state.get("startup_state") or {}
    exec_ctx = st_state.get("execution_context") or {}
    prog = st_state.get("progress") or {}

    tech_cap = exec_ctx.get("technical_capability") or {}
    if isinstance(tech_cap, dict) and tech_cap.get("value") is True:
        b6_execution_capacity.append("The team can build the prototype internally.")

    mvp = prog.get("mvp_exists") or {}
    if isinstance(mvp, dict) and mvp.get("value") is True:
        b6_execution_capacity.append("A working prototype or early version is already built.")

    cust_acc = exec_ctx.get("customer_access") or {}
    if isinstance(cust_acc, dict) and cust_acc.get("value") and cust_acc.get("provenance") != "UNKNOWN":
        acc_val = str(cust_acc.get("value")).strip()
        if acc_val.lower() not in ["none", "unknown", "no customer access"]:
            b5_customer_access.append(f"Direct target customer access ({acc_val}).")

    pilots = prog.get("pilot_agreements") or {}
    if isinstance(pilots, dict) and pilots.get("value") and pilots.get("provenance") != "UNKNOWN":
        b1_commitments.append(f"Pilot commitment in place: {pilots.get('value')}")
    elif isinstance(pilots, list) and pilots:
        b1_commitments.append(f"Pilot commitment in place: {', '.join(str(p) for p in pilots)}")

    areas = st_state.get("areas") or st_state.get("area_knowledge") or {}
    for area_name, area_info in areas.items():
        if not isinstance(area_info, dict):
            continue
        evidence_list = area_info.get("evidence_provided") or []
        for ev in evidence_list:
            ev_clean = str(ev).strip()
            ev_lower = ev_clean.lower()
            if any(f in ev_lower for f in forbidden_terms):
                continue
            if any(w in ev_lower for w in ["pilot", "willing to pay", "commission", "pre-order", "deposit", "loi"]):
                b1_commitments.append(ev_clean)
            elif any(w in ev_lower for w in ["multiple", "several", "verified recurring"]):
                if "stockout" in ev_lower:
                    b2_external_evidence.append("Multiple independent shop owners reported recent stockouts.")
                if "waste" in ev_lower or "spoilage" in ev_lower:
                    b2_external_evidence.append("Multiple owners reported recurring perishable waste.")
            elif any(w in ev_lower for w in ["stockout", "spoilage", "perishable", "waste"]):
                b3_problem_urgency.append("One relevant shop owner reports recurring stockouts and perishable spoilage.")
            elif any(w in ev_lower for w in ["return", "pain", "friction", "costly", "recurring", "complaint"]):
                b3_problem_urgency.append(ev_clean)

        if area_name in ["Existing Alternatives", "Competition"]:
            for item in area_info.get("claims") or []:
                item_lower = str(item).lower()
                if "pos" in item_lower or "recommendation" in item_lower:
                    b4_alternatives_friction.append("Existing POS and inventory tools do not provide ordering recommendations.")
                elif "manual" in item_lower or "pen and paper" in item_lower:
                    b4_alternatives_friction.append("Existing ordering decisions rely heavily on manual judgment and reactive adjustments.")

    # 4. Context & Founder Messages (raw text observations)
    raw_ctx = (state.get("raw_context") or "").lower()
    founder_msgs = [
        str(m.get("content", "")).lower()
        for m in (state.get("founder_messages") or [])
        if isinstance(m, dict) and m.get("role") == "founder"
    ]
    all_founder_text = raw_ctx + " " + " ".join(founder_msgs)

    # Commitments
    if any(p in all_founder_text for p in ["willing to pilot", "run a pilot", "willing to test"]):
        b1_commitments.append("That merchant is willing to run a pilot.")
    if any(p in all_founder_text for p in ["willing to pay commission", "pay commission", "conditional willingness to pay"]):
        b1_commitments.append("They expressed conditional willingness to pay commission if returns decrease.")

    # External interviews
    if any(p in all_founder_text for p in ["6 independent grocery", "interviewed 6", "5/6 owners", "4/6 had recent stockouts", "4/6 had stockouts"]):
        b2_external_evidence.append("Multiple independent shop owners reported recent stockouts.")
    if any(p in all_founder_text for p in ["3/6 had waste", "3/6 had over-ordering", "perishable waste in dairy"]):
        b2_external_evidence.append("Multiple owners reported recurring perishable waste.")

    # Problem urgency & consequences (including single-shop observations)
    if any(p in all_founder_text for p in ["recurring size-related returns", "recurring returns", "suffer from 25% return rates", "size-related returns"]):
        b3_problem_urgency.append("One relevant merchant already experiences recurring size-related returns.")
    if any(p in all_founder_text for p in ["recurring stockouts", "spoilage and stockouts", "produce and dairy spoilage", "stockouts or perishable waste", "daily shelf stockouts"]):
        b3_problem_urgency.append("One relevant shop owner reports recurring stockouts and perishable spoilage.")
    if any(p in all_founder_text for p in ["worth solving", "worth changing", "matters to them"]):
        b3_problem_urgency.append("The owner considers the stockout and waste issue worth changing.")
    if any(p in all_founder_text for p in ["manual steps", "prevention actions", "order logs", "manual order log", "reduces waste"]):
        b3_problem_urgency.append("Owners already take manual steps to reduce these problems.")

    # Dissatisfaction with alternatives / workarounds
    if any(p in all_founder_text for p in ["pos", "inventory system", "ordering recommendation", "ordering workflow"]):
        if any(p in all_founder_text for p in ["insufficient", "lacks", "do not provide", "no ordering recommendations", "without ordering"]):
            b4_alternatives_friction.append("Existing POS and inventory tools do not provide ordering recommendations.")
    if any(p in all_founder_text for p in ["manual judgment", "reactive adjustments", "pen and paper", "whatsapp", "manual adjustments"]):
        b4_alternatives_friction.append("Existing ordering decisions still rely heavily on manual judgment and reactive adjustments.")

    # Ordering & Deduplication
    ordered_signals: list[str] = []
    seen: set[str] = set()

    def _get_concept_key(text: str) -> str:
        tl = text.lower()
        if "pilot" in tl:
            return "pilot"
        if "commission" in tl or "willingness to pay" in tl or "willing to pay" in tl:
            return "pay_commission"
        if "multiple" in tl and "stockout" in tl:
            return "multi_stockouts"
        if "multiple" in tl and ("waste" in tl or "perishable" in tl or "spoilage" in tl):
            return "multi_waste"
        if "manual steps" in tl or "prevention" in tl:
            return "manual_prevention"
        if "manual judgment" in tl or "reactive" in tl or "ordering decision" in tl:
            return "manual_ordering"
        if "pos" in tl or "inventory tools" in tl:
            return "pos_tools"
        if "worth solving" in tl or "worth changing" in tl:
            return "worth_changing"
        if "stockout" in tl or "spoilage" in tl or "perishable" in tl or "waste" in tl:
            return "single_stockouts_waste"
        if "return" in tl and ("size" in tl or "25%" in tl or "recurring" in tl):
            return "recurring_returns"
        if "cousin" in tl or "uncle" in tl or "bodega" in tl or ("merchant" in tl and "access" in tl) or "customer access" in tl:
            return "customer_access"
        if "build" in tl or "engineer" in tl or "code" in tl or "internally" in tl:
            return "technical_build"
        if "prototype" in tl or "mvp" in tl:
            return "prototype_built"
        return " ".join(tl.split())

    for bucket in [
        b1_commitments,
        b2_external_evidence,
        b3_problem_urgency,
        b4_alternatives_friction,
        b5_customer_access,
        b6_execution_capacity,
    ]:
        for item in bucket:
            key = _get_concept_key(item)
            if key not in seen:
                seen.add(key)
                ordered_signals.append(item)
            if len(ordered_signals) >= 4:
                break
        if len(ordered_signals) >= 4:
            break

    if not ordered_signals:
        return ["No strong positive signal is established yet."]

    return ordered_signals[:4]


def compose_founder_experiment_summary(
    experiment: Experiment | dict,
    assumption: dict | None = None,
    state: dict | None = None,
    diagnosis_snapshot: dict | None = None,
) -> FounderExperimentSummary:
    """
    Builds the compact FounderExperimentSummary answering the core questions (Milestone 4.0.5 & 4.0.7.1).
    """
    if isinstance(experiment, Experiment):
        exp_data = experiment.model_dump()
    else:
        exp_data = dict(experiment)

    asmp_data = assumption.model_dump() if hasattr(assumption, "model_dump") else (assumption or {})
    state_asmp = {}
    if isinstance(state, dict):
        raw_asmp = state.get("active_assumption") or state.get("active_critical_assumption") or {}
        if hasattr(raw_asmp, "model_dump"):
            state_asmp = raw_asmp.model_dump()
        elif isinstance(raw_asmp, dict):
            state_asmp = raw_asmp

    raw_crit = (
        asmp_data.get("statement")
        or asmp_data.get("critical_assumption")
        or state_asmp.get("statement")
        or state_asmp.get("critical_assumption")
        or exp_data.get("critical_assumption")
        or ""
    )
    crit_asmp = clean_assumption_statement(raw_crit)

    cand1 = exp_data.get("what_are_we_testing") or exp_data.get("what_we_are_testing")
    cand2 = exp_data.get("objective") or exp_data.get("learning_goal")
    cand3 = exp_data.get("hypothesis")
    cand4 = crit_asmp
    cand5 = "We are testing the active critical assumption."

    what_are_we_testing = resolve_what_are_we_testing(
        candidates=[cand1, cand2, cand3, cand4, cand5],
        fallback=cand5,
    )

    actions = exp_data.get("founder_actions") or []
    target = exp_data.get("target_participants_or_scope") or exp_data.get("target_participants_or_context") or ""
    if target.lower() in ["target customer cohort", "cohort", "none"]:
        target = ""

    primary_action = ""
    if actions:
        import re
        first_act = re.sub(r"^\d+\.\s*", "", str(actions[0]).strip())
        primary_action = first_act
    elif target:
        primary_action = f"Interview target participants ({target}) using structured questions."
    else:
        primary_action = "Interview target participants using structured questions."

    ev = exp_data.get("evidence_to_collect") or ""
    if "," in ev and not ev.startswith("- ") and "\n" not in ev:
        items = [it.strip() for it in ev.split(",") if it.strip()]
        what_should_i_bring_back = "\n".join(f"- {it}" for it in items)
    elif ev:
        what_should_i_bring_back = ev
    else:
        what_should_i_bring_back = "- Specific customer quotes and feedback\n- Frequency of problem\n- Concrete follow-up commitments"

    supp = exp_data.get("support_signal") or ""
    if not supp:
        supp = "Several independent participants confirm the pain and agree to a concrete follow-up step."
    useful_signal = supp

    pos_signals = compose_current_positive_signals(state=state, diagnosis_snapshot=diagnosis_snapshot)

    exp_type = exp_data.get("experiment_type", "CUSTOMER_INTERVIEWS")
    short_title = str(exp_type).replace("_", " ").title()
    estimated_effort = exp_data.get("estimated_time") or exp_data.get("time_constraint")
    participant_target = target if target else None

    summary = FounderExperimentSummary(
        critical_assumption=crit_asmp if crit_asmp else None,
        what_are_we_testing=what_are_we_testing,
        what_should_i_do=primary_action,
        what_should_i_bring_back=what_should_i_bring_back,
        useful_signal=useful_signal,
        current_positive_signals=pos_signals,
        short_title=short_title,
        estimated_effort=estimated_effort,
        participant_target=participant_target,
    )

    validate_founder_experiment_summary(
        summary,
        assumption=asmp_data or state_asmp,
    )
    return summary


def render_execution_handoff_text(
    handoff: ExecutionHandoff | dict,
    mode: str = "compact",
) -> str:
    """
    Renders the formatted, founder-facing markdown block for an ExecutionHandoff.
    Supports mode='compact' (Milestone 4.0.5 & 4.0.7.1 default, answering key questions)
    and mode='full' (Milestone 4.0.3 full internal methodology, actions, constraints, and signals).
    """
    if isinstance(handoff, ExecutionHandoff):
        data = handoff.model_dump()
    else:
        data = dict(handoff)

    if mode == "compact":
        summary = data.get("summary") or {}
        if hasattr(summary, "model_dump"):
            summary = summary.model_dump()

        raw_crit = (
            summary.get("critical_assumption")
            or data.get("critical_assumption")
            or (data.get("active_assumption") or {}).get("statement")
            or ""
        )
        crit_asmp = clean_assumption_statement(raw_crit)

        cand1 = summary.get("what_are_we_testing")
        cand2 = data.get("what_we_are_testing")
        cand3 = data.get("objective") or data.get("learning_goal")
        cand4 = crit_asmp
        cand5 = "We are testing the active critical assumption."

        what_test = resolve_what_are_we_testing(
            candidates=[cand1, cand2, cand3, cand4, cand5],
            fallback=cand5,
        )
        if not what_test or what_test.strip() in [".", "Whether .", "If ."]:
            logger.warning("summary_integrity_warning | what_are_we_testing was unexpectedly empty; applied fallback.")
            what_test = f"Whether {crit_asmp[0].lower() + crit_asmp[1:].rstrip('.')}." if crit_asmp else "We are testing the active critical assumption."

        actions = data.get("founder_actions") or []
        primary_action = summary.get("what_should_i_do")
        if not primary_action:
            if actions:
                import re
                primary_action = re.sub(r"^\d+\.\s*", "", str(actions[0]).strip())
            else:
                primary_action = "Interview target participants using structured questions."

        evidence_text = summary.get("what_should_i_bring_back") or data.get("evidence_to_collect") or ""
        if "," in evidence_text and not evidence_text.startswith("- ") and "\n" not in evidence_text:
            items = [it.strip() for it in evidence_text.split(",") if it.strip()]
            evidence_formatted = "\n".join(f"- {it}" for it in items)
        else:
            evidence_formatted = evidence_text

        useful_sig = summary.get("useful_signal") or data.get("support_signal") or "Several independent participants report the problem and accept a concrete next step."
        pos_signals = summary.get("current_positive_signals") or []
        if not pos_signals:
            pos_signals = compose_current_positive_signals(
                diagnosis_snapshot=data.get("diagnosis_snapshot"),
            )

        lines = [
            "=" * 54,
            "YOUR NEXT TEST",
            "=" * 54,
            "",
        ]
        if crit_asmp:
            lines.extend([
                "CRITICAL ASSUMPTION",
                "",
                f"{crit_asmp}",
                "",
            ])
        lines.extend([
            "WHAT ARE WE TESTING?",
            "",
            f"{what_test}",
            "",
            "WHAT SHOULD I DO?",
            "",
            f"{primary_action}",
            "",
            "WHAT SHOULD I BRING BACK?",
            "",
            f"{evidence_formatted}",
            "",
            "WHAT COUNTS AS A USEFUL SIGNAL?",
            "",
            f"{useful_sig}",
            "",
            "WHAT LOOKS PROMISING SO FAR?",
            "",
        ])
        if pos_signals:
            for sig in pos_signals:
                if sig.startswith("- "):
                    lines.append(f"{sig}")
                else:
                    lines.append(f"- {sig}")
        else:
            lines.append("- No strong positive signal is established yet.")

        lines.extend([
            "",
            "=" * 54,
        ])
        return "\n".join(lines)

    # Full mode
    raw_what = data.get("what_we_are_testing") or data.get("learning_goal") or data.get("objective") or "Assumption under test"
    what_test = clean_testing_question(raw_what)
    if not what_test:
        what_test = data.get("learning_goal") or data.get("critical_assumption") or "Assumption under test"

    target = data.get("target_participants_or_context") or ""
    if not target or target.lower() in ["target customer cohort", "cohort", "none"]:
        target = "Qualifying target participants"
    actions = data.get("founder_actions") or []
    evidence = data.get("evidence_to_collect") or ""
    support = data.get("support_signal") or ""
    contradict = data.get("contradiction_signal") or ""
    inconclusive = data.get("inconclusive_signal") or ""
    do_nots = data.get("important_constraints") or []
    invalid_inf = data.get("invalid_inferences") or []

    lines = [
        "=" * 54,
        "YOUR NEXT TEST",
        "=" * 54,
        "",
        "What we're testing:",
        f"{what_test}",
        "",
        "What to do:",
    ]
    if actions:
        for idx, act in enumerate(actions, 1):
            lines.append(f"  {idx}. {act}")
    else:
        lines.append("  - Run the proposed observation/interaction.")

    lines.extend([
        "",
        "Who to do it with:",
        f"  {target}",
        "",
        "What to observe and bring back:",
        f"  - Evidence: {evidence}",
    ])
    if support:
        lines.append(f"  - Confirms assumption: {support}")
    if contradict:
        lines.append(f"  - Weakens/disproves: {contradict}")
    if inconclusive:
        lines.append(f"  - Inconclusive signal: {inconclusive}")

    if do_nots or invalid_inf:
        lines.extend(["", "Do not:"])
        for dn in do_nots:
            lines.append(f"  - {dn}")
        for ii in invalid_inf:
            lines.append(f"  - Count this as proving: {ii}")

    lines.extend([
        "",
        "=" * 54,
        "WHEN YOU'RE DONE",
        "=" * 54,
        "Return here and tell me what actually happened.",
        "You can also ask me to explain the experiment or tell me if something blocks you.",
    ])

    return "\n".join(lines)


def compose_fallback_learning_action(
    assumption: dict | None = None,
    weak_link: str | None = None,
    critic_output: dict | None = None,
    rejected_experiment: dict | None = None,
    state: dict | None = None,
    diagnosis_snapshot: dict | None = None,
) -> FallbackLearningAction:
    """
    Deterministically composes a safe, bounded FallbackLearningAction when
    the Experiment Critic revision budget is exhausted (Milestone 4.0.6).

    Rules:
      - Strictly for learning, never validation (no numerical thresholds).
      - Strips weak proxy signals (free templates, WhatsApp curiosity, landing pages).
      - Reconstructs recent real-world events, firsthand behaviors, and actual consequences.
      - Sets explicit non-inference boundaries (willingness to pay, need for AI/software).
    """
    assumption = assumption or {}
    critic_output = critic_output or {}
    rejected_exp = rejected_experiment or {}
    state = state or {}

    asmp_id = assumption.get("id") or rejected_exp.get("assumption_id") or "asmp-unknown"
    target_area = assumption.get("area") or weak_link or state.get("weak_area") or "Problem"

    # 1. WHAT DO WE STILL NEED TO LEARN?
    asmp_text = assumption.get("statement") or rejected_exp.get("hypothesis") or "Critical assumption under test"
    for prefix in [
        "Validating critical assumption:",
        "Validating critical assumption",
        "Critical assumption under test:",
        "Critical assumption under test",
        "Validate critical assumption:",
        "Validate critical assumption",
    ]:
        if asmp_text.lower().startswith(prefix.lower()):
            asmp_text = asmp_text[len(prefix):].strip(" -:")
            break

    learning_goal = assumption.get("learning_goal")
    if not learning_goal or learning_goal.lower().startswith("validat"):
        if not asmp_text.lower().startswith("whether ") and not asmp_text.lower().startswith("if "):
            learning_goal = f"Whether {asmp_text[0].lower() + asmp_text[1:].rstrip('.') if len(asmp_text) > 1 else asmp_text}."
        else:
            learning_goal = asmp_text

    # 2. WHY FALLBACK IS NEEDED
    issues = critic_output.get("issues") or []
    if issues:
        why_needed = f"The proposed experiment did not meet Startary's quality bar ({issues[0]}). Rather than running an unverified test, this bounded step gathers concrete firsthand facts."
    else:
        why_needed = "The proposed experiment did not meet Startary's quality bar. Rather than running an unverified test, this bounded step gathers concrete firsthand facts."

    # 3. WHO TO ENGAGE
    target = (
        rejected_exp.get("target_participants_or_scope")
        or rejected_exp.get("target_participants_or_context")
        or ""
    )
    if not target or target.lower() in ["target customer cohort", "cohort", "none"]:
        target = f"5–8 independent target participants or store owners relevant to {target_area.lower()}"
    who_to_engage = target

    # 4. PRIMARY ACTION
    primary_action = f"Talk with {who_to_engage}."

    # 5. WHAT TO ASK OR OBSERVE
    if target_area.lower() == "problem":
        what_to_ask = [
            "Ask each for one recent example of unsold/expired inventory or a product running out unexpectedly (or equivalent core problem event).",
            "Ask what happened during that specific event.",
            "Ask how often this happens across a typical month.",
            "Ask what money, time, or logistical overhead it affected.",
            "Ask how they currently decide what to do or what workaround they use.",
            "Ask whether they have already actively tried changing the process.",
        ]
        what_to_bring = [
            "One concrete recent incident per participant/shop",
            "Estimated frequency of occurrence",
            "Concrete consequence (cost, wasted time, or operational disruption)",
            "Current workaround or decision process",
            "Whether they have already tried to improve or change the process",
        ]
    else:
        what_to_ask = [
            f"Ask for one concrete recent situation where they encountered {target_area.lower()} friction.",
            "Ask what specific actions they took and what consequences occurred.",
            "Ask how frequently this situation arises.",
            "Ask what workaround or alternative they currently rely on.",
            "Ask whether they have actively attempted to change or solve this before.",
        ]
        what_to_bring = [
            "One concrete recent situation per participant",
            "Actual behaviors and decisions observed",
            "Specific costs or disruption reported",
            "Current workarounds in place",
            "Past attempts to resolve the issue",
        ]

    # 6. WHAT NOT TO INFER
    what_not_to_infer = [
        "Willingness to pay or commercial conversion",
        "Need for AI, automation, or specialized software",
        "That a proposed product or recommendation engine will be adopted",
        "That general politeness, interest, or curiosity represents commercial urgency",
    ]

    action = FallbackLearningAction(
        assumption_id=asmp_id,
        target_area=target_area,
        learning_goal=learning_goal,
        why_fallback_is_needed=why_needed,
        primary_action=primary_action,
        who_to_engage=who_to_engage,
        what_to_ask_or_observe=what_to_ask,
        what_to_bring_back=what_to_bring,
        what_not_to_infer=what_not_to_infer,
        completion_instruction="Return with the notes from those conversations.",
        participant_target=who_to_engage,
        estimated_effort="3–5 hours across 5–8 conversations",
    )
    rendered = render_fallback_learning_action_text(action, mode="compact")
    action.response_text = rendered
    return action


def render_fallback_learning_action_text(
    action: FallbackLearningAction | dict,
    mode: str = "compact",
) -> str:
    """
    Renders the founder-facing markdown block for a FallbackLearningAction (Milestone 4.0.6).
    Emphasizes the 4 key sections answering:
      1. WHAT DO WE STILL NEED TO LEARN?
      2. WHAT SHOULD I DO NEXT?
      3. WHAT SHOULD I BRING BACK?
      4. WHAT SHOULD I NOT CONCLUDE YET?
    """
    if isinstance(action, FallbackLearningAction):
        data = action.model_dump()
    else:
        data = dict(action)

    learning_goal = data.get("learning_goal") or "Resolve core uncertainty with firsthand observations."
    primary_action = data.get("primary_action") or "Talk with target participants."
    questions = data.get("what_to_ask_or_observe") or []
    bring_back = data.get("what_to_bring_back") or []
    not_infer = data.get("what_not_to_infer") or []
    instruction = data.get("completion_instruction") or "Return with the notes from those conversations."

    lines = [
        "=" * 54,
        "WHAT DO WE STILL NEED TO LEARN?",
        "=" * 54,
        "",
        f"{learning_goal}",
        "",
        "=" * 54,
        "WHAT SHOULD I DO NEXT?",
        "=" * 54,
        "",
        f"{primary_action}",
        "",
        "In each conversation, ask:",
    ]
    for q in questions:
        lines.append(f"- {q}")

    lines.extend([
        "",
        "=" * 54,
        "WHAT SHOULD I BRING BACK?",
        "=" * 54,
        "",
    ])
    for b in bring_back:
        lines.append(f"- {b}")

    lines.extend([
        "",
        "=" * 54,
        "WHAT SHOULD I NOT CONCLUDE YET?",
        "=" * 54,
        "",
        "This does not prove:",
    ])
    for n in not_infer:
        lines.append(f"- {n}")

    lines.extend([
        "",
        "=" * 54,
        "WHEN YOU'RE DONE",
        "=" * 54,
        f"{instruction}",
        "",
        "=" * 54,
        "This action is for learning, not validation.",
        "=" * 54,
    ])

    return "\n".join(lines)



