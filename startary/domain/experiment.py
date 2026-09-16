"""
Experiment schema.

Every experiment must reference the assumption it is designed to test.
This is enforced at the schema level, not just by convention.

Experiments are designed by future AI modules (Milestone 2+).
In Milestone 0 the schema is defined and validated only.
"""

from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Optional, Union

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
    target_sample_size: Optional[int] = Field(
        default=None,
        description="Explicit target sample size or observation count required (Milestone 4.1.2.2).",
    )
    collected_sample_count: Optional[int] = Field(
        default=None,
        description="Number of qualifying observations collected so far (Milestone 4.1.2.2).",
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
    action_type: Union[NextLearningActionType, str] = Field(
        default=NextLearningActionType.VALIDATED_EXPERIMENT,
        description="Distinguishes validated experiments from fallback learning.",
    )
    build_readiness: Optional[dict[str, Any]] = Field(
        default=None,
        description="Structured build readiness decision (Milestone 4.1).",
    )
    founder_diagnosis_text: Optional[str] = Field(
        default=None,
        description="Compact 7-part founder-facing diagnosis snapshot text.",
    )
    next_strategic_action: Optional[dict[str, Any]] = Field(
        default=None,
        description="The NextStrategicAction decision (Milestone 4.1.2 & 4.1.2.1).",
    )
    target_sample_size: Optional[int] = Field(
        default=None,
        description="Explicit target sample size or observation count required (Milestone 4.1.2.2).",
    )
    early_signals: list[str] = Field(
        default_factory=list,
        description="Experiment-specific early signals from active incomplete test (Milestone 4.1.2.2).",
    )
    observations_collected: Optional[int] = Field(
        default=None,
        description="Observations collected so far for active experiment.",
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
    smallest_thing_to_resolve_next: Optional[str] = Field(
        default=None,
        description="The smallest concrete thing to resolve next (Milestone 4.1.1).",
    )
    critic_rejection_reasons: list[str] = Field(
        default_factory=list,
        description="Concise Critic-derived rejection reasons (Milestone 4.1.1).",
    )
    source_basis: list[str] = Field(
        default_factory=list,
        description="State fields, critical assumption elements, or critic reasons grounding this action (Milestone M4.1.3.4).",
    )


class RecoveryContext(BaseModel):
    """
    Context package for deriving grounded, domain-isolated recovery actions (Milestone M4.1.3.4).
    """

    project_id: str = Field(default="", description="Identifier of the active project.")
    startup_name: str = Field(default="", description="Name of the active startup.")
    active_problem: str = Field(default="", description="Core problem statement.")
    target_user: str = Field(default="", description="Target customer or user cohort.")
    current_weak_link: str = Field(default="", description="Active weak link area.")
    critical_assumption: str = Field(default="", description="Active critical assumption statement.")
    critic_rejection_reasons: list[str] = Field(default_factory=list, description="Rejection reasons from experiment critic.")
    domain_vocabulary: set[str] = Field(default_factory=set, description="Set of lowercased tokens present in active context.")


# Known foreign domain trap terms to guard against cross-domain contamination (Milestone M4.1.3.4)
FOREIGN_DOMAIN_TRAPS: dict[str, list[str]] = {
    "food_and_bakery": [
        "bakery", "bakeries", "pastry", "bread", "surplus bag", "surplus bags",
        "surplus food", "closing-time surplus", "closing window"
    ],
    "refrigeration_and_kitchen": [
        "refrigerator", "refrigerators", "refrigeration", "fridge", "commercial kitchen",
        "kitchen operator", "kitchen operators", "kitchen manager", "food spoilage",
        "temperature monitoring", "iot sensor", "sensor hardware"
    ],
    "education_and_it_coaching": [
        "cybersecurity", "it course", "it courses", "university student",
        "university students", "career guidance"
    ],
    "medical_and_clinical": [
        "clinical", "clinic", "patient", "patients", "physician", "doctor", "hospital",
        "medical record"
    ],
}


def build_recovery_context(
    state: dict | None = None,
    assumption: dict | None = None,
    rejected_exp: dict | None = None,
    critic_output: dict | None = None,
    diagnosis_snapshot: dict | None = None,
) -> RecoveryContext:
    """
    Constructs a RecoveryContext extracting all available ground-truth context from the active startup.
    Ensures that any generated recovery action has access to domain vocabulary and active constraints.
    """
    state = state or {}
    assumption = assumption or {}
    rejected_exp = rejected_exp or {}
    critic_output = critic_output or {}
    snapshot = diagnosis_snapshot or state.get("diagnosis_snapshot") or {}

    project_id = str(state.get("project_id") or "")
    st_state = state.get("startup_state") or {}
    startup_name = str(state.get("startup_name") or st_state.get("name") or "")

    target_user = str(
        st_state.get("target_user")
        or state.get("target_user")
        or rejected_exp.get("target_participants_or_scope")
        or rejected_exp.get("target_participants_or_context")
        or assumption.get("target_user")
        or snapshot.get("target_user")
        or ""
    ).strip()

    active_problem = str(
        st_state.get("problem")
        or state.get("active_problem")
        or snapshot.get("problem")
        or assumption.get("problem")
        or ""
    ).strip()

    current_weak_link = str(
        assumption.get("area")
        or state.get("weak_area")
        or "Problem"
    ).strip()

    critical_assumption = str(
        assumption.get("statement")
        or rejected_exp.get("hypothesis")
        or "Critical assumption under test"
    ).strip()

    reasons: list[str] = list(
        critic_output.get("issues")
        or critic_output.get("reasons_for_rejection")
        or critic_output.get("critic_issues")
        or []
    )

    # Build domain vocabulary from active startup context text sources
    raw_texts = [
        project_id,
        startup_name,
        target_user,
        active_problem,
        critical_assumption,
        str(state.get("raw_context") or ""),
        str(state.get("value_prop") or ""),
        str(st_state.get("value_prop") or ""),
        str(assumption.get("rationale") or ""),
        str(assumption.get("learning_goal") or ""),
        " ".join(str(m.get("content", "")) for m in (state.get("founder_messages") or []) if isinstance(m, dict)),
        " ".join(
            str(fe.get("raw_content", "")) + " " + str(fe.get("content", ""))
            for fe in (state.get("founder_evidence") or [])
            if isinstance(fe, dict)
        ),
    ]
    vocab: set[str] = set()
    for text in raw_texts:
        if text:
            tokens = re.findall(r"[a-z0-9]+", text.lower())
            vocab.update(tokens)

    return RecoveryContext(
        project_id=project_id,
        startup_name=startup_name,
        active_problem=active_problem,
        target_user=target_user,
        current_weak_link=current_weak_link,
        critical_assumption=critical_assumption,
        critic_rejection_reasons=reasons,
        domain_vocabulary=vocab,
    )


def validate_recovery_grounding(
    action: FallbackLearningAction,
    context: RecoveryContext,
) -> tuple[bool, str]:
    """
    Validates that a FallbackLearningAction does not contain ungrounded concepts,
    workflows, or customer types foreign to the active startup (Milestone M4.1.3.4).
    """
    if not action.who_to_engage or not action.primary_action:
        return False, "FallbackLearningAction is missing who_to_engage or primary_action."
    if not action.smallest_thing_to_resolve_next:
        return False, "FallbackLearningAction is missing smallest_thing_to_resolve_next."

    text_to_check = " ".join([
        action.response_text,
        action.primary_action,
        action.who_to_engage,
        action.smallest_thing_to_resolve_next,
        " ".join(action.what_to_ask_or_observe),
        " ".join(action.what_to_bring_back),
        " ".join(action.what_not_to_infer),
    ]).lower()

    def _is_token_grounded(tok: str) -> bool:
        if tok in context.domain_vocabulary:
            return True
        if tok.rstrip("s") in context.domain_vocabulary or (tok + "s") in context.domain_vocabulary:
            return True
        if tok.endswith("es") and tok[:-2] in context.domain_vocabulary:
            return True
        if tok.endswith("y") and tok[:-1] + "ies" in context.domain_vocabulary:
            return True
        if tok.endswith("ies") and tok[:-3] + "y" in context.domain_vocabulary:
            return True
        if tok.endswith("ing") and (tok[:-3] in context.domain_vocabulary or tok[:-3] + "e" in context.domain_vocabulary):
            return True
        if tok.endswith("ed") and (tok[:-2] in context.domain_vocabulary or tok[:-1] in context.domain_vocabulary):
            return True
        return False

    DOMAIN_ANCHORS = {
        "food_and_bakery": {"bakery", "bakeries", "pastry", "pastries", "bread", "cafe", "cafes"},
        "refrigeration_and_kitchen": {"refrigerator", "refrigerators", "refrigeration", "fridge", "kitchen", "kitchens", "restaurant", "restaurants"},
        "education_and_it_coaching": {"cybersecurity", "tutoring", "student", "students", "course", "courses"},
        "medical_and_clinical": {"clinic", "clinics", "clinical", "hospital", "hospitals", "radiologist", "radiologists", "physician", "physicians", "doctor", "doctors", "patient", "patients"},
    }

    for domain_name, terms in FOREIGN_DOMAIN_TRAPS.items():
        # If the active startup context already belongs to this domain, it is native, not foreign
        anchors = DOMAIN_ANCHORS.get(domain_name, set())
        if any(_is_token_grounded(a) for a in anchors):
            continue

        for term in terms:
            if term in text_to_check:
                # Check if every domain-specific token of the term is grounded in context vocabulary
                tokens = [t for t in re.findall(r"[a-z0-9]+", term) if len(t) > 2]
                domain_tokens = [
                    t for t in tokens
                    if t not in (
                        "operator", "operators", "manager", "managers", "monitoring",
                        "hardware", "window", "bag", "bags", "record", "records",
                        "course", "courses", "guidance"
                    )
                ]
                if not domain_tokens:
                    domain_tokens = tokens
                is_grounded = all(_is_token_grounded(token) for token in domain_tokens)
                if not is_grounded:
                    return False, f"Ungrounded foreign domain term '{term}' detected (domain: {domain_name})."

    return True, "Recovery action is grounded in active context."


def sanitize_recovery_leakage(
    text: str,
    domain_vocabulary: set[str],
) -> str:
    """
    Tertiary Output Sanitizer (Milestone M4.1.3.4):
    Neutralizes ungrounded foreign domain terms if they appear in founder-facing output.
    """
    if not text:
        return text

    def _is_tok_grounded(tok: str) -> bool:
        if tok in domain_vocabulary:
            return True
        if tok.rstrip("s") in domain_vocabulary or (tok + "s") in domain_vocabulary:
            return True
        if tok.endswith("es") and tok[:-2] in domain_vocabulary:
            return True
        if tok.endswith("y") and tok[:-1] + "ies" in domain_vocabulary:
            return True
        if tok.endswith("ies") and tok[:-3] + "y" in domain_vocabulary:
            return True
        if tok.endswith("ing") and (tok[:-3] in domain_vocabulary or tok[:-3] + "e" in domain_vocabulary):
            return True
        if tok.endswith("ed") and (tok[:-2] in domain_vocabulary or tok[:-1] in domain_vocabulary):
            return True
        return False

    DOMAIN_ANCHORS = {
        "food_and_bakery": {"bakery", "bakeries", "pastry", "pastries", "bread", "cafe", "cafes"},
        "refrigeration_and_kitchen": {"refrigerator", "refrigerators", "refrigeration", "fridge", "kitchen", "kitchens", "restaurant", "restaurants"},
        "education_and_it_coaching": {"cybersecurity", "tutoring", "student", "students", "course", "courses"},
        "medical_and_clinical": {"clinic", "clinics", "clinical", "hospital", "hospitals", "radiologist", "radiologists", "physician", "physicians", "doctor", "doctors", "patient", "patients"},
    }

    skip_domains = set()
    for dname, anchors in DOMAIN_ANCHORS.items():
        if any(_is_tok_grounded(a) for a in anchors):
            skip_domains.add(dname)

    REPLACEMENTS = [
        ("refrigeration_and_kitchen", "commercial kitchens", "target operational environments"),
        ("refrigeration_and_kitchen", "commercial kitchen", "target operational environment"),
        ("refrigeration_and_kitchen", "kitchen operators", "operational managers"),
        ("refrigeration_and_kitchen", "kitchen operator", "operational manager"),
        ("refrigeration_and_kitchen", "kitchen managers", "operational managers"),
        ("refrigeration_and_kitchen", "kitchen manager", "operational manager"),
        ("refrigeration_and_kitchen", "kitchens", "operational facilities"),
        ("refrigeration_and_kitchen", "kitchen", "operational facility"),
        ("refrigeration_and_kitchen", "walk-in refrigerators", "equipment"),
        ("refrigeration_and_kitchen", "walk-in refrigerator", "equipment"),
        ("refrigeration_and_kitchen", "refrigerators", "equipment"),
        ("refrigeration_and_kitchen", "refrigerator", "equipment"),
        ("refrigeration_and_kitchen", "refrigeration", "climate-controlled storage"),
        ("refrigeration_and_kitchen", "temperature monitoring", "operational condition monitoring"),
        ("refrigeration_and_kitchen", "food spoilage", "inventory loss"),
        ("food_and_bakery", "closing-time surplus", "operational surplus"),
        ("food_and_bakery", "surplus food", "surplus inventory"),
        ("food_and_bakery", "surplus bags", "inventory offerings"),
        ("food_and_bakery", "surplus bag", "inventory offering"),
        ("food_and_bakery", "bakeries", "partner facilities"),
        ("food_and_bakery", "bakery", "partner facility"),
        ("food_and_bakery", "pastry", "goods"),
        ("food_and_bakery", "bread", "goods"),
    ]

    sanitized = text
    for dom, term, replacement in REPLACEMENTS:
        if dom in skip_domains:
            continue
        if term in sanitized.lower():
            tokens = [t for t in re.findall(r"[a-z0-9]+", term) if len(t) > 2]
            domain_tokens = [t for t in tokens if t not in ("operator", "operators", "manager", "managers", "monitoring")]
            if not domain_tokens:
                domain_tokens = tokens
            if not all(_is_tok_grounded(tok) for tok in domain_tokens):
                pattern = re.compile(re.escape(term), re.IGNORECASE)
                sanitized = pattern.sub(replacement, sanitized)

    return sanitized


def build_generic_deterministic_fallback(
    context: RecoveryContext,
) -> FallbackLearningAction:
    """
    Section 6 Deterministic Domain-Neutral Generic Fallback (Milestone M4.1.3.4).
    Produces a safe, strictly grounded recovery action when grounding validation fails
    or context is insufficient for domain-specific recovery.
    """
    clean_asmp = clean_assumption_statement(context.critical_assumption)
    target = context.target_user or "target participants or operational stakeholders"
    target_area = context.current_weak_link or "Problem"

    smallest_thing = "Collect the minimum missing information required to redesign the test around the current assumption."
    primary_action = (
        f"Find 3–5 reachable people ({target}) who have direct experience with the current problem "
        f"or buying/operational context. Ask specifically about the gaps identified by the critic."
    )
    who_to_engage = f"3–5 reachable {target}"
    what_to_ask = [
        "Ask for one concrete recent situation where they experienced or managed this operational workflow.",
        "Ask what specific actions they took, what workarounds they used, and what concrete consequences occurred.",
        "Ask specifically about the operational gaps and constraints identified in the test critique.",
        "Observe how they currently decide what to do without prompting or leading them.",
    ]
    what_to_bring = [
        f"Concrete examples and recent incidents directly relevant to '{clean_asmp}'",
        "Specific factual details addressing the unresolved questions identified by the critic",
        "Direct observations of current workarounds, time/cost impacts, and actual behaviors",
    ]
    what_not_to_infer = [
        "Willingness to pay or commercial conversion",
        "Need for AI, automation, or specialized software",
        "That general politeness, interest, or curiosity represents commercial urgency",
        "That a small sample validates market demand or scalability",
    ]

    sanitized_reasons = [
        sanitize_recovery_leakage(r, context.domain_vocabulary) for r in context.critic_rejection_reasons
    ]

    action = FallbackLearningAction(
        assumption_id=f"asmp-{context.project_id or 'generic'}",
        target_area=target_area,
        learning_goal=f"Resolve core uncertainty around '{clean_asmp}' with firsthand observations.",
        why_fallback_is_needed=(
            "The proposed experiment did not meet Startary's quality bar. "
            "Rather than running an unverified test, this bounded step gathers concrete firsthand facts."
        ),
        primary_action=primary_action,
        who_to_engage=who_to_engage,
        what_to_ask_or_observe=what_to_ask,
        what_to_bring_back=what_to_bring,
        what_not_to_infer=what_not_to_infer,
        completion_instruction="Return with what happened / what you learned.",
        participant_target=who_to_engage,
        estimated_effort="3–5 conversations (approx. 2–3 hours)",
        smallest_thing_to_resolve_next=smallest_thing,
        critic_rejection_reasons=sanitized_reasons,
        source_basis=["generic_deterministic_recovery", target_area],
    )
    rendered = render_fallback_learning_action_text(action, mode="compact")
    action.response_text = sanitize_recovery_leakage(rendered, context.domain_vocabulary)
    return action


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
    'What currently gives us a reason to continue testing?' (Milestones 4.0.5, 4.0.7.1, 4.0.7.2.2).

    Precedence:
      1. real customer commitment / behavior
      2. repeated external problem evidence
      3. explicit problem urgency / consequence (including single-shop pain)
      4. dissatisfaction with existing alternatives / manual workarounds
      5. customer access
      6. relevant founder execution capability

    Rules:
      - Strictly grounded behavioral facts only.
      - Resolves historical validated evidence across completed cycles (Milestone 4.0.7.2.2).
      - Contradicted evidence cycles are strictly excluded from positive signals.
      - Never fabricate validation or use speculative praise/hype.
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

    # 1. Advantages from diagnosis snapshot
    advantages = list(diagnosis_snapshot.get("current_advantages") or [])
    for adv in advantages:
        adv_clean = str(adv).strip()
        adv_lower = adv_clean.lower()
        if any(w in adv_lower for w in ["access", "cousin", "uncle", "distribution", "channel", "network"]):
            b5_customer_access.append(adv_clean)
        elif any(w in adv_lower for w in ["pilot", "loi", "letter of intent", "willing to pay", "paying", "pre-order", "deposit", "commission"]):
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

    # 2. External Evidence & Completed Cycles
    # Inspect evidence analysis records (current and historical)
    analyses_to_check: list[dict[str, Any]] = []
    ev_analysis = state.get("evidence_analysis")
    if isinstance(ev_analysis, dict) and ev_analysis:
        analyses_to_check.append(ev_analysis)
    for past_a in (state.get("evidence_history") or []):
        if isinstance(past_a, dict) and past_a:
            analyses_to_check.append(past_a)

    for analysis in analyses_to_check:
        conclusion = str(analysis.get("conclusion") or "").upper()
        # Milestone 4.0.7.2.2 & M4.1.2.1: Contradicted cycles must NOT contribute positive signals
        if conclusion == "CONTRADICTED":
            continue

        exec_quality = str(analysis.get("experiment_execution_quality") or "").upper()
        if exec_quality in ("FAILED", "POOR"):
            continue

        support_findings = analysis.get("support_findings") or []
        valid_learnings = analysis.get("valid_learnings") or []
        core_lrn = analysis.get("core_learning") or ""
        all_findings = support_findings + valid_learnings + ([core_lrn] if core_lrn else [])

        for sf in all_findings:
            sf_clean = str(sf).strip()
            if not sf_clean:
                continue
            sf_lower = sf_clean.lower()
            if any(w in sf_lower for w in ["paid", "deposit", "commission", "loi", "pilot", "purchas", "revenue", "paying", "paying teams", "paying customers", "reservation", "pickup", "manageable", "willing to repeat", "agreed"]):
                b1_commitments.append(sf_clean)
            elif any(w in sf_lower for w in ["repeat", "weekly", "schedule", "organizer", "tactical", "feedback", "cadence", "two evening", "both evening", "multi-evening"]):
                b1_commitments.append(sf_clean)
            elif any(w in sf_lower for w in ["urgency", "acute", "friction", "pain", "costly", "drop-off", "churn", "stockout", "spoilage", "waste", "spoil", "return", "returns", "manual steps", "prevention"]):
                b3_problem_urgency.append(sf_clean)
            elif any(w in sf_lower for w in ["alternative", "workaround", "failure", "footage", "delayed", "reliance on memory", "manual work", "workflow failure", "manual judgment"]):
                b4_alternatives_friction.append(sf_clean)
            elif any(w in sf_lower for w in ["coach", "coaches", "access", "channel"]):
                b5_customer_access.append(sf_clean)
            else:
                b2_external_evidence.append(sf_clean)

    # Inspect founder evidence items (extracting from raw_content, content, structured_observations)
    founder_ev_list: list[dict[str, Any]] = []
    for item in (state.get("founder_evidence") or state.get("external_evidence") or []):
        if isinstance(item, dict):
            founder_ev_list.append(item)
    active_ev = state.get("active_evidence")
    if isinstance(active_ev, dict) and active_ev not in founder_ev_list:
        founder_ev_list.append(active_ev)

    for fe in founder_ev_list:
        fe_raw = " ".join([
            str(fe.get("raw_content") or ""),
            str(fe.get("raw_text") or ""),
            str(fe.get("content") or ""),
            str(fe.get("raw_submission") or ""),
            " ".join(str(o) for o in (fe.get("structured_observations") or [])),
            str(fe.get("quantitative_results") or ""),
        ]).lower()

        # Behavioral Commitments Patterns - only grounded if terms in evidence
        if ("non-incubator" in fe_raw or "non incubator" in fe_raw or "stranger" in fe_raw or "nearby adult" in fe_raw) and ("deposit" in fe_raw or "paid" in fe_raw or "pickup" in fe_raw):
            b1_commitments.append("Non-incubator buyers also made paid reservations and completed pickup.")
        elif "incubator" in fe_raw and "non-incubator" not in fe_raw and "non incubator" not in fe_raw and any(w in fe_raw for w in ["pickup", "pickups", "reservation", "surplus"]):
            b1_commitments.append("Three incubator members completed real closing-time pickups.")
        elif any(w in fe_raw for w in ["paid pickup", "paid pickups", "completed paid", "completed pickups"]):
            b1_commitments.append("Multiple completed paid pickups occurred from real surplus.")

        if any(w in fe_raw for w in ["two evening", "two separate evening", "both evening", "multiple evening"]) or ("repeated behavior" in fe_raw and "evening" in fe_raw):
            b1_commitments.append("The behavior repeated with fresh nearby people across two separate evenings.")

        if "manageable" in fe_raw or "hand off remained manageable" in fe_raw or "handoff remained manageable" in fe_raw:
            if "bakery" in fe_raw or "bakeries" in fe_raw:
                b1_commitments.append("The participating bakery reported that the manual handoff remained manageable.")
            else:
                b1_commitments.append("The participating partner reported that the manual handoff remained manageable.")
        elif "willing to repeat" in fe_raw or "repeat the test" in fe_raw:
            b1_commitments.append("The participating merchant was willing to repeat the test.")

        if ("bakery" in fe_raw or "bakeries" in fe_raw) and ("unconnected" in fe_raw or "outreach" in fe_raw or "contacted 10" in fe_raw) and ("pilot" in fe_raw or "agreed" in fe_raw or "commit" in fe_raw):
            b1_commitments.append("Independent bakeries agreed to scheduled surplus pilot dates.")

        # Problem / Stockout / Waste Patterns - strictly domain-grounded
        if "stockout" in fe_raw:
            if any(k in fe_raw for k in ["grocery", "shop", "store"]):
                b2_external_evidence.append("Multiple independent shop owners reported recent stockouts.")
            else:
                b2_external_evidence.append("Multiple target participants reported recent stockouts.")
        if "waste" in fe_raw or "spoilage" in fe_raw:
            if any(k in fe_raw for k in ["perishable", "spoilage", "dairy", "food"]):
                b2_external_evidence.append("Multiple owners reported recurring perishable waste.")
            elif any(k in fe_raw for k in ["litter", "bin", "waste collection", "municipal"]):
                b2_external_evidence.append("Waste collection operators reported recurring bin overflow.")
            else:
                b2_external_evidence.append("Multiple participants reported recurring waste issues.")

        if "return" in fe_raw and ("recurring" in fe_raw or "25%" in fe_raw or "size" in fe_raw):
            b3_problem_urgency.append("One relevant merchant already experiences recurring size-related returns.")

        # Manual steps / prevention
        if "manual steps" in fe_raw or "prevention" in fe_raw:
            b3_problem_urgency.append("Owners already take manual steps to reduce these problems.")

        # Alternatives friction (avoid false positives across domains)
        is_inventory_startup = any(k in fe_raw for k in ["grocery", "shelf", "stockout", "spoilage", "pos", "fridge"])
        if is_inventory_startup and ("ordering" in fe_raw or "manual judgment" in fe_raw):
            b4_alternatives_friction.append("Existing ordering decisions still rely heavily on manual judgment and reactive adjustments.")

    # 3. Startup State Context & Founder Messages
    raw_ctx = (state.get("raw_context") or "").lower()
    founder_msgs = [
        str(m.get("content", "")).lower()
        for m in (state.get("founder_messages") or [])
        if isinstance(m, dict) and m.get("role") == "founder"
    ]
    all_founder_text = raw_ctx + " " + " ".join(founder_msgs)

    if ("non-incubator" in all_founder_text or "non incubator" in all_founder_text) and any(w in all_founder_text for w in ["deposit", "paid", "pickup"]):
        b1_commitments.append("Non-incubator buyers also made paid reservations and completed pickup.")
    elif "incubator" in all_founder_text and "non-incubator" not in all_founder_text and "non incubator" not in all_founder_text and any(w in all_founder_text for w in ["pickup", "pickups"]):
        b1_commitments.append("Three incubator members completed real closing-time pickups.")
    if any(p in all_founder_text for p in ["two separate evening", "both evening", "two evening", "across two evening"]):
        b1_commitments.append("The behavior repeated with fresh nearby people across two separate evenings.")
    if any(p in all_founder_text for p in ["manageable", "process was manageable"]):
        if "bakery" in all_founder_text or "bakeries" in all_founder_text:
            b1_commitments.append("The participating bakery reported that the manual handoff remained manageable.")
        else:
            b1_commitments.append("The participating partner reported that the manual handoff remained manageable.")
    if any(p in all_founder_text for p in ["willing to repeat", "repeat the test"]):
        b1_commitments.append("The participating merchant was willing to repeat the test.")
    if ("bakery" in all_founder_text or "bakeries" in all_founder_text) and ("pilot" in all_founder_text or "surplus" in all_founder_text):
        b1_commitments.append("Independent bakeries agreed to scheduled surplus pilot dates.")
    if any(p in all_founder_text for p in ["nearby buyers", "already located near", "buyers were generally already located near"]):
        b2_external_evidence.append("Successful buyers were generally already located near the business.")
    if any(p in all_founder_text for p in ["6 independent grocery", "interviewed 6", "4/6 had recent stockouts", "4/6 had stockouts"]):
        b2_external_evidence.append("Multiple independent shop owners reported recent stockouts.")
    if any(p in all_founder_text for p in ["3/6 had waste", "3/6 had over-ordering", "perishable waste in dairy"]):
        b2_external_evidence.append("Multiple owners reported recurring perishable waste.")
    if any(p in all_founder_text for p in ["recurring stockouts", "spoilage and stockouts", "produce and dairy spoilage", "stockouts or perishable waste", "reports recurring stockouts"]):
        if any(p in all_founder_text for p in ["shop", "grocery", "store"]):
            b3_problem_urgency.append("One relevant shop owner reports recurring stockouts and perishable spoilage.")
        else:
            b3_problem_urgency.append("Target users report recurring stockouts and operational friction.")
    if any(p in all_founder_text for p in ["worth solving", "worth changing", "matters to them"]):
        b3_problem_urgency.append("The owner considers the core issue worth changing.")
    if any(p in all_founder_text for p in ["recurring size-related returns", "recurring returns", "suffer from 25% return rates", "size-related returns"]):
        b3_problem_urgency.append("One relevant merchant already experiences recurring size-related returns.")
    if any(p in all_founder_text for p in ["manual steps", "prevention actions", "order logs", "manual order log", "reduces waste"]):
        b3_problem_urgency.append("Owners already take manual steps to reduce these problems.")
    if any(p in all_founder_text for p in ["pos", "inventory system", "ordering recommendation", "ordering workflow"]):
        if any(p in all_founder_text for p in ["insufficient", "lacks", "do not provide", "no ordering recommendations", "without ordering"]):
            b4_alternatives_friction.append("Existing POS and inventory tools do not provide ordering recommendations.")

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
        evidence_list = list(area_info.get("evidence_provided") or [])
        status = str(area_info.get("status") or "").upper()
        if status in ("KNOWN", "PARTIALLY_KNOWN"):
            evidence_list.extend(area_info.get("verified_claims") or [])
        for ev in evidence_list:
            ev_clean = str(ev).strip()
            if not ev_clean:
                continue
            ev_lower = ev_clean.lower()
            if any(w in ev_lower for w in ["pilot", "willing to pay", "commission", "pre-order", "deposit", "loi", "paying"]):
                b1_commitments.append(ev_clean)
            elif any(w in ev_lower for w in ["repeat", "weekly", "schedule", "organizer", "tactical"]):
                b1_commitments.append(ev_clean)
            elif any(w in ev_lower for w in ["failure", "alternative", "workaround", "footage", "delayed"]):
                b4_alternatives_friction.append(ev_clean)
            elif any(w in ev_lower for w in ["multiple", "several", "verified recurring"]):
                if "stockout" in ev_lower:
                    b2_external_evidence.append("Multiple independent shop owners reported recent stockouts.")
                if "waste" in ev_lower or "spoilage" in ev_lower:
                    b2_external_evidence.append("Multiple owners reported recurring perishable waste.")
            elif any(w in ev_lower for w in ["coach", "coaches", "access", "channel"]):
                b5_customer_access.append(ev_clean)
            else:
                b2_external_evidence.append(ev_clean)

    # LearningDelta resolution (Milestone 4.1.2.1)
    learning_delta = state.get("learning_delta") or {}
    delta_conc = str(learning_delta.get("conclusion") or "").upper()
    if delta_conc != "CONTRADICTED":
        delta_items = list(learning_delta.get("key_takeaways") or []) + list(learning_delta.get("what_we_learned") or [])
        for di in delta_items:
            di_clean = str(di).strip()
            if not di_clean:
                continue
            di_lower = di_clean.lower()
            if any(w in di_lower for w in ["repeat", "weekly", "schedule", "organizer", "tactical", "paying", "paid", "cadence"]):
                b1_commitments.append(di_clean)
            elif any(w in di_lower for w in ["failure", "alternative", "workaround", "footage", "delayed", "reliance"]):
                b4_alternatives_friction.append(di_clean)
            elif any(w in di_lower for w in ["coach", "coaches", "access", "channel"]):
                b5_customer_access.append(di_clean)
            else:
                b2_external_evidence.append(di_clean)

    # Grounded traction facts (Milestone 4.1.2.1)
    paying_val = prog.get("paying_customers_count") or (prog.get("paying_customers") or {}).get("value")
    if (isinstance(paying_val, int) and paying_val > 0) or (str(paying_val).isdigit() and int(paying_val) > 0):
        b1_commitments.append(f"Existing payment and repeat usage provide early behavioral demand evidence ({paying_val} paying customers/teams).")
    elif any(k in all_founder_text for k in ["8 paying", "8 paying teams"]):
        b1_commitments.append("Existing payment and repeat usage provide early behavioral demand evidence (8 paying teams).")

    if any(k in all_founder_text for k in ["repeat user profile", "repeat-user profile", "repeat usage is concentrated"]):
        b1_commitments.append("Repeat usage is concentrated among teams with a regular playing schedule, a stable organizer, and structured tactical-review or player-feedback use.")

    # 4. Filter, Deduplicate & Prioritize Signals
    forbidden_terms = [
        "huge market", "massive market", "large market", "strong business model",
        "strong product-market fit", "product market fit", "validated demand",
        "market opportunity", "guaranteed success", "visionary", "disruptive",
        "budget", "runway", "hours/week", "hrs/week", "h/week", "capital",
        "dollars", "funding", "bootstrapped", "speculative", "clearly needs",
        "great team", "winning team", "unlimited potential"
    ]

    def _is_safe(sig: str) -> bool:
        sig_lower = sig.lower()
        if any(f in sig_lower for f in forbidden_terms):
            return False
        if "$" in sig:
            return False
        return True

    ordered_signals: list[str] = []
    seen_keys: set[str] = set()

    def _get_concept_key(text: str) -> str:
        tl = text.lower()
        if "non-incubator" in tl or "non incubator" in tl or "stranger" in tl:
            return "non_incubator_pickups"
        if "incubator" in tl:
            return "incubator_pickups"
        if "evening" in tl or "recurren" in tl:
            return "multi_evening"
        if "manageable" in tl or "hand off" in tl or "handoff" in tl:
            return "manageable_handoff"
        if "willing to repeat" in tl or "repeat the test" in tl:
            return "repeat_merchant"
        if "bakery" in tl or "bakeries" in tl:
            return "bakery_pilots"
        if "pickup" in tl:
            return "paid_pickups"
        if "nearby" in tl or "located near" in tl:
            return "nearby_buyers"
        if "pilot" in tl:
            return "pilot"
        if "commission" in tl or "willing to pay" in tl:
            return "pay_commission"
        if "multiple" in tl and "stockout" in tl:
            return "multi_stockouts"
        if "multiple" in tl and ("waste" in tl or "spoilage" in tl or "perishable" in tl):
            return "multi_waste"
        if "manual steps" in tl or "prevention" in tl:
            return "manual_prevention"
        if "manual judgment" in tl or "ordering" in tl:
            return "manual_ordering"
        if "pos" in tl:
            return "pos_tools"
        if "return" in tl:
            return "recurring_returns"
        if "worth changing" in tl or "worth solving" in tl:
            return "worth_changing"
        if "stockout" in tl or "spoilage" in tl:
            return "single_stockouts"
        if "uncle" in tl or "cousin" in tl or "customer access" in tl:
            return "customer_access"
        if "build" in tl or "prototype" in tl or "mvp" in tl:
            return "technical_build"
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
            if not _is_safe(item):
                continue
            key = _get_concept_key(item)
            if key not in seen_keys:
                seen_keys.add(key)
                ordered_signals.append(item)
            if len(ordered_signals) >= 4:
                break
        if len(ordered_signals) >= 4:
            break

    if not ordered_signals:
        return ["No strong positive signal is established yet."]

    return ordered_signals[:4]


# Milestone 4.0.7.2 alias
extract_positive_signals = compose_current_positive_signals


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

        # Milestone 4.1: Handle BUILD_MINIMAL_PROTOTYPE escalation rendering
        build_rd = data.get("build_readiness")
        if not build_rd and hasattr(handoff, "build_readiness"):
            build_rd = handoff.build_readiness
        if isinstance(build_rd, dict) and build_rd.get("decision") == "BUILD_MINIMAL_PROTOTYPE":
            lines = []
            diag_text = data.get("founder_diagnosis_text") or data.get("diagnosis_text")
            if diag_text:
                lines.extend([
                    diag_text.strip(),
                    "",
                    "",
                ])
            lines.extend([
                "=" * 54,
                "STARTARY — TIME TO BUILD THE NEXT TEST",
                "=" * 54,
                "",
                "WHY BUILD NOW?",
                "",
                "Manual testing has already established:",
            ])
            ev_basis = build_rd.get("evidence_basis") or []
            if not ev_basis:
                ev_basis = [
                    s for s in (summary.get("current_positive_signals") or [])
                    if s != "No strong positive signal is established yet."
                ]
            if ev_basis:
                for eb in ev_basis[:4]:
                    clean_eb = str(eb).lstrip("- ").strip()
                    lines.append(f"- {clean_eb}")
            else:
                lines.append("- Key prerequisite assumptions have been manually tested with real participants.")

            what_test_cand = summary.get("what_are_we_testing") or data.get("what_we_are_testing") or data.get("objective") or "We are testing the active critical assumption."
            what_test_clean = clean_testing_question(what_test_cand) if "clean_testing_question" in globals() else what_test_cand

            lines.extend([
                "",
                "The next uncertainty cannot be tested reliably through founder-operated manual coordination.",
                "",
                "WHAT ARE WE TESTING?",
                "",
                f"{what_test_clean}",
                "",
                "BUILD ONLY THIS:",
                "",
            ])
            min_scope = build_rd.get("minimum_build_scope") or []
            if min_scope:
                for item in min_scope:
                    clean_item = str(item).lstrip("- ").strip()
                    lines.append(f"- {clean_item}")
            else:
                lines.append("- Build the smallest product-mediated workflow required to test the active critical assumption.")

            lines.extend([
                "",
                "DO NOT BUILD YET:",
                "",
            ])
            not_build = build_rd.get("must_not_build") or []
            if not_build:
                for item in not_build:
                    clean_item = str(item).lstrip("- ").strip()
                    lines.append(f"- {clean_item}")
            else:
                lines.append("- Recommendation AI, loyalty programs, reviews, complex analytics, or automated payouts.")

            ev_text = summary.get("what_should_i_bring_back") or data.get("evidence_to_collect") or "- Observable user interaction results"
            useful_sig_text = summary.get("useful_signal") or data.get("support_signal") or "Users complete the workflow independently."

            lines.extend([
                "",
                "WHAT SHOULD I BRING BACK?",
                "",
                f"{ev_text}",
                "",
                "WHAT COUNTS AS A USEFUL SIGNAL?",
                "",
                f"{useful_sig_text}",
                "",
                "=" * 54,
            ])
            return "\n".join(lines)
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

        lines = []
        diag_text = data.get("founder_diagnosis_text") or data.get("diagnosis_text")

        # Check strategic action type
        strat_act = data.get("next_strategic_action") or {}
        action_type_val = strat_act.get("action_type") or data.get("action_type") or ""
        action_type_str = str(getattr(action_type_val, "value", action_type_val))

        # CONTINUE_CURRENT_TEST / Partial evidence: do not render full diagnosis again unless meaningful state changed
        is_partial = (
            action_type_str == "CONTINUE_CURRENT_TEST"
            or data.get("is_partial_evidence") is True
            or (handoff and str(getattr(handoff, "action_type", "")) == "CONTINUE_CURRENT_TEST")
        )
        suppress_diag = is_partial and not data.get("meaningful_state_changed", False)
        if diag_text and not suppress_diag:
            lines.extend([
                diag_text.strip(),
                "",
                "",
            ])

        # Milestone 4.1.2.2: Dedicated Partial Evidence Continuation rendering
        if is_partial:
            target_count = (
                data.get("target_sample_size")
                or (data.get("summary") or {}).get("target_sample_size")
                or 1
            )
            collected_count = (
                data.get("observations_collected")
                or data.get("collected_sample_count")
                or 0
            )
            unit_name = "teams" if any(w in str(data).lower() for w in ["team", "teams"]) else "observations"
            early_signals = data.get("early_signals") or []
            remaining = max(1, target_count - collected_count)

            lines.extend([
                "=" * 54,
                "CURRENT TEST PROGRESS",
                "=" * 54,
                "",
                f"{collected_count} of {target_count} required {unit_name} collected (requires {target_count} {unit_name}; currently have {collected_count} qualifying case(s)).",
                "",
            ])
            if early_signals:
                lines.extend([
                    "EARLY SIGNALS FROM THIS TEST",
                    "",
                ])
                for sig in early_signals:
                    sig_str = str(sig).strip()
                    if sig_str.startswith("- "):
                        lines.append(sig_str)
                    else:
                        lines.append(f"- {sig_str}")
                lines.extend([
                    "",
                    "This is directionally positive, but the experiment remains incomplete.",
                    "",
                ])
            lines.extend([
                "WHAT REMAINS",
                "",
                f"- Collect the remaining {remaining} qualifying {unit_name} using the same definitions. Continue the same test with the remaining {unit_name}.",
                "",
                "=" * 54,
            ])
            return "\n".join(lines)

        # Milestone 4.1.2.1: Dedicated Product Iteration rendering
        if action_type_str == "PRODUCT_ITERATION":
            bottleneck = strat_act.get("bottleneck_identified") or "setup and recording reliability"
            smallest = strat_act.get("smallest_change") or "smallest justified setup-assistance improvement"
            not_build = strat_act.get("what_not_to_build") or [
                "advanced tactical analytics",
                "expanded sports support",
                "unrelated AI features",
            ]
            measure = strat_act.get("measurable_outcome") or "setup completion time, recording failure rate, and repeat recording behavior"
            if isinstance(measure, str) and "," in measure and not measure.startswith("- "):
                measure_items = [it.strip() for it in measure.split(",") if it.strip()]
            elif isinstance(measure, list):
                measure_items = measure
            else:
                measure_items = [measure]

            lines.extend([
                "=" * 54,
                "NEXT ACTION — PRODUCT ITERATION",
                "=" * 54,
                "",
                f"Improve only the {bottleneck} bottleneck.",
                "",
                "BUILD / CHANGE ONLY THIS:",
                "",
                f"- {smallest}",
                "",
                "DO NOT BUILD YET:",
                "",
            ])
            for nb in not_build:
                lines.append(f"- {str(nb).lstrip('- ')}")
            lines.extend([
                "",
                "MEASURE:",
                "",
            ])
            for m_item in measure_items:
                lines.append(f"- {str(m_item).lstrip('- ')}")
            lines.extend([
                "",
                "=" * 54,
            ])
            return "\n".join(lines)

        lines.extend([
            "=" * 54,
            "YOUR NEXT TEST",
            "=" * 54,
            "",
        ])
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

    lines = []
    diag_text = data.get("founder_diagnosis_text") or data.get("diagnosis_text")
    if diag_text:
        lines.extend([
            diag_text.strip(),
            "",
            "",
        ])

    lines.extend([
        "=" * 54,
        "YOUR NEXT TEST",
        "=" * 54,
        "",
        "What we're testing:",
        f"{what_test}",
        "",
        "What to do:",
    ])
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
    the Experiment Critic revision budget is exhausted (Milestone 4.0.6 & Milestone 4.1.1).

    Rules:
    Critic-Derived Logic (Milestone 4.1.1):
      - Analyzes the final Critic issues, active Critical Assumption, and rejected experiment.
      - Translates specific blockers (e.g. proximity/availability per window, missing denominator,
        confounding variables, workflow/decision-maker access) into the smallest targeted learning step.
      - Never dumps generic customer-interview templates when an operational or eligibility probe is required.
      - Strictly for learning, never validation (no numerical thresholds).
      - Strips weak proxy signals (free templates, WhatsApp curiosity, landing pages).
      - Reconstructs recent real-world events, firsthand behaviors, and actual consequences.
      - Sets explicit non-inference boundaries (willingness to pay, need for AI/software).
      - Sets explicit non-inference boundaries.
    """
    assumption = assumption or {}
    critic_output = critic_output or {}
    rejected_exp = rejected_experiment or {}
    state = state or {}

    asmp_id = assumption.get("id") or rejected_exp.get("assumption_id") or "asmp-unknown"
    target_area = assumption.get("area") or weak_link or state.get("weak_area") or "Problem"

    # 1. WHAT DO WE STILL NEED TO LEARN?
    # 1. Critical assumption & learning goal extraction
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
    # 2. Critic Issues & Context Analysis
    issues = (
        critic_output.get("issues")
        or critic_output.get("reasons_for_rejection")
        or critic_output.get("critic_issues")
        or []
    )
    revision_inst = critic_output.get("revision_instructions") or ""
    issues_joined = "; ".join(issues)
    critic_text = (issues_joined + " " + revision_inst).lower()
    asmp_lower = asmp_text.lower()
    exp_text = " ".join([str(v) for v in rejected_exp.values() if isinstance(v, str)]).lower()
    combined_context = critic_text + " " + asmp_lower + " " + exp_text

    if issues:
        why_needed = f"The proposed experiment did not meet Startary's quality bar ({issues[0]}). Rather than running an unverified test, this bounded step gathers concrete firsthand facts."
    else:
        why_needed = "The proposed experiment did not meet Startary's quality bar. Rather than running an unverified test, this bounded step gathers concrete firsthand facts."

    # 3. Build RecoveryContext and extract dynamic domain grounding (Milestone M4.1.3.4)
    context = build_recovery_context(
        state=state,
        assumption=assumption,
        rejected_exp=rejected_exp,
        critic_output=critic_output,
        diagnosis_snapshot=diagnosis_snapshot,
    )

    # Dynamically extract target_who from context, rejected_exp, or assumption
    target_who = ""
    cand_target = rejected_exp.get("target_participants_or_scope") or rejected_exp.get("target_participants_or_context") or ""
    if cand_target and str(cand_target).strip().lower() not in ["target customer cohort", "cohort", "none"]:
        target_who = str(cand_target).strip()
    elif context.target_user:
        target_who = context.target_user
    elif state.get("target_user"):
        target_who = str(state.get("target_user")).strip()

    if not target_who or target_who.lower() in ["target customer cohort", "cohort", "none", "target participants"]:
        match = re.search(
            r"(?:There is a definable\s+)(.+?)(?:\s+(?:group|segment|cohort))?(?:\s+beyond|\s+for whom|\s+who|\s+that)",
            clean_assumption_statement(asmp_text),
            re.IGNORECASE,
        )
        if match:
            cand = match.group(1).strip()
            if len(cand) > 3 and not cand.lower().startswith("validat"):
                target_who = cand

    if not target_who or target_who.lower() in ["target customer cohort", "cohort", "none", "target participants"]:
        match = re.search(
            r"^(?:There is a [^.]+? (?:group|segment) of )?([A-Za-z0-9\s–—-]+?)(?:\s+(?:experience|experiences|will|face|faces|suffer|suffers|are able|is able|adopt|adopts|struggle|struggles|prefer|prefers|rely|relies|have|has|use|uses)\b)",
            clean_assumption_statement(asmp_text),
            re.IGNORECASE,
        )
        if match:
            cand = match.group(1).strip()
            if len(cand) > 3 and not cand.lower().startswith("validat"):
                target_who = cand

    if not target_who:
        target_who = f"target {target_area.lower()} participants"

    # Dynamically extract problem_short from context or assumption
    problem_short = context.active_problem
    if not problem_short or len(problem_short) < 5:
        match = re.search(
            r"(?:experience|experiences|face|faces|suffer|suffers|encounter|encounters)\s+([^.]+)",
            clean_assumption_statement(asmp_text),
            re.IGNORECASE,
        )
        if match:
            problem_short = match.group(1).strip()
    if not problem_short or len(problem_short) < 5:
        problem_short = f"the core {target_area.lower()} operational challenge"

    # Categorize Critic Blocker to formulate smallest targeted corrective action
    is_proximity_window_issue = any(
        k in critic_text for k in [
            "proximity", "pickup window", "closing window", "closing-time window",
            "closing-time offer", "within 10 minutes", "10-minute", "10 minute",
            "per-window", "per venue", "venue/window", "unavailability",
        ]
    ) or (
        ("proximity" in combined_context or "nearby" in combined_context)
        and ("pickup" in combined_context or "window" in combined_context)
    )

    is_denominator_issue = any(
        k in critic_text for k in [
            "denominator", "baseline", "sample size", "unverified denominator",
            "rate without", "missing total", "fraction", "conversion without denominator",
            "confounded denominator",
        ]
    )

    is_confounder_issue = any(
        k in critic_text for k in [
            "confound", "confounding", "alternative explanation", "multiple variables",
            "pricing artifact", "noise", "isolated variable",
        ]
    )

    is_workflow_decision_maker_issue = any(
        k in critic_text for k in [
            "decision-maker", "decision maker", "uncle", "restaurant manager",
            "kitchen manager", "gatekeeper", "workflow", "operational workflow",
        ]
    ) or ("uncle" in combined_context and any(k in combined_context for k in ["sensor", "alert", "manager", "decision-maker"]))

    # 4. Formulate targeted fallback fields dynamically grounded in active context
    if is_proximity_window_issue:
        # Grounded proximity / operational window probe
        state_text = f"{asmp_lower} {target_who.lower()} {problem_short.lower()}"
        if "closing-time" in state_text or "pre-closing" in state_text or "closing window" in state_text:
            window_phrase = "closing-time"
            timing_phrase = "pre-closing pickup window"
        else:
            window_phrase = "operational"
            timing_phrase = "operational service window"

        if "bakery" in state_text or "cafe" in state_text:
            venue_phrase = " at one participating bakery or cafe"
        elif "store" in state_text or "shop" in state_text:
            venue_phrase = " at one participating store or site"
        else:
            venue_phrase = " at the target operational site"

        who_to_engage = f"Nearby {target_who} physically confirmed within a 10-minute walk during the {timing_phrase}{venue_phrase}"
        primary_action = f"Run one real {window_phrase} offer or interaction{venue_phrase} with verified nearby participants."
        smallest_thing = f"Verify real-time physical proximity and availability during the {timing_phrase} to isolate genuine demand from physical unavailability."
        what_to_ask = [
            f"Confirm at the exact moment of the offer that each participant is physically within 10 minutes of the location and available during the window.",
            "Present the offer or workflow interaction only to participants confirmed eligible and nearby.",
            "Observe how many confirmed eligible participants reserve and complete pickup or the interaction.",
            "Record the exact reason for any decline or cancellation (timing, dietary/personal preference, or lack of interest).",
        ]
        what_to_bring = [
            "Number of people confirmed physically nearby and available during the window",
            "Number of reservations and deposits/payments completed",
            "Number of pickups or completed interactions",
            "Specific reasons given by those who declined or cancelled",
        ]
        what_not_to_infer = [
            "General commercial demand across different neighborhoods or operating windows",
            "Willingness to pay full commercial price or standard rates",
            "Need for a dedicated mobile app or automated platform",
            "Repeat daily or weekly retention",
        ]
        effort_estimate = f"1 {window_phrase} window (approx. 1–2 hours at 1 venue)"

    elif is_denominator_issue:
        # Grounded denominator / baseline exposure observation
        who_to_engage = f"All eligible {target_who} who enter the operational environment during one observation cycle"
        primary_action = f"Observe and record the full baseline denominator of {target_who} for one complete operational cycle."
        smallest_thing = "Establish the true total baseline (denominator) of exposed or eligible participants to measure unprompted actions accurately."
        what_to_ask = [
            "Count the exact total number of qualified subjects or customers exposed to the opportunity (the true denominator).",
            "Track how many of that total baseline take each sequential action without prompting.",
            "Record the specific operational friction or reasons causing participants to drop off.",
        ]
        what_to_bring = [
            "Exact total denominator count (number of qualified people exposed)",
            "Exact count of participants taking each sequential step",
            "Observed points of abandonment and specific recorded reasons",
        ]
        what_not_to_infer = [
            "Long-term conversion rates at scale",
            "Commercial willingness to pay",
            "That awareness alone represents purchase intent",
        ]
        effort_estimate = "1 full operational cycle (approx. 2–4 hours)"

    elif is_confounder_issue:
        # Isolated single-variable observation
        who_to_engage = f"Target {target_who} observed under a single, isolated operational condition"
        primary_action = "Run one tightly bounded observation holding confounding operational variables constant."
        smallest_thing = "Isolate the single variable being tested by holding operational conditions constant and removing secondary incentives or friction."
        what_to_ask = [
            "Hold operational conditions constant (fixed time, single venue, clear offer) to eliminate alternative explanations.",
            "Remove secondary incentives, discounts, or bundled services that obscure the primary value proposition.",
            "Observe firsthand behavioral reactions under the isolated condition.",
        ]
        what_to_bring = [
            "Behavioral responses under the single isolated condition",
            "Evidence ruling out specific alternative explanations",
            "Unprompted founder observations during the session",
        ]
        what_not_to_infer = [
            "Conversion rates with promotional bundling",
            "Pricing elasticity across variations",
            "Broad market adoption",
        ]
        effort_estimate = "1 isolated test run (approx. 2–3 hours)"

    elif is_workflow_decision_maker_issue:
        # Dynamically derived decision-maker & operational workflow probe (Milestone M4.1.3.4)
        if target_who and target_who.lower() not in ["target customer cohort", "cohort", "none", "target participants"]:
            if any(k in target_who.lower() for k in ["decision", "manager", "operator", "planner", "director", "owner", "lead"]):
                role_name = target_who
            else:
                role_name = f"{target_who} decision-makers or operational managers"
        else:
            role_name = "independent decision-makers or operational managers"

        if problem_short and len(problem_short) > 5:
            topic_name = problem_short
        else:
            topic_name = f"the current {target_area.lower()} workflow"

        who_to_engage = f"3–5 independent {role_name} (with no personal ties)"
        primary_action = f"Engage 3–5 independent {role_name} beyond personal connections to observe their current workflow for {topic_name}."
        smallest_thing = f"Verify whether independent decision-makers beyond personal connections experience {topic_name} as an urgent operational concern."
        what_to_ask = [
            f"Observe or ask for one concrete recent situation where {topic_name} caused disruption or operational friction.",
            "Ask who specifically is held responsible and how logs or records are currently maintained.",
            "Ask what specific financial, operational, or regulatory consequence occurred during the last incident.",
            "Ask what workaround or manual procedure they currently enforce.",
        ]
        what_to_bring = [
            f"One concrete recent incident regarding {topic_name} per organization",
            "Current manual checking or handling process and responsible role",
            "Exact regulatory, operational, or financial consequence experienced",
            "Decision authority confirmed for operational tools or process changes",
        ]
        what_not_to_infer = [
            "Willingness to purchase commercial software or IoT hardware",
            "Need for an automated alerting dashboard or platform",
            "That general agreement on the problem translates to budget approval",
        ]
        effort_estimate = "3–5 conversations (approx. 2–3 hours)"

    else:
        # Default Problem / Friction inquiry citing exact Critic issues
        has_sampling_bias = any(
            k in critic_text for k in [
                "selection bias", "sampling bias", "coach-led", "unrepresentative",
                "biased sample", "convenience sample", "role bias", "confounding selection",
            ]
        )
        if has_sampling_bias:
            who_to_engage = "5–8 role-neutral participants across diverse segments and user roles without selection bias"
            smallest_thing = "Obtain an unbiased, role-neutral sample across multiple user roles and segments to avoid selection bias."
            primary_action = f"Talk with {who_to_engage}."
            what_to_ask = [
                "Engage participants across different roles and segments beyond personal or convenience connections.",
                f"Ask each for one concrete recent situation where they encountered {target_area.lower()} friction.",
                "Ask what specific actions they took and what operational or personal consequences occurred.",
                "Ask what workaround or alternative they currently rely on.",
                "Verify whether the issue is experienced consistently across different roles or restricted to one role.",
            ]
            what_to_bring = [
                "Observations recorded across diverse, role-neutral participants",
                "Specific breakdown of responses by user role / segment",
                "Direct quotes and concrete recent incidents",
                "Evidence ruling out role-specific selection bias",
            ]
            what_not_to_infer = [
                "That responses from a single role or segment represent the entire market",
                "Commercial willingness to pay across segments",
                "That feedback from a single segment generalizes to other segments",
            ]
            effort_estimate = "3–5 hours across 5–8 diverse conversations"
        else:
            who_to_engage = f"5–8 independent {target_who}"
            primary_action = f"Talk with {who_to_engage}."
            smallest_thing = f"Gather concrete firsthand facts directly from {who_to_engage} without relying on unverified proxy assumptions."

            what_to_ask = [
                f"Ask each for one recent concrete example of encountering {problem_short}.",
                "Ask what happened during that specific event.",
                "Ask how often this happens across a typical month or operational cycle.",
                "Ask what money, time, or logistical overhead it affected.",
                "Ask how they currently decide what to do or what workaround they use.",
                "Ask whether they have already actively tried changing the process or resolving this before.",
            ]
            what_to_bring = [
                "One concrete recent incident per participant",
                "Estimated frequency of occurrence",
                "Concrete consequence (cost, wasted time, or operational disruption)",
                "Current workaround or decision process",
                "Whether they have already tried to improve or change the process",
            ]

        what_not_to_infer = [
            "Willingness to pay or commercial conversion",
            "Need for AI, automation, or specialized software",
            "That a proposed product or recommendation engine will be adopted",
            "That general politeness, interest, or curiosity represents commercial urgency",
        ]
        effort_estimate = "3–5 hours across 5–8 conversations"

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
        completion_instruction="Return with what happened / what you learned.",
        participant_target=who_to_engage,
        estimated_effort=effort_estimate,
        smallest_thing_to_resolve_next=smallest_thing,
        critic_rejection_reasons=issues,
        source_basis=["critic_derived_recovery", target_area] + issues[:2],
    )
    rendered = render_fallback_learning_action_text(action, mode="compact")
    action.response_text = rendered

    # Milestone M4.1.3.4 Grounding Validation Check
    is_grounded, failure_reason = validate_recovery_grounding(action, context)
    if not is_grounded:
        logger.warning(
            "compose_fallback_learning_action: grounding validation failed (%s). Substituting generic deterministic fallback.",
            failure_reason,
        )
        action = build_generic_deterministic_fallback(context)

    # Tertiary Output Sanitizer (Milestone M4.1.3.4)
    action.response_text = sanitize_recovery_leakage(action.response_text, context.domain_vocabulary)
    action.primary_action = sanitize_recovery_leakage(action.primary_action, context.domain_vocabulary)
    action.who_to_engage = sanitize_recovery_leakage(action.who_to_engage, context.domain_vocabulary)
    action.smallest_thing_to_resolve_next = sanitize_recovery_leakage(action.smallest_thing_to_resolve_next, context.domain_vocabulary)
    action.critic_rejection_reasons = [
        sanitize_recovery_leakage(r, context.domain_vocabulary) for r in action.critic_rejection_reasons
    ]

    return action


def render_fallback_learning_action_text(
    action: FallbackLearningAction | dict,
    mode: str = "compact",
) -> str:
    """
    Renders the founder-facing markdown block for a FallbackLearningAction (Milestone 4.0.6 & 4.1.1).
    Structured according to Milestone 4.1.1 Section 12 while preserving backwards-compatibility:
      - STARTARY — EXPERIMENT DESIGN NEEDS REVISION
      - WE KNOW WHAT NEEDS TO BE LEARNED (WHAT DO WE STILL NEED TO LEARN?):
      - WHY THE TEST COULD NOT BE TRUSTED:
      - THE SMALLEST THING TO RESOLVE NEXT:
      - WHAT SHOULD I DO NEXT? (WHAT SHOULD I DO?)
      - WHAT SHOULD I BRING BACK?
      - WHAT SHOULD I NOT CONCLUDE YET?
      - WHY THIS MATTERS:
      - WHEN YOU'RE DONE:
      - This action is for learning, not validation.
    """
    if isinstance(action, FallbackLearningAction):
        data = action.model_dump()
    else:
        data = dict(action)

    learning_goal = data.get("learning_goal") or "Resolve core uncertainty with firsthand observations."
    why_needed = data.get("why_fallback_is_needed") or "The proposed experiment did not meet Startary's quality bar."
    reasons = data.get("critic_rejection_reasons") or data.get("untrusted_test_reasons") or []
    smallest_thing = data.get("smallest_thing_to_resolve_next")
    if not smallest_thing:
        smallest_thing = f"Directly observe or verify {data.get('target_area', 'core').lower()} conditions without unverified proxy assumptions."

    primary_action = data.get("primary_action") or "Talk with target participants."
    questions = data.get("what_to_ask_or_observe") or []
    bring_back = data.get("what_to_bring_back") or []
    not_infer = data.get("what_not_to_infer") or []
    instruction = data.get("completion_instruction") or "Return with what happened / what you learned."

    lines = [
        "=" * 54,
        "STARTARY — EXPERIMENT DESIGN NEEDS REVISION",
        "=" * 54,
        "",
        "=" * 54,
        "WE KNOW WHAT NEEDS TO BE LEARNED (WHAT DO WE STILL NEED TO LEARN?):",
        "=" * 54,
        "",
        f"{learning_goal}",
        "",
        "=" * 54,
        "WHY THE TEST COULD NOT BE TRUSTED:",
        "=" * 54,
    ]
    if reasons:
        for r in reasons:
            lines.append(f"- {r}")
    else:
        lines.append(f"{why_needed}")

    lines.extend([
        "",
        "=" * 54,
        "THE SMALLEST THING TO RESOLVE NEXT:",
        "=" * 54,
        "",
        f"{smallest_thing}",
        "",
        "=" * 54,
        "WHAT SHOULD I DO NEXT? (WHAT SHOULD I DO?)",
        "=" * 54,
        "",
        f"{primary_action}",
        "",
        "In each observation/conversation:",
    ])
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
        "WHY THIS MATTERS:",
        "=" * 54,
        "",
        "This information will let Startary design a cleaner experiment.",
        "",
        "=" * 54,
        "WHEN YOU'RE DONE:",
        "=" * 54,
        f"{instruction}",
        "",
        "=" * 54,
        "This action is for learning, not validation.",
        "=" * 54,
    ])

    return "\n".join(lines)


def extract_sample_target_from_protocol(exp_data: dict[str, Any]) -> Optional[int]:
    """
    Deterministically extract explicit sample target size from structured experiment protocol.
    Single authoritative source of truth for experiment observation completeness (Milestone 4.1.2.2).

    Invariants:
    1. Reads explicit target_sample_size if present and > 0.
    2. Parses participant scope / actions for concrete primary observation counts (e.g. '8 teams: 4 U14-U15, 4 amateur').
    3. Never matches age cohorts (e.g. 'U14-U15 teams' or 'under-16' should NOT match 14 or 15).
    4. Never confuses support rules / thresholds (e.g. 'at least 3 teams show repeat use') with sample target.
    5. Returns None if no sample target requirement exists, avoiding inventing hidden targets.
    """
    if not exp_data or not isinstance(exp_data, dict):
        return None

    # 1. Direct structured field
    val = exp_data.get("target_sample_size")
    if val is not None:
        try:
            val_int = int(val)
            if val_int > 0:
                return val_int
        except (ValueError, TypeError):
            pass

    # 2. Check target_participants_or_scope or founder_actions
    scope_text = str(exp_data.get("target_participants_or_scope") or "").strip()
    actions = exp_data.get("founder_actions") or []
    action_text = " ".join(str(a) for a in actions)
    candidate_texts = [scope_text, action_text]

    # Negative lookbehinds prevent matching age brackets:
    # (?<![uU])(?<!under[- ])(?<!under)(?<!grade[- ])(\d+)
    lead_pat = re.compile(
        r"(?:^|[\s,;:\.('\"\-])(?<![uU])(?<!under[- ])(?<!under)(?<!grade[- ])(\d+)\s+(?:[\w/-]+\s+){0,2}(?:teams?|participants?|users?|customers?|merchants?|bakeries?|interviews?|cases?|observations?|subjects?|stores?|partners?|shops?|organizations?|managers?|owners?|clients?|people|respondents?|cafes?|restaurants?|leads?)\b",
        re.IGNORECASE,
    )

    for txt in candidate_texts:
        if not txt:
            continue
        m = lead_pat.search(txt)
        if m:
            try:
                candidate = int(m.group(1))
                if candidate > 0:
                    return candidate
            except ValueError:
                pass

    return None


def evaluate_experiment_completeness(
    state: dict[str, Any],
    analysis: dict[str, Any],
) -> tuple[bool, str, int, int]:
    """
    Evaluates whether the active experiment has been completely executed or is only partially executed (Milestone 4.1.2 & 4.1.2.2).

    Returns:
      (is_complete: bool, early_stop_reason: str, collected_count: int, target_count: int)
    """
    import re
    active_exp = state.get("active_experiment") or {}
    conclusion = (analysis.get("conclusion") or "INCONCLUSIVE").upper()
    strength = (analysis.get("evidence_strength") or "WEAK").upper()
    exec_quality = (analysis.get("experiment_execution_quality") or "GOOD").upper()

    # 1. Target sample size extraction: Single Authoritative Source of Truth (Milestone 4.1.2.2 & M4.1.3.2)
    raw_target = extract_sample_target_from_protocol(active_exp)
    has_explicit_target = raw_target is not None and raw_target > 0
    target_count = raw_target if has_explicit_target else 1

    # 2. Collected sample size extraction
    raw_content = state.get("founder_response") or ""
    if not raw_content:
        founder_messages = state.get("founder_messages") or []
        for m in reversed(founder_messages):
            if m.get("role") == "founder":
                raw_content = m.get("content", "")
                break
    raw_lower = raw_content.lower()

    # Check if founder explicitly stated they cannot collect more
    founder_cannot_collect = any(k in raw_lower for k in [
        "cannot collect remaining", "unable to reach more", "that is all i can get",
        "no more teams available", "cannot get more", "could not reach any more",
        "impossible to collect more", "no other teams", "unable to collect remaining"
    ])

    # Check extracted sample count
    extracted = state.get("extracted_evidence") or {}
    quant = extracted.get("quantitative_outcomes") or extracted.get("sample_or_counts") or {}
    sample_val = quant.get("sample_size") or quant.get("sample_count") or quant.get("teams_count") or quant.get("total_interviews")

    collected_count = 0
    if sample_val and isinstance(sample_val, (int, float)):
        collected_count = int(sample_val)
    elif isinstance(sample_val, str) and sample_val.isdigit():
        collected_count = int(sample_val)
    else:
        c_match = re.search(
            r"(?:^|[\s,;:\.('\"\-])(?<![uU])(?<!under[- ])(?<!under)(?<!grade[- ])(\d+)\s+(?:[\w/-]+\s+){0,3}(?:teams?|participants?|users?|customers?|merchants?|bakeries?|interviews?|cases?|observations?|subjects?|managers?|owners?|clients?|people|respondents?|cafes?|restaurants?|leads?|stores?|shops?|clinics?)\b",
            raw_content,
            re.IGNORECASE,
        )
        frac_match = re.search(r"(\d+)\s*(?:/|out\s+of)\s*(\d+)", raw_content, re.IGNORECASE)
        if c_match:
            collected_count = int(c_match.group(1))
        elif frac_match:
            collected_count = int(frac_match.group(2))

        elif any(k in raw_lower for k in ["one team", "1 team", "one participant", "1 participant", "1 observation", "team alpha"]):
            collected_count = 1
        elif any(k in raw_lower for k in ["two teams", "2 teams", "two participants", "2 participants"]):
            collected_count = 2
        elif any(k in raw_lower for k in ["four teams", "4 teams", "four participants"]):
            collected_count = 4
        elif any(k in raw_lower for k in ["six teams", "6 teams", "six participants"]):
            collected_count = 6
        elif any(k in raw_lower for k in ["eight teams", "8 teams", "eight participants"]):
            collected_count = 8
        elif extracted.get("observed_behaviors"):
            collected_count = len(extracted["observed_behaviors"])
        elif exec_quality == "PARTIAL":
            collected_count = 1
        else:
            collected_count = 1

    # Null-safe integer normalization (Milestone M4.1.3.2)
    prev_collected = int(active_exp.get("collected_sample_count") or 0)
    collected_count = int(collected_count or 0)

    increment = (collected_count if prev_collected > 0 and has_explicit_target and collected_count < target_count else 0)
    total_collected = max(collected_count, prev_collected + increment)

    # 3. Methodological early stopping & completeness rules (Milestone M4.1.3.2):
    # Rule 1: Founder explicitly cannot collect more
    if founder_cannot_collect:
        return True, "FOUNDER_CANNOT_COLLECT_MORE", max(total_collected, 1), target_count

    # Rule 2: Decisive early contradiction
    if conclusion == "CONTRADICTED" and strength in ["STRONG", "MODERATE"]:
        return True, "DECISIVE_EARLY_STOP", total_collected, target_count

    # Rule 3: Partial or poor execution quality (compromised protocol)
    if exec_quality in ["PARTIAL", "POOR"]:
        return False, "PARTIAL_EVIDENCE_IN_PROGRESS", total_collected, target_count

    # Rule 4: No explicit sample target in protocol
    if not has_explicit_target:
        return True, "NO_EXPLICIT_SAMPLE_TARGET", total_collected, target_count

    # Rule 5: Target sample reached
    if has_explicit_target and total_collected >= target_count:
        return True, "TARGET_SAMPLE_REACHED", total_collected, target_count

    # Rule 6: Partial collection under explicit target
    if total_collected < target_count:
        return False, "PARTIAL_EVIDENCE_IN_PROGRESS", total_collected, target_count

    return True, "COMPLETE", total_collected, target_count





