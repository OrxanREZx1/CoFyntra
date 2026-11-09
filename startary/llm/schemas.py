"""
LLM Structured Output Schemas for Startary — Milestone 2.6.

These Pydantic models define the EXACT shape the LLM must return.
They are separate from domain models for a deliberate reason:

  LLM output schemas   — describe what the model is asked to produce
  Domain models        — describe validated business objects

The LLM output is validated by these schemas first.
If validation passes, deterministic merge/evaluation logic converts them into domain objects.
If validation fails, we retry once, then fail safely without touching state.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator

from startary.domain.area_knowledge import AreaStatus
from startary.domain.assessment import RiskFactorLevel
from startary.domain.enums import CriticDimensionResult, CriticVerdict, ExperimentType
from startary.domain.reality_check import EvidenceConclusion
from startary.domain.sufficiency import GroundingQuality
from startary.domain.taxonomy import StartupArea


# ---------------------------------------------------------------------------
# Milestone 1 Schemas: Diagnosis
# ---------------------------------------------------------------------------

class AreaDiagnosisOutput(BaseModel):
    """
    LLM's structured assessment of one startup area during initial diagnosis.
    """

    area: StartupArea = Field(
        description="Which startup area this assessment covers."
    )

    status: AreaStatus = Field(
        description=(
            "Knowledge status: UNKNOWN if nothing provided, "
            "PARTIALLY_KNOWN if some information is present with gaps, "
            "KNOWN if sufficient working picture exists, "
            "UNCLEAR if the founder provided conflicting or ambiguous information."
        )
    )

    understanding: str | None = Field(
        default=None,
        description=(
            "What is directly established about this area from the founder's input. "
            "State only what the founder actually said. "
            "Do not invent or infer beyond what is stated. "
            "Null if nothing is known."
        ),
    )

    founder_claims: list[str] = Field(
        default_factory=list,
        description=(
            "Claims or beliefs the founder stated about this area "
            "that are NOT supported by direct evidence. "
            "Example: 'Everyone will use this' or 'The market is huge'. "
            "Do not include validated facts here."
        ),
    )

    evidence_provided: list[str] = Field(
        default_factory=list,
        description=(
            "Concrete observations or data points the founder has actually collected. "
            "Customer interviews, surveys, usage data, or direct observations qualify. "
            "Informal personal observations count but should be noted as such. "
            "Assumptions and beliefs do NOT qualify as evidence."
        ),
    )

    missing_information: str | None = Field(
        default=None,
        description=(
            "What specific information would materially reduce uncertainty about this area. "
            "Be specific. Null if the area is sufficiently understood."
        ),
    )


class DiagnosisLLMOutput(BaseModel):
    """
    Complete structured diagnosis output from the LLM.
    """

    area_assessments: list[AreaDiagnosisOutput] = Field(
        description="One assessment per startup area evaluated from founder context."
    )

    clear_areas: list[StartupArea] = Field(
        default_factory=list,
        description="Areas that are relatively well-understood from the context provided.",
    )

    unclear_areas: list[StartupArea] = Field(
        default_factory=list,
        description="Areas that were mentioned but remain ambiguous or contradictory.",
    )

    missing_areas: list[StartupArea] = Field(
        default_factory=list,
        description="Areas with no information whatsoever in the founder's context.",
    )

    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="Claims the founder made that appear to lack supporting evidence.",
    )

    diagnosis_summary: str = Field(
        description=(
            "A concise analytical summary of what is currently known about this startup. "
            "Describe the current state of knowledge — do NOT advise or recommend next actions."
        )
    )

    startup_stage: str | None = Field(
        default=None,
        description=(
            "Conservative stage inference. Possible values: 'raw_idea', "
            "'problem_exploration', 'validation', 'mvp_planning', 'early_build'."
        ),
    )


# ---------------------------------------------------------------------------
# Milestone 2.5 Schemas: Context Sufficiency & Clarification
# ---------------------------------------------------------------------------

class ContextSufficiencyLLMOutput(BaseModel):
    """
    Structured evaluation of whether the current founder context is sufficient
    to proceed to Weak-Link decision risk assessment.
    """

    sufficient: bool = Field(
        description=(
            "True if there is enough grounded information to formulate a meaningful weak link "
            "and testable critical assumption without inventing missing context."
        )
    )

    confidence: RiskFactorLevel = Field(
        default=RiskFactorLevel.MEDIUM,
        description="Confidence level in this sufficiency judgment (LOW, MEDIUM, HIGH).",
    )

    grounding_quality: GroundingQuality = Field(
        default=GroundingQuality.WEAK,
        description=(
            "Grounding quality of founder input: "
            "WEAK (generalized beliefs, hypothetical stories, abstractions), "
            "MODERATE (at least one direct observation, personal experience, or customer conversation), "
            "STRONG (multiple concrete observations or qualitative evidence)."
        ),
    )

    would_one_more_answer_materially_change_direction: bool = Field(
        default=False,
        description=(
            "True if one additional clarifying answer has a realistic chance of materially "
            "changing the selected weak link, problem interpretation, or critical assumption."
        ),
    )

    reason: str = Field(
        description="Short analytical explanation of why the context is or is not sufficient."
    )

    blocking_unknowns: list[str] = Field(
        default_factory=list,
        description="Specific unknowns that actively prevent formulating a sound critical assumption.",
    )

    usable_grounding: list[str] = Field(
        default_factory=list,
        description="Concrete pieces of founder-provided information that are already grounded.",
    )

    prohibited_inferences: list[str] = Field(
        default_factory=list,
        description="Details or mechanisms Startary would have to invent if it proceeded now.",
    )

    recommended_question_area: Optional[StartupArea] = Field(
        default=None,
        description="The primary startup area that a clarifying question should target.",
    )

    information_goal: str = Field(
        default="",
        description="The specific missing information goal that must be resolved next.",
    )


class ClarifyingQuestionLLMOutput(BaseModel):
    """
    Structured selection of a single highest-information-value clarifying question.
    """

    question: str = Field(
        min_length=10,
        description=(
            "The natural-language question to ask the founder. "
            "Must NOT inject unstated product mechanisms or assumptions."
        ),
    )

    target_area: StartupArea = Field(
        description="The startup area targeted by this clarifying question.",
    )

    information_goal: str = Field(
        description="The specific missing context this question seeks to obtain.",
    )

    why_this_question_now: str = Field(
        description="Analytical rationale explaining why this question maximizes decision-relevant grounding.",
    )


# ---------------------------------------------------------------------------
# Milestone 2 Schemas: Area Risk Assessment
# ---------------------------------------------------------------------------

class AreaRiskAssessmentLLMOutput(BaseModel):
    """
    LLM's qualitative risk assessment of one startup area.
    """

    area: StartupArea = Field(
        description="Which startup area is being evaluated for decision risk."
    )

    importance: RiskFactorLevel = Field(
        description="How important is this area to the CURRENT startup direction and stage?"
    )

    uncertainty: RiskFactorLevel = Field(
        description="How uncertain is the current understanding of this area?"
    )

    evidence_gap: RiskFactorLevel = Field(
        description="How large is the gap between founder claims and supporting evidence?"
    )

    consequence_of_being_wrong: RiskFactorLevel = Field(
        description="If the founder is wrong about this area, how severely does it affect the direction?"
    )

    rationale: str = Field(
        description="Brief 1-2 sentence analytical justification for the assigned factor levels."
    )


class StartupAreaAssessmentBatchLLMOutput(BaseModel):
    """
    Batch evaluation of all 10 canonical startup areas in a single reasoning call.
    """

    assessments: list[AreaRiskAssessmentLLMOutput] = Field(
        description="Structured risk assessments for all 10 canonical startup areas."
    )


# ---------------------------------------------------------------------------
# Milestone 2 Schemas: Reality Check
# ---------------------------------------------------------------------------

class RealityCheckLLMOutput(BaseModel):
    """
    Structured epistemic Reality Check analyzing claims vs evidence for the weak link.
    """

    weak_link_area: StartupArea = Field(
        description="The selected weak-link startup area being scrutinized."
    )

    relevant_founder_claims: list[str] = Field(
        default_factory=list,
        description="Beliefs or assertions the founder holds regarding this area.",
    )

    available_founder_evidence: list[str] = Field(
        default_factory=list,
        description="Concrete data, interviews, or observations supplied by the founder.",
    )

    what_evidence_supports: list[str] = Field(
        default_factory=list,
        description="Specific elements of the founder's thesis that the evidence actually validates.",
    )

    what_evidence_does_not_support: list[str] = Field(
        default_factory=list,
        description="Assumptions, extrapolations, or claims that the evidence does NOT justify.",
    )

    conclusion: EvidenceConclusion = Field(
        description="Epistemic conclusion: SUPPORTED, PARTIALLY_SUPPORTED, UNSUPPORTED, CONTRADICTED, INCONCLUSIVE, or NO_EVIDENCE."
    )

    unresolved_uncertainty: str = Field(
        description="The core remaining unknown or strategic ambiguity in this area."
    )

    reality_check_summary: str = Field(
        description="Concise analytical synthesis of the Reality Check."
    )


# ---------------------------------------------------------------------------
# Milestone 2 Schemas: Critical Assumption
# ---------------------------------------------------------------------------

class CriticalAssumptionLLMOutput(BaseModel):
    """
    Structured generation of the single critical assumption to test.
    """

    area: StartupArea = Field(
        description="Must match the selected weak-link startup area."
    )

    statement: str = Field(
        min_length=10,
        description=(
            "One single, specific, testable, and falsifiable belief whose failure "
            "would materially change the startup direction. "
            "Must NOT combine multiple independent hypotheses."
        ),
    )

    rationale: str = Field(
        description="Why this specific assumption is the single most critical risk to test now."
    )

    what_falsifies_it: str = Field(
        description="Specific observable real-world evidence or outcome that would prove this assumption false."
    )

    grounding_points: list[str] = Field(
        default_factory=list,
        description="Specific factual grounding points from founder context or Reality Check supporting this assumption.",
    )

    unresolved_uncertainty_addressed: str = Field(
        default="",
        description="The exact unresolved uncertainty from Reality Check that this assumption formulates as a testable proposition.",
    )

    introduced_test_parameters: list[str] = Field(
        default_factory=list,
        description="Any specific test metrics, numbers, or thresholds. Must normally be empty in this milestone unless explicitly founder-supplied.",
    )


# ---------------------------------------------------------------------------
# Milestone 2.6 Schemas: Startup State Extractor
# ---------------------------------------------------------------------------

class ExtractedKnowledgeFieldLLMOutput(BaseModel):
    """A single extracted factual field with provenance metadata from the LLM."""

    value: Optional[str] = Field(
        default=None,
        description="The extracted value as a string. None if not found.",
    )
    provenance: str = Field(
        default="UNKNOWN",
        description="EXPLICIT | INFERRED | UNKNOWN — how this value was obtained.",
    )
    confidence: str = Field(
        default="LOW",
        description="LOW | MEDIUM | HIGH — qualitative confidence.",
    )
    notes: Optional[str] = Field(
        default=None,
        description="Optional note about extraction context.",
    )


class ExtractedTeamMemberLLMOutput(BaseModel):
    """A single team member extracted from founder messages."""

    role: Optional[str] = Field(default=None, description="Role or title.")
    capabilities: list[str] = Field(default_factory=list, description="Specific capabilities mentioned.")
    weekly_hours: Optional[float] = Field(default=None, description="Approximate weekly hours if stated.")


class StartupStateDeltaLLMOutput(BaseModel):
    """
    Structured extraction output for execution context and startup progress.

    The LLM extracts ONLY what is explicitly or safely inferrable from
    founder-authored messages. All fields are optional.

    Rules (enforced by prompts):
    - Do NOT invent values not present in founder messages.
    - Use provenance=INFERRED for safely derived values (e.g. total capacity).
    - Use provenance=EXPLICIT only when the founder directly stated the fact.
    - Do NOT equate: users / active_users / paying_customers / revenue.
    - Do NOT assume funding status, revenue=0, or users=0 if not stated.
    - AI-generated text (questions, analysis) is NOT founder source material.
    """

    # ── Team ────────────────────────────────────────────────────────
    founder_count: Optional[ExtractedKnowledgeFieldLLMOutput] = Field(
        default=None,
        description="Number of co-founders (integer as string). EXPLICIT only.",
    )
    active_team_size: Optional[ExtractedKnowledgeFieldLLMOutput] = Field(
        default=None,
        description="Total actively working team members (integer as string).",
    )
    team_members: list[ExtractedTeamMemberLLMOutput] = Field(
        default_factory=list,
        description="Individual team members if described separately.",
    )

    # ── Capabilities ─────────────────────────────────────────────────
    has_technical_capability: Optional[ExtractedKnowledgeFieldLLMOutput] = Field(
        default=None,
        description="'true' or 'false' — team has technical/engineering capability.",
    )
    has_design_capability: Optional[ExtractedKnowledgeFieldLLMOutput] = Field(
        default=None,
        description="'true' or 'false' — team has design capability.",
    )
    has_marketing_capability: Optional[ExtractedKnowledgeFieldLLMOutput] = Field(
        default=None,
        description="'true' or 'false' — team has marketing/GTM capability.",
    )
    domain_expertise_description: Optional[ExtractedKnowledgeFieldLLMOutput] = Field(
        default=None,
        description="Domain expertise description if mentioned.",
    )

    # ── Capacity ─────────────────────────────────────────────────────
    weekly_hours_per_member: Optional[ExtractedKnowledgeFieldLLMOutput] = Field(
        default=None,
        description="Approximate weekly hours per member (float as string). EXPLICIT.",
    )
    total_weekly_capacity_hours: Optional[ExtractedKnowledgeFieldLLMOutput] = Field(
        default=None,
        description="Total team weekly hours (float as string). Usually INFERRED.",
    )

    # ── Budget (separate from funding) ───────────────────────────────
    available_budget_amount: Optional[ExtractedKnowledgeFieldLLMOutput] = Field(
        default=None,
        description="Available budget amount (float as string). EXPLICIT only.",
    )
    budget_currency: Optional[ExtractedKnowledgeFieldLLMOutput] = Field(
        default=None,
        description="Currency (e.g. 'USD', 'AZN', 'EUR'). Do not assume USD.",
    )
    budget_period: Optional[ExtractedKnowledgeFieldLLMOutput] = Field(
        default=None,
        description="Budget period: 'month', 'week', 'total', etc.",
    )

    # ── Funding ──────────────────────────────────────────────────────
    funding_status: Optional[ExtractedKnowledgeFieldLLMOutput] = Field(
        default=None,
        description="Funding status (str). EXPLICIT only. Do NOT infer bootstrapped.",
    )
    funding_amount_raised: Optional[ExtractedKnowledgeFieldLLMOutput] = Field(
        default=None,
        description="Amount raised if explicitly stated (float as string).",
    )

    # ── Customer access ──────────────────────────────────────────────
    customer_access: Optional[ExtractedKnowledgeFieldLLMOutput] = Field(
        default=None,
        description="Access to potential customers (str). E.g. 'university club'.",
    )
    existing_audience_or_community: Optional[ExtractedKnowledgeFieldLLMOutput] = Field(
        default=None,
        description="Existing audience, community, email list, or social following.",
    )
    existing_assets: list[str] = Field(
        default_factory=list,
        description="Existing assets mentioned: prototype, waitlist, codebase, dataset, etc.",
    )
    geographic_market: Optional[ExtractedKnowledgeFieldLLMOutput] = Field(
        default=None,
        description="Target geographic market. EXPLICIT only if founder mentioned it.",
    )

    # ── Startup Progress ─────────────────────────────────────────────
    development_stage: Optional[ExtractedKnowledgeFieldLLMOutput] = Field(
        default=None,
        description=(
            "DevelopmentStage value: RAW_IDEA | PROBLEM_EXPLORATION | SOLUTION_EXPLORATION | "
            "PROTOTYPE | PRE_MVP | MVP | EARLY_USAGE | EARLY_REVENUE | GROWTH | UNKNOWN."
        ),
    )
    product_status: Optional[ExtractedKnowledgeFieldLLMOutput] = Field(
        default=None,
        description="ProductStatus: NOTHING | CONCEPT | PROTOTYPE | MVP_BUILT | LAUNCHED | UNKNOWN.",
    )
    prototype_exists: Optional[ExtractedKnowledgeFieldLLMOutput] = Field(
        default=None,
        description="'true' or 'false' — does a prototype exist? EXPLICIT only.",
    )
    mvp_exists: Optional[ExtractedKnowledgeFieldLLMOutput] = Field(
        default=None,
        description=(
            "'true' or 'false' — does an MVP exist? EXPLICIT only. "
            "'false' does NOT imply development_stage=PROBLEM_EXPLORATION."
        ),
    )
    launched: Optional[ExtractedKnowledgeFieldLLMOutput] = Field(
        default=None,
        description="'true' or 'false' — launched to real users? EXPLICIT only.",
    )
    validation_activity: Optional[ExtractedKnowledgeFieldLLMOutput] = Field(
        default=None,
        description="Summary of validation activities performed.",
    )
    interviews_completed: Optional[ExtractedKnowledgeFieldLLMOutput] = Field(
        default=None,
        description="Number of user interviews completed (integer as string). EXPLICIT only.",
    )
    active_users: Optional[ExtractedKnowledgeFieldLLMOutput] = Field(
        default=None,
        description="Number of active users (integer as string). Do NOT infer from revenue.",
    )
    paying_customers: Optional[ExtractedKnowledgeFieldLLMOutput] = Field(
        default=None,
        description="Number of paying customers. Semantically separate from active_users.",
    )
    revenue: Optional[ExtractedKnowledgeFieldLLMOutput] = Field(
        default=None,
        description="Revenue description as string. E.g. '$500/month'. Do NOT infer from user count.",
    )
    major_milestones: list[str] = Field(
        default_factory=list,
        description="Major milestones explicitly mentioned.",
    )


# ---------------------------------------------------------------------------
# Milestone 2.6 Schemas: Information Gap Planner
# ---------------------------------------------------------------------------

class InformationGapPlanLLMOutput(BaseModel):
    """
    Unified output of the Adaptive Information Gap Planner.

    Combines strategic sufficiency evaluation and profile gap assessment into one call.
    Produces one structured decision: ask one question, or proceed.
    """

    should_ask: bool = Field(
        description=(
            "True if exactly one clarifying question should be asked right now. "
            "False if context is sufficient to proceed, or asking would not materially improve the decision."
        )
    )

    strategic_sufficient: bool = Field(
        default=False,
        description=(
            "True if founder context is grounded enough to identify the next testable uncertainty. "
            "This is the strategic sufficiency signal."
        ),
    )

    resolution_mode: str = Field(
        default="FOUNDER_KNOWLEDGE",
        description="FOUNDER_KNOWLEDGE | EXTERNAL_EVIDENCE | NOT_YET_FORMULATABLE",
    )

    can_formulate_testable_uncertainty: bool = Field(
        default=False,
        description="True if context is sufficient to formulate a testable hypothesis/assumption for reality to answer.",
    )

    why_this_matters: str = Field(
        default="",
        description="Why this uncertainty or information gap matters for the next step.",
    )

    missing_information: str = Field(
        default="",
        description="Specific missing information or uncertainty identified.",
    )

    recommended_area: Optional[str] = Field(
        default=None,
        description="Startup area where the critical uncertainty lies (e.g. 'Problem', 'Target User').",
    )

    target_domain: Optional[str] = Field(
        default=None,
        description="STRATEGIC | EXECUTION | PROGRESS — where the gap is. None if should_ask=False.",
    )

    target_field_or_area: Optional[str] = Field(
        default=None,
        description="The specific startup area or field to ask about.",
    )

    priority: str = Field(
        default="USEFUL_LATER",
        description="REQUIRED_NOW | USEFUL_LATER | IRRELEVANT_NOW — gap relevance to current decision.",
    )

    information_goal: str = Field(
        default="",
        description="What specific information the question seeks to obtain.",
    )

    reason: str = Field(
        description="Short analytical explanation of why this question was selected, or why no question is needed.",
    )

    question: Optional[str] = Field(
        default=None,
        description=(
            "The single short natural-language question to ask. "
            "None if should_ask=False. Never a list or multi-part form."
        ),
    )


# ---------------------------------------------------------------------------
# Milestone 2.6.1 Schemas: Baseline Startup Check-In
# ---------------------------------------------------------------------------

class BaselineSufficiencyLLMOutput(BaseModel):
    """
    Structured judgment of whether Startary has gathered sufficient minimum operational
    context across the 5 core baseline dimensions.
    """
    is_sufficient: bool = Field(
        description="True if Startary has sufficient baseline knowledge across the 5 dimensions."
    )
    team_status: str = Field(
        default="MISSING",
        description="KNOWN_ENOUGH | MISSING | NOT_RELEVANT",
    )
    capacity_status: str = Field(
        default="MISSING",
        description="KNOWN_ENOUGH | MISSING | NOT_RELEVANT",
    )
    product_stage_status: str = Field(
        default="MISSING",
        description="KNOWN_ENOUGH | MISSING | NOT_RELEVANT",
    )
    validation_status: str = Field(
        default="MISSING",
        description="KNOWN_ENOUGH | MISSING | NOT_RELEVANT",
    )
    resource_status: str = Field(
        default="MISSING",
        description="KNOWN_ENOUGH | MISSING | NOT_RELEVANT",
    )
    missing_groups: list[str] = Field(
        default_factory=list,
        description="Dimensions that still need clarification.",
    )
    reason: str = Field(
        description="Brief analytical explanation of what is known versus what is still required.",
    )


class BaselineQuestionPlanLLMOutput(BaseModel):
    """
    Structured selection of the next grouped baseline question.
    """
    should_ask: bool = Field(
        description="True if a baseline prompt should be presented to the founder."
    )
    target_group: Optional[str] = Field(
        default=None,
        description="TEAM | CAPACITY | PRODUCT_STAGE | VALIDATION_OR_USER | RESOURCE",
    )
    target_fields: list[str] = Field(
        default_factory=list,
        description="Fields targeted by this question.",
    )
    question: Optional[str] = Field(
        default=None,
        description="Single concise, grouped natural-language question. Never a multi-part form.",
    )
    reason: str = Field(
        description="Why this group was chosen and what information it will compress.",
    )


# ---------------------------------------------------------------------------
# Milestone 2.6.1 Schemas: Weak-Link Discrimination
# ---------------------------------------------------------------------------

class WeakLinkDiscriminationLLMOutput(BaseModel):
    """
    Structured evaluation of selection confidence and candidate ambiguity for the weak link.
    """
    selection_confidence: str = Field(
        description="LOW | MEDIUM | HIGH — confidence that the leading candidate deserves priority over competitors.",
    )
    competing_areas: list[StartupArea] = Field(
        default_factory=list,
        description="Areas sharing the maximum risk score or within near-identical decision risk.",
    )
    discrimination_reason: str = Field(
        description="Analytical rationale explaining why the winner is distinct or why ambiguity remains.",
    )
    missing_discriminating_information: str = Field(
        default="",
        description="Specific real-world fact or observation that would separate the candidates.",
    )
    should_clarify: bool = Field(
        default=False,
        description="True if asking a clarifying question could plausibly separate the candidates.",
    )
    resolution_mode: str = Field(
        default="FOUNDER_KNOWLEDGE",
        description="FOUNDER_KNOWLEDGE | EXTERNAL_EVIDENCE — whether separating candidates requires founder knowledge or external evidence.",
    )
    recommended_information_goal: str = Field(
        default="",
        description="Goal for the discriminating question.",
    )


class WeakLinkClarifyingQuestionLLMOutput(BaseModel):
    """
    Structured formulation of a single discriminating question to separate weak-link candidates.
    """
    question: str = Field(
        min_length=10,
        description="The natural-language question crafted to differentiate tied candidates.",
    )
    competing_areas: list[StartupArea] = Field(
        default_factory=list,
        description="The areas this question aims to differentiate.",
    )
    information_goal: str = Field(
        description="What observation or fact this question seeks to obtain.",
    )
    why_this_question_now: str = Field(
        description="Analytical justification for why this question resolves candidate ambiguity.",
    )


# ---------------------------------------------------------------------------
# Milestone 3 Schemas: Experiment Generation & Experiment Critic
# ---------------------------------------------------------------------------

class ExperimentParameterLLM(BaseModel):
    """
    Structured parameter or decision rule for the experiment.
    """
    name: str = Field(
        description="Parameter identifier, e.g. 'sample_size', 'response_threshold', 'time_box'.",
    )
    value: str = Field(
        description="Specific provisional value or decision rule (e.g. '6-8 sellers', 'at least 5 of 8').",
    )
    provenance: str = Field(
        default="AI_PROPOSED",
        description="AI_PROPOSED (provisional heuristic) or FOUNDER_SUPPLIED (stated by founder).",
    )
    notes: str = Field(
        default="",
        description="Context explaining that this is a provisional test rule, not a market fact.",
    )


class ExperimentGeneratorLLMOutput(BaseModel):
    """
    Structured generation of ONE specific, falsifiable experiment testing the active critical assumption.
    """
    experiment_type: ExperimentType = Field(
        description="Controlled category of the experiment.",
    )
    objective: str = Field(
        min_length=5,
        description="What specific uncertainty this experiment is trying to reduce.",
    )
    hypothesis: str = Field(
        min_length=5,
        description="The exact active critical assumption statement under test.",
    )
    method: str = Field(
        min_length=5,
        description="How the experiment will be conducted in the real world.",
    )
    target_participants_or_scope: str = Field(
        description="Target participant profile, participant count, or channel scope.",
    )
    target_sample_size: Optional[int] = Field(
        default=None,
        description="Explicit target sample size or number of qualifying observations required (Milestone 4.1.2.2).",
    )
    founder_actions: list[str] = Field(
        min_length=1,
        description="Concrete sequential steps the founder must take.",
    )
    evidence_to_collect: str = Field(
        min_length=5,
        description="What specific data, behavioral observations, or artifacts the founder must capture.",
    )
    support_signal: str = Field(
        min_length=5,
        description="What observable result would increase confidence in the assumption.",
    )
    contradiction_signal: str = Field(
        min_length=5,
        description="What observable result would materially weaken or falsify the assumption.",
    )
    inconclusive_signal: str = Field(
        min_length=5,
        description="What outcome would fail to provide decision-relevant evidence.",
    )
    valid_inferences: list[str] = Field(
        min_length=1,
        description="What this experiment CAN teach us (epistemic boundary).",
    )
    invalid_inferences: list[str] = Field(
        min_length=1,
        description="What this experiment CANNOT teach us (epistemic guardrails).",
    )
    estimated_time: Optional[str] = Field(
        default="2-4 days",
        description="Estimated time to complete (e.g. '2-4 days', '1 week').",
    )
    estimated_cost: Optional[str] = Field(
        default="Near-zero / $0",
        description="Estimated financial budget (e.g. 'Near-zero / $0', '$50').",
    )
    required_capabilities: list[str] = Field(
        default_factory=list,
        description="Skills or capabilities required to run the test.",
    )
    required_access: list[str] = Field(
        default_factory=list,
        description="Customer, community, or distribution access required.",
    )
    parameters: list[ExperimentParameterLLM] = Field(
        default_factory=list,
        description="Explicit test parameters and decision thresholds with provenance.",
    )


class ExperimentCriticLLMOutput(BaseModel):
    """
    Structured critique evaluating experiment quality, alignment, feasibility, and falsifiability.
    """
    # Milestone 6.2.5: strict enums (exact tokens; trimming/case are normalized, nothing else).
    verdict: CriticVerdict = Field(
        description="ACCEPT or REVISE.",
    )
    issues: list[str] = Field(
        default_factory=list,
        description="Specific identified issues or weaknesses (empty if ACCEPT).",
    )
    assumption_alignment: CriticDimensionResult = Field(
        description="PASS or FAIL — does it directly test the active critical assumption?",
    )
    evidence_validity: CriticDimensionResult = Field(
        description="PASS or FAIL — could the evidence legitimately support/contradict the claim?",
    )
    feasibility: CriticDimensionResult = Field(
        description="PASS or FAIL — can this founder realistically execute it with current resources?",
    )
    cost_efficiency: CriticDimensionResult = Field(
        description="PASS or FAIL — is this the cheapest/fastest test of similar information value?",
    )
    falsifiability: CriticDimensionResult = Field(
        description="PASS or FAIL — can the assumption meaningfully fail?",
    )
    scope: CriticDimensionResult = Field(
        description="PASS or FAIL — is the test small and stage-appropriate (not build-heavy)?",
    )
    revision_instructions: str = Field(
        default="",
        description="Concise, actionable instructions for the generator if REVISE.",
    )
    rationale: str = Field(
        description="Summary rationale for the critic decision.",
    )

    @field_validator(
        "verdict", "assumption_alignment", "evidence_validity", "feasibility",
        "cost_efficiency", "falsifiability", "scope", mode="before",
    )
    @classmethod
    def _normalize_token(cls, value: Any) -> Any:
        return value.strip().upper() if isinstance(value, str) else value


# ---------------------------------------------------------------------------
# Milestone 6.2.6 (STARTARY_EXPERIMENT_REVISION_V2 only): revision architecture schemas
# ---------------------------------------------------------------------------

class RevisionScope(StrEnum):
    NONE = "NONE"            # verdict is ACCEPT
    TARGETED = "TARGETED"    # the defects can be fixed by editing specific fields of this draft
    REDESIGN = "REDESIGN"    # a different method, population, or evidence source is needed


class ExperimentCriticV2LLMOutput(ExperimentCriticLLMOutput):
    """Full critique (V2): identical evaluation plus the critic's judgement of the revision scope."""
    revision_scope: RevisionScope = Field(
        description="NONE if ACCEPT. On REVISE: TARGETED if editing specific fields fixes the issues; REDESIGN if a new method, population, or evidence source is required.",
    )

    @field_validator("revision_scope", mode="before")
    @classmethod
    def _normalize_scope(cls, value: Any) -> Any:
        return value.strip().upper() if isinstance(value, str) else value


class ExperimentFocusedCriticLLMOutput(ExperimentCriticLLMOutput):
    """
    Focused re-critique of a targeted edit. All six dimensions are still re-evaluated; acceptance
    additionally requires every original blocking issue resolved and no new blocking issue.
    """
    original_issues_resolved: bool = Field(
        description="True only if EVERY blocking issue from the previous critique is resolved by the revision.",
    )
    new_blocking_issues: list[str] = Field(
        default_factory=list,
        description="Blocking problems introduced by the revision itself (empty if none).",
    )


class ExperimentRevisionPatchLLMOutput(BaseModel):
    """
    Targeted edit of a rejected experiment: ONLY the fields that must change are filled; every other
    field stays null and is kept verbatim from the original draft. The merged draft must still satisfy
    ExperimentGeneratorLLMOutput.
    """
    experiment_type: Optional[ExperimentType] = Field(default=None, description="Only if the critique requires a different experiment type.")
    objective: Optional[str] = Field(default=None, description="Revised objective, or null to keep.")
    method: Optional[str] = Field(default=None, description="Revised method, or null to keep.")
    target_participants_or_scope: Optional[str] = Field(default=None, description="Revised participants/scope, or null to keep.")
    target_sample_size: Optional[int] = Field(default=None, description="Revised sample size, or null to keep.")
    founder_actions: Optional[list[str]] = Field(default=None, description="Full revised list of founder actions, or null to keep.")
    evidence_to_collect: Optional[str] = Field(default=None, description="Revised evidence to collect, or null to keep.")
    support_signal: Optional[str] = Field(default=None, description="Revised support signal, or null to keep.")
    contradiction_signal: Optional[str] = Field(default=None, description="Revised contradiction signal, or null to keep.")
    inconclusive_signal: Optional[str] = Field(default=None, description="Revised inconclusive signal, or null to keep.")
    valid_inferences: Optional[list[str]] = Field(default=None, description="Full revised list, or null to keep.")
    invalid_inferences: Optional[list[str]] = Field(default=None, description="Full revised list, or null to keep.")
    estimated_time: Optional[str] = Field(default=None, description="Revised time estimate, or null to keep.")
    estimated_cost: Optional[str] = Field(default=None, description="Revised cost estimate, or null to keep.")
    required_capabilities: Optional[list[str]] = Field(default=None, description="Full revised list, or null to keep.")
    required_access: Optional[list[str]] = Field(default=None, description="Full revised list, or null to keep.")
    parameters: Optional[list[ExperimentParameterLLM]] = Field(default=None, description="Full revised list, or null to keep.")


# ---------------------------------------------------------------------------
# Milestone 3.1 Schemas: Founder-Facing Startup Diagnosis Snapshot
# ---------------------------------------------------------------------------

class StartupDiagnosisSnapshotLLMOutput(BaseModel):
    """
    Structured output for composing the founder-facing startup diagnosis snapshot.
    Translates internal reasoning into an empathetic, clear, and actionable summary.
    """
    stage: str = Field(
        description="Current startup stage name (e.g. Early Build / Pre-MVP, Idea, Solution Exploration).",
    )
    stage_explanation: str = Field(
        description="1-2 sentence founder-friendly explanation of why the startup is at this stage.",
    )
    grounded_areas: list[str] = Field(
        default_factory=list,
        description="3-5 factual statements describing things Startary understands with reasonable grounding.",
    )
    current_advantages: list[str] = Field(
        default_factory=list,
        description="1-3 genuine practical execution or learning assets (or empty if none).",
    )
    execution_constraints: list[str] = Field(
        default_factory=list,
        description="2-4 factual limitations on team, budget, time, or capabilities.",
    )
    major_uncertainties: list[str] = Field(
        default_factory=list,
        description="Top 3 to 5 uncertainties that materially matter to the current stage and next decision.",
    )
    weak_link: str = Field(
        description="The active weak link area name matching the selected weak link.",
    )
    weak_link_reason: str = Field(
        description="Empathetic, clear explanation of why this area represents the highest immediate risk.",
    )
    weak_link_confidence: str = Field(
        description="Selection confidence level: LOW, MEDIUM, or HIGH.",
    )
    weak_link_confidence_explanation: str = Field(
        description="Founder-friendly explanation of what the confidence level means.",
    )
    competing_risks: list[str] = Field(
        default_factory=list,
        description="Up to 3 most relevant competing risk areas.",
    )
    reality_summary: str = Field(
        description="Concise 2-4 sentence founder-facing summary of the Reality Check.",
    )
    recommended_learning_goal: str = Field(
        description="The strategic learning objective (what we need to learn next) before the experiment.",
    )
    critical_assumption: str = Field(
        description="The verbatim statement of the active Critical Assumption.",
    )


# ---------------------------------------------------------------------------
# Milestone 3.2 Schemas: Founder Intent & Conversational Awareness
# ---------------------------------------------------------------------------

class FounderTurnInterpretationLLMOutput(BaseModel):
    """
    Structured output classifying founder intent, questions, concerns, corrections,
    and ability to answer/act.
    """
    intent_types: list[str] = Field(
        default_factory=list,
        description="List of intents: ANSWER, QUESTION, CONCERN, CORRECTION, DOES_NOT_KNOW, CANNOT_ANSWER_NOW, CANNOT_ACT_NOW, NEW_EVIDENCE, PREFERENCE, REQUEST_FOR_ADVICE, OTHER.",
    )
    direct_answer: bool = Field(
        default=False,
        description="True if the message answers the pending question.",
    )
    direct_question: Optional[str] = Field(
        default=None,
        description="The specific question the founder asked Startary, if any.",
    )
    concern_or_goal: Optional[str] = Field(
        default=None,
        description="Core fear, strategic concern, or underlying objective expressed.",
    )
    correction: Optional[str] = Field(
        default=None,
        description="Any misunderstanding or incorrect assumption by Startary that the founder corrected.",
    )
    cannot_answer_now: bool = Field(
        default=False,
        description="True if founder explicitly says they don't know or can't answer right now.",
    )
    cannot_act_now: bool = Field(
        default=False,
        description="True if founder indicates they cannot do external customer discovery or tasks right now.",
    )
    uncertainty_statement: Optional[str] = Field(
        default=None,
        description="Explicit uncertainty or risk the founder highlighted.",
    )
    new_evidence_summary: Optional[str] = Field(
        default=None,
        description="Summary of any new customer/operational facts introduced in this turn.",
    )
    founder_preference_or_constraint: Optional[str] = Field(
        default=None,
        description="Personal boundary, preference, or working style constraint expressed.",
    )
    detected_founder_goal: Optional[str] = Field(
        default=None,
        description="Direct founder objective: FIND_INITIAL_MARKET | DECIDE_IF_WORTH_CONTINUING | VALIDATE_PROBLEM | FIND_CUSTOMERS | UNDERSTAND_WHAT_TO_BUILD | OTHER",
    )
    is_already_answered_claim: bool = Field(
        default=False,
        description="True if founder stated they already answered this question or Startary already asked it.",
    )
    response_priority: str = Field(
        default="NORMAL",
        description="LOW | NORMAL | HIGH — HIGH if direct question, fear/concern, correction, or inability.",
    )
    interpretation_summary: str = Field(
        description="Concise 1-2 sentence summary of what the founder communicated.",
    )


class AdvisorResponseLLMOutput(BaseModel):
    """
    Structured output for the advisor response to the founder.
    Enforces strict binding to the planner's question_id and semantic information target.
    """
    question_id: str = Field(
        description="Must EXACTLY match the question_id provided in the prompt.",
    )
    next_question_target: str = Field(
        description="Must EXACTLY match the information_target provided in the prompt (e.g. CAPACITY, TEAM, RESOURCE).",
    )
    next_question_rephrased: Optional[str] = Field(
        default=None,
        description="The planner's canonical question rephrased naturally for conversation, preserving the exact same information target.",
    )
    acknowledge: Optional[str] = Field(
        default=None,
        description="1 sentence acknowledging founder input, specific concern, or correction.",
    )
    answer_to_founder: Optional[str] = Field(
        default=None,
        description="Substantive, grounded answer to founder's direct question/concern based on current evidence (no false guarantees).",
    )
    diagnosis_delta: Optional[str] = Field(
        default=None,
        description="Brief note if new evidence materially strengthened or weakened understanding (omit if trivial answer).",
    )
    response_mode: Optional[str] = Field(
        default="BRIEF_CONTINUE",
        description="The conversational response mode: BRIEF_CONTINUE | MATERIAL_UPDATE | DIRECT_ANSWER | CONCERN_RESPONSE | CORRECTION_RESPONSE | HANDOFF_TO_TEST",
    )
    should_ask_next_question: bool = Field(
        default=True,
        description="True if the next question should be asked now; False if answering founder is sufficient.",
    )
    response_text: str = Field(
        description="The final integrated conversational response (1 to 4 sentences total, strictly matching the response mode).",
    )



# ---------------------------------------------------------------------------
# Milestone 4 Schemas: Evidence Extraction & Evidence Analysis
# ---------------------------------------------------------------------------

class CountMetricItem(BaseModel):
    """A named count or sample size extracted from evidence."""
    name: str = Field(..., description="Metric or count name (e.g. 'interviewed', 'positive_responses', 'negative_responses').")
    value: int = Field(..., description="Integer count value.")


class MetricOutcomeItem(BaseModel):
    """A quantitative outcome or metric extracted from evidence."""
    name: str = Field(..., description="Metric name.")
    value: str = Field(..., description="Metric value or description.")


class RecommendedStateUpdateItem(BaseModel):
    """A recommended state update for a specific startup area."""
    area: str = Field(..., description="Startup area name (e.g. 'Problem', 'Demand Signal').")
    update_text: str = Field(..., description="Factual update summarizing evidence for this area.")


class ExtractedEvidenceItemLLMOutput(BaseModel):
    """A discrete observation or factual finding from founder-submitted evidence."""
    observation: str = Field(..., description="The concrete observation or quote.")
    evidence_type: str = Field(
        default="OBSERVATION",
        description="INTERVIEW_NOTES | OBSERVATION | USAGE_DATA | COMMITMENT | PAYMENT | OUTREACH_RESULT | PROTOTYPE_TEST_RESULT | OTHER",
    )
    provenance: str = Field(
        default="FOUNDER_REPORTED",
        description="FOUNDER_REPORTED | FOUNDER_UPLOADED | SYSTEM_DERIVED",
    )


class ExtractedEvidenceLLMOutput(BaseModel):
    """
    Structured extraction of raw founder-submitted experiment results.
    Separates what was actually observed and executed from subjective interpretation.
    """
    actions_completed: list[str] = Field(
        default_factory=list,
        description="List of concrete experiment steps or actions the founder completed.",
    )
    sample_or_counts_items: list[CountMetricItem] = Field(
        default_factory=list,
        description="Extracted sample size, participant counts, or outreach numbers (e.g. name='interviewed', value=7).",
    )
    observed_behaviors: list[str] = Field(
        default_factory=list,
        description="Actual behaviors observed (e.g. 'Managers covered shifts themselves', 'Used WhatsApp group').",
    )
    quotes: list[str] = Field(
        default_factory=list,
        description="Verbatim or direct paraphrased quotes supplied by founder.",
    )
    quantitative_outcomes_items: list[MetricOutcomeItem] = Field(
        default_factory=list,
        description="Specific metrics, percentages, dollar amounts, or ratios.",
    )
    incomplete_actions: list[str] = Field(
        default_factory=list,
        description="Planned experiment actions that were omitted, postponed, or could not be completed.",
    )
    deviations_from_plan: list[str] = Field(
        default_factory=list,
        description="How the founder's execution deviated from the generated experiment design.",
    )
    missing_result_information: list[str] = Field(
        default_factory=list,
        description="Important result dimensions that the founder did not report or clarify.",
    )
    is_future_intention_only: bool = Field(
        default=False,
        description="CRITICAL: True if the submission ONLY describes future intentions or plans (e.g. 'I will do 5 interviews tomorrow') rather than completed execution.",
    )
    evidence_items: list[ExtractedEvidenceItemLLMOutput] = Field(
        default_factory=list,
        description="Discrete evidence items extracted from submission.",
    )
    direct_evidence_summary: str = Field(
        ...,
        description="Objective, factual summary of the executed actions and observations without validating assumptions.",
    )

    def __init__(self, **data: Any):
        if "sample_or_counts" in data and "sample_or_counts_items" not in data:
            raw_counts = data.pop("sample_or_counts") or {}
            items = []
            for k, v in raw_counts.items():
                try:
                    val = int(v)
                except (ValueError, TypeError):
                    val = 0
                items.append(CountMetricItem(name=str(k), value=val))
            data["sample_or_counts_items"] = items
        if "quantitative_outcomes" in data and "quantitative_outcomes_items" not in data:
            raw_metrics = data.pop("quantitative_outcomes") or {}
            data["quantitative_outcomes_items"] = [
                MetricOutcomeItem(name=str(k), value=str(v))
                for k, v in raw_metrics.items()
            ]
        super().__init__(**data)

    @property
    def sample_or_counts(self) -> dict[str, int]:
        return {item.name: item.value for item in self.sample_or_counts_items}

    @property
    def quantitative_outcomes(self) -> dict[str, str]:
        return {item.name: item.value for item in self.quantitative_outcomes_items}

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        d = super().model_dump(*args, **kwargs)
        d["sample_or_counts"] = {item.name: item.value for item in self.sample_or_counts_items}
        d["quantitative_outcomes"] = {item.name: item.value for item in self.quantitative_outcomes_items}
        return d


class EvidenceAnalysisLLMOutput(BaseModel):
    """
    Structured epistemic evaluation of submitted evidence against the active experiment
    and active critical assumption.
    """
    conclusion: str = Field(
        ...,
        description="SUPPORTED | PARTIALLY_SUPPORTED | CONTRADICTED | INCONCLUSIVE",
    )
    evidence_strength: str = Field(
        ...,
        description="WEAK | MODERATE | STRONG — epistemic weight of the evidence.",
    )
    experiment_execution_quality: str = Field(
        default="GOOD",
        description="GOOD | PARTIAL | POOR — quality and fidelity of the founder's execution.",
    )
    support_findings: list[str] = Field(
        default_factory=list,
        description="Specific observations or data points that support the critical assumption.",
    )
    contradiction_findings: list[str] = Field(
        default_factory=list,
        description="Specific observations or data points that contradict, weaken, or falsify the assumption.",
    )
    inconclusive_findings: list[str] = Field(
        default_factory=list,
        description="Gaps, sample limitations, or confounding factors making parts of the result inconclusive.",
    )
    deviations_from_plan: list[str] = Field(
        default_factory=list,
        description="How execution deviations influenced the reliability of findings.",
    )
    valid_learnings: list[str] = Field(
        default_factory=list,
        description="Legitimate inferences permitted by this experiment's method and evidence.",
    )
    invalid_learnings: list[str] = Field(
        default_factory=list,
        description="Overreaching conclusions that this experiment cannot legitimately claim to establish.",
    )
    assumption_confidence_change: str = Field(
        ...,
        description="Qualitative summary of how belief in the critical assumption shifted.",
    )
    reasoning_summary: str = Field(
        ...,
        description="Epistemic rationale explaining how the findings compare to support/contradiction signals.",
    )
    recommended_state_updates_items: list[RecommendedStateUpdateItem] = Field(
        default_factory=list,
        description="Proposed candidate updates for StartupState areas directly informed by this evidence.",
    )

    def __init__(self, **data: Any):
        if "recommended_state_updates" in data and "recommended_state_updates_items" not in data:
            raw_updates = data.pop("recommended_state_updates") or {}
            data["recommended_state_updates_items"] = [
                RecommendedStateUpdateItem(area=str(k), update_text=str(v))
                for k, v in raw_updates.items()
            ]
        super().__init__(**data)

    @property
    def recommended_state_updates(self) -> dict[str, str]:
        return {item.area: item.update_text for item in self.recommended_state_updates_items}

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        d = super().model_dump(*args, **kwargs)
        d["recommended_state_updates"] = {item.area: item.update_text for item in self.recommended_state_updates_items}
        return d


# ---------------------------------------------------------------------------
# Milestone 4.0.3 Schemas: Evidence Admission Gate & Waiting-For-Evidence Advisor
# ---------------------------------------------------------------------------

class EvidenceTurnAssessmentLLMOutput(BaseModel):
    """
    Structured classification of a founder message received during WAITING_FOR_EVIDENCE.
    Decides whether the message represents actual completed real-world evidence or
    a conversational turn (question, request for help, future intention, blocker).
    """
    disposition: str = Field(
        description=(
            "ACTUAL_EVIDENCE | QUESTION_ABOUT_EXPERIMENT | REQUEST_FOR_EXECUTION_HELP | "
            "FUTURE_INTENTION | CANNOT_EXECUTE | CORRECTION | UNRELATED | AMBIGUOUS"
        )
    )
    contains_completed_external_action: bool = Field(
        default=False,
        description="True if message reports completed real-world actions with external parties/customers.",
    )
    contains_results: bool = Field(
        default=False,
        description="True if message includes observations, findings, metrics, quotes, or answers.",
    )
    contains_only_future_intent: bool = Field(
        default=False,
        description="True if message only expresses an intention/plan to do the test later.",
    )
    asks_for_help: bool = Field(
        default=False,
        description="True if founder is asking how to execute the test or find participants.",
    )
    asks_for_explanation: bool = Field(
        default=False,
        description="True if founder is asking what to do, why this test, or for clarification.",
    )
    cannot_execute: bool = Field(
        default=False,
        description="True if founder reports being blocked or unable to execute the test.",
    )
    evidence_candidate_text: Optional[str] = Field(
        default=None,
        description="The specific text that describes results or evidence (for mixed messages).",
    )
    rationale_summary: str = Field(
        description="Short rationale explaining the classification.",
    )


class WaitingForEvidenceAdvisorLLMOutput(BaseModel):
    """
    Structured advisor response to a founder message while remaining in WAITING_FOR_EVIDENCE.
    Preserves the accepted experiment while addressing founder questions, help requests, or blockers.
    """
    response_text: str = Field(
        description="Empathetic, clear conversational response to the founder.",
    )
    rationale: str = Field(
        description="Brief reasoning explaining how the response addresses the founder's specific intent.",
    )


# ---------------------------------------------------------------------------
# Milestone 4.1 Schemas: Build Readiness / Experiment Escalation
# ---------------------------------------------------------------------------

class BuildReadinessAssessmentLLMOutput(BaseModel):
    """
    Structured build readiness decision evaluating whether the next learning cycle
    should remain manual or requires a minimal product-mediated prototype (Milestone 4.1).
    """
    decision: str = Field(
        description="Controlled decision: 'CONTINUE_MANUAL' or 'BUILD_MINIMAL_PROTOTYPE'."
    )
    reason: str = Field(
        description="Concise rationale explaining why manual testing suffices or why software is now required."
    )
    manual_learning_sufficient: bool = Field(
        default=False,
        description="True if prerequisite uncertainties have been sufficiently reduced with empirical evidence."
    )
    next_uncertainty_requires_product: bool = Field(
        default=False,
        description="True if the active critical assumption strictly requires unmediated product interaction."
    )
    minimum_build_scope: list[str] = Field(
        default_factory=list,
        description="Minimal essential capabilities required to test this specific assumption."
    )
    must_not_build: list[str] = Field(
        default_factory=list,
        description="Explicitly excluded features to protect against feature creep and roadmapping."
    )
    evidence_basis: list[str] = Field(
        default_factory=list,
        description="Grounded historical evidence citations supporting this decision."
    )





