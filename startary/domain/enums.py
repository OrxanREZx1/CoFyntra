"""
Shared enumerations for the Startary domain.

Rationale for WorkflowPhase choices:
  UNDERSTAND          — founder is describing their startup; system is building initial state
  DIAGNOSE            — system is evaluating startup areas against known information
  TEST                — an assumption is being tested via an experiment
  WAITING_FOR_EVIDENCE — experiment is in progress; system waits for real-world data
  LEARN               — evidence has returned; system re-evaluates and updates state

DIAGNOSE and LEARN are included beyond the minimum four because the
product loop (DIAGNOSE → WEAK LINK → … → RE-DIAGNOSE → UPDATE STATE)
makes them architecturally distinct routing targets.  Without them,
routing between UNDERSTAND and TEST would require magic strings.
"""

from enum import StrEnum


class WorkflowPhase(StrEnum):
    UNDERSTAND = "UNDERSTAND"
    CLARIFYING_CONTEXT = "CLARIFYING_CONTEXT"
    DIAGNOSE = "DIAGNOSE"
    TEST = "TEST"
    WAITING_FOR_EVIDENCE = "WAITING_FOR_EVIDENCE"
    WAITING_FOR_LEARNING_INPUT = "WAITING_FOR_LEARNING_INPUT"
    LEARN = "LEARN"


class AssumptionStatus(StrEnum):
    UNTESTED = "UNTESTED"
    TESTING = "TESTING"
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    WEAKENED = "WEAKENED"
    CONTRADICTED = "CONTRADICTED"
    INCONCLUSIVE = "INCONCLUSIVE"
    SUPERSEDED = "SUPERSEDED"


class ExperimentStatus(StrEnum):
    DRAFT = "DRAFT"
    PROPOSED = "PROPOSED"
    READY = "READY"
    BLOCKED = "BLOCKED"
    IN_PROGRESS = "IN_PROGRESS"
    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"
    ABANDONED = "ABANDONED"


class ExperimentType(StrEnum):
    CUSTOMER_INTERVIEW = "CUSTOMER_INTERVIEW"
    PROBLEM_INTERVIEW = "PROBLEM_INTERVIEW"
    OBSERVATION = "OBSERVATION"
    CONCIERGE_TEST = "CONCIERGE_TEST"
    FAKE_DOOR = "FAKE_DOOR"
    LANDING_PAGE = "LANDING_PAGE"
    PROTOTYPE_TEST = "PROTOTYPE_TEST"
    MANUAL_SERVICE = "MANUAL_SERVICE"
    PREORDER_OR_COMMITMENT = "PREORDER_OR_COMMITMENT"
    PRICING_TEST = "PRICING_TEST"
    OUTREACH_TEST = "OUTREACH_TEST"
    CHANNEL_TEST = "CHANNEL_TEST"
    OTHER = "OTHER"


class ParameterProvenance(StrEnum):
    AI_PROPOSED = "AI_PROPOSED"
    FOUNDER_SUPPLIED = "FOUNDER_SUPPLIED"


class CriticVerdict(StrEnum):
    ACCEPT = "ACCEPT"
    REVISE = "REVISE"


class CriticDimensionResult(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"


class ExecutionReadiness(StrEnum):
    CAN_EXECUTE = "CAN_EXECUTE"
    NEEDS_HELP = "NEEDS_HELP"
    BLOCKED = "BLOCKED"


class EvidenceSourceType(StrEnum):
    FOUNDER_PROVIDED = "FOUNDER_PROVIDED"
    EXTERNALLY_RETRIEVED = "EXTERNALLY_RETRIEVED"


class EvidenceResult(StrEnum):
    PENDING = "PENDING"
    ANALYZED = "ANALYZED"


# ---------------------------------------------------------------------------
# Milestone 4: Evidence Intake, Analysis, and Learning Enums
# ---------------------------------------------------------------------------

class EvidenceType(StrEnum):
    INTERVIEW_NOTES = "INTERVIEW_NOTES"
    OBSERVATION = "OBSERVATION"
    EXPERIMENT_RESULT = "EXPERIMENT_RESULT"
    USAGE_DATA = "USAGE_DATA"
    COMMITMENT = "COMMITMENT"
    PAYMENT = "PAYMENT"
    OUTREACH_RESULT = "OUTREACH_RESULT"
    PROTOTYPE_TEST_RESULT = "PROTOTYPE_TEST_RESULT"
    FOUNDER_REPORT = "FOUNDER_REPORT"
    OTHER = "OTHER"


class EvidenceProvenance(StrEnum):
    FOUNDER_REPORTED = "FOUNDER_REPORTED"
    FOUNDER_UPLOADED = "FOUNDER_UPLOADED"
    SYSTEM_DERIVED = "SYSTEM_DERIVED"


class EvidenceSubmissionSufficiency(StrEnum):
    SUFFICIENT_FOR_ANALYSIS = "SUFFICIENT_FOR_ANALYSIS"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"
    INSUFFICIENT_BUT_ANALYZABLE = "INSUFFICIENT_BUT_ANALYZABLE"


class EvidenceAnalysisConclusion(StrEnum):
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    INCONCLUSIVE = "INCONCLUSIVE"


class EvidenceStrength(StrEnum):
    WEAK = "WEAK"
    MODERATE = "MODERATE"
    STRONG = "STRONG"


class ExecutionQuality(StrEnum):
    GOOD = "GOOD"
    PARTIAL = "PARTIAL"
    POOR = "POOR"


# ---------------------------------------------------------------------------
# Milestone 4.0.4: Grounding & Experiment Recovery Integrity Enums
# ---------------------------------------------------------------------------

class ExperimentDesignStatus(StrEnum):
    READY = "READY"
    REVISION_REQUIRED = "REVISION_REQUIRED"
    QUALITY_BAR_NOT_MET = "QUALITY_BAR_NOT_MET"


class ConversationalResponseMode(StrEnum):
    BRIEF_CONTINUE = "BRIEF_CONTINUE"
    MATERIAL_UPDATE = "MATERIAL_UPDATE"
    DIRECT_ANSWER = "DIRECT_ANSWER"
    CONCERN_RESPONSE = "CONCERN_RESPONSE"
    CORRECTION_RESPONSE = "CORRECTION_RESPONSE"
    HANDOFF_TO_TEST = "HANDOFF_TO_TEST"


# ---------------------------------------------------------------------------
# Milestone 4.0.6: Experiment Failure Recovery Enums
# ---------------------------------------------------------------------------

class NextLearningActionType(StrEnum):
    VALIDATED_EXPERIMENT = "VALIDATED_EXPERIMENT"
    FALLBACK_LEARNING_ACTION = "FALLBACK_LEARNING_ACTION"






