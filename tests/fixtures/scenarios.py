"""
Canonical founder scenarios for Startary Milestone 1, Milestone 2, and Milestone 2.5 testing.

Milestone 2.5 Scenarios:
  SCENARIO A: Extremely Sparse
  SCENARIO B: Mynera Sparse & Multi-turn
  SCENARIO C: Evidence-Rich Enough
  SCENARIO D: Strong Product Detail, No Problem
  SCENARIO E: Unsupported Pricing Claim
  SCENARIO F: Known Problem, Missing Revenue (Verifies Unknown != Insufficient)
"""

from __future__ import annotations

from typing import Any

from startary.domain.area_knowledge import AreaStatus
from startary.domain.assessment import RiskFactorLevel
from startary.domain.reality_check import EvidenceConclusion
from startary.domain.taxonomy import StartupArea

# ---------------------------------------------------------------------------
# Canonical Scenarios
# ---------------------------------------------------------------------------

SCENARIO_A_EXTREMELY_SPARSE = "I have an AI startup idea."

SCENARIO_B_MYNERA_SPARSE = (
    "Mynera is an AI tool that helps students and freshmen to find the correct IT branch for their future career."
)

MYNERA_ANSWER_1 = (
    "Many students choose IT because it is popular but do not understand the different branches. "
    "Some later realize that the area they chose does not match what they actually enjoy or want to do."
)

MYNERA_ANSWER_2 = (
    "I experienced this myself and talked with around 12 university students. "
    "Several told me they were confused between areas such as software engineering, cybersecurity and data science."
)

SCENARIO_C_EVIDENCE_RICH = (
    "We interviewed 20 first-time founders. Fifteen said they lose several hours each week "
    "deciding what startup task to work on next. Most currently ask mentors, ChatGPT, or search online. "
    "We do not know whether they would repeatedly use a structured startup guidance product."
)

SCENARIO_D_PRODUCT_NO_PROBLEM = (
    "We built an AI dashboard with CRM integration, automated reporting, voice controls, "
    "analytics, and Slack integration."
)

SCENARIO_E_PRICING_ONLY = "Small businesses will definitely pay $100 per month for this."

SCENARIO_F_KNOWN_PROBLEM_MISSING_REVENUE = (
    "We interviewed 25 university students choosing an IT specialization. 18 said they spent weeks "
    "comparing branches and felt unable to tell which direction matched their career goals. "
    "They currently rely mainly on YouTube, friends, and generic career tests. We have not explored pricing yet."
)

# For backward compatibility
CASE_1_SPARSE = "I want to make an AI app for restaurants."
CASE_2_AMBIGUITY = (
    "I originally thought university students were the users, but after talking to a few people "
    "I think accelerator founders may have the bigger problem. I'm not sure."
)
CASE_3_PRICING_CLAIM = "Every small business would pay $100 per month for this."
CASE_4_WEAK_ACQUISITION = (
    "We interviewed 20 founders and 15 reported this problem weekly. We built a landing page, "
    "but we don't know how to consistently reach founders beyond our university network."
)
CASE_5_FEATURE_HEAVY = (
    "We built dashboards, AI summaries, CRM integrations, voice controls and automated reporting."
)
CASE_6_INCONCLUSIVE = (
    "We interviewed 10 target users. Five strongly had the problem, three did not care, "
    "and two were unsure."
)

CASE_A_SPARSE = CASE_1_SPARSE
CASE_B_RICH = (
    "I am building a platform for student founders who get stuck deciding what "
    "startup task to do next. I have interviewed 8 student founders. Six described "
    "spending several days switching between random advice and AI tools, feeling lost "
    "about their priorities. I am a technical student founder with two teammates and "
    "almost no marketing budget. We have no revenue yet and haven't built anything, "
    "but I've seen this problem myself as a student founder."
)
CASE_C_CLAIM = CASE_3_PRICING_CLAIM
CASE_D_CONTRADICTORY = CASE_2_AMBIGUITY
CASE_E_SOLUTION_HEAVY = CASE_5_FEATURE_HEAVY

ALL_SCENARIOS = {
    "A_sparse": CASE_A_SPARSE,
    "B_rich": CASE_B_RICH,
    "C_claim": CASE_C_CLAIM,
    "D_contradictory": CASE_D_CONTRADICTORY,
    "E_solution_heavy": CASE_E_SOLUTION_HEAVY,
}

M2_SCENARIOS = {
    "case_1_sparse": CASE_1_SPARSE,
    "case_2_ambiguity": CASE_2_AMBIGUITY,
    "case_3_pricing_claim": CASE_3_PRICING_CLAIM,
    "case_4_weak_acquisition": CASE_4_WEAK_ACQUISITION,
    "case_5_feature_heavy": CASE_5_FEATURE_HEAVY,
    "case_6_inconclusive": CASE_6_INCONCLUSIVE,
}

M2_5_SCENARIOS = {
    "scenario_a_sparse": SCENARIO_A_EXTREMELY_SPARSE,
    "scenario_b_mynera": SCENARIO_B_MYNERA_SPARSE,
    "scenario_c_rich": SCENARIO_C_EVIDENCE_RICH,
    "scenario_d_product_no_problem": SCENARIO_D_PRODUCT_NO_PROBLEM,
    "scenario_e_pricing_only": SCENARIO_E_PRICING_ONLY,
    "scenario_f_missing_revenue": SCENARIO_F_KNOWN_PROBLEM_MISSING_REVENUE,
}


# ---------------------------------------------------------------------------
# Mock Response Helpers for Tests
# ---------------------------------------------------------------------------

def make_minimal_valid_llm_output_dict() -> dict:
    """Minimal valid DiagnosisLLMOutput dict for unit tests."""
    return {
        "area_assessments": [
            {
                "area": StartupArea.PROBLEM,
                "status": AreaStatus.PARTIALLY_KNOWN,
                "understanding": "Founder wants to build an AI app for restaurants.",
                "founder_claims": ["Restaurants need AI tools"],
                "evidence_provided": [],
                "missing_information": "What specific problem restaurants face.",
            },
        ],
        "clear_areas": [],
        "unclear_areas": [],
        "missing_areas": [
            StartupArea.TARGET_USER,
            StartupArea.PAIN_URGENCY,
            StartupArea.EXISTING_ALTERNATIVES,
            StartupArea.VALUE_PROPOSITION,
            StartupArea.DEMAND_SIGNAL,
            StartupArea.ACQUISITION,
            StartupArea.BUSINESS_MODEL,
            StartupArea.SOLUTION_MVP,
            StartupArea.FOUNDER_CONSTRAINTS,
        ],
        "unsupported_claims": ["Restaurants need AI tools"],
        "diagnosis_summary": "Founder stated intent to build an AI app for restaurants.",
        "startup_stage": "raw_idea",
    }


def make_rich_valid_llm_output_dict() -> dict:
    """Rich valid DiagnosisLLMOutput dict."""
    return {
        "area_assessments": [
            {
                "area": StartupArea.PROBLEM,
                "status": AreaStatus.PARTIALLY_KNOWN,
                "understanding": "Student founders struggle to decide what startup task to prioritize next.",
                "founder_claims": [],
                "evidence_provided": [
                    "8 student founder interviews conducted",
                    "6 of 8 described spending several days lost between advice and AI tools",
                ],
                "missing_information": "Frequency and cost of this confusion.",
            },
            {
                "area": StartupArea.TARGET_USER,
                "status": AreaStatus.PARTIALLY_KNOWN,
                "understanding": "Student founders who are early in their startup journey.",
                "founder_claims": [],
                "evidence_provided": ["8 student founders interviewed"],
                "missing_information": "More specific segment definition within student founders.",
            },
        ],
        "clear_areas": [StartupArea.PROBLEM],
        "unclear_areas": [],
        "missing_areas": [
            StartupArea.PAIN_URGENCY,
            StartupArea.EXISTING_ALTERNATIVES,
            StartupArea.VALUE_PROPOSITION,
            StartupArea.DEMAND_SIGNAL,
            StartupArea.ACQUISITION,
            StartupArea.BUSINESS_MODEL,
            StartupArea.SOLUTION_MVP,
            StartupArea.FOUNDER_CONSTRAINTS,
        ],
        "unsupported_claims": [],
        "diagnosis_summary": "Problem and target user partially supported by 8 interviews.",
        "startup_stage": "problem_exploration",
    }


def make_valid_context_sufficiency_output_dict(
    sufficient: bool = True,
    target_area: StartupArea = StartupArea.PROBLEM,
    grounding_quality: str | None = None,
    would_one_more_answer_materially_change_direction: bool | None = None,
) -> dict:
    """Mock valid ContextSufficiencyLLMOutput dict."""
    g_quality = grounding_quality or ("STRONG" if sufficient else "WEAK")
    one_more = would_one_more_answer_materially_change_direction if would_one_more_answer_materially_change_direction is not None else (not sufficient)
    return {
        "sufficient": sufficient,
        "confidence": RiskFactorLevel.HIGH,
        "grounding_quality": g_quality,
        "would_one_more_answer_materially_change_direction": one_more,
        "reason": "Sufficient problem context provided." if sufficient else "Sparse context without observable problem friction.",
        "blocking_unknowns": [] if sufficient else ["Specific difficulties or confusion experienced by users."],
        "usable_grounding": ["Target audience is students."] if sufficient else [],
        "prohibited_inferences": [] if sufficient else ["Unstated career matching algorithms or preferences."],
        "recommended_question_area": target_area,
        "information_goal": "" if sufficient else "Clarify specific observable friction faced by target users.",
    }


def make_valid_clarifying_question_output_dict(
    area: StartupArea = StartupArea.PROBLEM,
) -> dict:
    """Mock valid ClarifyingQuestionLLMOutput dict."""
    return {
        "question": "When you say students struggle to find the correct IT branch, what specific difficulties or confusion have you observed them experiencing?",
        "target_area": area,
        "information_goal": "Clarify specific observable friction faced by target users.",
        "why_this_question_now": "Grounding the problem is the primary blocker preventing a testable critical assumption.",
    }


def make_valid_area_assessments_batch_output_dict(top_area: StartupArea = StartupArea.PROBLEM) -> dict:
    """Mock valid StartupAreaAssessmentBatchLLMOutput dict."""
    assessments = []
    for area in StartupArea:
        is_top = (area == top_area)
        assessments.append({
            "area": area,
            "importance": RiskFactorLevel.HIGH if is_top else RiskFactorLevel.LOW,
            "uncertainty": RiskFactorLevel.HIGH if is_top else RiskFactorLevel.LOW,
            "evidence_gap": RiskFactorLevel.HIGH if is_top else RiskFactorLevel.LOW,
            "consequence_of_being_wrong": RiskFactorLevel.HIGH if is_top else RiskFactorLevel.LOW,
            "rationale": f"Risk assessment for {area}.",
        })
    return {"assessments": assessments}


def make_valid_reality_check_output_dict(
    area: StartupArea = StartupArea.PROBLEM,
    conclusion: EvidenceConclusion = EvidenceConclusion.NO_EVIDENCE,
) -> dict:
    """Mock valid RealityCheckLLMOutput dict."""
    return {
        "weak_link_area": area,
        "relevant_founder_claims": ["Founder claims students need IT branch guidance."],
        "available_founder_evidence": [],
        "what_evidence_supports": [],
        "what_evidence_does_not_support": ["Claims of career matching demand."],
        "conclusion": conclusion,
        "unresolved_uncertainty": "Whether students experience severe decision paralysis.",
        "reality_check_summary": "Zero evidence was provided to support the founder's thesis.",
    }


def make_valid_critical_assumption_output_dict(area: StartupArea = StartupArea.PROBLEM) -> dict:
    """Mock valid CriticalAssumptionLLMOutput dict."""
    return {
        "area": area,
        "statement": "University freshmen in IT experience branch selection confusion severe enough to actively use a guided decision tool.",
        "rationale": "If students do not experience this friction, the product has no core value proposition.",
        "what_falsifies_it": "Discovery interviews showing freshmen pick their branch without hesitation or regret.",
    }


def make_valid_gap_plan_output_dict(
    should_ask: bool = False,
    strategic_sufficient: bool = True,
    question: str | None = None,
    target_domain: str | None = None,
    priority: str = "USEFUL_LATER",
    information_goal: str = "",
    reason: str = "Context is sufficiently grounded to proceed.",
) -> dict:
    """Mock valid InformationGapPlanLLMOutput dict for M2.6 tests."""
    return {
        "should_ask": should_ask,
        "strategic_sufficient": strategic_sufficient,
        "target_domain": target_domain or ("STRATEGIC" if should_ask else None),
        "target_field_or_area": "Problem" if should_ask else None,
        "priority": priority,
        "information_goal": information_goal,
        "reason": reason,
        "question": question or ("Can you describe what specific difficulty users face?" if should_ask else None),
    }


# ---------------------------------------------------------------------------
# M2.6 Scenarios (Scenarios A–I from spec)
# ---------------------------------------------------------------------------

SCENARIO_M26_A_RICH_PROFILE = (
    "We are three students building this — two of us code and one does design. "
    "We can spend about 10 hours each per week. We don't have an MVP yet "
    "and can spend around $200 a month. "
    "We are building a tool to help IT students pick their specialization."
)

SCENARIO_M26_B_MVP_FALSE_ONLY = (
    "We don't have an MVP yet. We're exploring the problem space."
)

SCENARIO_M26_C_USERS_NO_REVENUE = (
    "We launched a beta version. We have 50 active users testing it. No revenue yet."
)

SCENARIO_M26_D_REVENUE_NO_USER_COUNT = (
    "We have two paying customers — a small SaaS deal, about $300 per month. "
    "Not sure how many total active users we have."
)

SCENARIO_M26_E_TEAM_UPDATE = (
    "I started alone, but now a friend joined me as a co-founder to handle the business side."
)

SCENARIO_M26_F_AMBIGUOUS_TEAM = (
    "There are three people helping with the project, but only two of us are really committed full-time."
)

SCENARIO_M26_G_GEOGRAPHY_EXPLICIT = (
    "We're targeting the German market specifically because of the vocational training system there."
)

SCENARIO_M26_H_EXISTING_USER_ACCESS = (
    "I run a Telegram community of 800 IT students. That's my main access to users."
)

SCENARIO_M26_I_STRATEGIC_REVEALS_PROFILE = (
    "I've done 20 interviews with students who are choosing between data science and backend engineering. "
    "I'm a final-year CS student myself. We're a team of 2."
)

M2_6_SCENARIOS = {
    "m26_a_rich_profile": SCENARIO_M26_A_RICH_PROFILE,
    "m26_b_mvp_false_only": SCENARIO_M26_B_MVP_FALSE_ONLY,
    "m26_c_users_no_revenue": SCENARIO_M26_C_USERS_NO_REVENUE,
    "m26_d_revenue_no_user_count": SCENARIO_M26_D_REVENUE_NO_USER_COUNT,
    "m26_e_team_update": SCENARIO_M26_E_TEAM_UPDATE,
    "m26_f_ambiguous_team": SCENARIO_M26_F_AMBIGUOUS_TEAM,
    "m26_g_geography_explicit": SCENARIO_M26_G_GEOGRAPHY_EXPLICIT,
    "m26_h_existing_user_access": SCENARIO_M26_H_EXISTING_USER_ACCESS,
    "m26_i_strategic_reveals_profile": SCENARIO_M26_I_STRATEGIC_REVEALS_PROFILE,
}


# ---------------------------------------------------------------------------
# Milestone 2.6.1 Helpers
# ---------------------------------------------------------------------------

def make_valid_baseline_sufficiency_output_dict(
    is_sufficient: bool = True,
    missing_groups: list[str] | None = None,
) -> dict[str, Any]:
    missing = missing_groups or ([] if is_sufficient else ["TEAM"])
    status = "KNOWN_ENOUGH" if is_sufficient else "MISSING"
    return {
        "is_sufficient": is_sufficient,
        "team_status": "MISSING" if "TEAM" in missing else "KNOWN_ENOUGH",
        "capacity_status": "MISSING" if "CAPACITY" in missing else "KNOWN_ENOUGH",
        "product_stage_status": "MISSING" if "PRODUCT_STAGE" in missing else "KNOWN_ENOUGH",
        "validation_status": "MISSING" if "VALIDATION_OR_USER" in missing else "KNOWN_ENOUGH",
        "resource_status": "MISSING" if "RESOURCE" in missing else "KNOWN_ENOUGH",
        "missing_groups": missing,
        "reason": "Baseline sufficiency evaluation" if is_sufficient else "Missing baseline operational dimensions",
    }


def make_valid_baseline_question_plan_output_dict(
    should_ask: bool = True,
    target_group: str = "TEAM",
    question: str = "Are you building this alone or with a team, and what does each person mainly work on?",
) -> dict[str, Any]:
    return {
        "should_ask": should_ask,
        "target_group": target_group,
        "target_fields": ["active_team_size", "member_roles"],
        "question": question if should_ask else None,
        "reason": f"Asking about {target_group} to fill operational baseline context.",
    }


def make_valid_weak_link_discrimination_output_dict(
    selection_confidence: str = "HIGH",
    competing_areas: list[StartupArea] | None = None,
    should_clarify: bool = False,
) -> dict[str, Any]:
    return {
        "selection_confidence": selection_confidence,
        "competing_areas": competing_areas or [],
        "discrimination_reason": "Clear score separation between leading candidate and competitors." if not should_clarify else "Multiple areas share maximum risk score.",
        "missing_discriminating_information": "" if not should_clarify else "Observation of actual user behavior under problem conditions",
        "should_clarify": should_clarify,
        "recommended_information_goal": "Separate problem versus demand risk",
    }


def make_valid_weak_link_clarifying_question_output_dict(
    question: str = "When students face this confusion, what do they actually do about it — delay choosing, spend money on private tutors, or actively search for guidance tools?",
    competing_areas: list[StartupArea] | None = None,
) -> dict[str, Any]:
    return {
        "question": question,
        "competing_areas": competing_areas or [StartupArea.PROBLEM, StartupArea.DEMAND_SIGNAL],
        "information_goal": "Observe whether students take costly action or passive delay.",
        "why_this_question_now": "Directly separates problem severity from active demand signal.",
    }


def make_valid_experiment_generator_output_dict(
    experiment_type: str = "PROBLEM_INTERVIEW",
    hypothesis: str = "Restaurants actively struggle to track food inventory accurately.",
) -> dict[str, Any]:
    return {
        "experiment_type": experiment_type,
        "objective": "Verify if restaurant owners experience painful inventory tracking errors.",
        "hypothesis": hypothesis,
        "method": "Conduct 6-8 structured retrospective interviews with independent restaurant managers.",
        "target_participants_or_scope": "Independent restaurant managers with 10-50 daily seatings.",
        "founder_actions": [
            "Identify 10 local restaurant managers within personal or professional network.",
            "Schedule 6-8 in-person or video interviews focused on recent inventory workflows.",
            "Record whether inventory tracking caused actual food waste or unexpected costs last month.",
        ],
        "evidence_to_collect": "Detailed notes on specific incidents of inventory errors, estimated dollar cost, and current workarounds.",
        "support_signal": "At least 5 of 8 managers describe concrete financial losses or missed orders due to inventory inaccuracies in the last 30 days.",
        "contradiction_signal": "Fewer than 2 managers recall any painful inventory discrepancies, or view current spreadsheets as completely adequate.",
        "inconclusive_signal": "Managers express vague general agreement but cannot cite a single concrete incident in the past 30 days.",
        "valid_inferences": [
            "Whether inventory tracking is an active, recognized pain point for independent restaurants.",
            "What manual workarounds or tools they currently rely on.",
        ],
        "invalid_inferences": [
            "Whether restaurants will pay $100/mo for an AI automation app.",
            "Long-term customer retention or scalability of sales channels.",
        ],
        "estimated_time": "3-5 days",
        "estimated_cost": "Near-zero / $0",
        "required_capabilities": ["Structured interviewing", "Cold outreach or network access"],
        "required_access": ["Access to local restaurant owners or managers"],
        "parameters": [
            {
                "name": "sample_size",
                "value": "6-8 managers",
                "provenance": "AI_PROPOSED",
                "notes": "Provisional sample size standard for exploratory problem interviews.",
            },
            {
                "name": "support_threshold",
                "value": "at least 5 of 8",
                "provenance": "AI_PROPOSED",
                "notes": "Provisional decision rule for problem signal.",
            },
        ],
    }


def make_valid_experiment_critic_output_dict(
    verdict: str = "ACCEPT",
    issues: list[str] | None = None,
    revision_instructions: str = "",
) -> dict[str, Any]:
    is_accept = (verdict == "ACCEPT")
    dim_res = "PASS" if is_accept else "FAIL"
    return {
        "verdict": verdict,
        "issues": issues or ([] if is_accept else ["Experiment tests willingness to pay rather than problem severity."]),
        "assumption_alignment": "PASS" if is_accept else dim_res,
        "evidence_validity": "PASS" if is_accept else dim_res,
        "feasibility": "PASS",
        "cost_efficiency": "PASS",
        "falsifiability": "PASS",
        "scope": "PASS",
        "revision_instructions": revision_instructions if not is_accept else "",
        "rationale": "Experiment directly tests what must be true at near-zero cost under current founder constraints." if is_accept else "Requires revision to tighten evidence validity.",
    }


def make_valid_diagnosis_snapshot_output_dict(
    stage: str = "Early Build / Pre-MVP",
    weak_link: str = "Problem",
    critical_assumption: str = "Restaurant managers experience recurring friction in scheduling.",
) -> dict[str, Any]:
    return {
        "stage": stage,
        "stage_explanation": "The startup is currently in early build where core market assumptions are untested.",
        "grounded_areas": [
            "Solo founder with ~15 hours/week capacity",
            "Initial wireframes built but no working software",
            "Direct personal access to 2 local restaurant managers",
        ],
        "current_advantages": [
            "Direct access to relevant target businesses",
            "In-house technical capability to build prototypes",
        ],
        "execution_constraints": [
            "Solo founder",
            "~$200 personal testing budget",
            "Limited weekly capacity",
        ],
        "major_uncertainties": [
            "Whether scheduling friction is severe enough to warrant a dedicated tool",
            "Whether current spreadsheet workarounds already solve the need",
            "Whether the problem generalizes beyond personal contacts",
        ],
        "weak_link": weak_link,
        "weak_link_reason": f"{weak_link} is currently the primary bottleneck with highest decision risk.",
        "weak_link_confidence": "LOW",
        "weak_link_confidence_explanation": "Several strategic areas remain highly uncertain, so this is the best current priority.",
        "competing_risks": ["Demand Signal", "Target User"],
        "reality_summary": "Two restaurant managers confirmed scheduling disruptions informally, but no documented audits exist.",
        "recommended_learning_goal": "Determine whether independent restaurant managers repeatedly experience scheduling disruption.",
        "critical_assumption": critical_assumption,
    }


def make_valid_founder_turn_interpretation_output_dict(
    intent_types: list[str] | None = None,
    direct_answer: bool = True,
    direct_question: str | None = None,
    concern_or_goal: str | None = None,
    correction: str | None = None,
    cannot_answer_now: bool = False,
    cannot_act_now: bool = False,
    response_priority: str = "NORMAL",
) -> dict[str, Any]:
    return {
        "intent_types": intent_types or ["ANSWER"],
        "direct_answer": direct_answer,
        "direct_question": direct_question,
        "concern_or_goal": concern_or_goal,
        "correction": correction,
        "cannot_answer_now": cannot_answer_now,
        "cannot_act_now": cannot_act_now,
        "uncertainty_statement": None,
        "new_evidence_summary": None,
        "founder_preference_or_constraint": None,
        "response_priority": response_priority,
        "interpretation_summary": "Founder answered the pending question.",
    }


def make_valid_advisor_response_output_dict(
    response_text: str = "Understood. What budget do you have available for testing?",
    next_question: str = "What budget do you have available for testing?",
    next_question_target: str = "Founder Constraints",
    question_id: str = "q_test_123",
    should_ask_next_question: bool = True,
) -> dict[str, Any]:
    return {
        "question_id": question_id,
        "next_question_target": next_question_target,
        "next_question_rephrased": next_question,
        "acknowledge": "Understood.",
        "answer_to_founder": None,
        "diagnosis_delta": None,
        "next_question": next_question,
        "should_ask_next_question": should_ask_next_question,
        "response_text": response_text,
    }



