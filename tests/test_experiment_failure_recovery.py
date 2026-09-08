"""
Tests for Milestone 4.0.6 — Experiment Failure Recovery Action.

Covers:
1. FallbackLearningAction validates and serializes required fields.
2. Fallback is created only when experiment_design_status == QUALITY_BAR_NOT_MET.
3. Accepted experiment never creates fallback.
4. Fallback preserves active critical assumption and diagnosis intact.
5. Fallback does not mark rejected experiment accepted or READY.
6. Fallback action contains no formal pass/fail decision thresholds.
7. ShelfSense scenario receives actionable, conservative learning step (grocery store owner interviews on stockout/waste incidents).
8. Returned fallback observations enrich startup state without automatically marking the assumption SUPPORTED.
9. Founder questions about fallback ("Who should I talk to?", "Can I start with 3 shops?", etc.) are answered directly from FallbackLearningAction.
10. Terminal state renders EXPERIMENT_DESIGN_NEEDS_REVISION with fallback banner.
"""

from __future__ import annotations

import pytest

from startary.domain.enums import (
    CriticDimensionResult,
    CriticVerdict,
    ExecutionReadiness,
    ExperimentDesignStatus,
    ExperimentType,
    NextLearningActionType,
    WorkflowPhase,
)
from startary.domain.experiment import (
    ExecutionHandoff,
    Experiment,
    FallbackLearningAction,
    FounderExperimentSummary,
    compose_fallback_learning_action,
    render_fallback_learning_action_text,
)
from startary.domain.terminal_state import (
    TerminalStatus,
    render_terminal_state,
    resolve_founder_facing_status,
)
from startary.graph.nodes import (
    compose_execution_handoff,
    handle_experiment_failure_stop,
    respond_to_waiting_founder,
)


def test_fallback_learning_action_model_validates():
    """1. FallbackLearningAction validates and serializes required fields."""
    action = FallbackLearningAction(
        assumption_id="ca-001",
        target_area="Problem",
        learning_goal="Whether independent grocery stores suffer frequent inventory stockouts.",
        why_fallback_is_needed="Experiment critic rejected proposed experiment twice.",
        primary_action="Talk with 5–8 independent grocery store owners.",
        who_to_engage="5–8 independent grocery store owners",
        what_to_ask_or_observe=[
            "Ask for recent stockout event",
            "Ask what consequences occurred",
        ],
        what_to_bring_back=[
            "Incident notes",
            "Observed consequences",
        ],
        what_not_to_infer=[
            "Willingness to pay",
            "Need for AI or software",
        ],
        completion_instruction="Return with notes from those conversations.",
        estimated_effort="3–5 hours across 5–8 conversations",
        participant_target="Independent grocery store owners",
    )

    data = action.model_dump()
    assert data["assumption_id"] == "ca-001"
    assert data["target_area"] == "Problem"
    assert data["action_type"] == NextLearningActionType.FALLBACK_LEARNING_ACTION.value
    assert "Talk with 5–8" in data["primary_action"]
    assert len(data["what_to_ask_or_observe"]) == 2
    assert len(data["what_to_bring_back"]) == 2
    assert len(data["what_not_to_infer"]) == 2
    assert data["estimated_effort"] == "3–5 hours across 5–8 conversations"


def test_fallback_created_only_when_quality_bar_not_met():
    """2. Fallback is created only when experiment_design_status == QUALITY_BAR_NOT_MET."""
    state = {
        "project_id": "proj-m406-02",
        "critic_revision_count": 2,
        "active_assumption": {
            "id": "ca-002",
            "statement": "Independent grocery store owners lose money from unexpected stockouts.",
            "area": "Problem",
        },
        "weak_area": "Problem",
        "experiment_critic_output": {
            "verdict": "REVISE",
            "issues": ["Proxy metric does not measure commercial intent."],
        },
    }

    res = handle_experiment_failure_stop(state)

    assert res["experiment_design_status"] == ExperimentDesignStatus.QUALITY_BAR_NOT_MET.value
    assert res["learning_input_mode"] == "FALLBACK"
    assert "fallback_learning_action" in res
    fallback = res["fallback_learning_action"]
    assert fallback["action_type"] == NextLearningActionType.FALLBACK_LEARNING_ACTION.value
    assert fallback["assumption_id"] == "ca-002"
    assert fallback["target_area"] == "Problem"


def test_accepted_experiment_never_creates_fallback():
    """3. Accepted experiment never creates fallback."""
    exp = Experiment(
        id="exp-accepted-01",
        project_id="proj-accepted",
        assumption_id="asmp-acc-01",
        target_area="Problem",
        type=ExperimentType.CUSTOMER_INTERVIEW,
        title="Store Owner Discovery",
        objective="Validate customer problem",
        hypothesis="Store owners experience stockouts regularly.",
        description="Interview store owners.",
        target_participants_or_context="Boutique store owners",
        evidence_to_collect="Return frequencies and estimated costs",
        support_signal="At least 5 owners report recurring friction",
        contradiction_signal="Fewer than 2 owners report friction",
        inconclusive_signal="Ambiguous responses",
        founder_actions=["Interview 6 store owners"],
        estimated_cost_hours=5.0,
        estimated_duration_days=3,
        execution_readiness=ExecutionReadiness.CAN_EXECUTE,
    )
    summary = FounderExperimentSummary(
        what_are_we_testing="Whether boutique store owners experience returns as high friction.",
        what_should_i_do="Interview 6 store owners.",
        what_should_i_bring_back="Return logs and friction notes.",
        useful_signal="Owners report recurring returns.",
        current_positive_signals=["One owner confirmed the problem."],
    )
    state = {
        "project_id": "proj-accepted",
        "active_experiment": exp.model_dump(),
        "active_assumption": {"id": "asmp-acc-01", "statement": "Store owners experience stockouts."},
        "founder_experiment_summary": summary.model_dump(),
    }

    res = compose_execution_handoff(state)

    assert res.get("fallback_learning_action") is None
    assert res.get("learning_input_mode") != "FALLBACK"
    assert "active_execution_handoff" in res
    assert res["active_execution_handoff"]["action_type"] == NextLearningActionType.VALIDATED_EXPERIMENT.value


def test_fallback_preserves_active_critical_assumption_and_diagnosis():
    """4. Fallback preserves active critical assumption and diagnosis intact."""
    ca_stmt = "Independent boutique merchants lose over 5 hours/week processing returns."
    state = {
        "project_id": "proj-preserve-01",
        "phase": WorkflowPhase.TEST,
        "weak_area": "Problem",
        "active_assumption": {"id": "ca-preserve-01", "statement": ca_stmt, "area": "Problem"},
        "diagnosis_snapshot": {
            "id": "snap-preserve-01",
            "weak_link": "Problem",
            "stage": "PRE_MVP",
            "critical_assumption": ca_stmt,
        },
        "experiment_critic_output": {"issues": ["Invalid proxy metric used for validation."]},
    }

    updates = handle_experiment_failure_stop(state)
    merged = {**state, **updates}

    # Diagnosis snapshot and weak link are preserved
    assert merged["weak_area"] == "Problem"
    assert merged["diagnosis_snapshot"]["weak_link"] == "Problem"
    assert merged["diagnosis_snapshot"]["critical_assumption"] == ca_stmt
    # Active critical assumption is preserved
    assert merged["active_assumption"]["id"] == "ca-preserve-01"
    assert merged["active_assumption"]["statement"] == ca_stmt
    # Fallback action links to active critical assumption
    assert merged["fallback_learning_action"]["assumption_id"] == "ca-preserve-01"
    assert merged["fallback_learning_action"]["target_area"] == "Problem"


def test_fallback_does_not_mark_rejected_experiment_accepted_or_ready():
    """5. Fallback does not mark rejected experiment accepted or READY."""
    state = {
        "phase": WorkflowPhase.TEST,
        "experiment_design_status": ExperimentDesignStatus.QUALITY_BAR_NOT_MET.value,
        "active_experiment": {
            "id": "exp-rejected-01",
            "execution_readiness": ExecutionReadiness.BLOCKED.value,
        },
        "experiment_critic_output": {
            "verdict": CriticVerdict.REVISE.value,
            "issues": ["Decision rule requires top-three priority."],
        },
        "status_message": "Experiment generation stopped: unable to produce an accepted experiment.",
    }

    updates = handle_experiment_failure_stop(state)
    merged = {**state, **updates}

    status = resolve_founder_facing_status(merged)
    assert status == TerminalStatus.EXPERIMENT_DESIGN_NEEDS_REVISION
    assert status != TerminalStatus.EXPERIMENT_READY
    assert status != TerminalStatus.WAITING_FOR_EVIDENCE
    assert status != TerminalStatus.INITIAL_DIAGNOSIS_COMPLETE
    # Experiment was NOT updated to CAN_EXECUTE
    assert merged["active_experiment"]["execution_readiness"] == ExecutionReadiness.BLOCKED.value


def test_fallback_action_contains_no_formal_pass_fail_thresholds():
    """6. Fallback action contains no formal pass/fail decision thresholds."""
    ca = {
        "id": "ca-no-thresh",
        "statement": "Independent grocery store owners experience perishable stockouts.",
        "area": "Problem",
    }
    critic = {
        "verdict": "REVISE",
        "issues": ["Arbitrary numerical pass criteria."],
    }
    fallback = compose_fallback_learning_action(
        assumption=ca,
        weak_link="Problem",
        critic_output=critic,
    )

    data = fallback.model_dump()
    rendered = fallback.response_text

    # No numerical threshold language
    for forbidden in ["at least", "of 8", "pass threshold", "fail threshold", "decision rule", "support criterion"]:
        assert forbidden not in rendered.lower()

    # Contains clear non-inference boundaries
    assert len(data["what_not_to_infer"]) > 0
    assert any("willingness to pay" in item.lower() for item in data["what_not_to_infer"])
    assert any("ai" in item.lower() or "software" in item.lower() for item in data["what_not_to_infer"])

    # Explicitly identifies itself as learning, not validation
    assert "This action is for learning, not validation." in rendered


def test_shelfsense_scenario_fallback_learning_action():
    """7. ShelfSense scenario receives actionable, conservative learning step."""
    assumption = {
        "id": "asmp-shelfsense-01",
        "statement": "Validating critical assumption: Independent grocery store owners experience stockouts and spoilage as an urgent, costly problem.",
        "area": "Problem",
    }
    critic_output = {
        "verdict": "REVISE",
        "issues": [
            "Offering a free tally sheet is a weak proxy for willingness to adopt an automated shelf monitor",
            "The experiment lacks behavioral commitment",
        ],
    }
    rejected_exp = {
        "target_participants_or_scope": "5–8 independent grocery store owners in urban centers",
        "hypothesis": "Grocery store owners will adopt automated inventory tracking.",
    }

    fallback = compose_fallback_learning_action(
        assumption=assumption,
        weak_link="Problem",
        critic_output=critic_output,
        rejected_experiment=rejected_exp,
    )

    # Cleaned learning goal without generic prefix
    assert not fallback.learning_goal.lower().startswith("validating critical assumption:")
    assert "stockouts" in fallback.learning_goal.lower() or "grocery" in fallback.learning_goal.lower()

    # Grounded target participants
    assert "grocery store owners" in fallback.who_to_engage.lower()

    # Reconstructs concrete past events and real consequences
    q_text = " ".join(fallback.what_to_ask_or_observe).lower()
    assert "recent" in q_text
    assert "consequences" in q_text or "cost" in q_text or "overhead" in q_text
    assert "workaround" in q_text or "current" in q_text

    # Bring back concrete facts
    b_text = " ".join(fallback.what_to_bring_back).lower()
    assert "incident" in b_text or "consequence" in b_text or "workaround" in b_text

    # Strict non-inference boundaries
    n_text = " ".join(fallback.what_not_to_infer).lower()
    assert "willingness to pay" in n_text
    assert "ai" in n_text or "software" in n_text

    # Check rendering
    rendered = fallback.response_text
    assert "WHAT DO WE STILL NEED TO LEARN?" in rendered
    assert "WHAT SHOULD I DO NEXT?" in rendered
    assert "WHAT SHOULD I BRING BACK?" in rendered
    assert "WHAT SHOULD I NOT CONCLUDE YET?" in rendered


def test_returned_fallback_observations_enrich_startup_state():
    """8. Returned fallback observations enrich startup state without automatically marking the assumption SUPPORTED."""
    fallback_dict = {
        "assumption_id": "asmp-shelfsense-01",
        "target_area": "Problem",
        "learning_goal": "Whether independent grocers suffer perishable stockouts.",
        "who_to_engage": "5–8 independent grocery store owners",
        "what_to_ask_or_observe": ["Ask for a recent stockout incident."],
        "what_to_bring_back": ["One concrete recent incident."],
        "what_not_to_infer": ["Willingness to pay for AI software."],
    }
    msg = "I spoke to 4 store owners today. 3 said expired dairy costs them $150 every Monday."
    state = {
        "phase": WorkflowPhase.WAITING_FOR_LEARNING_INPUT,
        "learning_input_mode": "FALLBACK",
        "fallback_learning_action": fallback_dict,
        "active_assumption": {
            "id": "asmp-shelfsense-01",
            "status": "ACTIVE",
            "statement": "Independent grocery store owners experience stockouts as an urgent problem.",
        },
        "startup_state": {
            "areas": {
                "Problem": {"evidence_provided": []},
            },
        },
        "founder_response": msg,
        "founder_turn": msg,
        "founder_messages": [
            {"role": "founder", "content": msg},
        ],
    }

    res = respond_to_waiting_founder(state)

    # Captured observations
    assert "I have captured these notes into your startup state" in res["waiting_response"]
    assert "without prematurely marking the assumption validated" in res["waiting_response"]
    assert res["startup_state"]["areas"]["Problem"]["evidence_provided"] == [msg]
    assert len(res["founder_evidence"]) == 1
    assert res["founder_evidence"][0]["source"] == "FALLBACK_LEARNING_ACTION"

    # Assumption is NOT automatically validated or marked SUPPORTED
    asmp = state["active_assumption"]
    assert asmp.get("status") != "SUPPORTED"
    assert asmp.get("status") == "ACTIVE"


def test_founder_fallback_questions_answered_directly_from_action():
    """9. Founder questions about fallback are answered directly from FallbackLearningAction."""
    fallback_dict = {
        "who_to_engage": "5–8 independent grocery store owners in urban bodegas",
        "what_to_ask_or_observe": [
            "Ask for one recent expired inventory incident",
            "Ask what costs were incurred",
        ],
        "what_to_bring_back": [
            "One concrete incident per store",
            "Specific dollar cost reported",
        ],
        "what_not_to_infer": [
            "Willingness to pay for automated software",
            "That interest represents commercial demand",
        ],
    }
    base_state = {
        "phase": WorkflowPhase.WAITING_FOR_LEARNING_INPUT,
        "learning_input_mode": "FALLBACK",
        "fallback_learning_action": fallback_dict,
    }

    # Query 1: Who should I talk to?
    res1 = respond_to_waiting_founder({
        **base_state,
        "founder_response": "Who should I talk to?",
        "founder_messages": [{"role": "founder", "content": "Who should I talk to?"}],
    })
    assert "5–8 independent grocery store owners in urban bodegas" in res1["waiting_response"]

    # Query 2: Can I start with 3 shops?
    res2 = respond_to_waiting_founder({
        **base_state,
        "founder_response": "Can I start with 3 shops?",
        "founder_messages": [{"role": "founder", "content": "Can I start with 3 shops?"}],
    })
    assert "Starting with 2–3 conversations is an excellent way" in res2["waiting_response"]

    # Query 3: What questions should I ask?
    res3 = respond_to_waiting_founder({
        **base_state,
        "founder_response": "What questions should I ask?",
        "founder_messages": [{"role": "founder", "content": "What questions should I ask?"}],
    })
    assert "Ask for one recent expired inventory incident" in res3["waiting_response"]
    assert "Ask what costs were incurred" in res3["waiting_response"]

    # Query 4: What should I bring back?
    res4 = respond_to_waiting_founder({
        **base_state,
        "founder_response": "What should I bring back?",
        "founder_messages": [{"role": "founder", "content": "What should I bring back?"}],
    })
    assert "One concrete incident per store" in res4["waiting_response"]

    # Query 5: What should I avoid?
    res5 = respond_to_waiting_founder({
        **base_state,
        "founder_response": "What should I avoid concluding?",
        "founder_messages": [{"role": "founder", "content": "What should I avoid concluding?"}],
    })
    assert "Willingness to pay for automated software" in res5["waiting_response"]
    assert "this step is for learning, not validation" in res5["waiting_response"].lower()


def test_terminal_state_renders_needs_revision_with_fallback_banner():
    """10. Terminal state renders EXPERIMENT_DESIGN_NEEDS_REVISION with fallback banner."""
    fallback_dict = {
        "assumption_id": "ca-010",
        "target_area": "Problem",
        "learning_goal": "Whether independent grocery store owners experience high inventory spoilage.",
        "primary_action": "Talk with 5–8 independent grocery store owners.",
        "who_to_engage": "5–8 independent grocery store owners",
        "what_to_ask_or_observe": ["Ask for one recent incident of inventory spoilage."],
        "what_to_bring_back": ["One concrete recent incident."],
        "what_not_to_infer": ["Willingness to pay for software."],
        "completion_instruction": "Return with the notes from those conversations.",
    }
    state = {
        "phase": WorkflowPhase.TEST,
        "experiment_design_status": ExperimentDesignStatus.QUALITY_BAR_NOT_MET.value,
        "active_assumption": {
            "statement": "Independent grocery store owners experience high inventory spoilage."
        },
        "experiment_critic_output": {
            "issues": [
                "Proxy metric lacks direct behavioral observation",
                "Sample size unfeasible within founder constraints",
            ]
        },
        "fallback_learning_action": fallback_dict,
    }

    banner = render_terminal_state(state)

    assert "STARTARY — EXPERIMENT DESIGN NEEDS REVISION" in banner
    assert "Independent grocery store owners experience high inventory spoilage." in banner
    assert "Proxy metric lacks direct behavioral observation" in banner
    assert "The diagnosis is still valid, but Startary has not yet produced a test reliable enough to run." in banner
    assert "Here is the safest next learning action:" in banner
    assert "WHAT DO WE STILL NEED TO LEARN?" in banner
    assert "WHAT SHOULD I DO NEXT?" in banner
    assert "WHAT SHOULD I BRING BACK?" in banner
    assert "WHAT SHOULD I NOT CONCLUDE YET?" in banner
    assert "This action is for learning, not validation." in banner
    assert "STARTARY — INITIAL DIAGNOSIS COMPLETE" not in banner
