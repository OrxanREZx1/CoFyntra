"""
Verification script for Milestone 4.0.4: Grounding & Experiment Recovery Integrity.

Scenarios verified:
Scenario A: CAPABILITY GROUNDING
  - Founder: "We're two founders and both know how to code."
  - technical capability = known (present)
  - design capability = UNKNOWN
  - marketing capability = UNKNOWN
  - Diagnosis snapshot execution constraints do NOT list design or marketing as limitations.

Scenario B: EXPERIMENT CRITIC FAILURE
  - Flawed experiment rejected twice by Critic.
  - Revision budget exhausted -> routes to handle_experiment_failure_stop.
  - experiment_design_status = QUALITY_BAR_NOT_MET.
  - Terminal status = EXPERIMENT_DESIGN_NEEDS_REVISION.
  - Rendered banner displays "STARTARY — EXPERIMENT DESIGN NEEDS REVISION".
  - Critical assumption and diagnosis snapshot survive intact.

Scenario C: CONVERSATIONAL NATURALNESS
  - Part 1: "15 hours each." -> mode is BRIEF_CONTINUE, diagnosis delta is suppressed, response is concise (1-2 sentences).
  - Part 2: "I don't know if this is worth shipping." -> mode is DIRECT_ANSWER, answers concern directly first.
"""

import sys
from unittest.mock import MagicMock

from startary.domain.enums import (
    ConversationalResponseMode,
    ExperimentDesignStatus,
    WorkflowPhase,
)
from startary.domain.execution_context import ExecutionContext
from startary.domain.provenance import (
    FactProvenance,
    unknown_field,
    validate_state_grounding,
)
from startary.domain.terminal_state import (
    TerminalStatus,
    render_terminal_state,
    resolve_founder_facing_status,
)
from startary.domain.turn_control import (
    NextTurnDecision,
    determine_conversational_response_mode,
    should_surface_diagnosis_delta,
)
from startary.graph.builder import route_on_experiment_critique
from startary.graph.nodes import (
    compose_advisor_response,
    compose_startup_diagnosis,
    handle_experiment_failure_stop,
    merge_execution_state,
)
from startary.llm.schemas import (
    AdvisorResponseLLMOutput,
    StartupDiagnosisSnapshotLLMOutput,
)


def verify_scenario_a():
    print("\n" + "=" * 60)
    print("SCENARIO A: CAPABILITY GROUNDING INTEGRITY")
    print("=" * 60)

    founder_turn = "We're two founders and both know how to code."
    print(f"Founder says: '{founder_turn}'")

    # Simulate extractor delta where an LLM might attempt speculative negative inferences
    candidate_delta = {
        "founder_count": {"value": "2", "provenance": "EXPLICIT"},
        "has_technical_capability": {"value": "true", "provenance": "EXPLICIT"},
        "has_design_capability": {"value": "false", "provenance": "INFERRED"},
        "has_marketing_capability": {"value": "false", "provenance": "INFERRED"},
    }

    validated_delta = validate_state_grounding({}, candidate_delta, founder_turn)

    assert validated_delta["has_technical_capability"]["value"] == "true", "Technical capability must be present"
    assert validated_delta["has_design_capability"] is None, "Design capability must NOT be inferred as false"
    assert validated_delta["has_marketing_capability"] is None, "Marketing capability must NOT be inferred as false"

    state = {
        "project_id": "proj-m404-scen-a",
        "founder_messages": [{"role": "founder", "content": founder_turn}],
        "startup_state_delta": validated_delta,
        "execution_context": ExecutionContext().model_dump(),
    }
    merged = merge_execution_state(state)
    exec_ctx = merged["execution_context"]

    tech_prov = exec_ctx["technical_capability"]["provenance"]
    tech_val = exec_ctx["technical_capability"]["value"]
    design_prov = exec_ctx["design_capability"]["provenance"]
    marketing_prov = exec_ctx["marketing_capability"]["provenance"]

    print(f"-> Technical capability: value={tech_val}, provenance={tech_prov}")
    print(f"-> Design capability: provenance={design_prov}")
    print(f"-> Marketing capability: provenance={marketing_prov}")

    assert tech_prov == "EXPLICIT" and tech_val == "true"
    assert design_prov == "UNKNOWN", f"Expected UNKNOWN, got {design_prov}"
    assert marketing_prov == "UNKNOWN", f"Expected UNKNOWN, got {marketing_prov}"

    # Now verify diagnosis snapshot filtering
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_snapshot = StartupDiagnosisSnapshotLLMOutput(
        stage="RAW_IDEA",
        stage_explanation="Pre-product problem exploration.",
        grounded_areas=["Founder engineering ability"],
        current_advantages=["Technical founders"],
        execution_constraints=[
            "Neither founder has design capability",
            "Neither founder has marketing capability",
            "Part-time hours available",
        ],
        major_uncertainties=["Merchant demand verification"],
        weak_link="Problem",
        weak_link_reason="Problem needs verification.",
        weak_link_confidence="MEDIUM",
        weak_link_confidence_explanation="Early stage uncertainty.",
        competing_risks=["Demand Signal"],
        reality_summary="Early concept described.",
        recommended_learning_goal="Validate returns friction with merchants.",
        critical_assumption="Boutique fashion stores experience severe friction from returns.",
    )
    mock_structured.invoke.return_value = mock_snapshot
    mock_llm.with_structured_output.return_value = mock_structured

    import startary.graph.nodes as _nodes
    orig_llm = _nodes.get_llm_client
    _nodes.get_llm_client = lambda s: mock_llm

    try:
        diag_state = {
            "project_id": "proj-m404-diag",
            "iteration": 1,
            "weak_area": "Problem",
            "active_assumption": {"statement": "Boutique fashion stores experience severe friction from returns."},
            "weak_link_discrimination": {"confidence": "MEDIUM"},
            "reality_check": {"summary": "Early indicators exist."},
            "execution_context": exec_ctx,
        }
        res = compose_startup_diagnosis(diag_state)
        constraints = res["diagnosis_snapshot"]["execution_constraints"]
        print(f"-> Rendered Execution Constraints: {constraints}")

        assert not any("design" in c.lower() for c in constraints), "Design must NOT appear as execution constraint"
        assert not any("marketing" in c.lower() for c in constraints), "Marketing must NOT appear as execution constraint"
        assert any("part-time" in c.lower() for c in constraints), "Grounded constraint must be preserved"
    finally:
        _nodes.get_llm_client = orig_llm

    print("[PASS] Scenario A: Capability Grounding Integrity verified successfully.")


def verify_scenario_b():
    print("\n" + "=" * 60)
    print("SCENARIO B: EXPERIMENT CRITIC FAILURE RECOVERY")
    print("=" * 60)

    ca_statement = "Independent fashion stores suffer direct inventory disruption from sizing returns."
    state = {
        "project_id": "proj-m404-scen-b",
        "phase": WorkflowPhase.TEST,
        "iteration": 1,
        "weak_area": "Problem",
        "active_assumption": {"id": "ca-m404", "statement": ca_statement},
        "diagnosis_snapshot": {
            "id": "snap-m404",
            "weak_link": "Problem",
            "reality_summary": "One boutique owner reported returns friction.",
        },
        "critic_revision_count": 2,
        "experiment_critic_output": {
            "verdict": "REVISE",
            "issues": [
                "Proposed experiment requires 12 store visits in 3 days (infeasible under part-time hours)",
                "Contradiction criteria require merchants to rank returns as #1 business priority (overpowering assumption)",
            ],
            "revision_instructions": "Reduce sample size and align decision threshold with burden claim.",
        },
        "proposed_experiment_output": {
            "title": "Aggressive Multi-Store In-Person Survey",
            "experiment_type": "CUSTOMER_INTERVIEW",
        },
    }

    # Step 1: Routing
    next_node = route_on_experiment_critique(state)
    print(f"-> Critic revision count: {state['critic_revision_count']}")
    print(f"-> Route on Experiment Critique: '{next_node}'")
    assert next_node == "handle_experiment_failure_stop", f"Expected handle_experiment_failure_stop, got {next_node}"

    # Step 2: Failure Node Execution
    updates = handle_experiment_failure_stop(state)
    final_state = {**state, **updates}

    exp_status = final_state.get("experiment_design_status")
    print(f"-> Experiment Design Status: {exp_status}")
    assert exp_status == ExperimentDesignStatus.QUALITY_BAR_NOT_MET.value

    # Verify diagnostic state preserved
    assert final_state["weak_area"] == "Problem"
    assert final_state["active_assumption"]["statement"] == ca_statement
    assert final_state["diagnosis_snapshot"]["weak_link"] == "Problem"
    assert final_state.get("active_experiment") is None, "Rejected experiment must NEVER be marked accepted / READY"

    # Step 3: Terminal Status Resolution
    term_status = resolve_founder_facing_status(final_state)
    print(f"-> Resolved Terminal Status: {term_status}")
    assert term_status == TerminalStatus.EXPERIMENT_DESIGN_NEEDS_REVISION
    assert term_status != TerminalStatus.INITIAL_DIAGNOSIS_COMPLETE

    # Step 4: Terminal Banner Rendering
    banner = render_terminal_state(final_state)
    print("\n--- Rendered Founder-Facing Banner ---")
    print(banner)
    print("--------------------------------------")

    assert "STARTARY — EXPERIMENT DESIGN NEEDS REVISION" in banner
    assert ca_statement in banner
    assert "Proposed experiment requires 12 store visits" in banner
    assert "Contradiction criteria require merchants to rank returns as #1 business priority" in banner
    assert "The diagnosis is still valid, but Startary has not yet produced a test reliable enough to run." in banner
    assert "INITIAL DIAGNOSIS COMPLETE" not in banner

    print("[PASS] Scenario B: Experiment Critic Failure Recovery verified successfully.")


def verify_scenario_c():
    print("\n" + "=" * 60)
    print("SCENARIO C: CONVERSATIONAL NATURALNESS & RESPONSE MODES")
    print("=" * 60)

    # Turn 1: Simple Capacity Answer
    founder_turn_1 = "15 hours each."
    print(f"Turn 1 Founder Input: '{founder_turn_1}'")

    state_1 = {
        "project_id": "proj-m404-scen-c",
        "phase": WorkflowPhase.UNDERSTAND,
        "startup_stage": "RAW_IDEA",
        "weak_area": "Problem",
        "active_next_turn_decision": {
            "question_id": "q_budget_01",
            "information_target": "RESOURCE",
            "canonical_question": "What budget can you realistically use for LoopCart?",
            "should_ask_question": True,
        },
        "founder_messages": [{"role": "founder", "content": founder_turn_1}],
    }

    mode_1 = determine_conversational_response_mode(
        state=state_1,
        founder_interpretation={"is_question": False, "is_correction": False},
        founder_turn=founder_turn_1,
        previous_state=dict(state_1),
    )
    surface_delta_1 = should_surface_diagnosis_delta(dict(state_1), state_1, founder_turn_1)

    print(f"-> Turn 1 Response Mode: {mode_1}")
    print(f"-> Turn 1 Surface Diagnosis Delta: {surface_delta_1}")
    assert mode_1 == ConversationalResponseMode.BRIEF_CONTINUE
    assert surface_delta_1 is False, "Routine operational answer must not surface diagnostic commentary"

    # Turn 2: Strategic Concern / Direct Question
    founder_turn_2 = "is this problem worth to ship?"
    print(f"\nTurn 2 Founder Input: '{founder_turn_2}'")

    state_2 = {
        "project_id": "proj-m404-scen-c",
        "phase": WorkflowPhase.UNDERSTAND,
        "startup_stage": "RAW_IDEA",
        "weak_area": "Problem",
        "active_next_turn_decision": {
            "question_id": "q_reach_02",
            "information_target": "CUSTOMER_ACCESS",
            "canonical_question": "Do you have access to other boutique store owners?",
            "should_ask_question": True,
        },
        "founder_messages": [
            {"role": "founder", "content": founder_turn_1},
            {"role": "founder", "content": founder_turn_2},
        ],
    }

    mode_2 = determine_conversational_response_mode(
        state=state_2,
        founder_interpretation={"is_question": True, "asks_for_guidance": True},
        founder_turn=founder_turn_2,
        previous_state=dict(state_1),
    )
    print(f"-> Turn 2 Response Mode: {mode_2}")
    assert mode_2 == ConversationalResponseMode.DIRECT_ANSWER

    # Mock response generation to test direct answer order
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    direct_resp_text = (
        "Not yet. One store is enough to justify testing the problem, but not enough to justify shipping broadly. "
        "Do you have access to other boutique store owners?"
    )
    mock_advisor_out = AdvisorResponseLLMOutput(
        question_id="q_reach_02",
        next_question_target="CUSTOMER_ACCESS",
        next_question_rephrased="Do you have access to other boutique store owners?",
        acknowledge="Understood.",
        answer_to_founder="Not yet. One store is enough to justify testing the problem, but not enough to justify shipping broadly.",
        response_mode="DIRECT_ANSWER",
        response_text=direct_resp_text,
    )
    mock_structured.invoke.return_value = mock_advisor_out
    mock_llm.with_structured_output.return_value = mock_structured

    import startary.graph.nodes as _nodes
    orig_llm = _nodes.get_llm_client
    _nodes.get_llm_client = lambda s: mock_llm

    try:
        adv_res = compose_advisor_response(state_2)
        resp = adv_res["latest_advisor_response"]
        print(f"-> Advisor Response Text: '{resp['response_text']}'")
        assert resp["response_mode"] == "DIRECT_ANSWER"
        assert resp["question_id"] == "q_reach_02"
        assert resp["next_question_target"] == "CUSTOMER_ACCESS"
        # Direct answer appears first
        assert resp["response_text"].startswith("Not yet")
    finally:
        _nodes.get_llm_client = orig_llm

    print("[PASS] Scenario C: Conversational Naturalness & Response Modes verified successfully.")


def main():
    print("================================================================")
    print("STARTARY MILESTONE 4.0.4 — INTEGRATION VERIFICATION")
    print("================================================================")
    try:
        verify_scenario_a()
        verify_scenario_b()
        verify_scenario_c()
        print("\n" + "=" * 60)
        print("ALL MILESTONE 4.0.4 SCENARIOS PASSED SUCCESSFULLY!")
        print("=" * 60)
    except AssertionError as exc:
        print(f"\n[FAIL] Assertion failed: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"\n[ERROR] Unexpected error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
