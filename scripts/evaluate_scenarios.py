"""
Scenario evaluation script for Milestone 2.5.1: Sufficiency Calibration + Weak-Link Scoring Backlog.

Runs canonical scenarios through the Startary system.
Supports live evaluation when OPENAI_API_KEY is present, and deterministic offline evaluation when absent.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from startary.config import get_settings
from startary.domain.enums import WorkflowPhase
from startary.domain.taxonomy import StartupArea
from startary.graph.builder import build_graph
from startary.graph.graph_state import StartaryGraphState
from startary.llm.provider import is_valid_api_key
from startary.llm.schemas import (
    BaselineQuestionPlanLLMOutput,
    BaselineSufficiencyLLMOutput,
    ClarifyingQuestionLLMOutput,
    ContextSufficiencyLLMOutput,
    CriticalAssumptionLLMOutput,
    DiagnosisLLMOutput,
    InformationGapPlanLLMOutput,
    RealityCheckLLMOutput,
    StartupAreaAssessmentBatchLLMOutput,
    StartupStateDeltaLLMOutput,
    WeakLinkClarifyingQuestionLLMOutput,
    WeakLinkDiscriminationLLMOutput,
)
from tests.fixtures.scenarios import (
    MYNERA_ANSWER_1,
    MYNERA_ANSWER_2,
    SCENARIO_A_EXTREMELY_SPARSE,
    SCENARIO_B_MYNERA_SPARSE,
    SCENARIO_C_EVIDENCE_RICH,
    SCENARIO_D_PRODUCT_NO_PROBLEM,
    SCENARIO_E_PRICING_ONLY,
    SCENARIO_F_KNOWN_PROBLEM_MISSING_REVENUE,
    make_minimal_valid_llm_output_dict,
    make_valid_area_assessments_batch_output_dict,
    make_valid_clarifying_question_output_dict,
    make_valid_context_sufficiency_output_dict,
    make_valid_critical_assumption_output_dict,
    make_valid_reality_check_output_dict,
)


def evaluate():
    settings = get_settings()
    has_key = is_valid_api_key(settings.openai_api_key)
    print(f"=== STARTARY MILESTONE 2.5.1 SCENARIO EVALUATION ===")
    print(f"OPENAI_API_KEY active: {has_key}")
    print(f"Model configured: {settings.model_name}\n")

    scenarios = [
        ("SCENARIO A — Extremely Sparse", SCENARIO_A_EXTREMELY_SPARSE, False, None),
        ("SCENARIO B — Mynera Sparse", SCENARIO_B_MYNERA_SPARSE, False, None),
        (
            "SCENARIO B — Mynera 3-Step Calibrated",
            SCENARIO_B_MYNERA_SPARSE,
            False,
            [MYNERA_ANSWER_1, MYNERA_ANSWER_2],
        ),
        ("SCENARIO C — Evidence-Rich", SCENARIO_C_EVIDENCE_RICH, True, None),
        ("SCENARIO D — Product Detail, No Problem", SCENARIO_D_PRODUCT_NO_PROBLEM, False, None),
        ("SCENARIO E — Unsupported Pricing Claim", SCENARIO_E_PRICING_ONLY, False, None),
        ("SCENARIO F — Known Problem, Missing Revenue", SCENARIO_F_KNOWN_PROBLEM_MISSING_REVENUE, True, None),
    ]

    for title, raw_input, expected_sufficient, follow_up_answers in scenarios:
        print("=" * 65)
        print(f"EVALUATING: {title}")
        print("=" * 65)
        print(f"INITIAL INPUT:\n{raw_input}\n")

        checkpointer = MemorySaver()
        graph = build_graph(checkpointer=checkpointer)
        thread_id = f"eval-{title[:10].replace(' ', '_')}"
        cfg = {"configurable": {"thread_id": thread_id}}

        initial_state: StartaryGraphState = {
            "project_id": thread_id,
            "phase": WorkflowPhase.UNDERSTAND,
            "iteration": 1,
            "raw_context": raw_input,
            "founder_messages": [{"role": "founder", "content": raw_input}],
            "clarification_turn_count": 0,
            "clarification_history": [],
        }

        if not has_key:
            # Deterministic mock runner
            turn_counter = {"count": 0}

            def mock_with_structured_output(schema):
                structured = MagicMock()
                if schema == DiagnosisLLMOutput:
                    structured.invoke.return_value = DiagnosisLLMOutput(**make_minimal_valid_llm_output_dict())
                elif schema == StartupStateDeltaLLMOutput:
                    structured.invoke.return_value = StartupStateDeltaLLMOutput()
                elif schema == InformationGapPlanLLMOutput:
                    from tests.fixtures.scenarios import make_valid_gap_plan_output_dict
                    is_suff = expected_sufficient or (follow_up_answers and turn_counter["count"] >= len(follow_up_answers))
                    structured.invoke.return_value = InformationGapPlanLLMOutput(
                        **make_valid_gap_plan_output_dict(
                            should_ask=not is_suff,
                            strategic_sufficient=is_suff,
                            question="What specific difficulties do students face?" if not is_suff else None,
                        )
                    )
                elif schema == ContextSufficiencyLLMOutput:
                    is_suff = expected_sufficient or (follow_up_answers and turn_counter["count"] >= len(follow_up_answers))
                    g_quality = "STRONG" if is_suff else ("WEAK" if turn_counter["count"] == 0 else "MODERATE")
                    structured.invoke.return_value = ContextSufficiencyLLMOutput(
                        **make_valid_context_sufficiency_output_dict(
                            sufficient=is_suff,
                            grounding_quality=g_quality,
                            would_one_more_answer_materially_change_direction=not is_suff,
                        )
                    )
                elif schema == ClarifyingQuestionLLMOutput:
                    structured.invoke.return_value = ClarifyingQuestionLLMOutput(
                        **make_valid_clarifying_question_output_dict(StartupArea.PROBLEM)
                    )
                elif schema == StartupAreaAssessmentBatchLLMOutput:
                    structured.invoke.return_value = StartupAreaAssessmentBatchLLMOutput(
                        **make_valid_area_assessments_batch_output_dict(StartupArea.PROBLEM)
                    )
                elif schema == RealityCheckLLMOutput:
                    structured.invoke.return_value = RealityCheckLLMOutput(
                        **make_valid_reality_check_output_dict(StartupArea.PROBLEM)
                    )
                elif schema == CriticalAssumptionLLMOutput:
                    structured.invoke.return_value = CriticalAssumptionLLMOutput(
                        **make_valid_critical_assumption_output_dict(StartupArea.PROBLEM)
                    )
                elif schema == BaselineSufficiencyLLMOutput:
                    from tests.fixtures.scenarios import make_valid_baseline_sufficiency_output_dict
                    structured.invoke.return_value = BaselineSufficiencyLLMOutput(
                        **make_valid_baseline_sufficiency_output_dict(is_sufficient=True)
                    )
                elif schema == BaselineQuestionPlanLLMOutput:
                    from tests.fixtures.scenarios import make_valid_baseline_question_plan_output_dict
                    structured.invoke.return_value = BaselineQuestionPlanLLMOutput(
                        **make_valid_baseline_question_plan_output_dict(should_ask=False)
                    )
                elif schema == WeakLinkDiscriminationLLMOutput:
                    from tests.fixtures.scenarios import make_valid_weak_link_discrimination_output_dict
                    structured.invoke.return_value = WeakLinkDiscriminationLLMOutput(
                        **make_valid_weak_link_discrimination_output_dict(selection_confidence="HIGH", should_clarify=False)
                    )
                elif schema == WeakLinkClarifyingQuestionLLMOutput:
                    from tests.fixtures.scenarios import make_valid_weak_link_clarifying_question_output_dict
                    structured.invoke.return_value = WeakLinkClarifyingQuestionLLMOutput(
                        **make_valid_weak_link_clarifying_question_output_dict()
                    )
                return structured

            with patch("startary.graph.nodes.get_llm_client") as mock_get_llm:
                mock_llm = MagicMock()
                mock_llm.with_structured_output.side_effect = mock_with_structured_output
                mock_get_llm.return_value = mock_llm

                graph.invoke(initial_state, cfg)

                if follow_up_answers:
                    for ans in follow_up_answers:
                        turn_counter["count"] += 1
                        print(f"  [Founder Turn {turn_counter['count']}]: {ans[:80]}...")
                        graph.invoke(Command(resume=ans), cfg)

        else:
            # Live execution
            graph.invoke(initial_state, cfg)

            if follow_up_answers:
                for idx, ans in enumerate(follow_up_answers, 1):
                    snapshot = graph.get_state(cfg)
                    if snapshot.tasks and snapshot.tasks[0].interrupts:
                        q_val = snapshot.tasks[0].interrupts[0].value
                        q_str = q_val.get("question") if isinstance(q_val, dict) else str(q_val)
                        print(f"  Startary Question: {q_str}")
                        print(f"  Founder Answer {idx}: {ans[:80]}...")
                        graph.invoke(Command(resume=ans), cfg)

        # Retrieve results
        snapshot = graph.get_state(cfg)
        final_values = snapshot.values

        cs = final_values.get("context_sufficiency") or {}
        cq = final_values.get("current_clarifying_question") or {}
        weak_area = final_values.get("weak_area")
        weak_details = final_values.get("weak_link_details") or {}
        rc = final_values.get("reality_check") or {}
        assumption = final_values.get("active_assumption") or {}

        print("\n--- RESULTS ---")
        print(f"Sufficiency Status   : {'SUFFICIENT' if cs.get('sufficient') else 'INSUFFICIENT'}")
        print(f"Grounding Quality    : {cs.get('grounding_quality')}")
        print(f"One More Answer Val  : {cs.get('would_one_more_answer_materially_change_direction')}")
        print(f"Sufficiency Reason   : {cs.get('reason')}")

        if not cs.get("sufficient"):
            print(f"Blocking Unknowns    : {cs.get('blocking_unknowns')}")
            print(f"Clarifying Question  : {cq.get('question') or (snapshot.tasks[0].interrupts[0].value.get('question') if snapshot.tasks else 'N/A')}")
            print(f"Weak Link Run?       : {'NO (Correct)' if not weak_area else 'YES (Error)'}")
            print(f"Assumption Created?  : {'NO (Correct)' if not assumption else 'YES (Error)'}")
        else:
            print(f"Selected Weak Link   : {weak_area}")
            if "saturation_telemetry" in weak_details:
                telemetry = weak_details["saturation_telemetry"]
                print(f"Saturation Telemetry : Max Score={telemetry.get('max_score')}, Ties={telemetry.get('max_score_tie_count')}, 81s={telemetry.get('saturated_area_count')}, TieBroken={telemetry.get('tie_breaker_used')}")
            print(f"Reality Check Concl. : {rc.get('conclusion')}")
            print(f"Critical Assumption  : {assumption.get('statement')}")

        print("\n")


if __name__ == "__main__":
    evaluate()
