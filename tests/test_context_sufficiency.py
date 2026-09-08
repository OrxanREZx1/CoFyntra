"""
Tests: Context Sufficiency Gate & Conversational Clarification Loop — Milestone 2.5 & 2.5.1.

All tests mock LLM responses and run offline without live network calls.
"""

from unittest.mock import MagicMock
import pytest
from langgraph.types import Command

from startary.domain.enums import WorkflowPhase
from startary.domain.sufficiency import ContextSufficiencyAssessment, GroundingQuality
from startary.domain.clarification import ClarifyingQuestion
from startary.domain.taxonomy import StartupArea
from startary.graph.builder import build_graph, route_on_context_sufficiency
from startary.graph.graph_state import StartaryGraphState
from startary.graph.nodes import (
    assess_context_sufficiency,
    select_clarifying_question,
    merge_founder_answer,
    handle_insufficient_context_stop,
)
from startary.llm.schemas import (
    AdvisorResponseLLMOutput,
    BaselineQuestionPlanLLMOutput,
    BaselineSufficiencyLLMOutput,
    ClarifyingQuestionLLMOutput,
    ContextSufficiencyLLMOutput,
    CriticalAssumptionLLMOutput,
    DiagnosisLLMOutput,
    ExperimentCriticLLMOutput,
    ExperimentGeneratorLLMOutput,
    FounderTurnInterpretationLLMOutput,
    RealityCheckLLMOutput,
    StartupAreaAssessmentBatchLLMOutput,
    StartupDiagnosisSnapshotLLMOutput,
    WeakLinkClarifyingQuestionLLMOutput,
    WeakLinkDiscriminationLLMOutput,
)
from tests.fixtures.scenarios import (
    make_minimal_valid_llm_output_dict,
    make_valid_advisor_response_output_dict,
    make_valid_context_sufficiency_output_dict,
    make_valid_clarifying_question_output_dict,
    make_valid_area_assessments_batch_output_dict,
    make_valid_diagnosis_snapshot_output_dict,
    make_valid_founder_turn_interpretation_output_dict,
    make_valid_reality_check_output_dict,
    make_valid_critical_assumption_output_dict,
    make_valid_experiment_critic_output_dict,
    make_valid_experiment_generator_output_dict,
    SCENARIO_B_MYNERA_SPARSE,
    MYNERA_ANSWER_1,
    SCENARIO_C_EVIDENCE_RICH,
)


class TestContextSufficiencyDomain:
    def test_grounding_quality_enum_values(self):
        assert GroundingQuality.WEAK == "WEAK"
        assert GroundingQuality.MODERATE == "MODERATE"
        assert GroundingQuality.STRONG == "STRONG"

    def test_sufficiency_model_validates(self):
        cs = ContextSufficiencyAssessment(
            sufficient=False,
            grounding_quality=GroundingQuality.WEAK,
            would_one_more_answer_materially_change_direction=True,
            reason="Problem specifics missing.",
            blocking_unknowns=["Specific user friction."],
            usable_grounding=["Students in IT."],
            prohibited_inferences=["Unstated recommendation algorithms."],
            recommended_question_area=StartupArea.PROBLEM,
            information_goal="Identify student difficulties.",
        )
        assert cs.sufficient is False
        assert cs.grounding_quality == GroundingQuality.WEAK
        assert cs.would_one_more_answer_materially_change_direction is True
        assert cs.recommended_question_area == StartupArea.PROBLEM
        assert len(cs.blocking_unknowns) == 1

    def test_clarifying_question_model_validates(self):
        cq = ClarifyingQuestion(
            question="What specific friction do students experience?",
            target_area=StartupArea.PROBLEM,
            information_goal="Identify student difficulties.",
            why_this_question_now="Core problem is ungrounded.",
        )
        assert cq.target_area == StartupArea.PROBLEM
        assert len(cq.question) >= 10


class TestSufficiencyCalibrationScenarios:
    """Tests for Milestone 2.5.1 calibration of epistemic levels."""

    def test_pure_concept_is_weak_and_insufficient(self):
        output = ContextSufficiencyLLMOutput(
            sufficient=False,
            confidence="MEDIUM",
            grounding_quality=GroundingQuality.WEAK,
            would_one_more_answer_materially_change_direction=True,
            reason="Pure concept without concrete user observations.",
            blocking_unknowns=["Target user segment specifics"],
            usable_grounding=[],
            prohibited_inferences=["Assumed recommendation logic"],
            recommended_question_area=StartupArea.PROBLEM,
            information_goal="Identify specific user and friction.",
        )
        assert output.grounding_quality == GroundingQuality.WEAK
        assert output.sufficient is False
        assert output.would_one_more_answer_materially_change_direction is True

    def test_generalized_plausible_story_is_weak_and_insufficient(self):
        output = ContextSufficiencyLLMOutput(
            sufficient=False,
            confidence="MEDIUM",
            grounding_quality=GroundingQuality.WEAK,
            would_one_more_answer_materially_change_direction=True,
            reason="Generalized hypothetical example without direct observation or founder experience.",
            blocking_unknowns=["Whether this is founder experience or observed in real students"],
            usable_grounding=["Belief that students drop out after 2 months"],
            prohibited_inferences=["Assumed frequency of course dropouts"],
            recommended_question_area=StartupArea.PROBLEM,
            information_goal="Determine source of founder belief.",
        )
        assert output.grounding_quality == GroundingQuality.WEAK
        assert output.sufficient is False

    def test_founder_personal_experience_is_moderate_grounding(self):
        output = ContextSufficiencyLLMOutput(
            sufficient=False,
            confidence="MEDIUM",
            grounding_quality=GroundingQuality.MODERATE,
            would_one_more_answer_materially_change_direction=True,
            reason="Direct founder experience provides grounding, but external prevalence remains unknown.",
            blocking_unknowns=["Whether others experience the same friction"],
            usable_grounding=["Founder personally experienced branch choice confusion"],
            prohibited_inferences=[],
            recommended_question_area=StartupArea.TARGET_USER,
            information_goal="Determine if other students report this.",
        )
        assert output.grounding_quality == GroundingQuality.MODERATE

    def test_customer_conversations_allow_sufficiency(self):
        output = ContextSufficiencyLLMOutput(
            sufficient=True,
            confidence="HIGH",
            grounding_quality=GroundingQuality.MODERATE,
            would_one_more_answer_materially_change_direction=False,
            reason="Interviews with 12 students provide direct qualitative grounding to identify weak link.",
            blocking_unknowns=[],
            usable_grounding=["12 student interviews", "Confusion between SWE and Cybersecurity"],
            prohibited_inferences=[],
            recommended_question_area=None,
            information_goal="",
        )
        assert output.grounding_quality == GroundingQuality.MODERATE
        assert output.sufficient is True
        assert output.would_one_more_answer_materially_change_direction is False


class TestSufficiencyNodes:
    def _create_state(self, is_suff: bool = False, turn: int = 0) -> StartaryGraphState:
        return StartaryGraphState(
            project_id="test-suff-proj",
            phase=WorkflowPhase.DIAGNOSE,
            iteration=1,
            raw_context="Mynera is an AI tool for IT branch discovery.",
            founder_messages=[{"role": "founder", "content": "Mynera is an AI tool for IT branch discovery."}],
            clarification_turn_count=turn,
            clarification_history=[],
            startup_state={"areas": {}},
            diagnosis={"summary": "Sparse context."},
            traces=[],
        )

    def test_assess_context_sufficiency_node(self, mocker):
        output_dict = make_valid_context_sufficiency_output_dict(sufficient=False)
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = ContextSufficiencyLLMOutput(**output_dict)
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured

        mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
        mocker.patch(
            "startary.graph.nodes.get_settings",
            return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"),
        )

        state = self._create_state()
        result = assess_context_sufficiency(state)

        assert "context_sufficiency" in result
        assert result["is_sufficient"] is False
        cs = ContextSufficiencyAssessment(**result["context_sufficiency"])
        assert cs.sufficient is False
        assert cs.grounding_quality == GroundingQuality.WEAK
        assert cs.would_one_more_answer_materially_change_direction is True

    def test_select_clarifying_question_node(self, mocker):
        output_dict = make_valid_clarifying_question_output_dict(area=StartupArea.PROBLEM)
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = ClarifyingQuestionLLMOutput(**output_dict)
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured

        mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
        mocker.patch(
            "startary.graph.nodes.get_settings",
            return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"),
        )

        state = self._create_state()
        state["context_sufficiency"] = make_valid_context_sufficiency_output_dict(sufficient=False)

        result = select_clarifying_question(state)

        assert "current_clarifying_question" in result
        assert result["phase"] == WorkflowPhase.CLARIFYING_CONTEXT
        cq = ClarifyingQuestion(**result["current_clarifying_question"])
        assert cq.target_area == StartupArea.PROBLEM
        assert len(cq.question) >= 10

    def test_merge_founder_answer_node(self):
        state = self._create_state(turn=0)
        state["current_clarifying_question"] = {
            "question": "What problem do students have?",
            "target_area": "Problem",
            "information_goal": "Clarify pain",
        }
        state["founder_response"] = "Students are confused between software and data science."

        result = merge_founder_answer(state)

        assert result["clarification_turn_count"] == 1
        assert len(result["founder_messages"]) == 3  # Initial + Startary Q + Founder A
        assert len(result["clarification_history"]) == 1
        assert result["phase"] == WorkflowPhase.UNDERSTAND
        assert "Students are confused" in result["raw_context"]

    def test_handle_insufficient_context_stop_node(self):
        state = self._create_state(turn=5)
        state["context_sufficiency"] = {
            "sufficient": False,
            "blocking_unknowns": ["User identity", "Problem friction"],
        }
        result = handle_insufficient_context_stop(state)

        assert result["phase"] == WorkflowPhase.UNDERSTAND
        assert "maximum clarification limit" in result["status_message"]
        assert "User identity" in result["status_message"]


class TestSufficiencyRouting:
    def test_routes_to_assess_startup_areas_when_sufficient(self):
        state: StartaryGraphState = {
            "context_sufficiency": {"sufficient": True},
            "clarification_turn_count": 0,
        }
        route = route_on_context_sufficiency(state)
        assert route == "assess_startup_areas"

    def test_routes_to_select_clarifying_question_when_insufficient_and_under_limit(self):
        state: StartaryGraphState = {
            "context_sufficiency": {"sufficient": False},
            "clarification_turn_count": 2,
        }
        route = route_on_context_sufficiency(state)
        assert route == "select_clarifying_question"

    def test_routes_to_handle_insufficient_context_stop_when_turn_limit_reached(self):
        state: StartaryGraphState = {
            "context_sufficiency": {"sufficient": False},
            "clarification_turn_count": 5,
        }
        route = route_on_context_sufficiency(state)
        assert route == "handle_insufficient_context_stop"


class TestGraphInterruptAndConversationalLoop:
    def _patch_llm_for_m2_5(self, mocker, sufficient_on_turn_0: bool = False):
        """
        Patch LLM client with M2.6-aware schema mocking.

        Supports M2.6 schemas (InformationGapPlanLLMOutput, StartupStateDeltaLLMOutput)
        in addition to existing M2.5 schemas for backward compat.
        """
        from startary.llm.schemas import InformationGapPlanLLMOutput, StartupStateDeltaLLMOutput
        from tests.fixtures.scenarios import make_valid_gap_plan_output_dict
        mock_llm = MagicMock()

        def mock_with_structured(schema):
            mock_s = MagicMock()
            if schema == DiagnosisLLMOutput:
                mock_s.invoke.return_value = DiagnosisLLMOutput(**make_minimal_valid_llm_output_dict())
            elif schema == StartupStateDeltaLLMOutput:
                # Empty delta — no profile extracted
                mock_s.invoke.return_value = StartupStateDeltaLLMOutput()
            elif schema == InformationGapPlanLLMOutput:
                # Map sufficient_on_turn_0 → gap planner decision
                mock_s.invoke.return_value = InformationGapPlanLLMOutput(
                    **make_valid_gap_plan_output_dict(
                        should_ask=not sufficient_on_turn_0,
                        strategic_sufficient=sufficient_on_turn_0,
                        question="Is this based on your own experience or conversations with students?" if not sufficient_on_turn_0 else None,
                    )
                )
            elif schema == ContextSufficiencyLLMOutput:
                # Backward compat (direct node tests still use this)
                mock_s.invoke.return_value = ContextSufficiencyLLMOutput(
                    **make_valid_context_sufficiency_output_dict(sufficient=sufficient_on_turn_0)
                )
            elif schema == ClarifyingQuestionLLMOutput:
                mock_s.invoke.return_value = ClarifyingQuestionLLMOutput(
                    **make_valid_clarifying_question_output_dict(StartupArea.PROBLEM)
                )
            elif schema == StartupAreaAssessmentBatchLLMOutput:
                mock_s.invoke.return_value = StartupAreaAssessmentBatchLLMOutput(
                    **make_valid_area_assessments_batch_output_dict(StartupArea.PROBLEM)
                )
            elif schema == RealityCheckLLMOutput:
                mock_s.invoke.return_value = RealityCheckLLMOutput(
                    **make_valid_reality_check_output_dict(StartupArea.PROBLEM)
                )
            elif schema == CriticalAssumptionLLMOutput:
                mock_s.invoke.return_value = CriticalAssumptionLLMOutput(
                    **make_valid_critical_assumption_output_dict(StartupArea.PROBLEM)
                )
            elif schema == BaselineSufficiencyLLMOutput:
                from tests.fixtures.scenarios import make_valid_baseline_sufficiency_output_dict
                mock_s.invoke.return_value = BaselineSufficiencyLLMOutput(
                    **make_valid_baseline_sufficiency_output_dict(is_sufficient=sufficient_on_turn_0)
                )
            elif schema == BaselineQuestionPlanLLMOutput:
                from tests.fixtures.scenarios import make_valid_baseline_question_plan_output_dict
                mock_s.invoke.return_value = BaselineQuestionPlanLLMOutput(
                    **make_valid_baseline_question_plan_output_dict(should_ask=not sufficient_on_turn_0)
                )
            elif schema == WeakLinkDiscriminationLLMOutput:
                from tests.fixtures.scenarios import make_valid_weak_link_discrimination_output_dict
                mock_s.invoke.return_value = WeakLinkDiscriminationLLMOutput(
                    **make_valid_weak_link_discrimination_output_dict(selection_confidence="HIGH", should_clarify=False)
                )
            elif schema == WeakLinkClarifyingQuestionLLMOutput:
                from tests.fixtures.scenarios import make_valid_weak_link_clarifying_question_output_dict
                mock_s.invoke.return_value = WeakLinkClarifyingQuestionLLMOutput(
                    **make_valid_weak_link_clarifying_question_output_dict()
                )
            elif schema == ExperimentGeneratorLLMOutput:
                mock_s.invoke.return_value = ExperimentGeneratorLLMOutput(
                    **make_valid_experiment_generator_output_dict()
                )
            elif schema == ExperimentCriticLLMOutput:
                mock_s.invoke.return_value = ExperimentCriticLLMOutput(
                    **make_valid_experiment_critic_output_dict()
                )
            elif schema == StartupDiagnosisSnapshotLLMOutput:
                mock_s.invoke.return_value = StartupDiagnosisSnapshotLLMOutput(
                    **make_valid_diagnosis_snapshot_output_dict()
                )
            elif schema == FounderTurnInterpretationLLMOutput:
                mock_s.invoke.return_value = FounderTurnInterpretationLLMOutput(
                    **make_valid_founder_turn_interpretation_output_dict()
                )
            elif schema == AdvisorResponseLLMOutput:
                mock_s.invoke.return_value = AdvisorResponseLLMOutput(
                    **make_valid_advisor_response_output_dict()
                )
            return mock_s

        mock_llm.with_structured_output.side_effect = mock_with_structured
        mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
        mocker.patch(
            "startary.graph.nodes.get_settings",
            return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"),
        )
        return mock_llm

    def test_sparse_mynera_input_interrupts_and_does_not_generate_assumption(self, mocker):
        """Sparse input halts at clarification interrupt without generating an assumption."""
        self._patch_llm_for_m2_5(mocker, sufficient_on_turn_0=False)
        from langgraph.checkpoint.memory import MemorySaver
        graph = build_graph(checkpointer=MemorySaver())
        thread_id = "test-mynera-sparse"
        cfg = {"configurable": {"thread_id": thread_id}}

        initial_state: StartaryGraphState = {
            "project_id": thread_id,
            "phase": WorkflowPhase.UNDERSTAND,
            "iteration": 1,
            "raw_context": SCENARIO_B_MYNERA_SPARSE,
        }

        # First run: should halt at interrupt
        graph.invoke(initial_state, cfg)

        snapshot = graph.get_state(cfg)
        assert snapshot.tasks and len(snapshot.tasks[0].interrupts) > 0
        interrupt_val = snapshot.tasks[0].interrupts[0].value
        assert "question" in interrupt_val
        assert len(interrupt_val["question"]) > 10

        # Verify downstream M2 nodes were NOT executed
        values = snapshot.values
        assert values.get("active_assumption") is None
        assert values.get("weak_area") is None
        assert values.get("reality_check") is None

    def test_sufficient_evidence_rich_input_runs_to_completion(self, mocker):
        """Evidence-rich input is evaluated as sufficient and proceeds through M2 to Critical Assumption."""
        self._patch_llm_for_m2_5(mocker, sufficient_on_turn_0=True)
        from langgraph.checkpoint.memory import MemorySaver
        graph = build_graph(checkpointer=MemorySaver())
        thread_id = "test-evidence-rich"
        cfg = {"configurable": {"thread_id": thread_id}}

        initial_state: StartaryGraphState = {
            "project_id": thread_id,
            "phase": WorkflowPhase.UNDERSTAND,
            "iteration": 1,
            "raw_context": SCENARIO_C_EVIDENCE_RICH,
        }

        graph.invoke(initial_state, cfg)

        snapshot = graph.get_state(cfg)
        assert len(snapshot.tasks) == 0  # Completed, not interrupted
        values = snapshot.values

        assert values.get("weak_area") == str(StartupArea.PROBLEM)
        assert values.get("reality_check") is not None
        assert values.get("active_assumption") is not None
        assert values.get("phase") == WorkflowPhase.TEST

    def test_multi_turn_resume_loop(self, mocker):
        """Graph interrupts on turn 0, resumes with founder answer, and loops back."""
        from startary.llm.schemas import InformationGapPlanLLMOutput, StartupStateDeltaLLMOutput
        from tests.fixtures.scenarios import make_valid_gap_plan_output_dict
        mock_llm = MagicMock()
        call_count = {"gap_planner": 0}

        def mock_with_structured(schema):
            mock_s = MagicMock()
            if schema == DiagnosisLLMOutput:
                mock_s.invoke.return_value = DiagnosisLLMOutput(**make_minimal_valid_llm_output_dict())
            elif schema == StartupStateDeltaLLMOutput:
                mock_s.invoke.return_value = StartupStateDeltaLLMOutput()
            elif schema == InformationGapPlanLLMOutput:
                # First call: insufficient (should_ask=True). Second+ call: sufficient!
                is_suff = (call_count["gap_planner"] > 0)
                call_count["gap_planner"] += 1
                mock_s.invoke.return_value = InformationGapPlanLLMOutput(
                    **make_valid_gap_plan_output_dict(
                        should_ask=not is_suff,
                        strategic_sufficient=is_suff,
                        question="Is this based on your own experience?" if not is_suff else None,
                    )
                )
            elif schema == ContextSufficiencyLLMOutput:
                mock_s.invoke.return_value = ContextSufficiencyLLMOutput(
                    **make_valid_context_sufficiency_output_dict(sufficient=False)
                )
            elif schema == ClarifyingQuestionLLMOutput:
                mock_s.invoke.return_value = ClarifyingQuestionLLMOutput(
                    **make_valid_clarifying_question_output_dict(StartupArea.PROBLEM)
                )
            elif schema == StartupAreaAssessmentBatchLLMOutput:
                mock_s.invoke.return_value = StartupAreaAssessmentBatchLLMOutput(
                    **make_valid_area_assessments_batch_output_dict(StartupArea.PROBLEM)
                )
            elif schema == RealityCheckLLMOutput:
                mock_s.invoke.return_value = RealityCheckLLMOutput(
                    **make_valid_reality_check_output_dict(StartupArea.PROBLEM)
                )
            elif schema == CriticalAssumptionLLMOutput:
                mock_s.invoke.return_value = CriticalAssumptionLLMOutput(
                    **make_valid_critical_assumption_output_dict(StartupArea.PROBLEM)
                )
            elif schema == BaselineSufficiencyLLMOutput:
                from tests.fixtures.scenarios import make_valid_baseline_sufficiency_output_dict
                mock_s.invoke.return_value = BaselineSufficiencyLLMOutput(
                    **make_valid_baseline_sufficiency_output_dict(is_sufficient=True)
                )
            elif schema == BaselineQuestionPlanLLMOutput:
                from tests.fixtures.scenarios import make_valid_baseline_question_plan_output_dict
                mock_s.invoke.return_value = BaselineQuestionPlanLLMOutput(
                    **make_valid_baseline_question_plan_output_dict(should_ask=False)
                )
            elif schema == WeakLinkDiscriminationLLMOutput:
                from tests.fixtures.scenarios import make_valid_weak_link_discrimination_output_dict
                mock_s.invoke.return_value = WeakLinkDiscriminationLLMOutput(
                    **make_valid_weak_link_discrimination_output_dict(selection_confidence="HIGH", should_clarify=False)
                )
            elif schema == WeakLinkClarifyingQuestionLLMOutput:
                from tests.fixtures.scenarios import make_valid_weak_link_clarifying_question_output_dict
                mock_s.invoke.return_value = WeakLinkClarifyingQuestionLLMOutput(
                    **make_valid_weak_link_clarifying_question_output_dict()
                )
            elif schema == ExperimentGeneratorLLMOutput:
                mock_s.invoke.return_value = ExperimentGeneratorLLMOutput(
                    **make_valid_experiment_generator_output_dict()
                )
            elif schema == ExperimentCriticLLMOutput:
                mock_s.invoke.return_value = ExperimentCriticLLMOutput(
                    **make_valid_experiment_critic_output_dict()
                )
            elif schema == StartupDiagnosisSnapshotLLMOutput:
                mock_s.invoke.return_value = StartupDiagnosisSnapshotLLMOutput(
                    **make_valid_diagnosis_snapshot_output_dict()
                )
            elif schema == FounderTurnInterpretationLLMOutput:
                mock_s.invoke.return_value = FounderTurnInterpretationLLMOutput(
                    **make_valid_founder_turn_interpretation_output_dict()
                )
            elif schema == AdvisorResponseLLMOutput:
                mock_s.invoke.return_value = AdvisorResponseLLMOutput(
                    **make_valid_advisor_response_output_dict()
                )
            return mock_s

        mock_llm.with_structured_output.side_effect = mock_with_structured
        mocker.patch("startary.graph.nodes.get_llm_client", return_value=mock_llm)
        mocker.patch(
            "startary.graph.nodes.get_settings",
            return_value=MagicMock(openai_api_key="sk-test", model_name="gpt-4o"),
        )

        from langgraph.checkpoint.memory import MemorySaver
        graph = build_graph(checkpointer=MemorySaver())
        thread_id = "test-multi-turn-flow"
        cfg = {"configurable": {"thread_id": thread_id}}

        initial_state: StartaryGraphState = {
            "project_id": thread_id,
            "phase": WorkflowPhase.UNDERSTAND,
            "iteration": 1,
            "raw_context": SCENARIO_B_MYNERA_SPARSE,
        }

        # 1. First invocation: halts at interrupt
        graph.invoke(initial_state, cfg)
        snapshot1 = graph.get_state(cfg)
        assert len(snapshot1.tasks[0].interrupts) > 0

        # 2. Resume with founder answer
        graph.invoke(Command(resume=MYNERA_ANSWER_1), cfg)

        # 3. Final state: sufficiency passed → completed to assumption!
        snapshot2 = graph.get_state(cfg)
        assert len(snapshot2.tasks) == 0
        final_values = snapshot2.values
        assert final_values.get("active_assumption") is not None
        assert final_values.get("clarification_turn_count") == 1
        assert len(final_values.get("founder_messages", [])) == 3
