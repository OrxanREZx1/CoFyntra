
import uuid
import pytest

from startary.domain.enums import (
    CriticVerdict,
    ExecutionReadiness,
    ExperimentDesignStatus,
    ExperimentStatus,
    ExperimentType,
    NextLearningActionType,
    WorkflowPhase,
)
from startary.domain.experiment import (
    ExecutionHandoff,
    Experiment,
    FallbackLearningAction,
    FounderExperimentSummary,
)
from startary.domain.terminal_state import (
    TerminalStatus,
    can_render_waiting_for_evidence,
    render_terminal_state,
    resolve_founder_facing_status,
)
from startary.graph.builder import (
    route_after_can_execute,
    route_on_experiment_critique,
)
from startary.graph.nodes import (
    handle_experiment_failure_stop,
    increment_iteration,
)


def test_1_accepted_experiment_waiting_for_evidence_works():
    exp_id = str(uuid.uuid4())
    asmp_id = 'asmp-acc-001'

    state = {
        'project_id': 'proj-test-1',
        'iteration': 1,
        'phase': WorkflowPhase.WAITING_FOR_EVIDENCE,
        'active_experiment': {
            'id': exp_id,
            'assumption_id': asmp_id,
            'status': ExperimentStatus.IN_PROGRESS.value,
            'created_iteration': 1,
        },
        'active_execution_handoff': {
            'experiment_id': exp_id,
            'assumption_id': asmp_id,
            'response_text': 'Interview 5 grocery store owners.',
        },
        'can_execute_result': {
            'readiness': ExecutionReadiness.CAN_EXECUTE.value,
            'reason': 'Founder has customer access.',
        },
    }

    assert can_render_waiting_for_evidence(state) is True
    status = resolve_founder_facing_status(state)
    assert status == TerminalStatus.WAITING_FOR_EVIDENCE

    rendered = render_terminal_state(state)
    assert 'STARTARY — WAITING FOR EVIDENCE' in rendered
    assert 'Your experiment is ready.' in rendered


def test_2_critic_revise_then_accept_waiting_for_evidence_works():
    state_revise = {
        'critic_revision_count': 1,
        'experiment_critic_output': {'verdict': CriticVerdict.REVISE.value, 'issues': ['Clarify metric']},
    }
    assert route_on_experiment_critique(state_revise) == 'generate_experiment'

    state_accept = {
        'critic_revision_count': 2,
        'experiment_critic_output': {'verdict': CriticVerdict.ACCEPT.value, 'issues': []},
    }
    assert route_on_experiment_critique(state_accept) == 'save_final_experiment'

    state_can_exec = {
        'can_execute_result': {'readiness': ExecutionReadiness.CAN_EXECUTE.value},
    }
    assert route_after_can_execute(state_can_exec) == 'compose_execution_handoff'

    exp_id = 'exp-revised-accepted'
    state_terminal = {
        'phase': WorkflowPhase.WAITING_FOR_EVIDENCE,
        'active_experiment': {'id': exp_id, 'status': 'IN_PROGRESS', 'created_iteration': 1},
        'active_execution_handoff': {'experiment_id': exp_id},
        'can_execute_result': {'readiness': 'CAN_EXECUTE'},
    }
    assert can_render_waiting_for_evidence(state_terminal) is True
    assert resolve_founder_facing_status(state_terminal) == TerminalStatus.WAITING_FOR_EVIDENCE
    assert 'STARTARY — WAITING FOR EVIDENCE' in render_terminal_state(state_terminal)


def test_3_critic_revise_twice_activates_fallback_action():
    state = {
        'project_id': 'proj-revise-twice',
        'iteration': 1,
        'critic_revision_count': 2,
        'experiment_critic_output': {
            'verdict': CriticVerdict.REVISE.value,
            'issues': [
                'Decision rule requires unfeasible sample size',
                'Proxy metric lacks direct customer observation',
            ],
        },
        'active_assumption': {
            'id': 'asmp-003',
            'statement': 'Independent grocery store owners will pay for automated ordering.',
            'area': 'Problem',
        },
        'proposed_experiment_output': {
            'objective': 'Test grocery store payment willingness',
            'method': 'Landing page test',
        },
    }

    assert route_on_experiment_critique(state) == 'handle_experiment_failure_stop'

    updates = handle_experiment_failure_stop(state)
    assert updates['learning_input_mode'] == 'FALLBACK'
    assert 'fallback_learning_action' in updates
    fallback = updates['fallback_learning_action']
    assert fallback['assumption_id'] == 'asmp-003'
    assert fallback['target_area'] == 'Problem'
    assert fallback['action_type'] == NextLearningActionType.FALLBACK_LEARNING_ACTION.value


def test_4_revision_exhaustion_sets_phase_waiting_for_learning_input():
    state = {
        'project_id': 'proj-phase-check',
        'iteration': 1,
        'critic_revision_count': 2,
        'experiment_critic_output': {
            'verdict': 'REVISE',
            'issues': ['Unfalsifiable signals'],
        },
        'active_assumption': {
            'id': 'asmp-004',
            'statement': 'Restaurant managers want automated prep scheduling.',
            'area': 'Problem',
        },
    }

    updates = handle_experiment_failure_stop(state)
    assert updates['phase'] == WorkflowPhase.WAITING_FOR_LEARNING_INPUT


def test_5_revision_exhaustion_clears_stale_execution_state():
    stale_exp_id = 'exp-stale-prev'
    state = {
        'project_id': 'proj-clear-stale',
        'iteration': 2,
        'phase': WorkflowPhase.TEST,
        'active_execution_handoff': {
            'experiment_id': stale_exp_id,
            'response_text': 'Old handoff instructions',
        },
        'execution_handoff_text': 'Old compact text',
        'can_execute_result': {
            'readiness': ExecutionReadiness.CAN_EXECUTE.value,
            'reason': 'Old readiness',
        },
        'waiting_response': 'Old waiting response',
        'post_evidence_answer': 'Old answer',
        'active_experiment': {
            'id': stale_exp_id,
            'status': 'READY',
            'created_iteration': 1,
        },
        'experiment_critic_output': {
            'verdict': 'REVISE',
            'issues': ['Could not isolate competitor differentiation'],
        },
        'active_assumption': {
            'id': 'asmp-iter2',
            'statement': 'Merchants switch from pen and paper.',
            'area': 'Existing Alternatives',
        },
    }

    updates = handle_experiment_failure_stop(state)

    assert updates['active_execution_handoff'] is None
    assert updates['execution_handoff_text'] is None
    assert updates['can_execute_result'] is None
    assert updates['waiting_response'] is None
    assert updates['post_evidence_answer'] is None
    assert updates['active_experiment'] is None
    assert updates['phase'] == WorkflowPhase.WAITING_FOR_LEARNING_INPUT


def test_6_revision_exhaustion_prevents_waiting_for_evidence_renderer():
    state = {
        'project_id': 'proj-renderer-test',
        'iteration': 1,
        'can_execute_result': {
            'readiness': ExecutionReadiness.CAN_EXECUTE.value,
            'reason': 'Old readiness before rejection',
        },
        'active_execution_handoff': {
            'experiment_id': 'exp-rejected-01',
            'response_text': 'Run this rejected test',
        },
        'experiment_critic_output': {
            'issues': ['Quality bar failed'],
        },
        'active_assumption': {
            'statement': 'Consumers will pay /month for AI fridge monitoring.',
        },
    }

    updates = handle_experiment_failure_stop(state)
    merged = {**state, **updates}

    assert can_render_waiting_for_evidence(merged) is False

    status = resolve_founder_facing_status(merged)
    assert status == TerminalStatus.EXPERIMENT_DESIGN_NEEDS_REVISION
    assert status != TerminalStatus.WAITING_FOR_EVIDENCE

    output = render_terminal_state(merged)
    assert 'STARTARY — WAITING FOR EVIDENCE' not in output
    assert 'Your experiment is ready.' not in output
    assert 'STARTARY — EXPERIMENT DESIGN NEEDS REVISION' in output
    assert 'Here is the safest next learning action:' in output


def test_7_prior_iteration_accepted_experiment_does_not_leak_into_failed_iteration():
    exp1_id = 'exp-iter-1-accepted'
    asmp1_id = 'asmp-iter-1'

    iter1_state = {
        'project_id': 'proj-fridgeguard',
        'iteration': 1,
        'phase': WorkflowPhase.WAITING_FOR_EVIDENCE,
        'weak_area': 'Problem',
        'active_assumption': {'id': asmp1_id, 'statement': 'Perishable food spoils frequently.'},
        'active_experiment': {
            'id': exp1_id,
            'assumption_id': asmp1_id,
            'status': ExperimentStatus.IN_PROGRESS.value,
            'created_iteration': 1,
        },
        'active_execution_handoff': {
            'experiment_id': exp1_id,
            'response_text': 'Track fridge spoilage for 7 days.',
        },
        'execution_handoff_text': 'Track fridge spoilage for 7 days.',
        'can_execute_result': {
            'readiness': ExecutionReadiness.CAN_EXECUTE.value,
            'reason': 'Founder has access.',
        },
        'founder_evidence': [
            {
                'iteration': 1,
                'assumption_id': asmp1_id,
                'evidence_type': 'USER_OBSERVATION',
                'content': '8 of 10 users had spoiled food weekly.',
                'quality_score': 0.85,
            }
        ],
        'evidence_analysis': {
            'iteration': 1,
            'conclusion': 'SUPPORTED',
            'findings': ['Direct evidence shows spoilage.'],
        },
        'startup_state': {
            'area_knowledge': {
                'Problem': {'status': 'KNOWN', 'confidence': 'HIGH'},
            }
        },
    }

    iter_updates = increment_iteration(iter1_state)
    state_iter2 = {**iter1_state, **iter_updates}

    assert state_iter2['iteration'] == 2
    assert state_iter2['phase'] == WorkflowPhase.LEARN
    assert state_iter2['active_experiment'] is None
    assert state_iter2['active_execution_handoff'] is None
    assert state_iter2['can_execute_result'] is None

    state_iter2['weak_area'] = 'Existing Alternatives'
    state_iter2['active_assumption'] = {
        'id': 'asmp-iter-2',
        'statement': 'Smart-fridge apps do not solve spoilage notifications.',
        'area': 'Existing Alternatives',
    }
    state_iter2['proposed_experiment_output'] = {
        'objective': 'Audit competitor reviews',
        'method': 'Online review analysis',
    }
    state_iter2['experiment_critic_output'] = {
        'verdict': 'REVISE',
        'issues': ['Could not distinguish true gap from review bias'],
    }
    state_iter2['critic_revision_count'] = 2

    fail_updates = handle_experiment_failure_stop(state_iter2)
    final_iter2_state = {**state_iter2, **fail_updates}

    assert final_iter2_state['iteration'] == 2
    assert final_iter2_state['phase'] == WorkflowPhase.WAITING_FOR_LEARNING_INPUT
    assert final_iter2_state['active_experiment'] is None
    assert final_iter2_state['active_execution_handoff'] is None
    assert final_iter2_state['can_execute_result'] is None
    assert can_render_waiting_for_evidence(final_iter2_state) is False

    status = resolve_founder_facing_status(final_iter2_state)
    assert status == TerminalStatus.EXPERIMENT_DESIGN_NEEDS_REVISION
    assert status != TerminalStatus.WAITING_FOR_EVIDENCE

    rendered = render_terminal_state(final_iter2_state)
    assert 'STARTARY — WAITING FOR EVIDENCE' not in rendered
    assert 'Your experiment is ready.' not in rendered
    assert 'STARTARY — EXPERIMENT DESIGN NEEDS REVISION' in rendered

    assert len(final_iter2_state['previous_assumptions']) == 1
    assert final_iter2_state['previous_assumptions'][0]['id'] == asmp1_id
    assert len(final_iter2_state['founder_evidence']) == 1


def test_8_handoff_experiment_id_mismatch_prevents_waiting_for_evidence_rendering():
    current_exp_id = 'exp-current-001'
    stale_handoff_id = 'exp-stale-000'

    state = {
        'phase': WorkflowPhase.WAITING_FOR_EVIDENCE,
        'iteration': 2,
        'active_experiment': {
            'id': current_exp_id,
            'status': ExperimentStatus.IN_PROGRESS.value,
            'created_iteration': 2,
        },
        'active_execution_handoff': {
            'experiment_id': stale_handoff_id,
            'response_text': 'Stale handoff instructions from previous experiment',
        },
        'can_execute_result': {
            'readiness': ExecutionReadiness.CAN_EXECUTE.value,
        },
    }

    assert can_render_waiting_for_evidence(state) is False
    status = resolve_founder_facing_status(state)
    assert status != TerminalStatus.WAITING_FOR_EVIDENCE

    rendered = render_terminal_state(state)
    assert 'STARTARY — WAITING FOR EVIDENCE' not in rendered
