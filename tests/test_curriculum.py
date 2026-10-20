import copy

import pytest

from noesis.curriculum import RESERVED_SEEDS, phase_decision, phase_jobs
from noesis.embodied_metrics import locomotion_reward
from noesis.experiments.curriculum_training import finish_round


def row(seed=10000, **kwargs):
    return dict(seed=seed, status='completed', simulated_seconds=5., learning=False,
                motor_exploration_mv_ms=0., training_fall_seconds=0.,
                standing_fraction=.95, heading_forward_distance_mm=2.,
                leg_step_cycles={'LF': 2, 'LM': 2, 'LH': 2, 'RF': 2, 'RM': 0, 'RH': 0},
                mean_stance_slip_mm_s=5., **kwargs)


def pairs():
    original = [row(10000), row(10001)]
    candidate = copy.deepcopy(original)
    for r in original:
        r['mean_stance_slip_mm_s'] = 10.
    return original, candidate


def test_stages_require_both_frozen_probes_and_no_final_seed_leakage():
    for stage in range(4):
        for r in range(12):
            jobs = phase_jobs(stage, r)
            assert not RESERVED_SEEDS & {j['seed'] for j in jobs}
            assert sum(j['learning'] for j in jobs) == 6
            assert all(j['arena'] == ('obstacle' if stage == 3 else 'quiet') for j in jobs)
    original, candidate = pairs()
    assert phase_decision(0, original, candidate)['accepted']
    assert not phase_decision(0, original[:1], candidate[:1])['accepted']
    assert not phase_decision(0, original, candidate[::-1])['accepted']


@pytest.mark.parametrize('field,value', [('standing_fraction', .8999), ('learning', True),
    ('motor_exploration_mv_ms', 1.5), ('training_fall_seconds', .1),
    ('simulated_seconds', 4.99), ('status', 'interrupted'), ('standing_fraction', float('nan'))])
def test_bad_frozen_probe_cannot_advance(field, value):
    original, candidate = pairs()
    candidate[1][field] = value
    assert not phase_decision(0, original, candidate)['accepted']


def test_displacement_and_leg_movement_are_insufficient_without_grip():
    original, candidate = pairs()
    assert phase_decision(1, original, candidate)['accepted']
    assert phase_decision(2, original, candidate)['accepted']
    candidate[0]['mean_stance_slip_mm_s'] = 10.01
    assert not phase_decision(1, original, candidate)['accepted']
    candidate[0]['mean_stance_slip_mm_s'] = 8.01
    assert not phase_decision(2, original, candidate)['accepted']
    candidate[0]['mean_stance_slip_mm_s'] = 5.
    candidate[0]['leg_step_cycles'] = {}
    assert not phase_decision(1, original, candidate)['accepted']


def test_obstacle_gate_needs_complete_response_and_matched_ablations():
    original, candidate = pairs()
    ablated = copy.deepcopy(candidate)
    for r in candidate:
        r.update(obstacle_observation_complete=True, obstacle_response_seconds=.8,
                 obstacle_contact_seconds=.3)
    for r in ablated:
        r['obstacle_contact_seconds'] = .8
    assert phase_decision(3, original, candidate, ablated)['accepted']
    assert not phase_decision(3, original, candidate)['accepted']
    candidate[1]['obstacle_response_seconds'] = 1.01
    assert not phase_decision(3, original, candidate, ablated)['accepted']


def reward(objective, speed=1., slip=0., support=6, up=.9):
    return locomotion_reward([0, 0, 1], [.005 * speed, 0, 1], up, [0.] * 42, .005,
        [1, 0, 0], [.1] * support + [0.] * (6 - support), [[slip, 0, 0]] * 6,
        objective=objective)


def test_standing_does_not_reward_displacement_and_step_credit_is_grounded():
    standing, step = 'curriculum-standing-v7', 'curriculum-gripped-step-v8'
    assert reward(standing, speed=0)['reward'] == reward(standing, speed=10)['reward']
    assert reward(standing, up=.1)['reward'] < 0
    assert reward(step, speed=1)['reward'] > reward(step, speed=0)['reward']
    assert reward(step, speed=-1)['reward'] < reward(step, speed=0)['reward']
    assert reward(step, slip=10)['reward'] < reward(step, slip=0)['reward']
    for kwargs in ({'support': 2}, {'up': .69}, {'speed': 31}):
        assert reward(step, **kwargs)['grounded_progress'] == 0
    assert reward(step, speed=0)['grounded_progress'] == 0


def test_progression_records_anchor_but_never_promotes_and_halts_failed_stage():
    original, candidate = pairs()
    state = {'round': 0, 'stage': 0, 'stage_rounds': 0, 'round_jobs': [], 'history': [],
                 'latest_training_checkpoint': 'last.npz', 'best_validated_checkpoint': None,
                 'status': 'pending', 'model_promoted': False}
    state['round_jobs'] = [{'learning': False, 'model': m, 'summary': r}
                           for m, rows in [('original', original), ('candidate', candidate)] for r in rows]
    assert finish_round(state)['accepted']
    assert state['stage'] == 1 and state['stage_passed_checkpoint'] == 'last.npz'
    assert state['best_validated_checkpoint'] is None and not state['model_promoted']
    for _ in range(3):
        assert not finish_round(state)['accepted']
    assert state['status'] == 'stage_not_learned' and state['stage'] == 1
