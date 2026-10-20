import json
import time

import numpy as np
import pytest

from noesis.curriculum import phase_jobs
from noesis.experiments import curriculum_training as ct
from noesis.experiments import learning_lab as lab
from test_circuit_pipeline import configure
from test_learning_lab import lab_environment as lab_environment  # noqa: PLC0414


@pytest.mark.parametrize('objective', ['curriculum-standing-v7', 'curriculum-gripped-step-v8'])
def test_context_transfer_preserves_weights_references_and_frozen_currents(lab_environment, objective):
    env = lab_environment
    args = configure(env)
    args.learning_scope = 'motor-circuit-v1'
    args.grounded_objective = 'standing-gate-smooth-grip-v6'
    _, previous = lab.run_episode(env.cn, args, 0, None)
    with np.load(previous) as bank:
        weights, refs = bank['weights'].copy(), bank['initial_weights'].copy()
    args.grounded_objective = objective
    args.learn, args.motor_exploration, args.import_weights = False, 0., True
    summary, new = lab.run_episode(env.cn, args, 1, previous)
    assert not summary['learning'] and summary['motor_exploration_mv_ms'] == 0
    assert summary['reward_objective'] == objective
    assert 'current_idx' not in env.networks[-1].last_drive
    with np.load(new) as bank:
        np.testing.assert_array_equal(bank['weights'], weights)
        np.testing.assert_array_equal(bank['initial_weights'], refs)
        for key in ('activity', 'eligibility', 'reward_baseline', 'reward_variance', 'reward_normalization_mass'):
            assert not bank[key].any()


def test_partial_training_restarts_from_completed_parent_and_probes_stay_frozen(tmp_path, monkeypatch):
    parent = tmp_path / 'parent.npz'
    parent.write_bytes(b'initial')
    output = tmp_path / 'output'
    output.mkdir()
    state = {'round': 0, 'stage': 0, 'round_jobs': [], 'attempts': [], 'completed_training_seconds': 0.,
                 'latest_training_checkpoint': str(parent), 'latest_training_sha256': ct.sha(parent)}
    calls = []

    def partial(cn, args, episode, checkpoint, *, deadline):
        calls.append((args, checkpoint))
        cp = args.output / 'episode_000.npz'
        cp.write_bytes(b'partial')
        return {'status': 'interrupted:wall', 'simulated_seconds': .1}, cp

    monkeypatch.setattr(lab, 'run_episode', partial)
    jobs = phase_jobs(0, 0)
    assert not ct.run_round(state, jobs, output, tmp_path / 'data', None, lab, time.monotonic() + 100)
    assert state['latest_training_checkpoint'] == str(parent)
    assert state['round_jobs'] == [] and state['completed_training_seconds'] == 0

    def completed(cn, args, episode, checkpoint, *, deadline):
        calls.append((args, checkpoint))
        cp = args.output / 'episode_000.npz'
        cp.write_bytes(b'complete')
        return {'status': 'completed', 'simulated_seconds': 5.}, cp

    monkeypatch.setattr(lab, 'run_episode', completed)
    assert ct.run_round(state, jobs, output, tmp_path / 'data', None, lab, time.monotonic() + 100)
    assert calls[0][1] == calls[1][1] == str(parent)
    assert calls[0][0].output != calls[1][0].output
    assert state['completed_training_seconds'] == 30 and len(state['round_jobs']) == 10
    for args, checkpoint in calls[-4:]:
        assert not args.learn and args.motor_exploration == args.training_fall_seconds == 0
        assert not args.no_senses and not args.no_balance
    snapshot = json.loads((output / 'curriculum.json').read_text())
    assert snapshot['attempts'][0]['status'] == 'interrupted:wall'
    assert snapshot['attempts'][1]['start_sha256'] == ct.sha(parent)


def test_ledger_retries_transient_windows_lock_without_losing_contents(tmp_path, monkeypatch):
    from pathlib import Path

    real_replace = Path.replace
    calls = []

    def sometimes_locked(path, target):
        calls.append(target)
        if len(calls) < 3:
            raise PermissionError('temporary scanner lock')
        return real_replace(path, target)

    monkeypatch.setattr(Path, 'replace', sometimes_locked)
    monkeypatch.setattr(ct.time, 'sleep', lambda seconds: None)
    target = tmp_path / 'curriculum.json'
    ct.atomic_json(target, {'completed': False, 'best_validated_checkpoint': None})
    assert len(calls) == 3
    assert json.loads(target.read_text()) == {'completed': False, 'best_validated_checkpoint': None}


def test_full_curriculum_main_keeps_final_validation_and_promotion_pending(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from noesis import embodiment
    from noesis.curriculum import RESERVED_SEEDS

    parent = tmp_path / 'parent.npz'
    parent.write_bytes(b'initial')
    calls = []

    def episode(cn, args, index, checkpoint, *, deadline):
        calls.append(args)
        cp = args.output / 'episode_000.npz'
        cp.write_bytes(b'weights')
        return {'seed': args.seed, 'status': 'completed', 'simulated_seconds': 5.,
                'learning': args.learn, 'motor_exploration_mv_ms': args.motor_exploration,
                'training_fall_seconds': args.training_fall_seconds,
                'standing_fraction': .95, 'heading_forward_distance_mm': 2.,
                'leg_step_cycles': dict.fromkeys(('LF', 'LM', 'LH', 'RF'), 3),
                'mean_stance_slip_mm_s': 4. if checkpoint else 10.,
                'obstacle_observation_complete': True, 'obstacle_response_seconds': .2,
                'obstacle_contact_seconds': .8 if args.no_environment_senses else .2}, cp

    monkeypatch.setattr(embodiment, 'load_training_brain', lambda *a: (SimpleNamespace(n=1), {}))
    monkeypatch.setattr(lab, 'run_episode', episode)
    output = tmp_path / 'curriculum'
    monkeypatch.setattr('sys.argv', ['curriculum', '--data', str(tmp_path / 'data'),
        '--output', str(output), '--import-weights', str(parent)])
    ct.main()
    state = json.loads((output / 'curriculum.json').read_text())
    assert state['status'] == 'ready_for_final_validation' and len(state['history']) == 4
    assert state['best_validated_checkpoint'] is None and not state['model_promoted']
    assert not state['normal_startup_changed'] and state['completed_training_seconds'] == 120
    assert len(calls) == 42 and not RESERVED_SEEDS & {a.seed for a in calls}
    frozen = [a for a in calls if not a.learn]
    assert all(a.motor_exploration == a.training_fall_seconds == 0 for a in frozen)
    assert all(not a.no_senses and not a.no_balance for a in calls)
    assert len([a for a in frozen if a.no_environment_senses]) == 2
