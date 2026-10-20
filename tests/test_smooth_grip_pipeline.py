"""Reward transfer and evaluation freeze through the existing circuit pipeline."""
import json

import numpy as np
import pytest

from noesis.experiments import learning_lab as lab
from noesis.experiments import train_validate as tv
from test_circuit_pipeline import configure
from test_learning_lab import lab_environment as lab_environment  # noqa: PLC0414

V5 = 'standing-gate-linear-slip-v5'
V6 = 'standing-gate-smooth-grip-v6'


def test_v5_to_v6_import_preserves_circuit_bank_and_frozen_has_no_exploration(lab_environment):
    env = lab_environment
    a = configure(env)
    a.learning_scope = 'motor-circuit-v1'
    a.motor_exploration = 1.5
    _, old = lab.run_episode(env.cn, a, 0, None)
    learner = env.learners[-1]
    learner.net.data[learner.edge_indices] *= 1.1
    learner.save(old)
    with np.load(old) as bank:
        previous = bank['weights'].copy()
        references = bank['initial_weights'].copy()
    a.grounded_objective = V6
    a.learn = False
    a.motor_exploration = 0.
    with pytest.raises(ValueError, match='context mismatch'):
        lab.run_episode(env.cn, a, 1, old)
    a.import_weights = True
    summary, new = lab.run_episode(env.cn, a, 2, old)
    assert summary['reward_objective'] == V6 and not summary['learning']
    assert summary['motor_exploration_mv_ms'] == 0
    assert summary['learning_scope'] == 'motor-circuit-v1'
    assert 'current_idx' not in env.networks[-1].last_drive
    context = json.loads(new.with_suffix('.model.json').read_text())
    assert context['schema'] == 9 and context['reward_objective'] == V6
    with np.load(new) as bank:
        assert bank['version'] == 4
        np.testing.assert_array_equal(bank['weights'], previous)
        np.testing.assert_array_equal(bank['initial_weights'], references)
        for key in ('activity', 'eligibility', 'reward_baseline', 'reward_variance', 'reward_normalization_mass'):
            assert not bank[key].any()
    assert all(s.closed for s in env.simulations)


def test_v6_reaches_all_frozen_controls_and_blocks_resume_with_old_reward(monkeypatch, tmp_path):
    calls = []

    def episode(cn, args, index, checkpoint, *, deadline):
        calls.append(vars(args).copy())
        summary = {'seed': args.seed, 'status': 'completed', 'simulated_seconds': 5.,
                   'learning': args.learn, 'motor_exploration_mv_ms': args.motor_exploration,
                   'standing_fraction': .95, 'mean_reward': .3 if checkpoint else .2,
                   'mean_stance_slip_mm_s': .1 if checkpoint else .2,
                   'heading_forward_distance_mm': 2., 'leg_step_cycles': dict.fromkeys(('LF', 'LM', 'LH', 'RF', 'RM', 'RH'), 3),
                   'obstacle_observation_complete': True, 'obstacle_response_seconds': .2,
                   'obstacle_contact_seconds': .4 if args.no_environment_senses else .1}
        cp = args.output / 'episode_000.npz'
        cp.write_bytes(b'orchestration-only')
        return summary, cp

    monkeypatch.setattr(tv, 'load_training_brain', lambda *a: (None, {}))
    monkeypatch.setattr(tv.lab, 'run_episode', episode)
    output = tmp_path / 'run'
    argv = ['train_validate', '--data', str(tmp_path / 'data'), '--output', str(output),
            '--episodes', '1', '--learning-scope', 'motor-circuit-v1',
            '--normalize-reward-error', '--bias-correct-reward-error',
            '--grounded-objective', V6, '--training-fall-seconds', '.1']
    monkeypatch.setattr('sys.argv', argv)
    tv.main()
    reportpath = output / 'report.json'
    report = json.loads(reportpath.read_text())
    assert report['status'] == 'completed' and len(report['jobs']) == 26
    assert all(c['grounded_objective'] == V6 for c in calls)
    frozen = [c for c in calls if not c['learn']]
    assert len(frozen) == 25
    assert all(c['motor_exploration'] == c['training_fall_seconds'] == 0 for c in frozen)
    assert all(not c['no_senses'] and not c['no_balance'] for c in frozen)
    assert any(c['no_environment_senses'] for c in frozen)
    before = reportpath.read_bytes()
    monkeypatch.setattr('sys.argv', [V5 if v == V6 else v for v in argv] + ['--resume'])
    with pytest.raises(SystemExit):
        tv.main()
    assert reportpath.read_bytes() == before and len(calls) == 26
