"""Exercise opt-in reward normalization across real episode/checkpoint plumbing."""
import json
from pathlib import Path

import numpy as np
import pytest

from noesis.experiments import learning_lab as lab
from noesis.experiments import train_validate as tv
from test_learning_lab import lab_environment as lab_environment  # noqa: PLC0414 -- pytest fixture re-export


def test_legacy_import_new_birth_and_frozen_credit_state(lab_environment):
    env = lab_environment
    args = env.args
    args.learning_rule = 'perturbation'
    args.normalize_reward_error = True
    args.motor_exploration = 1.5
    old_summary, old = lab.run_episode(env.cn, args, 0, None)
    old_context = json.loads(old.with_suffix('.model.json').read_text())
    assert old_context['schema'] == 7
    assert 'bias_correct_reward_error' not in old_context
    assert 'bias_correct_reward_error' not in old_summary
    with np.load(old) as bank:
        assert bank['version'] == 3
        previous = bank['weights'].copy()
    args.bias_correct_reward_error = True
    with pytest.raises(ValueError, match='model context mismatch'):
        lab.run_episode(env.cn, args, 1, old)
    args.import_weights = True
    # Disable step updates to isolate explicit import from subsequent training.
    args.learning_rate = 0.
    summary, trained = lab.run_episode(env.cn, args, 2, old)
    context = json.loads(trained.with_suffix('.model.json').read_text())
    assert context['schema'] == 8 and context['bias_correct_reward_error'] is True
    assert summary['bias_correct_reward_error'] is True
    with np.load(trained) as bank:
        assert bank['version'] == 4 and float(bank['reward_normalization_mass']) > 0
        np.testing.assert_array_equal(bank['weights'], previous)
    args.import_weights = False
    args.learn = False
    args.motor_exploration = 0.
    frozen, checkpoint = lab.run_episode(env.cn, args, 3, trained)
    assert not frozen['learning'] and frozen['motor_exploration_mv_ms'] == 0
    assert frozen['bias_correct_reward_error'] is True
    with np.load(checkpoint) as bank:
        np.testing.assert_array_equal(bank['weights'], previous)
        for name in ('activity', 'eligibility', 'reward_variance', 'reward_baseline',
                     'reward_normalization_mass'):
            assert not bank[name].any()
    assert all(sim.closed for sim in env.simulations)


def test_invalid_bias_configuration_precedes_body_or_output(lab_environment, monkeypatch, tmp_path):
    env = lab_environment
    env.args.bias_correct_reward_error = True
    with pytest.raises(SystemExit):
        lab.validate_args(lab.parser(), env.args)
    assert not env.simulations
    output = tmp_path/'invalid'
    monkeypatch.setattr('sys.argv', ['train_validate', '--data', str(tmp_path/'data'),
        '--output', str(output), '--bias-correct-reward-error'])
    with pytest.raises(SystemExit):
        tv.main()
    assert not output.exists() and not env.simulations


def test_opt_in_is_shared_by_all_controls_and_resume_is_strict(monkeypatch, tmp_path):
    assert Path(tv.__file__).is_relative_to(Path(__file__).resolve().parents[1]/'src')
    calls = []
    interrupt = [True]

    def episode(cn, args, index, checkpoint, *, deadline):
        calls.append(vars(args).copy())
        summary = {'seed': args.seed, 'status': 'completed', 'simulated_seconds': 5.,
            'learning': args.learn, 'motor_exploration_mv_ms': args.motor_exploration,
            'standing_fraction': .95, 'mean_reward': .3 if checkpoint else .2,
            'mean_stance_slip_mm_s': .1 if checkpoint else .2,
            'heading_forward_distance_mm': 2.,
            'leg_step_cycles': dict.fromkeys(('LF','LM','LH','RF','RM','RH'), 3),
            'obstacle_observation_complete': True, 'obstacle_response_seconds': .2,
            'obstacle_contact_seconds': .4 if args.no_environment_senses else .1}
        if interrupt[0]:
            summary['status'] = 'wall_budget'
            interrupt[0] = False
        saved = args.output/'episode_000.npz'
        saved.write_bytes(b'fake-checkpoint')
        return summary, saved

    monkeypatch.setattr(tv, 'load_training_brain', lambda *a: (None, {}))
    monkeypatch.setattr(tv.lab, 'run_episode', episode)
    output = tmp_path/'run'
    base = ['train_validate', '--data', str(tmp_path/'data'), '--output', str(output),
        '--episodes', '1', '--seed', '74', '--bg-weight', '40', '--normalize-reward-error']
    monkeypatch.setattr('sys.argv', base+['--bias-correct-reward-error'])
    tv.main()
    report = output/'report.json'
    before = report.read_bytes()
    assert not json.loads(before)['jobs']
    monkeypatch.setattr('sys.argv', base+['--resume'])
    with pytest.raises(SystemExit):
        tv.main()
    assert report.read_bytes() == before
    monkeypatch.setattr('sys.argv', base+['--resume', '--bias-correct-reward-error'])
    tv.main()
    result = json.loads(report.read_bytes())
    assert result['status'] == 'completed' and len(result['jobs']) == 26
    assert result['decision']['accepted'] and len(calls) == 27
    assert result['settings']['bias_correct_reward_error'] is True
    for c in calls:
        assert c['bias_correct_reward_error'] and c['normalize_reward_error']
        assert c['bg_weight'] == 40
        assert c['motor_exploration'] == (1.5 if c['learn'] else 0.)
        assert not c['no_senses'] and not c['no_balance']
    assert any(c['no_environment_senses'] and not c['learn'] for c in calls)
