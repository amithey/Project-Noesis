"""Training curricula change experience allocation, never frozen acceptance cases."""
import json

import pytest

from noesis.experiments import train_validate as tv


@pytest.mark.parametrize(('arena', 'expected'), [
    ('graded', ['quiet', 'obstacle', 'obstacle']),
    ('quiet', ['quiet', 'quiet', 'quiet']),
    ('obstacle', ['obstacle', 'obstacle', 'obstacle']),
])
def test_training_scene_choice_preserves_complete_frozen_matrix(arena, expected):
    reference = tv.build_jobs(3, 5, [10, 11], [200, 201, 202])
    jobs = tv.build_jobs(3, 5, [10, 11], [200, 201, 202], arena)
    assert [j['arena'] for j in jobs if j['stage']=='training'] == expected
    assert [j['seed'] for j in jobs if j['stage']=='training'] == [5, 6, 7]
    assert [j for j in jobs if j['stage']!='training'] == reference[3:]
    assert len(reference[3:]) == 25


def test_invalid_training_scene_rejected():
    with pytest.raises(ValueError, match='training arena'):
        tv.build_jobs(3, 5, [10, 11], [200], 'empty')


def test_short_quiet_episodes_chain_full_weights_and_replay_interrupted_episode(monkeypatch, tmp_path):
    calls, interrupted = [], [True]
    def episode(cn, args, index, checkpoint, *, deadline):
        calls.append((args.seed, args.arena, args.learn, args.seconds,
                      args.motor_exploration, checkpoint, args.no_environment_senses))
        summary = {'seed': args.seed, 'status': 'completed', 'simulated_seconds': 5.,
            'learning': args.learn, 'motor_exploration_mv_ms': args.motor_exploration,
            'mean_reward': .2 if checkpoint is None else .3, 'mean_uprightness': .95,
            'standing_fraction': .95, 'mean_stance_slip_mm_s': .2 if checkpoint is None else .1,
            'leg_step_cycles': {'LF': 2, 'LM': 2, 'LH': 2, 'RF': 2},
            'heading_forward_distance_mm': 1.1, 'obstacle_observation_complete': True,
            'obstacle_response_seconds': .2,
            'obstacle_contact_seconds': .4 if args.no_environment_senses else .1}
        if args.learn and args.seed == 7 and interrupted[0]:
            summary.update(status='wall_budget', simulated_seconds=2.)
            interrupted[0] = False
        saved = args.output/'episode_000.npz'
        saved.write_bytes(b'fake-checkpoint')
        return summary, saved
    monkeypatch.setattr(tv, 'load_training_brain', lambda *a: (None, {}))
    monkeypatch.setattr(tv.lab, 'run_episode', episode)
    output = tmp_path/'run'
    argv = ['train_validate', '--data', str(tmp_path/'data'), '--output', str(output),
            '--episodes', '3', '--seconds', '5', '--seed', '5', '--training-arena', 'quiet']
    monkeypatch.setattr('sys.argv', argv)
    tv.main()
    first = json.loads((output/'report.json').read_text())
    assert first['status']=='pending' and len(first['jobs'])==2
    assert first['best_validated_checkpoint'] is None
    last_full = first['last_training_checkpoint']
    assert last_full == str((output/'cycle_000/train-1/attempt_0001/episode_000.npz').resolve())
    assert calls[2][5] == last_full
    monkeypatch.setattr('sys.argv', argv+['--resume'])
    tv.main()
    second = json.loads((output/'report.json').read_text())
    assert calls[2] == calls[3]
    training = [call for call in calls if call[2]]
    assert [call[:5] for call in training] == [
        (5, 'quiet', True, 5., 1.5), (6, 'quiet', True, 5., 1.5),
        (7, 'quiet', True, 5., 1.5), (7, 'quiet', True, 5., 1.5)]
    frozen = [call for call in calls if not call[2]]
    assert len(frozen)==25 and all(call[3:5]==(5., 0.) for call in frozen)
    assert any(call[1]=='obstacle' and call[6] for call in frozen)
    assert second['status']=='completed' and second['decision']['accepted']
    assert second['best_validated_checkpoint']==second['last_training_checkpoint']
