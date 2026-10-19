"""Keep background gain matched across train/control jobs and strict on resume."""
import json
from pathlib import Path

import pytest

from noesis.experiments import train_validate as tv


@pytest.mark.parametrize('requested', [None, 40.])
def test_gain_covers_all_jobs_and_cannot_change_on_resume(monkeypatch, tmp_path, requested):
    expected = 60. if requested is None else requested
    assert Path(tv.__file__).is_relative_to(Path(__file__).resolve().parents[1] / 'src')
    calls = []
    interrupted = [True]

    def episode(cn, args, index, checkpoint, *, deadline):
        calls.append((args.bg_weight, args.learn, args.motor_exploration,
                      args.no_environment_senses, args.no_senses, args.no_balance))
        s = {'seed': args.seed, 'status': 'completed', 'simulated_seconds': 5.,
             'learning': args.learn, 'motor_exploration_mv_ms': args.motor_exploration,
             'standing_fraction': .95, 'mean_reward': .3 if checkpoint else .2,
             'mean_stance_slip_mm_s': .1 if checkpoint else .2,
             'heading_forward_distance_mm': 2.,
             'leg_step_cycles': dict.fromkeys(('LF', 'LM', 'LH', 'RF', 'RM', 'RH'), 3),
             'obstacle_observation_complete': True, 'obstacle_response_seconds': .2,
             'obstacle_contact_seconds': .4 if args.no_environment_senses else .1}
        if interrupted[0]:
            s['status'] = 'wall_budget'
            interrupted[0] = False
        saved = args.output / 'episode_000.npz'
        saved.write_bytes(b'fake-checkpoint')
        return s, saved

    monkeypatch.setattr(tv, 'load_training_brain', lambda *a: (None, {}))
    monkeypatch.setattr(tv.lab, 'run_episode', episode)
    output = tmp_path / 'run'
    argv = ['train_validate', '--data', str(tmp_path / 'data'), '--output', str(output), '--episodes', '1']
    if requested is not None:
        argv += ['--bg-weight', str(requested)]
    monkeypatch.setattr('sys.argv', argv)
    tv.main()
    report = output / 'report.json'
    before = report.read_bytes()
    pending = json.loads(before)
    assert pending['settings']['bg_weight'] == expected and not pending['jobs']
    changed = 40 if expected == 60 else 60
    monkeypatch.setattr('sys.argv', argv + ['--resume', '--bg-weight', str(changed)])
    with pytest.raises(SystemExit):
        tv.main()
    assert report.read_bytes() == before
    monkeypatch.setattr('sys.argv', argv + ['--resume'])
    tv.main()
    complete = json.loads(report.read_bytes())
    assert complete['status'] == 'completed' and len(complete['jobs']) == 26
    assert complete['decision']['accepted']
    assert len(calls) == 27 and {c[0] for c in calls} == {expected}
    assert all(noise == (1.5 if learn else 0.) for _, learn, noise, *_ in calls)
    assert all(not no_senses and not no_balance for *_, no_senses, no_balance in calls)
    assert any(ablated and not learn for _, learn, _, ablated, *_ in calls)


@pytest.mark.parametrize('gain', ['-1', 'nan', 'inf'])
def test_invalid_gain_rejected_before_any_output(monkeypatch, tmp_path, gain):
    output = tmp_path / 'run'
    monkeypatch.setattr('sys.argv', ['train_validate', '--data', str(tmp_path / 'data'),
                                  '--output', str(output), '--bg-weight', gain])
    with pytest.raises(SystemExit):
        tv.main()
    assert not output.exists()