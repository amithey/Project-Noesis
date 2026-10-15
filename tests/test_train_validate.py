import pytest

from noesis.experiments.train_validate import promotion_decision


def trials(reward=.2, up=.9, slip=.1, status='completed'):
    return [{'seed': seed, 'status': status, 'mean_reward': reward, 'mean_uprightness': up,
                 'mean_stance_slip_mm_s': slip, 'simulated_seconds': 5., 'learning': False,
                 'motor_exploration_mv_ms': 0., 'standing_fraction': up,
                 'leg_step_cycles': {leg: 2 for leg in ('LF', 'LM', 'LH', 'RF')},
                 'heading_forward_distance_mm': 1.1} for seed in (10, 11)]


def test_candidate_must_improve_without_exploiting_posture_or_slip():
    assert promotion_decision(trials(slip=.2), trials(reward=.3))['accepted']
    assert not promotion_decision(trials(), trials(reward=.1))['accepted']
    assert not promotion_decision(trials(), trials(reward=.3, up=.7))['accepted']
    assert not promotion_decision(trials(), trials(reward=.3, slip=.5))['accepted']
    assert not promotion_decision(trials(), trials(reward=.3, status='physics_error'))['accepted']
    assert not promotion_decision(trials(), trials(reward=float('nan')))['accepted']


def test_rejects_unmatched_or_missing_trials():
    candidate = trials()
    candidate[0]['seed'] = 200
    with pytest.raises(ValueError):
        promotion_decision(trials(), candidate)
    with pytest.raises(ValueError):
        promotion_decision([], [])


def test_pipeline_refuses_to_create_outputs_inside_source_data(monkeypatch, tmp_path):
    from noesis.experiments import train_validate
    data = tmp_path / 'data'
    data.mkdir()
    output = data / 'run'
    monkeypatch.setattr('sys.argv', ['train_validate', '--data', str(data), '--output', str(output)])
    with pytest.raises(SystemExit):
        train_validate.main()
    assert not output.exists()


def test_resume_replays_interrupted_episode_and_keeps_last_separate_from_best(monkeypatch, tmp_path):
    import json

    from noesis.experiments import train_validate as tv
    calls = []
    interrupted = [True]
    fail_candidate = [False]
    def episode(cn, args, index, checkpoint, *, deadline):
        calls.append((args.seed, args.learn, args.motor_exploration, checkpoint,
                      args.no_environment_senses, args.no_senses, args.no_balance))
        summary = trials(reward=.3, slip=.1)[0].copy()
        summary.update(seed=args.seed, learning=args.learn, motor_exploration_mv_ms=args.motor_exploration,
                       obstacle_observation_complete=True, obstacle_response_seconds=.2,
                       obstacle_contact_seconds=.4 if args.no_environment_senses else .1)
        if not args.learn and checkpoint is None:
            summary.update(mean_reward=.2, mean_stance_slip_mm_s=.2)
        if fail_candidate[0] and not args.learn and checkpoint:
            summary['standing_fraction'] = .5
        if interrupted[0]:
            summary['status'] = 'wall_budget'
            interrupted[0] = False
        saved = args.output/'episode_000.npz'
        saved.write_bytes(b'fake-checkpoint')
        return summary, saved
    monkeypatch.setattr(tv, 'load_training_brain', lambda *a: (None, {}))
    monkeypatch.setattr(tv.lab, 'run_episode', episode)
    output = tmp_path/'run'
    argv = ['train_validate', '--data', str(tmp_path/'data'), '--output', str(output), '--episodes', '1']
    monkeypatch.setattr('sys.argv', argv)
    tv.main()
    first = json.loads((output/'report.json').read_text())
    assert first['status'] == 'pending' and first['jobs'] == []
    assert first['last_training_checkpoint'] is None and first['best_validated_checkpoint'] is None
    monkeypatch.setattr('sys.argv', argv+['--resume'])
    tv.main()
    second = json.loads((output/'report.json').read_text())
    assert calls[0][:4] == calls[1][:4] == (0, True, 1.5, None)
    assert second['status'] == 'completed' and second['decision']['accepted']
    best = second['best_validated_checkpoint']
    assert best == second['last_training_checkpoint']
    assert all(not learn and noise == 0 for _, learn, noise, *_ in calls[2:])
    assert all(not no_senses and not no_balance for *_, no_senses, no_balance in calls)
    fail_candidate[0] = True
    tv.main()
    third = json.loads((output/'report.json').read_text())
    assert not third['decision']['accepted']
    assert third['best_validated_checkpoint'] == best
    assert third['last_training_checkpoint'] != best


def test_translation_without_leg_cycles_fails_promotion():
    baseline, candidate = trials(slip=.2), trials(reward=.3)
    for r in candidate:
        r['leg_step_cycles'] = {'LF': 0, 'RF': 0}
    assert not promotion_decision(baseline, candidate)['accepted']


def test_early_rejection_does_not_claim_unrun_controls_and_resume_continues_training(monkeypatch, tmp_path):
    import json

    from noesis.experiments import train_validate as tv
    calls = []
    failing = [True]
    def episode(cn, args, index, checkpoint, *, deadline):
        calls.append((args.seed, args.learn, checkpoint, args.normalize_reward_error))
        summary = trials(reward=.3, slip=.1)[0].copy()
        summary.update(seed=args.seed, learning=args.learn, motor_exploration_mv_ms=args.motor_exploration,
                       obstacle_observation_complete=True, obstacle_response_seconds=.2,
                       obstacle_contact_seconds=.4 if args.no_environment_senses else .1)
        if not args.learn and checkpoint is None:
            summary.update(mean_reward=.2, mean_stance_slip_mm_s=.2)
        if failing[0] and not args.learn and checkpoint:
            summary['standing_fraction'] = .8
        saved = args.output/'episode_000.npz'; saved.write_bytes(b'fake-checkpoint')
        return summary, saved
    monkeypatch.setattr(tv, 'load_training_brain', lambda *a: (None, {}))
    monkeypatch.setattr(tv.lab, 'run_episode', episode)
    output = tmp_path/'run'
    argv = ['train_validate', '--data', str(tmp_path/'data'), '--output', str(output),
            '--episodes', '1', '--early-reject', '--normalize-reward-error']
    monkeypatch.setattr('sys.argv', argv); tv.main()
    first = json.loads((output/'report.json').read_text())
    assert first['status'] == 'rejected_validation' and len(first['jobs']) == 4
    assert not first['decision']['accepted'] and first['best_validated_checkpoint'] is None
    assert not any(j['stage'] == 'heldout' or j['arena'] == 'obstacle' for j in first['jobs'])
    last = first['last_training_checkpoint']; count = len(calls)
    failing[0] = False
    monkeypatch.setattr('sys.argv', argv+['--resume']); tv.main()
    second = json.loads((output/'report.json').read_text())
    assert calls[count][1:3] == (True, last)
    assert second['status'] == 'completed' and len(second['jobs']) == 26
    assert second['decision']['accepted']
    assert all(c[3] for c in calls)
