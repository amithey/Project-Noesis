"""Safety checks for audited control reuse between rejected training cycles."""
import copy
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    'continue_young_fly', Path(__file__).resolve().parents[1]/'scripts'/'continue_young_fly.py')
continuation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(continuation)


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(continuation, 'source_hash', lambda: 'same-source')
    monkeypatch.setattr(continuation, 'input_manifest', lambda data: {'files': {}, 'versions': {}})
    output = tmp_path/'run'
    output.mkdir()
    arrays = {'topology': np.array([1]), 'motor_indices': np.array([0]),
                  'initial_weights': np.array([1.]), 'parameters': np.array([.2])}
    checkpoint = output/'trained.npz'
    np.savez(checkpoint, **arrays, weights=np.array([1.1]))
    checkpoint.with_suffix('.model.json').write_text('{}')
    jobs = []
    for seed in (10, 11):
        path = output/f'original-{seed}.npz'
        np.savez(path, **arrays, weights=arrays['initial_weights'])
        path.with_suffix('.model.json').write_text('{}')
        summary = {'seed': seed, 'status': 'completed', 'simulated_seconds': 5., 'learning': False,
                       'motor_exploration_mv_ms': 0., 'checkpoint': str(path), 'awake_samples': 990,
                       'mean_reward': .1, 'mean_stance_slip_mm_s': 5., 'standing_fraction': .9,
                       'leg_step_cycles': {'LF': 2, 'LM': 2, 'LH': 2, 'RF': 2},
                       'heading_forward_distance_mm': 2.}
        path.with_suffix('.json').write_text(json.dumps(summary))
        with path.with_suffix('.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['time_s', 'awake'])
            writer.writeheader()
            writer.writerows({'time_s': (i+1)*.005, 'awake': int(i >= 10)} for i in range(1000))
        jobs.append({'id': f'validation-quiet-original-{seed}', 'stage': 'validation',
                         'arena': 'quiet', 'model': 'original', 'seed': seed, 'summary': summary})
    candidate = copy.deepcopy(jobs[0])
    candidate.update(id='validation-quiet-candidate-10', model='candidate')
    candidate['summary'].update(mean_reward=.2, mean_stance_slip_mm_s=6.)
    jobs.append(candidate)
    report = {'schema': 3, 'status': 'rejected_validation', 'cycle': 0, 'jobs': jobs,
                  'settings': {'output': str(output), 'data': str(tmp_path/'data'), 'source_sha256': 'same-source',
                                'seed': 0, 'episodes': 1, 'validation_seeds': [10, 11], 'heldout_seeds': [200, 201, 202]},
                  'decision': {'accepted': False}, 'attempts': [], 'last_training_checkpoint': str(checkpoint),
                  'best_validated_checkpoint': None}
    path = output/'report.json'
    path.write_text(json.dumps(report))
    return path, report


def test_dry_run_and_continuation_keep_training_and_validation_separate(ledger):
    path, report = ledger
    before = path.read_bytes()
    new, audit = continuation.prepare_continuation(path, terminal_verified=True, dry_run=True)
    assert path.read_bytes() == before
    assert not (path.parent/'original-control-inputs.json').exists()
    assert new['cycle'] == 1 and new['settings']['seed'] == 1
    assert new['status'] == 'pending' and new['decision'] is None
    assert new['last_training_checkpoint'] == report['last_training_checkpoint']
    assert new['best_validated_checkpoint'] is None
    assert [j['model'] for j in new['jobs']] == ['original', 'original']
    assert all(j['original_execution_cycle'] == 0 for j in new['jobs'])
    assert new['history'][0]['jobs'] == report['jobs']
    assert audit['candidate_or_heldout_jobs_reused'] == []
    continuation.prepare_continuation(path, terminal_verified=True)
    assert json.loads(path.read_text())['settings']['seed'] == 1


@pytest.mark.parametrize('failure', ['terminal', 'active', 'source', 'partial', 'weights', 'manifest', 'tampered'])
def test_refuses_unverified_or_changed_controls(ledger, monkeypatch, failure):
    path, report = ledger
    if failure == 'active':
        report['attempts'] = [{'status': 'running'}]
    elif failure == 'source':
        monkeypatch.setattr(continuation, 'source_hash', lambda: 'changed')
    elif failure == 'partial':
        report['jobs'][0]['summary']['simulated_seconds'] = 4.9
    elif failure == 'weights':
        original = Path(report['jobs'][0]['summary']['checkpoint'])
        with np.load(original) as saved:
            arrays = {key: saved[key] for key in saved.files}
        arrays['weights'] = np.array([1.1])
        np.savez(original, **arrays)
    elif failure == 'manifest':
        (path.parent/'original-control-inputs.json').write_text('{"changed": true}')
    elif failure == 'tampered':
        report['jobs'][0]['cache_artifact_hashes'] = {'wrong': 'hash'}
    path.write_text(json.dumps(report))
    before = path.read_bytes()
    with pytest.raises(ValueError):
        continuation.prepare_continuation(path, terminal_verified=failure != 'terminal')
    assert path.read_bytes() == before


def test_training_seed_skips_reserved_validation_seeds(ledger):
    path, report = ledger
    report['settings']['seed'] = 9
    path.write_text(json.dumps(report))
    new, _ = continuation.prepare_continuation(path, terminal_verified=True, dry_run=True)
    assert new['settings']['seed'] == 12
