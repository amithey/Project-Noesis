"""D4 synthetic receipt tests, not evidence of physical learning."""
from pathlib import Path
import sys
import json
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_training_resource_parity import validate_training, exact_archive, TRACE_FIELDS, PARENT, SOURCES
from run_training_resource_parity import admission_window, prior_refusal, CORE, EXPECTED


def report():
    return dict(status='completed_training_birth', seed=1000, worlds=1, backend='cpu', learning=True,
        requested_seconds=5, simulated_seconds=5, completed_chunks=1000, learning_updates=990,
        learning_rate=.2, motor_exploration_mv_ms=1.5, cycle_weight=.1, eligibility_ms=100,
        original_weight_bound=.5, control_ms=5., physics_dt=.0001, birth_ms=50,
        reward_objective='curriculum-gripped-cycle-v9', import_checkpoint_sha256=PARENT,
        noesis_source_sha256=SOURCES['core'])


def test_only_original_complete_training_context_is_accepted():
    validate_training(report())
    for key, value in [('status', 'incomplete_wall_budget'), ('completed_chunks', 999),
                       ('learning_updates', 989), ('learning_rate', .002), ('seed', 1001),
                       ('noesis_source_sha256', 'changed'), ('import_checkpoint_sha256', 'partial')]:
        with pytest.raises(ValueError): validate_training({**report(), key: value})


@pytest.mark.parametrize('change', ['missing', 'truncated', 'different', 'nonfinite', 'dtype', None])
def test_all19_complete_finite_trace_fields_must_match(tmp_path, change):
    reference = {k: np.zeros((1000, 2), dtype=np.float64) for k in TRACE_FIELDS}
    candidate = {k: v.copy() for k, v in reference.items()}
    key = 'command'
    if change == 'missing': del candidate[key]
    if change == 'truncated': candidate[key] = candidate[key][:-1]
    if change == 'different': candidate[key][999, 0] = 1
    if change == 'nonfinite': candidate[key][0, 0] = np.nan
    if change == 'dtype': candidate[key] = candidate[key].astype(np.float32)
    a, b = tmp_path / 'candidate.npz', tmp_path / 'reference.npz'
    np.savez(a, **candidate); np.savez(b, **reference)
    if change:
        with pytest.raises(ValueError): exact_archive(a, b, trace=True)
    else:
        assert len(exact_archive(a, b, trace=True)) == 19


def test_log_indices_and_scalar_timing_are_compared_without_invented_rows(tmp_path):
    a, b = tmp_path / 'candidate.npz', tmp_path / 'reference.npz'
    values = dict(control_chunk_index=np.arange(10, 1000), control_ms=np.array(5.))
    np.savez(a, **values); np.savez(b, **values)
    assert all(exact_archive(a, b).values())
    values['control_chunk_index'] = np.arange(0, 990); np.savez(a, **values)
    with pytest.raises(ValueError): exact_archive(a, b)


def test_prospective_mean_contrast_keeps_original_decision_and_memory_floor():
    samples = [dict(available_gib=3., host_cpu_percent=30.) for _ in range(10)]
    samples[-1]['host_cpu_percent'] = 90.
    assert not admission_window(samples)['passed']
    mean = admission_window(samples, mean_cpu=True)
    assert mean['passed'] and not mean['original_max_rule_passed']
    assert mean['mean_host_cpu_percent'] == 36 and mean['maximum_host_cpu_percent'] == 90
    samples[0]['available_gib'] = 1.99
    assert not admission_window(samples, mean_cpu=True)['passed']
    for row in samples: row.update(available_gib=3., host_cpu_percent=61.)
    assert not admission_window(samples, mean_cpu=True)['passed']
    with pytest.raises(ValueError): admission_window(samples[:-1], mean_cpu=True)


def refusal_fixture(path):
    reference = {'reference-file': 'reference-hash'}
    protocol = dict(experiment='D4 guarded training replay', core_commit=CORE, source_hashes=SOURCES,
        parent_sha256=PARENT, expected_checkpoint_sha256=EXPECTED, reference_files=reference,
        seed=1000, seconds=5, chunks=1000, learning_updates=990, learning_rate=.2,
        cycle_weight=.1, exploration_mv_ms=1.5, aggregate_seconds=420, inner_seconds=300, process_seconds=330)
    result = dict(status='stopped', jobs=[], error='D4 resource preflight refused; no training launched', wall_seconds=10.078)
    samples = [dict(available_gib=3., host_cpu_percent=30.) for _ in range(10)]; samples[-1]['host_cpu_percent'] = 90.
    for name, value in [('protocol.json', protocol), ('result.json', result),
                        ('resource-preflight.json', dict(samples=samples, decision=admission_window(samples)))]:
        (path / name).write_text(json.dumps(value), encoding='utf-8')
    return reference, protocol, result


@pytest.mark.parametrize('change', ['job', 'train-folder', 'source', 'wall', 'already-allocated', None])
def test_mean_followup_only_accepts_unused_zero_job_matching_refusal(tmp_path, change):
    reference, protocol, result = refusal_fixture(tmp_path)
    if change == 'job': result['jobs'] = [dict(status='partial')]
    if change == 'train-folder': (tmp_path / 'train').mkdir()
    if change == 'source': protocol['source_hashes'] = {'core': 'other', 'pilot': 'other'}
    if change == 'wall': result['wall_seconds'] = float('nan')
    if change == 'already-allocated': (tmp_path / 'mean-policy-allocation.json').write_text('{}')
    (tmp_path / 'protocol.json').write_text(json.dumps(protocol), encoding='utf-8')
    (tmp_path / 'result.json').write_text(json.dumps(result), encoding='utf-8')
    if change:
        with pytest.raises(ValueError): prior_refusal(tmp_path, reference)
    else:
        receipt = prior_refusal(tmp_path, reference)
        assert receipt['wall_seconds'] == 10.078 and len(receipt['files']) == 3
