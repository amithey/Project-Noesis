import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from runtime_diagnostic import compare_traces, validate_report, PARENT


def test_only_exact_reviewed_pytest_identity_is_allowed(tmp_path):
    from runtime_diagnostic import allowed_test_worker,command_digest
    from compare_exploration import digest
    source=tmp_path/'test_synthetic.py';source.write_text('def test_small(): pass\n')
    identity=dict(pid=123,create_time=10.,cmdline=['python.exe','-m','pytest','test_synthetic.py','-q'],cwd=str(tmp_path))
    approved=dict(pid=123,create_time=10.,command_sha256=command_digest(identity['cmdline']),cwd=str(tmp_path),test_source=str(source),test_source_sha256=digest(source))
    assert allowed_test_worker(identity,[approved])
    assert not allowed_test_worker(dict(identity,pid=124),[approved])
    assert not allowed_test_worker(dict(identity,create_time=11.),[approved])
    assert not allowed_test_worker(dict(identity,cmdline=['python.exe','-m','noesis_gpu.train']),[approved])
    assert not allowed_test_worker(dict(identity,cwd=str(tmp_path.parent)),[approved])
    source.write_text('def test_changed(): pass\n')
    assert not allowed_test_worker(identity,[approved])


def report():
    return dict(status='completed_transport_pilot', learning=False, motor_exploration=0,
                weights_unchanged=True, completed_control_chunks=1000,
                simulated_seconds_per_world=5, checkpoint_sha256=PARENT, seeds=[10006],
                worlds=1, backend='cpu', controller='brain', setup_wall_seconds=1,
                loop_wall_seconds=2, total_wall_seconds=3)


@pytest.mark.parametrize('key,value', [('completed_control_chunks', 316), ('learning', True),
    ('weights_unchanged', False), ('seeds', [10]), ('checkpoint_sha256', 'wrong'),
    ('total_wall_seconds', float('nan'))])
def test_reject_invalid_repeat(key, value):
    candidate = report()
    candidate[key] = value
    with pytest.raises(ValueError):
        validate_report(candidate)


def test_accept_complete_frozen():
    validate_report(report())


def test_trace_mismatch_and_nonfinite(tmp_path):
    a, b = tmp_path/'a.npz', tmp_path/'b.npz'
    values = np.zeros((1000, 1, 3))
    np.savez(a, position=values)
    np.savez(b, position=values)
    assert compare_traces(a, b)['position']['exact_equal']
    values[-1, 0, 0] = 1
    np.savez(b, position=values)
    assert not compare_traces(a, b)['position']['exact_equal']
    values[0, 0, 0] = np.nan
    np.savez(b, position=values)
    with pytest.raises(ValueError, match='Nonfinite'):
        compare_traces(a, b)


def test_partial_trace(tmp_path):
    a, b = tmp_path/'a.npz', tmp_path/'b.npz'
    np.savez(a, position=np.zeros((316, 1, 3)))
    np.savez(b, position=np.zeros((316, 1, 3)))
    with pytest.raises(ValueError, match='Incomplete'):
        compare_traces(a, b)
