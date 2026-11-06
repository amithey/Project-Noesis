"""Scientific invariant and native-loader checks for the ES1 proposal operator."""
import hashlib
import inspect
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy import sparse

# Repository tests/ imports the operator from scripts/. The prepared host also
# places both files in work/ and supplies that directory on PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
pytest.importorskip('noesis_gpu.controller', reason='Requires the isolated native pilot environment')
from es1_proposal import propose, write_candidate, assess
from noesis.lif import LIFNetwork
from noesis.plasticity import MotorPlasticity
from noesis_gpu.controller import import_checkpoint

if 'bias_correct_reward_error' not in inspect.signature(MotorPlasticity).parameters:
    pytest.skip('Requires pinned core42c2076, not the hub legacy core', allow_module_level=True)


def test_antithetic_signed_proposals_preserve_parent_zero_topology_and_references():
    w, r = np.array([.9, -1.1, 0.]), np.array([1., -1., 0.])
    a, b = propose(w, r, np.array([0, 1, 0]), np.array([1, -1]))
    np.testing.assert_allclose((a+b)/2, w, rtol=0, atol=1e-15)
    np.testing.assert_allclose(a-w, [.02, -.02, 0], rtol=0, atol=1e-15)
    np.testing.assert_array_equal(w, [.9, -1.1, 0])
    assert np.sign(a).tolist() == np.sign(b).tolist() == np.sign(r).tolist()


@pytest.mark.parametrize('w', [[1.49], [float('nan')], [-1.], [0.]])
def test_invalid_or_bound_crossing_proposals_fail_instead_of_clipping(w):
    with pytest.raises(ValueError):
        propose(w, [1.], np.array([0]), np.array([1]))


def test_native_loader_accepts_only_new_weights_with_original_reference_and_reset_traces(tmp_path):
    matrix = sparse.csr_matrix(([2., -3.], ([0, 1], [2, 2])), shape=(3, 3))
    net = LIFNetwork(matrix)
    learner = MotorPlasticity(net, [2], learning_rate=.2, tau_eligibility_ms=100, max_change=.5,
                              rule='perturbation', normalize_reward_error=True, bias_correct_reward_error=True)
    learner.update([0], 5., .2, perturbation=[1.])
    parent, child = tmp_path/'parent.npz', tmp_path/'candidate.npz'
    learner.save(parent)
    before = hashlib.sha256(parent.read_bytes()).hexdigest()
    write_candidate(parent, child, [2.04, -2.94])
    restored = LIFNetwork(matrix)
    import_checkpoint(restored, child)
    np.testing.assert_array_equal(restored.data, [2.04, -2.94])
    with np.load(child, allow_pickle=False) as z:
        np.testing.assert_array_equal(z['initial_weights'], [2., -3.])
        assert not z['activity'].any() and not z['eligibility'].any() and not z['reward_normalization_mass']
    assert hashlib.sha256(parent.read_bytes()).hexdigest() == before
    with pytest.raises(FileExistsError):
        write_candidate(parent, child, [2.04, -2.94])


def row(standing=.95, slip=2., heading=0., net=0., cycles=10, tripod=.1):
    return dict(standing_fraction=standing, mean_stance_slip_mm_s=slip, heading_forward_distance_mm=heading,
                net_forward_displacement_mm=net, total_leg_steps=cycles,
                tripod_coordination=dict(tripod_coordination_score=tripod))


def test_progress_cannot_override_a_single_seed_retention_failure():
    parent = {s: row() for s in (10006,10007,10010,10011)}
    child = {s: row(heading=.5, net=.5) for s in parent}
    assert assess(parent, child, list(parent))['passed']
    child[10011] = row(standing=.89, heading=9., net=9.)
    assert not assess(parent, child, list(parent))['passed']
    child[10011] = row(slip=2.01, heading=9., net=9.)
    assert not assess(parent, child, list(parent))['passed']


def test_missing_or_nonfinite_evidence_is_not_a_pass():
    with pytest.raises(ValueError):
        assess({10006:row()}, {}, [10006])
    with pytest.raises(ValueError):
        assess({10006:row()}, {10006:row(heading=float('nan'))}, [10006])


def test_less_backward_is_not_forward_progress():
    assert not assess({10006:row(heading=-1.,net=-1.)},
                      {10006:row(heading=-.5,net=-.5)},[10006])['passed']
