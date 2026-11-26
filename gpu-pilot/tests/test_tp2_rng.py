"""Exact pinned-core RNG matching and informative mismatched control on tiny networks."""
import numpy as np
from scipy import sparse

from noesis.lif import LIFNetwork, LIFParams, _seed
from noesis_gpu.tp2_rng import rng_state


def trajectory(weight, extra=True, read=True):
    w = sparse.csr_matrix(([3., -2.], ([0, 1], [1, 2])), shape=(32, 32))
    net = LIFNetwork(w, LIFParams(dt=.5, da=1), seed=7)
    states, probes, spikes = [], [], []
    for chunk in range(20):
        _seed(chunk + 81)
        idx = np.arange(6 if extra else 0, dtype=np.int64)
        _, emitted = net.run(5, drive_idx=idx, drive_rate_hz=25, drive_weight=weight,
                             background_hz=3, background_weight=40)
        spikes.append(emitted.copy())
        states.append(np.concatenate([net.v, net.g, net.adapt, net.refr]))
        if read:
            probes.append(rng_state())
    return np.asarray(states), probes, spikes


def test_sham_preserves_rng_stream_even_when_neural_response_differs():
    sham, sham_rng, sham_spikes = trajectory(0)
    driven, driven_rng, driven_spikes = trajectory(400)
    np.testing.assert_array_equal(sham_rng, driven_rng)
    assert not np.array_equal(sham, driven)
    assert sum(map(len, driven_spikes)) > sum(map(len, sham_spikes))


def test_removing_extra_indices_changes_the_rng_control():
    _, sham_rng, _ = trajectory(0)
    _, absent_rng, _ = trajectory(0, extra=False)
    assert not np.array_equal(sham_rng, absent_rng)


def test_observational_readout_does_not_change_reseeded_trajectory():
    observed, _, observed_spikes = trajectory(400)
    pristine, _, pristine_spikes = trajectory(400, read=False)
    np.testing.assert_array_equal(observed, pristine)
    assert all(np.array_equal(a, b) for a, b in zip(observed_spikes, pristine_spikes))


def test_state_readout_is_complete_MT_state_and_does_not_consume_it():
    _seed(512)
    a, b = rng_state(), rng_state()
    assert a.shape == (625,) and a.dtype == np.uint32
    np.testing.assert_array_equal(a, b)
