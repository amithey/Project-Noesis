import numpy as np
from scipy import sparse

from noesis.lif import simulate


def chain(sign: float):
    # neuron 0 -> neuron 1, 150 synapses (a single weak input cannot reach threshold)
    return sparse.csr_matrix(np.array([[0, 150 * sign], [0, 0]], np.float32))


def test_no_input_no_spikes():
    t, _ = simulate(chain(1), 200.0)
    assert len(t) == 0


def test_excitatory_chain_propagates_with_delay():
    t, i = simulate(chain(1), 500.0, stim_idx=np.array([0]), stim_weight=100.0)
    assert (i == 0).any() and (i == 1).any()
    assert t[i == 1].min() > t[i == 0].min()  # downstream fires after upstream


def test_inhibitory_connection_does_not_drive_target():
    _, i = simulate(chain(-1), 500.0, stim_idx=np.array([0]), stim_weight=100.0)
    assert (i == 0).any() and not (i == 1).any()


def test_deterministic_given_seed():
    a = simulate(chain(1), 200.0, np.array([0]), stim_weight=100.0, seed=3)
    b = simulate(chain(1), 200.0, np.array([0]), stim_weight=100.0, seed=3)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_adaptation_lowers_rate_of_strongly_driven_neuron():
    from noesis.lif import LIFParams

    W = sparse.csr_matrix((1, 1))
    drive = {"stim_idx": np.array([0]), "stim_rate_hz": 200.0, "stim_weight": 60.0}
    base, _ = simulate(W, 1000.0, params=LIFParams(da=0.0), **drive)
    adapted, _ = simulate(W, 1000.0, params=LIFParams(da=0.5), **drive)
    assert 0 < len(adapted) < 0.8 * len(base)
