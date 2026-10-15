import numpy as np
import pytest
from scipy import sparse

from noesis.lif import LIFNetwork, LIFParams, depression_event


def test_exact_depletion_recovery_and_bounds():
    amplitude, x = depression_event(1., 0, 0, .5, .3, 300.)
    assert amplitude == 1 and x == pytest.approx(.7)
    amplitude, x = depression_event(x, 0, 600, .5, .3, 300.)
    assert amplitude == pytest.approx(1 - .3 / np.e)
    assert x == pytest.approx(amplitude * .7)
    assert 0 <= x <= amplitude <= 1
    assert depression_event(0., 0, 60000, .5, 1., 300.) == (1., 0.)


def network(u):
    return LIFNetwork(sparse.csr_matrix([[0., 350.], [0., 0.]]),
                      LIFParams(dt=.5, depression_u=u, depression_tau_ms=300), seed=3)


def test_depression_reduces_repeated_transmission_without_changing_external_drive():
    a = network(0)
    ta, ia = a.run(500, np.array([0]), 100, 100)
    b = network(.6)
    tb, ib = b.run(500, np.array([0]), 100, 100)
    assert np.array_equal(ta[ia == 0], tb[ib == 0])
    assert 0 < np.count_nonzero(ib == 1) < np.count_nonzero(ia == 1)
    assert b.resources[0] < 1


def test_resource_state_and_delay_survive_run_boundaries():
    one = network(.3)
    t, i = one.run(500, np.array([0]), 100, 100)
    split = network(.3)
    parts = [split.run(ms, np.array([0]), 100, 100) for ms in [1., 99., 200., 200.]]
    assert np.array_equal(t, np.concatenate([p[0] for p in parts]))
    assert np.array_equal(i, np.concatenate([p[1] for p in parts]))
    assert np.array_equal(one.resources, split.resources)
    assert np.array_equal(one.last_release, split.last_release)


def test_disabled_depression_does_not_touch_resources():
    net = network(0)
    net.run(200, np.array([0]), 100, 100)
    assert np.all(net.resources == 1) and np.all(net.last_release == 0)


def test_inhibitory_sign_and_zero_input_are_preserved():
    net = LIFNetwork(sparse.csr_matrix([[0., -150.], [0., 0.]]),
                     LIFParams(dt=.5, depression_u=.3))
    _, i = net.run(200, np.array([0]), 100, 100)
    assert (i == 0).any() and not (i == 1).any()
    silent = network(.3)
    assert len(silent.run(500)[0]) == 0


@pytest.mark.parametrize('u,tau', [(-.1, 300), (1.1, 300), (np.nan, 300), (.3, 0), (.3, np.inf)])
def test_invalid_depression_parameters_fail_early(u, tau):
    with pytest.raises(ValueError):
        LIFNetwork(sparse.csr_matrix((1, 1)), LIFParams(depression_u=u, depression_tau_ms=tau))


def test_excitatory_only_depression_preserves_inhibitory_transmission_exactly():
    W = sparse.csr_matrix([[0., -150., 350.], [0., 0., 0.], [0., 0., 0.]])
    a = LIFNetwork(W, LIFParams(dt=.5), seed=3)
    _, ia = a.run(500, np.array([0]), 100, 100)
    b = LIFNetwork(W, LIFParams(dt=.5, depression_u=.1), seed=3)
    _, ib = b.run(500, np.array([0]), 100, 100)
    assert a.g[1] == b.g[1] and a.v[1] == b.v[1]
    assert np.count_nonzero(ib == 2) < np.count_nonzero(ia == 2)
    np.testing.assert_array_equal(a.data, b.data)
