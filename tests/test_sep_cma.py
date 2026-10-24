import json

import numpy as np
import pytest

from noesis.sep_cma import SepCMA


def run(f, n=10, gens=120, pop=12, sigma=0.5, seed=0, noise=0.0):
    es = SepCMA(np.full(n, 2.0), sigma, pop, seed)
    rng = np.random.default_rng(99)
    for _ in range(gens):
        xs = es.ask()
        es.tell(xs, np.array([f(x) + noise * rng.standard_normal() for x in xs]))
    return es


def test_maximises_a_sphere_from_far_away():
    es = run(lambda x: -float(np.sum(x ** 2)))
    assert np.sum(es.m ** 2) < 0.05          # started at 40
    assert np.all(np.isfinite(es.m)) and es.sigma > 0


def test_improves_under_noisy_fitness():
    es = run(lambda x: -float(np.sum(x ** 2)), gens=150, noise=0.5)
    assert np.sum(es.m ** 2) < 3.0           # clearly better than the start (40), noise limits precision


def test_state_roundtrip_is_exact_and_resumes_identically():
    f = lambda x: -float(np.sum(x ** 2))
    a = SepCMA(np.full(6, 1.0), 0.4, 8, seed=3)
    for _ in range(5):
        xs = a.ask()
        a.tell(xs, np.array([f(x) for x in xs]))
    b = SepCMA.from_state(json.loads(json.dumps(a.state())), 8)
    xa, xb = a.ask(), b.ask()
    assert np.allclose(xa, xb) and a.gen == b.gen == 5


def test_rejects_bad_inputs():
    with pytest.raises(ValueError):
        SepCMA([1.0], 0.5, 8)
    with pytest.raises(ValueError):
        SepCMA([1.0, 2.0], 0.5, 2)
    es = SepCMA([0.0, 0.0], 0.5, 6)
    xs = es.ask()
    with pytest.raises(ValueError):
        es.tell(xs, np.array([np.nan] * 6))
