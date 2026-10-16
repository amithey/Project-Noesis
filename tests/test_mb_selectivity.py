import numpy as np
import pytest

from noesis.experiments.mb_selectivity import identity_index, identity_null, jaccard, pn_selectivity


def masks(*rows):
    return [np.array(r, dtype=bool) for r in rows]


def test_jaccard_basic_and_empty():
    a, b = masks([1, 1, 0, 0], [0, 1, 1, 0])
    assert jaccard(a, b) == pytest.approx(1 / 3)
    z = np.zeros(4, bool)
    assert jaccard(z, z) == 0.0


def test_identity_index_separates_distinct_codes():
    a1, a2, b1, b2 = masks([1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1])
    assert identity_index({0: [a1, a2], 1: [b1, b2]}) == pytest.approx(1.0)


def test_identity_index_zero_when_codes_identical():
    m = np.array([1, 0, 1, 0], bool)
    assert identity_index({0: [m, m], 1: [m, m]}) == pytest.approx(0.0)


def test_identity_index_needs_repeats_and_odours():
    m = np.ones(3, bool)
    with pytest.raises(ValueError):
        identity_index({0: [m]})


def test_identity_null_is_significant_only_for_real_structure():
    a, b = np.array([1, 1, 0, 0, 0, 0], bool), np.array([0, 0, 0, 0, 1, 1], bool)
    real = identity_null({0: [a, a], 1: [b, b]}, 200)
    assert real["observed"] == pytest.approx(1.0) and real["p_value"] < 0.35
    same = identity_null({0: [a, a], 1: [a, a]}, 200)
    assert same["observed"] == pytest.approx(0.0) and same["p_value"] > 0.5


def test_pn_selectivity_detects_elevated_own_set():
    rate = np.ones(100)
    rate[:5] = 50.0
    hit = pn_selectivity(rate, np.arange(100), np.arange(5), 300)
    assert hit["own_hz"] == 50.0 and hit["p_value"] < 0.01
    flat = pn_selectivity(np.ones(100), np.arange(100), np.arange(5), 300)
    assert flat["p_value"] > 0.9


def test_pn_selectivity_rejects_empty_own():
    with pytest.raises(ValueError):
        pn_selectivity(np.ones(10), np.arange(10), np.array([], dtype=int), 10)
