import numpy as np
import pytest

from noesis.experiments.dn_decoder import (
    loso_accuracy,
    permutation_null,
    ridge_predict,
    standardise,
)


def synthetic(signal: float, n_sets=6, per=6, p=40, seed=0):
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_sets), per)
    y = np.tile([1, -1], n_sets * per // 2)
    x = rng.normal(size=(len(y), p))
    x[:, :5] += signal * y[:, None]
    x += rng.normal(size=(n_sets, 1, p))[groups, 0] * 0.5   # odour-set-specific offsets
    return x, y, groups


def test_standardise_uses_training_statistics_and_handles_constants():
    tr = np.array([[1.0, 5.0], [3.0, 5.0]])
    te = np.array([[2.0, 5.0]])
    a, b = standardise(tr, te)
    assert b.tolist() == [[0.0, 0.0]] and np.isfinite(a).all()


def test_ridge_rejects_bad_labels():
    with pytest.raises(ValueError):
        ridge_predict(np.zeros((2, 2)), np.array([0, 1]), np.zeros((1, 2)), 0.1)


def test_decoder_generalises_across_groups_only_when_signal_exists():
    x, y, g = synthetic(2.0)
    assert loso_accuracy(x, y, g, 0.1)["accuracy"] > 0.9
    x0, y0, g0 = synthetic(0.0)
    assert loso_accuracy(x0, y0, g0, 0.1)["accuracy"] < 0.75


def test_permutation_null_separates_signal_from_noise():
    x, y, g = synthetic(2.0)
    hit = permutation_null(x, y, g, 0.1, 40)
    assert hit["p_value"] < 0.1 and hit["null_mean"] < 0.65
    x0, y0, g0 = synthetic(0.0)
    assert permutation_null(x0, y0, g0, 0.1, 40)["p_value"] > 0.1
