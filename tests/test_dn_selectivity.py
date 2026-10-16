import numpy as np
import pytest

from noesis.experiments.dn_selectivity import corr, lateralisation
from noesis.experiments.mb_selectivity import identity_index


def test_corr_handles_constant_vectors():
    assert corr(np.ones(5), np.arange(5.0)) == 0.0
    assert corr(np.arange(5.0), np.arange(5.0)) == pytest.approx(1.0)


def test_identity_index_with_correlation_similarity():
    a, b = np.array([1.0, 2, 3, 4]), np.array([4.0, 3, 2, 1])
    assert identity_index({0: [a, a * 2], 1: [b, b * 3]}, sim=corr) == pytest.approx(2.0)


def _trials(effect):
    rng = np.random.default_rng(0)
    out = []
    for sid in range(3):
        for side in ("left", "right"):
            for _ in range(2):
                rate = rng.normal(0, 0.1, 40)
                if side == "left":
                    rate[:10] += effect   # left-soma DNs respond to left drive
                else:
                    rate[10:20] += effect
                out.append({"set": sid, "side": side, "rate": rate})
    return out


def test_lateralisation_detects_ipsilateral_response():
    left, right, dn = np.arange(10), np.arange(10, 20), np.arange(40)
    hit = lateralisation(_trials(5.0), left, right, 300, dn)
    assert hit["S_hz"] > 8 and hit["p_two_sided"] < 0.05
    none = lateralisation(_trials(0.0), left, right, 300, dn)
    assert abs(none["S_hz"]) < 0.5 and none["p_two_sided"] > 0.05
