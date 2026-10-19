import numpy as np
import pytest

from noesis.experiments.paired_stats import compare, paired_bootstrap_ci, sign_test_greater, verdict

V25_CANDIDATE = {10: 0.934, 11: 0.159, 12: 0.202, 13: 0.891, 14: 0.942}
V25_ORIGINAL = {10: 0.442, 11: 0.929, 12: 0.742, 13: 0.863, 14: 0.435}


def test_bootstrap_interval_contains_mean_and_is_deterministic():
    mean, lo, hi = paired_bootstrap_ci([0.1, 0.2, 0.3, 0.4], seed=1)
    assert lo <= mean <= hi and mean == pytest.approx(0.25)
    assert paired_bootstrap_ci([0.1, 0.2, 0.3, 0.4], seed=1) == (mean, lo, hi)


def test_bootstrap_rejects_too_few_or_nonfinite():
    with pytest.raises(ValueError):
        paired_bootstrap_ci([0.1])
    with pytest.raises(ValueError):
        paired_bootstrap_ci([0.1, float("nan")])


def test_sign_test_counts_and_ties():
    assert sign_test_greater([1, 2, 3, 4]) == {"n_nonzero": 4, "n_better": 4, "p_one_sided": pytest.approx(1 / 16)}
    assert sign_test_greater([0, 0])["p_one_sided"] == 1.0
    assert sign_test_greater([1, -1, 1, -1])["p_one_sided"] == pytest.approx(11 / 16)


def test_verdict_rule():
    assert verdict(0.8, 0.1) == "improved"
    assert verdict(0.6, 0.1) == "better_but_below_threshold"
    assert verdict(0.9, -0.05) == "no_evidence_of_improvement"


def test_known_v25_data_gives_the_documented_numbers():
    out = compare(V25_CANDIDATE, V25_ORIGINAL)
    assert out["diff_mean"] == pytest.approx(-0.0566, abs=1e-3)
    assert out["diff_sd"] == pytest.approx(0.585, abs=0.005)
    assert out["candidate_ge_0.9"] == 2 and out["original_ge_0.9"] == 1
    assert out["verdict"] == "no_evidence_of_improvement"
    assert np.isfinite(out["diff_ci95"]).all()
