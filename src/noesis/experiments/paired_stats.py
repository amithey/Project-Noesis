"""Paired comparison of a candidate against the original weights over development seeds.

Used by ``scripts/dev_eval.py``. The primary criterion is fixed in advance (see ``verdict``): the paired mean
difference in standing fraction must have a 95% bootstrap interval above zero AND the candidate's mean standing must be at
least 0.75. All numbers are descriptive of the listed seeds; nothing here makes the seeds independent draws of anything.
"""

from __future__ import annotations

from math import comb

import numpy as np


def paired_bootstrap_ci(diffs, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """(mean, lower, upper) of the mean of paired differences, resampling seeds with replacement."""
    d = np.asarray(diffs, float)
    if d.ndim != 1 or len(d) < 2 or not np.all(np.isfinite(d)):
        raise ValueError("need at least two finite paired differences")
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(d.mean()), float(lo), float(hi)


def sign_test_greater(diffs) -> dict:
    """Exact one-sided sign test for 'candidate better' (ties dropped)."""
    d = [x for x in np.asarray(diffs, float) if x != 0]
    n, k = len(d), sum(x > 0 for x in d)
    p = 1.0 if n == 0 else sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return {"n_nonzero": n, "n_better": int(k), "p_one_sided": float(p)}


def verdict(candidate_mean: float, ci_lower: float, min_mean: float = 0.75) -> str:
    """Pre-registered rule: 'improved' needs the interval above zero and a candidate mean of at least ``min_mean``."""
    if ci_lower > 0 and candidate_mean >= min_mean:
        return "improved"
    if ci_lower > 0:
        return "better_but_below_threshold"
    return "no_evidence_of_improvement"


def compare(candidate: dict[int, float], original: dict[int, float]) -> dict:
    """Standing-fraction comparison over the seeds present in both dictionaries."""
    seeds = sorted(set(candidate) & set(original))
    if len(seeds) < 2:
        raise ValueError("need at least two common seeds")
    diffs = [candidate[s] - original[s] for s in seeds]
    mean, lo, hi = paired_bootstrap_ci(diffs)
    cand = np.array([candidate[s] for s in seeds])
    orig = np.array([original[s] for s in seeds])
    return {"seeds": seeds, "diff_mean": mean, "diff_ci95": [lo, hi], "diff_sd": float(np.std(diffs, ddof=1)),
            "candidate_mean": float(cand.mean()), "original_mean": float(orig.mean()),
            "candidate_ge_0.9": int((cand >= 0.9).sum()), "original_ge_0.9": int((orig >= 0.9).sum()),
            "sign_test": sign_test_greater(diffs), "verdict": verdict(float(cand.mean()), lo)}
