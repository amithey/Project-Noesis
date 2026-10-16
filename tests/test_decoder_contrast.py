import numpy as np

from noesis.experiments.decoder_contrast import RATIOS, contrast_table, loso_scores, wilson


def test_wilson_interval_contains_estimate_and_widens_for_small_n():
    lo, hi = wilson(8, 10)
    assert lo < 0.8 < hi
    lo2, hi2 = wilson(80, 100)
    assert (hi2 - lo2) < (hi - lo)
    assert wilson(0, 0) == (0.0, 1.0)


def test_ratios_are_symmetric_in_log_space_and_exclude_one():
    logs = sorted(np.round(np.log(RATIOS), 2))
    assert 1.0 not in RATIOS
    assert np.allclose(logs, -np.array(logs[::-1]), atol=0.02)


def test_contrast_table_groups_by_abs_log_ratio():
    block = np.repeat(np.arange(4), 3)
    y = np.repeat([1, -1, 1, -1], 3)
    ratio = np.repeat([2.0, 0.5, 1.25, 0.8], 3)
    scores = np.where(block >= 2, -y * 1.0, y * 1.0)   # perfect for strong contrast, wrong for weak
    rows = {r["abs_log_ratio"]: r for r in contrast_table(scores, y, block, ratio)}
    assert rows[0.7]["window_accuracy"] == 1.0 and rows[0.7]["block_accuracy"] == 1.0
    assert rows[0.2]["window_accuracy"] == 0.0


def test_loso_scores_never_use_the_held_out_group():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(4), 10)
    y = np.tile([1, -1], 20)
    x = rng.normal(size=(40, 20))
    x[:, :3] += 2.0 * y[:, None]
    assert (np.sign(loso_scores(x, y, groups, 0.1)) == y).mean() > 0.85
    x_noise = rng.normal(size=(40, 20))
    assert abs((np.sign(loso_scores(x_noise, y, groups, 0.1)) == y).mean() - 0.5) < 0.3
