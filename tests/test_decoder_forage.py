import numpy as np
import pytest

from noesis.experiments.decoder_forage import (
    CONDITIONS,
    LEVELS,
    RATIOS,
    antenna_rates,
    decode,
    fit_decoder,
    shuffle_block_labels,
)


def test_antenna_rates_left_stronger_and_capped():
    left, right = antenna_rates(0.15, 1.5)
    assert left > right > 0
    assert antenna_rates(100.0, 1.0) == (200.0, 200.0)
    with pytest.raises(ValueError):
        antenna_rates(0.1, 0.0)


def test_design_covers_both_sides_and_hungry_range():
    assert min(RATIOS) < 1 < max(RATIOS) and max(LEVELS) > 1.0
    assert {"shuffled_decoder", "reversed_sign", "shuffled_sides", "no_gradient", "diffusion_only"} <= set(CONDITIONS)


def test_fit_decoder_recovers_sign_on_new_rows():
    rng = np.random.default_rng(0)
    y = np.tile([1, -1], 50)
    x = rng.normal(size=(100, 30))
    x[:, :4] += 2.0 * y[:, None]
    dec = fit_decoder(x, y, 0.1)
    xt = rng.normal(size=(20, 30))
    yt = np.tile([1, -1], 10)
    xt[:, :4] += 2.0 * yt[:, None]
    acc = np.mean([np.sign(decode(dec, r)) == t for r, t in zip(xt, yt, strict=True)])
    assert acc > 0.9


def test_shuffle_block_labels_keeps_block_constant_and_level_balance():
    block = np.repeat(np.arange(6), 5)
    level = np.repeat([0, 0, 0, 1, 1, 1], 5)
    y = np.repeat([1, -1, 1, 1, -1, -1], 5)
    out = shuffle_block_labels(y, block, level, 3)
    for b in range(6):
        assert len(set(out[block == b])) == 1
    for lv in (0, 1):
        assert sorted(set(out[level == lv])) == sorted(set(y[level == lv]))
        assert out[level == lv].sum() == y[level == lv].sum()
