import numpy as np
import pytest

from noesis.reward_readout import (
    EpisodeSpec,
    FeatureLibrary,
    episode_return,
    evaluate,
    run_episode,
    train,
)
from noesis.virtual_agent import AgentParams


def synthetic_library(seed=0, d=12):
    """PN-like features that encode the left/right ratio linearly (feature 0), plus noise; 3 levels x 6 ratios."""
    rng = np.random.default_rng(seed)
    xs, level, ratio, block = [], [], [], []
    b = 0
    for lv in (0.05, 0.3, 2.0):
        for r in (0.5, 0.8, 0.95, 1.05, 1.25, 2.0):
            for _ in range(20):
                v = rng.normal(0, 0.3, d)
                v[0] += 2.0 * np.log(r)           # informative axis
                xs.append(v)
                level.append(lv)
                ratio.append(r)
                block.append(b)
            b += 1
    return FeatureLibrary(np.array(xs), np.array(level), np.array(ratio), np.array(block))


def test_library_returns_rows_of_the_nearest_block():
    lib = synthetic_library()
    rng = np.random.default_rng(1)
    z_hi = np.mean([lib.sample(0.3, 2.0, rng)[0] for _ in range(200)])
    z_lo = np.mean([lib.sample(0.3, 0.5, rng)[0] for _ in range(200)])
    assert z_hi > 0.5 > -0.5 > z_lo
    assert lib.nearest_block(1e9, 1e9) == lib.nearest_block(2.0, 2.0)       # clamped to the library range


def test_library_rejects_bad_input():
    with pytest.raises(ValueError):
        FeatureLibrary(np.zeros(5), np.ones(5), np.ones(5), np.zeros(5))


def test_episode_return_ordering():
    assert episode_return(20, 10, None, 12) == pytest.approx(0.5)
    assert episode_return(20, 3, 6.0, 12) > episode_return(20, 3, 11.0, 12) > episode_return(20, 3, None, 12)


def test_zero_readout_is_diffusion_only_and_reports_the_noise_statistic():
    lib = synthetic_library()
    rng = np.random.default_rng(2)
    r = run_episode(lib, np.zeros(lib.n_features), 0.0, EpisodeSpec(), AgentParams(), rng, noise_sigma=0.3)
    assert r["stat"].shape == (lib.n_features,) and np.isfinite(r["ret"])


def test_reward_training_learns_the_informative_axis_and_shuffled_reward_does_not():
    lib = synthetic_library()
    spec, params = EpisodeSpec(duration_s=10.0), AgentParams()
    base = evaluate(lib, np.zeros(lib.n_features), 0.0, spec, params, 80, 5)["mean_return"]
    kw = {"iters": 60, "batch": 16, "lr": 0.3, "sigma": 0.5, "seed": 3, "l2": 1e-2}
    good = train(lib, spec, params, **kw)
    assert evaluate(lib, good["w"], good["b"], spec, params, 80, 5)["mean_return"] > base + 0.15
    assert good["w"][0] > 0 and int(np.argmax(np.abs(good["w"]))) == 0
    bad = train(lib, spec, params, shuffle_reward=True, **kw)
    assert evaluate(lib, bad["w"], bad["b"], spec, params, 80, 5)["mean_return"] < evaluate(
        lib, good["w"], good["b"], spec, params, 80, 5)["mean_return"]


def test_hunger_multiplier_scales_the_update():
    lib = synthetic_library()
    spec, params = EpisodeSpec(duration_s=6.0), AgentParams()
    kw = {"iters": 1, "batch": 8, "lr": 1.0, "sigma": 0.5, "seed": 4, "momentum": 0.0, "l2": 0.0}
    w1 = train(lib, spec, params, reward_multiplier=1.0, **kw)["w"]
    w2 = train(lib, spec, params, reward_multiplier=2.0, **kw)["w"]
    assert np.allclose(w2, 2.0 * w1, rtol=1e-6, atol=1e-9)


def test_pca_reduction_roundtrips_to_full_weights():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 12)) @ rng.normal(size=(12, 12))
    level = np.repeat([0.1, 1.0], 100)
    ratio = np.tile([0.5, 2.0], 100)
    block = np.repeat(np.arange(10), 20)
    lib = FeatureLibrary(x, level, ratio, block, n_components=4)
    assert lib.n_features == 4 and lib.basis.shape == (12, 4)
    w = rng.normal(size=4)
    z_std = (x - lib.mu) / lib.sd
    assert np.allclose(z_std @ lib.full_weights(w), (z_std @ lib.basis) @ w, atol=1e-8)
    assert np.allclose((z_std @ lib.basis).std(axis=0), 1.0, atol=1e-6)
