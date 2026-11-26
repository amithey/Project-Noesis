"""Synthetic timing controls, independent of connectome data or LIF dynamics."""
import numpy as np
import pandas as pd
import pytest

from noesis.gait import LEGS, TRIPOD, antagonist_alternation, interleg_phase, rhythm_spectrum


def synthetic(*, tonic=False, cocontract=False, wrong_phase=False, seed=42):
    rng = np.random.default_rng(seed)
    times, indices, rows = [], [], []
    # 10 s, 5 ms generation bins, 12 neurons/pool; independent Poisson sampling.
    t = np.arange(2.5, 10000, 5.0)
    for leg in LEGS:
        phase = 0 if wrong_phase else TRIPOD[leg]
        for sign, muscle in ((1, "tibia_flexor"), (-1, "tibia_extensor")):
            wave = np.cos(2 * np.pi * 10 * t / 1000 + phase)
            rate = np.full(t.shape, 50.0) if tonic else 50 * (
                1 + .95 * wave * (1 if cocontract else sign))
            for _ in range(12):
                neuron = len(rows)
                rows.append({"idx": neuron, "leg": leg, "muscle": muscle})
                counts = rng.poisson(rate * .005)
                spikes = np.repeat(t - 2.5, counts) + rng.uniform(0, 5, counts.sum())
                times.extend(spikes)
                indices.extend([neuron] * len(spikes))
    table = pd.DataFrame(rows)
    groups = {leg: table.loc[(table.leg == leg) & (table.muscle == "tibia_flexor"),
                             "idx"].to_numpy() for leg in LEGS}
    return np.asarray(times), np.asarray(indices), table, groups


def test_known_tripod():
    times, idx, table, groups = synthetic()
    rhythm = rhythm_spectrum(times, idx, groups["LF"], 10000)
    assert rhythm["frequency_hz"] == 10
    assert rhythm["rhythmic"] and rhythm["p_value"] <= .05
    assert rhythm["power_hz2"] > rhythm["null95_hz2"]
    phase = interleg_phase(times, idx, groups, 10000)
    assert phase["phase_valid"] and phase["tripod_score"] > .99
    assert phase["common_frequency_hz"] == 10
    assert all(r["frequency_hz"] == 10 for r in phase["legs"].values())
    for leg, result in phase["legs"].items():
        delta = result["dominant_phase_rad"] - phase["legs"]["LF"]["dominant_phase_rad"]
        assert abs(np.angle(np.exp(1j * (delta - TRIPOD[leg])))) < .1
    assert all(r["correlation"] < -.65
               for r in antagonist_alternation(times, idx, table, 10000).values())


def test_tonic_poisson():
    times, idx, table, groups = synthetic(tonic=True)
    assert not rhythm_spectrum(times, idx, groups["LF"], 10000)["rhythmic"]
    phase = interleg_phase(times, idx, groups, 10000)
    assert not phase["phase_valid"] and phase["tripod_score"] is None
    assert all(abs(r["correlation"]) < .1
               for r in antagonist_alternation(times, idx, table, 10000).values())


def test_cocontraction():
    times, idx, table, _ = synthetic(cocontract=True)
    assert all(r["correlation"] > .65
               for r in antagonist_alternation(times, idx, table, 10000).values())


def test_synchronous_legs_are_not_tripod():
    times, idx, _, groups = synthetic(wrong_phase=True)
    phase = interleg_phase(times, idx, groups, 10000)
    assert phase["phase_valid"] and phase["tripod_score"] < .05


def test_silent_missing_constant_and_validation():
    times, idx = np.arange(5, 1000, 10), np.zeros(100, dtype=int)
    assert rhythm_spectrum(times, idx, [0], 1000)["frequency_hz"] is None
    assert not rhythm_spectrum([], [], [], 1000)["rhythmic"]
    table = pd.DataFrame({"idx": [0], "leg": ["LF"], "muscle": ["tibia_flexor"]})
    assert antagonist_alternation(times, idx, table, 1000)["LF_Tibia"]["correlation"] is None
    phase = interleg_phase([], [], {leg: [] for leg in LEGS}, 1000)
    assert phase["tripod_score"] is None
    with pytest.raises(ValueError):
        rhythm_spectrum([], [], [], 1000, n_shuffles=99)
    with pytest.raises(ValueError):
        rhythm_spectrum([1000], [0], [0], 1000)
    with pytest.raises(ValueError):
        rhythm_spectrum([], [], [], 1001)
    with pytest.raises(ValueError):
        interleg_phase([], [], {}, 1000)


def test_null_false_positive_rate_and_repeatability():
    # Fixed ensemble guards against accepting every noisy periodogram maximum.
    detections = []
    rng = np.random.default_rng(17)
    for seed in range(40):
        times = rng.uniform(0, 3000, 3000)
        result = rhythm_spectrum(times, np.zeros(3000, int), [0], 3000, seed=seed)
        detections.append(result["rhythmic"])
    assert sum(detections) <= 6
    assert result == rhythm_spectrum(times, np.zeros(3000, int), [0], 3000, seed=39)


def test_sparse_spectrum_numerical_ties():
    # One spike has identical power at every positive non-Nyquist frequency.
    # FFT roundoff must not turn equivalent shuffled maxima into significance.
    for time in [0, 120, 590, 2990]:
        result = rhythm_spectrum([time], [0], [0], 3000)
        assert result["p_value"] == 1.0
        assert not result["rhythmic"]


def test_phase_requires_shared_frequency_and_is_time_origin_invariant():
    times, idx, _, groups = synthetic()
    original = interleg_phase(times, idx, groups, 10000)
    shifted = interleg_phase((times + 20) % 10000, idx, groups, 10000)
    assert shifted["tripod_score"] == pytest.approx(original["tripod_score"])
    different = times.copy()
    mask = np.isin(idx, groups["LH"])
    different[mask] = (different[mask] * 2) % 10000
    result = interleg_phase(different, idx, groups, 10000)
    assert result["legs"]["LH"]["frequency_hz"] == 5
    assert result["legs"]["LH"]["rhythmic"]
    assert not result["phase_valid"] and result["tripod_score"] is None
