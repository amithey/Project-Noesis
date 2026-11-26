"""Spike-only gait measurements; no simulation state is changed.

The spectral null redistributes spike times uniformly over the observation window,
conditioning on the pool's spike count (equivalently each neuron's count). Each
surrogate contributes its MAXIMUM non-DC power, correcting the frequency search.
This tests stationary Poisson timing, not specifically walking: transients and
nonstationary drive can also reject it. Phase/alternation must be checked separately.
"""

from __future__ import annotations

import numpy as np

from noesis.motor import MUSCLE_JOINT

LEGS = ("LF", "LM", "LH", "RF", "RM", "RH")
TRIPOD = {"LF": 0.0, "RM": 0.0, "LH": 0.0, "RF": np.pi, "LM": np.pi, "RH": np.pi}


def _inputs(times_ms, idx, duration_ms, bin_ms):
    times, ids = np.asarray(times_ms, float), np.asarray(idx)
    if times.ndim != 1 or ids.shape != times.shape:
        raise ValueError("times_ms and idx must be matching one-dimensional arrays")
    if not np.isfinite(duration_ms) or not np.isfinite(bin_ms):
        raise ValueError("duration_ms and bin_ms must be finite")
    if bin_ms <= 0 or duration_ms < 4 * bin_ms:
        raise ValueError("need at least four positive-width bins")
    n = round(duration_ms / bin_ms)
    if not np.isclose(n * bin_ms, duration_ms):
        raise ValueError("duration_ms must be an integer multiple of bin_ms")
    if not np.all(np.isfinite(times)) or np.any((times < 0) | (times >= duration_ms)):
        raise ValueError("spike times must be finite and in [0, duration_ms)")
    return times, ids, n


def _counts(times, ids, group, n, bin_ms):
    members = np.unique(np.asarray(group, dtype=int))
    selected = times[np.isin(ids, members)]
    return np.bincount((selected / bin_ms).astype(int), minlength=n), len(members)


def _spectrum(counts, size, bin_ms):
    rate = counts / (max(size, 1) * bin_ms * 1e-3)
    fourier = np.fft.rfft(rate - rate.mean())
    power = 2 * np.abs(fourier) ** 2 / len(rate) ** 2
    if len(rate) % 2 == 0:
        power[-1] /= 2  # Nyquist has no negative-frequency partner.
    return fourier, power


def rhythm_spectrum(times_ms, idx, group, duration_ms, bin_ms=10.0, *,
                    n_shuffles=199, seed=0):
    """Return peak Hz, power (Hz²), null threshold, Monte Carlo p and rhythmic flag.

    All resolvable positive frequencies through Nyquist are searched. The flag
    requires BOTH power > the 95th percentile of surrogate maxima and p <= .05.
    Empty/silent/constant pools have frequency None and are never rhythmic.
    Local RNG makes this pure and repeatable; >=100 shuffles are mandatory.
    """
    times, ids, n = _inputs(times_ms, idx, duration_ms, bin_ms)
    if not isinstance(n_shuffles, (int, np.integer)) or n_shuffles < 100:
        raise ValueError("n_shuffles must be an integer >= 100")
    counts, size = _counts(times, ids, group, n, bin_ms)
    _, power = _spectrum(counts, size, bin_ms)
    peak = int(np.argmax(power[1:])) + 1
    observed = float(power[peak])
    rng = np.random.default_rng(seed)
    # Multinomial counts are exactly the binned uniform-time shuffled-spike null.
    shuffled = rng.multinomial(int(counts.sum()), np.full(n, 1 / n), size=n_shuffles)
    rates = shuffled / (max(size, 1) * bin_ms * 1e-3)
    fft = np.fft.rfft(rates - rates.mean(axis=1, keepdims=True), axis=1)
    null_power = 2 * np.abs(fft) ** 2 / n ** 2
    if n % 2 == 0:
        null_power[:, -1] /= 2
    maxima = null_power[:, 1:].max(axis=1)
    threshold = float(np.percentile(maxima, 95))
    tolerance = 64 * np.finfo(float).eps * max(observed, float(maxima.max()))
    p = float((1 + np.count_nonzero(maxima >= observed - tolerance)) / (n_shuffles + 1))
    return {"frequency_hz": float(peak * 1000 / duration_ms) if observed > 0 else None,
            "power_hz2": observed, "null95_hz2": threshold, "p_value": p,
            "rhythmic": bool(observed > threshold + tolerance and p <= .05),
            "spike_count": int(counts.sum()), "pool_size": size}


def antagonist_alternation(times_ms, idx, table, duration_ms, bin_ms=10.0):
    """Pearson correlation of mean pool rates per leg/joint from leg_motor_table.

    Positive means coincident rate fluctuations; negative means alternation.
    Missing or constant pools return None, not an invented zero correlation.
    Unsmoothed nonoverlapping bins keep the time scale explicit. Positive
    correlation alone does not prove simultaneous muscle force.
    """
    times, ids, n = _inputs(times_ms, idx, duration_ms, bin_ms)
    pools = {}
    for row in table.itertuples():
        for joint, sign in MUSCLE_JOINT.get(row.muscle, ()):
            pools.setdefault((row.leg, joint), {1: [], -1: []})[sign].append(row.idx)
    out = {}
    for (leg, joint), pair in sorted(pools.items()):
        a, na = _counts(times, ids, pair[1], n, bin_ms)
        b, nb = _counts(times, ids, pair[-1], n, bin_ms)
        valid = na and nb and a.std() > 0 and b.std() > 0
        out[f"{leg}_{joint}"] = {
            "correlation": float(np.corrcoef(a, b)[0, 1]) if valid else None,
            "agonist_size": na, "antagonist_size": nb,
            "agonist_hz": float(a.sum() / (max(na, 1) * duration_ms * 1e-3)),
            "antagonist_hz": float(b.sum() / (max(nb, 1) * duration_ms * 1e-3))}
    return out


def interleg_phase(times_ms, idx, groups, duration_ms, bin_ms=10.0, *,
                   n_shuffles=199, seed=0):
    """Dominant leg phases and common-frequency tripod score, invariant to time origin.

    groups maps all six leg names to matching muscle pools (e.g. tibia flexors).
    Score = abs(mean(exp(i*(phase - expected_tripod_phase)))), in [0,1].
    Common frequency maximizes sum of per-leg normalized spectra; phases are
    compared THERE, never at different frequencies. Score is None unless all six
    legs reject their spectral null and peak within one FFT bin of this frequency.
    This validity gate is not a statistical test of the tripod score itself.
    """
    if set(groups) != set(LEGS):
        raise ValueError("groups must contain exactly LF, LM, LH, RF, RM, RH")
    times, ids, n = _inputs(times_ms, idx, duration_ms, bin_ms)
    spectra, fouriers, legs = [], [], {}
    for leg in LEGS:
        counts, size = _counts(times, ids, groups[leg], n, bin_ms)
        fft, power = _spectrum(counts, size, bin_ms)
        fouriers.append(fft)
        spectra.append(power / power[1:].sum() if power[1:].sum() else power)
        result = rhythm_spectrum(times, ids, groups[leg], duration_ms, bin_ms,
                                 n_shuffles=n_shuffles, seed=seed)
        k = int(np.argmax(power[1:])) + 1
        result["dominant_phase_rad"] = float(np.angle(fft[k])) if power[k] > 0 else None
        legs[leg] = result
    combined = np.sum(spectra, axis=0)
    k = int(np.argmax(combined[1:])) + 1
    frequency = k * 1000 / duration_ms
    valid = combined[k] > 0
    errors = []
    for leg, fft in zip(LEGS, fouriers):
        result = legs[leg]
        result["common_phase_rad"] = float(np.angle(fft[k])) if abs(fft[k]) > 0 else None
        valid = valid and result["rhythmic"] and abs(
            (result["frequency_hz"] or 0) - frequency) <= 1000 / duration_ms + 1e-9
        valid = valid and abs(fft[k]) > 0
        errors.append(np.angle(fft[k]) - TRIPOD[leg])
    score = float(abs(np.mean(np.exp(1j * np.asarray(errors))))) if valid else None
    return {"common_frequency_hz": float(frequency) if combined[k] > 0 else None,
            "tripod_score": score, "phase_valid": bool(valid), "legs": legs}
