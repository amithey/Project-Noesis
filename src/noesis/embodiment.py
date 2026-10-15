"""Explicit sensorimotor training scaffolding, separate from the connectome model."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from noesis.connectome import corrected_scale, load_cached
from noesis.motor import MUSCLE_JOINT, leg_motor_table


def load_training_brain(data: Path, cache_dir: Path, brain="corrected"):
    """Read shared data/caches; build absent caches only in a local run directory."""
    data, cache_dir = data.resolve(), cache_dir.resolve()
    if cache_dir == data or data in cache_dir.parents:
        raise ValueError("training cache must be outside source data")
    scale = None
    if brain == "corrected":
        manc = (data / "manc" / "connections_princeton.csv",
                data.parent / "MANC" / "connections_princeton.csv")
        if not any(p.is_file() for p in manc):
            raise FileNotFoundError("MANC is required for the corrected training brain")
        scale = corrected_scale(data)
    elif brain != "raw":
        raise ValueError("brain must be raw or corrected")
    suffix = ""
    if scale:
        suffix = "_scaled_" + hashlib.md5(
            json.dumps(sorted(scale.items())).encode()).hexdigest()[:8]
    shared = data / "processed"
    complete = (shared / "neurons.parquet").is_file() and (
        shared / f"W_min5{suffix}.npz").is_file()
    return load_cached(data / "raw", shared if complete else cache_dir,
                       region_scale=scale), scale


class MotorExploration:
    """Seeded, bounded, nonperiodic currents into annotated leg motor pools.

    An assumed training aid, not an anatomical CPG or a walking trajectory.
    Independent Ornstein-Uhlenbeck traces per muscle pool let training try
    actions instead of repeatedly observing an identical tonic motor state.
    Positive and negative currents excite/inhibit neurons through LIFNetwork;
    no joint angle is generated here. Frozen evaluation defaults to zero.
    """

    def __init__(self, neurons, amplitude_mv_ms=0.0, tau_ms=100.0, seed=0):
        if not np.isfinite(amplitude_mv_ms) or amplitude_mv_ms < 0:
            raise ValueError("exploration amplitude must be finite and nonnegative")
        if not np.isfinite(tau_ms) or tau_ms <= 0:
            raise ValueError("exploration time constant must be finite and positive")
        self.amplitude, self.tau = amplitude_mv_ms, tau_ms
        self.rng = np.random.default_rng(seed)
        self.innovation = np.empty(0)
        if amplitude_mv_ms == 0:
            self.idx = np.empty(0, dtype=np.int64)
            self.pool = np.empty(0, dtype=np.int64)
            self.state = np.empty(0)
            return
        table = leg_motor_table(neurons)
        table = table[table.muscle.isin(MUSCLE_JOINT)]
        groups = list(table.groupby(["leg", "muscle"]))
        self.idx = np.concatenate([g.idx.to_numpy() for _, g in groups]).astype(np.int64) \
            if groups else np.empty(0, dtype=np.int64)
        self.pool = np.concatenate([np.full(len(g), k) for k, (_, g) in enumerate(groups)]) \
            if groups else np.empty(0, dtype=np.int64)
        self.state = np.zeros(len(groups))
        self.innovation = np.zeros(len(self.idx))

    def step(self, dt_ms):
        if not np.isfinite(dt_ms) or dt_ms <= 0:
            raise ValueError("dt_ms must be finite and positive")
        if not len(self.idx):
            return {}
        decay = np.exp(-dt_ms / self.tau)
        noise = self.rng.normal(size=len(self.state))
        self.innovation = np.tanh(noise[self.pool])
        self.state = decay * self.state + np.sqrt(1 - decay**2) * noise
        return {"current_idx": self.idx,
                "current_mv_per_ms": self.amplitude * np.tanh(self.state[self.pool])}


    def credit(self, motor_indices):
        """Bounded innovations aligned to learner motors; zeros for unperturbed pools."""
        result = np.zeros(len(motor_indices))
        if len(self.idx):
            slots = np.searchsorted(motor_indices, self.idx)
            valid = slots < len(motor_indices)
            valid[valid] &= np.asarray(motor_indices)[slots[valid]] == self.idx[valid]
            result[slots[valid]] = self.innovation[valid]
        return result


class BrainReadout:
    """Observed firing channels, not labels for subjective emotion or thought."""

    def __init__(self, neurons, motor_indices):
        ctype = neurons.get("Primary Cell Type")
        self.groups = {"motor_hz": np.unique(motor_indices)}
        for name, prefix in (("pam_hz", "PAM"), ("ppl1_hz", "PPL1"),
                             ("mbon_hz", "MBON"), ("kc_hz", "KC")):
            self.groups[name] = np.flatnonzero(ctype.fillna("").str.startswith(prefix)) \
                if ctype is not None else np.empty(0, dtype=int)

    def rates(self, rate):
        result = {name: float(np.mean(rate[ids])) if len(ids) else 0.0
                  for name, ids in self.groups.items()}
        result["active_motor_neurons"] = int(np.count_nonzero(rate[self.groups["motor_hz"]] > 1))
        return result


def gait_report(times_ms, idx, neurons, duration_ms, seed=0):
    """Assess recorded leg spikes, truncating only an incomplete final 10 ms bin."""
    from noesis.gait import (
        LEGS,
        antagonist_alternation,
        interleg_phase,
        rhythm_spectrum,
    )

    duration = 10 * int(duration_ms // 10)
    if duration < 40:
        return {"status": "too_short", "analysis_duration_ms": duration}
    table = leg_motor_table(neurons)
    keep = np.asarray(times_ms) < duration
    times, ids = np.asarray(times_ms)[keep], np.asarray(idx)[keep]
    groups = {leg: table.loc[table.leg.eq(leg) & table.muscle.isin(
        ['tibia_flexor', 'accessory_tibia_flexor']), 'idx'].to_numpy() for leg in LEGS}
    return {"status": "measured", "analysis_duration_ms": duration,
            "pooled_leg_spectra": {leg: rhythm_spectrum(
                times, ids, table.loc[table.leg.eq(leg), 'idx'], duration, seed=seed)
                for leg in LEGS},
            "antagonists": antagonist_alternation(times, ids, table, duration),
            "tibia_flexor_phase": interleg_phase(times, ids, groups, duration, seed=seed)}
