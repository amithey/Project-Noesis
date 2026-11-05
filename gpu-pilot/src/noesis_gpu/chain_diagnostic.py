"""Fixed-weight, learning-off transport diagnostic: exploration current -> neural firing -> muscle activation -> commands -> body.

Three paired arms share the checkpoint, the LIF noise stream and the (separately seeded) exploration stream:
``off`` injects nothing and never steps the exploration object, ``zero`` steps it (same random draws) but injects an all-zero
current, ``active`` injects the real current. Learning never runs and weights are checked unchanged. ``off`` and ``zero`` must
agree bit for bit at every recorded stage; that is the registration/zero-intervention invariance check and it is run BEFORE any
causal reading. No gait generator, leg order or external command exists here: the recorded commands are the controller's own.

The coupling statistic is descriptive: for each stage, the best pooled R^2 between the joint predictor (mean current of the
explored agonist neurons minus antagonist neurons, as in L1b) and that stage's joint response over lags 0..max_lag, compared
with a circular-shift reference. Reference percentiles are descriptive thresholds, not a detection power or an effect-size
bound. The ``off`` and ``zero`` arms are scored against the active arm's current, which they never received: that is the
matched negative control for the whole statistic.
"""

from __future__ import annotations

import numpy as np

from .learning import DEFAULT_LEARNING_RATE, LearningBrain

ARMS = ("off", "zero", "active")
STAGES = ("firing", "muscle", "command", "body")
RECORDED = ("drive", "circuit_spikes", "total_spikes", "a_pos", "a_neg", "command", "adhesion",
            "obs_angles", "obs_velocity", "obs_position", "obs_up_body", "obs_load")


class ChainBrain(LearningBrain):
    """Frozen-weight brain with an exploration arm; ``learn`` is forbidden and weights are never updated."""

    def __init__(self, connectome, rest, lo, hi, *, checkpoint, arm, seed=0, exploration_mv_ms=1.5, **options):
        if arm not in ARMS:
            raise ValueError(f"arm must be one of {ARMS}")
        options.pop("record_exploration", None)
        super().__init__(connectome, rest, lo, hi, checkpoint=checkpoint, seed=seed, exploration_mv_ms=exploration_mv_ms,
                         learning_rate=DEFAULT_LEARNING_RATE, record_exploration=False, **options)
        self.arm = arm
        # Same order as the exploration object injects: motor-pool targets first, then the circuit targets.
        self.circuit_idx = np.concatenate((self.exploration.idx, self.exploration.circuit_idx)).astype(np.int64)
        if len(np.unique(self.circuit_idx)) != len(self.circuit_idx):
            raise ValueError("Exploration targets overlap; per-neuron recording would be ambiguous")
        self._slot = np.full(connectome.n, -1, dtype=np.int64)
        self._slot[self.circuit_idx] = np.arange(len(self.circuit_idx))
        self._injected = np.zeros(len(self.circuit_idx), dtype=np.float32)
        self._rows = {key: [] for key in RECORDED}

    def learn(self, reward, dt_ms):
        raise RuntimeError("learning is off in the chain diagnostic")

    def _current_drive(self, dt_ms):
        self._injected = np.zeros(len(self.circuit_idx), dtype=np.float32)
        if self.arm == "off":
            return {}
        drive = self.exploration.step(dt_ms)  # same draws in the zero and active arms
        if not drive:
            return {}
        if len(drive["current_idx"]) != len(self.circuit_idx) or not np.array_equal(drive["current_idx"], self.circuit_idx):
            raise RuntimeError("Injected targets differ from the recorded targets")
        current = np.asarray(drive["current_mv_per_ms"], dtype=float)
        if self.arm == "zero":
            current = np.zeros_like(current)
        else:
            self.max_observed_exploration = max(self.max_observed_exploration, float(np.abs(current).max(initial=0)))
        self._injected = current.astype(np.float32)
        return {"current_idx": drive["current_idx"], "current_mv_per_ms": current}

    def step(self, observation, dt_ms):
        command, adhesion = super().step(observation, dt_ms)
        rows = self._rows
        spikes = np.asarray(self.last_spikes, dtype=np.int64)
        slots = self._slot[spikes] if len(spikes) else spikes
        rows["drive"].append(self._injected.copy())
        rows["circuit_spikes"].append(np.bincount(slots[slots >= 0], minlength=len(self.circuit_idx)).astype(np.uint16))
        rows["total_spikes"].append(len(spikes))
        rows["a_pos"].append(self.muscles.a_pos.copy())
        rows["a_neg"].append(self.muscles.a_neg.copy())
        rows["command"].append(np.asarray(command, dtype=float).copy())
        rows["adhesion"].append(np.asarray(adhesion, dtype=float).copy())
        for key, name in (("obs_angles", "angles"), ("obs_velocity", "velocity"), ("obs_position", "position"),
                          ("obs_up_body", "up_body"), ("obs_load", "load")):
            rows[key].append(np.asarray(observation[name], dtype=float).copy())
        return command, adhesion

    def arrays(self):
        out = {key: np.asarray(value) for key, value in self._rows.items()}
        out["drive_idx"], out["rest"] = self.circuit_idx.copy(), np.asarray(self.rest).copy()
        return out


def run_arm(brain, body, n_chunks, physics_steps, control_ms):
    """Run one world for ``n_chunks`` control steps; ``body`` needs ``observe()`` and ``advance(commands, adhesion, steps)``."""
    observation = body.observe()
    for _ in range(n_chunks):
        single = {key: value[0] for key, value in observation.items()}
        command, adhesion = brain.step(single, control_ms)
        body.advance(command[None], adhesion[None], physics_steps)
        observation = body.observe()
    if not brain.weights_unchanged():
        raise RuntimeError("Frozen weights changed during the chain diagnostic")
    return brain.arrays()


def invariance_report(off, zero):
    """Exact comparison of the off and zero arms; any difference means recording/zero-intervention altered the base run."""
    report = {}
    for key in RECORDED:
        a, b = np.asarray(off[key]), np.asarray(zero[key])
        if key == "drive":  # both must be exactly zero; shapes can differ only if the circuit differs
            report[key] = bool(a.shape == b.shape and not a.any() and not b.any())
        else:
            report[key] = bool(a.shape == b.shape and np.array_equal(a, b))
    report["identical"] = all(report.values())
    return report


def first_divergence(a, b):
    """First control step at which each recorded array of ``a`` and ``b`` differs (None when identical)."""
    out = {}
    for key in RECORDED:
        x, y = np.asarray(a[key]), np.asarray(b[key])
        n = min(len(x), len(y))
        diff = (x[:n] != y[:n]).reshape(n, -1).any(axis=1) if n else np.zeros(0, bool)
        out[key] = int(diff.argmax()) if diff.any() else (None if len(x) == len(y) else n)
    return out


def joint_predictor(drive, pos_mask, neg_mask):
    """(steps, joints): mean current of the explored agonist neurons minus antagonist neurons of each joint.

    Joints without both pools are NaN columns (skipped by ``coupling``).
    """
    pos, neg = np.asarray(pos_mask, bool), np.asarray(neg_mask, bool)
    ok = pos.any(axis=1) & neg.any(axis=1)
    weights = np.zeros(pos.shape, dtype=float)
    weights[ok] = pos[ok] / pos[ok].sum(axis=1, keepdims=True) - neg[ok] / neg[ok].sum(axis=1, keepdims=True)
    out = np.asarray(drive, dtype=float) @ weights.T
    out[:, ~ok] = np.nan
    return out


def stage_responses(arrays, pos_mask, neg_mask):
    """Per-joint (steps, joints) response of each stage; firing uses the same agonist-antagonist pools as the predictor."""
    rest = arrays["rest"]
    return {
        "firing": joint_predictor(arrays["circuit_spikes"].astype(float), pos_mask, neg_mask),
        "muscle": arrays["a_pos"] - arrays["a_neg"],
        "command": arrays["command"] - rest,
        "body": arrays["obs_velocity"],
    }


def _zscore(x):
    x = x - x.mean(axis=0)
    scale = np.sqrt((x ** 2).mean(axis=0))
    out = np.zeros_like(x)
    np.divide(x, scale, out=out, where=scale > 0)
    return out


def coupling(predictor, response, max_lag=40, permutations=1000, min_shift=200, seed=0):
    """Best pooled R^2 over lags 0..max_lag (response lagging the predictor) with a circular-shift reference.

    Pooled over joints with one shared coefficient (columns are z-scored first). Lags and shifts are circular, so
    the circular cross-correlation of each joint is computed once and a reference draw only rolls it by a random shift per joint.
    Returns observed/best lag/rank (share of draws with max R^2 >= observed) and the reference p95/p99, all descriptive.
    """
    x, y = np.asarray(predictor, float), np.asarray(response, float)
    if x.shape != y.shape or x.ndim != 2:
        raise ValueError("predictor and response must be (steps, joints) of equal shape")
    keep = np.isfinite(x).all(axis=0) & np.isfinite(y).all(axis=0) & (x.std(axis=0) > 0) & (y.std(axis=0) > 0)
    steps = len(x)
    if keep.sum() == 0 or steps <= 2 * min_shift:
        return {"joints_used": int(keep.sum()), "observed_r2": None, "status": "insufficient"}
    x, y = _zscore(x[:, keep]), _zscore(y[:, keep])
    cross = np.fft.irfft(np.conj(np.fft.rfft(x, axis=0)) * np.fft.rfft(y, axis=0), n=steps, axis=0)  # cross[l] = sum_t x[t] y[t+l]
    norm = float(keep.sum()) * steps  # both z-scored: sum x^2 = sum y^2 = steps per joint
    lags = np.arange(max_lag + 1)
    observed = (cross[lags].sum(axis=1) / norm) ** 2
    rng = np.random.default_rng(seed)
    joints = cross.shape[1]
    null = np.empty(permutations)
    for p in range(permutations):
        shifts = rng.integers(min_shift, steps - min_shift + 1, size=joints)
        idx = (lags[:, None] - shifts[None, :]) % steps
        null[p] = ((cross[idx, np.arange(joints)[None, :]].sum(axis=1) / norm) ** 2).max()
    best = int(observed.argmax())
    return {"joints_used": int(keep.sum()), "observed_r2": float(observed[best]), "best_lag_steps": best,
            "reference_rank": float((1 + np.sum(null >= observed[best])) / (permutations + 1)),
            "reference_p95": float(np.percentile(null, 95)), "reference_p99": float(np.percentile(null, 99)),
            "status": "ok", "sign": float(np.sign(cross[best].sum()))}


def chain_report(arms, pos_mask, neg_mask, **options):
    """Invariance first; then per-arm per-stage coupling against the ACTIVE arm's current (negative control for off and zero)."""
    inv = invariance_report(arms["off"], arms["zero"])
    predictor = joint_predictor(arms["active"]["drive"], pos_mask, neg_mask)
    out = {"invariance_off_vs_zero": inv,
           "first_divergence_active_vs_off": first_divergence(arms["active"], arms["off"]), "coupling": {}}
    for arm in ARMS:
        responses = stage_responses(arms[arm], pos_mask, neg_mask)
        out["coupling"][arm] = {stage: coupling(predictor, responses[stage], **options) for stage in STAGES}
    return out
