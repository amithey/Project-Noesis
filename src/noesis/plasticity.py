"""Experimental reward-modulated motor-synapse learning, not biological validation.

Only existing edges entering caller-selected motor/readout neurons can change. The
connectome supplies the topology and initial signed weights; it does not supply a
learning rule. This module assumes frame-binned Hebbian coactivity, a decaying
eligibility trace, and a scalar reward broadcast. It does not implement dopamine,
STDP, or claim that the animal learns this way. Positive reward prediction error
strengthens the *magnitude* of coactive excitatory AND inhibitory connections.

The optional ``perturbation`` rule instead uses a normalized presynaptic rate
trace times the signed motor exploration innovation, with weight sign conversion.
It is an action-credit heuristic, not a proven policy gradient or a biological
model. The caller must provide the actual bounded exploration innovations.

Legacy activity is binary per update window, so results depend on the window size.
Reward is clipped to [-1, 1]. Learned magnitudes remain within ``max_change`` of
their initial values. This is an exploratory controller, not proof of learning to
walk or fly; that requires held-out evaluation against a frozen-weight baseline.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def _indices(values, n, name):
    result = np.asarray(values)
    if result.ndim != 1 or (result.size and result.dtype.kind not in "iu"):
        raise ValueError(f"{name} must be a one-dimensional integer array")
    if result.size and (np.any(result < 0) or np.any(result >= n)):
        raise ValueError(f"{name} contains an out-of-range neuron")
    return result.astype(np.int64, copy=False)


class MotorPlasticity:
    """Mutate ``net.data`` in place; callers disable learning by skipping update.

    ``learning_rate`` is fractional initial-weight change per second at maximal
    eligibility and reward error. Checkpoints restore learning state and selected
    weights, not membrane state, body state, simulation time, or random generators.
    With normalization enabled, the raw prediction error is divided by its
    two-second RMS estimate (floor 0.01) and clipped to [-1, 1]. The learning
    rate then applies to that unitless signal; the reward baseline stays in raw
    reward units. Network topology must remain fixed throughout.
    """

    def __init__(self, net, motor_indices, learning_rate=0.02,
                 tau_eligibility_ms=500.0, max_change=0.5, rule="hebbian",
                 normalize_reward_error=False):
        params = np.array([learning_rate, tau_eligibility_ms, max_change], dtype=float)
        if not np.isfinite(params).all():
            raise ValueError("learning parameters must be finite")
        if learning_rate < 0 or tau_eligibility_ms <= 0 or not 0 <= max_change < 1:
            raise ValueError("require rate >= 0, tau > 0, and 0 <= max_change < 1")
        if net.data.dtype.kind != "f" or not np.isfinite(net.data).all():
            raise ValueError("network weights must be finite floating-point values")
        if rule not in ("hebbian", "perturbation"):
            raise ValueError("unknown motor learning rule")
        if not isinstance(normalize_reward_error, (bool, np.bool_)):
            raise TypeError("normalize_reward_error must be boolean")
        if normalize_reward_error and rule != "perturbation":
            raise ValueError("reward normalization requires perturbation learning")
        self.normalize_reward_error = bool(normalize_reward_error)
        self.reward_variance = 0.0
        self.rule = rule
        self.net = net
        self.learning_rate, self.tau_eligibility_ms, self.max_change = params
        self.motor_indices = np.unique(_indices(motor_indices, net.n, "motor_indices"))
        self.edge_indices = np.flatnonzero(np.isin(net.indices, self.motor_indices))
        self.initial_weights = net.data[self.edge_indices].copy()
        # CSR rows are presynaptic neurons in LIFNetwork.
        pre = np.searchsorted(net.indptr, self.edge_indices, side="right") - 1
        post = net.indices[self.edge_indices]
        self.neurons = np.unique(np.concatenate((pre, post)))
        self._pre = np.searchsorted(self.neurons, pre)
        self._post = np.searchsorted(self.neurons, post)
        self._post_motor = np.searchsorted(self.motor_indices, post)
        self.activity = np.zeros(len(self.neurons))
        self.eligibility = np.zeros(len(self.edge_indices))
        self.reward_baseline = 0.0
        magnitude = np.abs(self.initial_weights)
        self._lower = magnitude * (1 - self.max_change)
        self._upper = magnitude * (1 + self.max_change)
        if not np.isfinite(self._upper).all():
            raise ValueError("weight bounds overflow")
        digest = hashlib.sha256()
        for array in (np.array([net.n]), net.indptr, net.indices):
            digest.update(np.asarray(array, dtype="<i8").tobytes())
        self._topology = digest.hexdigest()

    def update(self, spike_idx, dt_ms, reward, *, perturbation=None):
        """Apply a frame's spikes and reward; return JSON-compatible metrics.

        Per-step work is proportional to selected edges/neurons plus input spikes,
        never to all connectome edges. Traces have a 100 ms activity time constant
        and a 2000 ms reward-baseline time constant. Invalid inputs do not mutate.
        """
        if not np.isfinite(dt_ms) or dt_ms <= 0 or not np.isfinite(reward):
            raise ValueError("dt_ms must be finite and positive; reward must be finite")
        spikes = _indices(spike_idx, self.net.n, "spike_idx")
        if self.rule == "perturbation":
            perturbation = np.zeros(len(self.motor_indices)) if perturbation is None else np.asarray(perturbation, float)
            if (perturbation.shape != self.motor_indices.shape or not np.isfinite(perturbation).all()
                    or np.any(np.abs(perturbation) > 1)):
                raise ValueError("perturbation must match motor_indices and lie in [-1, 1]")
        elif perturbation is not None:
            raise ValueError("perturbation requires the perturbation learning rule")
        reward = float(np.clip(reward, -1, 1))
        raw_error = reward - self.reward_baseline
        error = raw_error
        if self.normalize_reward_error:
            self.reward_variance += -np.expm1(-dt_ms / 2000.0) * (
                raw_error * raw_error - self.reward_variance)
            error = float(np.clip(raw_error / max(.01, np.sqrt(self.reward_variance)), -1, 1))
        active = np.isin(self.neurons, spikes)
        alpha = -np.expm1(-dt_ms / 100.0)
        if self.rule == "hebbian":
            self.activity += alpha * (active - self.activity)
        else:
            # Presynaptic rate trace normalized at 50 Hz, rather than multiplying
            # two tiny 5 ms spike probabilities as in the legacy rule.
            rate = np.bincount(spikes, minlength=self.net.n)[self.neurons] * (1000 / dt_ms / 50)
            self.activity = np.clip((1 - alpha) * self.activity + alpha * rate, 0, 1)
        alpha_e = -np.expm1(-dt_ms / self.tau_eligibility_ms)
        coactivity = self.activity[self._pre] * self.activity[self._post]
        if self.rule == "perturbation":
            # A rewarded positive motor perturbation credits excitatory input
            # and reduces inhibitory magnitude; a negative perturbation reverses it.
            coactivity = (self.activity[self._pre] * perturbation[self._post_motor]
                          * np.sign(self.initial_weights))
        self.eligibility += alpha_e * (coactivity - self.eligibility)
        old = self.net.data[self.edge_indices].copy()
        # Clip the dimensionless increment before multiplication to avoid overflow
        # even when a caller supplies a very large finite duration or rate.
        with np.errstate(over="ignore", invalid="ignore"):
            step = self.learning_rate * (dt_ms / 1000.0)
        if error == 0 or self.learning_rate == 0:
            delta = np.zeros_like(self.eligibility)
        else:
            with np.errstate(over="ignore", invalid="ignore"):
                delta = step * error * self.eligibility
            delta = np.nan_to_num(delta, nan=0.0, posinf=2.0, neginf=-2.0)
            delta = np.clip(delta, -2 * self.max_change, 2 * self.max_change)
        initial = np.abs(self.initial_weights)
        relative = np.divide(np.abs(old), initial, out=np.ones_like(initial), where=initial > 0)
        relative = np.clip(relative + delta, 1 - self.max_change, 1 + self.max_change)
        new = self.initial_weights * relative
        self.net.data[self.edge_indices] = new
        self.reward_baseline += -np.expm1(-dt_ms / 2000.0) * raw_error
        return {
            "reward": reward, "reward_baseline": float(self.reward_baseline),
            "prediction_error": float(raw_error), "learning_signal": float(error),
            "changed_edges": int(np.count_nonzero(new != old)),
            **self.metrics(),
        }

    def metrics(self):
        """Report current deviation from original weights without advancing learning.

        Suitable for frozen evaluation after loading a checkpoint. Explicit zero
        edges remain zero and count as zero relative change.
        """
        initial = np.abs(self.initial_weights)
        difference = np.abs(self.net.data[self.edge_indices] - self.initial_weights)
        changes = np.divide(difference, initial, out=np.zeros_like(initial), where=initial > 0)
        return {
            "plastic_edges": len(self.edge_indices),
            "reward_variance": float(self.reward_variance),
            "reward_error_scale": float(max(.01, np.sqrt(self.reward_variance)))
            if self.normalize_reward_error else 1.0,
            "mean_relative_change": float(np.mean(changes)) if len(changes) else 0.0,
            "max_relative_change": float(np.max(changes)) if len(changes) else 0.0,
        }

    def save(self, path: str | Path):
        """Write a pickle-free NPZ checkpoint at the exact requested path."""
        with Path(path).open("wb") as handle:
            np.savez_compressed(
                handle, version=np.array(3 if self.normalize_reward_error else 1 if self.rule == "hebbian" else 2),
                normalize_reward_error=np.array(self.normalize_reward_error),
                reward_variance=np.array(self.reward_variance),
                rule=np.array(self.rule), topology=np.array(self._topology),
                motor_indices=self.motor_indices, initial_weights=self.initial_weights,
                parameters=np.array([self.learning_rate, self.tau_eligibility_ms, self.max_change]),
                weights=self.net.data[self.edge_indices], activity=self.activity,
                eligibility=self.eligibility, reward_baseline=np.array(self.reward_baseline),
            )

    def load(self, path: str | Path, restore_traces: bool = False, *, weights_only=False):
        """Validate before restoring; explicit weight transfer may cross learning rules.

        Topology, motor identities, original weights, signs and current magnitude
        bounds are checked even for weights-only imports. Normal loading also
        checks learning configuration and state. Fresh episodes reset all traces,
        the raw reward baseline and normalization statistics.
        """
        if weights_only and restore_traces:
            raise ValueError("weights-only import cannot restore traces")
        try:
            with np.load(path, allow_pickle=False) as checkpoint:
                version = checkpoint["version"]
                if version.shape != () or version.dtype.kind not in "iu" or int(version) not in (1, 2, 3):
                    raise ValueError("unsupported checkpoint version")
                expected = {"topology": np.array(self._topology),
                            "motor_indices": self.motor_indices,
                            "initial_weights": self.initial_weights}
                if not weights_only:
                    expected.update(version=np.array(3 if self.normalize_reward_error
                                    else 1 if self.rule == "hebbian" else 2),
                                    parameters=np.array([self.learning_rate,
                                        self.tau_eligibility_ms, self.max_change]))
                for key, value in expected.items():
                    if not np.array_equal(checkpoint[key], value):
                        raise ValueError(f"checkpoint mismatch: {key}")
                stored_rule = str(checkpoint["rule"]) if "rule" in checkpoint else "hebbian"
                normalized = bool(checkpoint["normalize_reward_error"]) if "normalize_reward_error" in checkpoint else False
                if not weights_only and (stored_rule != self.rule or normalized != self.normalize_reward_error):
                    raise ValueError("checkpoint mismatch: learning rule or normalization")
                weights = checkpoint["weights"].copy()
                if weights.shape != self.initial_weights.shape or not np.isfinite(weights).all():
                    raise ValueError("invalid checkpoint state")
                if (np.any(np.sign(weights) != np.sign(self.initial_weights))
                        or np.any(np.abs(weights) < self._lower)
                        or np.any(np.abs(weights) > self._upper)):
                    raise ValueError("checkpoint state exceeds learning bounds")
                activity, eligibility = np.zeros_like(self.activity), np.zeros_like(self.eligibility)
                baseline, variance = 0., 0.
                if not weights_only:
                    activity = checkpoint["activity"].copy()
                    eligibility = checkpoint["eligibility"].copy()
                    baseline = checkpoint["reward_baseline"].copy()
                    variance = checkpoint["reward_variance"].copy() if self.normalize_reward_error or "reward_variance" in checkpoint else np.array(0.)
                    for value, shape in ((activity, self.activity.shape),
                                         (eligibility, self.eligibility.shape), (baseline, ()), (variance, ())):
                        if value.shape != shape or not np.isfinite(value).all():
                            raise ValueError("invalid checkpoint state")
                    if (np.any((activity < 0) | (activity > 1))
                            or np.any((eligibility < (-1 if self.rule == "perturbation" else 0)) | (eligibility > 1))
                            or not -1 <= float(baseline) <= 1 or not 0 <= float(variance) <= 4):
                        raise ValueError("checkpoint state exceeds learning bounds")
        except (KeyError, TypeError, OverflowError) as error:
            raise ValueError("invalid plasticity checkpoint") from error
        self.net.data[self.edge_indices] = weights
        self.activity[:] = activity if restore_traces else 0
        self.eligibility[:] = eligibility if restore_traces else 0
        self.reward_baseline = float(baseline) if restore_traces else 0.0
        self.reward_variance = float(variance) if restore_traces else 0.0
