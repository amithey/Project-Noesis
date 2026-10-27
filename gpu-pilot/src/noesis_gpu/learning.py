"""Single-world experimental learning using the existing bounded Noesis rule."""

from __future__ import annotations

import hashlib

import numpy as np

from .controller import FrozenBrain


class LearningBrain(FrozenBrain):
    """Original-reference bounds are captured BEFORE importing any learned weights."""

    def __init__(self, connectome, rest, lo, hi, *, checkpoint, seed=0, **options):
        if "shared_weights" in options:
            raise ValueError("Learning brains require independent mutable weights")
        from noesis.circuit_learning import CircuitExploration
        from noesis.plasticity import MotorPlasticity

        super().__init__(connectome, rest, lo, hi, seed=seed, **options)
        self.net.data = self.net.data.copy()  # independent mutable weights, no population sharing
        with np.load(checkpoint, allow_pickle=False) as cp:
            targets = cp["motor_indices"].copy()
        self.learner = MotorPlasticity(
            self.net, targets, learning_rate=0.2, tau_eligibility_ms=100, max_change=0.5,
            rule="perturbation", normalize_reward_error=True, bias_correct_reward_error=True,
        )
        self.learner.load(checkpoint, weights_only=True)
        self.initial_weights_sha256 = hashlib.sha256(memoryview(self.net.data)).hexdigest()
        classes = connectome.neurons.iloc[self.learner.motor_indices]["Class"]
        circuit = self.learner.motor_indices[classes.isin(
            ("single_leg_neuromere", "sequential_leg_neuromeres")).to_numpy()]
        self.exploration = CircuitExploration(connectome.neurons, circuit,
                                              amplitude_mv_ms=1.5, tau_ms=100, seed=seed)
        if not np.isin(self.exploration.idx, self.learner.motor_indices).all():
            raise ValueError("Checkpoint does not cover all explored leg motor neurons")
        self.learning_active = False
        self.updates = 0
        self.max_observed_exploration = 0.0

    def _current_drive(self, dt_ms):
        if not self.learning_active:
            return {}
        drive = self.exploration.step(dt_ms)
        if drive:
            self.max_observed_exploration = max(self.max_observed_exploration,
                float(np.abs(drive["current_mv_per_ms"]).max(initial=0)))
        return drive

    def learn(self, reward, dt_ms):
        if not self.learning_active:
            return self.learner.metrics()
        result = self.learner.update(
            self.last_spikes, dt_ms, reward,
            perturbation=self.exploration.credit(self.learner.motor_indices),
        )
        self.updates += 1
        return result


def score_transition(before, after, dt_ms, objective="curriculum-standing-v7"):
    from noesis.embodied_metrics import locomotion_reward

    dt = dt_ms / 1000
    return locomotion_reward(
        before["position"], after["position"], float(np.clip(after["up_body"][2], -1, 1)),
        after["velocity"], dt, before["heading"], after["foot_load"],
        (after["feet_position"] - before["feet_position"]) / dt, objective=objective,
    )


def standing_mask(observation):
    """Physical posture/height and actual claw-to-floor support, independent of reward."""
    height = observation["position"][..., 2]
    return ((observation["up_body"][..., 2] >= .7) & (height >= .3) & (height <= 1.5)
            & (observation["foot_load"] > 1e-5).any(axis=-1))
