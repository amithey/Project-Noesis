"""Single-world experimental learning using the existing bounded Noesis rule."""

from __future__ import annotations

import hashlib

import numpy as np

from .controller import FrozenBrain
from .reward_profiles import resolve_objective

MAX_EXPLORATION_MV_MS = 1.5  # original exploration cap; options may only lower it
DEFAULT_LEARNING_RATE = 0.2   # per second; the pilot's value since the first births (core default is 0.02)
MAX_LEARNING_RATE = 2.0       # sanity cap for experiments; not a recommendation


class LearningBrain(FrozenBrain):
    """Original-reference bounds are captured BEFORE importing any learned weights.

    ``learning_rate`` only scales the size of every weight update; synthetic tests show it does not
    change the signal-to-noise of the credit. ``record_exploration`` keeps the per-step injected
    exploration current and the innovation credit for offline credit-timing analysis; it never
    changes learning.
    """

    def __init__(self, connectome, rest, lo, hi, *, checkpoint, seed=0,
                 exploration_mv_ms=MAX_EXPLORATION_MV_MS, learning_rate=DEFAULT_LEARNING_RATE,
                 record_exploration=False, **options):
        if "shared_weights" in options:
            raise ValueError("Learning brains require independent mutable weights")
        if not np.isfinite(exploration_mv_ms) or not 0 < exploration_mv_ms <= MAX_EXPLORATION_MV_MS:
            raise ValueError("exploration_mv_ms must be in (0, 1.5]; the original cap is not raised")
        if not np.isfinite(learning_rate) or not 0 < learning_rate <= MAX_LEARNING_RATE:
            raise ValueError(f"learning_rate must be finite and in (0, {MAX_LEARNING_RATE}]")
        from noesis.circuit_learning import CircuitExploration
        from noesis.plasticity import MotorPlasticity

        super().__init__(connectome, rest, lo, hi, seed=seed, **options)
        self.net.data = self.net.data.copy()  # independent mutable weights, no population sharing
        with np.load(checkpoint, allow_pickle=False) as cp:
            targets = cp["motor_indices"].copy()
        self.learner = MotorPlasticity(
            self.net, targets, learning_rate=float(learning_rate), tau_eligibility_ms=100, max_change=0.5,
            rule="perturbation", normalize_reward_error=True, bias_correct_reward_error=True,
        )
        self.learner.load(checkpoint, weights_only=True)
        self.initial_weights_sha256 = hashlib.sha256(memoryview(self.net.data)).hexdigest()
        classes = connectome.neurons.iloc[self.learner.motor_indices]["Class"]
        circuit = self.learner.motor_indices[classes.isin(
            ("single_leg_neuromere", "sequential_leg_neuromeres")).to_numpy()]
        self.exploration = CircuitExploration(connectome.neurons, circuit,
                                              amplitude_mv_ms=exploration_mv_ms, tau_ms=100, seed=seed)
        if not np.isin(self.exploration.idx, self.learner.motor_indices).all():
            raise ValueError("Checkpoint does not cover all explored leg motor neurons")
        self.exploration_mv_ms = float(exploration_mv_ms)
        self.learning_rate = float(learning_rate)
        self.learning_active = False
        self.updates = 0
        self.max_observed_exploration = 0.0
        self._log = {"drive": [], "credit": [], "drive_idx": None} if record_exploration else None

    def _current_drive(self, dt_ms):
        if not self.learning_active:
            return {}
        drive = self.exploration.step(dt_ms)
        if drive:
            self.max_observed_exploration = max(self.max_observed_exploration,
                float(np.abs(drive["current_mv_per_ms"]).max(initial=0)))
            if self._log is not None:
                self._log["drive"].append(np.asarray(drive["current_mv_per_ms"], np.float32).copy())
                self._log["drive_idx"] = np.asarray(drive["current_idx"]).copy()
        return drive

    def learn(self, reward, dt_ms):
        if not self.learning_active:
            return self.learner.metrics()
        credit = self.exploration.credit(self.learner.motor_indices)
        if self._log is not None:
            self._log["credit"].append(np.asarray(credit, np.float32).copy())
        result = self.learner.update(self.last_spikes, dt_ms, reward, perturbation=credit)
        self.updates += 1
        return result

    def exploration_log(self):
        """Arrays recorded when ``record_exploration`` is on, else None.

        ``drive`` (steps, drive targets) is the injected current in mV/ms for ``drive_idx`` neurons;
        ``credit`` (steps, motor targets) is the innovation credit given to ``motor_indices``. Row i of
        both belongs to the i-th learning-active control step.
        """
        if self._log is None:
            return None
        drive, credit = self._log["drive"], self._log["credit"]
        return {"drive": np.array(drive, np.float32), "credit": np.array(credit, np.float32),
                "drive_idx": self._log["drive_idx"], "motor_indices": self.learner.motor_indices.copy()}


def score_transition(before, after, dt_ms, objective="curriculum-standing-v7", cycle_credit=0.,
                     cycle_weight=None):
    from noesis.embodied_metrics import locomotion_reward

    dt = dt_ms / 1000
    extra = {"cycle_credit": cycle_credit} if cycle_credit else {}  # core rejects it for non-v9
    if cycle_weight is not None:  # None keeps the core default and works with a core that lacks the option
        extra["cycle_weight"] = cycle_weight
    return locomotion_reward(
        before["position"], after["position"], float(np.clip(after["up_body"][2], -1, 1)),
        after["velocity"], dt, before["heading"], after["foot_load"],
        (after["feet_position"] - before["feet_position"]) / dt,
        objective=resolve_objective(objective), **extra,
    )


_base_score_transition = score_transition  # CycleScoring must not call the (possibly patched) public name


def standing_mask(observation):
    """Physical posture/height and actual claw-to-floor support, independent of reward."""
    height = observation["position"][..., 2]
    return ((observation["up_body"][..., 2] >= .7) & (height >= .3) & (height <= 1.5)
            & (observation["foot_load"] > 1e-5).any(axis=-1))


class CycleScoring:
    """Stateful ``score_transition`` for the opt-in v9 objective: one instance per brain and episode.

    Owns a core ``LegCycleTracker`` (lift -> touchdown cycles per leg, same persistence debounce as the
    evaluation) fed once per control step, and passes its credit to the core reward. Call signature is
    ``score_transition``'s. The tracker starts empty, so no credit is paid for the first ``speed_window``
    of an episode; build a new instance (or call ``reset``) for each birth. Records how much of the
    bonus is credited and how often the reward is clipped, for the run report.
    """

    def __init__(self, objective="curriculum-gripped-cycle-v9", dt_ms=5., cycle_weight=None):
        try:
            from noesis.embodied_metrics import CYCLE_WEIGHT, LegCycleTracker
        except ImportError as error:
            raise ValueError("The v9 cycle objective needs a core with LegCycleTracker "
                             "(branch claude/flygym2-core-learning)") from error
        self.objective = resolve_objective(objective)
        if self.objective != "curriculum-gripped-cycle-v9":
            raise ValueError("CycleScoring only serves the v9 cycle objective")
        self.dt_ms = float(dt_ms)
        self._weight = CYCLE_WEIGHT if cycle_weight is None else float(cycle_weight)
        self._explicit_weight = cycle_weight
        if not np.isfinite(self._weight) or not 0 < self._weight <= 1:
            raise ValueError("cycle_weight must be finite and within (0, 1]")
        self._tracker = LegCycleTracker(dt_seconds=self.dt_ms / 1000)
        self.reset()

    def reset(self):
        self._tracker.reset()
        self.steps = self.credit_steps = self.unclipped_over_one = 0
        self.credit_sum = self.bonus_sum = self.actual_bonus_sum = 0.

    def __call__(self, before, after, dt_ms, objective=None):
        if dt_ms != self.dt_ms:
            raise ValueError(f"tracker was built for {self.dt_ms} ms steps, got {dt_ms} ms")
        heading = np.asarray(before["heading"], float)[:2]
        direction = heading / max(np.linalg.norm(heading), 1e-12)
        speed = float((np.asarray(after["position"], float)[:2]
                       - np.asarray(before["position"], float)[:2]) @ direction / (dt_ms / 1000))
        credit = self._tracker.update(after["foot_load"], speed, bool(standing_mask(after)))
        result = _base_score_transition(before, after, dt_ms, self.objective, cycle_credit=credit,
                                        cycle_weight=self._explicit_weight)
        self.steps += 1
        if credit > 0:
            self.credit_steps += 1
            self.credit_sum += credit
            self.bonus_sum += self._weight * credit * result["grip_factor"]  # before the [-1, 1] clip
            # what actually reached learning: the v9 reward minus the v8 reward of the same step
            self.actual_bonus_sum += result["reward"] - _base_score_transition(
                before, after, dt_ms, "curriculum-gripped-step-v8")["reward"]
        self.unclipped_over_one += result["reward_unclipped"] > 1
        return result

    def summary(self):
        steps = max(self.steps, 1)
        return {"cycle_weight": self._weight, "steps": self.steps, "credit_step_share": self.credit_steps / steps,
                "mean_credit_when_paid": self.credit_sum / max(self.credit_steps, 1),
                "intended_bonus_sum_before_clipping": self.bonus_sum,
                "actual_bonus_sum_after_clipping": self.actual_bonus_sum,
                "reward_unclipped_over_1_share": self.unclipped_over_one / steps}
