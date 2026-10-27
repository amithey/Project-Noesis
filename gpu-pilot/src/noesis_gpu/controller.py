"""Frozen CPU connectome -> muscle commands, with batched GPU body feedback.

Neural execution is intentionally still CPU. This is a transport/compatibility
pilot, not a new learning rule, validated curriculum, or GPU neural implementation.
"""

from __future__ import annotations

import hashlib
import inspect

import numpy as np

from .bridge import LEGACY_JOINTS, LEGS, Muscles, adhesion_from_command


def import_checkpoint(net, path):
    """Delegate compatibility, topology, sign and bounds checks to the existing loader."""
    from noesis.plasticity import MotorPlasticity

    with np.load(path, allow_pickle=False) as cp:
        indices = cp["motor_indices"].copy()
        parameters = cp["parameters"].copy()
        rule = str(cp["rule"]) if "rule" in cp else "hebbian"
        normalized = bool(cp["normalize_reward_error"]) if "normalize_reward_error" in cp else False
        corrected = (bool(cp["bias_correct_reward_error"])
                     if "bias_correct_reward_error" in cp else False)
    if parameters.shape != (3,) or parameters[2] != 0.5:
        raise ValueError("Pilot requires original-reference checkpoint bounds of 50%")
    options = {"learning_rate": parameters[0], "tau_eligibility_ms": parameters[1], "max_change": 0.5}
    if "rule" in inspect.signature(MotorPlasticity).parameters:
        options.update(rule=rule, normalize_reward_error=normalized,
                       bias_correct_reward_error=corrected)
    elif rule != "hebbian" or normalized or corrected:
        raise ValueError("Selected Noesis source cannot read this checkpoint version")
    loader = MotorPlasticity(net, indices, **options)
    if "weights_only" in inspect.signature(loader.load).parameters:
        loader.load(path, weights_only=True)
    else:
        loader.load(path)
    # The loader and all learning traces are discarded. No update() is called.


class FrozenBrain:
    def __init__(self, connectome, rest, lo, hi, seed=0, checkpoint=None,
                 bg_hz=3.0, bg_weight=40.0, sensory_weight=400.0, shared_weights=None):
        from noesis.balance import BalanceMap
        from noesis.body import BodyParams
        from noesis.lif import LIFNetwork, LIFParams
        from noesis.motor import RateTracker, build_muscle_matrices
        from noesis.sensory import SensoryMap

        self.net = LIFNetwork(connectome.W, LIFParams(dt=0.5, da=1.0), seed=seed)
        if shared_weights is not None:
            self.net.indptr = shared_weights.indptr
            self.net.indices = shared_weights.indices
            self.net.data = shared_weights.data
        elif checkpoint is not None:
            import_checkpoint(self.net, checkpoint)
        for array in (self.net.indptr, self.net.indices, self.net.data):
            array.flags.writeable = False
        self.initial_weights_sha256 = hashlib.sha256(memoryview(self.net.data)).hexdigest()
        self.tracker = RateTracker(connectome.n)
        self.senses = SensoryMap(connectome.neurons)
        self.balance = BalanceMap(connectome.neurons, seed=0)
        pos, neg = build_muscle_matrices(connectome.neurons, list(LEGACY_JOINTS))
        self.muscles = Muscles(pos, neg, rest, lo, hi, BodyParams())
        self.rest, self.command = rest.copy(), rest.copy()
        self.rng = np.random.default_rng(seed)
        self.bg_hz, self.bg_weight, self.sensory_weight = bg_hz, bg_weight, sensory_weight
        self.total_spikes = 0
        self.last_spikes = np.empty(0, dtype=np.int64)

    def _current_drive(self, dt_ms):
        return {}

    def step(self, observation, dt_ms):
        from noesis.environment_senses import merge_drive
        from noesis.lif import _seed
        from noesis.sensory import LegState

        angles = (observation["angles"] - self.rest).reshape(6, 7)
        velocity = observation["velocity"].reshape(6, 7)
        states = {leg: LegState(angles[i, 5], velocity[i, 5], angles[i, 0],
                                angles[i, 3], observation["load"][i])
                  for i, leg in enumerate(LEGS)}
        drive = merge_drive([
            (self.senses.idx, self.senses.drive(states), self.sensory_weight),
            (self.balance.idx, self.balance.drive(observation["up_body"], observation["omega"]),
             self.sensory_weight),
        ])
        # Existing LIF uses a thread-global Numba RNG. A per-world seed stream on
        # each chunk makes world 0 independent of how many other worlds run.
        # This is explicitly a new RNG schedule, not bitwise legacy replay.
        _seed(int(self.rng.integers(0, 2**31)))
        _, spikes = self.net.run(dt_ms, background_hz=self.bg_hz,
                                 background_weight=self.bg_weight, **drive,
                                 **self._current_drive(dt_ms))
        self.last_spikes = spikes
        self.total_spikes += len(spikes)
        rates = self.tracker.update(spikes, dt_ms)
        desired = self.muscles.step(rates, dt_ms)
        self.command += np.clip(desired - self.command, -0.06, 0.06)
        return self.command.copy(), adhesion_from_command(desired, self.rest)

    def array_bytes(self):
        return sum(value.nbytes for value in vars(self.net).values()
                   if isinstance(value, np.ndarray)) + self.tracker.rate.nbytes

    def weights_unchanged(self):
        return hashlib.sha256(memoryview(self.net.data)).hexdigest() == self.initial_weights_sha256


class FrozenPopulation:
    def __init__(self, connectome, body, seed=0, memory_budget_gib=8.0, **options):
        first = FrozenBrain(connectome, body.rest, body.lo, body.hi, seed=seed, **options)
        base = sum(array.nbytes for array in (connectome.W.data, connectome.W.indices,
                                              connectome.W.indptr))
        shared_bytes = sum(array.nbytes for array in (first.net.data, first.net.indices,
                                                      first.net.indptr))
        state_bytes = first.array_bytes() - shared_bytes
        # Include the original graph, shared frozen graph, per-world state and a
        # transient graph copy while a subsequent LIFNetwork is constructed.
        self.estimated_host_bytes = int(1.3 * (base + 2 * shared_bytes
                                               + body.n_worlds * state_bytes))
        if self.estimated_host_bytes > memory_budget_gib * 1024**3:
            raise MemoryError(f"Estimated neural arrays plus headroom: "
                              f"{self.estimated_host_bytes / 1024**3:.2f} GiB; "
                              f"budget {memory_budget_gib:.2f} GiB. Reduce --worlds or "
                              "set --host-memory-budget-gib for a measured suitable server.")
        self.brains = [first] + [
            FrozenBrain(connectome, body.rest, body.lo, body.hi, seed=seed + i,
                        shared_weights=first.net, **options)
            for i in range(1, body.n_worlds)
        ]

    def step(self, observation, dt_ms):
        commands, pads = [], []
        for i, brain in enumerate(self.brains):
            command, adhesion = brain.step({key: value[i] for key, value in observation.items()},
                                           dt_ms)
            commands.append(command)
            pads.append(adhesion)
        return np.stack(commands), np.stack(pads)
