"""Population-based training of the connectome's existing plastic synapses (Claude track).

Why a new method: every prior per-step three-factor run (E2..EC1) produced tiny,
noisy weight changes whose effect did not survive frozen evaluation. Here the
brain is trained by *episode-level* black-box optimisation instead:

* The plastic edge set is exactly the existing one (edges entering the
  checkpoint's ``motor_indices``: leg motor neurons and annotated leg-neuromere
  circuit cells). Topology and signs never change; every magnitude stays within
  the original-reference +-50% bound used by the whole project.
* Edges are grouped by their postsynaptic population (muscle pool / circuit
  class and side) and presynaptic sign. Training searches one multiplicative
  gain per group, so the search space is O(100) instead of O(85k).
* A candidate is scored by running the frozen brain (no exploration current, no
  per-step plasticity) in the body, so the score *is* the frozen behaviour that
  is later evaluated; there is no train/test mismatch from exploration noise.
* Antithetic pairs share body/neural seeds (common random numbers).

Nothing here generates a gait trajectory, CPG or joint target: muscles are still
driven only by connectome motor-neuron rates through the existing Muscles model.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

BOUND = 0.5  # original-reference magnitude bound used project-wide
CONTROL_MS = 5.0
SUBSTEPS = 50  # 5 ms control / 0.1 ms physics
SETTLE_CHUNKS = 10  # 50 ms birth settling, excluded from scores


@dataclass
class EdgeGroups:
    edge_indices: np.ndarray      # positions in net.data
    original: np.ndarray          # signed original-reference weights
    start: np.ndarray             # signed starting weights (parent checkpoint)
    group: np.ndarray             # group id per edge
    names: list
    motor_indices: np.ndarray

    @property
    def n_groups(self):
        return len(self.names)

    def weights(self, gains):
        """Signed weights for a gain vector; magnitudes clipped to original +-50%."""
        gains = np.asarray(gains, float)
        if gains.shape != (self.n_groups,) or not np.isfinite(gains).all():
            raise ValueError("gain vector has wrong shape or is non-finite")
        magnitude = np.abs(self.start) * np.clip(gains, 0.0, None)[self.group]
        reference = np.abs(self.original)
        magnitude = np.clip(magnitude, reference * (1 - BOUND), reference * (1 + BOUND))
        return np.sign(self.original) * magnitude

    def describe(self):
        counts = np.bincount(self.group, minlength=self.n_groups)
        return [{"id": i, "name": n, "edges": int(c)} for i, (n, c) in enumerate(zip(self.names, counts))]


def _side(neurons):
    for column in ("Soma side", "side", "Side"):
        if column in neurons:
            return neurons[column].fillna("?").astype(str).str[0].str.upper()
    return np.full(len(neurons), "?")


def _neuromere(neurons):
    for column in ("Soma Neuromere", "somaNeuromere", "Neuromere", "neuromere"):
        if column in neurons:
            return neurons[column].fillna("?").astype(str)
    return np.full(len(neurons), "?")


def build_groups(connectome, net, motor_indices, start_data=None, grouping="fine"):
    """Group existing plastic edges. ``net`` must still hold ORIGINAL weights."""
    from noesis.motor import build_muscle_matrices

    from .bridge import LEGACY_JOINTS

    motor_indices = np.unique(np.asarray(motor_indices, np.int64))
    edges = np.flatnonzero(np.isin(net.indices, motor_indices))
    original = net.data[edges].copy()
    start = original.copy() if start_data is None else np.asarray(start_data, float)[edges].copy()
    pre = np.searchsorted(net.indptr, edges, side="right") - 1
    post = net.indices[edges]
    neurons = connectome.neurons
    pos, neg = build_muscle_matrices(neurons, list(LEGACY_JOINTS))
    pool = {}
    for sign, matrix in ((+1, pos), (-1, neg)):
        coo = matrix.tocoo()
        for j, idx in zip(coo.row, coo.col):
            joint = LEGACY_JOINTS[j].removeprefix("joint_")
            if grouping == "coarse":  # share a gain across the six legs
                joint = joint[2:]
            pool.setdefault(int(idx), f"MN:{joint}:{'+' if sign > 0 else '-'}")
    classes = neurons["Class"].fillna("?").astype(str).to_numpy() if "Class" in neurons else \
        np.full(len(neurons), "?")
    side, mere = np.asarray(_side(neurons)), np.asarray(_neuromere(neurons))
    label = {}
    for idx in np.unique(post):
        if idx in pool:
            label[idx] = pool[idx]
        elif grouping == "coarse":
            label[idx] = f"IN:{classes[idx]}"
        else:
            label[idx] = f"IN:{classes[idx]}:{side[idx]}:{mere[idx]}"
    names, group = {}, np.empty(len(edges), np.int64)
    pre_sign = np.where(original >= 0, "E", "I")
    for k, (p, s) in enumerate(zip(post, pre_sign)):
        key = f"{label[int(p)]}|{s}"
        group[k] = names.setdefault(key, len(names))
    return EdgeGroups(edges, original, start, group, list(names), motor_indices)


def infer_motor_sides(connectome):
    """Opt-in body-interface repair: give leg motor neurons with no annotated side
    (neither Nerve nor Soma side) the side that supplies most of their synaptic input.

    The connectome (edges, weights, signs) is unchanged; only the neuron-to-leg
    mapping used by the muscle model gains assignments. Returns an audit dict.
    """
    import pandas as pd

    from noesis.motor import _nerve_or_soma_side

    n = connectome.neurons
    side = _nerve_or_soma_side(n.fillna({"Nerve": ""}) if "Nerve" in n else n)
    side = side.where(side.isin(["L", "R"]))
    mn = n["Class"].eq("leg_motor_neuron").to_numpy()
    unknown = np.flatnonzero(mn & side.isna().to_numpy())
    csc = connectome.W.tocsc()
    code = side.map({"L": 1.0, "R": -1.0}).fillna(0.0).to_numpy()
    assigned = {"L": 0, "R": 0, "undecided": 0}
    new = n["Soma side"].astype(object).copy()
    for idx in unknown:
        rows = csc.indices[csc.indptr[idx]:csc.indptr[idx + 1]]
        w = np.abs(csc.data[csc.indptr[idx]:csc.indptr[idx + 1]])
        vote = float((w * code[rows]).sum()) / max(float((w * np.abs(code[rows])).sum()), 1e-12)
        if abs(vote) < 0.2:
            assigned["undecided"] += 1
            continue
        s_ = "L" if vote > 0 else "R"
        new.iloc[idx] = "left" if s_ == "L" else "right"
        assigned[s_] += 1
    import dataclasses

    # the connectome is a frozen dataclass: return a copy with the new neuron table
    connectome = dataclasses.replace(
        connectome, neurons=n.assign(**{"Soma side": pd.Series(new.to_numpy(), index=n.index)}))
    return connectome, {"unknown_side_leg_motor_neurons": int(len(unknown)), **assigned,
                        "rule": "input-synapse-weighted side majority, |vote| >= 0.2"}


class _WeightHolder:
    def __init__(self, indptr, indices, data):
        self.indptr, self.indices, self.data = indptr, indices, data


class Evaluator:
    """Per-process cache of the connectome; evaluates gain vectors in fresh bodies."""

    def __init__(self, data, cache, checkpoint, grouping="fine", infer_sides=False):
        from noesis.embodiment import load_training_brain
        from noesis.lif import LIFNetwork, LIFParams

        self.cn, _ = load_training_brain(Path(data), Path(cache))
        self.side_audit = None
        if infer_sides:
            self.cn, self.side_audit = infer_motor_sides(self.cn)
        net = LIFNetwork(self.cn.W, LIFParams(dt=0.5, da=1.0), seed=0)
        self.indptr, self.indices = net.indptr, net.indices
        self.base_data = net.data.copy()
        with np.load(checkpoint, allow_pickle=False) as cp:
            motor = cp["motor_indices"].copy()
        from .controller import import_checkpoint

        original = net.data.copy()
        import_checkpoint(net, checkpoint)  # validates topology/signs/bounds
        self.parent_data = net.data.copy()
        net.data = original
        self.groups = build_groups(self.cn, net, motor, self.parent_data, grouping)

    def data_for(self, gains):
        data = self.parent_data.copy()
        data[self.groups.edge_indices] = self.groups.weights(gains)
        return data

    def episode(self, gains, seed, seconds, objective, record=None, frame_cb=None):
        from .backend import FlyGymBatch
        from .controller import FrozenBrain
        from .learning import score_transition, standing_mask

        started = time.perf_counter()
        body = FlyGymBatch(1, "cpu")
        holder = _WeightHolder(self.indptr, self.indices, self.data_for(gains))
        brain = FrozenBrain(self.cn, body.rest, body.lo, body.hi, seed=seed, shared_weights=holder)
        chunks = int(round(seconds * 1000 / CONTROL_MS))
        before = {k: v[0] for k, v in body.observe().items()}
        start_pos, heading = before["position"].copy(), before["heading"].copy()
        heading = heading[:2] / max(np.linalg.norm(heading[:2]), 1e-12)
        rewards, stand, slip, up, height, feet_load, speeds = [], [], [], [], [], [], []
        reach, ref_reach = [], None  # horizontal claw-to-thorax distance per leg
        both = {"curriculum-standing-v7": [], "curriculum-gripped-step-v8": []}
        if frame_cb:
            frame_cb(body, 0.0, before, start_pos, heading)
        for chunk in range(chunks):
            command, pads = brain.step(before, CONTROL_MS)
            body.advance(command[None], pads[None], SUBSTEPS)
            after = {k: v[0] for k, v in body.observe().items()}
            if chunk >= SETTLE_CHUNKS:
                for name, values in both.items():
                    values.append(score_transition(before, after, CONTROL_MS, objective=name)["reward"])
                s = score_transition(before, after, CONTROL_MS, objective=objective)
                rewards.append(s["reward"])
                slip.append(s["stance_slip_mm_s"])
                speeds.append(s["forward_speed_mm_s"])
                stand.append(bool(standing_mask(after)))
                up.append(float(after["up_body"][2]))
                height.append(float(after["position"][2]))
                feet_load.append(after["foot_load"] > 1e-5)
                d = np.linalg.norm(after["feet_position"][:, :2] - after["position"][:2], axis=1)
                ref_reach = d.copy() if ref_reach is None else ref_reach
                reach.append(d)
            if record is not None:
                record.append({k: np.asarray(v).copy() for k, v in after.items()})
            if frame_cb:
                frame_cb(body, (chunk + 1) * CONTROL_MS / 1000, after, start_pos, heading)
            before = after
        if not rewards:
            raise ValueError("episode shorter than settling interval")
        loads = np.array(feet_load)
        # a leg "really supports" only with its claw on the floor AND the leg extended
        # (>= 70% of its settled reach); a leg folded under the body does not count
        extended = np.array(reach) >= 0.7 * ref_reach
        support_share = (loads & extended).mean(axis=0)
        touchdowns = (loads[1:] & ~loads[:-1]).sum(axis=0)  # raw contact flicker count
        steps = debounced_steps(loads)
        displacement = (before["position"][:2] - start_pos[:2])
        return {
            "seed": int(seed), "seconds": float(seconds), "objective": objective,
            "mean_reward": float(np.mean(rewards)),
            "mean_reward_v7_stand": float(np.mean(both["curriculum-standing-v7"])),
            "mean_reward_v8_step": float(np.mean(both["curriculum-gripped-step-v8"])),
            "standing_fraction": float(np.mean(stand)),
            "mean_stance_slip_mm_s": float(np.mean(slip)),
            "mean_uprightness": float(np.mean(up)),
            "mean_height_mm": float(np.mean(height)),
            "min_height_mm": float(np.min(height)),
            "mean_supporting_feet": float(loads.sum(axis=1).mean()),
            "forward_mm": float(displacement @ heading),
            "path_mm": float(np.sum(np.abs(speeds)) * CONTROL_MS / 1000),
            "touchdowns_per_leg": touchdowns.astype(int).tolist(),
            "debounced_steps_per_leg": steps,
            "extended_support_share_per_leg": [round(float(x), 4) for x in support_share],
            "extended_share_per_leg": [round(float(x), 4) for x in extended.mean(axis=0)],
            "wall_seconds": time.perf_counter() - started,
            "total_spikes": int(brain.total_spikes),
        }

    def save_checkpoint(self, gains, path):
        """Write a MotorPlasticity-compatible checkpoint (loadable by every project tool)."""
        from noesis.lif import LIFNetwork, LIFParams
        from noesis.plasticity import MotorPlasticity

        net = LIFNetwork(self.cn.W, LIFParams(dt=0.5, da=1.0), seed=0)
        learner = MotorPlasticity(net, self.groups.motor_indices, learning_rate=0.2,
                                  tau_eligibility_ms=100, max_change=BOUND, rule="perturbation",
                                  normalize_reward_error=True, bias_correct_reward_error=True)
        if not np.array_equal(learner.edge_indices, self.groups.edge_indices):
            raise RuntimeError("plastic edge set mismatch")
        net.data[self.groups.edge_indices] = self.groups.weights(gains)
        learner.save(path)
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def debounced_steps(loads, min_swing=8, min_stance=8):
    """Real steps per leg: a swing (no claw load) lasting >= min_swing control steps (40 ms)
    followed by a stance lasting >= min_stance steps. Raw contact flicker does not count."""
    loads = np.asarray(loads, bool)
    out = []
    for leg in loads.T:
        runs, value, length = [], leg[0], 0
        for x in leg:
            if x == value:
                length += 1
            else:
                runs.append((bool(value), length))
                value, length = x, 1
        runs.append((bool(value), length))
        out.append(int(sum(1 for (a, la), (b, lb) in zip(runs, runs[1:])
                           if not a and la >= min_swing and b and lb >= min_stance)))
    return out


def fitness(result, stage, displacement_weight=0.05):
    """Episode-level scalar to maximise.

    stand: mean curriculum-standing-v7 reward (support, posture, low slip) plus a bonus
           for every leg carrying weight while extended (mean and minimum over legs).
    walk : mean curriculum-gripped-step-v8 reward plus net forward displacement
           per simulated second and the share of legs that really step, both
           multiplied by the standing fraction (all six legs must participate).
    """
    share = np.clip(np.asarray(result.get("extended_support_share_per_leg", [0] * 6)) / 0.5, 0, 1)
    all_legs = 0.1 * share.mean() + 0.1 * share.min()  # every leg must carry weight, extended
    if stage == "stand":
        return result["mean_reward"] + all_legs
    per_second = result["forward_mm"] / result["seconds"]
    # all-leg participation: share of the six legs making >= 1 real (debounced) step per second
    need = max(1, int(round(result["seconds"])))
    legs = sum(n >= need for n in result.get("debounced_steps_per_leg", [])) / 6
    return result["mean_reward"] + all_legs + result["standing_fraction"] * (
        displacement_weight * per_second + 0.15 * legs)


STAGE_OBJECTIVE = {"stand": "curriculum-standing-v7", "walk": "curriculum-gripped-step-v8"}


def _json_default(o):
    """Convert numpy scalars/arrays (e.g. np.bool_ from gate checks) to plain Python."""
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False, default=_json_default), encoding="utf-8")
