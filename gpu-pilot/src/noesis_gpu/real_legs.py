"""Opt-in realistic leg ranges for the FlyGym 2 body (Amit, 2026-11-26).

The pilot body uses the native v2 neutral pose as muscle rest and an arbitrary
+-0.7 rad range on every joint. That gives every leg the same symmetric window,
which is not a fly's range: knees saturate straight, hind legs end up as rigid props.

This applies the calibration the original Noesis body (core ``noesis.body``) used:
* rest (muscle-off equilibrium) = step-cycle mean of each pitch joint in real fly
  walking kinematics (NeuroMechFly ``single_steps_untethered.pkl``),
* range = observed stepping range widened by 25% of the span + 0.35 rad,
with the v1/v2 convention safeguards described in ``real_ranges``.
The legs are posed to that stance before the episode. Connectome and muscle
equations are unchanged; only the body's rest pose and joint limits change.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

MARGIN_FRAC, MARGIN_RAD = 0.25, 0.35


def find_kinematics(path=None):
    candidates = [Path(path)] if path else []
    try:
        import flygym

        candidates.append(Path(flygym.__file__).parent / "data/behavior/single_steps_untethered.pkl")
    except ImportError:
        pass
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError("single_steps_untethered.pkl not found; pass --kinematics PATH "
                            "(it ships with FlyGym 1.x under flygym/data/behavior/)")


PITCH_SHIFT_LIMIT = 0.6  # rad; larger kinematic-vs-native rest offsets are treated as convention gaps


def real_ranges(kinematics_path, joint_names, native_rest):
    """Per-joint rest and limits from real walking, made safe for FlyGym 2 conventions.

    v1 kinematics mirror right-side roll/yaw angles while v2 does not, and femur-roll
    signs disagree (audit 2026-11-26, legs-fix.md). So:
    * pitch joints (Coxa, Femur, Tibia, Tarsus1): rest = real step-cycle mean and the
      real min/max (+margin), unless the mean is > 0.6 rad from the native rest; then the
      real window shape is shifted onto the native rest;
    * roll/yaw joints: native rest, symmetric window of the real half-span + margin.
    """
    with open(kinematics_path, "rb") as f:
        steps = pickle.load(f)  # trusted NeuroMechFly data file
    rest, lo, hi, rule = [], [], [], []
    for j, nat in zip(joint_names, native_rest):
        a = np.asarray(steps[j], float)
        mean, span = a.mean(), a.max() - a.min()
        m = MARGIN_FRAC * span + MARGIN_RAD
        if j.endswith(("_roll", "_yaw")):
            half = span / 2 + m
            rest.append(nat), lo.append(nat - half), hi.append(nat + half), rule.append("sym")
        elif abs(mean - nat) <= PITCH_SHIFT_LIMIT:
            rest.append(mean), lo.append(a.min() - m), hi.append(a.max() + m), rule.append("real")
        else:
            rest.append(nat), lo.append(nat + a.min() - mean - m), hi.append(nat + a.max() - mean + m)
            rule.append("shifted")
    return np.array(rest), np.array(lo), np.array(hi), rule


def apply_real_legs(body, kinematics_path):
    """Set rest/limits of the 42 active leg joints and pose the legs at the real stance."""
    import mujoco

    from .bridge import LEGACY_JOINTS

    sim = body.sim
    joint_ids = sim._internal_jointids_by_fly[body.fly.name][body.read_order]
    qadr = sim.mj_model.jnt_qposadr[joint_ids]
    native = sim.mj_data.qpos[qadr].copy()
    rest, lo, hi, rule = real_ranges(kinematics_path, LEGACY_JOINTS, native)
    sim.mj_model.jnt_range[joint_ids] = np.column_stack((lo, hi))
    sim.mj_data.qpos[qadr] = rest
    sim.mj_data.qvel[:] = 0
    mujoco.mj_forward(sim.mj_model, sim.mj_data)
    body.rest, body.lo, body.hi = rest.copy(), lo.copy(), hi.copy()
    return {"kinematics": str(kinematics_path),
            "max_abs_shift_from_native_rad": float(np.max(np.abs(rest - native))),
            "mean_range_rad": float(np.mean(hi - lo)),
            "rules": {k: rule.count(k) for k in ("real", "shifted", "sym")},
            "per_joint": {j: [round(float(r), 3), round(float(a), 3), round(float(b), 3), u]
                          for j, r, a, b, u in zip(LEGACY_JOINTS, rest, lo, hi, rule)}}
