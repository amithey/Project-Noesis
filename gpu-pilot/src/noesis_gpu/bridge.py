"""Anatomical ordering and muscle dynamics, with no generated stepping trajectory."""

from __future__ import annotations

import numpy as np

LEGS = ("LF", "LM", "LH", "RF", "RM", "RH")
DOFS = ("Coxa", "Coxa_roll", "Coxa_yaw", "Femur", "Femur_roll", "Tibia", "Tarsus1")
LEGACY_JOINTS = tuple(f"joint_{leg}{dof}" for leg in LEGS for dof in DOFS)


def legacy_joint_name(dof) -> str:
    """Map FlyGym 2 JointDOFs by anatomy, never by array position.

    This maps *channels*, not a proof of v1/v2 kinematic equivalence. FlyGym 2's
    native axis conventions and neutral posture are used in this pilot.
    """
    child, axis = dof.child, dof.axis.value
    leg = child.pos.upper()
    joint = {"coxa": "Coxa", "trochanterfemur": "Femur",
             "tibia": "Tibia", "tarsus1": "Tarsus1"}.get(child.link)
    allowed = {"Coxa": ("pitch", "roll", "yaw"), "Femur": ("pitch", "roll"),
               "Tibia": ("pitch",), "Tarsus1": ("pitch",)}
    if leg not in LEGS or joint is None or axis not in allowed[joint]:
        raise ValueError(f"Unsupported anatomical joint: {dof}")
    return f"joint_{leg}{joint}" + (f"_{axis}" if axis != "pitch" else "")


def joint_permutation(dofs) -> np.ndarray:
    """Indices that reorder native channels into Noesis's canonical 42 channels."""
    names = [legacy_joint_name(dof) for dof in dofs]
    if len(names) != len(set(names)) or set(names) != set(LEGACY_JOINTS):
        raise ValueError("Expected exactly one of each of the 42 leg joint channels")
    return np.array([names.index(name) for name in LEGACY_JOINTS], dtype=np.int64)


def checked_batch(value, shape, name):
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return array


def up_in_body(quaternions):
    """World up in the body frame, from MuJoCo's (w,x,y,z) quaternions."""
    q = np.asarray(quaternions, dtype=float)
    if q.ndim != 2 or q.shape[1] != 4 or not np.isfinite(q).all():
        raise ValueError("Expected finite batched quaternions")
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    if np.any(norm < 1e-12):
        raise ValueError("Zero quaternion")
    w, x, y, z = (q / norm).T
    return np.column_stack((2 * (x * z - w * y), 2 * (y * z + w * x),
                            1 - 2 * (x * x + y * y)))


class Muscles:
    """Same activation/equilibrium equations as Noesis MuscleModel, native v2 limits.

    Each controller owns its activation state; no shared state across worlds.
    """

    def __init__(self, g_pos, g_neg, rest, lo, hi, params):
        self.g_pos, self.g_neg, self.p = g_pos, g_neg, params
        self.rest = checked_batch(rest, (42,), "rest").copy()
        self.lo = checked_batch(lo, (42,), "lower limits").copy()
        self.hi = checked_batch(hi, (42,), "upper limits").copy()
        if np.any(self.lo > self.rest) or np.any(self.hi < self.rest):
            raise ValueError("Neutral pose must be inside all joint limits")
        self.hi_eq = self.rest + params.muscle_reach * (self.hi - self.rest)
        self.lo_eq = self.rest - params.muscle_reach * (self.rest - self.lo)
        self.a_pos, self.a_neg = np.zeros(42), np.zeros(42)

    def step(self, rates, dt_ms):
        if not np.isfinite(dt_ms) or dt_ms <= 0:
            raise ValueError("Muscle timestep must be positive")
        pos, neg = (np.maximum(g @ rates, 0) for g in (self.g_pos, self.g_neg))
        k = 1 - np.exp(-dt_ms / self.p.tau_act_ms)
        self.a_pos += k * (pos / (pos + self.p.r_half) - self.a_pos)
        self.a_neg += k * (neg / (neg + self.p.r_half) - self.a_neg)
        return np.clip(self.rest + self.a_pos * (self.hi_eq - self.rest)
                       - self.a_neg * (self.rest - self.lo_eq), self.lo, self.hi)


def adhesion_from_command(desired, rest):
    """Release pads in response to flexor commands only; no leg order or oscillator."""
    lift = (np.asarray(desired) - rest).reshape(6, 7)[:, 3]
    return (lift <= 0.08).astype(float)
