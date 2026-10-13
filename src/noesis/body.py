"""Physical body: skeleton, joint limits, passive tissue and antagonistic muscles.

This module only describes what the body *is*. It makes no decisions: the brain's motor
neurons activate muscles and physics does the rest.

    motor-neuron rates -> agonist/antagonist activation (with activation dynamics)
    -> equilibrium joint angle of the muscle pair -> stiff tissue servo -> physics

Muscles are modelled with the equilibrium-point idea: an antagonistic pair sets the angle the
joint is pulled toward, and the tissue supplies the stiffness. Zero activation of both muscles
leaves the joint at the standing posture; full activation of the agonist (antagonist) pulls it
to the upper (lower) end of its range of motion. Direct torque control was tried first and is
numerically infeasible for this body: leg-segment inertias (~1e-7) are ~9 orders of magnitude
below the torques needed to bear the body's weight, so any soft joint limit is overpowered.

Everything here comes from real kinematics where possible and is otherwise a labelled assumption:

* **Standing posture** and **joint limits** come from the real stepping kinematics shipped with
  NeuroMechFly (``single_steps_untethered.pkl``): the posture is the step-cycle mean of every
  joint; the limits are the observed step range widened by a margin (the true range of motion is
  larger than what walking uses, so the margin is an assumption).
* **Activation curve and activation time constant** are assumptions of the model.
* The muscle->joint-direction table (``noesis.motor.MUSCLE_JOINT``) is anatomical judgement.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse

LEGS = ("LF", "LM", "LH", "RF", "RM", "RH")
DOFS = ("Coxa", "Coxa_roll", "Coxa_yaw", "Femur", "Femur_roll", "Tibia", "Tarsus1")
JOINTS = tuple(f"joint_{leg}{d}" for leg in LEGS for d in DOFS)  # NeuroMechFly leg order


@dataclass(frozen=True)
class BodyParams:
    r_half: float = 25.0        # Hz of pool firing giving half activation
    tau_act_ms: float = 30.0    # muscle activation/relaxation time constant
    servo_kp: float = 45.0      # tissue stiffness toward the equilibrium angle (NeuroMechFly default)
    servo_forcerange: float = 65.0
    joint_damping: float = 1.0   # tissue viscosity; 0.06 (NeuroMechFly default) overshoots limits
    limit_margin_frac: float = 0.25   # joint-limit margin, as a fraction of the stepping span
    limit_margin_rad: float = 0.35    # ... plus this many radians
    limit_solref: float = 0.002       # joint-limit time constant (s): small = hard stop
    muscle_reach: float = 0.9         # muscles reach this fraction of the way to a limit; the
                                      # last 10% is only entered by external forces


def _kinematics() -> dict[str, np.ndarray]:
    import flygym

    path = Path(flygym.__file__).parent / "data" / "behavior" / "single_steps_untethered.pkl"
    with open(path, "rb") as f:
        steps = pickle.load(f)
    return {j: np.asarray(steps[j], float) for j in JOINTS}


def standing_pose() -> dict[str, float]:
    """Step-cycle mean of every leg joint of a real walking fly."""
    return {j: float(a.mean()) for j, a in _kinematics().items()}


def joint_limits(params: BodyParams | None = None) -> dict[str, tuple[float, float]]:
    """Observed stepping range widened by a margin; always contains the standing posture."""
    p = params or BodyParams()
    out = {}
    for j, a in _kinematics().items():
        span = a.max() - a.min()
        m = p.limit_margin_frac * span + p.limit_margin_rad
        out[j] = (float(a.min() - m), float(a.max() + m))
    return out


def make_fly(params: BodyParams | None = None, spawn_pos=(0.0, 0.0, 1.5), floor="all", **fly_kwargs):
    """A NeuroMechFly body in the standing posture with joint limits and tissue servos."""
    from flygym import Fly
    from flygym.state import KinematicPose

    p = params or BodyParams()
    pose = standing_pose()
    kwargs = {"enable_adhesion": True, "floor_collisions": floor}
    kwargs.update(fly_kwargs)
    fly = Fly(
        init_pose=KinematicPose(joint_pos=pose), control="position", spawn_pos=spawn_pos,
        actuator_gain=p.servo_kp, actuator_forcerange=p.servo_forcerange,
        joint_damping=p.joint_damping, **kwargs,
    )
    for name, (lo, hi) in joint_limits(p).items():
        joint = fly.model.find("joint", name)
        joint.limited = "true"
        joint.range = (lo, hi)
        joint.solreflimit = (p.limit_solref, 1.0)
    return fly


class MuscleModel:
    """Antagonistic muscle pair per joint; equilibrium angle from motor-neuron pool firing rates."""

    def __init__(self, g_pos: sparse.csr_matrix, g_neg: sparse.csr_matrix, joint_names: list[str],
                 params: BodyParams | None = None):
        self.g_pos, self.g_neg = g_pos, g_neg
        self.p = params or BodyParams()
        pose, limits = standing_pose(), joint_limits(self.p)
        self.stand = np.array([pose[j] for j in joint_names])
        self.lo = np.array([limits[j][0] for j in joint_names])
        self.hi = np.array([limits[j][1] for j in joint_names])
        r = self.p.muscle_reach
        self.hi_eq = self.stand + r * (self.hi - self.stand)  # what full agonist activation reaches
        self.lo_eq = self.stand - r * (self.stand - self.lo)
        self.a_pos = np.zeros(len(joint_names))
        self.a_neg = np.zeros(len(joint_names))

    def reset(self) -> None:
        self.a_pos[:] = 0.0
        self.a_neg[:] = 0.0

    def step(self, rates_hz: np.ndarray, dt_ms: float) -> np.ndarray:
        """Advance muscle activation by ``dt_ms`` and return the joint equilibrium angles."""
        pos = np.maximum(self.g_pos @ rates_hz, 0.0)
        neg = np.maximum(self.g_neg @ rates_hz, 0.0)
        k = 1.0 - np.exp(-dt_ms / self.p.tau_act_ms)
        self.a_pos += k * (pos / (pos + self.p.r_half) - self.a_pos)
        self.a_neg += k * (neg / (neg + self.p.r_half) - self.a_neg)
        return self.equilibrium()

    def equilibrium(self) -> np.ndarray:
        theta = (self.stand + self.a_pos * (self.hi_eq - self.stand)
                 - self.a_neg * (self.stand - self.lo_eq))
        return np.clip(theta, self.lo_eq, self.hi_eq)
