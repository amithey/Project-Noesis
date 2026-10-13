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

# Solid body: every leg segment collides with the head, eyes, thorax and abdomen, so a leg cannot
# pass through them (NeuroMechFly lets legs collide only with each other). The body parts are
# meshes, which are too slow to collide every step (24x slower), so each gets an invisible
# ellipsoid proxy sized from the mesh's bounding box. Coxae are left out: fused to the thorax.
BODY_GEOMS = ("Thorax", "Head", "LEye", "REye", "A1A2", "A3", "A4", "A5", "A6")
LEG_GEOMS = tuple(f"{side}{pos}{seg}" for side in "LR" for pos in "FMH"
                  for seg in ("Femur", "Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5"))
PROXY_SCALE = 0.9  # proxies are slightly smaller than the mesh so the resting posture is not in contact


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


def make_fly(params: BodyParams | None = None, spawn_pos=(0.0, 0.0, 1.5), floor="all",
             body_proxies=True, **fly_kwargs):
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
    if body_proxies:
        _add_body_proxies(fly)
    for name, (lo, hi) in joint_limits(p).items():
        joint = fly.model.find("joint", name)
        joint.limited = "true"
        joint.range = (lo, hi)
        joint.solreflimit = (p.limit_solref, 1.0)
    return fly


def _add_body_proxies(fly) -> None:
    """Add invisible ellipsoid proxies (sized from each mesh's bounding box) for distance queries.

    They have no contacts: making them collide dynamically crashed the physics (``BADQACC``) even
    in plain standing, because stiff servos and tiny segment masses do not mix with extra contact
    pairs. ``CollisionGuard`` measures leg-to-proxy distances instead.
    """
    from dm_control import mjcf

    phys = mjcf.Physics.from_mjcf_model(fly.model)  # compile once to read the mesh bounding boxes
    named = phys.named.model
    for g in BODY_GEOMS:
        key = g if g in named.geom_size.axes.row.names else f"{fly.name}/{g}"
        geom = fly.model.find("geom", g)
        geom.parent.add(
            "geom", name=f"proxy_{g}", type="ellipsoid",
            size=PROXY_SCALE * np.asarray(named.geom_size[key]),
            pos=named.geom_pos[key], quat=named.geom_quat[key],
            contype=0, conaffinity=0, rgba=(0, 0, 0, 0), group=5,
            mass=0,  # distance queries only: must not add inertia to the body
            dclass=geom.dclass,  # else FlyGym mistakes the proxy for part of the floor
        )


class CollisionGuard:
    """Keeps muscle commands from driving a leg into the head, eyes, thorax or abdomen.

    A stiff tissue servo overpowers contact forces (segment inertias are ~1e-8), so contact alone
    cannot stop a leg from entering the head, and extra contact pairs destabilise the physics. A real
    muscle cannot push a leg through its own body, so the commanded equilibrium pose is shortened
    along the line from the current pose to the last pose that keeps a clearance from the body
    proxies. Distances are exact (``mj_geomDistance``); leg-leg collisions stay with FlyGym.
    """

    def __init__(self, physics, fly, joint_names, buffer_mm: float = 0.1, slack_mm: float = 0.005,
                 steps: int = 8, reach_mm: float = 0.6):
        import mujoco

        self._mj = mujoco
        self.model = physics.model.ptr
        self.live = physics.data.ptr
        self.scratch = mujoco.MjData(self.model)
        obj = mujoco.mjtObj.mjOBJ_GEOM
        ident = lambda g: mujoco.mj_name2id(self.model, obj, f"{fly.name}/{g}")
        self.qadr = np.array([
            self.model.jnt_qposadr[mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{fly.name}/{j}")] for j in joint_names])
        self.pairs = [(ident(leg), ident(f"proxy_{b}")) for leg in LEG_GEOMS for b in BODY_GEOMS]
        self.buffer, self.slack, self.steps, self.reach = buffer_mm, slack_mm, steps, reach_mm
        self.shortened = 0  # how many commands had to be shortened (diagnostic)
        self._fromto = np.zeros(6)

    def _gap(self, data) -> float:
        """Smallest signed distance (mm, negative = overlap) from any leg segment to a body proxy."""
        d = self.reach
        best = d
        for g1, g2 in self.pairs:
            best = min(best, self._mj.mj_geomDistance(self.model, data, g1, g2, d, self._fromto))
        return best

    def live_gap(self) -> float:
        """Current leg-to-body gap in the running simulation (use after a physics step)."""
        return self._gap(self.live)

    def _clearance(self, q: np.ndarray) -> float:
        d = self.scratch
        d.qpos[:] = self.live.qpos
        d.qpos[self.qadr] = q
        self._mj.mj_kinematics(self.model, d)
        return self._gap(d)

    def filter(self, current: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Largest fraction of the way from ``current`` to ``target`` that keeps the clearance.

        A pose is acceptable if no leg comes closer than ``buffer_mm`` to the body -- or, when it is
        already closer than that, no closer than now.
        """
        need = min(self.buffer, self._clearance(current)) - self.slack
        if self._clearance(target) >= need:
            return target
        best = 0.0
        for k in range(1, self.steps):
            frac = k / self.steps
            if self._clearance(current + frac * (target - current)) < need:
                break
            best = frac
        self.shortened += 1
        return current + best * (target - current)


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
