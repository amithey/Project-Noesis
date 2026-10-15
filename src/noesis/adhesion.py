"""Release adhesive feet from predicted physical lift, independent of joint sign."""
from __future__ import annotations

import numpy as np

from noesis.body import LEGS


def adhesion_from_lift(lift_mm, max_swing=3, threshold_mm=.02):
    """Release up to max_swing feet whose predicted vertical lift exceeds threshold."""
    lift = np.asarray(lift_mm, float)
    if lift.shape != (6,) or not np.isfinite(lift).all():
        raise ValueError("lift must contain six finite heights")
    if not isinstance(max_swing, (int, np.integer)) or not 0 <= max_swing <= 3:
        raise ValueError("max_swing must be an integer from 0 to 3")
    if not np.isfinite(threshold_mm) or threshold_mm < 0:
        raise ValueError("lift threshold must be finite and nonnegative")
    adhesion = np.ones(6, dtype=int)
    order = np.argsort(-lift, kind='stable')
    chosen = order[lift[order] > threshold_mm][:max_swing]
    adhesion[chosen] = 0
    return adhesion


class FootAdhesion:
    """Use MuJoCo forward kinematics in scratch data; live state is never changed.

    The 20 micrometre lift threshold and maximum three released pads are explicit
    controller assumptions. Selection follows commanded physical lift, with no
    preset leg sequence, gait clock or tripod grouping.
    """

    def __init__(self, physics, fly, joint_names, max_swing=3):
        import mujoco
        self.mj, self.model, self.live = mujoco, physics.model.ptr, physics.data.ptr
        self.scratch = mujoco.MjData(self.model)
        joints = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT,
                                  f'{fly.name}/{name}') for name in joint_names]
        self.feet = np.array([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                               f'{fly.name}/{leg}Tarsus5') for leg in LEGS])
        if min(joints, default=-1) < 0 or self.feet.min() < 0:
            raise ValueError("missing joint or foot in physical model")
        self.qadr = self.model.jnt_qposadr[joints]
        self.max_swing = max_swing

    def command(self, target):
        target = np.asarray(target, float)
        if target.shape != self.qadr.shape or not np.isfinite(target).all():
            raise ValueError("target must match actuated joints and be finite")
        self.scratch.qpos[:] = self.live.qpos
        self.mj.mj_kinematics(self.model, self.scratch)
        current_z = self.scratch.xpos[self.feet, 2].copy()
        self.scratch.qpos[self.qadr] = target
        self.mj.mj_kinematics(self.model, self.scratch)
        lift = self.scratch.xpos[self.feet, 2] - current_z
        return adhesion_from_lift(lift, max_swing=self.max_swing)
