"""Pinned FlyGym 2.1 adapter; GPU physics remains the upstream implementation."""

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path

import numpy as np

from .bridge import checked_batch, joint_permutation, up_in_body

UPSTREAM_COMMIT = "ca65a510c2afe6ac61c51df4f274c8d190c2f95f"


def embed_mesh_assets(spec):
    """Use MuJoCo's VFS so native mesh loading also works in Unicode Windows paths."""
    for i, mesh in enumerate(spec.meshes):
        if not mesh.file:
            continue
        path = Path(mesh.file)
        if path.is_file():
            key = f"noesis_mesh_{i}{path.suffix}"
            spec.assets[key] = path.read_bytes()
            mesh.file = key


def require_cuda():
    try:
        import warp as wp
    except ImportError as error:
        raise RuntimeError("GPU backend requires FlyGym[warp] and an NVIDIA CUDA GPU; "
                           "no CPU fallback was substituted") from error

    devices = wp.get_cuda_devices()
    if not devices:
        raise RuntimeError("NVIDIA CUDA GPU unavailable; no CPU fallback was substituted")
    return str(devices[0])


class FlyGymBatch:
    """One model, independent batched states; the CPU reference accepts one world."""

    def __init__(self, n_worlds=1, backend="gpu", timestep=1e-4):
        if isinstance(n_worlds, bool) or int(n_worlds) != n_worlds or n_worlds < 1:
            raise ValueError("n_worlds must be a positive integer")
        if backend not in ("cpu", "gpu") or (backend == "cpu" and n_worlds != 1):
            raise ValueError("CPU reference supports exactly one world")
        if not np.isfinite(timestep) or timestep <= 0:
            raise ValueError("Invalid physics timestep")
        if version("flygym") != "2.1.0":
            raise RuntimeError("This adapter is pinned to FlyGym 2.1.0; use the isolated environment")
        self.device = require_cuda() if backend == "gpu" else "cpu"
        from flygym.anatomy import ActuatedDOFPreset, AxisOrder, JointPreset, Skeleton
        from flygym.compose import ActuatorType, FlatGroundWorld, KinematicPosePreset, NeuroMechFly
        from flygym.utils.math import Rotation3D

        self.backend, self.n_worlds, self.timestep = backend, n_worlds, timestep
        self.actuator_type = ActuatorType.POSITION
        self.fly = fly = NeuroMechFly()
        embed_mesh_assets(fly.mjcf_root)
        skeleton = Skeleton(axis_order=AxisOrder.YAW_PITCH_ROLL, joint_preset=JointPreset.LEGS_ONLY)
        pose = KinematicPosePreset.NEUTRAL
        fly.add_joints(skeleton, neutral_pose=pose, damping=1.0)
        dofs = skeleton.get_actuated_dofs_from_preset(ActuatedDOFPreset.LEGS_ACTIVE_ONLY)
        fly.add_actuators(dofs, actuator_type=self.actuator_type, kp=45.0,
                          forcerange=(-65.0, 65.0), neutral_input=pose)
        fly.add_leg_adhesion()
        world = FlatGroundWorld()
        world.add_fly(fly, (0, 0, 0.8), Rotation3D("quat", (1, 0, 0, 0)))
        # Keep whole-leg load for the existing sensory map, but score support
        # using distinct claw-to-floor sensors. A femur touching down is not a footstep.
        import mujoco as mj

        segments = fly.get_bodysegs_order()
        self.feet = np.array([next(i for i, b in enumerate(segments)
                                   if b.pos == leg and b.link == "tarsus5")
                              for leg in ("lf", "lm", "lh", "rf", "rm", "rh")])
        foot_sensor_names = []
        for i in self.feet:
            segment = segments[i]
            name = f"noesis_foot_ground_{segment.pos}"
            world.mjcf_root.add_sensor(
                name=name, type=mj.mjtSensor.mjSENS_CONTACT,
                objtype=mj.mjtObj.mjOBJ_XBODY, objname=fly.bodyseg_to_mjcfbody[segment].name,
                reftype=mj.mjtObj.mjOBJ_GEOM, refname=world.ground_geoms[0].name,
                intprm=[3, 3, 1],
            )
            foot_sensor_names.append(name)
        # Match the CPU reference to the solver option MuJoCo-Warp supports.
        # This is not a claim of equivalence to the old v1 training body.
        world.mjcf_root.option.noslip_iterations = 0
        if backend == "gpu":
            from flygym.warp import GPUSimulation

            self.sim = GPUSimulation(world, n_worlds, timestep=timestep)
        else:
            from flygym.simulation import Simulation

            self.sim = Simulation(world, timestep=timestep)
        # The native skeleton also has passive tarsus2..5 joints. Read only the
        # 42 active channels, while leaving passive joints in the physics model.
        all_dofs = fly.get_jointdofs_order()
        active_indices = np.array([i for i, dof in enumerate(all_dofs)
                                   if dof.child.link not in ("tarsus2", "tarsus3", "tarsus4", "tarsus5")])
        self.read_order = active_indices[joint_permutation([all_dofs[i] for i in active_indices])]
        self.action_order = joint_permutation(fly.get_actuated_jointdofs_order(self.actuator_type))
        self.native_action_order = np.argsort(self.action_order)
        self.thorax = next(i for i, b in enumerate(fly.get_bodysegs_order()) if b.link == "thorax")
        # Private mappings are isolated here and guarded by the exact dependency pin.
        sim = self.sim
        self.contact_ids = sim._intern_groundcontactsensorids_by_fly[fly.name]
        if len(self.contact_ids) != 96:
            raise RuntimeError("Expected six 16-channel ground-only contact sensors")
        foot_sensor_ids = [mj.mj_name2id(sim.mj_model, mj.mjtObj.mjOBJ_SENSOR, name)
                           for name in foot_sensor_names]
        if any(i < 0 or sim.mj_model.sensor_dim[i] != 4 for i in foot_sensor_ids):
            raise RuntimeError("Expected six 4-channel claw-to-ground sensors")
        self.foot_sensor_ids = np.concatenate([
            np.arange(sim.mj_model.sensor_adr[i], sim.mj_model.sensor_adr[i] + 4)
            for i in foot_sensor_ids])
        self.body_id = sim._internal_bodyids_by_fly[fly.name][self.thorax]
        joint_ids = sim._internal_jointids_by_fly[fly.name][self.read_order]
        qadr = sim.mj_model.jnt_qposadr[joint_ids]
        self.rest = sim.mj_data.qpos[qadr].copy()
        # Native v2 range is a pilot modelling assumption, not the v1 calibrated range.
        self.lo, self.hi = self.rest - 0.7, self.rest + 0.7
        sim.mj_model.jnt_limited[joint_ids] = True
        sim.mj_model.jnt_range[joint_ids] = np.column_stack((self.lo, self.hi))
        # GPU structs were created before our range edit; rebuild explicitly.
        if backend == "gpu":
            sim.mjw_model, sim.mjw_data = sim._mj_structs_to_mjw_structs()

    def _query(self, name):
        value = getattr(self.sim, name)(self.fly.name)
        return value.numpy() if self.backend == "gpu" else np.asarray(value)[None]

    def observe(self):
        sim = self.sim
        if self.backend == "gpu":
            import mujoco_warp as mjw

            mjw.forward(sim.mjw_model, sim.mjw_data)
            all_sensors = sim.mjw_data.sensordata.numpy()
            # cvel is [rotation, translation], expressed in world axes.
            omega_world = sim.mjw_data.cvel.numpy()[:, self.body_id, :3]
        else:
            import mujoco

            mujoco.mj_forward(sim.mj_model, sim.mj_data)
            all_sensors = sim.mj_data.sensordata[None]
            omega_world = sim.mj_data.cvel[self.body_id, :3][None]
        sensors = all_sensors[:, self.contact_ids].reshape(self.n_worlds, 6, 16)
        foot_sensors = all_sensors[:, self.foot_sensor_ids].reshape(self.n_worlds, 6, 4)
        q = self._query("get_body_rotations")[:, self.thorax]
        from scipy.spatial.transform import Rotation

        rotations = Rotation.from_quat(q[:, [1, 2, 3, 0]])
        positions = self._query("get_body_positions")
        observation = {
            "angles": self._query("get_joint_angles")[:, self.read_order],
            "velocity": self._query("get_joint_velocities")[:, self.read_order],
            "position": positions[:, self.thorax],
            "feet_position": positions[:, self.feet],
            "heading": rotations.apply(np.tile([1., 0., 0.], (self.n_worlds, 1))),
            "up_body": up_in_body(q),
            "omega": rotations.inv().apply(omega_world),
            "ground_contact": sensors[:, :, 0] > 0,
            "load": np.where(sensors[:, :, 0] > 0,
                             np.linalg.norm(sensors[:, :, 1:4], axis=2), 0.0),
            "foot_load": np.where(foot_sensors[:, :, 0] > 0,
                                  np.linalg.norm(foot_sensors[:, :, 1:4], axis=2), 0.0),
        }
        if not all(np.isfinite(value).all() for value in observation.values()):
            raise RuntimeError("Non-finite body observation")
        return observation

    def advance(self, commands, adhesion, steps):
        commands = checked_batch(commands, (self.n_worlds, 42), "commands")
        adhesion = checked_batch(adhesion, (self.n_worlds, 6), "adhesion")
        if np.any((commands < self.lo) | (commands > self.hi)):
            raise ValueError("Commands outside configured joint limits")
        if np.any((adhesion < 0) | (adhesion > 1)):
            raise ValueError("Adhesion outside [0, 1]")
        if isinstance(steps, bool) or int(steps) != steps or steps < 1:
            raise ValueError("Physics steps must be a positive integer")
        inputs = commands[:, self.native_action_order]
        pads = adhesion
        if self.backend == "cpu":
            inputs, pads = inputs[0], pads[0]
        self.sim.set_actuator_inputs(self.fly.name, self.actuator_type, inputs)
        self.sim.set_leg_adhesion_states(self.fly.name, pads)
        for _ in range(steps):
            self.sim.step()
