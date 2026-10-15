"""Real MuJoCo drop tests; skipped when optional body dependencies are absent."""

import numpy as np
import pytest

flygym = pytest.importorskip("flygym")
mujoco = pytest.importorskip("mujoco")

from noesis.world import (
    CONTACT_SOLREF,
    FLOOR_COLLISIONS,
    FLOOR_CONTACT_MARGIN_MM,
    GRAVITY,
    configure_floor_contacts,
    floor_arena,
)


@pytest.mark.parametrize("roll", [0.0, np.pi / 2, np.pi])
def test_gravity_and_whole_body_floor_collision(roll):
    fly = flygym.Fly(init_pose="stretch", control="position", enable_adhesion=True,
                     floor_collisions=FLOOR_COLLISIONS, spawn_pos=(0, 0, 4),
                     contact_solref=CONTACT_SOLREF,
                     spawn_orientation=(roll, 0, 0))
    sim = flygym.SingleFlySimulation(fly=fly, cameras=[], arena=floor_arena(),
                                    gravity=GRAVITY, timestep=1e-4)
    configure_floor_contacts(sim)
    try:
        obs, _ = sim.reset()
        model, data = sim.physics.model.ptr, sim.physics.data.ptr
        ground = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
        floor_pairs = (model.pair_geom1 == ground) | (model.pair_geom2 == ground)
        body_geoms = np.unique(np.concatenate([
            model.pair_geom1[floor_pairs], model.pair_geom2[floor_pairs]]))
        body_geoms = body_geoms[body_geoms != ground]
        # In particular the head, thorax and abdomen must have floor contact pairs.
        for part in ("Head", "Thorax", "A1A2", "A6", "LWing", "RWing"):
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{fly.name}/{part}")
            assert gid in body_geoms
        np.testing.assert_array_equal(sim.gravity, GRAVITY)
        assert model.geom_pos[ground, 2] == 0
        assert model.geom_type[ground] == mujoco.mjtGeom.mjGEOM_PLANE

        start_z = data.subtree_com[0, 2]
        sim.step({"joints": obs["joints"][0], "adhesion": np.ones(6, int)})
        sim.physics.step(9)
        drop = start_z - data.subtree_com[0, 2]
        assert 0.003 < drop < 0.007, f"incorrect free fall over 1 ms: {drop} mm"

        minimum_gap = float("inf")
        for _ in range(1200):  # Check every 0.1 ms step throughout impact and settling.
            sim.physics.step()
            distances = [mujoco.mj_geomDistance(model, data, ground, int(g), 100, None)
                         for g in body_geoms]
            minimum_gap = min(minimum_gap, min(distances))
            assert np.isfinite(data.qpos).all()
        assert minimum_gap < FLOOR_CONTACT_MARGIN_MM + 0.01, "fly never reached the floor"
        print(f"roll={roll:.3f}: min floor gap={minimum_gap:.6f} mm")
        # Permit only 2 micrometres of numerical error at the visible surface.
        assert minimum_gap > -0.002, f"body penetrated floor by {-minimum_gap:.6f} mm"
        assert data.subtree_com[0, 2] > 0, "body fell through the floor"
    finally:
        sim.close()


def test_extending_head_and_wings_keeps_independent_episode_bodies_identical():
    from noesis.body import make_fly
    from noesis.world import enable_wing_and_head_joints
    default_joints = list(flygym.preprogrammed.all_leg_dofs)
    snapshots = []
    for _ in range(2):
        fly = make_fly(body_proxies=False)
        enable_wing_and_head_joints(fly)
        assert flygym.preprogrammed.all_leg_dofs == default_joints
        assert fly.monitored_joints is not flygym.preprogrammed.all_leg_dofs
        # Repeated extension of the same body must also be idempotent.
        first_names = list(fly.actuated_joints)
        enable_wing_and_head_joints(fly)
        assert fly.actuated_joints == first_names
        sim = flygym.SingleFlySimulation(fly=fly, cameras=[], arena=floor_arena(),
                                        gravity=GRAVITY, timestep=1e-4)
        try:
            obs, _ = sim.reset(seed=10)
            m = sim.physics.model.ptr
            snapshots.append({k: np.asarray(getattr(m, k)).copy() for k in
                              ('dof_damping', 'actuator_gainprm', 'actuator_biasprm', 'jnt_stiffness')})
            # Check actual pre-neural settling too, not only the parameter manifest.
            for _ in range(200):
                sim.step({'joints': obs['joints'][0], 'adhesion': np.ones(6, int)})
            snapshots[-1]['settled_qpos'] = sim.physics.data.qpos.copy()
        finally:
            sim.close()
    for name in snapshots[0]:
        np.testing.assert_array_equal(snapshots[0][name], snapshots[1][name], err_msg=name)
