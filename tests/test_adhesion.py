import numpy as np
import pytest

from noesis.adhesion import FootAdhesion, adhesion_from_lift


def test_release_follows_physical_lift_with_three_support_pads_remaining():
    assert np.array_equal(adhesion_from_lift([.1, -.1, .2, .3, -.2, .4]), [1, 1, 0, 0, 1, 0])
    assert adhesion_from_lift(np.ones(6)).sum() == 3
    assert adhesion_from_lift(np.ones(6), max_swing=2).sum() == 4
    assert adhesion_from_lift(np.ones(6), max_swing=0).sum() == 6
    assert adhesion_from_lift(np.zeros(6)).sum() == 6
    with pytest.raises(ValueError):
        adhesion_from_lift(np.ones(6), max_swing=4)


def test_real_femur_sign_and_release_do_not_mutate_physics():
    pytest.importorskip('flygym')
    from flygym import SingleFlySimulation

    from noesis.body import make_fly
    from noesis.world import GRAVITY, floor_arena
    fly = make_fly(body_proxies=False)
    sim = SingleFlySimulation(fly=fly, cameras=[], timestep=1e-4,
                              arena=floor_arena(), gravity=GRAVITY)
    try:
        obs, _ = sim.reset()
        names = list(fly.actuated_joints)
        controller = FootAdhesion(sim.physics, fly, names)
        original = sim.physics.data.ptr.qpos.copy()
        target = obs['joints'][0].astype(float).copy()
        femur = names.index('joint_LFFemur')
        target[femur] -= .05
        assert controller.command(target)[0] == 0  # Negative Femur actually lifts LF.
        target[femur] += .10
        assert controller.command(target)[0] == 1  # Positive Femur lowers, must retain grip.
        np.testing.assert_array_equal(sim.physics.data.ptr.qpos, original)
    finally:
        sim.close()
