"""Rendered vision and physical sensor integration, with optional FlyGym."""

from types import SimpleNamespace

import numpy as np
import pytest

flygym = pytest.importorskip('flygym')
mujoco = pytest.importorskip('mujoco')

from noesis.environment_senses import contact_body_touch, contact_tastes
from noesis.habitat import Habitat, habitat_arena
from noesis.world import CONTACT_SOLREF, FLOOR_COLLISIONS, GRAVITY, configure_floor_contacts


def test_real_eyes_antennae_and_obstacle_contacts_follow_world_changes():
    arena = habitat_arena()
    fly = flygym.Fly(enable_vision=True, enable_olfaction=True, vision_refresh_rate=20,
                     floor_collisions=FLOOR_COLLISIONS, contact_solref=CONTACT_SOLREF)
    sim = flygym.SingleFlySimulation(fly=fly, arena=arena, cameras=[], gravity=GRAVITY)
    configure_floor_contacts(sim)
    try:
        original, _ = sim.reset()
        assert original['vision'].shape == (2, 721, 2)
        assert original['odor_intensity'].shape == (6, 4)
        assert np.isfinite(original['vision']).all()
        physics = sim.physics
        model = physics.model.ptr
        obstacle = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'obstacle')
        thorax = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f'{fly.name}/Thorax')
        assert np.any(((model.pair_geom1 == obstacle) & (model.pair_geom2 == thorax)) |
                      ((model.pair_geom2 == obstacle) & (model.pair_geom1 == thorax)))
        target = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'sensor_visual_target')
        assert not np.any((model.pair_geom1 == target) | (model.pair_geom2 == target))

        # Change a real rendered material, not the observation array.
        physics.named.model.mat_rgba['grid'] = (0, 0, 1, 1)
        fly._last_vision_update_time = -np.inf
        blue = fly.get_observation(sim)['vision'].copy()
        physics.named.model.mat_rgba['grid'] = (0, 1, 0, 1)
        fly._last_vision_update_time = -np.inf
        green = fly.get_observation(sim)['vision'].copy()
        assert np.mean(np.abs(green - blue)) > 0.01
        assert green[:, :, 0].mean() > blue[:, :, 0].mean()
        assert blue[:, :, 1].mean() > green[:, :, 1].mean()

        # Move the whole fly; its physical antenna positions determine odor input.
        root_joint = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)[0]
        start = model.jnt_qposadr[root_joint]
        with physics.reset_context():
            physics.data.qpos[start:start + 2] += (3.2, 1.8)
        near = fly.get_observation(sim)['odor_intensity']
        assert near[0].mean() > original['odor_intensity'][0].mean()

        before = physics.named.data.mocap_pos['visual_target'].copy()
        arena.step(0.5, physics)
        assert not np.allclose(physics.named.data.mocap_pos['visual_target'], before)
    finally:
        sim.close()


@pytest.mark.parametrize('part,location', [('LFTarsus1', 0), ('LHFemur', 2), ('Haustellum', 6)])
def test_real_contact_force_activates_only_touching_taste_organ(part, location):
    model = mujoco.MjModel.from_xml_string(f'''<mujoco>
      <option timestep="0.0001" gravity="0 0 -9810"/>
      <default><geom solref="0.0002 1" margin="0.05"/></default>
      <worldbody><geom name="ground" type="plane" size="10 10 1"/>
        <body pos="3.2 1.8 0.1"><freejoint/>
          <geom name="test/{part}" type="sphere" size="0.05" mass="0.001"/>
        </body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    sim = SimpleNamespace(physics=SimpleNamespace(model=SimpleNamespace(ptr=model),
                                                  data=SimpleNamespace(ptr=data)))
    fly = SimpleNamespace(name='test')
    for _ in range(100):
        mujoco.mj_step(model, data)
    taste = contact_tastes(sim, fly, Habitat())
    assert taste[location, 0] == 1
    assert taste.sum() == 1
    touch = contact_body_touch(sim, fly)
    expected_touch_site = 2 if location == 6 else 3 + location
    assert touch[expected_touch_site] > 0
    assert np.count_nonzero(touch) == 1
    data.qpos[0] = 0  # Still touching the floor, but outside the food patch.
    mujoco.mj_forward(model, data)
    assert not contact_tastes(sim, fly, Habitat()).any()
