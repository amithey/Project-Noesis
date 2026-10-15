"""Real rendering regression: evidence camera keeps distant and fallen flies visible."""
import numpy as np
import pytest

from noesis.body import make_fly
from noesis.recording import make_follow_camera
from noesis.world import floor_arena


def test_follow_camera_keeps_body_visible_without_mutating_body_or_observations(tmp_path):
    flygym = pytest.importorskip('flygym')
    mj = pytest.importorskip('mujoco')
    fly = make_fly(body_proxies=False, enable_vision=True, vision_refresh_rate=20)
    arena = floor_arena()
    camera = make_follow_camera(fly, arena, window_size=(320, 240))
    sim = flygym.SingleFlySimulation(fly=fly, cameras=[camera], timestep=1e-4, arena=arena)
    try:
        sim.reset(seed=0)
        model, data = sim.physics.model.ptr, sim.physics.data.ptr
        base = data.qpos.copy()
        rotation = None
        for index, (x, y, fallen) in enumerate(((0., 0., False), (15., -8., False), (-20., 10., True))):
            data.qpos[:] = base
            data.qpos[:2] += (x, y)
            if fallen:
                data.qpos[3:7] = (0., 1., 0., 0.)
            sim.physics.forward()
            sim.curr_time = float(index+1)
            observation = fly.get_observation(sim)
            before_obs = {key: np.array(value, copy=True) for key, value in observation.items()}
            before_last = {key: np.array(value, copy=True) for key, value in fly.last_obs.items()}
            before_state = [data.qpos.copy(), data.qvel.copy(), data.ctrl.copy(),
                            data.qacc.copy(), data.qacc_warmstart.copy()]
            state_spec = mj.mjtState.mjSTATE_INTEGRATION
            integration = np.empty(mj.mj_stateSize(model, state_spec))
            mj.mj_getState(model, data, integration, state_spec)
            cam_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, camera.camera_id)
            other_cameras = np.arange(model.ncam) != cam_id
            eye_positions = data.cam_xpos[other_cameras].copy()
            eye_rotations = data.cam_xmat[other_cameras].copy()
            frame = camera.render(sim.physics, 0., sim.curr_time, [fly.last_obs])
            assert frame.shape == (240, 320, 3)
            # The fly's eyes/body have red/brown pixels; gray floor is excluded.
            visible = (frame[..., 0].astype(float) > 1.3*frame[..., 1]) & (frame[..., 0] > 100)
            assert visible.sum() > 100
            for actual, expected in zip((data.qpos, data.qvel, data.ctrl, data.qacc, data.qacc_warmstart), before_state, strict=True):
                np.testing.assert_array_equal(actual, expected)
            after_integration = np.empty_like(integration)
            mj.mj_getState(model, data, after_integration, state_spec)
            np.testing.assert_array_equal(after_integration, integration)
            np.testing.assert_array_equal(data.cam_xpos[other_cameras], eye_positions)
            np.testing.assert_array_equal(data.cam_xmat[other_cameras], eye_rotations)
            for key, expected in before_obs.items():
                np.testing.assert_array_equal(observation[key], expected)
            for key, expected in before_last.items():
                np.testing.assert_array_equal(fly.last_obs[key], expected)
            cam = sim.physics.bind(camera._cam)
            np.testing.assert_allclose(cam.xpos, [fly.last_obs['pos'][0], fly.last_obs['pos'][1]-7., 4.8], atol=1e-10)
            if rotation is None:
                rotation = cam.xmat.copy()
            np.testing.assert_array_equal(cam.xmat, rotation)
            # Keep actual frames for visual inspection when run outside pytest cleanup.
            import imageio.v2 as imageio
            imageio.imwrite(tmp_path/f'camera-{index}.png', frame)
    finally:
        sim.close()
