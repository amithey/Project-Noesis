"""External evidence camera; follows position without changing sensory inputs."""
import numpy as np


def make_follow_camera(fly, arena, window_size=(960, 640)):
    from flygym import Camera

    class FloorFollowingCamera(Camera):
        def _update_camera(self, physics, floor_height, obs):
            # Keep world orientation and floor height so turns/falls stay visible.
            # Copy the observation: some FlyGym camera variants mutate obs['pos'].
            position = np.asarray(obs['pos'], dtype=float).copy()
            position[2] = floor_height
            physics.bind(self._cam).xpos = position + self.camera_base_offset

    return FloorFollowingCamera(
        attachment_point=arena.root_element.worldbody,
        camera_name='training_follow', targeted_fly_names=[fly.name],
        camera_parameters={'mode': 'fixed', 'pos': (0., -7., 4.8),
                           'euler': (float(np.arctan2(7., 4.8)), 0., 0.), 'fovy': 45},
        window_size=window_size, play_speed=.1, fps=30, timestamp_text=True)
