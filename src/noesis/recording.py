"""External evidence camera with optional bounded-memory video encoding."""
import logging
from pathlib import Path

import numpy as np


class _FrameCount:
    """Preserve FlyGym's render clock without retaining its image arrays."""

    def __init__(self):
        self.count = 0

    def __len__(self):
        return self.count

    def append(self, frame):
        self.count += 1

    def clear(self):
        self.count = 0


def make_follow_camera(fly, arena, window_size=(960, 640), output_path=None):
    """Create one episode's camera; stream when an output path is supplied.

    FlyGym uses len(_frames) to schedule rendering. A count-only replacement
    retains the same clock while each image is sent directly to the encoder.
    The temporary movie becomes final only after explicit save_video succeeds.
    """
    from flygym import Camera

    destination = Path(output_path).resolve() if output_path is not None else None

    class FloorFollowingCamera(Camera):
        def __init__(self, *args, **kwargs):
            self._destination = destination
            self._writer = None
            self._encoded_frames = 0
            self._closed = False
            self._saved = False
            super().__init__(*args, **kwargs)
            if destination is not None:
                self._frames = _FrameCount()

        @property
        def retained_frame_count(self):
            return len(self._frames) if self._destination is None else 0

        @property
        def encoded_frame_count(self):
            return self._encoded_frames

        @property
        def partial_path(self):
            return self._destination.with_name(
                self._destination.stem + '.partial' + self._destination.suffix)

        def reset(self):
            if self._destination is not None and self._encoded_frames:
                raise RuntimeError('create a new recording camera for each episode')
            super().reset()

        def _update_camera(self, physics, floor_height, obs):
            position = np.asarray(obs['pos'], dtype=float).copy()
            position[2] = floor_height
            physics.bind(self._cam).xpos = position + self.camera_base_offset

        def render(self, physics, floor_height, curr_time, last_obs=None):
            if self._destination is not None and self._closed:
                raise RuntimeError('recording is already closed')
            frame = super().render(physics, floor_height, curr_time, last_obs)
            if self._destination is not None and frame is not None and curr_time >= .02:
                if self._writer is None:
                    import imageio.v2 as imageio
                    self._destination.parent.mkdir(parents=True, exist_ok=True)
                    self._writer = imageio.get_writer(self.partial_path, fps=self.fps)
                self._writer.append_data(frame)
                self._encoded_frames += 1
            return frame

        def close(self):
            writer, self._writer = self._writer, None
            if self._destination is not None:
                self._closed = True
            if writer is not None:
                writer.close()

        def save_video(self, path, stabilization_time=.02):
            if self._destination is None:
                return super().save_video(path, stabilization_time=stabilization_time)
            if Path(path).resolve() != self._destination:
                raise ValueError('streamed movie must use its configured output path')
            if stabilization_time != .02:
                raise ValueError('streaming uses the configured 20 ms stabilization window')
            if self._saved:
                return
            self.close()
            if not self._encoded_frames:
                logging.getLogger(__name__).warning('No post-stabilization frames; no final movie saved.')
                return
            self.partial_path.replace(self._destination)
            self._saved = True

    return FloorFollowingCamera(
        attachment_point=arena.root_element.worldbody,
        camera_name='training_follow', targeted_fly_names=[fly.name],
        camera_parameters={'mode': 'fixed', 'pos': (0., -7., 4.8),
                           'euler': (float(np.arctan2(7., 4.8)), 0., 0.), 'fovy': 45},
        window_size=window_size, play_speed=.1, fps=30, timestamp_text=True)
