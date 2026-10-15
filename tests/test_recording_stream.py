"""Long capture keeps the same clock/pixels and releases each frame immediately."""
import gc
import hashlib
import weakref
from pathlib import Path
from types import SimpleNamespace

import imageio
import imageio.v2
import numpy as np
import pytest
from dm_control import mjcf

from noesis.recording import make_follow_camera


@pytest.fixture
def encoder(monkeypatch):
    state = SimpleNamespace(writers={}, fail_on=None)

    class Writer:
        def __init__(self, path, **kwargs):
            self.path = Path(path)
            self.path.write_bytes(b'partial')
            self.count, self.closes = 0, 0
            self.digest = hashlib.sha256()
            state.writers[str(self.path.resolve())] = self

        def append_data(self, frame):
            if state.fail_on == self.count+1:
                raise OSError('injected encoder failure')
            self.digest.update(frame.tobytes())
            self.count += 1

        def close(self):
            self.closes += 1

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    monkeypatch.setattr(imageio, 'get_writer', Writer)
    monkeypatch.setattr(imageio.v2, 'get_writer', Writer)
    return state


class Physics:
    def __init__(self):
        self.bound = SimpleNamespace(xpos=np.zeros(3))
        self.count = 0

    def bind(self, camera):
        return self.bound

    def render(self, width, height, camera_id):
        self.count += 1
        return np.full((height, width, 3), self.count % 255, dtype=np.uint8)


def camera(path=None):
    arena = SimpleNamespace(root_element=mjcf.RootElement(model='record_test'))
    fly = SimpleNamespace(name='test_fly')
    return make_follow_camera(fly, arena, window_size=(32, 32), output_path=path)


def test_twenty_second_stream_matches_buffered_clock_and_pixels_without_retaining_frames(tmp_path, encoder):
    streamed_path, buffered_path = tmp_path/'streamed.mp4', tmp_path/'buffered.mp4'
    streamed, buffered = camera(streamed_path), camera()
    stream_physics, buffer_physics = Physics(), Physics()
    observation = {'pos': np.array([2., 3., .8])}
    references = []
    for step in range(200000):
        now = (step+1)*.0001
        frame = streamed.render(stream_physics, 0., now, [observation])
        if frame is not None:
            references.append(weakref.ref(frame))
        buffered.render(buffer_physics, 0., now, [observation])
    del frame
    gc.collect()
    assert all(ref() is None for ref in references)
    assert len(streamed._frames) == len(buffered._frames) > 5000
    assert streamed._timestamp_per_frame == buffered._timestamp_per_frame
    assert streamed.retained_frame_count == 0
    streamed.save_video(streamed_path)
    buffered.save_video(buffered_path)
    stream_writer = encoder.writers[str(streamed.partial_path)]
    buffer_writer = encoder.writers[str(buffered_path)]
    assert stream_writer.count == buffer_writer.count == streamed.encoded_frame_count
    assert stream_writer.digest.hexdigest() == buffer_writer.digest.hexdigest()
    assert stream_writer.closes == 1
    assert streamed_path.is_file() and not streamed.partial_path.exists()
    np.testing.assert_array_equal(observation['pos'], [2., 3., .8])


@pytest.mark.parametrize('fail_encoder', [False, True])
def test_interrupted_or_failed_encoder_keeps_partial_movie_unpublished(tmp_path, encoder, fail_encoder):
    destination = tmp_path/'unfinished.mp4'
    recorded = camera(destination)
    physics = Physics()
    obs = {'pos': np.array([0., 0., .8])}
    recorded.render(physics, 0., .03, [obs])
    if fail_encoder:
        encoder.fail_on = 2
        with pytest.raises(OSError, match='injected encoder failure'):
            recorded.render(physics, 0., .04, [obs])
    recorded.close()
    recorded.close()
    assert recorded.partial_path.is_file() and not destination.exists()
    assert encoder.writers[str(recorded.partial_path)].closes == 1
    with pytest.raises(RuntimeError, match='closed'):
        recorded.render(physics, 0., .05, [obs])
