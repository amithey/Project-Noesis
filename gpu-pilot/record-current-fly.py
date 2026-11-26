"""Record ten real simulated seconds, at real-time playback, without learning."""
import argparse
import json
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from noesis_gpu.backend import FlyGymBatch
from noesis_gpu.controller import FrozenPopulation
from noesis_gpu.run import file_digest, source_digest

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--output", type=Path, required=True)
ap.add_argument("--preview-only", action="store_true")
args = ap.parse_args()
root = Path(__file__).resolve().parent
project = root.parent
source = Path("C:/Users/heyma/.codex/worktrees/brain-body-feedback/Project Noesis/src")
state = json.loads((root / "continuation-status.json").read_text(encoding="utf-8"))
checkpoint = Path(state["latest_experimental_training_checkpoint"])
assert file_digest(checkpoint) == state["latest_experimental_training_sha256"]
args.output.mkdir(parents=True, exist_ok=False)
sys.path.insert(0, str(source))
from noesis.embodiment import load_training_brain

started = time.perf_counter()
body = FlyGymBatch(1, "cpu")
body.observe()  # populate world transforms before the first camera frame
model, data = body.sim.mj_model, body.sim.mj_data
model.vis.global_.offwidth = 960
model.vis.global_.offheight = 640
renderer = mujoco.Renderer(model, height=640, width=960)
camera = mujoco.MjvCamera()
mujoco.mjv_defaultCamera(camera)
camera.lookat[:] = [0, 0, .5]
camera.distance = 7
camera.azimuth = 125
camera.elevation = -25
font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 22)
small = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 17)


def frame(t):
    renderer.update_scene(data, camera=camera)
    result = Image.new("RGB", (960, 720), (18, 22, 29))
    result.paste(Image.fromarray(renderer.render()), (0, 80))
    draw = ImageDraw.Draw(result)
    draw.text((18, 10), f"Noesis brain + FlyGym 2 | Latest experimental weights | t = {t:05.2f} s",
              font=font, fill="white")
    draw.text((18, 44), "CPU | No learning or motor exploration | 1x speed | Standing improved; walking not yet verified",
              font=small, fill=(203, 214, 229))
    return np.asarray(result)


Image.fromarray(frame(0)).save(args.output / "preview.png")
if args.preview_only:
    renderer.close()
    print(args.output / "preview.png")
    raise SystemExit(0)

report = {"status": "initializing", "seconds": 10, "fps": 30, "seed": 10007,
          "backend": "cpu", "learning": False, "motor_exploration": 0,
          "checkpoint": str(checkpoint), "checkpoint_sha256": file_digest(checkpoint),
          "noesis_source_sha256": source_digest(source),
          "script_sha256": file_digest(__file__), "camera": "fixed; no following or stabilization",
          "model_promoted": False, "walking_proven": False, "completed_chunks": 0}
trace = []
writer = None
try:
    cn, _ = load_training_brain(project / "data", args.output / "brain-cache")
    population = FrozenPopulation(cn, body, seed=10007, checkpoint=checkpoint)
    writer = imageio.get_writer(str(args.output / "fly.partial.mp4"), fps=30, codec="libx264",
                               quality=8, pixelformat="yuv420p", macro_block_size=1)
    writer.append_data(frame(0))
    frames = 1
    observation = body.observe()
    for chunk in range(2000):
        if time.perf_counter() - started > 1800:
            raise TimeoutError("Recording wall budget exhausted; partial is not a completed video")
        commands, adhesion = population.step(observation, 5.)
        body.advance(commands, adhesion, 50)
        observation = body.observe()
        trace.append({key: value.copy() for key, value in observation.items()})
        t = (chunk + 1) * .005
        if frames < 300 and t + 1e-9 >= frames / 30:
            writer.append_data(frame(t))
            frames += 1
        report["completed_chunks"] = chunk + 1
        if (chunk + 1) % 100 == 0:
            print(json.dumps({"seconds": t, "frames": frames}), flush=True)
    assert frames == 300
    assert all(brain.weights_unchanged() for brain in population.brains)
    assert file_digest(checkpoint) == report["checkpoint_sha256"]
    assert source_digest(source) == report["noesis_source_sha256"]
    writer.close()
    writer = None
    count, duration = imageio_ffmpeg.count_frames_and_secs(str(args.output / "fly.partial.mp4"))
    assert count == 300 and abs(duration - 10) < .05, (count, duration)
    (args.output / "fly.partial.mp4").rename(args.output / "fly-10-seconds.mp4")
    report.update(status="completed", encoded_frames=count, encoded_duration=duration,
                  simulated_seconds=10., weights_unchanged=True,
                  video_sha256=file_digest(args.output / "fly-10-seconds.mp4"))
except BaseException as error:
    report.update(status="incomplete", error=str(error))
    raise
finally:
    if writer is not None:
        writer.close()
    renderer.close()
    if trace:
        np.savez_compressed(args.output / "observations.npz", **{
            key: np.stack([sample[key] for sample in trace]) for key in trace[0]})
    report["wall_seconds"] = time.perf_counter() - started
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
