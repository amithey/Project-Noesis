"""Record ten real simulated seconds of the fly at 30 fps (1x real-time playback)."""
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

# Add pilot src and core source
pilot_root = Path(__file__).resolve().parent
project_root = Path(r"C:\Users\heyma\OneDrive\Desktop\מדעי המחשב\פרוייקטים\Project Noesis")
source_root = Path(r"C:\Users\heyma\.codex\worktrees\brain-body-feedback\Project Noesis\src")
sys.path.insert(0, str(pilot_root / "src"))
sys.path.insert(0, str(source_root))

from noesis.embodiment import load_training_brain
from noesis_gpu.backend import FlyGymBatch
from noesis_gpu.controller import FrozenPopulation
from noesis_gpu.run import file_digest, source_digest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=10006)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--preview-only", action="store_true")
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    default_checkpoint = (
        project_root / "gpu-pilot/outputs/gripped-001-train-11001/weights-last.npz"
        if (project_root / "gpu-pilot/outputs/gripped-001-train-11001/weights-last.npz").exists()
        else project_root / "gpu-pilot/outputs/standing-001-train-11000/weights-last.npz"
    )
    checkpoint = (args.checkpoint or default_checkpoint).resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    data_dir = project_root / "data"

    started = time.perf_counter()
    body = FlyGymBatch(1, "cpu")
    body.observe()

    model, data = body.sim.mj_model, body.sim.mj_data
    model.vis.global_.offwidth = 960
    model.vis.global_.offheight = 640
    renderer = mujoco.Renderer(model, height=640, width=960)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = [0.0, 0.0, 0.5]
    camera.distance = 7.0
    camera.azimuth = 125.0
    camera.elevation = -25.0

    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 22)
        small = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 17)
    except Exception:
        font = ImageFont.load_default()
        small = ImageFont.load_default()

    def render_frame(t_sec, forward_mm=0.0):
        renderer.update_scene(data, camera=camera)
        canvas = Image.new("RGB", (960, 720), (18, 22, 29))
        canvas.paste(Image.fromarray(renderer.render()), (0, 80))
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (18, 10),
            f"Noesis Brain + FlyGym 2 | Candidate Stage 2 (Seed {args.seed}) | t = {t_sec:05.2f} s",
            font=font,
            fill="white",
        )
        draw.text(
            (18, 44),
            f"CPU Sequential | Frozen Evaluation | Fwd Dist: {forward_mm:+.2f} mm | 1x Speed",
            font=small,
            fill=(203, 214, 229),
        )
        return np.asarray(canvas)

    preview_img = render_frame(0.0, 0.0)
    preview_path = args.output / "preview.png"
    Image.fromarray(preview_img).save(preview_path)
    print(f"Saved preview to {preview_path}")

    if args.preview_only:
        renderer.close()
        return

    cn, _ = load_training_brain(data_dir, args.output / "brain-cache")
    population = FrozenPopulation(cn, body, seed=args.seed, checkpoint=checkpoint)

    total_chunks = round(args.seconds / 0.005)  # 2000 chunks for 10 s
    fps = 30
    total_video_frames = round(args.seconds * fps)  # 300 frames

    mp4_partial = args.output / "fly.partial.mp4"
    mp4_final = args.output / "fly-10-seconds.mp4"

    writer = imageio.get_writer(
        str(mp4_partial),
        fps=fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=1,
    )
    writer.append_data(preview_img)
    written_frames = 1

    trace = []
    report = {
        "status": "recording",
        "seconds": args.seconds,
        "fps": fps,
        "seed": args.seed,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_digest(checkpoint),
        "noesis_source_sha256": source_digest(source_root),
        "total_chunks": total_chunks,
    }

    try:
        observation = body.observe()
        initial_pos = observation["position"][0].copy()
        initial_head = observation["heading"][0].copy()
        unit_head = initial_head / (np.linalg.norm(initial_head) + 1e-12)

        for chunk in range(total_chunks):
            if time.perf_counter() - started > 1800:
                raise TimeoutError("Recording wall budget exceeded 1800 s")

            commands, adhesion = population.step(observation, 5.0)
            body.advance(commands, adhesion, 50)
            observation = body.observe()
            trace.append({k: v.copy() for k, v in observation.items()})

            t_sim = (chunk + 1) * 0.005
            curr_pos = observation["position"][0]
            fwd_mm = float(np.dot(curr_pos - initial_pos, unit_head))

            if written_frames < total_video_frames and t_sim + 1e-9 >= written_frames / fps:
                writer.append_data(render_frame(t_sim, fwd_mm))
                written_frames += 1

            if (chunk + 1) % 200 == 0:
                print(
                    json.dumps({
                        "simulated_seconds": round(t_sim, 2),
                        "frames_encoded": written_frames,
                        "forward_mm": round(fwd_mm, 3),
                        "elapsed_wall_s": round(time.perf_counter() - started, 1),
                    }),
                    flush=True,
                )

        writer.close()
        writer = None

        count, duration = imageio_ffmpeg.count_frames_and_secs(str(mp4_partial))
        if mp4_final.exists():
            mp4_final.unlink()
        mp4_partial.rename(mp4_final)

        report.update(
            status="completed",
            encoded_frames=count,
            encoded_duration=duration,
            video_sha256=file_digest(mp4_final),
            wall_seconds=time.perf_counter() - started,
        )
        (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nSuccessfully generated {mp4_final} ({count} frames, {duration:.2f} s)")

    finally:
        if writer is not None:
            writer.close()
        renderer.close()


if __name__ == "__main__":
    main()
