"""Held-out acceptance gates and the 10-second progress video for a trained gain vector.

Gates are fixed in advance (docs/claude-training/GATES.md) and never relaxed:
  held-out seeds 20001, 20002, 20003 (never used in training), 5 s frozen each,
  compared pairwise with the starting parent (all gains = 1) on the same seeds.

Usage:
  python es_evaluate.py --gains outputs/claude-es/R1/best-gains.npy --output outputs/claude-es/R1/eval
  python es_evaluate.py --gains ... --output ... --video-only --label "R1 ES stand gen 12"
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from es_train import DEFAULT_CORE, DEFAULT_PARENT, DEFAULT_PROJECT, _init, _paths  # noqa: E402

HELD_OUT = (20001, 20002, 20003)


def _eval(args):
    import es_train

    gains, seed, seconds, objective = args
    try:
        return es_train._EVAL.episode(gains, seed, seconds, objective)
    except Exception as error:
        return {"seed": seed, "error": f"{type(error).__name__}: {error}"}


def stand_gate(c, p):
    ok = all("error" not in r for r in c)
    checks = {
        "no_physics_errors": ok,
        "standing_ge_0.97_every_seed": ok and all(r["standing_fraction"] >= 0.97 for r in c),
        "uprightness_ge_0.85_every_seed": ok and all(r["mean_uprightness"] >= 0.85 for r in c),
        "slip_not_worse_than_parent_mean": ok and np.mean([r["mean_stance_slip_mm_s"] for r in c])
        <= np.mean([r["mean_stance_slip_mm_s"] for r in p]),
        # added 2026-11-26 after the leg audit (Amit: a hind leg is unused); tightening only
        "every_leg_extended_support_ge_0.3_every_seed": ok and all(
            min(r.get("extended_support_share_per_leg", [0])) >= 0.3 for r in c),
        "v7_reward_beats_parent_every_seed": ok and all(
            a["mean_reward_v7_stand"] > b["mean_reward_v7_stand"] + 1e-6 for a, b in zip(c, p)),
    }
    return checks


def walk_gate(c, p):
    ok = all("error" not in r for r in c)
    checks = {
        "no_physics_errors": ok,
        "forward_ge_2mm_every_seed": ok and all(r["forward_mm"] >= 2.0 for r in c),
        "standing_ge_0.90_every_seed": ok and all(r["standing_fraction"] >= 0.90 for r in c),
        "stepping_4plus_legs_3plus_touchdowns": ok and all(
            sum(t >= 3 for t in r["touchdowns_per_leg"]) >= 4 for r in c),
        # added 2026-11-25 after R0 showed raw touchdowns are contact flicker (tightening only)
        "real_steps_4plus_legs_2plus_debounced_steps": ok and all(
            sum(t >= 2 for t in r.get("debounced_steps_per_leg", [])) >= 4 for r in c),
        # added 2026-11-25 per Amit: stable walking must use ALL legs
        "all_6_legs_2plus_debounced_steps": ok and all(
            all(t >= 2 for t in r.get("debounced_steps_per_leg", [0])) and
            len(r.get("debounced_steps_per_leg", [])) == 6 for r in c),
        "every_leg_extended_support_ge_0.2_every_seed": ok and all(
            min(r.get("extended_support_share_per_leg", [0])) >= 0.2 for r in c),
        # added 2026-11-26 (Amit: a hind leg is dragged like a rigid prop)
        "every_leg_stride_ge_0.3mm_every_seed": ok and all(
            min(r.get("stride_mm_per_leg", [0])) >= 0.3 for r in c),
        "forward_beats_parent_every_seed": ok and all(
            a["forward_mm"] > b["forward_mm"] + 0.5 for a, b in zip(c, p)),
    }
    return checks


def video(evaluator, gains, seed, seconds, path, label):
    import imageio.v2 as imageio
    import mujoco
    from PIL import Image, ImageDraw, ImageFont

    state = {"renderer": None, "writer": None, "frames": 0, "camera": None}
    fps = 30
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 22)
        small = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 17)
    except Exception:
        font = small = ImageFont.load_default()

    def frame(body, t, obs, start, heading):
        if state["renderer"] is None:
            model = body.sim.mj_model
            model.vis.global_.offwidth, model.vis.global_.offheight = 960, 640
            state["renderer"] = mujoco.Renderer(model, height=640, width=960)
            cam = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(cam)
            cam.distance, cam.azimuth, cam.elevation = 7.0, 125.0, -25.0
            state["camera"] = cam
            state["writer"] = imageio.get_writer(str(path), fps=fps, codec="libx264", quality=7,
                                                 pixelformat="yuv420p", macro_block_size=1)
        if state["frames"] > 0 and t + 1e-9 < state["frames"] / fps:
            return
        cam = state["camera"]
        cam.lookat[:] = [obs["position"][0], obs["position"][1], 0.5]  # follow the fly
        r = state["renderer"]
        r.update_scene(body.sim.mj_data, camera=cam)
        canvas = Image.new("RGB", (960, 730), (18, 22, 29))
        canvas.paste(Image.fromarray(r.render()), (0, 90))
        d = ImageDraw.Draw(canvas)
        fwd = float((obs["position"][:2] - start[:2]) @ heading)
        feet = int((obs["foot_load"] > 1e-5).sum())
        d.text((18, 8), f"Noesis | {label} | t = {t:05.2f} s", font=font, fill="white")
        d.text((18, 42), f"frozen brain, seed {seed} | forward {fwd:+.2f} mm | height "
               f"{obs['position'][2]:.2f} mm | upright {obs['up_body'][2]:.2f} | feet on floor {feet}/6",
               font=small, fill=(203, 214, 229))
        d.text((18, 66), "connectome-driven muscles, no scripted gait | 1x speed", font=small,
               fill=(150, 160, 175))
        state["writer"].append_data(np.asarray(canvas))
        state["frames"] += 1

    try:
        result = evaluator.episode(gains, seed, seconds, "curriculum-gripped-step-v8", frame_cb=frame)
    finally:
        if state["writer"] is not None:
            state["writer"].close()
        if state["renderer"] is not None:
            state["renderer"].close()
    return {**result, "video": str(path), "frames": state["frames"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gains", type=Path, default=None, help="omit for the parent (all ones)")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--label", default="candidate")
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--video-seed", type=int, default=HELD_OUT[0])
    ap.add_argument("--video-only", action="store_true")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--grouping", choices=("fine", "coarse"), default="fine")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    ap.add_argument("--core", type=Path, default=DEFAULT_CORE)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_PARENT)
    ap.add_argument("--infer-mn-sides", action="store_true")
    ap.add_argument("--muscle", choices=("mean", "size"), default="mean")
    ap.add_argument("--legs", choices=("native", "real"), default="native")
    ap.add_argument("--kinematics", default=None)
    args = ap.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    data = (args.project / "data").resolve()
    cache = (args.project / "gpu-pilot/outputs/claude-es/brain-cache").resolve()
    _paths(args.core)
    import es_train
    from noesis_gpu.es_brain import Evaluator, dump

    es_train._init(args.core, data, cache, args.checkpoint, args.grouping, 1, args.infer_mn_sides,
                   args.muscle, args.legs, args.kinematics)
    ev = es_train._EVAL
    gains = (es_train.load_gains(args.gains, ev.groups.names) if args.gains
             else np.ones(ev.groups.n_groups))
    started = time.time()
    report = {"legs": args.legs, "muscle_model": args.muscle, "motor_side_inference": ev.side_audit, "label": args.label, "gains": str(args.gains), "held_out_seeds": HELD_OUT,
              "seconds": args.seconds}
    if not args.video_only:
        parent = np.ones(ev.groups.n_groups)
        jobs = [(g, s, args.seconds, "curriculum-gripped-step-v8") for g in (gains, parent) for s in HELD_OUT]
        ctx = mp.get_context("spawn")
        with ctx.Pool(min(args.workers, len(jobs)), initializer=_init,
                      initargs=(args.core, data, cache, args.checkpoint, args.grouping, 1,
                                args.infer_mn_sides, args.muscle, args.legs, args.kinematics)) as pool:
            res = pool.map(_eval, jobs, chunksize=1)
        cand, par = res[:3], res[3:]
        sg, wg = stand_gate(cand, par), walk_gate(cand, par)
        report.update(candidate=cand, parent=par, stand_gate=sg, walk_gate=wg,
                      stand_gate_passed=all(sg.values()), walk_gate_passed=all(wg.values()),
                      model_promoted=False,
                      walking_proven=False)
    if not args.no_video:
        report["video"] = video(ev, gains, args.video_seed, 10.0, out / "fly-10s.mp4", args.label)
    report["wall_seconds"] = time.time() - started
    dump(out / "evaluation.json", report)
    print(json.dumps({k: report.get(k) for k in ("label", "stand_gate_passed", "walk_gate_passed",
                                                 "stand_gate", "walk_gate")}, indent=2,
                     default=lambda o: o.item() if isinstance(o, np.generic) else str(o)))


if __name__ == "__main__":
    main()
