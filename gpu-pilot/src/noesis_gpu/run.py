"""Run the upstream GPU body or a frozen-brain compatibility pilot; never promotes weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from .backend import UPSTREAM_COMMIT, FlyGymBatch


def source_digest(root):
    digest = hashlib.sha256()
    for path in sorted((root / "noesis").rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def timing_steps(seconds, control_ms, physics_dt):
    values = np.asarray([seconds, control_ms, physics_dt], dtype=float)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("Durations must be finite and positive")
    n_chunks = seconds * 1000 / control_ms
    physics_steps = control_ms / 1000 / physics_dt
    neural_steps = control_ms / 0.5
    if any(not np.isclose(value, round(value), rtol=0, atol=1e-8)
           for value in (n_chunks, physics_steps, neural_steps)):
        raise ValueError("Duration must contain whole control, physics and neural steps")
    return round(n_chunks), round(physics_steps)


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", choices=("gpu", "cpu"), default="gpu")
    ap.add_argument("--controller", choices=("brain", "neutral"), default="brain")
    ap.add_argument("--worlds", type=int, default=1)
    ap.add_argument("--seconds", type=float, default=0.1)
    ap.add_argument("--control-ms", type=float, default=5.0)
    ap.add_argument("--physics-dt", type=float, default=0.0001)
    ap.add_argument("--wall-seconds", type=float, default=600)
    ap.add_argument("--seed", type=int, default=10)
    ap.add_argument("--noesis-source", type=Path, help="explicit existing src/ with noesis package")
    ap.add_argument("--data", type=Path)
    ap.add_argument("--checkpoint", type=Path, help="frozen weight import using existing loader")
    ap.add_argument("--bg-hz", type=float, default=3.0)
    ap.add_argument("--bg-weight", type=float, default=40.0)
    ap.add_argument("--host-memory-budget-gib", type=float, default=8)
    ap.add_argument("--output", type=Path, required=True, help="new directory; never overwrites")
    return ap


def run(args):
    n_chunks, physics_steps = timing_steps(args.seconds, args.control_ms, args.physics_dt)
    if args.worlds < 1 or args.seed < 0 or args.seed + args.worlds >= 2**32:
        raise ValueError("Invalid world count or seed range")
    if not np.isfinite(args.wall_seconds) or args.wall_seconds <= 0:
        raise ValueError("Invalid wall budget")
    for name in ("bg_hz", "bg_weight", "host_memory_budget_gib"):
        value = getattr(args, name)
        if not np.isfinite(value) or value < 0 or (name == "host_memory_budget_gib" and value == 0):
            raise ValueError(f"Invalid {name}")
    if args.controller == "brain":
        if not args.noesis_source or not args.data:
            raise ValueError("Brain pilot requires --noesis-source and --data")
        args.noesis_source = args.noesis_source.resolve()
        if not (args.noesis_source / "noesis" / "lif.py").is_file():
            raise ValueError("--noesis-source must point to the intended src directory")
        sys.path.insert(0, str(args.noesis_source))
    elif args.checkpoint is not None:
        raise ValueError("Neutral body smoke test cannot use a brain checkpoint")
    if args.data is not None:
        output, data = args.output.resolve(), args.data.resolve()
        if output == data or data in output.parents:
            raise ValueError("Pilot output must be outside the read-only data directory")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "initializing", "backend": args.backend, "controller": args.controller,
        "worlds": args.worlds, "requested_simulated_seconds_per_world": args.seconds,
        "control_ms": args.control_ms, "physics_dt": args.physics_dt,
        "bg_hz": args.bg_hz, "bg_weight": args.bg_weight,
        "upstream_commit": UPSTREAM_COMMIT, "python": sys.version,
        "packages": {}, "learning": False, "motor_exploration": 0,
        "model_promoted": False, "walking_proven": False, "v1_body_equivalence_verified": False,
        "neural_execution": "cpu_sequential" if args.controller == "brain" else "none",
        "senses": ["joint_position", "joint_velocity", "ground_load", "balance"],
        "external_vision_and_chemical_senses": False,
        "rng_schedule": "per-world per-chunk reseeding; not legacy bitwise replay",
        "seeds": list(range(args.seed, args.seed + args.worlds)),
        "completed_control_chunks": 0,
    }
    for package in ("flygym", "mujoco", "mujoco-warp", "warp-lang", "numba", "numpy"):
        try:
            report["packages"][package] = version(package)
        except PackageNotFoundError:
            report["packages"][package] = None
    started = time.perf_counter()
    trace, population, body = [], None, None
    try:
        body = FlyGymBatch(args.worlds, args.backend, args.physics_dt)
        report["device"] = body.device
        report["body"] = {"neutral_pose": body.rest.tolist(), "lower_limits": body.lo.tolist(),
                          "upper_limits": body.hi.tolist(), "servo_kp": 45.0,
                          "force_range": [-65.0, 65.0], "damping": 1.0,
                          "noslip_iterations": 0, "axis_order": "yaw_pitch_roll"}
        if args.controller == "brain":
            from noesis.embodiment import load_training_brain

            from .controller import FrozenPopulation

            report["noesis_source"] = str(args.noesis_source)
            report["noesis_source_sha256"] = source_digest(args.noesis_source)
            if args.checkpoint:
                report["checkpoint"] = str(args.checkpoint.resolve())
                report["checkpoint_sha256"] = file_digest(args.checkpoint)
            cn, scale = load_training_brain(args.data, args.output / "brain-cache")
            report["region_scale"] = scale
            report["neurons"], report["edges"] = cn.n, cn.W.nnz
            population = FrozenPopulation(cn, body, seed=args.seed,
                                          memory_budget_gib=args.host_memory_budget_gib,
                                          checkpoint=args.checkpoint, bg_hz=args.bg_hz,
                                          bg_weight=args.bg_weight)
            report["estimated_neural_host_bytes"] = population.estimated_host_bytes
        observation = body.observe()
        initial_position = observation["position"].copy()
        loop_start = time.perf_counter()
        report["setup_wall_seconds"] = loop_start - started
        for chunk in range(n_chunks):
            if time.perf_counter() - started >= args.wall_seconds:
                report["status"] = "incomplete_wall_budget"
                break
            if population is None:
                commands = np.broadcast_to(body.rest, (args.worlds, 42)).copy()
                adhesion = np.ones((args.worlds, 6))
            else:
                commands, adhesion = population.step(observation, args.control_ms)
            body.advance(commands, adhesion, physics_steps)
            observation = body.observe()
            trace.append({key: value.copy() for key, value in observation.items()})
            report["completed_control_chunks"] = chunk + 1
        else:
            report["status"] = "completed_transport_pilot"
        loop_seconds = time.perf_counter() - loop_start
        completed_seconds = report["completed_control_chunks"] * args.control_ms / 1000
        report.update(simulated_seconds_per_world=completed_seconds,
                      total_simulated_world_seconds=completed_seconds * args.worlds,
                      loop_wall_seconds=loop_seconds,
                      world_seconds_per_wall_second=completed_seconds * args.worlds / loop_seconds,
                      displacement_mm=(observation["position"] - initial_position).tolist())
        if population is not None:
            report["spikes_per_world"] = [brain.total_spikes for brain in population.brains]
            report["weights_unchanged"] = all(brain.weights_unchanged() for brain in population.brains)
            if not report["weights_unchanged"]:
                raise RuntimeError("Frozen weights changed during the transport pilot")
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report["total_wall_seconds"] = time.perf_counter() - started
        if trace:
            np.savez_compressed(args.output / "observations.npz", **{
                key: np.stack([sample[key] for sample in trace]) for key in trace[0]})
        (args.output / "report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return report


def main():
    ap = parser()
    args = ap.parse_args()
    try:
        report = run(args)
    except (ValueError, RuntimeError, MemoryError, ImportError, FileExistsError) as error:
        print(f"GPU pilot: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "completed_transport_pilot" else 3


if __name__ == "__main__":
    raise SystemExit(main())
