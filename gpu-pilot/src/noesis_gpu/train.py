"""One bounded training birth on the upstream body; no curriculum or model promotion."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from .backend import UPSTREAM_COMMIT, FlyGymBatch
from .run import file_digest, source_digest, timing_steps


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--noesis-source", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    ap.add_argument("--seconds", type=float, default=5.)
    ap.add_argument("--wall-seconds", type=float, default=600.)
    ap.add_argument("--seed", type=int, default=11000)
    return ap


def run(args):
    chunks, substeps = timing_steps(args.seconds, 5., .0001)
    if (not np.isfinite(args.wall_seconds) or not 0 < args.wall_seconds <= 3600
            or args.seconds <= .05 or not 0 <= args.seed < 2**32):
        raise ValueError("Require >50 ms duration, valid seed and 0 < wall budget <=3600 s")
    output, data = args.output.resolve(), args.data.resolve()
    if output == data or data in output.parents:
        raise ValueError("Output must be outside read-only data")
    source = args.noesis_source.resolve()
    if not (source / "noesis" / "circuit_learning.py").is_file():
        raise ValueError("Training requires the current isolated Noesis source")
    sys.path.insert(0, str(source))
    from noesis.embodiment import load_training_brain

    from .learning import LearningBrain, score_transition, standing_mask

    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = {"status": "initializing", "experimental": True, "learning": True,
              "worlds": 1, "backend": args.backend, "seed": args.seed,
              "upstream_commit": UPSTREAM_COMMIT, "noesis_source_sha256": source_digest(source),
              "import_checkpoint": str(args.checkpoint.resolve()),
              "import_checkpoint_sha256": file_digest(args.checkpoint),
              "reward_objective": "curriculum-standing-v7", "learning_rate": .2,
              "eligibility_ms": 100, "motor_exploration_mv_ms": 1.5,
              "original_weight_bound": .5, "control_ms": 5., "physics_dt": .0001,
              "birth_ms": 50, "requested_seconds": args.seconds, "completed_chunks": 0,
              "latest_training_checkpoint": None, "best_validated_checkpoint": None,
              "model_promoted": False, "walking_proven": False,
              "support_measurement": "tarsus5-to-floor-only",
              "external_vision_and_chemical_senses": False,
              "legacy_body_equivalence_verified": False}
    records, brain = [], None
    try:
        body = FlyGymBatch(1, args.backend)
        cn, _ = load_training_brain(data, output / "brain-cache")
        brain = LearningBrain(cn, body.rest, body.lo, body.hi,
                              checkpoint=args.checkpoint, seed=args.seed)
        report.update(neurons=cn.n, edges=cn.W.nnz,
                      plasticity_targets=len(brain.learner.motor_indices),
                      plasticity_edges=len(brain.learner.edge_indices))
        before = {key: value[0] for key, value in body.observe().items()}
        for chunk in range(chunks):
            if time.perf_counter() - started >= args.wall_seconds:
                report["status"] = "incomplete_wall_budget"
                break
            brain.learning_active = chunk >= 10
            command, adhesion = brain.step(before, 5.)
            body.advance(command[None], adhesion[None], substeps)
            after = {key: value[0] for key, value in body.observe().items()}
            score = score_transition(before, after, 5.)
            learning = brain.learn(score["reward"], 5.)
            records.append({**after, "command": command.copy(), "adhesion": adhesion.copy(),
                            "reward": score["reward"], "standing": bool(standing_mask(after)),
                            "stance_slip_mm_s": score["stance_slip_mm_s"],
                            "learning_active": brain.learning_active})
            before = after
            report["completed_chunks"] = chunk + 1
            if (chunk + 1) % 100 == 0:
                print(json.dumps({"event": "training_progress", "seconds": (chunk+1)*.005,
                    "standing": float(np.mean([r["standing"] for r in records[10:]])),
                    "reward": score["reward"], "learning": learning}), flush=True)
        else:
            report["status"] = "completed_training_birth"
        report.update(simulated_seconds=report["completed_chunks"] * .005,
                      learning_updates=brain.updates,
                      max_observed_exploration=brain.max_observed_exploration,
                      weights_changed=not brain.weights_unchanged(), learning_metrics=brain.learner.metrics())
        if len(records) > 10:
            report["standing_fraction"] = float(np.mean([r["standing"] for r in records[10:]]))
            report["mean_stance_slip_mm_s"] = float(np.mean([r["stance_slip_mm_s"] for r in records[10:]]))
        name = "weights-last.npz" if report["status"] == "completed_training_birth" else "weights-partial.npz"
        checkpoint = output / name
        brain.learner.save(checkpoint)
        context = {**report, "context": "experimental-v2-standing-only; not a validated curriculum bank"}
        checkpoint.with_suffix(".model.json").write_text(json.dumps(context, indent=2), encoding="utf-8")
        report["saved_checkpoint_sha256"] = file_digest(checkpoint)
        if report["status"] == "completed_training_birth":
            report["latest_training_checkpoint"] = str(checkpoint)
    except BaseException as error:
        report.update(status="incomplete_interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                      error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report["wall_seconds"] = time.perf_counter() - started
        if records:
            np.savez_compressed(output / "training-observations.npz", **{
                key: np.stack([row[key] for row in records]) for key in records[0]})
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


def main():
    args = parser().parse_args()
    report = run(args)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "completed_training_birth" else 3


if __name__ == "__main__":
    raise SystemExit(main())
