"""Persistent worker for the pool-gain evolution: loads the brain once, then runs frozen episodes on request.

Each task is one frozen locomotion episode (quiet arena, learning off, exploration off; same settings as
``scripts/frozen_probe.py``) with an optional leg muscle-pool genome (see ``noesis.pool_genome``). Per-task output
directories are deleted after the summary is extracted, because only the summary numbers are needed.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

_STATE: dict = {}


def init(data: str, cache: str) -> None:
    from noesis.embodiment import load_training_brain
    cn, _ = load_training_brain(Path(data), Path(cache))
    _STATE["cn"], _STATE["data"] = cn, Path(data)


def make_case(data: Path, output: Path, seed: int, awake_seconds: float, genome=None, birth: float = 0.05):
    """The protocol's frozen job (V24/V25 registration) for one seed; mirrors ``scripts/frozen_probe.py``."""
    from noesis.experiments import learning_lab as lab
    template = lab.parser().parse_args(['--data', str(data), '--output', str(output / 'probe'), '--seconds', '5',
                                        '--learning-rate', '0.2', '--arena', 'quiet', '--depression-u', '0', '--learn'])
    template.grounded_objective = 'standing-gate-linear-slip-v5'
    template.training_fall_seconds = 0.1
    template.proprioceptive_encoding = 'legacy'
    template.normalize_reward_error = True
    template.bias_correct_reward_error = True
    template.bg_weight = 40.0
    lab.validate_args(lab.parser(), template)
    case = argparse.Namespace(**vars(template))
    case.seed, case.arena, case.learn = seed, 'quiet', False
    case.seconds, case.training_fall_seconds = birth + awake_seconds, 0.0
    case.motor_exploration = 0.0
    case.birth, case.warmup_ms = birth, 0.0
    case.pool_genome = None if genome is None else [float(g) for g in genome]
    case.pool_gains = None
    case.import_weights, case.no_environment_senses, case.video = False, False, False
    case.output = output
    return case


def run_task(task: dict) -> dict:
    """task: {key, seed, genome (list or None), awake_seconds, workdir}. Returns metrics; simulation failures score 0."""
    from noesis.experiments import learning_lab as lab
    out = Path(task["workdir"])
    out.mkdir(parents=True, exist_ok=True)
    try:
        case = make_case(_STATE["data"], out, task["seed"], task["awake_seconds"], task["genome"])
        summary, _ = lab.run_episode(_STATE["cn"], case, 0, None)
        keep = ("status", "simulated_seconds", "standing_fraction", "mean_uprightness", "mean_stance_slip_mm_s",
                "heading_forward_distance_mm", "mean_supporting_feet", "mean_reward")
        res = {k: summary[k] for k in keep if k in summary}
        res["ok"] = summary.get("status") == "completed"
    except Exception as exc:  # noqa: BLE001  a failed simulation scores as a fall; it must not stop the evolution
        res = {"ok": False, "error": f"{type(exc).__name__}: {exc}", "standing_fraction": 0.0}
    finally:
        shutil.rmtree(out, ignore_errors=True)
    res["key"] = task["key"]
    res["standing_fraction"] = float(np.nan_to_num(res.get("standing_fraction", 0.0)))
    return res
