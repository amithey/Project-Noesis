"""One exploratory training birth with a selectable reward profile and exploration amplitude.

``train.py`` (Codex-owned) is fixed to curriculum-standing-v7 and 1.5 mV/ms. This runner leaves it
untouched and only swaps the objective passed to ``score_transition`` and the exploration amplitude
passed to ``LearningBrain`` for the duration of one ``train.run`` call. It never promotes weights.

Exit codes: 0 completed birth; 1 ``train.run`` raised; 2 rejected before running (existing output
directory or invalid option); 3 ``train.run`` returned a non-completed status (for example
``incomplete_wall_budget``). Existing output is never modified; for a new output directory the
effective objective is always written to ``profile-run.json`` and, only for a completed birth,
also annotated into ``report.json`` and the checkpoint's ``.model.json``.
"""

from __future__ import annotations

import functools
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

from . import learning, train
from .reward_profiles import resolve_objective

COMPLETED = "completed_training_birth"
SIDECAR = "profile-run.json"


@contextmanager
def _patched(objective, exploration_mv_ms):
    """Yield the CycleScoring in use (v9 only, else None); restore the learning module on exit."""
    score, brain = learning.score_transition, learning.LearningBrain
    scorer = (learning.CycleScoring(objective) if objective == "curriculum-gripped-cycle-v9"
              else None)  # built first: a core without the v9 tracker fails before anything is patched
    learning.score_transition = scorer or functools.partial(score, objective=objective)
    learning.LearningBrain = functools.partial(brain, exploration_mv_ms=exploration_mv_ms)
    try:
        yield scorer
    finally:
        learning.score_transition, learning.LearningBrain = score, brain


def _write_json(path: Path, data) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _annotate(output: Path, objective, exploration_mv_ms) -> None:
    for path in (output / "report.json", *output.glob("*.model.json")):
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        data["reward_objective_trainpy_label"] = data.get("reward_objective")
        data["reward_objective"] = objective
        data["motor_exploration_mv_ms"] = exploration_mv_ms
        _write_json(path, data)


def build_parser(train_module=train):
    ap = train_module.parser()
    ap.add_argument("--profile", default="gripped-steps")
    ap.add_argument("--exploration-mv-ms", type=float, default=learning.MAX_EXPLORATION_MV_MS)
    return ap


def run_birth(args, train_module=train) -> int:
    objective = resolve_objective(args.profile)
    if not 0 < args.exploration_mv_ms <= learning.MAX_EXPLORATION_MV_MS:
        print("--exploration-mv-ms must be in (0, 1.5]", file=sys.stderr)
        return 2
    output = Path(args.output)
    if output.exists():
        print(f"Refusing to run: output directory already exists: {output}", file=sys.stderr)
        return 2
    report, error, scorer = None, None, None
    try:
        with _patched(objective, args.exploration_mv_ms) as scorer:
            report = train_module.run(args)
    except BaseException as caught:  # noqa: BLE001 - recorded, then re-raised below
        error = caught
    status = "failed" if error is not None else (report or {}).get("status")
    if output.is_dir():  # created by this run
        if error is None and status == COMPLETED:
            _annotate(output, objective, args.exploration_mv_ms)
        sidecar = {
            "profile": args.profile, "reward_objective": objective,
            "motor_exploration_mv_ms": args.exploration_mv_ms, "seed": args.seed,
            "status": status, "checkpoint": str(args.checkpoint)}
        if scorer is not None:  # v9: how much cycle bonus was credited and how often the reward clipped
            sidecar["cycle_scoring"] = scorer.summary()
        _write_json(output / SIDECAR, sidecar)
    if error is not None:
        raise error
    return 0 if status == COMPLETED else 3


def main(argv=None, train_module=train) -> int:
    return run_birth(build_parser(train_module).parse_args(argv), train_module)


if __name__ == "__main__":
    raise SystemExit(main())
