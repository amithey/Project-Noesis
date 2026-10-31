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
from types import SimpleNamespace

import numpy as np

from . import learning, train
from .reward_profiles import resolve_objective

COMPLETED = "completed_training_birth"
SIDECAR = "profile-run.json"


@contextmanager
def _patched(objective, exploration_mv_ms, cycle_weight=None, learning_rate=None, record_exploration=False,
             credit_control=None):
    """Yield (scorer, brains): the CycleScoring in use (v9 only, else None) and every LearningBrain built
    during the run; restore the learning module on exit."""
    score, brain = learning.score_transition, learning.LearningBrain
    scorer = (learning.CycleScoring(objective, cycle_weight=cycle_weight) if objective == "curriculum-gripped-cycle-v9"
              else None)  # built first: a core without the v9 tracker fails before anything is patched
    learning.score_transition = scorer or functools.partial(score, objective=objective)
    brains, options = [], {"exploration_mv_ms": exploration_mv_ms}
    if learning_rate is not None:
        options["learning_rate"] = learning_rate
    if record_exploration:
        options["record_exploration"] = True
    if credit_control is not None:
        options["credit_control"] = credit_control

    def make_brain(*a, **k):  # same override order as functools.partial: the call's keywords win
        built = brain(*a, **{**options, **k})
        brains.append(built)
        return built

    make_brain.func, make_brain.keywords = brain, options  # introspectable like the functools.partial it replaces
    learning.LearningBrain = make_brain
    try:
        yield SimpleNamespace(scorer=scorer, brains=brains)
    finally:
        learning.score_transition, learning.LearningBrain = score, brain


def _write_json(path: Path, data) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _annotate(output: Path, objective, exploration_mv_ms, cycle_weight=None, learning_rate=None) -> None:
    for path in (output / "report.json", *output.glob("*.model.json")):
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        data["reward_objective_trainpy_label"] = data.get("reward_objective")
        data["reward_objective"] = objective
        data["motor_exploration_mv_ms"] = exploration_mv_ms
        if cycle_weight is not None:
            data["cycle_weight"] = cycle_weight
        if learning_rate is not None:
            data["learning_rate"] = learning_rate
        _write_json(path, data)


def build_parser(train_module=train):
    ap = train_module.parser()
    ap.add_argument("--profile", default="gripped-steps")
    ap.add_argument("--exploration-mv-ms", type=float, default=learning.MAX_EXPLORATION_MV_MS)
    ap.add_argument("--cycle-weight", type=float, default=None,
                    help="v9 only: weight of the cycle bonus in (0, 1]; default is the core's 0.1")
    ap.add_argument("--learning-rate", type=float, default=None,
                    help=f"weight-update rate per second in (0, {learning.MAX_LEARNING_RATE}]; default "
                         f"{learning.DEFAULT_LEARNING_RATE}")
    ap.add_argument("--credit-control", choices=("independent",), default=None,
                    help="matched noise control: the learner receives credit from an independent exploration "
                         "stream; injected current and reward are unchanged")
    ap.add_argument("--log-exploration", action="store_true",
                    help="save the injected exploration current and innovation credit to exploration-log.npz")
    return ap


def run_birth(args, train_module=train) -> int:
    objective = resolve_objective(args.profile)
    if not 0 < args.exploration_mv_ms <= learning.MAX_EXPLORATION_MV_MS:
        print("--exploration-mv-ms must be in (0, 1.5]", file=sys.stderr)
        return 2
    if args.cycle_weight is not None:
        if objective != "curriculum-gripped-cycle-v9":
            print("--cycle-weight is only valid with the gripped-cycle profile", file=sys.stderr)
            return 2
        if not 0 < args.cycle_weight <= 1:
            print("--cycle-weight must be in (0, 1]", file=sys.stderr)
            return 2
    if args.learning_rate is not None and not 0 < args.learning_rate <= learning.MAX_LEARNING_RATE:
        print(f"--learning-rate must be in (0, {learning.MAX_LEARNING_RATE}]", file=sys.stderr)
        return 2
    output = Path(args.output)
    if output.exists():
        print(f"Refusing to run: output directory already exists: {output}", file=sys.stderr)
        return 2
    report, error, scorer, brains = None, None, None, []
    try:
        with _patched(objective, args.exploration_mv_ms, args.cycle_weight, args.learning_rate,
                      args.log_exploration, args.credit_control) as context:
            scorer, brains = context.scorer, context.brains
            report = train_module.run(args)
    except BaseException as caught:  # noqa: BLE001 - recorded, then re-raised below
        error = caught
    status = "failed" if error is not None else (report or {}).get("status")
    if output.is_dir():  # created by this run
        if error is None and status == COMPLETED:
            _annotate(output, objective, args.exploration_mv_ms, args.cycle_weight, args.learning_rate)
        sidecar = {
            "profile": args.profile, "reward_objective": objective,
            "motor_exploration_mv_ms": args.exploration_mv_ms, "seed": args.seed,
            "status": status, "checkpoint": str(args.checkpoint)}
        if args.cycle_weight is not None:
            sidecar["cycle_weight"] = args.cycle_weight
        if args.learning_rate is not None:
            sidecar["learning_rate"] = args.learning_rate
        if args.credit_control is not None:
            sidecar["credit_control"] = args.credit_control
        if scorer is not None:  # v9: how much cycle bonus was credited and how often the reward clipped
            sidecar["cycle_scoring"] = scorer.summary()
        if args.log_exploration:
            logs = [b.exploration_log() for b in brains if b.exploration_log() is not None]
            if logs and len(logs[-1]["credit"]):
                # Native train exports explicit observation-row alignment; retain it.
                if not (output / "exploration-log.npz").exists():
                    temporary = output / "exploration-log.tmp.npz"
                    np.savez_compressed(temporary, **logs[-1])
                    os.replace(temporary, output / "exploration-log.npz")
                sidecar["exploration_log"] = {"file": "exploration-log.npz",
                                              "steps": len(logs[-1]["credit"])}
        _write_json(output / SIDECAR, sidecar)
    if error is not None:
        raise error
    return 0 if status == COMPLETED else 3


def main(argv=None, train_module=train) -> int:
    return run_birth(build_parser(train_module).parse_args(argv), train_module)


if __name__ == "__main__":
    raise SystemExit(main())
