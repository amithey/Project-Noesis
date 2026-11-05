import functools
import json
from types import SimpleNamespace

import pytest

from noesis_gpu import learning, train
from noesis_gpu.profile_run import COMPLETED, SIDECAR, main

LABEL = "curriculum-standing-v7"


def fake_train(status, seen=None, raises=None):
    def run(args):
        if seen is not None:
            seen.append((learning.score_transition, learning.LearningBrain))
        args.output.mkdir(parents=True)
        report = {"status": status, "reward_objective": LABEL, "seed": args.seed}
        (args.output / "report.json").write_text(json.dumps(report), encoding="utf-8")
        (args.output / "weights-last.model.json").write_text(
            json.dumps({"reward_objective": LABEL}), encoding="utf-8")
        if raises:
            raise raises
        return report
    return SimpleNamespace(parser=train.parser, run=run)


def argv(tmp_path, *extra, name="out"):
    return ["--noesis-source", str(tmp_path), "--data", str(tmp_path), "--checkpoint",
            str(tmp_path / "parent.npz"), "--output", str(tmp_path / name), *extra]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_completed_birth_exits_zero_and_records_the_real_objective(tmp_path):
    seen = []
    code = main(argv(tmp_path, "--exploration-mv-ms", ".75"), fake_train(COMPLETED, seen))
    assert code == 0
    patched_score, patched_brain = seen[0]
    assert patched_score.keywords["objective"] == "curriculum-gripped-step-v8"
    assert patched_brain.keywords["exploration_mv_ms"] == .75
    report = read(tmp_path / "out" / "report.json")
    assert report["reward_objective"] == "curriculum-gripped-step-v8"
    assert report["reward_objective_trainpy_label"] == LABEL
    assert report["motor_exploration_mv_ms"] == .75
    assert read(tmp_path / "out" / "weights-last.model.json")["reward_objective"] == report["reward_objective"]
    assert read(tmp_path / "out" / SIDECAR)["status"] == COMPLETED


@pytest.mark.parametrize("status", ["incomplete_wall_budget", "incomplete_interrupted", "failed", None])
def test_every_non_completed_birth_exits_nonzero_and_leaves_train_reports_alone(tmp_path, status):
    code = main(argv(tmp_path), fake_train(status))
    assert code == 3
    assert read(tmp_path / "out" / "report.json")["reward_objective"] == LABEL
    assert read(tmp_path / "out" / "weights-last.model.json")["reward_objective"] == LABEL
    assert read(tmp_path / "out" / SIDECAR)["status"] == status


def test_exception_propagates_and_reports_are_untouched(tmp_path):
    with pytest.raises(RuntimeError, match="boom"):
        main(argv(tmp_path), fake_train(COMPLETED, raises=RuntimeError("boom")))
    assert read(tmp_path / "out" / "report.json")["reward_objective"] == LABEL
    assert read(tmp_path / "out" / SIDECAR)["status"] == "failed"


def test_existing_output_is_rejected_without_running_or_touching_files(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    history = {"reward_objective": LABEL, "note": "historical"}
    (out / "report.json").write_text(json.dumps(history), encoding="utf-8")
    (out / "other.json").write_text(json.dumps(history), encoding="utf-8")
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    called = []

    def run(args):
        called.append(1)

    code = main(argv(tmp_path), SimpleNamespace(parser=train.parser, run=run))
    assert code == 2 and not called
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before
    assert not (out / SIDECAR).exists()


@pytest.mark.parametrize("bad", ["0", "-1", "1.51", "nan"])
def test_invalid_exploration_is_rejected_before_anything_is_created(tmp_path, bad):
    assert main(argv(tmp_path, "--exploration-mv-ms", bad), fake_train(COMPLETED)) == 2
    assert not (tmp_path / "out").exists()


def test_unknown_profile_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        main(argv(tmp_path, "--profile", "nope"), fake_train(COMPLETED))
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("raises", [None, RuntimeError("x")])
def test_learning_module_is_restored_even_after_a_failure(tmp_path, raises):
    score, brain = learning.score_transition, learning.LearningBrain
    try:
        main(argv(tmp_path), fake_train(COMPLETED, raises=raises))
    except RuntimeError:
        pass
    assert learning.score_transition is score and learning.LearningBrain is brain
    assert not isinstance(learning.score_transition, functools.partial)
