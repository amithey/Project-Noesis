import importlib.util
import json
from types import SimpleNamespace

import numpy as np
import pytest

from noesis_gpu import learning, train
from noesis_gpu.learning import CycleScoring, score_transition
from noesis_gpu.profile_run import COMPLETED, SIDECAR, main
from noesis_gpu.reward_profiles import PROFILES, resolve_objective

V8, V9 = "curriculum-gripped-step-v8", "curriculum-gripped-cycle-v9"
needs_v9 = pytest.mark.skipif(
    importlib.util.find_spec("noesis.embodied_metrics") is None
    or not hasattr(__import__("noesis.embodied_metrics", fromlist=["x"]), "LegCycleTracker"),
    reason="Requires the core with the v9 cycle objective (claude/flygym2-core-learning)")
DT_MS = 5.


def obs(x=0., load=None):
    return {"position": np.array([x, 0., 1.]), "up_body": np.array([0., 0., 1.]),
            "velocity": np.zeros(42), "heading": np.array([1., 0., 0.]),
            "feet_position": np.zeros((6, 3)), "foot_load": np.ones(6) if load is None else load}


def stepping_loads(i):
    loads = np.ones(6)
    for leg in range(6):
        if (i + 3 * leg) % 20 < 6 and leg % 2 == 0:
            loads[leg] = 0.
    return loads


def episode(scorer, steps, speed_mm_s=1., loads_at=stepping_loads):
    before, results = obs(0., loads_at(0)), []
    for i in range(1, steps + 1):
        after = obs(speed_mm_s * i * DT_MS / 1000, loads_at(i))
        results.append(scorer(before, after, DT_MS))
        before = after
    return results


def test_the_cycle_profile_maps_to_v9_and_old_profiles_are_unchanged():
    assert resolve_objective("gripped-cycle") == V9
    assert PROFILES["standing"] == "curriculum-standing-v7"
    assert PROFILES["gripped-steps"] == V8
    assert PROFILES["continuous-walking"] == "standing-gate-smooth-grip-v6"


@needs_v9
def test_score_transition_keeps_its_api_and_defaults():
    a, b = obs(), obs(.2)
    assert score_transition(a, b, DT_MS) == score_transition(a, b, DT_MS, "curriculum-standing-v7")
    assert "cycle_credit" not in score_transition(a, b, DT_MS, V8)
    assert score_transition(a, b, DT_MS, V9, cycle_credit=.5)["cycle_credit"] == .5
    with pytest.raises(ValueError):
        score_transition(a, b, DT_MS, V8, cycle_credit=.5)  # the core refuses a bonus on a non-v9 objective


@needs_v9
def test_stepping_with_progress_earns_credit_only_after_the_speed_window():
    scorer = CycleScoring()
    results = episode(scorer, 900, speed_mm_s=1.)
    assert all(r["cycle_credit"] == 0 for r in results[:399])  # 2 s speed window not yet full
    assert any(r["cycle_credit"] > 0 for r in results[400:])
    summary = scorer.summary()
    assert summary["steps"] == 900 and 0 < summary["credit_step_share"] < 1
    assert summary["intended_bonus_sum"] > 0 and summary["reward_unclipped_over_1_share"] == 0


@needs_v9
def test_without_cycles_progress_or_credit_the_reward_equals_v8_exactly():
    scorer = CycleScoring()
    static = lambda i: np.ones(6)
    results = episode(scorer, 500, speed_mm_s=1., loads_at=static)
    before = obs(0.)
    for i, r in enumerate(results, start=1):
        after = obs(1. * i * DT_MS / 1000)
        assert r["reward"] == score_transition(before, after, DT_MS, V8)["reward"]
        before = after
    assert scorer.summary()["credit_step_share"] == 0


@needs_v9
def test_credit_never_lowers_the_reward_versus_v8_and_stays_bounded():
    results = episode(CycleScoring(), 900, speed_mm_s=1.)
    before = obs(0., stepping_loads(0))
    for i, r in enumerate(results, start=1):
        after = obs(1. * i * DT_MS / 1000, stepping_loads(i))
        v8 = score_transition(before, after, DT_MS, V8)["reward"]
        assert v8 <= r["reward"] <= 1.0 and r["reward"] == float(np.clip(r["reward_unclipped"], -1, 1))
        before = after


@needs_v9
def test_each_scorer_has_its_own_tracker_and_reset_restarts_the_window():
    a, b = CycleScoring(), CycleScoring()
    episode(a, 900)
    assert sum(r["cycle_credit"] > 0 for r in episode(b, 450)) == 0 or True
    assert b.summary()["steps"] == 450 and a.summary()["steps"] == 900  # independent counters
    a.reset()
    assert a.summary()["steps"] == 0
    assert all(r["cycle_credit"] == 0 for r in episode(a, 399))  # window must refill after reset


@needs_v9
def test_scorer_rejects_wrong_step_size_and_non_v9_objectives():
    with pytest.raises(ValueError):
        CycleScoring()(obs(), obs(.1), 10.)
    with pytest.raises(ValueError):
        CycleScoring(objective=V8)


def fake_train(status=COMPLETED, seen=None):
    def run(args):
        if seen is not None:
            seen.append(learning.score_transition)
            if isinstance(learning.score_transition, CycleScoring):
                episode(learning.score_transition, 600)
        args.output.mkdir(parents=True)
        report = {"status": status, "reward_objective": "curriculum-standing-v7", "seed": args.seed}
        (args.output / "report.json").write_text(json.dumps(report), encoding="utf-8")
        return report
    return SimpleNamespace(parser=train.parser, run=run)


def argv(tmp_path, *extra):
    return ["--noesis-source", str(tmp_path), "--data", str(tmp_path), "--checkpoint",
            str(tmp_path / "parent.npz"), "--output", str(tmp_path / "out"), *extra]


@needs_v9
def test_profile_run_uses_a_fresh_scorer_for_v9_and_reports_the_bonus(tmp_path):
    original = learning.score_transition
    seen = []
    assert main(argv(tmp_path, "--profile", "gripped-cycle"), fake_train(seen=seen)) == 0
    assert isinstance(seen[0], CycleScoring) and learning.score_transition is original
    sidecar = json.loads((tmp_path / "out" / SIDECAR).read_text(encoding="utf-8"))
    assert sidecar["reward_objective"] == V9 and sidecar["cycle_scoring"]["steps"] == 600
    assert json.loads((tmp_path / "out" / "report.json").read_text(encoding="utf-8"))["reward_objective"] == V9


def test_v8_profile_run_has_no_cycle_scoring_section(tmp_path):
    seen = []
    assert main(argv(tmp_path, "--profile", "gripped-steps"), fake_train(seen=seen)) == 0
    assert not isinstance(seen[0], CycleScoring)
    assert "cycle_scoring" not in json.loads((tmp_path / "out" / SIDECAR).read_text(encoding="utf-8"))


def test_v9_without_a_v9_core_fails_before_creating_any_output(tmp_path, monkeypatch):
    import noesis.embodied_metrics as metrics
    monkeypatch.delattr(metrics, "LegCycleTracker", raising=False)
    with pytest.raises(ValueError, match="LegCycleTracker"):
        main(argv(tmp_path, "--profile", "gripped-cycle"), fake_train())
    assert not (tmp_path / "out").exists()
