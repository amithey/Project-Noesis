import json

import pytest
from test_cycle_scoring import V8, V9, argv, episode, fake_train, needs_v9, obs

from noesis_gpu.learning import CycleScoring, score_transition
from noesis_gpu.profile_run import SIDECAR, main


@needs_v9
def test_default_weight_is_the_core_default_and_matches_an_explicit_one():
    implicit, explicit = CycleScoring(), CycleScoring(cycle_weight=.1)
    assert implicit.summary()["cycle_weight"] == .1
    a, b = episode(implicit, 700), episode(explicit, 700)
    assert [r["reward"] for r in a] == [r["reward"] for r in b]


@needs_v9
def test_larger_weight_pays_proportionally_more_bonus_on_the_same_trajectory():
    low, high = CycleScoring(), CycleScoring(cycle_weight=.5)
    episode(low, 900, speed_mm_s=1.)
    results = episode(high, 900, speed_mm_s=1.)
    ratio = (high.summary()["intended_bonus_sum_before_clipping"]
             / low.summary()["intended_bonus_sum_before_clipping"])
    assert ratio == pytest.approx(5.)
    assert all(r["cycle_weight"] == .5 for r in results)
    assert high.summary()["cycle_weight"] == .5
    assert 0 < high.summary()["actual_bonus_sum_after_clipping"] <= high.summary()["intended_bonus_sum_before_clipping"] + 1e-9


@needs_v9
def test_weight_changes_only_the_bonus_not_the_credit_schedule():
    low, high = CycleScoring(), CycleScoring(cycle_weight=.9)
    a, b = episode(low, 900), episode(high, 900)
    assert [r["cycle_credit"] for r in a] == [r["cycle_credit"] for r in b]
    assert all(y["reward"] >= x["reward"] for x, y in zip(a, b))


@pytest.mark.parametrize("bad", [0, -.1, 1.5, float("nan"), float("inf")])
def test_invalid_weight_is_rejected(bad):
    with pytest.raises(ValueError):
        CycleScoring(cycle_weight=bad)


@needs_v9
def test_score_transition_accepts_the_weight_only_for_v9_in_the_core():
    a, b = obs(), obs(.1)
    assert score_transition(a, b, 5., V9, cycle_credit=.5, cycle_weight=.4)["cycle_weight"] == .4
    with pytest.raises(ValueError):
        score_transition(a, b, 5., V8, cycle_weight=.4)
    assert "cycle_weight" not in score_transition(a, b, 5., V8)  # unchanged without the option


@needs_v9
def test_profile_run_records_the_weight_in_the_sidecar_and_report(tmp_path):
    seen = []
    assert main(argv(tmp_path, "--profile", "gripped-cycle", "--cycle-weight", "0.5"), fake_train(seen=seen)) == 0
    assert seen[0].summary()["cycle_weight"] == .5
    sidecar = json.loads((tmp_path / "out" / SIDECAR).read_text(encoding="utf-8"))
    assert sidecar["cycle_weight"] == .5 and sidecar["cycle_scoring"]["cycle_weight"] == .5
    assert json.loads((tmp_path / "out" / "report.json").read_text(encoding="utf-8"))["cycle_weight"] == .5


@needs_v9
def test_profile_run_without_the_option_keeps_the_default_and_writes_no_weight_field(tmp_path):
    assert main(argv(tmp_path, "--profile", "gripped-cycle"), fake_train()) == 0
    sidecar = json.loads((tmp_path / "out" / SIDECAR).read_text(encoding="utf-8"))
    assert "cycle_weight" not in sidecar and sidecar["cycle_scoring"]["cycle_weight"] == .1


@pytest.mark.parametrize("profile, weight", [("gripped-steps", "0.5"), ("gripped-cycle", "0"), ("gripped-cycle", "1.5")])
def test_profile_run_refuses_a_misplaced_or_out_of_range_weight_before_creating_output(tmp_path, profile, weight):
    assert main(argv(tmp_path, "--profile", profile, "--cycle-weight", weight), fake_train()) == 2
    assert not (tmp_path / "out").exists()
