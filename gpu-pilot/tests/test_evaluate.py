import json

import numpy as np
import pytest

from noesis_gpu.evaluate import compare, standing_gate, summarize


def fixture(directory, *, standing=True):
    directory.mkdir()
    report = {"status": "completed_transport_pilot", "controller": "brain", "worlds": 1,
              "learning": False, "motor_exploration": 0, "weights_unchanged": True,
              "control_ms": 5., "completed_control_chunks": 1000,
              "requested_simulated_seconds_per_world": 5., "simulated_seconds_per_world": 5.,
              "backend": "cpu", "physics_dt": .0001, "bg_hz": 3., "bg_weight": 40.,
              "upstream_commit": "pin", "packages": {"flygym": "2.1"}, "body": {"kp": 45},
              "noesis_source_sha256": "source", "senses": ["joint", "balance"],
              "external_vision_and_chemical_senses": False, "rng_schedule": "per-chunk",
              "seeds": [10007], "checkpoint_sha256": "bank"}
    (directory / "report.json").write_text(json.dumps(report), encoding="utf-8")
    position = np.zeros((1000, 1, 3))
    position[..., 2] = 1
    up = position.copy()
    feet = np.zeros((1000, 1, 6, 3))
    feet[..., 0] = np.arange(1000)[:, None, None] * .005 * 2  # exactly 2 mm/s sliding
    loads = np.ones((1000, 1, 6)) if standing else np.zeros((1000, 1, 6))
    loads[:10] = 0  # warmup excluded from gate, separately reported
    np.savez(directory / "observations.npz", position=position, up_body=up, heading=up,
             feet_position=feet, foot_load=loads, load=np.ones_like(loads))
    return report


def test_actual_claw_support_and_warmup_separation(tmp_path):
    fixture(tmp_path / "supported")
    _, result = summarize(tmp_path / "supported")
    assert result["warmup_standing_fraction"] == 0
    assert result["standing_fraction"] == 1
    assert result["measured_samples"] == 990
    assert result["mean_stance_slip_mm_s"] == pytest.approx(2)
    fixture(tmp_path / "body-contact-only", standing=False)
    _, result = summarize(tmp_path / "body-contact-only")
    assert result["standing_fraction"] == 0


@pytest.mark.parametrize("key,value", [("status", "incomplete_wall_budget"),
    ("completed_control_chunks", 999), ("weights_unchanged", False),
    ("learning", True), ("motor_exploration", 1.5)])
def test_reject_incomplete_or_active_evaluation(tmp_path, key, value):
    report = fixture(tmp_path / "case")
    report[key] = value
    (tmp_path / "case/report.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError):
        summarize(tmp_path / "case")


def test_comparison_rejects_changed_body_or_seed(tmp_path):
    fixture(tmp_path / "a")
    report = fixture(tmp_path / "b")
    assert compare(tmp_path / "a", tmp_path / "b")["same_conditions_verified"]
    for key, changed in (("body", {"kp": 90}), ("seeds", [10008])):
        other = dict(report, **{key: changed})
        (tmp_path / "b/report.json").write_text(json.dumps(other), encoding="utf-8")
        with pytest.raises(ValueError):
            compare(tmp_path / "a", tmp_path / "b")


def test_standing_gate_requires_distinct_training_seeds_same_candidate(tmp_path):
    pairs = [(tmp_path / f"parent-{i}", tmp_path / f"candidate-{i}") for i in range(2)]
    for pair in pairs:
        for directory in pair:
            fixture(directory)
    with pytest.raises(ValueError, match="distinct training seeds"):
        standing_gate(pairs)
    for directory in pairs[1]:
        path = directory / "report.json"
        report = json.loads(path.read_text())
        report["seeds"] = [10008]
        path.write_text(json.dumps(report))
    gate = standing_gate(pairs)
    assert gate["standing_training_gate_passed"]
    assert not gate["stage_advanced"] and not gate["model_promoted"]
    path = pairs[1][1] / "report.json"
    report = json.loads(path.read_text())
    report["checkpoint_sha256"] = "different-candidate"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="Same parent and same candidate"):
        standing_gate(pairs)


def test_summarize_includes_versioned_gait_metrics(tmp_path):
    fixture(tmp_path / "case")
    _, result = summarize(tmp_path / "case")
    assert result["schema_version"] == "gait-metrics-v1.0"
    assert "heading_forward_distance_mm" in result
    assert "leg_step_cycles" in result
    assert "locomotion_regime" in result
    assert result["locomotion_regime"] in ("STATIC_STANDING", "DRAGGING_OR_SLIPPING", "COORDINATED_TRIPOD_WALKING")
    for leg in ("LF", "LM", "LH", "RF", "RM", "RH"):
        assert leg in result["leg_step_cycles"]
        assert "step_count" in result["leg_step_cycles"][leg]


def test_compare_includes_gait_differences(tmp_path):
    fixture(tmp_path / "parent")
    fixture(tmp_path / "cand")
    comp = compare(tmp_path / "parent", tmp_path / "cand")
    assert comp["schema_version"] == "gait-metrics-v1.0"
    assert "heading_forward_change_mm" in comp
    assert "total_steps_change" in comp
    assert "locomotion_regime_before" in comp
    assert "locomotion_regime_after" in comp
