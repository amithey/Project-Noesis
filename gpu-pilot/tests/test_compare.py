import json

import numpy as np
import pytest

from noesis_gpu.compare import compare


def make_pair(tmp_path):
    for backend in ("cpu", "gpu"):
        directory = tmp_path / backend
        directory.mkdir()
        (directory / "report.json").write_text(json.dumps({
            "backend": backend, "status": "completed_transport_pilot", "controller": "neutral",
            "worlds": 1, "learning": False, "motor_exploration": 0,
            "seeds": [10], "completed_control_chunks": 2,
            "upstream_commit": "test", "requested_simulated_seconds_per_world": 0.01,
            "control_ms": 5, "physics_dt": 0.0001, "bg_hz": 3, "bg_weight": 40,
            "senses": ["joint_position"], "rng_schedule": "test", "body": {},
            "packages": {"flygym": "2.1.0", "mujoco": "3.9.0"},
        }))
        np.savez(directory / "observations.npz", angles=np.zeros((2, 1, 42)))
    return tmp_path / "cpu", tmp_path / "gpu"


def test_comparison_reports_measured_difference(tmp_path):
    cpu, gpu = make_pair(tmp_path)
    values = np.zeros((2, 1, 42))
    values[0, 0, 5] = 0.2
    np.savez(gpu / "observations.npz", angles=values)
    result = compare(cpu, gpu)
    assert result["metrics"]["angles"]["max_absolute_difference"] == 0.2
    assert result["walking_proven"] is False


@pytest.mark.parametrize("key,value", [("status", "incomplete_wall_budget"),
                                       ("seeds", [11]), ("worlds", 100)])
def test_comparison_rejects_incomplete_or_unmatched_controls(tmp_path, key, value):
    cpu, gpu = make_pair(tmp_path)
    path = gpu / "report.json"
    report = json.loads(path.read_text())
    report[key] = value
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        compare(cpu, gpu)
