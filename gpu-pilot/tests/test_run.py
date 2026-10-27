import json
from importlib.metadata import PackageNotFoundError, version

import numpy as np
import pytest

from noesis_gpu.run import parser, run, timing_steps


@pytest.mark.parametrize("values", [(0, 5, 1e-4), (0.101, 5, 1e-4),
                                         (1, 5.1, 1e-4), (np.nan, 5, 1e-4)])
def test_no_silent_time_rounding(values):
    with pytest.raises(ValueError):
        timing_steps(*values)


def test_budget_counts_full_control_chunks_only(tmp_path, monkeypatch):
    import noesis_gpu.run as module

    class Body:
        n_worlds = 1
        rest = np.zeros(42)
        lo = np.full(42, -0.7)
        hi = np.full(42, 0.7)
        device = "fake-test-device"

        def __init__(self, *args):
            self.steps = 0

        def observe(self):
            return {"position": np.zeros((1, 3)), "angles": np.zeros((1, 42))}

        def advance(self, commands, adhesion, steps):
            self.steps += steps

    monkeypatch.setattr(module, "FlyGymBatch", Body)
    times = iter([0, 0, 1, 1, 1])
    monkeypatch.setattr(module.time, "perf_counter", lambda: next(times))
    args = parser().parse_args(["--controller", "neutral", "--wall-seconds", "0.5",
                                "--output", str(tmp_path / "run")])
    report = run(args)
    assert report["status"] == "incomplete_wall_budget"
    assert report["simulated_seconds_per_world"] == 0
    assert not (args.output / "observations.npz").exists()


def test_missing_gpu_records_failure_without_falling_back(tmp_path, monkeypatch):
    import noesis_gpu.run as module

    def missing(*args):
        raise RuntimeError("NVIDIA CUDA GPU unavailable")

    monkeypatch.setattr(module, "FlyGymBatch", missing)
    args = parser().parse_args(["--controller", "neutral", "--output", str(tmp_path / "run")])
    with pytest.raises(RuntimeError):
        run(args)
    report = json.loads((args.output / "report.json").read_text())
    assert report["status"] == "failed"
    assert report["backend"] == "gpu"
    assert report["completed_control_chunks"] == 0
    assert report["model_promoted"] is False


def test_actual_upstream_cpu_body_observations_and_commands():
    try:
        installed = version("flygym")
    except PackageNotFoundError:
        installed = None
    if installed != "2.1.0":
        pytest.skip("Requires isolated FlyGym 2.1 runtime")
    from noesis_gpu.backend import FlyGymBatch

    body = FlyGymBatch(1, "cpu")
    obs = body.observe()
    assert obs["angles"].shape == (1, 42)
    assert obs["ground_contact"].shape == (1, 6)
    assert obs["foot_load"].shape == (1, 6)
    assert obs["feet_position"].shape == (1, 6, 3)
    np.testing.assert_allclose(obs["angles"][0], body.rest, atol=1e-10)
    command = body.rest.copy()
    command[5] += 0.1
    body.advance(command[None], np.ones((1, 6)), 20)
    after = body.observe()
    assert all(np.isfinite(value).all() for value in after.values())
    assert after["angles"][0, 5] > obs["angles"][0, 5]


def test_actual_cuda_worlds_have_independent_body_states():
    try:
        installed = version("flygym")
    except PackageNotFoundError:
        installed = None
    if installed != "2.1.0":
        pytest.skip("Requires isolated FlyGym 2.1 runtime")
    wp = pytest.importorskip("warp", reason="Requires the GPU extra and NVIDIA hardware")

    if not wp.get_cuda_devices():
        pytest.skip("Real NVIDIA CUDA hardware required; CPU is not GPU verification")
    from noesis_gpu.backend import FlyGymBatch

    body = FlyGymBatch(2, "gpu")
    initial = body.observe()
    np.testing.assert_allclose(initial["angles"][0], initial["angles"][1], atol=1e-6)
    commands = np.tile(body.rest, (2, 1))
    commands[0, 5] += 0.1
    body.advance(commands, np.ones((2, 6)), 20)
    after = body.observe()
    assert abs(after["angles"][0, 5] - after["angles"][1, 5]) > 1e-8
    assert after["load"].shape == (2, 6)
