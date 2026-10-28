from pathlib import Path

import pytest

from noesis_gpu.train import parser, run


def args(tmp_path, *extra):
    return parser().parse_args([
        "--noesis-source", str(tmp_path / "src"), "--data", str(tmp_path / "data"),
        "--checkpoint", str(tmp_path / "parent.npz"), "--output", str(tmp_path / "out"),
        *extra,
    ])


def test_native_defaults_and_explicit_profile(tmp_path):
    default = args(tmp_path)
    assert default.profile == "standing" and default.exploration_mv_ms == 1.5
    selected = args(tmp_path, "--profile", "continuous-walking", "--exploration-mv-ms", "1.0")
    assert selected.profile == "continuous-walking" and selected.exploration_mv_ms == 1.0


@pytest.mark.parametrize("value", ["0", "-1", "1.6", "nan", "inf"])
def test_invalid_exploration_rejected_before_output_creation(tmp_path, value):
    selected = args(tmp_path, "--exploration-mv-ms", value)
    with pytest.raises(ValueError, match="Exploration"):
        run(selected)
    assert not Path(selected.output).exists()


def test_legacy_wrapper_can_override_defaults_without_argument_conflict(tmp_path):
    from noesis_gpu.profile_run import build_parser

    wrapper = build_parser()
    selected = wrapper.parse_args([
        "--noesis-source", "src", "--data", "data", "--checkpoint", "parent.npz",
        "--output", str(tmp_path / "out"),
    ])
    assert selected.profile == "gripped-steps"


def test_native_v9_uses_fresh_tracker_and_old_objective_stays_stateless():
    import noesis.embodied_metrics as metrics
    if not hasattr(metrics, "LegCycleTracker"):
        pytest.skip("experimental v9 core required")
    from noesis_gpu.train import episode_scorer
    from noesis_gpu.learning import CycleScoring
    a = episode_scorer("curriculum-gripped-cycle-v9")
    b = episode_scorer("curriculum-gripped-cycle-v9")
    assert isinstance(a, CycleScoring) and isinstance(b, CycleScoring)
    assert a._tracker is not b._tracker
    assert not isinstance(episode_scorer("curriculum-gripped-step-v8"), CycleScoring)


def test_native_v9_rejects_core_without_tracker(monkeypatch):
    import noesis.embodied_metrics as metrics
    from noesis_gpu.train import episode_scorer
    monkeypatch.delattr(metrics, "LegCycleTracker", raising=False)
    with pytest.raises(ValueError, match="LegCycleTracker"):
        episode_scorer("curriculum-gripped-cycle-v9")
