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
    assert default.learning_rate == .2 and default.log_exploration is False
    selected = args(tmp_path, "--profile", "continuous-walking", "--exploration-mv-ms", "1.0")
    assert selected.profile == "continuous-walking" and selected.exploration_mv_ms == 1.0


@pytest.mark.parametrize("value", ["0", "-1", "2.1", "nan", "inf"])
def test_invalid_native_rate_rejected_before_output(tmp_path, value):
    selected = args(tmp_path, "--learning-rate", value)
    with pytest.raises(ValueError, match="Learning rate"):
        run(selected)
    assert not selected.output.exists()


def test_native_log_matches_reward_rows_without_changing_weights(tmp_path, monkeypatch):
    """Real small learning brain, synthetic fixed body; not a physical success test."""
    import json
    import numpy as np
    import noesis
    import noesis.embodiment
    from noesis_gpu import train
    from test_weight_integrity import build
    from test_learning import state

    cn, parent = build(tmp_path)
    class FixedBody:
        rest = np.zeros(42)
        lo, hi = np.full(42, -.7), np.full(42, .7)
        def __init__(self, *a): pass
        def observe(self):
            return {k: np.asarray(v)[None] for k, v in state().items()}
        def advance(self, *a): pass
    monkeypatch.setattr(train, 'FlyGymBatch', FixedBody)
    monkeypatch.setattr(noesis.embodiment, 'load_training_brain', lambda *a:(cn, None))
    for logged in (False, True):
        selected = args(tmp_path, '--seconds', '.1')
        selected.noesis_source = Path(noesis.__file__).resolve().parents[1]
        selected.checkpoint = parent
        selected.output = tmp_path/('logged' if logged else 'plain')
        selected.log_exploration = logged
        report = run(selected)
        assert report['status'] == 'completed_training_birth'
        if logged:
            with np.load(selected.output/'exploration-log.npz', allow_pickle=False) as log:
                np.testing.assert_array_equal(log['control_chunk_index'], np.arange(10, 20))
                assert log['drive'].shape[0] == log['credit'].shape[0] == 10
            assert report['exploration_log']['first_control_chunk'] == 10
            saved = json.loads((selected.output/'report.json').read_text())
            assert saved['exploration_log']['steps'] == 10
    with np.load(tmp_path/'plain/weights-last.npz') as a, np.load(tmp_path/'logged/weights-last.npz') as b:
        for key in a.files: np.testing.assert_array_equal(a[key], b[key])
    with np.load(tmp_path/'plain/training-observations.npz') as a, np.load(tmp_path/'logged/training-observations.npz') as b:
        for key in a.files: np.testing.assert_array_equal(a[key], b[key])


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
