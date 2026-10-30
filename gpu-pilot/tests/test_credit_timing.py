import numpy as np
import pandas as pd
import pytest
from scipy.signal import lfilter

from noesis_gpu import credit_timing as ct

T, M_PER_LEG, TAU = 990, 10, 20  # steps per birth, explored neurons per leg, OU time constant in steps


def ou(rng, n, columns, tau=TAU):
    decay = np.exp(-1 / tau)
    return np.tanh(lfilter([np.sqrt(1 - decay ** 2)], [1, -decay], rng.normal(size=(n, columns)), axis=0))


def birth(rng, effect_leg=None, lag=8, kappa=0.0, noise=0.5):
    """Injected current for 6 legs x M neurons with a shared per-leg component, plus responses."""
    latent = ou(rng, T, 6)
    drive = np.concatenate([latent[:, [leg]] * rng.uniform(.5, 1.5, M_PER_LEG) + 0.3 * ou(rng, T, M_PER_LEG)
                            for leg in range(6)], axis=1)
    legs = np.repeat(np.arange(6), M_PER_LEG)
    joint = {leg: noise * ou(rng, T, 1)[:, 0] + noise * rng.normal(size=T) for leg in range(6)}
    reward = 0.3 * rng.normal(size=T) + 0.2 * ou(rng, T, 1, tau=400)[:, 0]
    if effect_leg is not None:
        shifted = np.zeros(T)
        shifted[lag:] = latent[:-lag if lag else None, effect_leg][: T - lag]
        joint[effect_leg] = joint[effect_leg] + kappa * shifted
        reward = reward + kappa * 0.5 * shifted
    return drive, legs, joint, reward


def prepare(births):
    features = {leg: [ct.leg_features(d, l, leg) for d, l, _, _ in births] for leg in range(6)}
    joints = {leg: [j[leg] for _, _, j, _ in births] for leg in range(6)}
    reward = {leg: [r for _, _, _, r in births] for leg in range(6)}
    return features, joints, reward


def test_assign_legs_uses_side_and_neuromere_and_marks_ambiguous_neurons():
    neurons = pd.DataFrame({
        "Soma side": ["left", "right", "left", "right", "left", "right"],
        "Top in/out region": ["T1_PRONM", "T2_MESONM", "T3_METANM", "T3_METANM", "T2_MESONM.T3_METANM", "ABDNM"]})
    legs = ct.assign_legs(neurons, np.arange(6))
    assert list(legs) == [0, 4, 2, 5, -1, -1]  # LF, RM, LH, RH, ambiguous, other


def test_reward_residual_removes_a_slow_baseline_and_keeps_fast_changes():
    reward = np.concatenate([np.zeros(400), np.ones(400)])
    residual = ct.reward_residual(reward)
    assert residual[:400].max() == 0 and residual[400] == pytest.approx(1.0)
    assert residual[-1] == pytest.approx(np.exp(-399 * 5 / 2000), rel=.02)  # 2 s after the jump the baseline has closed 63%


def test_a_planted_effect_is_recovered_at_the_right_leg_and_lag():
    rng = np.random.default_rng(1)
    births = [birth(rng, effect_leg=2, lag=8, kappa=1.0) for _ in range(6)]
    features, joints, _ = prepare(births)
    result, null = ct.analyse(features, joints, max_lag=40, permutations=100, seed=3)
    assert result[2]["family_p"] < .02 and abs(result[2]["best_lag_steps"] - 8) <= 1
    assert result[2]["r2"] > null["null_max_p99"]
    assert sum(result[leg]["family_p"] < .05 for leg in range(6) if leg != 2) <= 1  # other legs stay quiet


def test_without_any_effect_false_positives_stay_near_the_nominal_rate():
    hits = 0
    for seed in range(12):
        rng = np.random.default_rng(100 + seed)
        births = [birth(rng) for _ in range(5)]
        features, joints, _ = prepare(births)
        result, _ = ct.analyse(features, joints, max_lag=30, permutations=60, seed=seed)
        hits += min(r["family_p"] for r in result.values()) < .05
    assert hits <= 3  # 12 null datasets at a 5% family-wise level; allow generous slack


def test_an_effect_on_the_reward_is_found_at_the_right_leg_and_lag_and_a_wrong_leg_is_not():
    rng = np.random.default_rng(7)
    births = [birth(rng, effect_leg=0, lag=10, kappa=6.0, noise=0.5) for _ in range(10)]
    features, _, reward = prepare(births)
    result, _ = ct.analyse(features, reward, max_lag=40, permutations=100, seed=1)
    assert result[0]["family_p"] < .05 and result[0]["best_lag_steps"] == 10
    assert all(result[leg]["family_p"] > .05 for leg in range(1, 6))


def test_a_small_effect_needs_many_births_to_stand_out_from_the_null():
    """Documents the method's power: R^2 of a few percent is only borderline even with ~9000 samples."""
    rng = np.random.default_rng(7)
    births = [birth(rng, effect_leg=0, lag=10, kappa=2.5) for _ in range(10)]
    features, _, reward = prepare(births)
    result, null = ct.analyse(features, reward, max_lag=40, permutations=100, seed=1)
    assert result[0]["best_lag_steps"] == 10  # the lag is still the best-fitting one
    assert result[0]["r2"] < .1 and null["null_max_p95"] > .01  # but R^2 stays small and the null is not far below it


def test_run_reads_the_log_and_observation_files_and_checks_alignment(tmp_path):
    rng = np.random.default_rng(5)
    neurons = pd.DataFrame({
        "Soma side": ["left"] * (M_PER_LEG * 3) + ["right"] * (M_PER_LEG * 3),
        "Top in/out region": (["T1_PRONM"] * M_PER_LEG + ["T2_MESONM"] * M_PER_LEG + ["T3_METANM"] * M_PER_LEG) * 2})
    dirs = []
    for b in range(4):
        drive, _, joint, reward = birth(rng, effect_leg=1, lag=6, kappa=1.2)
        steps = len(drive)
        velocity = np.zeros((steps + 10, 42))
        for leg in range(6):
            velocity[10:, leg * 7] = joint[leg]
        active = np.r_[np.zeros(10, bool), np.ones(steps, bool)]
        d = tmp_path / f"birth{b}"
        d.mkdir()
        np.savez(d / "exploration-log.npz", drive=drive.astype(np.float32), credit=np.zeros((steps, 3), np.float32),
                 drive_idx=np.arange(drive.shape[1]), motor_indices=np.arange(3))
        np.savez(d / "training-observations.npz", reward=np.r_[np.zeros(10), reward], velocity=velocity,
                 learning_active=active)
        dirs.append(d)
    out = ct.run(dirs, neurons, max_lag=30, permutations=50, seed=2)
    assert out["births"] == 4 and out["joint_velocity"]["per_leg"][1]["family_p"] < .05
    # a log that does not line up with the observations is refused
    np.savez(dirs[0] / "exploration-log.npz", drive=np.zeros((steps - 5, 60), np.float32), credit=np.zeros((steps - 5, 3)),
             drive_idx=np.arange(60), motor_indices=np.arange(3))
    with pytest.raises(ValueError):
        ct.run(dirs[:1], neurons, max_lag=10, permutations=5)


def test_command_line_writes_a_json_report_and_never_overwrites(tmp_path):
    rng = np.random.default_rng(11)
    neurons = pd.DataFrame({
        "Soma side": ["left"] * (M_PER_LEG * 3) + ["right"] * (M_PER_LEG * 3),
        "Top in/out region": (["T1_PRONM"] * M_PER_LEG + ["T2_MESONM"] * M_PER_LEG + ["T3_METANM"] * M_PER_LEG) * 2})
    neurons.to_parquet(tmp_path / "neurons.parquet")
    dirs = []
    for b in range(2):
        drive, _, joint, reward = birth(rng)
        steps = len(drive)
        velocity = np.zeros((steps + 10, 42))
        for leg in range(6):
            velocity[10:, leg * 7] = joint[leg]
        d = tmp_path / f"b{b}"
        d.mkdir()
        np.savez(d / "exploration-log.npz", drive=drive.astype(np.float32), credit=np.zeros((steps, 3), np.float32),
                 drive_idx=np.arange(drive.shape[1]), motor_indices=np.arange(3))
        np.savez(d / "training-observations.npz", reward=np.r_[np.zeros(10), reward], velocity=velocity,
                 learning_active=np.r_[np.zeros(10, bool), np.ones(steps, bool)])
        dirs.append(str(d))
    out = tmp_path / "result.json"
    argv = [*dirs, "--neurons", str(tmp_path / "neurons.parquet"), "--out", str(out), "--max-lag", "10", "--permutations", "5"]
    ct.main(argv)
    report = __import__("json").loads(out.read_text(encoding="utf-8"))
    assert report["births"] == 2 and set(report["joint_velocity"]["per_leg"]) == {str(i) for i in range(6)}
    with pytest.raises(SystemExit):
        ct.main(argv)  # the existing report is never overwritten
