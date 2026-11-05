import json

import numpy as np
import pandas as pd
import pytest
from scipy.signal import lfilter

from noesis_gpu import credit_timing as ct

T, M_PER_LEG, TAU = 990, 10, 20  # steps per birth, explored neurons per leg, OU time constant in steps
GAINS = np.array([1.0, -0.5, 0.8, -1.0, 0.3, 0.6, -0.7])  # mixed-sign effect on the 7 joints of the affected leg


def ou(rng, n, columns, tau=TAU):
    decay = np.exp(-1 / tau)
    return np.tanh(lfilter([np.sqrt(1 - decay ** 2)], [1, -decay], rng.normal(size=(n, columns)), axis=0))


def birth(rng, effect_leg=None, lag=8, kappa=0.0, noise=0.5):
    """Injected current for 6 legs x M neurons with a shared per-leg component, joint velocities (T, 7) per leg, reward."""
    latent = ou(rng, T, 6)
    drive = np.concatenate([latent[:, [leg]] * rng.uniform(.5, 1.5, M_PER_LEG) + 0.3 * ou(rng, T, M_PER_LEG)
                            for leg in range(6)], axis=1)
    legs = np.repeat(np.arange(6), M_PER_LEG)
    joint = {leg: noise * ou(rng, T, 7) + noise * rng.normal(size=(T, 7)) for leg in range(6)}
    reward = 0.3 * rng.normal(size=T) + 0.2 * ou(rng, T, 1, tau=400)[:, 0]
    if effect_leg is not None:
        shifted = np.zeros(T)
        shifted[lag:] = latent[:T - lag, effect_leg]
        joint[effect_leg] = joint[effect_leg] + kappa * shifted[:, None] * GAINS
        reward = reward + kappa * 0.5 * shifted
    return drive, legs, joint, reward


def prepare(births, components=4):
    drives = [b[0] for b in births]
    legs = births[0][1]
    features = {leg: [] for leg in range(6)}
    for leg in range(6):
        basis = ct.leg_basis(drives, legs, leg, components)
        features[leg] = [ct.project(d, basis) for d in drives]
    joints = {leg: [b[2][leg] for b in births] for leg in range(6)}
    reward = {leg: [b[3] for b in births] for leg in range(6)}
    return features, joints, reward


def test_assign_legs_uses_side_and_neuromere_and_marks_ambiguous_neurons():
    neurons = pd.DataFrame({
        "Soma side": ["left", "right", "left", "right", "left", "right"],
        "Top in/out region": ["T1_PRONM", "T2_MESONM", "T3_METANM", "T3_METANM", "T2_MESONM.T3_METANM", "ABDNM"]})
    legs = ct.assign_legs(neurons, np.arange(6))
    assert list(legs) == [0, 4, 2, 5, -1, -1]  # LF, RM, LH, RH, ambiguous, other


def test_nerve_prefix_takes_precedence_over_the_soma_side():
    neurons = pd.DataFrame({"Soma side": ["left", "left"], "Top in/out region": ["T1_PRONM", "T1_PRONM"],
                            "Nerve": ["right_prothoracic_leg_nerve", None]})
    assert list(ct.assign_legs(neurons, np.arange(2))) == [3, 0]


def test_ema_residual_removes_a_slow_baseline_and_keeps_fast_changes():
    reward = np.concatenate([np.zeros(400), np.ones(400)])
    residual = ct.ema2s_reward_residual(reward)
    assert residual[:400].max() == 0 and residual[400] == pytest.approx(1.0)
    assert residual[-1] == pytest.approx(np.exp(-399 * 5 / 2000), rel=.02)  # 2 s after the jump the baseline has closed 63%


def test_learning_signal_is_the_learners_own_normalized_error_and_differs_from_the_plain_residual():
    rng = np.random.default_rng(2)
    reward = 0.2 + 0.1 * rng.normal(size=600)
    signal = ct.learning_signal(reward)
    assert signal[0] == 0.0  # bias correction: the first observation only sets the baseline
    assert np.abs(signal).max() <= 1.0 and signal.std() > 0
    # normalization makes it independent of the reward scale (away from the 0.01 floor and the clip)
    np.testing.assert_allclose(ct.learning_signal(reward * 0.5), ct.learning_signal(reward), atol=1e-6)
    plain = ct.ema2s_reward_residual(reward)
    assert plain.std() < signal.std()  # the plain residual is not on the learner's normalized scale
    assert not np.allclose(plain / plain.std(), signal / signal.std(), atol=0.05)  # and not the same signal


def test_per_birth_feature_signs_can_erase_a_perfect_relation_but_the_shared_basis_cannot():
    """Codex's counterexample, reproduced: identical relation in two births, the second one expressed with flipped features."""
    rng = np.random.default_rng(4)
    x = rng.normal(size=(600, 1))
    y = x[:, 0]
    same = ct.LagFit([x, x], 0, 0).r2([y, y])[0]
    flipped_in_birth_two = ct.LagFit([x, -x], 0, 0).r2([y, y])[0]  # what a per-birth PCA sign flip would do
    assert same > 0.999 and flipped_in_birth_two < 0.01  # the failure mode is real
    # the shared basis is one fixed matrix applied to every birth, so no per-birth sign or rotation exists
    drive = ou(rng, 600, 12)
    legs = np.zeros(12, dtype=int)
    basis = ct.leg_basis([drive, drive], legs, 0, components=3)
    np.testing.assert_array_equal(ct.project(drive, basis), ct.project(drive, basis))
    first = ct.project(drive, basis)
    second = ct.project(drive.copy(), basis)
    np.testing.assert_array_equal(first, second)  # same current -> same features in every birth


def test_r2_is_invariant_to_rotating_or_flipping_the_shared_features():
    rng = np.random.default_rng(6)
    births = [birth(rng, effect_leg=1, lag=5, kappa=1.0) for _ in range(3)]
    features, joints, _ = prepare(births)
    base = ct.LagFit(features[1], 12, 12).r2(joints[1])
    q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    rotated = [f @ q for f in features[1]]  # the same rotation/flip in every birth, as a shared basis would give
    np.testing.assert_allclose(ct.LagFit(rotated, 12, 12).r2(joints[1]), base, atol=1e-9)


def test_a_planted_effect_is_recovered_at_the_right_leg_and_lag_with_multi_joint_response():
    rng = np.random.default_rng(1)
    births = [birth(rng, effect_leg=2, lag=8, kappa=1.0) for _ in range(4)]
    features, joints, _ = prepare(births)
    result, null = ct.analyse(features, joints, max_lag=30, permutations=60, seed=3)
    assert result[2]["family_rank"] < .02 and abs(result[2]["best_lag_steps"] - 8) <= 1
    assert result[2]["r2"] > null["null_max_p99"]
    assert sum(result[leg]["family_rank"] < .05 for leg in range(6) if leg != 2) <= 1  # other legs stay quiet


def test_the_effect_is_found_even_when_births_use_different_random_seeds_and_orders():
    rng = np.random.default_rng(21)
    births = [birth(rng, effect_leg=4, lag=6, kappa=1.0) for _ in range(5)]
    forward, _, _ = prepare(births)
    backward, joints_b, _ = prepare(births[::-1])
    a = ct.LagFit(forward[4], 20, 20).r2([b[2][4] for b in births])
    b = ct.LagFit(backward[4], 20, 20).r2(joints_b[4])
    np.testing.assert_allclose(a, b, atol=1e-9)  # births are exchangeable: no hidden per-birth convention


def test_without_any_effect_false_positives_stay_near_the_nominal_rate():
    hits = 0
    for seed in range(8):
        rng = np.random.default_rng(100 + seed)
        births = [birth(rng) for _ in range(4)]
        features, joints, _ = prepare(births)
        result, _ = ct.analyse(features, joints, max_lag=20, permutations=40, seed=seed)
        hits += min(r["family_rank"] for r in result.values()) < .05
    assert hits <= 2  # 8 null datasets at a 5% family-wise level; allow generous slack


def test_an_effect_on_the_learning_signal_is_found_at_the_right_leg_and_lag_and_a_wrong_leg_is_not():
    rng = np.random.default_rng(7)
    births = [birth(rng, effect_leg=0, lag=10, kappa=1.0, noise=0.5) for _ in range(5)]
    features, _, reward = prepare(births)
    result, _ = ct.analyse(features, reward, max_lag=30, permutations=60, seed=1)
    assert result[0]["family_rank"] < .05 and result[0]["best_lag_steps"] == 10
    assert all(result[leg]["family_rank"] > .05 for leg in range(1, 6))


def test_the_design_detects_an_effect_explaining_a_few_percent_of_the_variance():
    """Documents the detection floor with the shared basis: the null maximum R^2 is about 1% for 8 births of 5 s."""
    rng = np.random.default_rng(31)
    births = [birth(rng, effect_leg=0, lag=10, kappa=0.2) for _ in range(8)]
    features, joints, reward = prepare(births)
    result, null = ct.analyse(features, reward, max_lag=40, permutations=60, seed=2)
    assert null["null_max_p99"] < .03  # nothing below about 3% R^2 is ever called an effect by chance
    assert result[0]["best_lag_steps"] == 10 and result[0]["r2"] > null["null_max_p99"]
    assert result[0]["family_rank"] < .05
    joint_result, _ = ct.analyse(features, joints, max_lag=40, permutations=60, seed=2)
    assert joint_result[0]["best_lag_steps"] == 10 and joint_result[0]["family_rank"] < .05


def write_birth(directory, drive, joint, reward, drive_idx=None):
    steps = len(drive)
    velocity = np.zeros((steps + 10, 42))
    for leg in range(6):
        velocity[10:, leg * 7:(leg + 1) * 7] = joint[leg]
    directory.mkdir()
    np.savez(directory / "exploration-log.npz", drive=drive.astype(np.float32), credit=np.zeros((steps, 3), np.float32),
             drive_idx=np.arange(drive.shape[1]) if drive_idx is None else drive_idx, motor_indices=np.arange(3))
    np.savez(directory / "training-observations.npz", reward=np.r_[np.zeros(10), reward], velocity=velocity,
             learning_active=np.r_[np.zeros(10, bool), np.ones(steps, bool)])
    return directory


def neurons_table():
    return pd.DataFrame({
        "Soma side": ["left"] * (M_PER_LEG * 3) + ["right"] * (M_PER_LEG * 3),
        "Top in/out region": (["T1_PRONM"] * M_PER_LEG + ["T2_MESONM"] * M_PER_LEG + ["T3_METANM"] * M_PER_LEG) * 2})


def test_run_reads_the_log_and_observation_files_and_checks_alignment(tmp_path):
    rng = np.random.default_rng(5)
    dirs = []
    for b in range(4):
        drive, _, joint, reward = birth(rng, effect_leg=1, lag=6, kappa=1.2)
        dirs.append(write_birth(tmp_path / f"birth{b}", drive, joint, reward))
    out = ct.run(dirs, neurons_table(), max_lag=30, permutations=50, seed=2)
    assert out["births"] == 4 and out["joint_velocity"]["per_leg"][1]["family_rank"] < .05
    assert set(out) >= {"learning_signal", "joint_velocity"}  # the error signal is reported under its real name
    # a log that does not line up with the observations is refused
    steps = T
    np.savez(dirs[0] / "exploration-log.npz", drive=np.zeros((steps - 5, 60), np.float32), credit=np.zeros((steps - 5, 3)),
             drive_idx=np.arange(60), motor_indices=np.arange(3))
    with pytest.raises(ValueError):
        ct.run(dirs[:1], neurons_table(), max_lag=10, permutations=5)


def test_run_refuses_births_that_explore_different_neurons(tmp_path):
    rng = np.random.default_rng(8)
    dirs = []
    for b in range(2):
        drive, _, joint, reward = birth(rng)
        ids = np.arange(drive.shape[1]) if b == 0 else np.arange(drive.shape[1])[::-1].copy()
        dirs.append(write_birth(tmp_path / f"b{b}", drive, joint, reward, drive_idx=ids))
    with pytest.raises(ValueError, match="different neurons"):
        ct.run(dirs, neurons_table(), max_lag=10, permutations=5)


def test_command_line_writes_a_json_report_and_never_overwrites(tmp_path):
    rng = np.random.default_rng(11)
    neurons_table().to_parquet(tmp_path / "neurons.parquet")
    dirs = []
    for b in range(2):
        drive, _, joint, reward = birth(rng)
        dirs.append(str(write_birth(tmp_path / f"b{b}", drive, joint, reward)))
    out = tmp_path / "result.json"
    argv = [*dirs, "--neurons", str(tmp_path / "neurons.parquet"), "--out", str(out), "--max-lag", "10", "--permutations", "5"]
    ct.main(argv)
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["births"] == 2 and set(report["joint_velocity"]["per_leg"]) == {str(i) for i in range(6)}
    with pytest.raises(SystemExit):
        ct.main(argv)  # the existing report is never overwritten


def test_leg_assignment_reports_a_reason_for_every_neuron_and_never_guesses_no_consensus_labels():
    neurons = pd.DataFrame({
        "Soma side": ["left", "right", "left", "right", "left", "left", "right"],
        "Top in/out region": ["T1_PRONM.NO_CONS", "T3_METANM.NO_CONS", "T2_MESONM.NO_CONS", "T1_PRONM",
                              "T2_MESONM.T3_METANM", "T1_PRONM", "ABDNM"],
        "Nerve": ["left_prothoracic_leg_nerve", "right_metathoracic_leg_nerve", "left_ventral_prothoracic_nerve",
                  None, None, None, None]})
    legs, reasons = ct.assign_legs_detailed(neurons, np.arange(7))
    assert list(legs) == [0, 5, -1, 3, -1, 0, -1]
    assert reasons == ["nerve", "nerve", "non_leg_nerve", "region", "region_ambiguous", "region", "no_side_or_region"]
    unlabelled = pd.DataFrame({"Soma side": ["left"], "Top in/out region": ["T2_MESONM.NO_CONS"], "Nerve": [None]})
    assert ct.assign_legs_detailed(unlabelled, np.arange(1)) == (pytest.approx(np.array([-1])), ["region_no_consensus"])


def test_run_reports_coverage_provenance_and_the_descriptive_interpretation(tmp_path):
    rng = np.random.default_rng(13)
    table = neurons_table()
    # make the whole right-hind leg unusable: no consensus label and no nerve, and one constant column on the left front
    table.loc[M_PER_LEG * 5:, "Top in/out region"] = "T3_METANM.NO_CONS"
    dirs = []
    for b in range(3):
        drive, _, joint, reward = birth(rng, effect_leg=1, lag=6, kappa=1.0)
        drive[:, 0] = 0.25  # constant column
        dirs.append(write_birth(tmp_path / f"c{b}", drive, joint, reward))
    out = ct.run(dirs, table, max_lag=15, permutations=20, seed=1, neurons_sha256="abc")
    cov = out["coverage"]
    assert cov["explored_neurons"] == 6 * M_PER_LEG and cov["assigned"] == 5 * M_PER_LEG
    assert cov["unassigned_by_reason"]["region_no_consensus"] == M_PER_LEG
    assert cov["constant_columns"] == 1 and cov["constant_columns_per_birth"] == [1, 1, 1]
    assert cov["per_leg"][0]["constant"] == 1 and cov["per_leg"][0]["usable"] == M_PER_LEG - 1
    assert cov["per_leg"][5]["status"] == "INSUFFICIENT_COVERAGE" and 5 not in out["analysed_legs"]
    assert out["joint_velocity"]["per_leg"][5] == {"status": "INSUFFICIENT_COVERAGE"}
    assert out["joint_velocity"]["per_leg"][1]["rank_is_calibrated_p_value"] is False
    assert "calibrated" in out["interpretation"] and "not established" in out["interpretation"]
    prov = out["provenance"]
    assert len(prov["inputs"]) == 3 and all(len(i["exploration_log_sha256"]) == 64 for i in prov["inputs"])
    assert prov["neurons_table_sha256"] == "abc" and len(prov["module_sha256"]) == 64 and prov["parameters"]["seed"] == 1
    assert len(out["joint_velocity"]["per_birth"]) == 3 and set(out["joint_velocity"]["per_birth"][0]) == set(out["analysed_legs"])
