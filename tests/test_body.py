import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from noesis.motor import build_muscle_matrices

pytest.importorskip("flygym")  # the body needs NeuroMechFly's shipped kinematics

from noesis.body import JOINTS, BodyParams, MuscleModel, joint_limits, standing_pose


def _muscles(rate_pos=0.0, rate_neg=0.0, joints=("joint_LFTibia",)):
    n = len(joints)
    g_pos = sparse.csr_matrix(np.ones((n, 1)))   # one neuron per pool, rate passed straight through
    g_neg = sparse.csr_matrix(np.ones((n, 1)))
    return MuscleModel(g_pos, g_neg, list(joints)), np.array([rate_pos]), np.array([rate_neg])


def test_standing_pose_is_inside_every_joint_limit():
    pose, lim = standing_pose(), joint_limits()
    assert set(pose) == set(JOINTS)
    assert all(lim[j][0] < pose[j] < lim[j][1] for j in JOINTS)


def test_relaxed_muscles_leave_the_joint_at_the_standing_posture():
    m, _, _ = _muscles()
    assert np.allclose(m.step(np.array([0.0]), 5.0), m.stand)


def test_full_agonist_or_antagonist_reaches_the_muscle_reach_not_the_hard_stop():
    p = BodyParams()
    m, _, _ = _muscles()
    for _ in range(200):
        eq_hi = m.step(np.array([1e6]), 5.0)  # g_pos and g_neg both see this rate -> co-contraction
    assert m.a_pos[0] > 0.99 and m.a_neg[0] > 0.99
    # co-contraction of equal muscles cancels: back to (almost) the standing posture
    assert abs(eq_hi[0] - m.stand[0]) < 0.5 * (m.hi_eq[0] - m.lo_eq[0])
    assert m.hi_eq[0] < m.hi[0] and m.lo_eq[0] > m.lo[0]
    assert np.isclose(m.hi_eq[0] - m.stand[0], p.muscle_reach * (m.hi[0] - m.stand[0]))


def test_activation_rises_gradually_with_the_activation_time_constant():
    m = MuscleModel(sparse.csr_matrix([[1.0]]), sparse.csr_matrix([[0.0]]), ["joint_LFTibia"])
    m.step(np.array([100.0]), 5.0)
    first = m.a_pos[0]
    for _ in range(100):
        m.step(np.array([100.0]), 5.0)
    assert 0 < first < 0.3 * m.a_pos[0] < 1  # slow rise, then saturates below 1
    assert m.a_pos[0] == pytest.approx(100 / (100 + m.p.r_half), rel=0.02)


def test_agonist_pulls_up_antagonist_pulls_down_and_output_stays_in_range():
    up = MuscleModel(sparse.csr_matrix([[1.0]]), sparse.csr_matrix([[0.0]]), ["joint_LFTibia"])
    down = MuscleModel(sparse.csr_matrix([[0.0]]), sparse.csr_matrix([[1.0]]), ["joint_LFTibia"])
    for _ in range(300):
        a = up.step(np.array([500.0]), 5.0)
        b = down.step(np.array([500.0]), 5.0)
    assert a[0] > up.stand[0] and b[0] < down.stand[0]
    assert up.lo_eq[0] <= a[0] <= up.hi_eq[0] and down.lo_eq[0] <= b[0] <= down.hi_eq[0]


def test_muscle_matrices_split_agonists_and_antagonists_per_joint():
    rows = [
        ("leg_motor_neuron", "front_leg_motor_neuron", "left_prothoracic_leg_nerve", "left", "tibia_flexor"),
        ("leg_motor_neuron", "front_leg_motor_neuron", "left_prothoracic_leg_nerve", "left", "tibia_flexor"),
        ("leg_motor_neuron", "front_leg_motor_neuron", "left_prothoracic_leg_nerve", "left", "tibia_extensor"),
    ]
    df = pd.DataFrame(rows, columns=["Class", "Sub Class", "Nerve", "Soma side", "Primary Cell Type"])
    gp, gn = build_muscle_matrices(df, ["joint_LFTibia"])
    assert np.allclose(gp.toarray(), [[0.5, 0.5, 0.0]])   # flexors pull positive (mean of the pool)
    assert np.allclose(gn.toarray(), [[0.0, 0.0, 1.0]])   # the extensor pulls negative


def test_tonic_adaptation_rebalances_chronic_asymmetry_while_preserving_phasic_drive():
    gp = sparse.csr_matrix([[1.0, 0.0], [1.0, 0.0]])
    gn = sparse.csr_matrix([[0.0, 1.0], [0.0, 1.0]])
    joints = ["joint_RHFemur", "joint_RHTibia"]
    unadapted = MuscleModel(gp, gn, joints, tonic_tau_ms=0.0)
    adapted = MuscleModel(gp, gn, joints, tonic_tau_ms=250.0, tonic_balance=1.0, femur_stance_bias=0.15)
    tonic_rates = np.array([2.0, 80.0])  # strong antagonist tonic imbalance (like RH_Femur)
    for _ in range(400):
        eq_unadapted = unadapted.step(tonic_rates, 5.0)
        eq_adapted = adapted.step(tonic_rates, 5.0)
    # Without tonic adaptation, RHFemur is pulled strongly toward lo_eq (lifted above thorax)
    assert eq_unadapted[0] < unadapted.stand[0] - 0.4 * (unadapted.stand[0] - unadapted.lo_eq[0])
    # With tonic adaptation (balance=1.0) and femur_stance_bias=0.15, RHFemur settles slightly into stance (> stand)
    # and RHTibia settles back at its standing angle
    assert eq_adapted[0] > adapted.stand[0]
    assert abs(eq_adapted[1] - adapted.stand[1]) < 0.02 * (adapted.hi_eq[1] - adapted.lo_eq[1])
    # Phasic burst on agonist or antagonist still deflects the joint bidirectionally
    eq_burst_pos = adapted.step(np.array([200.0, 80.0]), 5.0)
    assert eq_burst_pos[1] > eq_adapted[1]


def test_tonic_adaptation_rejects_invalid_parameters():
    gp = sparse.csr_matrix([[1.0]])
    gn = sparse.csr_matrix([[1.0]])
    with pytest.raises(ValueError, match="tonic_tau_ms"):
        MuscleModel(gp, gn, ["joint_LFTibia"], tonic_tau_ms=-1.0)
    with pytest.raises(ValueError, match="tonic_balance"):
        MuscleModel(gp, gn, ["joint_LFTibia"], tonic_tau_ms=100.0, tonic_balance=1.5)
    with pytest.raises(ValueError, match="femur_stance_bias"):
        MuscleModel(gp, gn, ["joint_LFTibia"], tonic_tau_ms=100.0, femur_stance_bias=2.0)
