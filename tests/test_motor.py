import numpy as np
import pandas as pd
from scipy import sparse

from noesis.lif import LIFNetwork
from noesis.motor import RateTracker, build_motor_matrix


def _neurons():
    rows = [
        # Class, Sub Class, Nerve, Soma side, Primary Cell Type
        ("leg_motor_neuron", "front_leg_motor_neuron", "left_prothoracic_leg_nerve", "left", "tibia_flexor"),
        ("leg_motor_neuron", "front_leg_motor_neuron", "left_prothoracic_leg_nerve", "left", "tibia_flexor"),
        ("leg_motor_neuron", "front_leg_motor_neuron", "left_prothoracic_leg_nerve", "left", "tibia_extensor"),
        ("leg_motor_neuron", "hind_leg_motor_neuron", None, "right", "tarsus_levator"),
        ("descending", None, None, "left", "DNx"),
    ]
    return pd.DataFrame(rows, columns=["Class", "Sub Class", "Nerve", "Soma side", "Primary Cell Type"])


def test_motor_matrix_signs_and_group_means():
    joints = ["joint_LFTibia", "joint_RHTarsus1", "joint_LFFemur"]
    G = build_motor_matrix(_neurons(), joints).toarray()
    assert np.allclose(G[0], [0.5, 0.5, -1.0, 0, 0])   # mean flexor rate - extensor rate
    assert np.allclose(G[1], [0, 0, 0, -1.0, 0])      # levator is the antagonist, side from soma
    assert not G[2].any()                              # no femur muscles present
    rates = np.array([10.0, 20.0, 4.0, 0.0, 0.0])
    assert (G @ rates)[0] == 11.0                      # 15 Hz flexor mean - 4 Hz extensor


def test_rate_tracker_converges_to_rate():
    tr = RateTracker(1, tau_ms=50.0)
    for _ in range(100):  # 1 spike per 5 ms window = 200 Hz
        r = tr.update(np.array([0]), 5.0)
    assert abs(r[0] - 200.0) < 20.0


def test_chunked_run_matches_single_run():
    W = sparse.csr_matrix(np.array([[0, 150.0], [0, 0]]))
    a = LIFNetwork(W, seed=1)
    ta, ia = a.run(100.0, np.array([0]), 100.0, 100.0)
    b = LIFNetwork(W, seed=1)
    t1, i1 = b.run(50.0, np.array([0]), 100.0, 100.0)
    t2, i2 = b.run(50.0, np.array([0]), 100.0, 100.0)
    assert np.allclose(ta, np.concatenate([t1, t2])) and np.array_equal(ia, np.concatenate([i1, i2]))


def test_wing_and_neck_motor_mappings():
    rows = [
        ("wing_motor_neuron", "wing_power_motor_neuron", "left_anterior_dorsal_mesothoracic_nerve", "left", "DVM1a-c"),
        ("wing_motor_neuron", "wing_power_motor_neuron", "left_posterior_dorsal_mesothoracic_nerve", "left", "DLM1-4"),
        ("wing_motor_neuron", "wing_steering_motor_neuron", "right_anterior_dorsal_mesothoracic_nerve", "right", "b1"),
        ("neck_motor_neuron", "neck_yaw_motor_neuron", "left_cervical_nerve", "left", "CvN1"),
        ("neck_motor_neuron", "neck_pitch_motor_neuron", "left_cervical_nerve", "left", "CvN4"),
    ]
    df = pd.DataFrame(rows, columns=["Class", "Sub Class", "Nerve", "Soma side", "Primary Cell Type"])
    joints = ["joint_LWing", "joint_RWing_yaw", "joint_Head_yaw", "joint_Head"]
    G = build_motor_matrix(df, joints).toarray()
    assert np.allclose(G[0], [1.0, -1.0, 0.0, 0.0, 0.0])   # DVM (+1) vs DLM (-1) on left wing
    assert np.allclose(G[1], [0.0, 0.0, 1.0, 0.0, 0.0])    # right b1 (+1) on right wing yaw
    assert np.allclose(G[2], [0.0, 0.0, 0.0, 1.0, 0.0])    # left neck yaw (+1)
    assert np.allclose(G[3], [0.0, 0.0, 0.0, 0.0, 1.0])    # CvN4 neck pitch (+1)

