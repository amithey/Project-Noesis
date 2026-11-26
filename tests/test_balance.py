import numpy as np
import pandas as pd

from noesis.balance import BalanceMap


def _neurons():
    rows = [("afferent", "johnstons_organ_E_neuron")] * 40 + [
        ("afferent", "haltere_campaniform_sensillum_neuron")] * 40 + [
        ("afferent", "johnstons_organ_B_neuron"),  # sound neuron: excluded
        ("intrinsic", "johnstons_organ_E_neuron"),  # not afferent: excluded
    ]
    return pd.DataFrame(rows, columns=["Flow", "Sub Class"])


def test_selects_gravity_and_rotation_afferents_only():
    m = BalanceMap(_neurons())
    assert len(m.jo) == 40 and len(m.hal) == 40 and len(m.idx) == 80


def test_upright_and_still_is_base_rate_tilt_and_spin_raise_it():
    m = BalanceMap(_neurons())
    still = m.drive(np.array([0, 0, 1.0]), np.zeros(3))
    assert np.allclose(still, m.base_hz)
    tilted = m.drive(np.array([0.9, 0.0, 0.1]), np.zeros(3))
    assert tilted[:40].max() > 50 and np.allclose(tilted[40:], m.base_hz)  # only gravity cells
    spinning = m.drive(np.array([0, 0, 1.0]), np.array([0.0, 8.0, 0.0]))
    assert spinning[40:].max() > 50 and np.allclose(spinning[:40], m.base_hz)  # only gyro cells


def test_tilt_direction_is_encoded_by_which_cells_fire():
    m = BalanceMap(_neurons())
    fwd = m.drive(np.array([0.6, 0, 0.8]), np.zeros(3))[:40]
    back = m.drive(np.array([-0.6, 0, 0.8]), np.zeros(3))[:40]
    # cells preferring +x respond more to forward tilt, cells preferring -x to backward tilt
    assert np.corrcoef(fwd - back, m.jo_dir[:, 0])[0, 1] > 0.8
