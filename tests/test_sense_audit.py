import numpy as np
import pandas as pd
import pytest

from noesis.experiments.sense_audit import asymmetry_index, channel_groups, responsive


def test_asymmetry_index():
    assert asymmetry_index(3.0, 1.0) == 0.5
    assert asymmetry_index(1.0, 1.0) == 0.0
    assert asymmetry_index(0.0, 0.0) is None
    with pytest.raises(ValueError):
        asymmetry_index(-1.0, 1.0)


def test_responsive_needs_baseline_and_control():
    stim = np.array([10.0, 10.0, 1.0])
    base = np.array([1.0, 9.5, 0.0])
    ctrl = np.array([2.0, 2.0, 0.0])
    assert responsive(stim, base, ctrl, 2.0).tolist() == [True, False, False]


def test_channel_groups_split_by_side_and_label():
    n = pd.DataFrame({
        "Flow": ["afferent"] * 4 + ["intrinsic"],
        "Class": ["olfactory_receptor_neuron"] * 4 + ["olfactory_receptor_neuron"],
        "Function": ["olfactory,fruity_volatile", "olfactory,fruity_volatile", "olfactory,yeasty_volatile",
                     "olfactory,fruity_volatile", "olfactory,fruity_volatile"],
        "Nerve": ["left_antennal", "right_antennal", "left_antennal", None, None],
        "Soma side": ["left", "right", "left", "right", "left"],
    })
    groups = channel_groups(n)
    assert groups["fruity_volatile"]["left"].tolist() == [0]
    assert groups["fruity_volatile"]["right"].tolist() == [1, 3]
    assert groups["yeasty_volatile"]["left"].tolist() == [2]
    assert len(groups["animal_matter_volatile"]["left"]) == 0
