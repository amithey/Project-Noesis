import numpy as np
import pandas as pd
import pytest

from noesis_gpu.external_senses import ExternalSensoryAdapter
from noesis_gpu.senses_bridge import MODALITIES, build_channels, select_modality


def table():
    rows = [("afferent", "x", "antenna_thermosensory_receptor_neuron", "Gr28a,heating,thermosensory"),
            ("afferent", "x", "antenna_thermosensory_receptor_neuron", "Ir21a,cooling,thermosensory"),
            ("afferent", "x", "antenna_hygrosensory_receptor_neuron", "Ir68a,humid,hygrosensory"),
            ("afferent", "x", "antenna_hygrosensory_receptor_neuron", "Ir40a,dry,hygrosensory"),
            ("afferent", "x", "johnstons_organ_A_neuron", "auditory"),
            ("afferent", "x", "johnstons_organ_B_neuron", "auditory"),
            ("afferent", "olfactory_receptor_neuron", "antenna_olfactory_receptor_neuron", "olfactory"),
            ("efferent", "x", "antenna_thermosensory_receptor_neuron", "Gr28a,heating,thermosensory"),  # wrong flow
            ("intrinsic", "x", None, None),
            ("afferent", "x", "antenna_thermosensory_receptor_neuron", "Gr28a,heating,thermosensory")]
    return pd.DataFrame(rows, columns=["Flow", "Class", "Sub Class", "Function"])


def test_selection_uses_annotations_and_afferents_only():
    n = table()
    assert select_modality(n, "thermo_warm").tolist() == [0, 9]
    assert select_modality(n, "thermo_cool").tolist() == [1]
    assert select_modality(n, "hygro_humid").tolist() == [2] and select_modality(n, "hygro_dry").tolist() == [3]
    assert select_modality(n, "sound_low").tolist() == [4] and select_modality(n, "olfaction").tolist() == [6]


def test_missing_unknown_and_overlapping_modalities_fail_loudly():
    n = table()
    with pytest.raises(ValueError):
        select_modality(n, "vision")  # no photoreceptors in this table
    with pytest.raises(KeyError):
        select_modality(n, "echolocation")
    with pytest.raises(ValueError):
        select_modality(n, "thermo_warm", taken=[9])


def test_build_channels_requires_explicit_gains_and_stays_disjoint_from_taken():
    n = table()
    ch = build_channels(n, {"thermo_warm": 50., "hygro_dry": 20.}, taken=[4, 5])
    assert [c.name for c in ch] == ["thermo_warm", "hygro_dry"]
    assert not set(ch[0].neuron_indices) & set(ch[1].neuron_indices)
    with pytest.raises(ValueError):
        build_channels(n, {"thermo_warm": 50.}, taken=[0])
    with pytest.raises(KeyError):
        build_channels(n, {"thermo_warm": 50., "thermo_warm2": 1.})  # unknown modality is an error, not an empty channel


def test_adapter_integration_has_no_drive_without_signal_and_respects_gain():
    n = table()
    adapter = ExternalSensoryAdapter(len(n), build_channels(n, {"thermo_warm": 40.}))
    idx, rate, _ = adapter.as_drive_part({})
    assert len(idx) == 0 and len(rate) == 0
    idx, rate, _ = adapter.as_drive_part({"thermo_warm": .5})
    assert idx.tolist() == [0, 9] and np.allclose(rate, 20.)


def test_every_declared_modality_rule_names_known_columns():
    assert all(col in ("Class", "Sub Class", "Function") for rules in MODALITIES.values() for col, _ in rules)
