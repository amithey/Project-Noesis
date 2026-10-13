import pandas as pd

from noesis.sensory import LegState, SensoryMap


def _neurons():
    cols = ["Flow", "Sub Class", "Nerve", "Soma side"]
    rows = [
        ("afferent", "front_leg_claw_chordotonal_organ_neuron", "left_prothoracic_leg_nerve", "left"),
        ("afferent", "front_leg_hook_chordotonal_organ_neuron", "left_prothoracic_leg_nerve", "left"),
        ("afferent", "hind_leg_bristle_neuron", None, "right"),
        ("afferent", "front_leg_taste_bristle_gustatory_neuron", "left_prothoracic_leg_nerve", "left"),
        ("intrinsic", "front_leg_bristle_neuron", None, "left"),
    ]
    return pd.DataFrame(rows, columns=cols)


def test_groups_only_known_leg_organs_and_sides():
    m = SensoryMap(_neurons())
    assert sorted(m.idx) == [0, 1, 2]  # taste neuron and non-afferent are excluded
    assert {(k, lg) for k, lg, _ in m.groups} == {("claw", "LF"), ("hook", "LF"), ("bristle", "RH")}


def test_flexion_drives_claw_extension_drives_hook_and_touch_drives_bristle():
    m = SensoryMap(_neurons())
    state = {lg: LegState() for lg in ("LF", "RH")}
    state["LF"] = LegState(tibia=+1.0)
    state["RH"] = LegState(load=10.0)
    r = dict(zip(m.idx, m.drive(state), strict=True))
    assert r[0] > r[1]   # claw (flexion) above hook (extension) when flexed
    assert r[2] > 100.0  # loaded leg: bristle near max rate
    state["LF"] = LegState(tibia=-1.0)
    r = dict(zip(m.idx, m.drive(state), strict=True))
    assert r[1] > r[0]
