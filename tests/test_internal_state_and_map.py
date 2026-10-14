"""Unit tests for the internal hemolymph/interoceptive state and live neuron map HUD."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from noesis.internal_state import InternalParams, InternalState
from noesis.viz.brain_map import (
    BrainMap,
    describe_triggers,
    neuron_positions,
    read_activity,
    read_activity_full,
    write_activity,
)


def _mock_neurons() -> pd.DataFrame:
    rows = [
        # index 0: Gr43a hemolymph sugar sensor
        (100, "afferent", "nutrient_sensory_neuron", "hemolymph_sensory_neuron",
         "SLP444", "Gr43a,gustatory,hemolymph_sugar", "DA", "SLP", "right"),
        # index 1: ISN interoceptive sensor
        (101, "afferent", "nutrient_sensory_neuron", "hemolymph_sensory_neuron",
         "ISN", "interoceptive", "SER", "GNG", "left"),
        # index 2: abdominal oxygenation sensor
        (102, "afferent", "oxygenation_neuron", "abdomen_oxygenation_neuron",
         "SAabg01", "oxygenation", "ACH", "ABDNM", "left"),
        # index 3: aorta contractile sensor
        (103, "afferent", "aorta_neuron", "aorta_sensory_neuron",
         "CB0991", "contractile", "ACH", "GNG", "right"),
        # index 4: internal pharyngeal water sensor
        (104, "afferent", "internal_taste_sensillum_gustatory_neuron",
         "pharynx_internal_taste_sensillum_gustatory_neuron", "PhG", "gustatory,ppk28,water", "ACH", "GNG", "left"),
        # index 5: internal pharyngeal nutrient sensor
        (105, "afferent", "internal_taste_sensillum_gustatory_neuron",
         "pharynx_internal_taste_sensillum_gustatory_neuron", "PhG1a", "Gr64e,fatty_acids,gustatory", "ACH", "GNG", "right"),
        # index 6: internal pharyngeal tactile sensor
        (106, "afferent", "internal_sensillum_tactile_neuron",
         "pharynx_internal_sensillum_tactile_neuron", "aPhM1", "tactile", "ACH", "GNG", "left"),
        # index 7: MN9 proboscis motor neuron
        (107, "efferent", "proboscis_motor_neuron", "", "MN9",
         "proboscis_motor,proboscis_positioning", "ACH", "GNG", "left"),
        # index 8: MN11V pharyngeal pumping motor neuron
        (108, "efferent", "pharynx_motor_neuron", "", "MN11V",
         "pharyngeal_pumping,pharynx_motor", "ACH", "GNG", "right"),
        # index 9: DILP insulin-like neurosecretory cell
        (109, "efferent", "pars_intercerebralis_neurosecretory_cell",
         "retrocerebral_complex_neurosecretory_cell", "m_NSC_DILP",
         "energy_metabolism,food_intake", "GLUT", "SMP", "right"),
        # index 10: CRZ cardiostimulatory neurosecretory cell
        (110, "efferent", "pars_lateralis_neurosecretory_cell",
         "retrocerebral_complex_neurosecretory_cell", "l_NSC_CRZ",
         "cardiostimulation,energy_metabolism", "GLUT", "SLP", "left"),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "root_id", "Flow", "Class", "Sub Class", "Primary Cell Type",
            "Function", "Predicted NT type", "Top in/out region", "Soma side",
        ],
    )


def test_internal_state_routes_hemolymph_and_pharyngeal_sensors_and_updates_reserves():
    neurons = _mock_neurons()
    state = InternalState(neurons, InternalParams(initial_sugar=0.5, initial_hydration=0.5, initial_oxygen=0.9))
    assert list(state.idx) == [0, 1, 2, 3, 4, 5, 6]

    # Exercise without food depletes hemolymph sugar/hydration/oxygen and raises ISN + hypoxia drive
    rates_brain = np.zeros(len(neurons))
    r0, tel0 = state.step(rates_brain, dt_s=0.0, muscle_effort=0.0)
    r1, tel1 = state.step(rates_brain, dt_s=10.0, muscle_effort=0.8, humidity=0.2)
    assert tel1["hemolymph_sugar"] < tel0["hemolymph_sugar"]
    assert tel1["hemolymph_hydration"] < tel0["hemolymph_hydration"]
    assert tel1["hemolymph_oxygen"] < tel0["hemolymph_oxygen"]
    assert r1[0] < r0[0]  # Gr43a sugar sensor rate drops as sugar drops
    assert r1[1] > r0[1]  # ISN hunger/thirst rate rises as reserves drop
    assert r1[2] > r0[2]  # Oxygenation hypoxia sensor rises under exertion

    # Feeding on sugar + water with active MN9 & pharyngeal pumping replenishes reserves
    rates_brain[7] = 40.0  # MN9
    rates_brain[8] = 50.0  # pharynx motor
    rates_brain[9] = 25.0  # DILP
    r2, tel2 = state.step(rates_brain, dt_s=5.0, muscle_effort=0.1, taste_sugar=1.0, taste_water=1.0, humidity=0.8)
    assert tel2["hemolymph_sugar"] > tel1["hemolymph_sugar"]
    assert tel2["hemolymph_hydration"] > tel1["hemolymph_hydration"]
    assert tel2["mn9_hz"] == pytest.approx(40.0)
    assert tel2["dilp_hz"] == pytest.approx(25.0)
    assert r2[4] > r1[4] and r2[5] > r1[5]  # internal pharyngeal water & nutrient sensors fire


def test_brain_map_activity_roundtrip_and_hud_rendering(tmp_path):
    import matplotlib
    matplotlib.use("Agg")

    neurons = _mock_neurons()
    pos = neuron_positions(neurons, seed=1)
    assert pos.shape == (len(neurons), 2)
    assert np.isfinite(pos).all()

    rates = np.array([15.0, 0.0, 42.0, 5.0, 0.0, 80.0, 0.0, 30.0, 12.0, 8.0, 0.0], dtype=np.float32)
    triggers = {
        "odor_food": 0.75, "odor_aversive": 0.1, "taste_sugar": 1.0, "taste_water": 0.0, "taste_bitter": 0.0,
        "vision_mean": 0.44, "temperature_c": 25.2, "humidity": 0.6,
        "hemolymph_sugar": 0.58, "hemolymph_hydration": 0.66, "hemolymph_oxygen": 0.91,
        "heart_rate_hz": 4.5, "mn9_hz": 30.0, "dilp_hz": 8.0,
        "body_z_mm": 1.08, "uprightness": 0.98, "muscle_pos_mean": 0.22, "muscle_neg_mean": 0.19,
        "da_hz": 4.2, "oa_hz": 6.1, "ser_hz": 3.0,
    }
    act_file = tmp_path / "live_activity.npz"
    assert write_activity(act_file, rates, 1.25, awake=True, note="BANC corrected | DN 12.0 Hz", triggers=triggers)

    # Backward-compatible 4-tuple and full 5-tuple readers both work
    r4, t4, aw4, note4 = read_activity(act_file)
    r5, t5, aw5, note5, trig5 = read_activity_full(act_file)
    assert aw4 is True and aw5 is True
    assert t4 == pytest.approx(1.25) and t5 == pytest.approx(1.25)
    assert "BANC corrected" in note4 and note4 == note5
    assert trig5["hemolymph_sugar"] == pytest.approx(0.58)
    assert np.allclose(r4, rates, rtol=1e-2)

    hud_text = describe_triggers(trig5)
    assert "world  |" in hud_text
    assert "inside | hemolymph sugar" in hud_text
    assert "body   | z 1.08 mm" in hud_text

    bm = BrainMap(pos, "Test Map")
    bm.update(r5, t5, awake=aw5, note=note5, triggers=trig5)
    out_png = tmp_path / "snapshot.png"
    bm.fig.savefig(out_png, dpi=100, facecolor="black")
    assert out_png.exists() and out_png.stat().st_size > 1000
