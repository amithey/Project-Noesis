from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from noesis.environment_senses import TOUCH_SITES, EnvironmentSenses, merge_drive
from noesis.habitat import ODORS, Habitat


def sensory_table():
    rows = [
        ('photoreceptor_neuron', 'retina_photoreceptor_neuron', 'R7', 'left', '', ''),
        ('photoreceptor_neuron', 'retina_photoreceptor_neuron', 'R8', 'right', '', ''),
        ('olfactory_receptor_neuron', 'antenna_olfactory_receptor_neuron', 'ORN', 'left',
         'left_antennal_nerve', 'olfactory,fruity_volatile'),
        ('olfactory_receptor_neuron', 'maxillary_palp_olfactory_receptor_neuron', 'ORN', 'right',
         'right_maxillary-labial_nerve', 'olfactory,aversive_volatile'),
        ('taste_bristle_gustatory_neuron', 'front_leg_taste_bristle_gustatory_neuron', 'SN', 'left',
         'left_prothoracic_leg_nerve', 'gustatory,sugar'),
        ('taste_bristle_gustatory_neuron', 'labellum_taste_bristle_gustatory_neuron', 'LB', 'left',
         'left_maxillary-labial_nerve', 'gustatory,bitter'),
        ('thermosensory_receptor_neuron', '', '', 'left', '', 'heating,thermosensory'),
        ('thermosensory_receptor_neuron', '', '', 'left', '', 'cooling,thermosensory'),
        ('hygrosensory_receptor_neuron', '', '', 'left', '', 'humid,hygrosensory'),
        ('hygrosensory_receptor_neuron', '', '', 'left', '', 'dry,hygrosensory'),
        ('chordotonal_organ_neuron', 'johnstons_organ_B_neuron', '', 'left', '', 'auditory'),
        ('leg_motor_neuron', '', '', 'left', '', ''),
    ]
    n = pd.DataFrame(rows, columns=['Class', 'Sub Class', 'Primary Cell Type', 'Soma side',
                                   'Nerve', 'Function'])
    n['Flow'] = ['afferent'] * 11 + ['efferent']
    return n


def readings():
    vision = np.zeros((2, 4, 2))
    vision[0, :, 1] = 0.2
    vision[1, :, 0] = 0.8
    odor = np.zeros((len(ODORS), 4))
    odor[0, 2], odor[4, 1] = 0.6, 0.4
    feet = np.tile([0, 0, 0.05], (6, 1))
    feet[0] = (3.2, 1.8, 0.05)
    taste = np.zeros((7, 3))
    taste[0, 0], taste[6, 2] = 1, 1
    return {'vision': vision, 'odor_intensity': odor, 'end_effectors': feet, 'surface_taste': taste}


def test_odors_are_spatial_temporal_and_finite_at_source():
    h = Habitat()
    near = h.odors_at([[3.2, 1.8, 0.25]], 0)[0, 0]
    far = h.odors_at([[-50, -50, 0.25]], 0)[0, 0]
    assert 0 <= far < near <= 1
    sample = h.odors_at([[1, -1, 0.5], [1, 1, 0.5]], 0)
    assert sample.shape == (len(ODORS), 2)
    assert sample[0, 0] != sample[0, 1]
    assert not np.allclose(sample, h.odors_at([[1, -1, 0.5], [1, 1, 0.5]], 1))
    assert np.isfinite(h.odors_at(h.odor_positions)).all()


def test_taste_requires_local_surface_contact():
    h = Habitat()
    values = h.tastes_at([[3.2, 1.8, 0.05], [3.2, 1.8, 1], [0, 0, 0.05],
                         [-3.2, 2.5, 0.05], [2, -4, 0.05], [3.2, 1.8, -1]])
    np.testing.assert_array_equal(values, [[1, 0, 0], [0, 0, 0], [0, 0, 0],
                                          [0, 1, 0], [0, 0, 1], [0, 0, 0]])


def test_distinct_warm_cool_humid_and_dry_regions_and_autonomous_motion():
    h = Habitat()
    assert h.local_conditions([-3.5, -3, 0], 0)['temperature_c'] > 30
    assert h.local_conditions([-6, -1, 0], 0)['temperature_c'] < 20
    assert h.local_conditions([-3.2, 2.5, 0], 0)['humidity'] > 0.8
    assert h.local_conditions([6, -5, 0], 0)['humidity'] < 0.25
    assert h.shadow(0) != h.shadow(1)
    assert h.illumination(0) > h.illumination(4)


def test_annotation_mapping_routes_eyes_antennae_palps_and_contact_taste():
    retina = SimpleNamespace(pale_type_mask=np.array([0, 1, 0, 1]))
    senses = EnvironmentSenses(sensory_table(), retina)
    rates, info = senses.drive(readings(), Habitat(), [2, -4, 0.05], 0)
    mapped = dict(zip(senses.idx, rates))
    assert 11 not in mapped  # The environment never drives motor neurons directly.
    for idx, activation in ((0, .2), (1, .8), (2, .6), (3, .4), (4, 1), (5, 1)):
        assert mapped[idx] == pytest.approx(2 + 118 * activation)
    assert info['taste_sugar'] == info['taste_bitter'] == 1
    obs = readings()
    obs['end_effectors'][0, 2] = 1
    obs['surface_taste'][:] = 0
    rates2, _ = senses.drive(obs, Habitat(), [0, 0, 1], 0)
    mapped2 = dict(zip(senses.idx, rates2))
    assert mapped2[4] == mapped2[5] == 2
    assert np.isfinite(rates).all() and np.all((rates >= 2) & (rates <= 120))
    del obs['surface_taste']
    no_contact, _ = senses.drive(obs, Habitat(), [3.2, 1.8, 0.05], 0)
    assert dict(zip(senses.idx, no_contact))[4] == 2


def test_visual_map_is_reproducible_and_colour_change_reaches_rates():
    retina = SimpleNamespace(pale_type_mask=np.array([0, 1, 0, 1]))
    a = EnvironmentSenses(sensory_table(), retina, seed=4)
    b = EnvironmentSenses(sensory_table(), retina, seed=4)
    obs = readings()
    first, _ = a.drive(obs, Habitat(), [0, 0, 1], 0)
    np.testing.assert_array_equal(first, b.drive(obs, Habitat(), [0, 0, 1], 0)[0])
    obs['vision'][0, :, 1] = 1
    changed, _ = a.drive(obs, Habitat(), [0, 0, 1], 0)
    assert changed[0] > first[0]
    np.testing.assert_array_equal(changed[1:], first[1:])


def test_merge_keeps_gains_and_rejects_double_stimulation():
    drive = merge_drive([(np.array([3]), np.array([10.]), 400),
                         (np.array([5]), np.array([20.]), 100)])
    np.testing.assert_array_equal(drive['drive_idx'], [3, 5])
    np.testing.assert_array_equal(drive['drive_weight'], [400, 100])
    with pytest.raises(ValueError, match='overlap'):
        merge_drive([(np.array([3]), np.array([10.]), 400)] * 2)
    assert merge_drive([]) == {}


@pytest.mark.parametrize('bad', [[[np.nan, 0, 0]], [[0, 0]], [0, 0, 0]])
def test_field_rejects_invalid_sensor_positions(bad):
    with pytest.raises(ValueError):
        Habitat().odors_at(bad)


def test_dynamic_waves_trigger_heat_cool_humid_dry_and_wind_at_origin():
    h = Habitat()
    times = np.linspace(0.0, 2.5, 26)
    samples = [h.local_conditions([0.0, 0.0, 0.5], float(t)) for t in times]
    temps = [s['temperature_c'] for s in samples]
    humids = [s['humidity'] for s in samples]
    winds = [s['airflow_envelope'] for s in samples]
    assert max(temps) > 28.0 and min(temps) < 20.0
    assert max(humids) > 0.65 and min(humids) < 0.25
    assert max(winds) > 0.8 and min(winds) < 0.1
    # Nearby micro-patches allow foot contact taste near the spawn workspace without polluting (0, 0).
    near_touches = h.tastes_at([[1.15, 0.15, 0.05], [-0.65, 1.95, 0.05], [-1.05, 0.22, 0.05]])
    np.testing.assert_array_equal(near_touches, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])


def test_expanded_receptors_route_wind_touch_evaporation_and_labellar_tastes():
    rows = [
        ('hygrosensory_receptor_neuron', 'antenna_hygrosensory_receptor_neuron', '', 'left', '',
         'Ir40a,cooling,hygrosensory'),
        ('thermosensory_receptor_neuron', 'antenna_thermosensory_receptor_neuron', '', 'left', '',
         'Ir68a,evaporation,thermosensory'),
        ('taste_peg_gustatory_neuron', 'labellum_taste_peg_gustatory_neuron', '', 'left', '',
         'Gr64e,Ir56d,fatty_acids,gustatory'),
        ('taste_bristle_gustatory_neuron', 'labellum_taste_bristle_gustatory_neuron', '', 'left', '',
         'aversive,gustatory'),
        ('bristle_neuron', 'antenna_bristle_neuron', '', 'left', '', 'mechanosensory'),
        ('bristle_neuron', 'thorax_bristle_neuron', '', 'left', '', 'mechanosensory'),
        ('taste_bristle_tactile_neuron', 'front_leg_taste_bristle_tactile_neuron', '', 'left',
         'left_prothoracic_leg_nerve', 'mechanosensory'),
    ]
    n = pd.DataFrame(rows, columns=['Class', 'Sub Class', 'Primary Cell Type', 'Soma side',
                                   'Nerve', 'Function'])
    n['Flow'] = 'afferent'
    retina = SimpleNamespace(pale_type_mask=np.array([0, 1, 0, 1]))
    senses = EnvironmentSenses(n, retina)
    obs = readings()
    obs['surface_taste'][:] = 0
    obs['surface_taste'][6, 0] = 1.0  # nutritive mouth taste
    obs['surface_taste'][6, 2] = 0.5  # aversive/bitter mouth taste
    touch = np.zeros(len(TOUCH_SITES))
    touch[0] = 0.75  # thorax
    touch[3] = 0.25  # LF leg tactile bristle
    obs['surface_touch'] = touch
    # At [6, -5, 0.05] and t=1.5s, temperature < 24 (cool) and humidity < 0.45 (dry/evaporation).
    rates, info = senses.drive(obs, Habitat(), [6.0, -5.0, 0.05], 1.5)
    mapped = dict(zip(senses.idx, rates))
    assert mapped[0] > 2.0  # Ir40a cooling hygrosensory
    assert mapped[1] > 2.0  # Ir68a evaporation thermosensory
    assert mapped[2] == pytest.approx(2 + 118 * 1.0)
    assert mapped[3] == pytest.approx(2 + 118 * 0.5)
    assert mapped[4] == pytest.approx(2 + 118 * info['airflow_envelope'])
    assert mapped[5] == pytest.approx(2 + 118 * 0.75)
    assert mapped[6] == pytest.approx(2 + 118 * 0.25)
    assert info['touch_contact'] == pytest.approx(0.75)
