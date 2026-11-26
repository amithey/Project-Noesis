import numpy as np
import pandas as pd
import pytest

from noesis.embodiment import BrainReadout, MotorExploration, load_training_brain


def neurons():
    return pd.DataFrame({
        'Class': ['leg_motor_neuron'] * 3 + ['other'],
        'Sub Class': ['front_leg_motor_neuron'] * 3 + ['other'],
        'Nerve': ['left_prothoracic_leg_nerve'] * 3 + [''],
        'Soma side': ['left'] * 4,
        'Primary Cell Type': ['tibia_flexor', 'tibia_flexor', 'tibia_extensor', 'PAM_test'],
    })


def test_exploration_is_neural_bounded_nonperiodic_and_repeatable():
    a, b = [MotorExploration(neurons(), 2.0, seed=8) for _ in range(2)]
    samples = []
    for _ in range(200):
        x, y = a.step(5), b.step(5)
        np.testing.assert_array_equal(x['current_idx'], y['current_idx'])
        np.testing.assert_array_equal(x['current_mv_per_ms'], y['current_mv_per_ms'])
        assert set(x['current_idx']) == {0, 1, 2}
        current = dict(zip(x['current_idx'], x['current_mv_per_ms']))
        assert current[0] == current[1]  # Same muscle shares a current.
        assert abs(current[0]) < 2 and abs(current[2]) < 2
        samples.append([current[0], current[2]])
    assert np.std(samples, axis=0).min() > .1
    assert abs(np.corrcoef(np.asarray(samples).T)[0, 1]) < .8


def test_frozen_exploration_has_no_current_and_validates_before_mutation():
    e = MotorExploration(pd.DataFrame({'root_id': [1]}))
    assert e.step(5) == {}
    active = MotorExploration(neurons(), 2)
    before = active.state.copy()
    with pytest.raises(ValueError):
        active.step(-1)
    np.testing.assert_array_equal(active.state, before)
    with pytest.raises(ValueError):
        MotorExploration(neurons(), float('nan'))


def test_neural_readout_reports_measured_channels_and_missing_pools():
    readout = BrainReadout(neurons(), [0, 1, 2])
    result = readout.rates(np.array([10., 20., 30., 40.]))
    assert result['motor_hz'] == 20
    assert result['pam_hz'] == 40
    assert result['ppl1_hz'] == 0
    assert result['active_motor_neurons'] == 3


def test_shared_data_are_never_used_as_writable_fallback(tmp_path):
    with pytest.raises(ValueError, match='outside source data'):
        load_training_brain(tmp_path, tmp_path / 'cache')
    with pytest.raises(FileNotFoundError, match='MANC'):
        load_training_brain(tmp_path, tmp_path.parent / 'cache')


def test_gait_report_skips_short_recording_and_handles_partial_bin():
    from noesis.embodiment import gait_report
    assert gait_report([], [], pd.DataFrame(), 20)['status'] == 'too_short'
    report = gait_report([1., 11., 21., 31., 44.], [0, 0, 0, 0, 0], neurons(), 45)
    assert report['analysis_duration_ms'] == 40
    assert report['pooled_leg_spectra']['LF']['spike_count'] == 4
    assert report['tibia_flexor_phase']['tripod_score'] is None
