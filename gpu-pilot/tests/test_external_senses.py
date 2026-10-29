"""Synthetic unit tests for the external senses drive adapter.

Validates the opt-in neural sensory-drive adapter without requiring connectome,
MuJoCo physics, GPU hardware, or network access.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from noesis.environment_senses import merge_drive
from noesis_gpu.external_senses import ChannelSpec, ExternalSensoryAdapter


def test_channel_spec_validation():
    """Verify channel specification bounds, types, and units."""
    # Valid spec
    ch = ChannelSpec("light_left", neuron_indices=[10, 11], gain=120.0, min_signal=0.0, max_signal=1.0)
    assert ch.name == "light_left"
    assert ch.neuron_indices == (10, 11)
    assert ch.gain == 120.0
    assert ch.min_signal == 0.0
    assert ch.max_signal == 1.0

    # Invalid names
    with pytest.raises(ValueError, match="non-empty string"):
        ChannelSpec("", neuron_indices=[1])
    with pytest.raises(ValueError, match="non-empty string"):
        ChannelSpec("   ", neuron_indices=[1])

    # Negative or non-integer indices
    with pytest.raises(ValueError, match="non-negative"):
        ChannelSpec("test", neuron_indices=[-1])
    with pytest.raises(ValueError, match="non-negative"):
        ChannelSpec("test", neuron_indices=-1)
    with pytest.raises(ValueError, match="non-negative"):
        ChannelSpec("test", neuron_indices=np.array([1, -2]))

    # Floats must raise TypeError without silent truncation
    with pytest.raises(TypeError, match="must be an integer"):
        ChannelSpec("test", neuron_indices=[1.5])  # type: ignore
    with pytest.raises(TypeError, match="must be an integer"):
        ChannelSpec("test", neuron_indices=1.5)  # type: ignore
    with pytest.raises(TypeError, match="must be an integer"):
        ChannelSpec("test", neuron_indices=np.float64(1.5))  # type: ignore
    with pytest.raises(TypeError, match="must be an integer"):
        ChannelSpec("test", neuron_indices=np.array([1.5, 2.5]))  # type: ignore

    # Booleans (scalar and sequence, Python and NumPy) must raise TypeError
    with pytest.raises(TypeError, match="must be an integer"):
        ChannelSpec("test", neuron_indices=True)  # type: ignore
    with pytest.raises(TypeError, match="must be an integer"):
        ChannelSpec("test", neuron_indices=False)  # type: ignore
    with pytest.raises(TypeError, match="must be an integer"):
        ChannelSpec("test", neuron_indices=np.bool_(True))  # type: ignore
    with pytest.raises(TypeError, match="must be an integer"):
        ChannelSpec("test", neuron_indices=[True])  # type: ignore
    with pytest.raises(TypeError, match="must be an integer"):
        ChannelSpec("test", neuron_indices=[False, True])  # type: ignore
    with pytest.raises(TypeError, match="must be an integer"):
        ChannelSpec("test", neuron_indices=[np.bool_(True)])  # type: ignore
    with pytest.raises(TypeError, match="must be an integer"):
        ChannelSpec("test", neuron_indices=np.array([True, False]))  # type: ignore

    # Multidimensional arrays and nested sequences must raise
    with pytest.raises(ValueError, match="array must be 1D"):
        ChannelSpec("test", neuron_indices=np.array([[1, 2], [3, 4]]))
    with pytest.raises(TypeError, match="must be an integer"):
        ChannelSpec("test", neuron_indices=[[1, 2], [3, 4]])  # type: ignore

    # Valid scalar ints and numpy integer types
    ch_scalar = ChannelSpec("scalar", neuron_indices=5)
    assert ch_scalar.neuron_indices == (5,)
    ch_np32 = ChannelSpec("np32", neuron_indices=np.int32(7))
    assert ch_np32.neuron_indices == (7,)
    ch_np64 = ChannelSpec("np64", neuron_indices=np.int64(9))
    assert ch_np64.neuron_indices == (9,)
    ch_arr32 = ChannelSpec("arr32", neuron_indices=np.array([2, 4], dtype=np.int32))
    assert ch_arr32.neuron_indices == (2, 4)
    ch_arr64 = ChannelSpec("arr64", neuron_indices=np.array([10, 12], dtype=np.int64))
    assert ch_arr64.neuron_indices == (10, 12)

    # Negative or non-finite gain
    with pytest.raises(ValueError, match="gain"):
        ChannelSpec("test", neuron_indices=[1], gain=-10.0)
    with pytest.raises(ValueError, match="gain"):
        ChannelSpec("test", neuron_indices=[1], gain=float("nan"))
    with pytest.raises(ValueError, match="gain"):
        ChannelSpec("test", neuron_indices=[1], gain=float("inf"))

    # Invalid signal ranges
    with pytest.raises(ValueError, match="cannot exceed max_signal"):
        ChannelSpec("test", neuron_indices=[1], min_signal=2.0, max_signal=1.0)
    with pytest.raises(ValueError, match="min_signal"):
        ChannelSpec("test", neuron_indices=[1], min_signal=float("nan"))
    with pytest.raises(ValueError, match="max_signal"):
        ChannelSpec("test", neuron_indices=[1], max_signal=float("inf"))


def test_adapter_initialization_and_n_neurons_bounds():
    """Verify n_neurons validation and neuron index boundary checks."""
    # Invalid n_neurons
    for bad_n in (0, -5, True, False, 1.5, np.bool_(True)):
        with pytest.raises((ValueError, TypeError)):
            ExternalSensoryAdapter(n_neurons=bad_n)

    # Valid adapter
    adapter = ExternalSensoryAdapter(n_neurons=100)
    assert adapter.n_neurons == 100

    # Adding channel with neuron_index >= n_neurons must fail
    bad_ch = ChannelSpec("out_of_bounds", neuron_indices=[99, 100])
    with pytest.raises(ValueError, match="exceeds connectome size"):
        adapter.add_channel(bad_ch)

    # Valid channel addition
    good_ch = ChannelSpec("valid", neuron_indices=[0, 50, 99])
    adapter.add_channel(good_ch)
    assert "valid" in adapter.channels

    # Duplicate channel registration must fail
    with pytest.raises(ValueError, match="already registered"):
        adapter.add_channel(good_ch)


def test_opt_in_enabled_disabled():
    """Disabled adapter produces NO drive (empty arrays), avoiding bias/random activity."""
    ch = ChannelSpec("ch1", neuron_indices=[5], gain=50.0)
    adapter = ExternalSensoryAdapter(n_neurons=20, channels=[ch], enabled=False)

    # When disabled, returns empty arrays regardless of signal
    idx, rates = adapter.process_signals({"ch1": 1.0})
    assert len(idx) == 0
    assert len(rates) == 0

    drive_dict = adapter.as_drive_dict({"ch1": 1.0})
    assert len(drive_dict["drive_idx"]) == 0
    assert len(drive_dict["drive_rate_hz"]) == 0
    assert len(drive_dict["drive_weight"]) == 0

    part_idx, part_rates, part_w = adapter.as_drive_part({"ch1": 1.0})
    assert len(part_idx) == 0
    assert len(part_rates) == 0
    assert part_w == 400.0

    # Re-enabling adapter produces expected drive
    adapter.enable()
    assert adapter.enabled
    idx, rates = adapter.process_signals({"ch1": 1.0})
    assert len(idx) == 1
    assert idx[0] == 5
    assert rates[0] == 50.0

    # Disabling again
    adapter.disable()
    assert not adapter.enabled
    idx, rates = adapter.process_signals({"ch1": 1.0})
    assert len(idx) == 0


def test_no_signal_and_missing_signal_policies():
    """Missing or absent signals produce NO drive under safe policies."""
    ch1 = ChannelSpec("temp", neuron_indices=[3], gain=20.0)
    ch2 = ChannelSpec("humidity", neuron_indices=[4], gain=30.0)

    # 1. missing_policy='skip' (default)
    adapter = ExternalSensoryAdapter(n_neurons=10, channels=[ch1, ch2], missing_policy="skip")
    # None signals -> NO drive
    idx, rates = adapter.process_signals(None)
    assert len(idx) == 0
    assert len(rates) == 0

    # Empty dict -> NO drive
    idx, rates = adapter.process_signals({})
    assert len(idx) == 0
    assert len(rates) == 0

    # Only temp supplied -> humidity skipped without error or baseline activity
    idx, rates = adapter.process_signals({"temp": 0.5})
    assert list(idx) == [3]
    assert list(rates) == [10.0]
    # Verify missing channel 'humidity' (neuron 4) contributes no baseline or entry
    assert 4 not in idx

    # Truly unmapped signals must fail clearly under default unmapped_policy='error'
    with pytest.raises(KeyError, match="Unmapped sensory signals"):
        adapter.process_signals({"temp": 0.5, "unmapped_sensor": 1.0})

    # 2. missing_policy='error'
    adapter_strict = ExternalSensoryAdapter(n_neurons=10, channels=[ch1, ch2], missing_policy="error")
    with pytest.raises(KeyError, match="Missing required sensory signals"):
        adapter_strict.process_signals({"temp": 0.5})

    with pytest.raises(KeyError, match="Signals mapping is None"):
        adapter_strict.process_signals(None)

    # 3. missing_policy='zero'
    adapter_zero = ExternalSensoryAdapter(n_neurons=10, channels=[ch1, ch2], missing_policy="zero")
    # Missing ch2 evaluates at 0.0 -> rate 0.0 -> no drive produced for ch2
    idx, rates = adapter_zero.process_signals({"temp": 0.5})
    assert list(idx) == [3]
    assert list(rates) == [10.0]
    assert 4 not in idx


def test_unmapped_signals_policy():
    """Unmapped signals must fail clearly under default policy, or ignore if opted-in."""
    ch = ChannelSpec("valid_ch", neuron_indices=[2], gain=10.0)

    # Default policy: error
    adapter = ExternalSensoryAdapter(n_neurons=10, channels=[ch], unmapped_policy="error")
    with pytest.raises(KeyError, match="Unmapped sensory signals"):
        adapter.process_signals({"valid_ch": 1.0, "unexpected_ch": 0.5})

    # Policy: ignore
    adapter_ignore = ExternalSensoryAdapter(n_neurons=10, channels=[ch], unmapped_policy="ignore")
    idx, rates = adapter_ignore.process_signals({"valid_ch": 1.0, "unexpected_ch": 0.5})
    assert list(idx) == [2]
    assert list(rates) == [10.0]


def test_nan_and_non_finite_signals():
    """Adapter rejects NaN and infinite values strictly."""
    ch = ChannelSpec("ch", neuron_indices=[1], gain=10.0)
    adapter = ExternalSensoryAdapter(n_neurons=10, channels=[ch])

    for bad_val in (float("nan"), float("inf"), float("-inf"), np.nan, np.inf):
        with pytest.raises(ValueError, match="non-finite"):
            adapter.process_signals({"ch": bad_val})

    # Booleans are rejected as non-numeric measurements
    for bad_bool in (True, False, np.bool_(True), np.bool_(False), [True], np.array(True)):
        with pytest.raises(TypeError, match="must be numeric"):
            adapter.process_signals({"ch": bad_bool})


def test_signal_bounds_and_clamping():
    """Test clamping and non-clamping boundary checks."""
    # Clamp = True
    ch_clamped = ChannelSpec("ch_c", neuron_indices=[1], gain=100.0, min_signal=0.0, max_signal=1.0, clamp=True)
    adapter_c = ExternalSensoryAdapter(n_neurons=10, channels=[ch_clamped])

    # Value above max_signal is clamped to 1.0 -> rate 100.0
    _, rates = adapter_c.process_signals({"ch_c": 1.5})
    assert rates[0] == 100.0

    # Value below min_signal is clamped to 0.0 -> rate 0.0 -> no drive
    idx, _ = adapter_c.process_signals({"ch_c": -0.5})
    assert len(idx) == 0

    # Clamp = False
    ch_unclamped = ChannelSpec("ch_u", neuron_indices=[1], gain=100.0, min_signal=0.0, max_signal=1.0, clamp=False)
    adapter_u = ExternalSensoryAdapter(n_neurons=10, channels=[ch_unclamped])

    with pytest.raises(ValueError, match="outside configured range"):
        adapter_u.process_signals({"ch_u": 1.5})

    with pytest.raises(ValueError, match="outside configured range"):
        adapter_u.process_signals({"ch_u": -0.1})


def test_threshold_behavior():
    """Signals at or below threshold produce zero drive."""
    ch = ChannelSpec("ch", neuron_indices=[2], gain=100.0, threshold=0.2)
    adapter = ExternalSensoryAdapter(n_neurons=10, channels=[ch])

    # Signal <= threshold -> zero rate -> no drive entries
    idx, _ = adapter.process_signals({"ch": 0.15})
    assert len(idx) == 0
    idx, _ = adapter.process_signals({"ch": 0.20})
    assert len(idx) == 0

    # Signal > threshold -> (0.25 - 0.20) * 100.0 = 5.0 Hz
    idx, rates = adapter.process_signals({"ch": 0.25})
    assert list(idx) == [2]
    assert np.isclose(rates[0], 5.0)


def test_input_immutability():
    """Caller input dictionaries and arrays must never be mutated."""
    indices = np.array([1, 2, 3], dtype=np.int64)
    indices_orig = indices.copy()
    ch = ChannelSpec("immutable", neuron_indices=indices, gain=50.0)
    assert np.array_equal(indices, indices_orig)

    adapter = ExternalSensoryAdapter(n_neurons=10, channels=[ch])
    signals = {"immutable": 0.8}
    signals_orig = dict(signals)

    adapter.process_signals(signals)
    adapter.as_drive_part(signals)
    adapter.as_drive_dict(signals)

    assert signals == signals_orig


def test_direction_independent_deterministic_mapping():
    """Order of channel definitions and signal dict keys does not affect drive output.

    Sorting target neuron indices in canonical ascending order ensures bitwise
    deterministic and direction-independent mappings.
    """
    ch_left = ChannelSpec("sensor_left", neuron_indices=[10], gain=100.0)
    ch_right = ChannelSpec("sensor_right", neuron_indices=[5], gain=100.0)

    # Order A: left then right
    adapter_a = ExternalSensoryAdapter(n_neurons=20, channels=[ch_left, ch_right])
    # Order B: right then left
    adapter_b = ExternalSensoryAdapter(n_neurons=20, channels=[ch_right, ch_left])

    sig_order1 = {"sensor_left": 0.6, "sensor_right": 0.4}
    sig_order2 = {"sensor_right": 0.4, "sensor_left": 0.6}

    out_a1 = adapter_a.process_signals(sig_order1)
    out_a2 = adapter_a.process_signals(sig_order2)
    out_b1 = adapter_b.process_signals(sig_order1)
    out_b2 = adapter_b.process_signals(sig_order2)

    # Ascending order: neuron 5 (rate 40), then neuron 10 (rate 60)
    expected_idx = np.array([5, 10], dtype=np.int64)
    expected_rates = np.array([40.0, 60.0], dtype=np.float64)

    for idx, rates in (out_a1, out_a2, out_b1, out_b2):
        np.testing.assert_array_equal(idx, expected_idx)
        np.testing.assert_allclose(rates, expected_rates)


def test_duplicate_aggregation_modes():
    """Multiple channels targeting the same neuron combine deterministically."""
    # Two distinct channels targeting the same neuron (neuron 7)
    ch_a = ChannelSpec("ch_a", neuron_indices=[7], gain=60.0)
    ch_b = ChannelSpec("ch_b", neuron_indices=[7], gain=40.0)

    # 1. Sum mode: 60 + 40 = 100 Hz
    adapter_sum = ExternalSensoryAdapter(n_neurons=20, channels=[ch_a, ch_b], aggregation_mode="sum")
    idx, rates = adapter_sum.process_signals({"ch_a": 1.0, "ch_b": 1.0})
    assert list(idx) == [7]
    assert np.isclose(rates[0], 100.0)

    # 2. Max mode: max(60, 40) = 60 Hz
    adapter_max = ExternalSensoryAdapter(n_neurons=20, channels=[ch_a, ch_b], aggregation_mode="max")
    idx, rates = adapter_max.process_signals({"ch_a": 1.0, "ch_b": 1.0})
    assert list(idx) == [7]
    assert np.isclose(rates[0], 60.0)

    # 3. Mean mode: (60 + 40) / 2 = 50 Hz
    adapter_mean = ExternalSensoryAdapter(n_neurons=20, channels=[ch_a, ch_b], aggregation_mode="mean")
    idx, rates = adapter_mean.process_signals({"ch_a": 1.0, "ch_b": 1.0})
    assert list(idx) == [7]
    assert np.isclose(rates[0], 50.0)


def test_single_channel_internal_duplicate_aggregation():
    """A channel with duplicate neuron indices in its own target list aggregates cleanly."""
    ch = ChannelSpec("dup_channel", neuron_indices=[4, 4, 8], gain=50.0)
    adapter = ExternalSensoryAdapter(n_neurons=20, channels=[ch], aggregation_mode="sum")

    # In sum mode, duplicate index 4 should aggregate 50 + 50 = 100 Hz
    idx, rates = adapter.process_signals({"dup_channel": 1.0})
    assert list(idx) == [4, 8]
    assert np.isclose(rates[0], 100.0)
    assert np.isclose(rates[1], 50.0)


def test_core_merge_drive_contract_compatibility():
    """Output from adapter integrates seamlessly with core `merge_drive`."""
    ch = ChannelSpec("wind", neuron_indices=[12, 15], gain=80.0)
    adapter = ExternalSensoryAdapter(n_neurons=50, channels=[ch], default_weight=350.0)

    # Part format
    part = adapter.as_drive_part({"wind": 0.5})
    assert isinstance(part, tuple)
    assert len(part) == 3
    assert part[0].dtype == np.int64
    assert part[1].dtype == np.float64
    assert part[2] == 350.0

    # Merging via core merge_drive
    merged = merge_drive([part])
    assert "drive_idx" in merged
    assert "drive_rate_hz" in merged
    assert "drive_weight" in merged
    np.testing.assert_array_equal(merged["drive_idx"], np.array([12, 15], dtype=np.int64))
    np.testing.assert_allclose(merged["drive_rate_hz"], np.array([40.0, 40.0], dtype=np.float64))
    np.testing.assert_allclose(merged["drive_weight"], np.array([350.0, 350.0], dtype=np.float64))

    # Test as_drive_dict direct output matches merge_drive schema
    direct_dict = adapter.as_drive_dict({"wind": 0.5})
    np.testing.assert_array_equal(direct_dict["drive_idx"], merged["drive_idx"])
    np.testing.assert_allclose(direct_dict["drive_rate_hz"], merged["drive_rate_hz"])
    np.testing.assert_allclose(direct_dict["drive_weight"], merged["drive_weight"])

    # Merging with a mock distinct sensory pathway succeeds
    mock_proprio_part = (np.array([1, 2, 3], dtype=np.int64), np.array([20.0, 25.0, 30.0]), 400.0)
    combined = merge_drive([mock_proprio_part, part])
    assert len(combined["drive_idx"]) == 5

    # Overlap with existing pathway triggers core merge_drive exception
    overlapping_part = (np.array([15, 20], dtype=np.int64), np.array([10.0, 10.0]), 400.0)
    with pytest.raises(ValueError, match="sensory pathways overlap"):
        merge_drive([overlapping_part, part])
