"""Opt-in neural sensory-drive adapter for external signals.

Translates caller-provided external scalar sensor signals into Poisson drive
parameters (drive_idx, drive_rate_hz, drive_weight) compatible with Noesis's
core `merge_drive` contract and LIFNetwork execution.

This is an architectural transport adapter and integration contract:
- Does NOT assume or invent connectome labels or target neuron IDs.
- Does NOT implement motor policy, reflexes, gait generators, or synthetic sensors.
- All channel-to-neuron mappings, gains, and signal ranges are caller-specified.
- Opt-in: disabled or missing signals produce NO drive (zero target neurons driven,
  no baseline bias, no spontaneous activity).
- Guarantees input immutability and deterministic multi-channel aggregation onto
  unique neuron indices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

import numpy as np


@dataclass(frozen=True)
class ChannelSpec:
    """Specification of an external scalar sensory input channel.

    Attributes:
        name: Unique string identifier for the channel.
        neuron_indices: Non-negative integer indices of target neurons in the connectome.
        gain: Scaling factor converting input signal units to firing rate in Hz.
            Must be finite and >= 0.0. Units: Hz / signal unit.
        min_signal: Lower bound of expected input scalar range. Must be finite.
        max_signal: Upper bound of expected input scalar range. Must be finite, >= min_signal.
        clamp: If True, input signals are clamped to [min_signal, max_signal].
            If False, input signals outside this range raise ValueError.
        threshold: Minimum input signal required to evoke drive (default 0.0).
            Signals below threshold produce 0.0 Hz.
        description: Informational description of the sensor channel.
    """

    name: str
    neuron_indices: Tuple[int, ...]
    gain: float = 100.0
    min_signal: float = 0.0
    max_signal: float = 1.0
    clamp: bool = True
    threshold: float = 0.0
    description: str = ""

    def __init__(
        self,
        name: str,
        neuron_indices: Union[int, Sequence[int], np.ndarray],
        gain: float = 100.0,
        min_signal: float = 0.0,
        max_signal: float = 1.0,
        clamp: bool = True,
        threshold: float = 0.0,
        description: str = "",
    ):
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Channel name must be a non-empty string, got: {name!r}")

        # Reject booleans for neuron_indices before integer checks
        if isinstance(neuron_indices, (bool, np.bool_)):
            raise TypeError(
                f"Neuron index must be an integer, got boolean: {neuron_indices!r}"
            )

        # Normalize and validate neuron indices without silent truncation
        if isinstance(neuron_indices, (int, np.integer)):
            raw_items = [neuron_indices]
        elif isinstance(neuron_indices, np.ndarray):
            if neuron_indices.ndim > 1:
                raise ValueError(
                    f"neuron_indices array must be 1D, got ndim={neuron_indices.ndim} with shape {neuron_indices.shape}"
                )
            if neuron_indices.ndim == 0:
                raw_items = [neuron_indices.item()]
            else:
                raw_items = list(neuron_indices)
        elif isinstance(neuron_indices, (list, tuple)):
            raw_items = list(neuron_indices)
        else:
            raise TypeError(
                f"Neuron index must be an integer or sequence of integers, got: {type(neuron_indices)}"
            )

        indices_list: List[int] = []
        for raw_idx in raw_items:
            if isinstance(raw_idx, (bool, np.bool_)):
                raise TypeError(
                    f"Neuron index must be an integer, got boolean: {raw_idx!r}"
                )
            if not isinstance(raw_idx, (int, np.integer)):
                raise TypeError(
                    f"Neuron index must be an integer, got {type(raw_idx).__name__}: {raw_idx!r}"
                )
            idx_val = int(raw_idx)
            if idx_val < 0:
                raise ValueError(f"Neuron index must be non-negative, got {idx_val}")
            indices_list.append(idx_val)

        # Validate gain
        gain_val = float(gain)
        if not math.isfinite(gain_val) or gain_val < 0.0:
            raise ValueError(f"gain must be a finite non-negative float, got {gain!r}")

        # Validate signal bounds
        min_s = float(min_signal)
        max_s = float(max_signal)
        if not math.isfinite(min_s):
            raise ValueError(f"min_signal must be finite, got {min_signal!r}")
        if not math.isfinite(max_s):
            raise ValueError(f"max_signal must be finite, got {max_signal!r}")
        if min_s > max_s:
            raise ValueError(f"min_signal ({min_s}) cannot exceed max_signal ({max_s})")

        thresh = float(threshold)
        if not math.isfinite(thresh):
            raise ValueError(f"threshold must be finite, got {threshold!r}")

        object.__setattr__(self, "name", str(name))
        object.__setattr__(self, "neuron_indices", tuple(indices_list))
        object.__setattr__(self, "gain", gain_val)
        object.__setattr__(self, "min_signal", min_s)
        object.__setattr__(self, "max_signal", max_s)
        object.__setattr__(self, "clamp", bool(clamp))
        object.__setattr__(self, "threshold", thresh)
        object.__setattr__(self, "description", str(description))


class ExternalSensoryAdapter:
    """Opt-in neural sensory-drive adapter.

    Translates measured scalar environment signals into Poisson drive arrays
    targeting specified connectome neurons.

    Core contract compatibility:
    - `as_drive_part(signals)` returns `(drive_idx, drive_rate_hz, drive_weight)`
      ready for inclusion in `noesis.environment_senses.merge_drive(...)`.
    - `as_drive_dict(signals)` returns `{"drive_idx": ..., "drive_rate_hz": ..., "drive_weight": ...}`
      matching the exact dictionary schema returned by `merge_drive`.
    - When disabled or signals are missing, produces zero-length drive arrays,
      guaranteeing NO spontaneous bias or random activity.
    - Multiple channels targeting the same neuron combine deterministically
      (via `sum`, `max`, or `mean`) with unique sorted neuron indices.
    - Inputs are never mutated.
    """

    ALLOWED_AGGREGATIONS = ("sum", "max", "mean")
    ALLOWED_MISSING_POLICIES = ("skip", "error", "zero")
    ALLOWED_UNMAPPED_POLICIES = ("error", "ignore")

    def __init__(
        self,
        n_neurons: int,
        channels: Optional[Union[Sequence[ChannelSpec], Mapping[str, ChannelSpec]]] = None,
        default_weight: float = 400.0,
        aggregation_mode: str = "sum",
        missing_policy: str = "skip",
        unmapped_policy: str = "error",
        enabled: bool = True,
    ):
        """Initialize the external sensory adapter.

        Args:
            n_neurons: Total number of neurons in the target connectome. Must be > 0.
            channels: Optional sequence or dict of ChannelSpec instances.
            default_weight: Default synaptic conductance weight for drive (dimensionless, >= 0).
                Matches `sensory_weight=400.0` in FrozenBrain.
            aggregation_mode: Combination rule for duplicate target neurons ('sum', 'max', 'mean').
            missing_policy: Policy when a configured channel is missing from input signals:
                - 'skip': Produce no drive for that channel (safe default).
                - 'error': Raise KeyError.
                - 'zero': Evaluate with signal value 0.0.
            unmapped_policy: Policy when an input signal is not a configured channel:
                - 'error': Raise KeyError.
                - 'ignore': Silently ignore extra signals.
            enabled: Opt-in flag. If False, adapter produces empty drive unconditionally.
        """
        if (
            isinstance(n_neurons, (bool, np.bool_))
            or not isinstance(n_neurons, (int, np.integer))
            or n_neurons <= 0
        ):
            raise ValueError(f"n_neurons must be a positive integer, got: {n_neurons!r}")
        self._n_neurons = int(n_neurons)

        if isinstance(default_weight, (bool, np.bool_)):
            raise TypeError(f"default_weight must be numeric, got boolean: {default_weight!r}")
        weight_val = float(default_weight)
        if not math.isfinite(weight_val) or weight_val < 0.0:
            raise ValueError(f"default_weight must be a finite non-negative float, got: {default_weight!r}")
        self._default_weight = weight_val

        if aggregation_mode not in self.ALLOWED_AGGREGATIONS:
            raise ValueError(
                f"aggregation_mode must be one of {self.ALLOWED_AGGREGATIONS}, got: {aggregation_mode!r}"
            )
        self._aggregation_mode = aggregation_mode

        if missing_policy not in self.ALLOWED_MISSING_POLICIES:
            raise ValueError(
                f"missing_policy must be one of {self.ALLOWED_MISSING_POLICIES}, got: {missing_policy!r}"
            )
        self._missing_policy = missing_policy

        if unmapped_policy not in self.ALLOWED_UNMAPPED_POLICIES:
            raise ValueError(
                f"unmapped_policy must be one of {self.ALLOWED_UNMAPPED_POLICIES}, got: {unmapped_policy!r}"
            )
        self._unmapped_policy = unmapped_policy

        self._enabled = bool(enabled)
        self._channels: Dict[str, ChannelSpec] = {}

        if channels is not None:
            if isinstance(channels, Mapping):
                for ch_name, spec in channels.items():
                    if spec.name != ch_name:
                        raise ValueError(
                            f"Channel mapping key '{ch_name}' does not match spec.name '{spec.name}'"
                        )
                    self.add_channel(spec)
            elif isinstance(channels, Iterable):
                for spec in channels:
                    self.add_channel(spec)
            else:
                raise TypeError(f"channels must be a sequence or mapping of ChannelSpec, got {type(channels)}")

    @property
    def n_neurons(self) -> int:
        """Total number of neurons in the target connectome."""
        return self._n_neurons

    @property
    def enabled(self) -> bool:
        """Whether the adapter is currently active."""
        return self._enabled

    @property
    def default_weight(self) -> float:
        """Default synaptic conductance weight."""
        return self._default_weight

    @property
    def aggregation_mode(self) -> str:
        """Duplicate neuron aggregation rule ('sum', 'max', 'mean')."""
        return self._aggregation_mode

    @property
    def missing_policy(self) -> str:
        """Missing signal policy ('skip', 'error', 'zero')."""
        return self._missing_policy

    @property
    def unmapped_policy(self) -> str:
        """Unmapped signal policy ('error', 'ignore')."""
        return self._unmapped_policy

    @property
    def channels(self) -> Dict[str, ChannelSpec]:
        """Mapping of configured channels (read-only copy)."""
        return dict(self._channels)

    def enable(self) -> None:
        """Enable sensory drive generation."""
        self._enabled = True

    def disable(self) -> None:
        """Disable sensory drive generation (produces NO drive)."""
        self._enabled = False

    def set_enabled(self, enabled: bool) -> None:
        """Set opt-in enabled status."""
        self._enabled = bool(enabled)

    def add_channel(self, channel: ChannelSpec) -> None:
        """Register a new sensory channel.

        Args:
            channel: ChannelSpec instance to register.

        Raises:
            TypeError: If channel is not a ChannelSpec.
            ValueError: If channel name already exists or neuron indices exceed n_neurons.
        """
        if not isinstance(channel, ChannelSpec):
            raise TypeError(f"Expected ChannelSpec, got {type(channel)}")
        if channel.name in self._channels:
            raise ValueError(f"Channel '{channel.name}' is already registered")

        # Validate indices against n_neurons
        for idx in channel.neuron_indices:
            if idx >= self._n_neurons:
                raise ValueError(
                    f"Neuron index {idx} in channel '{channel.name}' exceeds connectome size n_neurons={self._n_neurons}"
                )

        self._channels[channel.name] = channel

    def remove_channel(self, name: str) -> None:
        """Remove a registered channel by name.

        Args:
            name: Channel identifier.
        """
        if name not in self._channels:
            raise KeyError(f"Channel '{name}' is not registered")
        del self._channels[name]

    def process_signals(
        self, signals: Optional[Mapping[str, Any]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert scalar sensory measurements into target neuron indices and firing rates.

        Guarantees:
        - Does NOT mutate `signals` or channel definitions.
        - Returns deterministically sorted unique indices.
        - When disabled, returns empty arrays.
        - When no signals drive neurons, returns empty arrays (NO drive, no bias).

        Args:
            signals: Optional mapping from channel names to scalar measurements.

        Returns:
            Tuple of (drive_idx, drive_rate_hz):
            - drive_idx: 1D np.ndarray of np.int64, unique neuron indices in ascending order.
            - drive_rate_hz: 1D np.ndarray of np.float64, firing rates in Hz.
        """
        empty_idx = np.empty(0, dtype=np.int64)
        empty_rates = np.empty(0, dtype=np.float64)

        if not self._enabled:
            return empty_idx, empty_rates

        if signals is None:
            if self._missing_policy == "error" and self._channels:
                raise KeyError(
                    f"Signals mapping is None, but {len(self._channels)} channels are configured with missing_policy='error'"
                )
            return empty_idx, empty_rates

        if not isinstance(signals, Mapping):
            raise TypeError(f"signals must be a Mapping[str, scalar], got {type(signals)}")

        # Check unmapped signals
        signal_keys = set(signals.keys())
        configured_keys = set(self._channels.keys())
        unmapped_keys = signal_keys - configured_keys
        if unmapped_keys and self._unmapped_policy == "error":
            raise KeyError(
                f"Unmapped sensory signals encountered: {sorted(unmapped_keys)}. "
                f"Configured channels: {sorted(configured_keys)}"
            )

        # Check missing signals
        missing_keys = configured_keys - signal_keys
        if missing_keys and self._missing_policy == "error":
            raise KeyError(
                f"Missing required sensory signals for channels: {sorted(missing_keys)}"
            )

        # Collect raw rates per neuron
        rates_per_neuron: Dict[int, List[float]] = {}

        for ch_name, ch in self._channels.items():
            if ch_name not in signals:
                if self._missing_policy == "skip":
                    continue
                elif self._missing_policy == "zero":
                    raw_val = 0.0
                else:
                    # 'error' was already caught above
                    continue
            else:
                raw_val = signals[ch_name]

            # Validate scalar measurement
            if isinstance(raw_val, (bool, np.bool_)) or (isinstance(raw_val, np.ndarray) and raw_val.dtype.kind == "b"):
                raise TypeError(
                    f"Signal for channel '{ch_name}' must be numeric scalar, got boolean: {raw_val!r}"
                )
            if isinstance(raw_val, (np.ndarray, list, tuple)):
                arr = np.asarray(raw_val)
                if arr.ndim != 0:
                    raise TypeError(
                        f"Signal for channel '{ch_name}' must be numeric scalar, got shape {arr.shape}"
                    )
                item = arr.item()
                if isinstance(item, (bool, np.bool_)):
                    raise TypeError(
                        f"Signal for channel '{ch_name}' must be numeric scalar, got boolean: {item!r}"
                    )
                scalar_val = float(item)
            elif isinstance(raw_val, (int, float, np.number)):
                scalar_val = float(raw_val)
            else:
                raise TypeError(
                    f"Signal for channel '{ch_name}' must be numeric scalar, got {type(raw_val)}"
                )

            if not math.isfinite(scalar_val):
                raise ValueError(
                    f"Signal for channel '{ch_name}' is non-finite: {scalar_val}"
                )

            # Check bounds and clamp
            if not ch.clamp:
                if scalar_val < ch.min_signal or scalar_val > ch.max_signal:
                    raise ValueError(
                        f"Signal {scalar_val} for channel '{ch_name}' is outside "
                        f"configured range [{ch.min_signal}, {ch.max_signal}] with clamp=False"
                    )
                clamped_val = scalar_val
            else:
                clamped_val = min(max(scalar_val, ch.min_signal), ch.max_signal)

            # Compute rate in Hz
            if clamped_val <= ch.threshold:
                rate_hz = 0.0
            else:
                rate_hz = float((clamped_val - ch.threshold) * ch.gain)

            if rate_hz <= 0.0:
                continue

            for target_idx in ch.neuron_indices:
                if target_idx not in rates_per_neuron:
                    rates_per_neuron[target_idx] = []
                rates_per_neuron[target_idx].append(rate_hz)

        if not rates_per_neuron:
            return empty_idx, empty_rates

        # Deterministic aggregation onto unique neuron indices
        sorted_indices = sorted(rates_per_neuron.keys())
        final_rates: List[float] = []

        for idx in sorted_indices:
            channel_rates = rates_per_neuron[idx]
            if self._aggregation_mode == "sum":
                agg = sum(channel_rates)
            elif self._aggregation_mode == "max":
                agg = max(channel_rates)
            elif self._aggregation_mode == "mean":
                agg = sum(channel_rates) / len(channel_rates)
            else:
                agg = sum(channel_rates)

            final_rates.append(agg)

        # Filter out quiescent rates (<= 0.0)
        active_pairs = [
            (idx, r) for idx, r in zip(sorted_indices, final_rates) if r > 0.0
        ]
        if not active_pairs:
            return empty_idx, empty_rates

        out_idx = np.array([p[0] for p in active_pairs], dtype=np.int64)
        out_rates = np.array([p[1] for p in active_pairs], dtype=np.float64)
        return out_idx, out_rates

    def as_drive_part(
        self,
        signals: Optional[Mapping[str, Any]] = None,
        weight: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Build a drive part tuple compatible with core `merge_drive(parts)`.

        Each part in `merge_drive` has the schema `(drive_idx, drive_rate_hz, weight)`.

        Args:
            signals: Optional sensory signal measurements.
            weight: Optional synaptic weight override. If None, uses default_weight.

        Returns:
            Tuple of:
            - drive_idx: 1D np.ndarray (int64) of unique target neuron indices.
            - drive_rate_hz: 1D np.ndarray (float64) of Poisson event rates in Hz.
            - weight: float synaptic weight.
        """
        if weight is not None:
            if isinstance(weight, (bool, np.bool_)):
                raise TypeError(f"weight must be numeric, got boolean: {weight!r}")
            w = float(weight)
        else:
            w = self._default_weight
        if not math.isfinite(w) or w < 0.0:
            raise ValueError(f"weight must be a finite non-negative float, got {weight!r}")

        drive_idx, drive_rate_hz = self.process_signals(signals)
        return drive_idx, drive_rate_hz, w

    def as_drive_dict(
        self,
        signals: Optional[Mapping[str, Any]] = None,
        weight: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        """Build a drive dictionary matching the exact schema returned by `merge_drive`.

        Schema:
        {
            'drive_idx': 1D np.ndarray[np.int64],
            'drive_rate_hz': 1D np.ndarray[np.float64],
            'drive_weight': 1D np.ndarray[np.float64],
        }

        Args:
            signals: Optional sensory signal measurements.
            weight: Optional synaptic weight override.

        Returns:
            Dictionary matching core merge_drive schema.
        """
        drive_idx, drive_rate_hz, w = self.as_drive_part(signals, weight)
        drive_weight = np.full(len(drive_idx), w, dtype=np.float64)
        return {
            "drive_idx": drive_idx,
            "drive_rate_hz": drive_rate_hz,
            "drive_weight": drive_weight,
        }
