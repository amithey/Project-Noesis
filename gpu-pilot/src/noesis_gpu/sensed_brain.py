"""Proposal: drive one extra sensory modality (thermo) into a ``FrozenBrain`` through the ``_extra_drive_parts`` hook.

Needs the 5-line controller hook in docs/workstreams/claude/senses/hook-proposal/controller-extra-drive-parts.patch (a Codex file; NOT applied here).
The stimulus is an experimenter-defined field, not physics: ``GradientField`` maps the body position from the EXISTING observation to a signed
temperature deviation u in [-1, 1] (warm channel = max(u, 0), cool channel = max(-u, 0)). Hz per unit signal (gain) is an explicit modelling
assumption of the caller. Afferents are selected from the connectome annotation by ``senses_bridge`` and must not overlap the already wired
leg and balance afferents. Modes (matched controls, same code path and same neurons):
``off`` adapter disabled; ``zero`` channels enabled, signal 0; ``constant`` signal fixed at ``level`` regardless of position; ``active`` the field.
No motor policy, no learning, no reward change.
"""

from __future__ import annotations

import numpy as np

from .external_senses import ExternalSensoryAdapter
from .senses_bridge import build_channels

MODES = ("off", "zero", "constant", "active")
THERMO = ("thermo_warm", "thermo_cool")


class GradientField:
    """u(position) = clip((position[axis] - origin) / length, -1, 1), in the same units as ``observation['position']`` (unit not verified here)."""

    def __init__(self, axis=0, origin=0.0, length=1.0):
        if axis not in (0, 1, 2) or not np.isfinite(origin) or not np.isfinite(length) or length <= 0:
            raise ValueError("axis must be 0..2, origin finite, length finite and positive")
        self.axis, self.origin, self.length = int(axis), float(origin), float(length)

    def __call__(self, observation):
        return float(np.clip((float(np.asarray(observation["position"])[self.axis]) - self.origin) / self.length, -1.0, 1.0))


def thermo_signals(u):
    """Opponent split of a signed deviation into the two channel signals."""
    return {"thermo_warm": max(u, 0.0), "thermo_cool": max(-u, 0.0)}


def sensed_class(base):
    """Subclass of ``base`` (a FrozenBrain with the ``_extra_drive_parts`` hook) driving the thermo afferents."""
    if not hasattr(base, "_extra_drive_parts"):
        raise TypeError("controller lacks the _extra_drive_parts hook; apply controller-extra-drive-parts.patch to a copy first")

    class SensedBrain(base):
        def __init__(self, connectome, rest, lo, hi, *, field, gain, mode="active", level=0.5, **options):
            if mode not in MODES:
                raise ValueError(f"mode must be one of {MODES}")
            if not np.isfinite(gain) or gain < 0 or not np.isfinite(level) or not -1 <= level <= 1:
                raise ValueError("gain must be finite and >= 0; level in [-1, 1]")
            super().__init__(connectome, rest, lo, hi, **options)
            taken = np.union1d(self.senses.idx, self.balance.idx)
            channels = build_channels(connectome.neurons, {name: gain for name in THERMO}, taken=taken,
                                      min_signal=0.0, max_signal=1.0)
            self.thermo = ExternalSensoryAdapter(connectome.n, channels, default_weight=self.sensory_weight)
            if mode == "off":
                self.thermo.disable()
            self.mode, self.field, self.level = mode, field, float(level)
            self.thermo_idx = np.sort(np.concatenate([np.asarray(c.neuron_indices) for c in channels]))
            self.last_u = 0.0

        def _extra_drive_parts(self, observation):
            u = {"off": 0.0, "zero": 0.0, "constant": self.level, "active": self.field(observation)}[self.mode]
            self.last_u = u
            idx, rate, weight = self.thermo.as_drive_part(thermo_signals(u))
            return [(idx, rate, weight)] if len(idx) else []

    return SensedBrain
