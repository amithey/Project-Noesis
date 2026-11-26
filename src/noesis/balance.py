"""Gravity and rotation senses for the connectome: Johnston's organ and the halteres.

Two organ systems, each driven by one body signal:

    Johnston's organ C/D/E/F   gravity direction in the body frame (antennal deflection)
    haltere afferents          body angular velocity (the halteres are gyroscopes)

FlyWire does not say which direction an individual neuron prefers, so every neuron gets a
fixed random preferred direction in 3-D (seeded, reproducible) and a rectified response:

    rate = base + (max - base) * max(0, tanh(preferred . signal / scale))

Gravity enters as the deviation of the body-frame "up" vector from upright (0, 0, 1), so an
upright fly drives these neurons at the base rate. All of this is a modelling assumption.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

UPRIGHT = np.array([0.0, 0.0, 1.0])
_JO_PREFIXES = tuple(f"johnstons_organ_{c}_neuron" for c in "CDEF")
_HALTERE = ("haltere_campaniform_sensillum_neuron", "haltere_chordotonal_organ_neuron",
            "haltere_orphan_neuron")


def _directions(n: int, rng: np.random.Generator) -> np.ndarray:
    d = rng.normal(size=(n, 3))
    return d / np.linalg.norm(d, axis=1, keepdims=True)


class BalanceMap:
    """Johnston's-organ and haltere afferents with random preferred directions."""

    def __init__(self, neurons: pd.DataFrame, base_hz: float = 2.0, max_hz: float = 150.0,
                 tilt_scale: float = 0.3, spin_scale: float = 3.0, seed: int = 0):
        self.base_hz, self.max_hz = base_hz, max_hz
        self.tilt_scale, self.spin_scale = tilt_scale, spin_scale
        sub = neurons["Sub Class"].fillna("")
        afferent = neurons["Flow"].eq("afferent")
        jo = np.flatnonzero((afferent & sub.isin(_JO_PREFIXES)).to_numpy())
        hal = np.flatnonzero((afferent & sub.isin(_HALTERE)).to_numpy())
        rng = np.random.default_rng(seed)
        self.jo, self.hal = jo, hal
        self.jo_dir, self.hal_dir = _directions(len(jo), rng), _directions(len(hal), rng)
        self.idx = np.concatenate([jo, hal])

    def drive(self, up_body: np.ndarray, omega: np.ndarray) -> np.ndarray:
        """Firing rate (Hz) for every neuron in ``self.idx`` (JO first, then haltere)."""
        tilt = np.asarray(up_body, float) - UPRIGHT
        g = np.maximum(0.0, np.tanh(self.jo_dir @ tilt / self.tilt_scale))
        s = np.maximum(0.0, np.tanh(self.hal_dir @ np.asarray(omega, float) / self.spin_scale))
        span = self.max_hz - self.base_hz
        return self.base_hz + span * np.concatenate([g, s])
