"""Turn body state into drive for the leg sensory neurons of the connectome.

FlyWire annotates leg afferents by organ (``Sub Class``) and leg/side (``Nerve``). Each organ
class is driven by one simple body signal:

    claw chordotonal   femur-tibia flexion        (Mamiya et al. 2018: claw = flexion)
    hook chordotonal   femur-tibia extension      (hook = extension)
    club chordotonal   tibia movement speed       (club = vibration/movement)
    hair plate         coxa + femur displacement  (joint-limit sensors)
    campaniform        leg load (floor contact force)
    bristle            leg touch (floor contact force)

The signal-to-rate transfer functions are modelling assumptions, not data. Each sensory
neuron is driven by a Poisson process at the resulting rate (one event ~ one spike).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

LEGS = ("LF", "LM", "LH", "RF", "RM", "RH")
_LEG = {"front": "F", "middle": "M", "hind": "H"}
_KIND = {
    "claw_chordotonal_organ_neuron": "claw",
    "hook_chordotonal_organ_neuron": "hook",
    "club_chordotonal_organ_neuron": "club",
    "hair_plate_neuron": "hairplate",
    "campaniform_sensillum_neuron": "campaniform",
    "bilateral_campaniform_sensillum_neuron": "campaniform",
    "bristle_neuron": "bristle",
}


@dataclass(frozen=True)
class LegState:
    """Per-leg body signals (angles are offsets from the rest pose, rad)."""

    tibia: float = 0.0
    tibia_vel: float = 0.0
    coxa: float = 0.0
    femur: float = 0.0
    load: float = 0.0  # summed contact force of the leg's segments


def _signal(kind: str, s: LegState, vel0: float, load0: float) -> float:
    """Normalised 0..1 activation of an organ class."""
    if kind == "claw":
        return float(1.0 / (1.0 + np.exp(-s.tibia / 0.3)))
    if kind == "hook":
        return float(1.0 / (1.0 + np.exp(s.tibia / 0.3)))
    if kind == "club":
        return float(np.tanh(abs(s.tibia_vel) / vel0))
    if kind == "hairplate":
        return float(np.tanh((abs(s.coxa) + abs(s.femur)) / 0.5))
    return float(np.tanh(s.load / load0))  # campaniform, bristle


class SensoryMap:
    """Leg sensory neurons grouped by (organ kind, leg); ``drive`` returns their rates."""

    def __init__(self, neurons: pd.DataFrame, base_hz: float = 2.0, max_hz: float = 150.0,
                 vel0: float = 2.0, load0: float = 2.0):
        self.base_hz, self.max_hz, self.vel0, self.load0 = base_hz, max_hz, vel0, load0
        a = neurons[neurons["Flow"].eq("afferent") & neurons["Sub Class"].fillna("").str.contains("_leg_")]
        sub = a["Sub Class"].str.split("_leg_", n=1, expand=True)
        kind = sub[1].map(_KIND)
        side = a["Nerve"].fillna("").str.extract(r"^(left|right)_")[0].fillna(a["Soma side"])
        leg = side.str[0].str.upper() + sub[0].map(_LEG)
        t = pd.DataFrame({"idx": a.index.to_numpy(), "kind": kind, "leg": leg}).dropna()
        self.groups: list[tuple[str, str, np.ndarray]] = [
            (k, lg, g["idx"].to_numpy()) for (k, lg), g in t.groupby(["kind", "leg"])
        ]
        self.idx = np.concatenate([g[2] for g in self.groups])
        self._bounds = np.cumsum([0] + [len(g[2]) for g in self.groups])

    def drive(self, state: dict[str, LegState]) -> np.ndarray:
        """Firing rate (Hz) for every neuron in ``self.idx``."""
        rates = np.empty(len(self.idx))
        for k, (kind, leg, _) in enumerate(self.groups):
            sig = _signal(kind, state[leg], self.vel0, self.load0)
            rates[self._bounds[k]:self._bounds[k + 1]] = self.base_hz + (self.max_hz - self.base_hz) * sig
        return rates
