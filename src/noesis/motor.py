"""Map leg motor-neuron activity (FlyWire muscle annotations) onto NeuroMechFly joint targets.

FlyWire labels every leg motor neuron with the muscle it drives (``Primary Cell Type``), the
leg (``Sub Class``) and the side (``Nerve``). Each muscle is assigned to one joint degree of
freedom with a sign (agonist +, antagonist -). This table is a *modelling assumption* from
leg anatomy, not something stored in the connectome; NeuroMechFly's sign conventions are
checked only by eye. Muscles not listed (e.g. the TTMn jump muscle) are ignored.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

DOFS = ("Coxa", "Coxa_roll", "Coxa_yaw", "Femur", "Femur_roll", "Tibia", "Tarsus1")

# muscle -> list of (joint dof, sign)
MUSCLE_JOINT: dict[str, list[tuple[str, int]]] = {
    "sternal_anterior_rotator": [("Coxa", +1)],
    "sternal_posterior_rotator": [("Coxa", -1)],
    "tergopleural_promotor": [("Coxa", +1)],
    "pleural_remotor_abductor": [("Coxa", -1)],
    "sternal_adductor": [("Coxa_roll", -1)],
    "trochanter_flexor": [("Femur", +1)],
    "accessory_trochanter_flexor": [("Femur", +1)],
    "trochanter_extensor": [("Femur", -1)],
    "tergotrochanter": [("Femur", -1)],
    "sternotrochanter": [("Femur", -1)],
    "femur_reductor": [("Femur_roll", +1)],
    "tibia_flexor": [("Tibia", +1)],
    "accessory_tibia_flexor": [("Tibia", +1)],
    "tibia_extensor": [("Tibia", -1)],
    "tarsus_depressor": [("Tarsus1", +1)],
    "long_tendon_muscle": [("Tarsus1", +1)],
    "tarsus_levator": [("Tarsus1", -1)],
}

_LEG = {"front": "F", "middle": "M", "hind": "H"}


def leg_motor_table(neurons: pd.DataFrame) -> pd.DataFrame:
    """Leg motor neurons with columns ``idx`` (matrix index), ``leg`` ('LF', 'RM', ...), ``muscle``."""
    m = neurons[neurons["Class"].eq("leg_motor_neuron")].copy()
    nerve_side = m["Nerve"].fillna("").str.extract(r"^(left|right)_")[0]
    side = nerve_side.fillna(m["Soma side"]).str[0].str.upper()
    leg = m["Sub Class"].str.replace("_leg_motor_neuron", "").map(_LEG)
    out = pd.DataFrame(
        {"idx": m.index.to_numpy(), "leg": (side + leg).to_numpy(), "muscle": m["Primary Cell Type"]}
    )
    return out.dropna().reset_index(drop=True)


def build_motor_matrix(neurons: pd.DataFrame, joint_names: list[str]) -> sparse.csr_matrix:
    """(n_joints x n_neurons) matrix; ``G @ rates_hz`` = mean agonist minus antagonist rate."""
    col = {name: k for k, name in enumerate(joint_names)}
    table = leg_motor_table(neurons)
    sizes = table.groupby(["leg", "muscle"]).size()
    rows, cols, vals = [], [], []
    for r in table.itertuples():
        for dof, sign in MUSCLE_JOINT.get(r.muscle, ()):
            j = col.get(f"joint_{r.leg}{dof}")
            if j is not None:
                rows.append(j)
                cols.append(r.idx)
                vals.append(sign / sizes[(r.leg, r.muscle)])
    return sparse.csr_matrix((vals, (rows, cols)), shape=(len(joint_names), len(neurons)))


class RateTracker:
    """Exponentially smoothed per-neuron firing rate (Hz) over a chosen set of neurons."""

    def __init__(self, n: int, tau_ms: float = 50.0):
        self.rate = np.zeros(n)
        self.tau = tau_ms

    def update(self, spike_idx: np.ndarray, window_ms: float) -> np.ndarray:
        counts = np.bincount(spike_idx, minlength=self.rate.size)
        self.rate *= np.exp(-window_ms / self.tau)
        self.rate += counts / (self.tau * 1e-3)
        return self.rate
