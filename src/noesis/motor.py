"""Map motor-neuron activity (FlyWire muscle annotations) onto NeuroMechFly joint targets.

FlyWire labels every leg motor neuron with the muscle it drives (``Primary Cell Type``), the
leg (``Sub Class``) and the side (``Nerve``). Each muscle is assigned to one joint degree of
freedom with a sign (agonist +, antagonist -). This table is a *modelling assumption* from
leg anatomy, not something stored in the connectome; NeuroMechFly's sign conventions are
checked only by eye. Muscles not listed (e.g. the TTMn jump muscle) are ignored.

When wing (``joint_LWing*``, ``joint_RWing*``) or neck (``joint_Head*``) joints are included
in ``joint_names``, ``build_motor_matrix`` and ``build_muscle_matrices`` also map FlyWire's
``wing_motor_neuron`` (indirect power muscles DVM/DLM, direct steering muscles b/i/iii/hg,
and tension muscles tp/ps) and ``neck_motor_neuron`` (pitch, roll, yaw) onto those joints.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

DOFS = ("Coxa", "Coxa_roll", "Coxa_yaw", "Femur", "Femur_roll", "Tibia", "Tarsus1")
WING_JOINTS = (
    "joint_LWing_yaw", "joint_LWing", "joint_LWing_roll",
    "joint_RWing_yaw", "joint_RWing", "joint_RWing_roll",
)
HEAD_JOINTS = ("joint_Head_yaw", "joint_Head", "joint_Head_roll")

# muscle -> list of (joint dof, sign)
MUSCLE_JOINT: dict[str, list[tuple[str, int]]] = {
    "sternal_anterior_rotator": [("Coxa", +1), ("Coxa_yaw", +1)],
    "sternal_posterior_rotator": [("Coxa", -1), ("Coxa_yaw", -1)],
    "tergopleural_promotor": [("Coxa", +1)],
    "pleural_remotor_abductor": [("Coxa", -1), ("Coxa_roll", +1)],
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

# Wing motor neurons -> list of (wing dof suffix, sign)
# Power muscles: DVM (dorsoventral) elevates (+1), DLM (dorsolongitudinal) depresses (-1).
# Steering muscles: basalare (b1-b3) & first/third axillary (i1-i2, iii1-iii4) vs hg1-hg4 & ps/tp.
WING_MUSCLE_JOINT: dict[str, list[tuple[str, int]]] = {
    "DVM1a-c": [("", +1)],
    "DVM2a-b": [("", +1)],
    "DVM3a-b": [("", +1)],
    "DLM1-4": [("", -1)],
    "DLM5": [("", -1)],
    "b1": [("_yaw", +1), ("_roll", +1)],
    "b2": [("_yaw", +1), ("_roll", +1)],
    "b3": [("_yaw", +1)],
    "i1": [("_yaw", +1), ("", +1)],
    "i2": [("_yaw", +1), ("", +1)],
    "iii1": [("_yaw", -1)],
    "iii3": [("_yaw", -1)],
    "iii4": [("_yaw", -1)],
    "hg1": [("_roll", -1), ("", -1)],
    "hg2": [("_roll", -1)],
    "hg3": [("_roll", -1)],
    "hg4": [("_roll", -1), ("", -1)],
    "ps1": [("_roll", +1)],
    "ps2": [("_roll", +1)],
    "PSn_u": [("_roll", +1)],
    "tp1": [("_yaw", -1)],
    "tp2": [("_yaw", -1)],
    "tpn": [("_yaw", -1)],
}

_LEG = {"front": "F", "middle": "M", "hind": "H"}


def _nerve_or_soma_side(df: pd.DataFrame) -> pd.Series:
    nerve_side = df["Nerve"].fillna("").str.extract(r"^(left|right)_")[0]
    return nerve_side.fillna(df["Soma side"]).str[0].str.upper()


def leg_motor_table(neurons: pd.DataFrame) -> pd.DataFrame:
    """Leg motor neurons with columns ``idx`` (matrix index), ``leg`` ('LF', 'RM', ...), ``muscle``."""
    m = neurons[neurons["Class"].eq("leg_motor_neuron")].copy()
    side = _nerve_or_soma_side(m)
    leg = m["Sub Class"].str.replace("_leg_motor_neuron", "").map(_LEG)
    out = pd.DataFrame(
        {"idx": m.index.to_numpy(), "leg": (side + leg).to_numpy(), "muscle": m["Primary Cell Type"]}
    )
    return out.dropna().reset_index(drop=True)


def wing_motor_table(neurons: pd.DataFrame) -> pd.DataFrame:
    """Wing motor neurons with columns ``idx``, ``side`` ('L' or 'R'), ``muscle``."""
    if "Class" not in neurons:
        return pd.DataFrame(columns=["idx", "side", "muscle"])
    m = neurons[neurons["Class"].eq("wing_motor_neuron")].copy()
    if m.empty:
        return pd.DataFrame(columns=["idx", "side", "muscle"])
    side = _nerve_or_soma_side(m)
    out = pd.DataFrame(
        {"idx": m.index.to_numpy(), "side": side.to_numpy(), "muscle": m["Primary Cell Type"]}
    )
    return out.dropna().reset_index(drop=True)


def neck_motor_table(neurons: pd.DataFrame) -> pd.DataFrame:
    """Neck motor neurons with columns ``idx``, ``side`` ('L' or 'R'), ``subclass``, ``muscle``."""
    if "Class" not in neurons:
        return pd.DataFrame(columns=["idx", "side", "subclass", "muscle"])
    m = neurons[neurons["Class"].eq("neck_motor_neuron")].copy()
    if m.empty:
        return pd.DataFrame(columns=["idx", "side", "subclass", "muscle"])
    side = _nerve_or_soma_side(m)
    out = pd.DataFrame({
        "idx": m.index.to_numpy(),
        "side": side.to_numpy(),
        "subclass": m["Sub Class"].fillna("neck_unclassified"),
        "muscle": m["Primary Cell Type"].fillna(""),
    })
    return out.dropna(subset=["idx", "side"]).reset_index(drop=True)


def _extra_joint_entries(neurons: pd.DataFrame, col: dict[str, int]) -> list[tuple[int, int, str, int]]:
    """Return (joint_col, neuron_idx, group_key, sign) for wing and neck motor neurons."""
    entries: list[tuple[int, int, str, int]] = []
    if any(k.startswith(("joint_LWing", "joint_RWing")) for k in col):
        wtable = wing_motor_table(neurons)
        for r in wtable.itertuples():
            for suffix, sign in WING_MUSCLE_JOINT.get(r.muscle, ()):
                j = col.get(f"joint_{r.side}Wing{suffix}")
                if j is not None:
                    entries.append((j, int(r.idx), f"{r.side}Wing_{r.muscle}", sign))
    if any(k.startswith("joint_Head") for k in col):
        ntable = neck_motor_table(neurons)
        for r in ntable.itertuples():
            side_sign = +1 if r.side == "L" else -1
            if r.subclass == "neck_yaw_motor_neuron":
                j = col.get("joint_Head_yaw")
                if j is not None:
                    entries.append((j, int(r.idx), f"Head_yaw_{r.side}", side_sign))
            elif r.subclass == "neck_roll_motor_neuron":
                j = col.get("joint_Head_roll")
                if j is not None:
                    entries.append((j, int(r.idx), f"Head_roll_{r.side}", side_sign))
            elif r.subclass == "neck_roll_pitch_motor_neuron":
                for jname, sg in (("joint_Head_roll", side_sign), ("joint_Head", +1)):
                    j = col.get(jname)
                    if j is not None:
                        entries.append((j, int(r.idx), f"{jname}_{r.muscle}", sg))
            elif r.subclass == "neck_pitch_motor_neuron":
                j = col.get("joint_Head")
                if j is not None:
                    pitch_sign = +1 if r.muscle in ("CvN4", "CvN5") else -1
                    entries.append((j, int(r.idx), f"Head_pitch_{r.muscle}", pitch_sign))
            else:
                j = col.get("joint_Head")
                if j is not None:
                    pitch_sign = +1 if r.muscle.startswith("CvN") else -1
                    entries.append((j, int(r.idx), f"Head_unclass_{r.muscle}", pitch_sign))
    return entries


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
    extra = _extra_joint_entries(neurons, col)
    if extra:
        extra_counts: dict[tuple[int, str], int] = {}
        for j, _, gkey, _ in extra:
            extra_counts[(j, gkey)] = extra_counts.get((j, gkey), 0) + 1
        for j, idx, gkey, sign in extra:
            rows.append(j)
            cols.append(idx)
            vals.append(sign / extra_counts[(j, gkey)])
    return sparse.csr_matrix((vals, (rows, cols)), shape=(len(joint_names), len(neurons)))


def build_muscle_matrices(
    neurons: pd.DataFrame, joint_names: list[str]
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """Agonist and antagonist pools per joint: each (n_joints x n_neurons), rows are pool means.

    ``Gpos @ rates`` is the mean firing rate of the neurons that pull a joint in the positive
    direction, ``Gneg @ rates`` of those that pull it negative; a joint with only one pool has an
    all-zero row in the other matrix.
    """
    col = {name: k for k, name in enumerate(joint_names)}
    table = leg_motor_table(neurons)
    pools: dict[tuple[int, int], list[int]] = {}
    for r in table.itertuples():
        for dof, sign in MUSCLE_JOINT.get(r.muscle, ()):
            j = col.get(f"joint_{r.leg}{dof}")
            if j is not None:
                pools.setdefault((j, sign), []).append(r.idx)
    for j, idx, _, sign in _extra_joint_entries(neurons, col):
        pools.setdefault((j, sign), []).append(idx)
    n = len(neurons)
    mats = {}
    for sign in (+1, -1):
        rows, cols, vals = [], [], []
        for (j, sg), idx in pools.items():
            if sg == sign:
                rows += [j] * len(idx)
                cols += idx
                vals += [1.0 / len(idx)] * len(idx)
        mats[sign] = sparse.csr_matrix((vals, (rows, cols)), shape=(len(joint_names), n))
    return mats[+1], mats[-1]


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
