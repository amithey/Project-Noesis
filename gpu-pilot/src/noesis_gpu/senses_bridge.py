"""Select connectome afferents for sensory modalities that the pilot does not yet drive.

Only a selection layer: modality -> neuron indices from the connectome's own annotation columns (``Flow``, ``Class``,
``Sub Class``, ``Function``), never hard-coded ids, with a hard guarantee of no overlap with the already-wired leg and balance
afferents (``merge_drive`` raises on overlap). It builds ``ChannelSpec`` objects for ``ExternalSensoryAdapter`` and does nothing else:
no signal source (the FlyGym 2 backend produces none), no gain default (gains are the caller's explicit modelling assumption),
no motor policy. A modality with no matching neurons is an error, not an empty channel.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

from .external_senses import ChannelSpec

# modality -> (column, substring) rules, ALL must match (case-sensitive, taken from the FlyWire annotations)
MODALITIES: dict[str, tuple[tuple[str, str], ...]] = {
    "thermo_warm": (("Sub Class", "thermosensory_receptor"), ("Function", "heating")),
    "thermo_cool": (("Sub Class", "thermosensory_receptor"), ("Function", "cooling")),
    "hygro_humid": (("Sub Class", "hygrosensory_receptor"), ("Function", "humid")),
    "hygro_dry": (("Sub Class", "hygrosensory_receptor"), ("Function", "dry")),
    "sound_low": (("Sub Class", "johnstons_organ_A_neuron"),),
    "sound_high": (("Sub Class", "johnstons_organ_B_neuron"),),
    "olfaction": (("Class", "olfactory_receptor_neuron"),),
    "vision": (("Class", "photoreceptor_neuron"),),
}


def select_modality(neurons: pd.DataFrame, modality: str, taken: Iterable[int] = ()) -> np.ndarray:
    """Sorted positional indices of afferents of one modality, excluding ``taken``; raises if none or if any already taken."""
    if modality not in MODALITIES:
        raise KeyError(f"unknown modality {modality!r}; known: {sorted(MODALITIES)}")
    mask = neurons["Flow"].eq("afferent").to_numpy().copy()
    for column, token in MODALITIES[modality]:
        mask &= neurons[column].fillna("").str.contains(token, regex=False).to_numpy()
    idx = np.flatnonzero(mask).astype(np.int64)
    if not len(idx):
        raise ValueError(f"no afferents match modality {modality!r}")
    overlap = np.intersect1d(idx, np.fromiter(taken, dtype=np.int64))
    if len(overlap):
        raise ValueError(f"modality {modality!r} overlaps {len(overlap)} already-driven neurons")
    return idx


def build_channels(neurons: pd.DataFrame, gains: Mapping[str, float], taken: Iterable[int] = (),
                   **spec) -> list[ChannelSpec]:
    """One ``ChannelSpec`` per requested modality; ``gains`` (Hz per signal unit) must be given explicitly, modalities must be disjoint."""
    taken = np.asarray(sorted({int(i) for i in taken}), dtype=np.int64)
    channels, used = [], taken
    for modality, gain in gains.items():
        idx = select_modality(neurons, modality, used)
        channels.append(ChannelSpec(modality, idx, gain=float(gain), description="senses_bridge", **spec))
        used = np.union1d(used, idx)
    return channels
