"""Type-aware proprioception hypotheses; no gait clock or motor commands.

Hook movement selectivity: Mamiya et al. 2018, doi:10.1016/j.neuron.2018.09.009.
Opposing hook/claw circuit effects: MANC annotation, Figure 59,
https://elifesciences.org/reviewed-preprints/97766.
The preferred signs below are INFERRED from a resistance-reflex hypothesis,
not direct physiological labels. The reversed assignment is an explicit control.
Unknown types retain the legacy transfer; connectivity and neuron order stay fixed.
"""
from __future__ import annotations

import numpy as np

from noesis.sensory import SensoryMap

ENCODINGS = ('legacy', 'typed-resistance', 'typed-reversed')
# Positive model Tibia coordinates/velocity mean flexion (physical FK tests).
TUNING = {'SNpp50': ('claw', 1), 'SNpp51': ('claw', -1),
          'SNpp39': ('hook', -1), 'SNpp41': ('hook', 1)}


class TypedSensoryMap:
    """Preserve afferent identities/order; replace only annotated tuning signals.

    Claw types receive signed joint position, hook types rectified signed velocity.
    Reversal changes all four inferred preferences, without changing anatomy.
    Rate curves/scales are assumptions. All six leg states must be supplied.
    """

    def __init__(self, neurons, encoding='typed-resistance', base_hz=2., max_hz=150.,
                 vel0=2., load0=2., kinds=None):
        if encoding not in ENCODINGS[1:]:
            raise ValueError('unknown typed proprioceptive encoding')
        self.encoding = encoding
        self._legacy = SensoryMap(neurons, base_hz, max_hz, vel0, load0)
        self.base_hz, self.max_hz, self.vel0 = base_hz, max_hz, vel0
        n = len(self._legacy.idx)
        legs, old_kinds = np.empty(n, object), np.empty(n, object)
        positions = {int(idx): pos for pos, idx in enumerate(self._legacy.idx)}
        for kind, leg, ids in self._legacy.groups:
            slots = np.array([positions[int(idx)] for idx in ids])
            legs[slots], old_kinds[slots] = leg, kind
        primary = (neurons['Primary Cell Type'].reindex(self._legacy.idx).fillna('').to_numpy()
                   if 'Primary Cell Type' in neurons else np.full(n, '', object))
        effective_kinds, preferences = old_kinds.copy(), np.zeros(n, int)
        polarity = 1 if encoding == 'typed-resistance' else -1
        for cell, (kind, preference) in TUNING.items():
            selected = primary == cell
            effective_kinds[selected] = kind
            preferences[selected] = polarity * preference
        keep = np.ones(n, bool) if kinds is None else np.isin(effective_kinds, list(kinds))
        self._keep = np.flatnonzero(keep)
        self.idx = self._legacy.idx[keep].copy()
        legs, primary = legs[keep], primary[keep]
        effective_kinds, preferences = effective_kinds[keep], preferences[keep]
        self.groups = [(kind, leg, self.idx[(effective_kinds == kind) & (legs == leg)])
                       for kind, leg in sorted(set(zip(effective_kinds, legs, strict=True)))]
        self._tuning = []
        for cell in TUNING:
            for leg in sorted(set(legs[primary == cell])):
                slots = np.flatnonzero((primary == cell) & (legs == leg))
                self._tuning.append((cell, leg, TUNING[cell][0],
                                     polarity * TUNING[cell][1], slots))
        self._coverage = {
            'encoding': encoding, 'preferred_directions_are_inferred': True,
            'mapped_neurons': len(self.idx), 'typed_neurons': int(np.count_nonzero(preferences)),
            'unchanged_legacy_neurons': int(np.count_nonzero(preferences == 0)),
            'unresolved_hook_neurons': int(np.count_nonzero((effective_kinds == 'hook') & (preferences == 0))),
            'types': {cell: int(np.count_nonzero(primary == cell)) for cell in TUNING},
        }

    def drive(self, state):
        """Only actual joint state determines input; stationary hooks return base."""
        rates = self._legacy.drive(state)[self._keep].copy()
        for _cell, leg, kind, preference, slots in self._tuning:
            s = state[leg]
            if kind == 'hook':
                signal = np.tanh(max(0., preference * s.tibia_vel) / self.vel0)
            else:
                signal = 1. / (1. + np.exp(-preference * s.tibia / .3))
            rates[slots] = self.base_hz + (self.max_hz - self.base_hz) * signal
        return rates

    def coverage(self):
        return {**self._coverage, 'types': dict(self._coverage['types'])}


def make_proprioception_map(neurons, encoding='legacy'):
    if encoding == 'legacy':
        return SensoryMap(neurons)
    return TypedSensoryMap(neurons, encoding)
