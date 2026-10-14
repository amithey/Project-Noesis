"""Map actual world observations onto annotated sensory neurons.

Odor/taste/thermal identities come from CSV Function labels. Spatial retinotopy,
R7->blue/R8->green assignments, response curves and gains are explicit assumptions;
the connectome has no measured receptive fields or complete UV spectral model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from noesis.habitat import ODORS, TASTES, Habitat
from noesis.sensory import LEGS


class EnvironmentSenses:
    def __init__(self, neurons: pd.DataFrame, retina, seed=0):
        self.groups = []
        self.retina = retina
        self.last_rates = np.zeros(0)
        rng = np.random.default_rng(seed)
        sub = neurons['Sub Class'].fillna('')
        function = neurons['Function'].fillna('')
        labels = {token: function.str.contains(fr'(?:^|,){token}(?:,|$)', regex=True)
                  for token in (*ODORS, *TASTES, 'heating', 'cooling', 'humid', 'dry')}
        afferent = neurons['Flow'].eq('afferent')
        side = neurons['Nerve'].fillna('').str.extract(r'^(left|right)_')[0]
        side = side.fillna(neurons['Soma side'])
        self.visual = []
        for eye, label in enumerate(('left', 'right')):
            for cell, channel in (('R7', 1), ('R8', 0)):
                mask = afferent & neurons['Class'].eq('photoreceptor_neuron') & side.eq(label)
                mask &= neurons['Primary Cell Type'].fillna('').str.startswith(cell)
                indices = np.flatnonzero(mask.to_numpy())
                pixels = np.flatnonzero(retina.pale_type_mask == channel)
                if len(indices) and not len(pixels):
                    raise ValueError('retina has no pixels for a required colour channel')
                assigned = np.resize(rng.permutation(pixels), len(indices))
                self.visual.append((indices, eye, assigned, channel))
        for sensor_side, label in enumerate(('left', 'right')):
            for organ, sensor_offset in (('antenna', 2), ('maxillary_palp', 0)):
                mask = afferent & neurons['Class'].eq('olfactory_receptor_neuron') & side.eq(label)
                mask &= sub.str.startswith(organ + '_')
                for odor, name in enumerate(ODORS):
                    ids = np.flatnonzero((mask & labels[name]).to_numpy())
                    self.groups.append(('odor', ids, (odor, sensor_offset + sensor_side)))
        # Leg/labelled mouth taste, not internal pharyngeal input without swallowing.
        for location in (*LEGS, 'mouth'):
            if location == 'mouth':
                loc_mask = sub.str.startswith('labellum_')
            else:
                leg_name = {'F': 'front', 'M': 'middle', 'H': 'hind'}[location[1]]
                loc_mask = sub.str.startswith(leg_name + '_leg_') & side.eq(
                    'left' if location[0] == 'left' or location[0] == 'L' else 'right')
            mask = afferent & neurons['Class'].fillna('').str.contains('gustatory') & loc_mask
            for taste, name in enumerate(TASTES):
                ids = np.flatnonzero((mask & labels[name]).to_numpy())
                self.groups.append(('taste', ids, (location, taste)))
        for token, kind in [('heating', 'heat'), ('cooling', 'cool'),
                            ('humid', 'humid'), ('dry', 'dry')]:
            category = 'thermosensory_receptor_neuron' if kind in ('heat', 'cool') else 'hygrosensory_receptor_neuron'
            mask = afferent & neurons['Class'].eq(category)
            mask &= labels[token]
            self.groups.append((kind, np.flatnonzero(mask.to_numpy()), None))
        sound_mask = afferent & sub.isin(('johnstons_organ_A_neuron', 'johnstons_organ_B_neuron'))
        self.groups.append(('sound', np.flatnonzero(sound_mask.to_numpy()), None))
        arrays = [g[0] for g in self.visual] + [g[1] for g in self.groups]
        self.idx = np.concatenate(arrays).astype(np.int64)
        if len(np.unique(self.idx)) != len(self.idx):
            raise ValueError('environmental sensory groups overlap')

    def coverage(self):
        result = {'vision': sum(len(g[0]) for g in self.visual)}
        for kind, ids, _ in self.groups:
            result[kind] = result.get(kind, 0) + len(ids)
        return result

    def drive(self, obs, habitat: Habitat, mouth_position, time_s):
        vision = np.asarray(obs['vision'], float)
        odors = np.asarray(obs['odor_intensity'], float)
        feet = np.asarray(obs['end_effectors'], float)
        if vision.shape != (2, len(self.retina.pale_type_mask), 2):
            raise ValueError('unexpected binocular retina shape')
        if odors.shape != (len(ODORS), 4) or feet.shape != (6, 3):
            raise ValueError('unexpected odor/foot sensor shape')
        if not np.isfinite(vision).all() or not np.isfinite(odors).all():
            raise ValueError('world sensor readings must be finite')
        conditions = habitat.local_conditions(mouth_position, time_s)
        # Missing contact evidence means no taste, even directly above food.
        taste = np.asarray(obs.get('surface_taste', np.zeros((7, len(TASTES)))), float)
        if taste.shape != (7, len(TASTES)) or not np.isfinite(taste).all():
            raise ValueError('unexpected physical taste contacts')
        activation = [np.clip(vision[eye, pixel, channel], 0, 1)
                      for _, eye, pixel, channel in self.visual]
        for kind, ids, source in self.groups:
            if kind == 'odor':
                value = odors[source]
            elif kind == 'taste':
                location, channel = source
                value = taste[6 if location == 'mouth' else LEGS.index(location), channel]
            elif kind == 'heat':
                value = (conditions['temperature_c'] - 24) / 10
            elif kind == 'cool':
                value = (24 - conditions['temperature_c']) / 10
            elif kind == 'humid':
                value = (conditions['humidity'] - 0.45) / 0.4
            elif kind == 'dry':
                value = (0.45 - conditions['humidity']) / 0.4
            else:
                value = conditions['sound_envelope']
            activation.append(np.full(len(ids), np.clip(value, 0, 1)))
        self.last_rates = 2.0 + 118.0 * np.concatenate(activation)
        telemetry = {**conditions, 'vision_mean': float(vision.mean()),
                     'odor_food': float(odors[0].mean()), 'odor_aversive': float(odors[4].mean()),
                     'taste_sugar': float(taste[:, 0].max()), 'taste_water': float(taste[:, 1].max()),
                     'taste_bitter': float(taste[:, 2].max()),
                     'environment_driven_neurons': int(np.count_nonzero(self.last_rates > 2.01))}
        return self.last_rates, telemetry


def mouth_position(sim, fly):
    """Actual world position of the proboscis, not the fly's centre."""
    return np.asarray(sim.physics.named.data.xpos[f'{fly.name}/Haustellum']).copy()


def contact_tastes(sim, fly, habitat):
    """Taste only at force-bearing tarsal/proboscis contacts with edible surfaces."""
    import mujoco

    model, data = sim.physics.model.ptr, sim.physics.data.ptr
    taste = np.zeros((7, len(TASTES)))
    force = np.zeros(6)
    direct = {'fruit_piece': 'sugar', 'patch_fruit': 'sugar',
              'patch_water': 'water', 'patch_bitter': 'bitter'}
    prefix = fly.name + '/'
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        first = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)) or ''
        second = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)) or ''
        if first.startswith(prefix) and not second.startswith(prefix):
            part, surface = first[len(prefix):], second
        elif second.startswith(prefix) and not first.startswith(prefix):
            part, surface = second[len(prefix):], first
        else:
            continue
        if part[:2] in LEGS and part[2:].startswith('Tarsus'):
            location = LEGS.index(part[:2])
        elif part in ('Haustellum', 'Rostrum'):
            location = 6
        else:
            continue
        mujoco.mj_contactForce(model, data, contact_index, force)
        if force[0] <= 1e-8:
            continue
        if surface in direct:
            taste[location, TASTES.index(direct[surface])] = 1
        elif surface == 'ground':
            taste[location] = np.maximum(taste[location], habitat.tastes_at(contact.pos[None])[0])
    return taste


def merge_drive(parts):
    """Combine (indices, rates, weight) without duplicating overlapping neurons."""
    if not parts:
        return {}
    idx = np.concatenate([np.asarray(p[0], np.int64) for p in parts])
    if len(np.unique(idx)) != len(idx):
        raise ValueError('sensory pathways overlap')
    return {'drive_idx': idx, 'drive_rate_hz': np.concatenate([p[1] for p in parts]),
            'drive_weight': np.concatenate([np.full(len(p[0]), p[2]) for p in parts])}
