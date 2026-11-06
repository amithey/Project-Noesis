"""One bounded antithetic synaptic proposal; never produces joint commands."""
import hashlib
from pathlib import Path

import numpy as np


def propose(weights, reference, target_slots, direction, epsilon=.02):
    w, r = np.asarray(weights, float), np.asarray(reference, float)
    slots, d = np.asarray(target_slots), np.asarray(direction)
    if (w.ndim != 1 or r.shape != w.shape or slots.shape != w.shape or slots.dtype.kind not in 'iu'
            or d.ndim != 1 or not len(d) or not np.isin(d, [-1, 1]).all()
            or (slots < 0).any() or (slots >= len(d)).any()
            or not np.isfinite(w).all() or not np.isfinite(r).all()
            or not np.isfinite(epsilon) or not 0 < epsilon < .5):
        raise ValueError('Invalid proposal context')
    validate_weights(w, r)
    with np.errstate(over='ignore', invalid='ignore'):
        offset = epsilon * np.abs(r) * d[slots]
        plus, minus = w + offset, w - offset
    validate_weights(plus, r)
    validate_weights(minus, r)
    return plus, minus


def validate_weights(weights, reference):
    if (not np.isfinite(weights).all() or not np.array_equal(np.sign(weights), np.sign(reference))
            or (np.abs(weights) < .5*np.abs(reference)).any()
            or (np.abs(weights) > 1.5*np.abs(reference)).any()):
        raise ValueError('Refuse sign/bound/nonfinite violation; no clipping')


def write_candidate(parent, output, weights):
    parent, output = Path(parent), Path(output)
    before = hashlib.sha256(parent.read_bytes()).hexdigest()
    with np.load(parent, allow_pickle=False) as z:
        content = {key: z[key].copy() for key in z.files}
    weights = np.asarray(weights, dtype=content['weights'].dtype)
    if weights.shape != content['weights'].shape:
        raise ValueError('Candidate shape differs')
    validate_weights(weights, content['initial_weights'])
    content['weights'] = weights.copy()
    for key in ('activity', 'eligibility', 'reward_baseline', 'reward_variance', 'reward_normalization_mass'):
        if key in content:
            content[key] = np.zeros_like(content[key])
    with output.open('xb') as handle:
        np.savez_compressed(handle, **content)
    assert hashlib.sha256(parent.read_bytes()).hexdigest() == before


def assess(parent, candidate, seeds):
    if set(parent) != set(candidate) or set(parent) != set(seeds):
        raise ValueError('Missing/extra seed context')
    scalar = ('standing_fraction', 'mean_stance_slip_mm_s', 'heading_forward_distance_mm',
              'net_forward_displacement_mm', 'total_leg_steps')
    if any(not np.isfinite(row[key]) for bank in (parent, candidate) for row in bank.values() for key in scalar):
        raise ValueError('Nonfinite summary')
    heading = float(np.median([candidate[s]['heading_forward_distance_mm']-parent[s]['heading_forward_distance_mm'] for s in seeds]))
    net = float(np.median([candidate[s]['net_forward_displacement_mm']-parent[s]['net_forward_displacement_mm'] for s in seeds]))
    forward = (np.median([candidate[s]['heading_forward_distance_mm'] for s in seeds]) > 0 and
               np.median([candidate[s]['net_forward_displacement_mm'] for s in seeds]) > 0)
    standing = all(candidate[s]['standing_fraction'] >= .9 and
                   candidate[s]['standing_fraction'] >= parent[s]['standing_fraction'] for s in seeds)
    slip = all(candidate[s]['mean_stance_slip_mm_s'] <= parent[s]['mean_stance_slip_mm_s'] for s in seeds)
    steps = np.median([candidate[s]['total_leg_steps'] for s in seeds]) >= np.median([parent[s]['total_leg_steps'] for s in seeds])
    tripod = np.median([candidate[s]['tripod_coordination']['tripod_coordination_score'] for s in seeds]) >= np.median([parent[s]['tripod_coordination']['tripod_coordination_score'] for s in seeds])
    return dict(passed=bool(standing and slip and heading >= .25 and net >= .25 and forward and steps and tripod),
                standing_retained=standing, slip_retained=slip, cycles_retained=bool(steps), tripod_retained=bool(tripod),
                positive_forward_progress=bool(forward),
                median_heading_gain_mm=heading, median_net_forward_gain_mm=net,
                ranking=min(heading, net), seeds=list(seeds), walking_proven=False, model_promoted=False)
