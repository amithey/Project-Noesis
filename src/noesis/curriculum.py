"""Training-only curriculum decisions; no motor commands or leg ordering."""
from __future__ import annotations

import math

STAGES = (
    ('standing', 'curriculum-standing-v7', 'quiet'),
    ('gripped-steps', 'curriculum-gripped-step-v8', 'quiet'),
    ('continuous-walking', 'standing-gate-smooth-grip-v6', 'quiet'),
    ('obstacles', 'standing-gate-smooth-grip-v6', 'obstacle'),
)
RESERVED_SEEDS = frozenset((10, 11, 200, 201, 202))


def phase_jobs(stage, round_index, episodes=6, *, fixed_objective=False):
    """Both comparison arms get identical scenes, seeds, probes and horizons."""
    if not 0 <= stage < len(STAGES) or not 0 <= round_index < 12 or episodes != 6:
        raise ValueError('curriculum uses four stages, at most twelve rounds, six births per round')
    _, objective, arena = STAGES[stage]
    if fixed_objective:
        objective = 'standing-gate-smooth-grip-v6'
    jobs = [{'id': f'train-{i}', 'seed': 1000 + 6 * round_index + i,
                 'model': 'candidate', 'learning': True, 'arena': arena} for i in range(episodes)]
    for model in ('original', 'candidate'):
        jobs += [{'id': f'probe-{model}-{i}', 'seed': 10000 + 2 * round_index + i,
                      'model': model, 'learning': False, 'arena': arena} for i in range(2)]
    if stage == 3:
        jobs += [{'id': f'probe-ablated-{i}', 'seed': 10000 + 2 * round_index + i,
                      'model': 'ablated', 'learning': False, 'arena': arena} for i in range(2)]
    for job in jobs:
        job['objective'] = objective
        if job['seed'] in RESERVED_SEEDS:
            raise ValueError('training probes must not use final evaluation seeds')
    return jobs


def _finite(row, key):
    value = row.get(key)
    return isinstance(value, (int, float)) and math.isfinite(value)


def phase_decision(stage, originals, candidates, ablated=()):
    """Intermediate gates never certify final gait or promote deployment weights."""
    if not 0 <= stage < len(STAGES):
        raise ValueError('unknown curriculum stage')
    checks = []
    if (len(originals) != 2 or len(candidates) != 2
            or len({r['seed'] for r in originals}) != 2
            or [r['seed'] for r in originals] != [r['seed'] for r in candidates]):
        return {'accepted': False, 'checks': [], 'reason': 'two matched training probes required'}
    for original, candidate in zip(originals, candidates, strict=True):
        rows = (original, candidate)
        complete = all(r.get('status') == 'completed' and r.get('simulated_seconds', 0) >= 5
                       and r.get('learning') is False
                       and r.get('motor_exploration_mv_ms') == 0
                       and r.get('training_fall_seconds') == 0 for r in rows)
        standing = _finite(candidate, 'standing_fraction') and candidate['standing_fraction'] >= .9
        check = {'seed': candidate['seed'], 'complete': complete, 'standing': standing}
        if stage >= 1:
            check['forward'] = (_finite(candidate, 'heading_forward_distance_mm')
                                and candidate['heading_forward_distance_mm'] >= (.25 if stage == 1 else 1.))
            cycles = candidate.get('leg_step_cycles', {})
            # Only the existing debounced ground-support/lift detector supplies these counts.
            check['cycles'] = sum(v >= (1 if stage == 1 else 2) for v in cycles.values()) >= (2 if stage == 1 else 4)
            check['slip'] = (_finite(candidate, 'mean_stance_slip_mm_s')
                             and _finite(original, 'mean_stance_slip_mm_s')
                             and candidate['mean_stance_slip_mm_s'] <= (
                                 10. if stage == 1 else .8 * original['mean_stance_slip_mm_s']))
        checks.append(check)
    if stage == 3:
        matched = len(ablated) == 2 and [r['seed'] for r in ablated] == [r['seed'] for r in candidates]
        for i, candidate in enumerate(candidates):
            other = ablated[i] if matched else {}
            checks[i]['obstacle'] = bool(
                other.get('status') == 'completed' and other.get('simulated_seconds', 0) >= 5
                and other.get('learning') is False and other.get('motor_exploration_mv_ms') == 0
                and other.get('training_fall_seconds') == 0
                and candidate.get('obstacle_observation_complete', False)
                and _finite(candidate, 'obstacle_response_seconds')
                and 0 <= candidate['obstacle_response_seconds'] <= 1
                and _finite(candidate, 'obstacle_contact_seconds')
                and _finite(other, 'obstacle_contact_seconds')
                and candidate['obstacle_contact_seconds'] < other['obstacle_contact_seconds'])
    accepted = all(all(v for k, v in check.items() if k != 'seed') for check in checks)
    return {'accepted': accepted, 'checks': checks, 'reason': 'training-only stage gate'}
