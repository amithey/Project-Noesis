"""Staged reward hypothesis: old objectives exact, new credit remains grounded."""
import hashlib
import json
from itertools import pairwise

import numpy as np
import pytest

from noesis import embodied_metrics as NEW

LEGACY_OBJECTIVES = ('standing-first-linear-slip-v3', 'standing-gate-linear-slip-v4',
                     'standing-gate-linear-slip-v5')
V6 = 'standing-gate-smooth-grip-v6'
V5 = 'standing-gate-linear-slip-v5'


def reward(speed=10., slip=10., *, support=6, up=.9, height=1., objective=V6):
    loads = np.zeros(6)
    loads[:support] = .1
    feet = np.zeros((6, 3))
    feet[:, 0] = slip
    return NEW.locomotion_reward([0, 0, height], [.005 * speed, 0, height], up,
                                 np.zeros(42), .005, [1, 0, 0], loads, feet,
                                 objective=objective)


def test_all_existing_objectives_are_bit_exact():
    # Canonical outputs frozen from the unmodified V28/CF8515 reward source,
    # before creating this objective. All output fields participate in the hash.
    expected = 'c18e44ff145d5aae24054e7eba429d45aaf0b9c535ef85cf0fa65eed7321f8fd'
    rng = np.random.default_rng(148)
    results = []
    for objective in LEGACY_OBJECTIVES:
        for _ in range(120):
            previous = rng.normal(size=3)
            position = previous + rng.normal(scale=.05, size=3)
            position[2] = rng.uniform(.1, 2.)
            args = (previous, position, rng.uniform(-1, 1), rng.normal(size=42), .005,
                    rng.normal(size=3), rng.choice([0., .01], size=6),
                    rng.normal(scale=15, size=(6, 3)), rng.uniform(0, 2))
            saved = [x.copy() for x in args if isinstance(x, np.ndarray)]
            results.append(NEW.locomotion_reward(*args, objective=objective))
            for a, b in zip(saved, [x for x in args if isinstance(x, np.ndarray)], strict=True):
                np.testing.assert_array_equal(a, b)
    canonical = json.dumps(results, sort_keys=True, separators=(',', ':'), allow_nan=False)
    assert hashlib.sha256(canonical.encode()).hexdigest() == expected
    assert NEW.GROUNDED_OBJECTIVES[:3] == LEGACY_OBJECTIVES


def test_partial_sliding_keeps_directional_credit_and_costs_more_slip():
    assert reward(slip=15, objective=V5)['grounded_progress'] == 0
    assert reward(slip=15)['grounded_progress'] > 0
    # Keep the speed low enough that the total reward is below its upper clip.
    samples = [reward(speed=2, slip=s) for s in (0, 1, 5, 10, 15, 20)]
    assert samples[0]['grip_factor'] == 1
    assert all(a['grounded_progress'] > b['grounded_progress'] for a, b in pairwise(samples))
    assert all(a['reward'] > b['reward'] for a, b in pairwise(samples))
    assert reward(speed=-10, slip=15)['grounded_progress'] < 0


@pytest.mark.parametrize('kwargs', [{'support': 0}, {'support': 2}, {'up': .69},
                                   {'height': .29}, {'height': 1.51}, {'speed': 30.01}])
def test_progress_requires_existing_posture_support_and_speed_gates(kwargs):
    assert reward(**kwargs)['grounded_progress'] == 0


def test_new_objective_changes_only_grip_and_progress_contribution():
    old, new = reward(slip=8, objective=V5), reward(slip=8)
    for key in old.keys() - {'reward', 'grip_factor', 'grounded_progress'}:
        assert old[key] == new[key]
    assert new['reward'] - old['reward'] == pytest.approx(.5 * (new['grounded_progress'] - old['grounded_progress']))
    assert reward(up=.1)['reward'] < 0
    assert reward(slip=10000)['reward'] == -1
    assert NEW.GROUNDED_OBJECTIVES[0] == LEGACY_OBJECTIVES[0]
