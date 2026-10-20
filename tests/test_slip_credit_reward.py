"""Stronger scalar slip credit, with unchanged physical measurements and old rewards."""
import json

import numpy as np
import pytest

from noesis.embodied_metrics import GROUNDED_OBJECTIVES, locomotion_reward
from noesis.experiments import learning_lab as lab
from test_learning_lab import lab_environment as lab_environment  # noqa: PLC0414 -- pytest fixture re-export

V4 = 'standing-gate-linear-slip-v4'
V5 = 'standing-gate-linear-slip-v5'


def stationary(*, feet=3, slip=0., up=.9, height=.8, objective=V5):
    loads = np.zeros(6)
    loads[:feet] = 1.
    return locomotion_reward([0,0,height],[0,0,height],up,np.zeros(3),.005,
        [1,0,0],loads,np.tile([slip,0,0],(6,1)),objective=objective)


@pytest.mark.parametrize('slip', [0., .1, 1., 5., 10., 17., 30.])
def test_only_linear_slip_slope_changes_fivefold(slip):
    assert GROUNDED_OBJECTIVES[:2] == ('standing-first-linear-slip-v3',V4)
    assert GROUNDED_OBJECTIVES[2] == V5
    old = stationary(slip=slip,objective=V4)
    new = stationary(slip=slip)
    assert old['reward'] == pytest.approx(.8-.006*slip,abs=1e-14)
    assert new['reward'] == pytest.approx(.8-.03*slip,abs=1e-14)
    assert new['reward']-old['reward'] == pytest.approx(-.024*slip,abs=1e-14)
    assert {k:v for k,v in old.items() if k!='reward'} == {k:v for k,v in new.items() if k!='reward'}


def test_every_physical_measurement_remains_exact_on_random_body_states():
    rng = np.random.default_rng(112)
    for _ in range(300):
        prev = np.array([0.,0.,.8])
        args = (prev,prev+rng.uniform(-.1,.1,3),rng.uniform(-1,1),
            rng.uniform(-20,20,42),.005,rng.normal(size=3),
            rng.uniform(0,1,6)*(rng.random(6)>.4),rng.uniform(-100,100,(6,3)),rng.uniform(0,1.3))
        a = locomotion_reward(*args,objective=V4)
        b = locomotion_reward(*args,objective=V5)
        assert {k:v for k,v in a.items() if k!='reward'} == {k:v for k,v in b.items() if k!='reward'}
        assert -1 <= b['reward'] <= 1


@pytest.mark.parametrize('slip', [9.7,14.6,16.8,20.])
def test_observed_standing_slip_range_still_beats_sustained_fall(slip):
    stable = stationary(slip=slip)['reward']
    fallen = stationary(up=-1.,height=.2,feet=0)['reward']
    assert stable > 0 and fallen < 0 and stable > fallen


def test_standing_time_is_not_traded_for_one_extra_loaded_foot():
    stable = stationary(feet=2,slip=15.)['reward']
    risky = stationary(feet=3,slip=14.)['reward']
    fallen = stationary(up=-1.,height=.2,feet=0)['reward']
    assert .9*stable+.1*fallen > .8*risky+.2*fallen


@pytest.mark.parametrize('kwargs', [{'up':.69},{'height':.29},{'height':1.51},{'feet':0}])
def test_no_standing_credit_without_posture_height_and_ground(kwargs):
    assert stationary(**kwargs)['reward'] < 0


def test_linear_bursts_and_total_clipping_do_not_hide_the_measured_slip():
    assert (stationary(slip=0.)['reward']+stationary(slip=10.)['reward'])/2 == pytest.approx(stationary(slip=5.)['reward'])
    extreme = stationary(slip=10000.)
    assert extreme['reward'] == -1 and extreme['stance_slip_mm_s'] == 10000.


def test_v4_import_to_v5_is_explicit_and_frozen_statistics_reset(lab_environment):
    env = lab_environment
    args = env.args
    args.learning_rule = 'perturbation'
    args.normalize_reward_error = args.bias_correct_reward_error = True
    args.learning_rate = .2
    args.reward_mode = 'grounded'
    args.grounded_objective = V4
    args.motor_exploration = 1.5
    _,old = lab.run_episode(env.cn,args,0,None)
    with np.load(old) as bank:
        assert bank['version'] == 4 and float(bank['reward_normalization_mass']) > 0
        previous = bank['weights'].copy()
    args.grounded_objective = V5
    args.learn = False
    args.motor_exploration = 0.
    with pytest.raises(ValueError,match='model context mismatch'):
        lab.run_episode(env.cn,args,1,old)
    args.import_weights = True
    summary,new = lab.run_episode(env.cn,args,2,old)
    assert summary['reward_objective'] == V5 and not summary['learning']
    context = json.loads(new.with_suffix('.model.json').read_text())
    assert context['schema'] == 8 and context['reward_objective'] == V5
    with np.load(new) as bank:
        assert bank['version'] == 4
        np.testing.assert_array_equal(bank['weights'],previous)
        for key in ('activity','eligibility','reward_baseline','reward_variance','reward_normalization_mass'):
            assert not bank[key].any()
    assert all(sim.closed for sim in env.simulations)
