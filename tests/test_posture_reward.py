import importlib.util
from pathlib import Path

import numpy as np
import pytest

from noesis.embodied_metrics import locomotion_reward


@pytest.mark.parametrize('up', [.7, .8, 1.])
@pytest.mark.parametrize('slip', [0., 16., 60., 100.])
def test_stationary_supported_posture_beats_unsupported_fall_in_observed_quiet_range(up, slip):
    args=([0,0,.8],[0,0,.8],up,np.array([10.,-10.]),.005,[1,0,0])
    standing=locomotion_reward(*args,np.ones(6),np.tile([slip,0,0],(6,1)))
    fallen=locomotion_reward([0,0,.2],[0,0,.2],-1.,np.zeros(2),.005,[1,0,0],np.zeros(6),np.zeros((6,3)))
    assert standing['reward']>fallen['reward']


def test_penalty_remains_linear_for_equal_mean_slip_and_strictly_orders_sliding():
    args=([0,0,.8],[0,0,.8],1.,np.zeros(3),.005,[1,0,0],np.ones(6))
    def reward(slip):
        return locomotion_reward(*args,np.tile([slip,0,0],(6,1)))['reward']
    assert (reward(3.)+reward(7.))/2==pytest.approx(reward(5.))
    values=[reward(s) for s in (0.,.1,3.,7.,16.,30.,60.,100.)]
    assert np.all(np.diff(values)<0)


def test_candidate_changes_only_reward_on_random_physical_samples():
    spec=importlib.util.spec_from_file_location('unchanged_reward',Path(__file__).with_name('support_slip_v2_reference.py'))
    unchanged=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(unchanged)
    rng=np.random.default_rng(81)
    for _ in range(1000):
        previous=np.array([0.,0.,.8]);position=previous+rng.uniform(-.1,.1,size=3)
        args=(previous,position,rng.uniform(-1,1),rng.uniform(-20,20,size=42),.005,
              rng.normal(size=3),rng.uniform(0,1,size=6)*(rng.random(6)>.4),
              rng.uniform(-100,100,size=(6,3)),rng.uniform(0,1.3))
        reference=unchanged.locomotion_reward(*args)
        candidate=locomotion_reward(*args)
        assert set(reference)==set(candidate)
        assert {k:v for k,v in reference.items() if k!='reward'}=={k:v for k,v in candidate.items() if k!='reward'}
        assert -1<=candidate['reward']<=1
