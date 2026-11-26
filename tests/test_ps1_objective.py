import numpy as np
import pytest
from ps1_objective import physical_score


def score(reaction=(0,10,0,0),velocity=(0,0,0),omega=(0,0,0),up=1,feet=(1,1,1,0,0,0),weight=10):
    return physical_score(reaction,velocity,omega,up,feet,weight)


def test_static_supported_counterexamples_are_formula_contract_not_fly_success():
    assert score()['reward']==1
    assert score(reaction=(0,0,0,0))['reward']==-1
    assert score(reaction=(10,0,0,0))['reward']==-1
    assert score(reaction=(0,0,0,10))['reward']==-1
    assert score(up=.7)['reward']==-1
    assert score(feet=(0,0,0,0,0,0))['reward']==-1
    assert score(feet=(1,0,0,0,0,0))['reward']<score(feet=(1,1,0,0,0,0))['reward']<score()['reward']
    assert score(reaction=(0,200,0,0))['reward']<-.999
    assert score(reaction=(0,10,0,0))['reward']==score(reaction=(0,4,6,0))['reward']


def test_joint_motion_support_and_load_tradeoffs_are_not_hidden_by_one_proxy():
    stationary=score()['quality']
    assert stationary>score(velocity=(1,0,0))['quality']>score(velocity=(3,0,0))['quality']
    assert stationary>score(omega=(0,1,0))['quality']>score(omega=(0,3,0))['quality']
    assert stationary>score(up=.9)['quality']>score(up=.8)['quality']
    assert stationary>score(reaction=(5,5,0,0))['quality']>score(reaction=(9,1,0,0))['quality']
    assert stationary>score(reaction=(0,20,0,0))['quality']
    # No thorax-height field exists, so raising/lowering it alone cannot change feedback.
    assert score(reaction=(0,100,0,0),weight=100)['reward']==score()['reward']


def test_batch_scalar_identity_and_readonly_input():
    r=np.array([[0,10,0,0],[4,6,0,0],[0,0,0,0]],float);v=np.arange(9).reshape(3,3)/3
    w=np.zeros((3,3));u=np.array([1,.8,.7]);f=np.ones((3,6));before=[x.copy() for x in (r,v,w,u,f)]
    actual=physical_score(r,v,w,u,f,10)
    for i in range(3):
        scalar=physical_score(r[i],v[i],w[i],u[i],f[i],10)
        for k in scalar:assert actual[k][i]==scalar[k]
    for a,b in zip((r,v,w,u,f),before):np.testing.assert_array_equal(a,b)
    assert ((actual['reward']>=-1)&(actual['reward']<=1)).all()


@pytest.mark.parametrize('kwargs',[{'reaction':(-1,10,0,0)},{'weight':0},{'weight':float('nan')},{'up':1.1},
    {'velocity':(0,float('nan'),0)},{'feet':(-1,1,1,0,0,0)},{'reaction':(0,1,0)},{'omega':(0,0)},
    {'feet':(1,1)},{'up':[1]}])
def test_invalid_features_stop_instead_of_becoming_favorable_feedback(kwargs):
    with pytest.raises(ValueError):score(**kwargs)
