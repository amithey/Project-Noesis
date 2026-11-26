import copy
import numpy as np
import pytest
from ew1_update import (TRAIN_SEEDS,DIRECTIONS,PROBE_STEP,UPDATE_STEP,
                        fixed_directions,apply_delta,estimate_update,training_panel)


def parent():
    return dict(weights=np.array([1.1,-1.8,0.]),initial_weights=np.array([1.,-2.,0.]),
                motor_indices=np.array([2,5]),topology=np.array('same-graph'),eligibility=np.array([.1,.2,.3]))


def test_only_existing_weights_change_and_parent_is_not_mutated():
    cp=parent();saved=copy.deepcopy(cp);delta=np.array([.01,-.01,.01])
    plus,minus=apply_delta(cp,delta),apply_delta(cp,-delta)
    np.testing.assert_allclose(plus['weights']-cp['weights'],delta*np.abs(cp['initial_weights']),rtol=0,atol=1e-15)
    np.testing.assert_allclose(minus['weights']-cp['weights'],-delta*np.abs(cp['initial_weights']),rtol=0,atol=1e-15)
    assert plus['weights'][2]==minus['weights'][2]==0
    for k in cp:
        np.testing.assert_array_equal(cp[k],saved[k])
        if k!='weights':np.testing.assert_array_equal(plus[k],cp[k])


@pytest.mark.parametrize('delta',[[.5,0.,0.],[-.7,0.,0.],[np.nan,0.,0.],[0.,0.],['x','y','z']])
def test_invalid_update_rejects_without_clipping(delta):
    with pytest.raises(ValueError):apply_delta(parent(),np.asarray(delta))


def test_directions_reproduce_without_touching_global_rng():
    np.random.seed(123);before=np.random.get_state()
    first,second=fixed_directions(50),fixed_directions(50)
    np.testing.assert_array_equal(first,second);assert first.shape==(4,50)
    after=np.random.get_state()
    assert before[0]==after[0] and before[2:]==after[2:]
    np.testing.assert_array_equal(before[1],after[1])


def test_linear_objective_recovers_known_gradient_in_orthogonal_fixture():
    # Pure arithmetic control, not a neural or physical success test.
    h=np.array([[1.,1.,1.,1.],[1.,-1.,1.,-1.],[1.,1.,-1.,-1.],[1.,-1.,-1.,1.]])
    true=np.array([.1,-.2,.3,.4])
    diff=PROBE_STEP*(h@true)
    plus=np.tile(.2+diff,(2,1));minus=np.tile(.2-diff,(2,1))
    result=estimate_update(h,plus,minus)
    np.testing.assert_allclose(result['gradient'],true,atol=1e-15,rtol=0)
    delta=result['relative_delta']
    assert true@delta>0 and np.max(np.abs(delta))==UPDATE_STEP
    # Reversing rewards must give the exactly opposite proposed update.
    reversed_result=estimate_update(h,-plus,-minus)
    np.testing.assert_array_equal(reversed_result['relative_delta'],-delta)
    same=estimate_update(h,plus+.1,minus+.1)
    np.testing.assert_allclose(same['relative_delta'],delta,atol=1e-15,rtol=0)


def test_constant_returns_do_not_create_learning():
    result=estimate_update(fixed_directions(10),np.ones((2,4))*.2,np.ones((2,4))*.2)
    assert result['zero_signal'];np.testing.assert_array_equal(result['relative_delta'],np.zeros(10))


@pytest.mark.parametrize('change',['missing','duplicate','heldout','partial','unverified','nonfinite','no_provenance'])
def test_training_return_panel_excludes_partial_duplicate_or_evaluation_data(change):
    records=[dict(seed=s,direction=d,sign=sign,return_ignored=0,**{'return':.1},complete_horizon=True,
                  actual_import_verified=True,checkpoint_sha256='a'*64,return_source_sha256='b'*64)
             for s in TRAIN_SEEDS for d in range(DIRECTIONS) for sign in (-1,1)]
    plus,minus=training_panel(records);assert plus.shape==minus.shape==(2,4)
    if change=='missing':records.pop()
    elif change=='duplicate':records.append(records[0])
    elif change=='heldout':records[0]['seed']=10006
    elif change=='partial':records[0]['complete_horizon']=False
    elif change=='unverified':records[0]['actual_import_verified']=False
    elif change=='nonfinite':records[0]['return']=float('nan')
    elif change=='no_provenance':records[0]['return_source_sha256']=''
    with pytest.raises(ValueError):training_panel(records)
