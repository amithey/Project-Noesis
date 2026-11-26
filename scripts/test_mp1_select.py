import numpy as np
import pytest
from mp1_select import TRAIN_SEEDS,STEP,MIN_GAIN,fixed_directions,select_policy,matched_gradient,training_panel


def test_worst_context_gain_not_mean_selects_complete_policy():
    returns=np.zeros((2,8));returns[:,0]=[.8,-.2];returns[:,1]=[.11,.12];returns[:,2]=[-.2,-.3]
    positive=select_policy([0.,0.],returns)
    negative=select_policy([0.,0.],returns,-1)
    assert positive['policy_index']==1 and positive['accepted_for_training_update']
    assert negative['policy_index']==2 and negative['accepted_for_training_update']
    assert not positive['model_promoted'] and positive['training_condition_not_independent_efficacy']


def test_ties_are_canonical_and_no_gain_returns_parent():
    returns=np.full((2,8),.1)
    assert select_policy([0.,0.],returns)['policy_index']==0
    for value in (0.,MIN_GAIN,-.1):
        result=select_policy([0.,0.],np.full((2,8),value))
        assert result['parent_fallback'] and result['policy_index']==-1


def test_gradient_comparator_uses_same_probe_update_scale():
    h=np.array([[1.,1.,1.,1.],[1.,-1.,1.,-1.],[1.,1.,-1.,-1.],[1.,-1.,-1.,1.]])
    true=np.array([.1,-.2,.3,.4]);returns=np.zeros((2,8))
    returns[:,1::2]=STEP*(h@true);returns[:,::2]=-STEP*(h@true)
    result=matched_gradient(h,returns)
    np.testing.assert_allclose(result['gradient'],true,rtol=0,atol=1e-15)
    assert abs(result['relative_delta']).max()==STEP
    reversed_result=matched_gradient(h,-returns)
    np.testing.assert_array_equal(reversed_result['relative_delta'],-result['relative_delta'])


def test_new_directions_are_reproducible_without_using_global_rng():
    np.random.seed(10);before=np.random.get_state();a=fixed_directions(20);after=np.random.get_state()
    np.testing.assert_array_equal(a,fixed_directions(20));np.testing.assert_array_equal(before[1],after[1])
    assert before[2:]==after[2:] and a.shape==(4,20)


@pytest.mark.parametrize('fault',['missing','duplicate','old_train_seed','fresh_eval_seed','partial','wrong_import','nonfinite'])
def test_only_fresh_complete_trial_records_can_select_policy(fault):
    records=[dict(seed=s,direction=d,sign=sign,**{'return':.1},complete_horizon=True,actual_import_verified=True,
        checkpoint_sha256='a'*64,return_source_sha256='b'*64) for s in TRAIN_SEEDS for d in range(4) for sign in (-1,1)]
    assert training_panel(records).shape==(2,8)
    if fault=='missing':records.pop()
    elif fault=='duplicate':records.append(records[0])
    elif fault=='old_train_seed':records[0]['seed']=1008
    elif fault=='fresh_eval_seed':records[0]['seed']=11008
    elif fault=='partial':records[0]['complete_horizon']=False
    elif fault=='wrong_import':records[0]['actual_import_verified']=False
    else:records[0]['return']=float('nan')
    with pytest.raises(ValueError):training_panel(records)
