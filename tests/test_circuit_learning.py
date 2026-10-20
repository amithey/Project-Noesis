import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from noesis.circuit_learning import CircuitExploration, import_subset_weights
from noesis.embodiment import MotorExploration
from noesis.lif import LIFNetwork
from noesis.plasticity import MotorPlasticity


def neurons():
    return pd.DataFrame({'Class':['single_leg_neuromere','leg_motor_neuron',
        'sequential_leg_neuromeres','single_leg_neuromere','leg_motor_neuron'],
        'Sub Class':['','front_leg_motor_neuron','','','front_leg_motor_neuron'],
        'Primary Cell Type':['','tibia_flexor','','','tibia_flexor'],
        'Nerve':['','left_leg','','','right_leg'],'Soma side':['','left','','','right']})


def test_motor_stream_exact_with_independent_bounded_circuit_innovations():
    n=neurons(); old=MotorExploration(n,1.5,seed=17)
    a=CircuitExploration(n,np.array([0,2]),1.5,seed=17)
    b=CircuitExploration(n,np.array([0,2]),1.5,seed=17)
    currents=[]
    for _ in range(100):
        original=old.step(5); extended=a.step(5); repeat=b.step(5)
        np.testing.assert_array_equal(original['current_idx'],extended['current_idx'][:2])
        np.testing.assert_array_equal(original['current_mv_per_ms'],extended['current_mv_per_ms'][:2])
        for key in extended: np.testing.assert_array_equal(extended[key],repeat[key])
        credit=a.credit(np.arange(5)); old_credit=old.credit(np.arange(5))
        np.testing.assert_array_equal(credit[[1,4]],old_credit[[1,4]])
        np.testing.assert_array_equal(credit[[0,2]],a.circuit_innovation)
        assert credit[3]==0 and np.max(abs(credit))<=1
        assert np.max(abs(extended['current_mv_per_ms']))<=1.5
        currents.extend(extended['current_mv_per_ms'][2:])
    assert min(currents)<0<max(currents)


def test_frozen_circuit_has_no_currents_or_rng_advance():
    e=CircuitExploration(neurons(),np.array([0,2]),0,seed=1)
    before=repr(e.circuit_rng.bit_generator.state)
    for _ in range(10):
        assert e.step(5)=={}
        assert not e.credit(np.arange(5)).any()
    assert repr(e.circuit_rng.bit_generator.state)==before
    assert not e.circuit_state.any()


@pytest.mark.parametrize('ids,amplitude',[(np.array([1]),1.5),(np.array([-1]),1.5),
    (np.array([5]),1.5),(np.array([0.5]),1.5),(np.array([0]),2),
    (np.array([0]),np.nan),(np.array([0]),-1)])
def test_bad_circuit_targets_and_current_bounds(ids,amplitude):
    with pytest.raises(ValueError): CircuitExploration(neurons(),ids,amplitude)


def net():
    return LIFNetwork(sparse.csr_matrix([[0.,2.,0.,0.,0.],
        [0.,0.,-3.,0.,0.],[4.,0.,0.,0.,0.],[0.,0.,5.,0.,0.],[-2.,-2.,0.,6.,0.]]))


def learner(network,targets):
    return MotorPlasticity(network,targets,learning_rate=.2,tau_eligibility_ms=100,
        rule='perturbation',normalize_reward_error=True,bias_correct_reward_error=True)


def old_bank(tmp_path):
    old=learner(net(),[2]); old.net.data[old.edge_indices]*=[1.2,.8]
    old.activity[:]=.3; old.eligibility[:]=.2; old.reward_baseline=.4
    old.reward_variance=.1; old.reward_normalization_mass=.5
    path=tmp_path/'old.npz'; old.save(path)
    return old,path


def test_subset_transfer_preserves_expanded_original_references_and_roundtrip(tmp_path):
    old,path=old_bank(tmp_path); fresh=net(); original=fresh.data.copy(); expanded=learner(fresh,[0,1,2])
    result=import_subset_weights(expanded,path)
    assert result=={'imported_targets':1,'imported_edges':2}
    np.testing.assert_array_equal(expanded.initial_weights,original[expanded.edge_indices])
    np.testing.assert_array_equal(fresh.data[old.edge_indices],old.net.data[old.edge_indices])
    other=np.setdiff1d(np.arange(len(original)),old.edge_indices)
    np.testing.assert_array_equal(fresh.data[other],original[other])
    assert not expanded.activity.any() and not expanded.eligibility.any()
    assert expanded.reward_baseline==expanded.reward_variance==expanded.reward_normalization_mass==0
    saved=tmp_path/'expanded.npz'; expanded.save(saved)
    copy=learner(net(),[0,1,2]); copy.load(saved)
    np.testing.assert_array_equal(copy.net.data,fresh.data)
    with pytest.raises(ValueError): learner(net(),[2]).load(saved)


@pytest.mark.parametrize('key,value',[('weights',np.array([20.,5.])),('weights',np.array([3.,5.])),
    ('weights',np.array([np.nan,5.])),('initial_weights',np.array([3.,5.])),
    ('topology',np.array('wrong')),('motor_indices',np.array([4]))])
def test_bad_subset_rejected_without_any_weight_mutation(tmp_path,key,value):
    _,path=old_bank(tmp_path)
    with np.load(path) as z: fields={k:z[k].copy() for k in z.files}
    fields[key]=value; np.savez_compressed(path,**fields)
    expanded=learner(net(),[0,1,2]); before=expanded.net.data.copy()
    with pytest.raises(ValueError): import_subset_weights(expanded,path)
    np.testing.assert_array_equal(expanded.net.data,before)


def test_subset_import_requires_fresh_state_and_original_network(tmp_path):
    _,path=old_bank(tmp_path)
    for dirty in ('state','weights'):
        expanded=learner(net(),[0,1,2])
        if dirty=='state': expanded.activity[0]=.1
        else: expanded.net.data[expanded.edge_indices[0]]*=1.1
        before=expanded.net.data.copy()
        with pytest.raises(ValueError,match='fresh'): import_subset_weights(expanded,path)
        np.testing.assert_array_equal(expanded.net.data,before)


def test_positive_circuit_credit_changes_excitatory_and_inhibitory_input_in_correct_direction():
    a=learner(net(),[0,2]); original=a.net.data.copy()
    a.update(np.arange(5),5,0,perturbation=np.zeros(2))
    a.update(np.arange(5),5,1,perturbation=np.array([1.,0.]))
    post=a.net.indices; chosen=(post==0)
    excitatory=chosen&(original>0); inhibitory=chosen&(original<0)
    assert np.all(a.net.data[excitatory]>original[excitatory])
    assert np.all(a.net.data[inhibitory]>original[inhibitory])
    np.testing.assert_array_equal(a.net.data[~chosen],original[~chosen])
    np.testing.assert_array_equal(np.sign(a.net.data),np.sign(original))
