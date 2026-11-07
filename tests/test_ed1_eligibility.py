import hashlib
import numpy as np
import pytest
from scipy import sparse
from noesis.lif import LIFNetwork
from noesis.plasticity import MotorPlasticity
from ed1_eligibility import delayed_learner


def learner(delay=None,rate=.2):
    net=LIFNetwork(sparse.csr_matrix(np.array([[0.,3.,4.],[0.,0.,-2.],[1.,0.,0.]])))
    cls=MotorPlasticity if delay is None else delayed_learner(MotorPlasticity,delay)
    return cls(net,[2],learning_rate=rate,tau_eligibility_ms=100,max_change=.5,rule='perturbation',normalize_reward_error=True,bias_correct_reward_error=True)


def frames():
    rng=np.random.default_rng(731)
    return [(np.array([0,1,2]) if i%3 else np.array([0,2]),float(r),np.array([float(p)])) for i,(r,p) in enumerate(zip(rng.uniform(-1,1,200),rng.uniform(-1,1,200)))]


def test_zero_delay_exact_native_update_state_and_checkpoint_bytes(tmp_path):
    a,b=learner(),learner(0)
    for spikes,reward,credit in frames():
        assert a.update(spikes,5,reward,perturbation=credit)==b.update(spikes,5,reward,perturbation=credit)
        for x,y in ((a.net.data,b.net.data),(a.activity,b.activity),(a.eligibility,b.eligibility)):np.testing.assert_array_equal(x,y)
    a.save(tmp_path/'a.npz');b.save(tmp_path/'b.npz');assert (tmp_path/'a.npz').read_bytes()==(tmp_path/'b.npz').read_bytes()


def test_positive_delay_uses_entire_past_eligibility_with_native_reward_moments():
    trace,delayed=learner(rate=0),learner(10);history=[];expected=delayed.net.data[delayed.edge_indices].copy()
    for i,(spikes,reward,credit) in enumerate(frames()):
        native=trace.update(spikes,5,reward,perturbation=credit);history.append(trace.eligibility.copy())
        used=history[i-10] if i>=10 else np.zeros_like(history[0]);delta=.2*.005*native['learning_signal']*used
        initial=delayed.initial_weights;relative=np.abs(expected)/np.abs(initial)
        expected=initial*np.clip(relative+delta,.5,1.5)
        actual=delayed.update(spikes,5,reward,perturbation=credit)
        np.testing.assert_array_equal(delayed.net.data[delayed.edge_indices],expected)
        for k in ('reward_baseline','reward_variance','prediction_error','learning_signal'):assert actual[k]==native[k]
        np.testing.assert_array_equal(delayed.eligibility,trace.eligibility)
        record=delayed.ed1_records[-1]
        assert record['used_source_update_index']==(i-10 if i>=10 else -1)
        assert record['used_eligibility_sha256']==hashlib.sha256(memoryview(used)).hexdigest()
    assert len(delayed.ed1_queue)==10


def test_invalid_call_preserves_queue_state_and_learning_rate():
    d=learner(10);d.update([0,1,2],5,-.9,perturbation=np.array([.2]));weights=d.net.data.copy();records=list(d.ed1_records);eligibility=d.eligibility.copy()
    for dt,reward in ((4,.2),(5,float('nan'))):
        with pytest.raises(ValueError):d.update([0,1,2],dt,reward,perturbation=np.array([.2]))
        assert d.learning_rate==.2 and d.ed1_records==records and len(d.ed1_queue)==1
        np.testing.assert_array_equal(d.net.data,weights);np.testing.assert_array_equal(d.eligibility,eligibility)


def test_original_sign_bounds_and_weights_only_reload_reset(tmp_path):
    d=learner(10,rate=1e300);original=d.net.data.copy();indices=d.net.indices.copy();indptr=d.net.indptr.copy()
    for spikes,reward,credit in frames():d.update(spikes,5,reward,perturbation=credit)
    weights=d.net.data[d.edge_indices];np.testing.assert_array_equal(np.sign(weights),np.sign(d.initial_weights))
    np.testing.assert_array_equal(d.net.indices,indices);np.testing.assert_array_equal(d.net.indptr,indptr)
    other=np.setdiff1d(np.arange(len(original)),d.edge_indices);np.testing.assert_array_equal(d.net.data[other],original[other])
    assert np.all(np.abs(weights)>=.5*np.abs(d.initial_weights)) and np.all(np.abs(weights)<=1.5*np.abs(d.initial_weights))
    d.save(tmp_path/'d.npz')
    with pytest.raises(ValueError):d.load(tmp_path/'d.npz',restore_traces=True)
    d.load(tmp_path/'d.npz',weights_only=True);assert len(d.ed1_queue)==len(d.ed1_records)==0 and not d.eligibility.any()
    np.testing.assert_array_equal(d.net.data[d.edge_indices],weights)


def test_unregistered_delay_rejected():
    for value in (True,-1,1,11):
        with pytest.raises(ValueError):delayed_learner(MotorPlasticity,value)
