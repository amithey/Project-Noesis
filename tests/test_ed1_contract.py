import numpy as np
import pytest
from scipy import sparse
from noesis.lif import LIFNetwork
from noesis.plasticity import MotorPlasticity
from ed1_contract import audited_learner


def make(delay):
    instances=[];cls=audited_learner(MotorPlasticity,delay,instances)
    learner=cls(LIFNetwork(sparse.csr_matrix(np.array([[0.,3.,4.],[0.,0.,-2.],[1.,0.,0.]]))),[2],
        learning_rate=.2,tau_eligibility_ms=100,max_change=.5,rule='perturbation',normalize_reward_error=True,bias_correct_reward_error=True)
    assert instances==[learner];return learner


@pytest.mark.parametrize('delay',[0,10])
def test_actual_queue_hashes_weight_steps_and_global_clock(delay):
    learner=make(delay)
    for i in range(50):learner.update([0,1,2],5,float(np.sin(i/7)),perturbation=np.array([float(np.cos(i/3))]))
    rows=learner.ed1_audit;assert len(rows)==50
    for i,r in enumerate(rows):
        assert r['control_chunk_index']==i+10 and r['independent_weight_step_exact']
        if i>=delay:
            assert r['used_source_update_index']==i-delay
            assert r['used_eligibility_sha256']==rows[i-delay]['current_eligibility_sha256']
        else:assert r['used_source_update_index']==-1 and r['used_eligibility_L2']==0


def test_independent_guard_detects_wrong_eligibility_provenance():
    learner=make(10);learner.update([0,1,2],5,-.4,perturbation=np.array([.5]))
    # Corrupt only the prototype's private history; the independent queue must catch it.
    learner.ed1_queue.clear()
    for i in range(10):
        if i==9:
            with pytest.raises(AssertionError):learner.update([0,1,2],5,.6,perturbation=np.array([.5]))
        else:learner.update([0,1,2],5,.6,perturbation=np.array([.5]))


def test_fresh_weight_transfer_resets_both_independent_and_actual_queues(tmp_path):
    learner=make(10)
    for i in range(20):learner.update([0,1,2],5,.2,perturbation=np.array([.5]))
    learner.save(tmp_path/'model.npz');learner.load(tmp_path/'model.npz',weights_only=True)
    assert not learner.ed1_independent_queue and not learner.ed1_queue and not learner.ed1_audit and not learner.ed1_records
    learner.update([0,1,2],5,.4,perturbation=np.array([.3]));assert learner.ed1_audit[0]['used_source_update_index']==-1
