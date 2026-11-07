import numpy as np
import pytest
from scipy import sparse
from sc1_signal import normalized_signal,lag_correlation
from noesis.lif import LIFNetwork
from noesis.plasticity import MotorPlasticity


def test_reconstruction_matches_actual_native_updates_without_network_integration():
    net=LIFNetwork(sparse.csr_matrix(np.array([[0.,3.,4.],[0.,0.,-2.],[1.,0.,0.]])))
    learner=MotorPlasticity(net,[2],learning_rate=0,rule='perturbation',normalize_reward_error=True,bias_correct_reward_error=True)
    rewards=np.r_[np.full(30,-.94),np.linspace(-1,.6,200),np.full(20,-1)];saved=rewards.copy();r=normalized_signal(rewards)
    for i,reward in enumerate(rewards):
        actual=learner.update([0,1,2],5,float(reward),perturbation=np.array([.2]))
        for a,b in (('baseline','reward_baseline'),('raw_error','prediction_error'),('variance','reward_variance'),('scale','reward_error_scale'),('signal','learning_signal')):
            assert r[a][i]==actual[b]
        assert r['mass'][i]==learner.reward_normalization_mass
    np.testing.assert_array_equal(rewards,saved)
    assert r['signal'][0]==0 and np.all(r['signal'][:30]==0)


def test_lag_direction_and_shift_control_are_statistical_fixture_not_neural_efficacy():
    rng=np.random.default_rng(702);x=rng.normal(size=(1000,3));y=rng.normal(size=1000);y[10:]=x[:-10,0]
    r,valid=lag_correlation(x,y,10);assert valid.all() and r[0]>1-1e-12
    backwards,_=lag_correlation(x,y,-10);assert abs(backwards[0])<.2
    shifted,_=lag_correlation(x,np.roll(y,300),10);assert abs(shifted[0])<.2
    r,valid=lag_correlation(np.c_[x,np.zeros(1000)],np.zeros(1000),0)
    assert np.all(r==0) and not valid.any()


@pytest.mark.parametrize('values',[[np.nan],[1.1],[],[[0.1]]])
def test_invalid_reward_stream_stops(values):
    with pytest.raises(ValueError):normalized_signal(values)
