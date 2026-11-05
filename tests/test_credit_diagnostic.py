import sys
import os
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'gpu-pilot/tests'))
if os.environ.get('NOESIS_SOURCE'):
    sys.path.insert(0,os.environ['NOESIS_SOURCE'])
from credit_diagnostic import replay_rewards, lag_correlations, association_reference


def test_actual_small_brain_normalization_is_exactly_replayed(tmp_path):
    from noesis_gpu.learning import LearningBrain
    from test_weight_integrity import build
    from test_learning import state
    cn,parent=build(tmp_path)
    brain=LearningBrain(cn,np.zeros(42),np.full(42,-.7),np.full(42,.7),checkpoint=parent,seed=7)
    rewards=np.random.default_rng(12).normal(.2,.4,150)
    actual=[]
    for reward in rewards:
        brain.learning_active=True
        brain.step(state(),5)
        actual.append(brain.learn(float(reward),5))
    replay,final=replay_rewards(rewards)
    for key in replay:
        np.testing.assert_array_equal(replay[key],[row[key] for row in actual])
    for key in final:assert final[key]==getattr(brain.learner,key)


def test_fft_lags_match_direct_pearson_and_detect_known_delay():
    rng=np.random.default_rng(14);x=rng.normal(size=(600,4));y=np.r_[np.zeros(7),x[:-7,2]]
    corr,active,width=lag_correlations(x,y,20)
    for lag in [-20,-7,0,7,20]:
        for col in range(4):
            direct=np.corrcoef(x[20:580,col],y[20+lag:580+lag])[0,1]
            assert corr[lag%len(y),col]==pytest.approx(direct,abs=1e-12)
    r=association_reference(x,y,20,99)
    assert r['best_lag_steps']==7 and r['best_column']==2
    assert r['observed_max_abs_r']==pytest.approx(1)
    assert r['rank_is_calibrated_p_value'] is False


def test_reject_nonfinite_or_constant_inputs():
    with pytest.raises(ValueError):replay_rewards([0,np.nan])
    with pytest.raises(ValueError):lag_correlations(np.zeros((600,2)),np.zeros(600),20)
