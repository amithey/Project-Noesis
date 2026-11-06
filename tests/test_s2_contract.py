"""Reject incomplete births and single-seed regressions before S2 model interpretation."""
from pathlib import Path
import sys

import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from s2_contract import validate_training_report,standing_retention


def report():
    return dict(status='completed_training_birth',requested_seconds=25,simulated_seconds=25,
                completed_chunks=5000,learning_updates=4990,reward_objective='curriculum-standing-v7',
                learning_rate=.2,eligibility_ms=100,motor_exploration_mv_ms=1.5,original_weight_bound=.5,
                control_ms=5,physics_dt=.0001,birth_ms=50,learning=True,worlds=1,
                exploration_log=dict(steps=4990,first_control_chunk=10))


def test_full_exact_contract_is_accepted():validate_training_report(report())


@pytest.mark.parametrize('key,value',[('status','incomplete_wall_budget'),('completed_chunks',4999),
                                    ('learning_updates',4989),('learning_rate',.02),('simulated_seconds',24.995)])
def test_partial_or_changed_birth_cannot_pass(key,value):
    r=report();r[key]=value
    with pytest.raises(ValueError):validate_training_report(r)


def test_missing_log_is_not_complete_evidence():
    r=report();r.pop('exploration_log')
    with pytest.raises(ValueError):validate_training_report(r)


def row(standing=.95,slip=2.,heading=.2,net=.1,cycles=10,tripod=.1):
    return dict(standing_fraction=standing,mean_stance_slip_mm_s=slip,heading_forward_distance_mm=heading,
                net_forward_displacement_mm=net,total_leg_steps=cycles,
                tripod_coordination=dict(tripod_coordination_score=tripod))


def test_one_seed_regression_cannot_hide_behind_median_improvement():
    seeds=[10006,10007,10010,10011];parent={s:row() for s in seeds}
    child={s:row(standing=.96,slip=1.9,heading=.3,net=.2) for s in seeds}
    assert standing_retention(parent,child,seeds)['passed']
    child[10011]=row(standing=.89,heading=3.,net=3.)
    assert not standing_retention(parent,child,seeds)['passed']
    child[10011]=row(slip=2.01,heading=3.,net=3.)
    assert not standing_retention(parent,child,seeds)['passed']


def test_missing_and_nonfinite_seed_evidence_is_rejected():
    with pytest.raises(ValueError):standing_retention({10006:row()},{},[10006])
    with pytest.raises(ValueError):standing_retention({10006:row()},{10006:row(heading=np.nan)},[10006])
