import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_duration_study import PROBES,duration_decision,prefix_check


def cases():
    return {arm:{seed:dict(seed=seed,standing_fraction=.95,mean_stance_slip_mm_s=2.,
             heading_forward_distance_mm=.3+(arm=='child25'),net_forward_displacement_mm=.3+(arm=='child25'),
             leg_step_counts={leg:3 for leg in ['LF','LM','LH','RF','RM','RH']}) for seed in PROBES}
            for arm in ['parent','control5','child25']}


def test_progress_cannot_hide_regression_on_any_probe():
    c=cases();assert duration_decision(c)['exploratory_method_rule_pass']
    c['child25'][10011]['standing_fraction']=.949
    assert not duration_decision(c)['exploratory_method_rule_pass']
    c=cases();c['child25'][10010]['mean_stance_slip_mm_s']=2.001
    assert not duration_decision(c)['exploratory_method_rule_pass']


def test_require_complete_finite_comparison_and_both_progress_metrics():
    c=cases();del c['parent'][10011]
    with pytest.raises(ValueError):duration_decision(c)
    c=cases();c['control5'][10006]['standing_fraction']=float('nan')
    with pytest.raises(ValueError):duration_decision(c)
    c=cases()
    for s in c['child25'].values():s['net_forward_displacement_mm']=.3
    assert not duration_decision(c)['exploratory_method_rule_pass']


def test_prefix_duration_and_alignment_are_exact(tmp_path):
    a,b=tmp_path/'long',tmp_path/'short';a.mkdir();b.mkdir()
    long=np.zeros((5000,3));np.savez(a/'training-observations.npz',position=long)
    np.savez(b/'training-observations.npz',position=np.zeros((1000,3)))
    np.savez(a/'exploration-log.npz',control_chunk_index=np.arange(10,5000),drive=np.zeros((4990,2)),credit=np.zeros((4990,3)),drive_idx=[1,2],motor_indices=[1,2,3])
    assert prefix_check(a,b)==['position']
    long[999,0]=1;np.savez(a/'training-observations.npz',position=long)
    with pytest.raises(ValueError,match='prefix'):prefix_check(a,b)
