import copy
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_duration_study import (TRACE_FIELDS,V9_DIAGNOSTICS,PROBES,standing_trace_check,
    standing_report_check,standing_decision)


def traces(tmp_path):
    a,b=tmp_path/'standing',tmp_path/'gripped';a.mkdir();b.mkdir()
    data={k:np.zeros((5000,1)) for k in TRACE_FIELDS};data['learning_active']=np.arange(5000)>=10
    log=dict(drive=np.zeros((4990,2)),credit=np.zeros((4990,3)),drive_idx=np.array([2,4]),
        motor_indices=np.array([2,4,6]),control_chunk_index=np.arange(10,5000),control_ms=np.array(5.))
    np.savez(b/'training-observations.npz',**data)
    candidate={k:v.copy() for k,v in data.items() if k not in V9_DIAGNOSTICS}
    candidate['reward'][:]=.7
    for p in (a,b):np.savez(p/'exploration-log.npz',**log)
    np.savez(a/'training-observations.npz',**candidate)
    return a,b,candidate,log


def test_intended_reward_and_schema_change_is_allowed_but_physical_warmup_is_exact(tmp_path):
    a,b,data,_=traces(tmp_path)
    assert set(standing_trace_check(a,b))==TRACE_FIELDS-V9_DIAGNOSTICS-{'reward'}
    data['position'][10:]=1;np.savez(a/'training-observations.npz',**data)
    standing_trace_check(a,b)
    data['position'][9]=1;np.savez(a/'training-observations.npz',**data)
    with pytest.raises(ValueError,match='prelearning'):standing_trace_check(a,b)


@pytest.mark.parametrize('change',['partial','nonfinite','extra_cycle','exploration','learning_window'])
def test_profile_replacement_does_not_authorize_incomplete_data_or_extra_factors(tmp_path,change):
    a,b,data,log=traces(tmp_path)
    if change=='partial':data={k:v[:-1] for k,v in data.items()}
    elif change=='nonfinite':data['reward'][100]=np.nan
    elif change=='extra_cycle':data['cycle_credit']=np.zeros((5000,1))
    elif change=='exploration':log['drive'][0,0]=1
    elif change=='learning_window':data['learning_active'][10]=False
    np.savez(a/'training-observations.npz',**data);np.savez(a/'exploration-log.npz',**log)
    with pytest.raises(ValueError):standing_trace_check(a,b)


def report():
    keys=('status','seed','worlds','backend','noesis_source_sha256','import_checkpoint_sha256',
        'eligibility_ms','motor_exploration_mv_ms','original_weight_bound','control_ms','physics_dt','birth_ms',
        'requested_seconds','completed_chunks','simulated_seconds','learning_updates','external_vision_and_chemical_senses')
    return dict(**dict.fromkeys(keys,1),learning_rate=.2,reward_objective='curriculum-gripped-cycle-v9',cycle_weight=.1)


@pytest.mark.parametrize('change',[None,'learning_rate','parent','cycle','profile'])
def test_only_registered_profile_metadata_may_change(change):
    control=report();candidate=copy.deepcopy(control);candidate['reward_objective']='curriculum-standing-v7';del candidate['cycle_weight']
    if change=='learning_rate':candidate['learning_rate']=.02
    elif change=='parent':candidate['import_checkpoint_sha256']='other'
    elif change=='cycle':candidate['cycle_weight']=.1
    elif change=='profile':candidate['reward_objective']='other'
    if change:
        with pytest.raises(ValueError):standing_report_check(candidate,control)
    else:standing_report_check(candidate,control)


def cases():
    return {a:{s:dict(seed=s,standing_fraction=.96 if a=='child25' else .95,mean_stance_slip_mm_s=2.,
        heading_forward_distance_mm=0.,net_forward_displacement_mm=0.,leg_step_counts={leg:0 for leg in ('LF','LM','LH','RF','RM','RH')})
        for s in PROBES} for a in ('parent','control25','child25')}


def test_standing_success_is_separate_from_unchanged_walking_diagnostic():
    r=standing_decision(cases());assert r['standing_method_rule_passed']
    assert not r['paired_metrics']['exploratory_method_rule_pass']
    assert not r['walking_proven'] and not r['stage_advanced'] and not r['model_promoted']


def test_any_single_probe_regression_rejects_standing_method_even_above_absolute_gate():
    c=cases();c['child25'][10011]['standing_fraction']=.949
    r=standing_decision(c);assert r['all_four_standing_gates_passed'] and not r['standing_method_rule_passed']
    c=cases();c['child25'][10007]['mean_stance_slip_mm_s']=2.001
    assert not standing_decision(c)['standing_method_rule_passed']
