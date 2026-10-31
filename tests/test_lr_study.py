import copy
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_duration_study import (TRACE_FIELDS,PROBES,duration_decision,lr_trace_check,
                               lr_report_check,verify_lr_control,T1B_CHILD,lr_bank_audit)


def traces(tmp_path):
    control,candidate=tmp_path/'control',tmp_path/'candidate'
    control.mkdir();candidate.mkdir()
    data={k:np.zeros((5000,1)) for k in TRACE_FIELDS}
    data['learning_active']=np.arange(5000)>=10
    log=dict(drive=np.zeros((4990,2)),credit=np.zeros((4990,3)),drive_idx=np.array([2,4]),
             motor_indices=np.array([2,4,6]),control_chunk_index=np.arange(10,5000),control_ms=np.array(5.))
    for p in (control,candidate):
        np.savez(p/'training-observations.npz',**data)
        np.savez(p/'exploration-log.npz',**log)
    return control,candidate,data,log


def test_postlearning_trajectory_divergence_is_allowed_but_prelearning_is_exact(tmp_path):
    control,candidate,data,_=traces(tmp_path)
    data['position'][10:]=1
    np.savez(candidate/'training-observations.npz',**data)
    assert set(lr_trace_check(candidate,control))==TRACE_FIELDS
    data['position'][9]=1
    np.savez(candidate/'training-observations.npz',**data)
    with pytest.raises(ValueError,match='warmup'):lr_trace_check(candidate,control)


@pytest.mark.parametrize('changed',['missing_trace','partial','learning_window','drive','credit','alignment','control_ms','nonfinite'])
def test_refuse_incomplete_or_confounding_lr_data(tmp_path,changed):
    control,candidate,data,log=traces(tmp_path)
    if changed=='missing_trace':del data['position']
    elif changed=='partial':data={k:v[:-1] for k,v in data.items()}
    elif changed=='learning_window':data['learning_active'][10]=False
    elif changed in ('drive','credit'):log[changed][0,0]=1
    elif changed=='alignment':log['control_chunk_index'][0]=11
    elif changed=='control_ms':log['control_ms']=np.array(10.)
    elif changed=='nonfinite':data['position'][100]=np.nan
    np.savez(candidate/'training-observations.npz',**data)
    np.savez(candidate/'exploration-log.npz',**log)
    with pytest.raises(ValueError):lr_trace_check(candidate,control)


def report():
    keys=('status','seed','worlds','backend','noesis_source_sha256','import_checkpoint_sha256',
          'reward_objective','eligibility_ms','motor_exploration_mv_ms','original_weight_bound',
          'control_ms','physics_dt','birth_ms','requested_seconds','completed_chunks',
          'simulated_seconds','learning_updates','cycle_weight','external_vision_and_chemical_senses')
    return dict(**dict.fromkeys(keys,1),learning_rate=.2)


@pytest.mark.parametrize('changed',[None,'learning_rate','cycle_weight','import_checkpoint_sha256','simulated_seconds','noesis_source_sha256'])
def test_only_explicit_learning_rate_metadata_may_differ(changed):
    control=report();candidate=copy.deepcopy(control);candidate['learning_rate']=.02
    if changed:candidate[changed]=.1 if changed=='learning_rate' else 'other'
    if changed:
        with pytest.raises(ValueError):lr_report_check(candidate,control)
    else:lr_report_check(candidate,control)


def test_control25_names_preserve_all_original_retention_and_progress_gates():
    cases={arm:{seed:dict(seed=seed,standing_fraction=.95,mean_stance_slip_mm_s=2.,
          heading_forward_distance_mm=1.3 if arm=='child25' else .3,
          net_forward_displacement_mm=1.3 if arm=='child25' else .3,
          leg_step_counts={leg:3 for leg in ('LF','LM','LH','RF','RM','RH')})
          for seed in PROBES} for arm in ('parent','control25','child25')}
    d=duration_decision(cases,control_arm='control25')
    assert d['exploratory_method_rule_pass'] and 'control25' in d['medians']
    cases['child25'][10011]['standing_fraction']=.949
    assert not duration_decision(cases,control_arm='control25')['exploratory_method_rule_pass']
    cases['child25'][10011]['standing_fraction']=.95
    for s in cases['child25'].values():s['net_forward_displacement_mm']=.3
    assert not duration_decision(cases,control_arm='control25')['exploratory_method_rule_pass']
    with pytest.raises(ValueError):duration_decision(cases)


@pytest.mark.parametrize('bad',[False,True])
def test_only_audited_complete_known_t1b_control_is_accepted(tmp_path,monkeypatch,bad):
    import audit_duration_result
    monkeypatch.setattr(audit_duration_result,'audit',lambda *args:dict(
        child25_sha256='partial' if bad else T1B_CHILD,complete_frozen_probes=12))
    (tmp_path/'result.json').write_text('{}')
    if bad:
        with pytest.raises(ValueError):verify_lr_control(tmp_path,tmp_path,tmp_path)
    else:assert set(verify_lr_control(tmp_path,tmp_path,tmp_path))=={'result.json'}


@pytest.mark.parametrize('changed',[None,'learning_rate','eligibility','bound','sign','topology','original','normalization'])
def test_bank_exception_allows_only_registered_lr_metadata(tmp_path,changed):
    original=dict(topology=np.array([1,2]),motor_indices=np.array([0,1]),initial_weights=np.array([2.,-2.]),
        weights=np.array([2.,-2.]),parameters=np.array([.2,100.,.5]),rule=np.array('perturbation'),
        version=np.array(4),normalize_reward_error=np.array(True),bias_correct_reward_error=np.array(True))
    parent=tmp_path/'parent.npz';child=tmp_path/'child.npz';np.savez(parent,**original)
    candidate=copy.deepcopy(original);candidate['parameters'][0]=.02
    if changed=='learning_rate':candidate['parameters'][0]=.03
    elif changed=='eligibility':candidate['parameters'][1]=200
    elif changed=='bound':candidate['weights'][0]=3.001
    elif changed=='sign':candidate['weights'][0]=-2
    elif changed=='topology':candidate['topology'][0]=3
    elif changed=='original':candidate['initial_weights'][0]=3
    elif changed=='normalization':candidate['bias_correct_reward_error']=np.array(False)
    np.savez(child,**candidate)
    if changed:
        with pytest.raises(ValueError):lr_bank_audit(child,parent)
    else:assert len(lr_bank_audit(child,parent))==64
