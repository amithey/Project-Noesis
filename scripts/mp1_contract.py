"""MP1 provenance/physical comparison rules; never motor control."""
import hashlib
import json
from types import SimpleNamespace
import numpy as np
from ew1_contract import episode_rewards
from mp1_select import TRAIN_SEEDS,DEV,FRESH
from mp1_worker import validate_arguments
from s2_contract import standing_retention
from ap2_contract import anatomical_rule


def validate_report(report):
    expected=dict(status='completed_transport_pilot',controller='brain',worlds=1,learning=False,
        motor_exploration=0,weights_unchanged=True,requested_simulated_seconds_per_world=5,
        simulated_seconds_per_world=5,completed_control_chunks=1000,control_ms=5.,physics_dt=.0001,mp1_worker_completed=True)
    if any(report.get(k)!=v for k,v in expected.items()):raise ValueError('Require complete native MP1 worker')
    c=report['mp1_context'];validate_arguments(SimpleNamespace(**c))
    if (c.get('worker_completed') is not True or c.get('online_learning') is not False
        or c.get('frozen_native_neural_motor_execution') is not True or report['seeds']!=[c['seed']]
        or report['checkpoint_sha256']!=c['checkpoint_sha256'] or len(c['import_records'])!=1
        or c['import_records'][0].get('actual_import_exact') is not True):
        raise ValueError('Invalid actual MP1 import/context')


def return_record(report,observations,anatomy,model,source_hashes):
    validate_report(report);c=report['mp1_context']
    if c['phase']!='trial' or c['seed'] not in TRAIN_SEEDS:raise ValueError('Evaluation cannot train MP1')
    if set(source_hashes)!={'report','observations','anatomy','model'} or any(len(h)!=64 for h in source_hashes.values()):
        raise ValueError('Require raw return-source provenance')
    rewards=episode_rewards(observations,anatomy,model)
    digest=hashlib.sha256(json.dumps(source_hashes,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return dict(seed=c['seed'],direction=c['direction'],sign=c['sign'],**{'return':float(rewards[10:].mean())},
        complete_horizon=True,actual_import_verified=True,checkpoint_sha256=c['checkpoint_sha256'],
        return_source_sha256=digest,source_hashes=source_hashes,warmup_samples=10,measured_samples=990,
        return_objective='physical-standing-ps1-v1')


def comparison(banks,selection_accepted):
    expected=set((*TRAIN_SEEDS,*DEV,*FRESH))
    if set(banks)!={'parent','candidate','negative','gradient'} or any(set(b)!=expected for b in banks.values()):
        raise ValueError('Missing/extra fixed MP1 comparison cells')
    for bank in banks.values():
        for seed,cell in bank.items():
            if cell['summary']['seed']!=seed:raise ValueError('Mislabeled frozen seed')
    controls=('parent','negative','gradient')
    retention={group:{b:standing_retention({s:banks[b][s]['summary'] for s in seeds},
        {s:banks['candidate'][s]['summary'] for s in seeds},seeds) for b in controls}
        for group,seeds in (('existing',DEV),('fresh',FRESH))}
    anatomy={b:anatomical_rule(*({s:banks[name][s]['anatomical'] for s in DEV} for name in ('parent',b,'candidate')))
        for b in ('negative','gradient')}
    physical={};existing_pass=True;fresh_pass=True
    for seed in (*DEV,*FRESH):
        physical[seed]={};candidate=banks['candidate'][seed]
        for bank in controls:
            reference=banks[bank][seed]
            delta={k:candidate['physical'][k]-reference['physical'][k] for k in
                ('mean_PS1_quality','mean_COM_speed_mm_s','mean_angular_speed_rad_s','mean_non_tarsal_fraction')}
            if not np.isfinite(list(delta.values())).all():raise ValueError('Nonfinite physical comparison')
            strict=seed in DEV
            passed=(delta['mean_PS1_quality']>0 and all(v<0 for k,v in delta.items() if k!='mean_PS1_quality')) if strict else (
                    delta['mean_PS1_quality']>=0 and all(v<=0 for k,v in delta.items() if k!='mean_PS1_quality'))
            if not strict:
                a,c=candidate['anatomical'],reference['anatomical']
                values=[a['mode_fractions']['tarsal_only'],c['mode_fractions']['tarsal_only'],a['mean_non_tarsal_fraction'],c['mean_non_tarsal_fraction']]
                if not np.isfinite(values).all() or any(not 0<=v<=1 for v in values):raise ValueError('Invalid fresh anatomical fraction')
                passed=passed and values[0]>=values[1] and values[2]<=values[3]
            physical[seed][bank]=dict(changes=delta,passed=bool(passed),strict_improvement_required=strict)
            if strict:existing_pass=existing_pass and passed
            else:fresh_pass=fresh_pass and passed
    all_retention=all(row['passed'] for group in retention.values() for row in group.values())
    return dict(retention=retention,original_anatomical=anatomy,paired_physical=physical,
        existing_all_physical_directions_pass=bool(existing_pass),fresh_all_no_regression_pass=bool(fresh_pass),
        combined_exploratory_pilot_pass=bool(selection_accepted and existing_pass and fresh_pass and all_retention and all(x['passed'] for x in anatomy.values())),
        training_acceptance_is_by_construction=True,first_use_fresh_contexts_not_final_qualification=True,
        model_promoted=False,standing_proven=False,walking_proven=False)
