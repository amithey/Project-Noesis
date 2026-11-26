"""EW1 complete-trial physical returns and preregistered comparison rules."""
import hashlib
import json
import numpy as np
from ps1_objective import physical_score,OBJECTIVE
from support_descriptor_c1 import REGIONS
from s2_contract import standing_retention
from ap2_contract import anatomical_rule
from ew1_update import TRAIN_SEEDS
from ew1_worker import DEV,validate_arguments
from types import SimpleNamespace


def validate_report(report):
    required=dict(status='completed_transport_pilot',controller='brain',worlds=1,learning=False,
        motor_exploration=0,weights_unchanged=True,requested_simulated_seconds_per_world=5,
        simulated_seconds_per_world=5,completed_control_chunks=1000,control_ms=5.,physics_dt=.0001,
        ew1_worker_completed=True)
    if any(report.get(k)!=v for k,v in required.items()):raise ValueError('Require complete EW1 native frozen five-second worker')
    c=report['ew1_context'];validate_arguments(SimpleNamespace(**c))
    if (not c.get('worker_completed') or not c.get('frozen_native_neural_motor_execution')
            or c.get('online_learning') is not False or report['seeds']!=[c['seed']]
            or report['checkpoint_sha256']!=c['checkpoint_sha256'] or len(c['import_records'])!=1
            or c['import_records'][0].get('actual_import_exact') is not True):
        raise ValueError('Missing/mismatched actual EW1 native import/context')


def episode_rewards(observations,anatomy,model):
    if list(model['region_columns'])!=list(REGIONS):raise ValueError('Unmatched anatomical columns')
    needed=('omega','up_body','foot_load')
    shapes=((1000,1,3),(1000,1,3),(1000,1,6))
    if any(np.asarray(observations[k]).shape!=shape for k,shape in zip(needed,shapes)):
        raise ValueError('Require full single-world observations')
    if (np.asarray(anatomy['region_vertical_reaction']).shape!=(1001,4)
            or np.asarray(anatomy['com_velocity']).shape!=(1001,3)
            or np.asarray(anatomy['time']).shape!=(1001,)):
        raise ValueError('Require complete anatomical horizon including initial frame')
    np.testing.assert_allclose(anatomy['time'],np.arange(1001)*.005,rtol=0,atol=1e-10)
    score=physical_score(anatomy['region_vertical_reaction'][1:],anatomy['com_velocity'][1:],
        observations['omega'][:,0],observations['up_body'][:,0,2],observations['foot_load'][:,0],
        model['total_mass']*abs(model['gravity'][2]))
    return score['reward']


def return_record(report,observations,anatomy,model,source_hashes):
    validate_report(report);c=report['ew1_context']
    if c['phase']!='trial' or c['seed'] not in TRAIN_SEEDS:raise ValueError('Evaluation data cannot train EW1')
    if set(source_hashes)!={'report','observations','anatomy','model'} or any(len(h)!=64 for h in source_hashes.values()):
        raise ValueError('Require full raw return-source provenance')
    rewards=episode_rewards(observations,anatomy,model)
    digest=hashlib.sha256(json.dumps(source_hashes,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return dict(seed=c['seed'],direction=c['direction'],sign=c['sign'],
        **{'return':float(np.mean(rewards[10:]))},return_objective=OBJECTIVE,warmup_samples=10,
        measured_samples=990,complete_horizon=True,actual_import_verified=True,
        checkpoint_sha256=c['checkpoint_sha256'],return_source_sha256=digest,source_hashes=source_hashes)


def comparison(banks):
    expected=set((*TRAIN_SEEDS,*DEV))
    if set(banks)!={'parent','candidate','negative'} or any(set(b)!=expected for b in banks.values()):
        raise ValueError('Require every fixed post-update comparison cell')
    candidate={s:banks['candidate'][s]['summary'] for s in DEV}
    retention={b:standing_retention({s:banks[b][s]['summary'] for s in DEV},candidate,DEV) for b in ('parent','negative')}
    anatomy=anatomical_rule(*({s:banks[b][s]['anatomical'] for s in DEV} for b in ('parent','negative','candidate')))
    paired={};directions_pass=True
    for seed in DEV:
        paired[seed]={}
        c=banks['candidate'][seed]['physical']
        for bank in ('parent','negative'):
            reference=banks[bank][seed]['physical']
            delta={k:c[k]-reference[k] for k in ('mean_PS1_quality','mean_COM_speed_mm_s','mean_angular_speed_rad_s','mean_non_tarsal_fraction')}
            passed=delta['mean_PS1_quality']>0 and all(v<0 for k,v in delta.items() if k!='mean_PS1_quality')
            directions_pass=directions_pass and passed
            paired[seed][bank]=dict(changes=delta,all_physical_directions_improve=bool(passed))
    training={s:{b:banks['candidate'][s]['return']>banks[b][s]['return'] for b in ('parent','negative')} for s in TRAIN_SEEDS}
    train_pass=all(all(v.values()) for v in training.values())
    return dict(retention=retention,anatomical=anatomy,paired_development=paired,training_return_direction=training,
        both_training_seeds_improve=bool(train_pass),each_development_physical_direction_improves=bool(directions_pass),
        combined_exploratory_pilot_pass=bool(train_pass and directions_pass and anatomy['passed'] and all(x['passed'] for x in retention.values())),
        one_between_episode_update_only=True,model_promoted=False,standing_proven=False,walking_proven=False,
        limits='Single update from four random directions; reused development benchmarks are exploratory, not untouched final qualification')
