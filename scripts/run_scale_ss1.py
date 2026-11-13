"""SS1 bounded native scale sensitivity; diagnostic probes, no learned model."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import uuid
import numpy as np
import run_training_context_mp2 as base
from mp2_select import TRAIN
from ss1_scale import SCALES,NEW_SCALES,SIGNAL,COMPATIBILITY,key,panel,analyze

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'work'
SOURCE=('ss1_scale.py','ss1_worker.py','run_scale_ss1.py','audit_scale_ss1.py','test_ss1_scale.py','test_ss1_orchestration.py')
LIMIT=7500
read,write,sha,arrays=base.read,base.write,base.sha,base.arrays


def jobs():
    result=[dict(name='baseline-1010',phase='baseline',bank='parent',seed=1010,direction=None,sign=None,amplitude=None)]
    for s in TRAIN:
        for h in NEW_SCALES:
            for k in (-1,1):result.append(dict(name=f'trial-{s}-{h}-{k}',phase='trial',bank='probe',seed=s,direction=0,sign=k,amplitude=h))
    return result


def reference(p,seed,amplitude=None,sign=None):
    old=Path(p['runtime_root'])/'outputs/measured-policy-mp1-20261110'
    prior=old if seed in (1010,1011) else Path(p['mp2_output'])
    return prior/(f'parent-{seed}' if amplitude is None else f'trial-0-{seed}-{sign}')


def checkpoint(p,j):
    return Path(p['runtime_root'])/p['parent_checkpoint'] if j['phase']=='baseline' else Path(p['inputs'])/f"probe-{j['amplitude']}-{j['sign']}.npz"


def validate_report(r,j,cp_hash,worker_hash):
    expected=dict(status='completed_transport_pilot',controller='brain',worlds=1,learning=False,motor_exploration=0,
        weights_unchanged=True,requested_simulated_seconds_per_world=5,simulated_seconds_per_world=5,
        completed_control_chunks=1000,control_ms=5.,physics_dt=.0001,ss1_worker_completed=True)
    if any(r.get(k)!=v for k,v in expected.items()):raise ValueError('Incomplete native SS1 worker')
    c=r['ss1_context'];key(c,baseline=j['phase']=='baseline')
    if any(c[k]!=j[k] for k in ('phase','bank','seed','direction','sign','amplitude')):raise ValueError('Wrong SS1 context')
    if (c['checkpoint_sha256']!=cp_hash or r['checkpoint_sha256']!=cp_hash or c['worker_sha256']!=worker_hash
            or r['seeds']!=[j['seed']] or c.get('worker_completed') is not True
            or c.get('frozen_native_neural_motor_execution') is not True or c.get('online_learning') is not False
            or len(c['import_records'])!=1):raise ValueError('Wrong native provenance')
    if not all(c['import_records'][0].get(k) is True for k in ('actual_import_exact','topology_unchanged','nonselected_weights_unchanged')):
        raise ValueError('Actual native full-graph import failed')


def reused(p,old):
    result=[]
    for s in TRAIN:
        for h,k in [(None,None),(.01,-1),(.01,1)]:
            context=dict(phase='parent' if h is None else 'trial',bank='parent' if h is None else 'probe',seed=s,
                direction=None if h is None else 0,sign=k,amplitude=h)
            folder=reference(p,s,h,k);r=read(folder/'report.json')
            if s in (1010,1011):
                from mp1_contract import validate_report as validate_original
                validate_original(r);c=r['mp1_context']
            else:
                original={k:v for k,v in context.items() if k!='amplitude'}
                cp=Path(p['runtime_root'])/p['parent_checkpoint'] if h is None else Path(p['runtime_root'])/'outputs/measured-policy-mp1-inputs-20261110'/f'probe-0-{k}.npz'
                base.validate_report(r,original,sha(cp),p['prior_source_hashes'][str(HERE/'mp2_worker.py')]);c=r['mp2_context']
            if any(c[k]!=v for k,v in context.items() if k not in ('phase','amplitude')):raise ValueError('Prior context mismatch')
            record=base.measured_record(folder,context,old);result.append(record)
    return result


def prepare(runtime,out):
    runtime,out=runtime.resolve(),out.resolve();inputs=out.with_name(out.name+'-inputs');prepared=out.with_name(out.name+'-prepared-private.json')
    if out.is_relative_to(runtime) or any(x.exists() for x in (out,inputs,prepared)):raise ValueError('Never overwrite prior allocation/preparation')
    prior_index=read(ROOT/'outputs/lf1-input-index-private-20261113.json');base.check_pins(prior_index['pins'])
    prior=read(ROOT/'outputs/training-context-mp2-20261113/protocol.json');r=read(ROOT/'outputs/training-context-mp2-20261113/result.json')
    if r['status']!='completed_MP2_training_panel' or not r['lease_released']:raise ValueError('Prior allocation incomplete')
    if Path(prior['runtime_root']).resolve()!=runtime:raise ValueError('Runtime mismatch')
    _,old,_=base.legacy(runtime)
    parent=arrays(runtime/prior['parent_checkpoint']);directions=arrays(old.inputs/'directions.npz')['directions']
    np.testing.assert_array_equal(directions,old.fixed_directions(len(parent['weights'])))
    for k in (-1,1):
        expected=old.apply_delta(parent,k*.01*directions[0]);actual=arrays(old.inputs/f'probe-0-{k}.npz')
        if set(actual)!=set(expected):raise ValueError('Original probe schema mismatch')
        for field in expected:np.testing.assert_array_equal(actual[field],expected[field])
    p=dict(experiment='SS1 finite-probe scale stability',runtime_root=str(runtime),output=str(out),inputs=str(inputs),
        mp2_output=str(ROOT/'outputs/training-context-mp2-20261113'),parent_checkpoint=prior['parent_checkpoint'],parent_sha256=prior['parent_sha256'],
        prior_source_hashes=prior['source_hashes'],prior_hashes=prior_index['pins'],source_hashes={str(HERE/n):sha(HERE/n) for n in SOURCE},
        training_seeds=list(TRAIN),scales=list(SCALES),direction=0,aggregate_seconds=LIMIT,worker_seconds=300,maximum_workers=25,
        signal_floor=SIGNAL,compatibility_threshold=COMPATIBILITY,model_promoted=False,walking_proven=False)
    old_records=reused(p,old)
    if len(old_records)!=18:raise ValueError('Missing reused cases')
    base.check_pins(p['prior_hashes']);out.parent.mkdir(parents=True,exist_ok=True);inputs.mkdir(exist_ok=False)
    for h in NEW_SCALES:
        for k in (-1,1):
            cp=old.apply_delta(parent,k*h*directions[0]);path=inputs/f'probe-{h}-{k}.npz'
            np.savez_compressed(path,**cp);loaded=arrays(path)
            for field in cp:np.testing.assert_array_equal(loaded[field],cp[field])
    p['input_hashes']={str(path):sha(path) for path in inputs.iterdir() if path.is_file()}
    write(prepared,p)
    public=dict(study=p['experiment'],canonical_direction=0,training_contexts=6,reused_parent_and_original_probe_cases=18,
        new_scales=list(NEW_SCALES),maximum_new_workers=25,seconds_per_worker=5,worker_seconds=300,aggregate_seconds=LIMIT,
        signal_floor=SIGNAL,relative_slope_and_midpoint_ratio_threshold=COMPATIBILITY,
        sequence='Exact parent1010replay; all24scale/sign/TRAIN cells; signed independent scale-analysis audit',
        stop='Any source/input/data/created/resource/lease/awake/import/horizon/settings/replay/timeout/containment/budget failure; no retry',
        no_model_selection=True,model_promoted=False,standing_proven=False,walking_proven=False,
        limits='One canonical direction and finite scales, engineering diagnostic thresholds only; no smaller-step remedy or gradient/skill qualification.')
    write(out.with_name(out.name+'-public-protocol.json'),public);print('SS1 prepared; four bounded diagnostic probes, no simulation allocated',flush=True)


def remaining(started):
    if time.monotonic()-started+300>LIMIT:raise TimeoutError('No full worker budget')


def execute(prepared):
    p=read(prepared);runtime=Path(p['runtime_root']);out=Path(p['output']);_,old,AwakeRequest=base.legacy(runtime)
    pins={**p['prior_hashes'],**p['source_hashes'],**p['input_hashes'],str(prepared.resolve()):sha(prepared)};created={}
    base.resources();base.check_pins(pins)
    if out.exists():raise ValueError('No retry/overwrite')
    lock=runtime/'work/Project-Noesis/gpu-pilot/outputs/codex-training-host.lock';token=uuid.uuid4().hex;started=time.monotonic()
    awake=AwakeRequest();empty=True
    with os.fdopen(os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY),'w') as f:json.dump(dict(owner='codex-SS1',pid=os.getpid(),token=token,deadline_unix=time.time()+LIMIT),f)
    result=dict(status='running_SS1',jobs=[],model_promoted=False,standing_proven=False,walking_proven=False)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1',MPLCONFIGDIR=str(out.parent/'matplotlib-cache'),NUMBA_CACHE_DIR=str(out.parent/'numba-cache'))
    try:
        out.mkdir(exist_ok=False);write(out/'protocol.json',p);created[str(out/'protocol.json')]=sha(out/'protocol.json')
        records=reused(p,old)
        with awake:
            for j in jobs():
                remaining(started);base.resources();base.check_pins({**pins,**created});cp=checkpoint(p,j);folder=out/j['name']
                print('START',j['name'],flush=True);empty=False
                args=[sys.executable,str(HERE/'ss1_worker.py'),'--runtime-root',str(runtime),'--output',str(folder),'--checkpoint',str(cp),
                    '--phase',j['phase'],'--bank',j['bank'],'--seed',str(j['seed'])]
                if j['phase']=='trial':args+=['--direction','0','--sign',str(j['sign']),'--amplitude',str(j['amplitude'])]
                try:
                    with (out/(j['name']+'.log')).open('w',encoding='utf-8') as log:receipt=old.native.run_windows_job(args,timeout=300,stdout=log,cwd=ROOT,env=env)
                    empty=receipt['tree_empty']
                except old.native.WatchdogError as e:empty=e.cleanup_verified;raise
                write(out/(j['name']+'-receipt.json'),receipt);result['jobs'].append(dict(**j,receipt=receipt));write(out/'result.json',result)
                if receipt['exit_code']!=0 or receipt['timed_out'] or not empty:raise RuntimeError('Worker failed; no retry')
                report=read(folder/'report.json');validate_report(report,j,sha(cp),p['source_hashes'][str(HERE/'ss1_worker.py')])
                if report['ss1_context']!=read(out/(j['name']+'-ss1-context.json')):raise ValueError('Context mismatch')
                ref=reference(p,j['seed']);old.native.compare(ref,folder)
                if j['phase']=='baseline':
                    for n in ('observations.npz','adhesion-observations.npz','anatomy-observations.npz'):old.native.exact(ref/n,folder/n)
                    if report['spikes_per_world']!=read(ref/'report.json')['spikes_per_world']:raise ValueError('Baseline spikes differ')
                    result['baseline_exact']=True
                else:records.append(base.measured_record(folder,{k:j[k] for k in ('phase','bank','seed','direction','sign','amplitude')},old))
                for path in folder.iterdir():
                    if path.is_file():created[str(path)]=sha(path)
                for suffix in ('-receipt.json','-ss1-context.json','.log'):
                    path=out/(j['name']+suffix);created[str(path)]=sha(path)
                print('COMPLETE',j['name'],flush=True)
            if len(result['jobs'])!=25 or len(records)!=42:raise ValueError('Incomplete panel')
            parent,returns=panel(records);analysis=analyze(parent,returns);write(out/'training-returns.json',records)
            write(out/'scale-analysis-private.json',{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in analysis.items()})
            for n in ('training-returns.json','scale-analysis-private.json'):created[str(out/n)]=sha(out/n)
            result.update(status='completed_SS1',compatible_contexts=analysis['compatible_contexts'],all_contexts_compatible=analysis['all_contexts_compatible'],
                smallest_scale_cross_context_sign_consistent=analysis['smallest_scale_cross_context_sign_consistent'])
            base.check_pins({**pins,**created})
            if time.monotonic()-started>LIMIT:raise TimeoutError('Aggregate budget exceeded')
    except BaseException as e:result.update(status='failed_SS1',error_type=type(e).__name__,error=str(e));raise
    finally:
        guards=True
        try:base.check_pins({**pins,**created})
        except (OSError,ValueError):guards=False
        if empty and lock.exists() and read(lock).get('token')==token:lock.unlink()
        result.update(seconds=time.monotonic()-started,source_input_data_created_unchanged=guards,process_tree_empty_verified=empty,
            lease_released=not lock.exists(),awake_request_released=awake.released,awake_request_acquired=awake.previous is not None,created_hashes=created)
        if out.exists():write(out/'result.json',result)
        print('RESULT',result['status'],result['seconds'],flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='mode',required=True)
    a=sub.add_parser('prepare');a.add_argument('--runtime-root',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    a=sub.add_parser('run');a.add_argument('--prepared',type=Path,required=True);a=ap.parse_args()
    prepare(a.runtime_root,a.output) if a.mode=='prepare' else execute(a.prepared)
