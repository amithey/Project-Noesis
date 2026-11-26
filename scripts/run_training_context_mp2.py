"""Bounded MP2 allocation with immutable legacy evidence and explicit new outputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import uuid
import numpy as np
import psutil
from mp2_select import OLD,NEW,TRAIN,key,panel,select

HERE=Path(__file__).resolve().parent
SOURCE=('mp2_select.py','mp2_worker.py','run_training_context_mp2.py','audit_training_context_mp2.py',
        'test_mp2_select.py','test_mp2_orchestration.py')
LIMIT=11100
FILES=('report.json','observations.npz','anatomy-observations.npz','anatomy-model.json')


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while block:=f.read(1024*1024):h.update(block)
    return h.hexdigest()


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def write(p,value):Path(p).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def arrays(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def legacy(root):
    sys.path.insert(0,str(root/'work'))
    import audit_mp1_interruption as prefix
    import run_measured_mp1 as old
    from awake_guard import AwakeRequest
    return prefix,old,AwakeRequest


def jobs():
    result=[dict(name='baseline-1010',phase='baseline',bank='parent',seed=1010,direction=None,sign=None)]
    for s in NEW:
        result.append(dict(name=f'parent-{s}',phase='parent',bank='parent',seed=s,direction=None,sign=None))
        result += [dict(name=f'trial-{d}-{s}-{k}',phase='trial',bank='probe',seed=s,direction=d,sign=k)
            for d in range(4) for k in (-1,1)]
    return result


def checkpoint(p,job):
    root=Path(p['runtime_root'])
    return root/p['parent_checkpoint'] if job['bank']=='parent' else root/'outputs/measured-policy-mp1-inputs-20261110'/f"probe-{job['direction']}-{job['sign']}.npz"


def inventory(root):
    entries=[];found=[]
    scopes=('outputs','work/evidence','work/Project-Noesis/gpu-pilot/outputs','work/Noesis-Sensory/gpu-pilot/outputs')
    for base in [*[root/s for s in scopes],HERE.parent/'outputs']:
        if not base.exists():continue
        for folder,dirs,files in os.walk(base):
            dirs[:]=[d for d in dirs if d not in ('brain-cache','__pycache__','.git','node_modules')]
            for name in files:
                if name not in ('report.json','protocol.json') and not name.endswith('-context.json'):continue
                path=Path(folder)/name;v=read(path);seeds=[]
                def visit(x):
                    if isinstance(x,dict):
                        for k,value in x.items():
                            if 'seed' in k.lower():
                                for s in value if isinstance(value,list) else [value]:
                                    if type(s) is int and s in NEW:seeds.append(s)
                            if isinstance(value,(dict,list)):visit(value)
                    elif isinstance(x,list):
                        for value in x:
                            if isinstance(value,(dict,list)):visit(value)
                visit(v);entries.append(dict(path=str(path),sha256=sha(path)))
                if seeds:found.append(str(path))
    if found:raise ValueError('Proposed training seeds already declared/used in available inventory; amend protocol, never substitute silently')
    return dict(files=entries,matches=found,scopes=list(scopes),
        limits='Available local report/context/protocol headers only, not a complete historical guarantee')


def prepare(root,out):
    root,out=root.resolve(),out.resolve()
    if out.is_relative_to(root) or out.exists():raise ValueError('Require new output outside immutable runtime/evidence')
    prefix,old,_=legacy(root);verified=prefix.verify();oldp=read(old.out/'protocol.json')
    source={str(HERE/n):sha(HERE/n) for n in SOURCE}
    inv=inventory(root)
    oldpins={str(root/n):h for n,h in verified['original_hashes'].items()}
    data={str(root/'work/NoesisData'/e['path']):e['sha256'] for e in read(root/'outputs/noesis-required-data-sha256.json')['files']}
    p=dict(experiment='MP2 training-context expansion',runtime_root=str(root),output=str(out),
        source_hashes=source,old_hashes=oldpins,data_hashes=data,inventory=inv,
        parent_checkpoint=oldp['parent_checkpoint'],parent_sha256=oldp['parent_sha256'],
        training_seeds=list(TRAIN),new_training_seeds=list(NEW),worker_seconds=300,aggregate_seconds=LIMIT,
        maximum_workers=37,seconds_per_worker=5,objective='physical-standing-ps1-v1',
        stop='Any source/input/data/created/resource/lease/awake/import/horizon/settings/replay/timeout/containment/budget failure; no retry',
        model_promoted=False,walking_proven=False)
    check_pins({**oldpins,**source,**data});out.parent.mkdir(parents=True,exist_ok=True)
    target=out.with_name(out.name+'-prepared-private.json')
    if target.exists():raise ValueError('Never overwrite prepared allocation')
    write(target,p);print('MP2 prepared; no simulation allocated',flush=True)


def check_pins(pins):
    for path,h in pins.items():
        if sha(path)!=h:raise ValueError('Signed source/data/input/created file changed')


def validate_report(report,job,cp_hash,worker_hash):
    expected=dict(status='completed_transport_pilot',controller='brain',worlds=1,learning=False,motor_exploration=0,
        weights_unchanged=True,requested_simulated_seconds_per_world=5,simulated_seconds_per_world=5,
        completed_control_chunks=1000,control_ms=5.,physics_dt=.0001,mp2_worker_completed=True)
    if any(report.get(k)!=v for k,v in expected.items()):raise ValueError('Incomplete native MP2 episode')
    c=report['mp2_context'];key(c,baseline=job['phase']=='baseline')
    if any(c[k]!=job[k] for k in ('phase','bank','seed','direction','sign')):raise ValueError('Wrong context')
    if (c['checkpoint_sha256']!=cp_hash or report['checkpoint_sha256']!=cp_hash or c['worker_sha256']!=worker_hash
            or report['seeds']!=[job['seed']] or c.get('worker_completed') is not True
            or c.get('frozen_native_neural_motor_execution') is not True or c.get('online_learning') is not False
            or len(c['import_records'])!=1):raise ValueError('Wrong native provenance')
    imported=c['import_records'][0]
    if not all(imported.get(k) is True for k in ('actual_import_exact','topology_unchanged','nonselected_weights_unchanged')):
        raise ValueError('Actual full-graph import not verified')


def measured_record(directory,context,old):
    r=read(directory/'report.json');o=arrays(directory/'observations.npz');a=arrays(directory/'anatomy-observations.npz')
    m=read(directory/'anatomy-model.json');reward=old.episode_rewards(o,a,m)
    hashes={n:sha(directory/n) for n in FILES}
    digest=hashlib.sha256(json.dumps(hashes,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return dict(context=context,**{'return':float(reward[10:].mean())},complete_horizon=True,actual_import_verified=True,
        checkpoint_sha256=r['checkpoint_sha256'],return_source_sha256=digest,source_hashes=hashes,
        warmup_samples=10,measured_samples=990,return_objective='physical-standing-ps1-v1')


def reused_records(root,old):
    records=[]
    for s in OLD:
        jobs0=[dict(phase='parent',bank='parent',seed=s,direction=None,sign=None)]
        jobs0 += [dict(phase='trial',bank='probe',seed=s,direction=d,sign=k) for d in range(4) for k in (-1,1)]
        for j in jobs0:
            folder=old.out/(f'parent-{s}' if j['phase']=='parent' else f"trial-{j['direction']}-{s}-{j['sign']}")
            report=read(folder/'report.json')
            from mp1_contract import validate_report as original_validate
            original_validate(report)
            records.append(measured_record(folder,j,old))
    return records


def resources():
    me=psutil.Process();own={me.pid,*[p.pid for p in me.parents()]}
    if any(p.pid not in own and (p.info['name'] or '').lower() in ('python.exe','pythonw.exe') for p in psutil.process_iter(['name'])):
        raise RuntimeError('Competing Python')
    if psutil.virtual_memory().available<2*2**30 or psutil.disk_usage(str(HERE)).free<2*2**30:
        raise RuntimeError('RAM/disk below2GiB')


def remaining(started):
    if time.monotonic()-started+300>LIMIT:raise TimeoutError('No budget for a complete worker')


def save_selection(p,out,records,old):
    parent,probes=panel(records)
    original=read(old.out/'selection.json')
    for bank,sign in (('candidate',1),('negative',-1)):
        replay=old.select_policy(parent[:2],probes[:2],sign)
        if replay!=original[bank]:raise ValueError('Original two-context decision not reproduced')
    selected={b:select(parent,probes,s) for b,s in (('candidate',1),('negative',-1))}
    write(out/'training-returns.json',records);write(out/'selection.json',selected)
    state=out/'optimizer-state.npz'
    if state.exists():raise ValueError('Never overwrite optimizer state')
    np.savez_compressed(state,parent_returns=parent,policy_returns=probes,training_seeds=np.array(TRAIN),
        selection_json=np.array(json.dumps(selected,sort_keys=True)),parent_sha256=np.array(p['parent_sha256']),
        return_records_json=np.array(json.dumps(records,sort_keys=True)))
    loaded=arrays(state)
    np.testing.assert_array_equal(loaded['parent_returns'],parent);np.testing.assert_array_equal(loaded['policy_returns'],probes)
    if json.loads(str(loaded['return_records_json']))!=records:raise ValueError('Return lineage reload mismatch')
    for bank,s in (('candidate',1),('negative',-1)):
        if select(loaded['parent_returns'],loaded['policy_returns'],s)!=selected[bank]:raise ValueError('Selection reload mismatch')
        i=selected[bank]['policy_index'];d,k=divmod(i,2)
        cp=checkpoint(p,dict(bank='parent' if i<0 else 'probe',direction=d,sign=-1 if k==0 else 1))
        target=out/(bank+'.npz')
        if target.exists():raise ValueError('Never overwrite checkpoint')
        shutil.copyfile(cp,target)
        if sha(cp)!=sha(target):raise ValueError('Selected checkpoint identity mismatch')
        write(target.with_suffix('.model.json'),dict(experiment='MP2',bank=bank,selection=selected[bank],
            checkpoint_sha256=sha(target),common_parent_sha256=p['parent_sha256'],measured_policy_source_sha256=sha(cp),
            native_online_learning=False,historical_compatibility_fields_not_optimizer_state=True,
            training_cases=54,training_contexts=6,model_promoted=False,standing_proven=False,walking_proven=False))
    index=selected['candidate']['policy_index']
    return dict(candidate_parent_fallback=index<0,candidate_identity_changed=index!=original['candidate']['policy_index'],
        independent_physical_evaluation_required=index>=0 and index!=original['candidate']['policy_index'],
        training_acceptance_not_efficacy=True,model_promoted=False,standing_proven=False,walking_proven=False)


def execute(prepared):
    prepared=prepared.resolve();prepared_hash=sha(prepared);p=read(prepared)
    root,out=Path(p['runtime_root']),Path(p['output']);_,old,AwakeRequest=legacy(root)
    pins={**p['old_hashes'],**p['data_hashes'],**p['source_hashes'],str(prepared):prepared_hash};created={}
    resources();check_pins(pins)
    if out.exists():raise ValueError('Never retry/overwrite allocation')
    inventory(root) # Same available-scope unused-seed gate immediately before lease.
    lock=root/'work/Project-Noesis/gpu-pilot/outputs/codex-training-host.lock';token=uuid.uuid4().hex
    started=time.monotonic();awake=AwakeRequest();empty=True
    with os.fdopen(os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY),'w') as f:
        json.dump(dict(owner='codex-MP2',pid=os.getpid(),token=token,deadline_unix=time.time()+LIMIT),f)
    result=dict(status='running_MP2',jobs=[],model_promoted=False,standing_proven=False,walking_proven=False)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1',MPLCONFIGDIR=str(out.parent/'matplotlib-cache'),
        NUMBA_CACHE_DIR=str(out.parent/'numba-cache'))
    try:
        out.mkdir(exist_ok=False);write(out/'protocol.json',p);created[str(out/'protocol.json')]=sha(out/'protocol.json')
        records=reused_records(root,old)
        with awake:
            for j in jobs():
                remaining(started);resources();check_pins({**pins,**created})
                cp=checkpoint(p,j);folder=out/j['name'];print('START',j['name'],flush=True);empty=False
                args=[sys.executable,str(HERE/'mp2_worker.py'),'--runtime-root',str(root),'--output',str(folder),
                    '--checkpoint',str(cp),'--phase',j['phase'],'--bank',j['bank'],'--seed',str(j['seed'])]
                if j['direction'] is not None:args+=['--direction',str(j['direction']),'--sign',str(j['sign'])]
                try:
                    with (out/(j['name']+'.log')).open('w',encoding='utf-8') as log:
                        receipt=old.native.run_windows_job(args,timeout=300,stdout=log,cwd=out.parent.parent,env=env)
                    empty=receipt['tree_empty']
                except old.native.WatchdogError as e:
                    empty=e.cleanup_verified;raise
                write(out/(j['name']+'-receipt.json'),receipt)
                result['jobs'].append(dict(**j,receipt=receipt));write(out/'result.json',result)
                if receipt['exit_code']!=0 or receipt['timed_out'] or not empty:raise RuntimeError('Worker failed; no retry')
                report=read(folder/'report.json');validate_report(report,j,sha(cp),p['source_hashes'][str(HERE/'mp2_worker.py')])
                if report['mp2_context']!=read(out/(j['name']+'-mp2-context.json')):raise ValueError('Context mismatch')
                reference=old.out/'parent-1010' if j['phase']=='baseline' else out/f"parent-{j['seed']}"
                old.native.compare(reference,folder)
                if j['phase']=='baseline':
                    for n in ('observations.npz','adhesion-observations.npz','anatomy-observations.npz'):old.native.exact(reference/n,folder/n)
                    if report['spikes_per_world']!=read(reference/'report.json')['spikes_per_world']:raise ValueError('Baseline spike mismatch')
                    result['baseline_exact']=True
                else:records.append(measured_record(folder,{k:j[k] for k in ('phase','bank','seed','direction','sign')},old))
                for path in folder.iterdir():
                    if path.is_file():created[str(path)]=sha(path)
                for suffix in ('-receipt.json','-mp2-context.json','.log'):
                    path=out/(j['name']+suffix);created[str(path)]=sha(path)
                print('COMPLETE',j['name'],flush=True)
            if len(result['jobs'])!=37 or len(records)!=54:raise ValueError('Incomplete allocation')
            result['science']=save_selection(p,out,records,old)
            for n in ('training-returns.json','selection.json','optimizer-state.npz','candidate.npz','candidate.model.json','negative.npz','negative.model.json'):
                created[str(out/n)]=sha(out/n)
            check_pins({**pins,**created})
            if time.monotonic()-started>LIMIT:raise TimeoutError('Aggregate budget exceeded')
            result['status']='completed_MP2_training_panel'
    except BaseException as e:
        result.update(status='failed_MP2',error_type=type(e).__name__,error=str(e));raise
    finally:
        guards=True
        try:check_pins({**pins,**created})
        except (OSError,ValueError):guards=False
        if empty and lock.exists() and read(lock).get('token')==token:lock.unlink()
        result.update(seconds=time.monotonic()-started,source_input_data_created_unchanged=guards,
            process_tree_empty_verified=empty,lease_released=not lock.exists(),awake_request_released=awake.released,
            awake_request_acquired=awake.previous is not None,created_hashes=created)
        if out.exists():write(out/'result.json',result)
        print('RESULT',result['status'],result['seconds'],flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='mode',required=True)
    a=sub.add_parser('prepare');a.add_argument('--runtime-root',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    a=sub.add_parser('run');a.add_argument('--prepared',type=Path,required=True)
    args=ap.parse_args()
    prepare(args.runtime_root,args.output) if args.mode=='prepare' else execute(args.prepared)
