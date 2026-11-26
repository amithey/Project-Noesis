"""SR1 independent training births with matched own-seed and shared development probes."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import uuid
import zipfile
import numpy as np
import psutil
from s2_contract import validate_training_report
from sr1_contract import replication_summary

root=Path(__file__).resolve().parents[1];out=root/'outputs/training-seed-sr1-20261106'
pilot=root/'work/Noesis-Sensory/gpu-pilot/src';core=root/'work/Noesis-Core/src'
sys.path[:0]=[str(pilot),str(core)]
from noesis_gpu.evaluate import summarize,compare
from noesis_gpu.retention_audit import compare_checkpoints
from noesis_gpu.process_watchdog import run_windows_job,WatchdogError


def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while b:=f.read(1024*1024):h.update(b)
    return h.hexdigest()
def sources():
    return {str(f.relative_to(root)):{p.relative_to(f).as_posix():sha(p) for p in sorted(f.rglob('*.py'))} for f in (pilot,core)}|{
        'scripts':{n:sha(root/'work'/n) for n in ('sr1_contract.py','run_training_seed_sr1.py','s2_worker.py','s2_contract.py','adhesion_mapping.py')}}
own={os.getpid()}|{p.pid for p in psutil.Process().parents()}
def resources():
    if any(p.pid not in own and (p.info['name'] or '').lower() in ('python.exe','pythonw.exe') for p in psutil.process_iter(['name'])):
        raise RuntimeError('Competing Python')
    ram=psutil.virtual_memory().available/2**30
    if ram<2:raise RuntimeError('RAM below2GiB')
    return ram


a1=root/'outputs/adhesion-sign-a1-20261106';s2=root/'outputs/standing-reacquisition-s2-20261106';x1=root/'outputs/exploration-transfer-x1-20261106'
assert read(s2/'result.json')['status']=='completed_S2' and read(x1/'result.json')['status']=='completed_X1'
for n,h in read(s2/'protocol.json')['source_hashes']['scripts'].items():assert sha(root/'work'/n)==h
provenance=read(root/'outputs/thermo-pilot-tp2-20261106-source.json')
for label,folder in [('pilot',pilot),('core',core)]:
    assert len(list(folder.rglob('*.py')))==len(provenance[label])
    for f in provenance[label]:assert sha(folder/f['path'])==f['sha256']
parent=root/'work/evidence/s1-restored/gpu-pilot/outputs/e4-exploration-20261029/train-control-1000/weights-last.npz'
assert sha(parent)==read(s2/'protocol.json')['parent_sha256']
dev=[10006,10007,10010,10011];prior={};parent_dev={};children_dev={1004:{}};own_pairs={};parent_dirs={}
for directory,artifact,selected in [
    (a1,'adhesion-sign-a1-20261106',[a1/f'legacy-{s}'/n for s in dev for n in ('report.json','observations.npz')]),
    (s2,'standing-reacquisition-s2-20261106',[s2/f'child-legacy-{s}'/n for s in dev for n in ('report.json','observations.npz')]),
    (x1,'exploration-transfer-x1-20261106',[x1/f'probe-{b}-legacy-off'/n for b in ('parent','child') for n in ('report.json','observations.npz')])]:
    receipt=read(root/'outputs'/f'{artifact}-artifact.json');archive=root/'outputs'/receipt['path'];assert sha(archive)==receipt['sha256']
    with zipfile.ZipFile(archive) as z:
        manifest=json.loads(z.read('manifest.json'))
        for f in selected:
            h=manifest[f.relative_to(root/'outputs').as_posix()]['sha256'];assert sha(f)==h;prior[str(f.relative_to(root))]=h
for seed in dev:
    parent_dirs[seed]=a1/f'legacy-{seed}'
    _,parent_dev[seed]=summarize(parent_dirs[seed]);_,children_dev[1004][seed]=summarize(s2/f'child-legacy-{seed}')
    compare(parent_dirs[seed],s2/f'child-legacy-{seed}')
own_pairs[1004]=(summarize(x1/'probe-parent-legacy-off')[1],summarize(x1/'probe-child-legacy-off')[1])
prior[str(parent.relative_to(root))]=sha(parent)
data=read(root/'outputs/noesis-required-data-sha256.json')['files'];source=sources();child_paths={};child_hashes={}
resources();out.mkdir(exist_ok=False)
lock=root/'work/Project-Noesis/gpu-pilot/outputs/codex-training-host.lock';token=uuid.uuid4().hex
with os.fdopen(os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY),'w') as f:
    json.dump(dict(owner='codex',pid=os.getpid(),token=token,start_unix=time.time(),deadline_unix=time.time()+3000),f)
started=time.monotonic();empty=True
protocol=dict(experiment='SR1 training-seed sensitivity replication',date='2026-11-06',source_hashes=source,prior_hashes=prior,
    parent_sha256=sha(parent),prior_training_seed=1004,new_training_seeds=[1005,1006],development_seeds=dev,
    training_seconds=25,frozen_seconds=5,training_updates=4990,learning_rate=.2,exploration_mv_ms=1.5,
    eligibility_ms=100,original_weight_bound=.5,policy='legacy-positive-Femur',reward='curriculum-standing-v7',
    aggregate_seconds=3000,training_worker_seconds=600,frozen_worker_seconds=300,maximum_new_jobs=14,
    analysis='all three births, own-seed vs shared-development gains, original strict standing/slip/proxy retention screen; no selection',
    stop='technical/hash/horizon/log/context/checkpoint/resource/timeout failure or insufficient full worker budget; no retry/retune/partial interpretation',
    model_promoted=False,walking_proven=False)
result=dict(status='running',jobs=[],training={},frozen=[],replication=None,model_promoted=False,walking_proven=False)
env=dict(os.environ,PYTHONUTF8='1',MPLCONFIGDIR=str(root/'work/matplotlib-cache'))
def unchanged():
    return sources()==source and all(sha(root/n)==h for n,h in prior.items()) and all(sha(p)==child_hashes[s] for s,p in child_paths.items())


def launch(phase,seed,cp,name,training_seed):
    global empty
    assert unchanged()
    for f in data:assert sha(root/'work/NoesisData'/f['path'])==f['sha256']
    ram=resources();budget=600 if phase=='train' else 300
    if time.monotonic()-started>3000-budget:raise RuntimeError('Insufficient remaining worker budget')
    directory=out/name
    write(out/(name+'-experiment-context.json'),dict(experiment='SR1',training_seed=training_seed,phase=phase,seed=seed,
        checkpoint_sha256=sha(cp),worker_sha256=source['scripts']['s2_worker.py'],policy='legacy-positive-Femur'))
    print('START',name,flush=True);empty=False
    try:
        with (out/(name+'.log')).open('w',encoding='utf-8') as log:
            receipt=run_windows_job([sys.executable,str(root/'work/s2_worker.py'),'--phase',phase,'--mode','legacy','--seed',str(seed),
                '--root',str(root),'--checkpoint',str(cp),'--output',str(directory)],timeout=budget,stdout=log,cwd=root,env=env)
        empty=receipt['tree_empty']
    except WatchdogError as error:
        empty=error.cleanup_verified;raise
    write(out/(name+'-receipt.json'),receipt)
    result['jobs'].append(dict(name=name,phase=phase,seed=seed,training_seed=training_seed,receipt=receipt,available_ram_before_gib=ram));write(out/'result.json',result)
    assert empty and receipt['exit_code']==0 and not receipt['timed_out']
    report=read(directory/'report.json');assert report['adhesion_probe_mode']=='legacy' and report['s2_phase']==phase
    assert (report['seed']==seed if phase=='train' else report['seeds']==[seed])
    assert report['import_checkpoint_sha256' if phase=='train' else 'checkpoint_sha256']==sha(cp)
    with np.load(directory/'adhesion-observations.npz',allow_pickle=False) as z:
        count=5000 if phase=='train' else 1000
        assert z['desired'].shape==(count,42) and z['pads'].shape==z['Femur_delta'].shape==(count,6)
        assert all(np.isfinite(z[k]).all() for k in z.files)
        np.testing.assert_array_equal((z['desired']-report['body']['neutral_pose']).reshape(count,6,7)[:,:,3],z['Femur_delta'])
        np.testing.assert_array_equal(z['pads'],(z['Femur_delta']<=.08).astype(float))
    return directory,report


try:
    write(out/'protocol.json',protocol)
    for train_seed in (1005,1006):
        parent_dir,_=launch('frozen',train_seed,parent,f'parent-own-{train_seed}',train_seed)
        own_parent=summarize(parent_dir)[1]
        directory,report=launch('train',train_seed,parent,f'train-{train_seed}',train_seed)
        validate_training_report(report);assert report['max_observed_exploration']<=1.5
        with np.load(directory/'training-observations.npz',allow_pickle=False) as z:
            assert all(len(z[k])==5000 and np.isfinite(z[k]).all() for k in z.files)
            np.testing.assert_array_equal(z['learning_active'],np.arange(5000)>=10)
        with np.load(directory/'exploration-log.npz',allow_pickle=False) as z:
            np.testing.assert_array_equal(z['control_chunk_index'],np.arange(10,5000))
            assert z['drive'].shape[0]==z['credit'].shape[0]==4990 and all(np.isfinite(z[k]).all() for k in z.files)
            with np.load(s2/'train-legacy-1004/exploration-log.npz',allow_pickle=False) as old:
                assert not np.array_equal(z['drive'],old['drive'])
        cp=directory/'weights-last.npz';assert sha(cp)==report['saved_checkpoint_sha256']
        check=compare_checkpoints(parent,cp);assert check['compatible'] and check['parameters']['match'] and check['rule']['match']
        with np.load(parent,allow_pickle=False) as old,np.load(cp,allow_pickle=False) as new:
            for k in ('version','normalize_reward_error','bias_correct_reward_error'):np.testing.assert_array_equal(old[k],new[k])
        sidecar=read(cp.with_suffix('.model.json'))
        for k,v in report.items():assert sidecar[k]==v
        child_paths[train_seed]=cp;child_hashes[train_seed]=sha(cp)
        result['training'][str(train_seed)]=dict(checkpoint_sha256=sha(cp),report_sha256=sha(directory/'report.json'),checkpoint_audit=check,
            observations_sha256=sha(directory/'training-observations.npz'),exploration_sha256=sha(directory/'exploration-log.npz'),
            standing_fraction=report['standing_fraction'])
        children_dev[train_seed]={};write(out/'result.json',result)
        print('BIRTH_COMPLETE',train_seed,sha(cp),flush=True)
        for seed in [train_seed,*dev]:
            folder,report=launch('frozen',seed,cp,f'child-{train_seed}-{seed}',train_seed)
            summary=summarize(folder)[1];baseline=parent_dir if seed==train_seed else parent_dirs[seed]
            paired=compare(baseline,folder)
            if seed==train_seed:own_pairs[train_seed]=(own_parent,summary)
            else:children_dev[train_seed][seed]=summary
            result['frozen'].append(dict(training_seed=train_seed,seed=seed,own_seed=seed==train_seed,summary=summary,paired=paired,
                report_sha256=sha(folder/'report.json'),observations_sha256=sha(folder/'observations.npz'),pads_sha256=sha(folder/'adhesion-observations.npz')))
            write(out/'result.json',result);print('PROBE_COMPLETE',train_seed,seed,'standing',summary['standing_fraction'],flush=True)
    result['replication']=replication_summary(parent_dev,children_dev,own_pairs);result['status']='completed_SR1'
except BaseException as error:
    result.update(status='failed',error_type=type(error).__name__,error=str(error));raise
finally:
    result.update(seconds=time.monotonic()-started,source_parent_prior_children_unchanged=unchanged(),
        data_unchanged=all(sha(root/'work/NoesisData'/f['path'])==f['sha256'] for f in data),process_tree_empty_verified=empty)
    if empty and lock.exists() and read(lock)['token']==token:lock.unlink()
    result['lease_released']=not lock.exists();write(out/'result.json',result)
    print('RESULT',result['status'],result['seconds'],flush=True)
