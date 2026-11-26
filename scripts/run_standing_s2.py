"""S2 paired standing reacquisition, one canonical lease and mapping-matched frozen retention."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import uuid

import numpy as np
import psutil
from s2_contract import validate_training_report,standing_retention

root=Path(__file__).resolve().parents[1];pilot=root/'work/Noesis-Sensory';core=root/'work/Noesis-Core/src'
data=root/'work/NoesisData';out=root/'outputs/standing-reacquisition-s2-20261106'
parent=root/'work/evidence/s1-restored/gpu-pilot/outputs/e4-exploration-20261029/train-control-1000/weights-last.npz'
expected='9fd7d41177bf2c745809401c3b142a832661d1a18d0b117b30cb02aea63fea3c'
seeds=[10006,10007,10010,10011];modes=['legacy','corrected'];train_seed=1004
sys.path[:0]=[str(pilot/'gpu-pilot/src'),str(core)]
from noesis_gpu.evaluate import summarize,compare,MATCHED_SETTINGS
from noesis_gpu.retention_audit import compare_checkpoints
from noesis_gpu.process_watchdog import run_windows_job,WatchdogError


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while chunk:=f.read(1024*1024):h.update(chunk)
    return h.hexdigest()


def tree(path):
    h=hashlib.sha256()
    for p in sorted(path.rglob('*.py')):
        h.update(p.relative_to(path).as_posix().encode());h.update(p.read_bytes())
    return h.hexdigest()


def sources():
    return dict(pilot=tree(pilot/'gpu-pilot/src'),core=tree(core),scripts={n:sha(root/'work'/n) for n in
                 ('adhesion_mapping.py','s2_contract.py','s2_worker.py','run_standing_s2.py')})


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def exact(a,b):
    with np.load(a,allow_pickle=False) as x,np.load(b,allow_pickle=False) as y:
        return set(x.files)==set(y.files) and all(np.array_equal(x[k],y[k]) for k in x.files)


own={os.getpid()}|{p.pid for p in psutil.Process().parents()}


def resources():
    for p in psutil.process_iter(['pid','name']):
        if p.info['pid'] not in own and (p.info['name'] or '').lower() in ('python.exe','pythonw.exe'):
            raise RuntimeError('Competing Python; no new worker')
    ram=psutil.virtual_memory().available/2**30
    if ram<2:raise RuntimeError('Available RAM below2GiB')
    return ram


manifest=json.loads((root/'outputs/noesis-required-data-sha256.json').read_text())['files']
provenance=json.loads((root/'outputs/thermo-pilot-tp2-20261106-source.json').read_text())
for label,folder in [('pilot',pilot/'gpu-pilot/src'),('core',core)]:
    assert len(list(folder.rglob('*.py')))==len(provenance[label])
    for f in provenance[label]:assert sha(folder/f['path'])==f['sha256']
assert sha(parent)==expected
a1=root/'outputs/adhesion-sign-a1-20261106'
a1_result=json.loads((a1/'result.json').read_text());assert a1_result['status']=='completed_A1'
assert sha(root/'work/adhesion_mapping.py')==json.loads((a1/'protocol.json').read_text())['source_hashes']['scripts']['adhesion_mapping.py']
hashes=sources();baselines={m:{} for m in modes};baseline_reports={m:{} for m in modes}
for mode in modes:
    for seed in seeds:
        previous=next(r for r in a1_result['cases'] if r['mode']==mode and r['seed']==seed)
        folder=a1/f'{mode}-{seed}'
        assert sha(folder/'report.json')==previous['report_sha256'] and sha(folder/'observations.npz')==previous['observations_sha256']
        report,summary=summarize(folder);baseline_reports[mode][seed]=report;baselines[mode][seed]=summary
        assert summary==previous['summary'] and report['checkpoint_sha256']==expected
resources();out.mkdir(exist_ok=False)
lock=root/'work/Project-Noesis/gpu-pilot/outputs/codex-training-host.lock';token=uuid.uuid4().hex
with os.fdopen(os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY),'w') as f:
    json.dump(dict(owner='codex',pid=os.getpid(),token=token,start_unix=time.time(),deadline_unix=time.time()+2400),f)
started=time.monotonic();empty=True
protocol=dict(experiment='S2 paired standing reacquisition after pad-sign change',date='2026-11-06',
              runtime_commit='e0cbd109bb6e018bd253c9f1b5e1698e0421b276',core_commit='42c20769376bda5694b4c1c66ca75e52cbd9f1f4',
              source_hashes=hashes,parent_sha256=expected,modes=modes,training_seed=train_seed,training_seconds=25,
              training_chunks=5000,learning_updates=4990,frozen_seeds=seeds,frozen_seconds=5,
              learning_rate=.2,eligibility_ms=100,exploration_mv_ms=1.5,reward='curriculum-standing-v7',
              bridge_gate='both frozen parent modes10006 exactly replay A1 body AND pad arrays before training',
              stream_gate='both complete25s births have exactly equal drive/credit/index/time exploration logs',
              stage_gate='EACH child seed standing>=.90 AND>=own mapping parent,slip<=parent; median paired heading/net/cycle/tripod gains>=0',
              aggregate_budget_seconds=2400,training_worker_budget_seconds=600,frozen_worker_budget_seconds=300,
              stop='technical/horizon/log/context/hash/resource/timeout failure; no retry/retune/partial-bank probes or insufficient remaining worker budget',
              model_promoted=False,walking_proven=False,one_training_pair_only=True)
result=dict(status='running',jobs=[],bridge_gate={},training={},frozen=[],stage_gates={},model_promoted=False,walking_proven=False)
children={};child_hashes={};env=dict(os.environ,PYTHONUTF8='1',MPLCONFIGDIR=str(root/'work/matplotlib-cache'))


def launch(phase,mode,seed,checkpoint,name):
    global empty
    budget=600 if phase=='train' else 300
    if time.monotonic()-started>2400-budget:raise RuntimeError('Insufficient remaining job budget')
    if sources()!=hashes or sha(parent)!=expected:raise RuntimeError('Source/parent drift')
    for m,p in children.items():
        if sha(p)!=child_hashes[m]:raise RuntimeError('Completed child changed')
    for f in manifest:
        if sha(data/f['path'])!=f['sha256']:raise RuntimeError('Data changed')
    ram=resources();target=out/name
    if time.monotonic()-started>2400-budget:raise RuntimeError('Insufficient remaining budget after preflight')
    print('START',phase,mode,seed,flush=True);empty=False
    try:
        with (out/f'{name}.log').open('w',encoding='utf-8') as log:
            receipt=run_windows_job([sys.executable,str(root/'work/s2_worker.py'),'--phase',phase,'--mode',mode,
                          '--seed',str(seed),'--root',str(root),'--checkpoint',str(checkpoint),'--output',str(target)],
                          timeout=budget,stdout=log,cwd=root,env=env)
        empty=receipt['tree_empty']
    except WatchdogError as error:
        empty=error.cleanup_verified;raise
    write(out/f'{name}-receipt.json',receipt)
    result['jobs'].append(dict(name=name,phase=phase,mode=mode,seed=seed,receipt=receipt,available_ram_before_gib=ram))
    write(out/'result.json',result)
    assert empty and not receipt['timed_out'] and receipt['exit_code']==0
    report=json.loads((target/'report.json').read_text())
    assert report['s2_phase']==phase and report['adhesion_probe_mode']==mode
    assert report['seed']==seed if phase=='train' else report['seeds']==[seed]
    assert report['import_checkpoint_sha256' if phase=='train' else 'checkpoint_sha256']==sha(checkpoint)
    with np.load(target/'adhesion-observations.npz',allow_pickle=False) as z:
        count=5000 if phase=='train' else 1000
        assert z['desired'].shape==(count,42) and z['pads'].shape==z['Femur_delta'].shape==(count,6)
        assert all(np.isfinite(z[k]).all() for k in z.files)
        np.testing.assert_array_equal(z['pads'],((z['Femur_delta'] if mode=='legacy' else -z['Femur_delta'])<=.08).astype(float))
    return target,report


try:
    write(out/'protocol.json',protocol)
    for mode in modes:
        folder,report=launch('frozen',mode,10006,parent,f'bridge-{mode}-10006')
        for file in ('observations.npz','adhesion-observations.npz'):
            assert exact(folder/file,a1/f'{mode}-10006'/file),'Bridge parity failed: '+file
        compare(a1/f'{mode}-10006',folder)
        result['bridge_gate'][mode]=True;write(out/'result.json',result)
    for mode in modes:
        folder,report=launch('train',mode,train_seed,parent,f'train-{mode}-{train_seed}')
        validate_training_report(report)
        assert report['import_checkpoint_sha256']==expected and report['max_observed_exploration']<=1.5
        with np.load(folder/'training-observations.npz',allow_pickle=False) as z:
            assert all(len(z[k])==5000 and np.isfinite(z[k]).all() for k in z.files)
            np.testing.assert_array_equal(z['learning_active'],np.arange(5000)>=10)
        with np.load(folder/'exploration-log.npz',allow_pickle=False) as z:
            np.testing.assert_array_equal(z['control_chunk_index'],np.arange(10,5000))
            assert z['drive'].shape[0]==z['credit'].shape[0]==4990 and all(np.isfinite(z[k]).all() for k in z.files)
        cp=folder/'weights-last.npz';assert cp.exists() and sha(cp)==report['saved_checkpoint_sha256']
        audit=compare_checkpoints(parent,cp)
        assert audit['compatible'] and audit['parameters']['match'] and audit['rule']['match']
        with np.load(parent,allow_pickle=False) as a,np.load(cp,allow_pickle=False) as b:
            for key in ('version','normalize_reward_error','bias_correct_reward_error'):
                np.testing.assert_array_equal(a[key],b[key])
        sidecar=json.loads(cp.with_suffix('.model.json').read_text())
        for key in ('status','completed_chunks','learning_updates','body','saved_checkpoint_sha256','reward_objective'):
            assert sidecar[key]==report[key], 'Stale checkpoint context: '+key
        children[mode]=cp;child_hashes[mode]=sha(cp)
        result['training'][mode]=dict(checkpoint_sha256=child_hashes[mode],report_sha256=sha(folder/'report.json'),
                         checkpoint_audit=audit,standing_fraction=report['standing_fraction'],mean_stance_slip_mm_s=report['mean_stance_slip_mm_s'],
                         observations_sha256=sha(folder/'training-observations.npz'),exploration_sha256=sha(folder/'exploration-log.npz'))
        write(out/'result.json',result);print('BIRTH_COMPLETE',mode,child_hashes[mode],flush=True)
    assert exact(out/f'train-legacy-{train_seed}/exploration-log.npz',out/f'train-corrected-{train_seed}/exploration-log.npz'),'Exploration streams differ'
    result['exploration_stream_gate']=True;write(out/'result.json',result)
    summaries={m:{} for m in modes}
    for seed in seeds:
        for mode in modes:
            folder,report=launch('frozen',mode,seed,children[mode],f'child-{mode}-{seed}')
            _,summary=summarize(folder)
            assert report['checkpoint_sha256']==child_hashes[mode]
            paired=compare(a1/f'{mode}-{seed}',folder)
            summaries[mode][seed]=summary
            result['frozen'].append(dict(mode=mode,seed=seed,summary=summary,paired=paired,
                                        report_sha256=sha(folder/'report.json'),observations_sha256=sha(folder/'observations.npz'),
                                        adhesion_sha256=sha(folder/'adhesion-observations.npz')))
            write(out/'result.json',result)
            print('PROBE_COMPLETE',mode,seed,'standing',summary['standing_fraction'],flush=True)
    for mode in modes:result['stage_gates'][mode]=standing_retention(baselines[mode],summaries[mode],seeds)
    result['status']='completed_S2'
except BaseException as error:
    result.update(status='failed',error_type=type(error).__name__,error=str(error));raise
finally:
    result.update(seconds=time.monotonic()-started,parent_unchanged=sha(parent)==expected,source_unchanged=sources()==hashes,
                  data_unchanged=all(sha(data/f['path'])==f['sha256'] for f in manifest),process_tree_empty_verified=empty)
    if empty and lock.exists() and json.loads(lock.read_text())['token']==token:lock.unlink()
    result['lease_released']=not lock.exists();write(out/'result.json',result)
    print('RESULT',result['status'],'seconds',result['seconds'],flush=True)
