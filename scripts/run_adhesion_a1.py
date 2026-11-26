"""A1 frozen dynamic comparison of legacy and geometry-consistent pad-sign mappings."""
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import uuid

import numpy as np
import psutil

from adhesion_mapping import screen

root=Path(__file__).resolve().parents[1]
pilot=root/'work/Noesis-Sensory';core=root/'work/Noesis-Core/src'
data=root/'work/NoesisData';out=root/'outputs/adhesion-sign-a1-20261106'
parent=root/'work/evidence/s1-restored/gpu-pilot/outputs/e4-exploration-20261029/train-control-1000/weights-last.npz'
expected='9fd7d41177bf2c745809401c3b142a832661d1a18d0b117b30cb02aea63fea3c'
seeds=[10006,10007,10010,10011]
sys.path[:0]=[str(pilot/'gpu-pilot/src'),str(core)]
from noesis_gpu.evaluate import summarize,MATCHED_SETTINGS
from noesis_gpu.process_watchdog import run_windows_job,WatchdogError


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while chunk:=f.read(1024*1024):h.update(chunk)
    return h.hexdigest()


def tree(folder):
    h=hashlib.sha256()
    for p in sorted(folder.rglob('*.py')):
        h.update(p.relative_to(folder).as_posix().encode());h.update(p.read_bytes())
    return h.hexdigest()


def sources():
    return dict(pilot=tree(pilot/'gpu-pilot/src'),core=tree(core),
                scripts={name:sha(root/'work'/name) for name in ('adhesion_mapping.py','adhesion_ab_worker.py','run_adhesion_a1.py')})


def write(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')


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
for f in manifest:assert sha(data/f['path'])==f['sha256']
hashes=sources()
protocol=dict(experiment='A1 frozen adhesion-sign one-factor comparison',date='2026-11-06',
              parent_sha256=expected,runtime_commit='e0cbd109bb6e018bd253c9f1b5e1698e0421b276',
              core_commit='42c20769376bda5694b4c1c66ca75e52cbd9f1f4',source_hashes=hashes,
              seeds=seeds,modes=['legacy','corrected'],seconds_per_job=5,chunks_per_job=1000,
              factor='pad release at desiredFemur-rest>.08 versus < -.08; threshold magnitude unchanged, no time/leg sequence input',
              process_budget_seconds=300,aggregate_budget_seconds=1200,
              technical_gate='legacy raw arrays exactly ES1parent per seed; full horizon/finite/pad formulas, unchanged weights/source/data/parent, all settings equal except explicit body.adhesion_mapping',
              physical_screen='EACH seed standing>=.90 AND>=legacy/slip<=legacy; median heading AND net gains>=.25mm with positive candidate medians; cycles/tripod medians not lower',
              stop='technical/resource/context/hash/replay/timeout failure; no retry/threshold/tuning/new seeds; no launch with<300sremaining',
              within_episode_learning=False,model_promoted=False,walking_proven=False)
resources();out.mkdir(exist_ok=False)
lock=root/'work/Project-Noesis/gpu-pilot/outputs/codex-training-host.lock';token=uuid.uuid4().hex
with os.fdopen(os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY),'w') as f:
    json.dump(dict(owner='codex',pid=os.getpid(),token=token,start_unix=time.time(),deadline_unix=time.time()+1200),f)
started=time.monotonic();empty=True
result=dict(status='running',cases=[],legacy_replay={},one_factor_context={},model_promoted=False,walking_proven=False)
summaries={m:{} for m in ('legacy','corrected')}
env=dict(os.environ,PYTHONUTF8='1',MPLCONFIGDIR=str(root/'work/matplotlib-cache'))
try:
    write(out/'protocol.json',protocol)
    for seed in seeds:
        reports={}
        for mode in ('legacy','corrected'):
            if time.monotonic()-started>900:raise RuntimeError('Remaining budget below300s')
            if sources()!=hashes or sha(parent)!=expected:raise RuntimeError('Source/parent drift')
            for f in manifest:
                if sha(data/f['path'])!=f['sha256']:raise RuntimeError('Data changed')
            ram=resources();target=out/f'{mode}-{seed}'
            print('START',seed,mode,flush=True)
            empty=False
            try:
                with (out/f'{mode}-{seed}.log').open('w',encoding='utf-8') as log:
                    receipt=run_windows_job([sys.executable,str(root/'work/adhesion_ab_worker.py'),'--mode',mode,
                               '--seed',str(seed),'--root',str(root),'--output',str(target)],
                               timeout=300,stdout=log,cwd=root,env=env)
                empty=receipt['tree_empty']
            except WatchdogError as error:
                empty=error.cleanup_verified
                raise
            write(out/f'{mode}-{seed}-receipt.json',receipt)
            assert empty and not receipt['timed_out'] and receipt['exit_code']==0
            report,summary=summarize(target)
            assert report['adhesion_probe_mode']==mode and report['checkpoint_sha256']==expected and report['weights_unchanged']
            with np.load(target/'adhesion-observations.npz',allow_pickle=False) as z:
                assert z['desired'].shape==(1000,42) and z['Femur_delta'].shape==z['pads'].shape==(1000,6)
                assert all(np.isfinite(z[k]).all() for k in z.files)
                expected_pads=((z['Femur_delta'] if mode=='legacy' else -z['Femur_delta'])<=.08).astype(float)
                np.testing.assert_array_equal(z['pads'],expected_pads)
                release=(z['pads'][10:]==0).mean(axis=0).tolist()
            if mode=='legacy':
                assert exact(target/'observations.npz',root/f'outputs/episodic-search-es1-20261106/parent-{seed}/observations.npz'),'Legacy replay failed'
                result['legacy_replay'][str(seed)]=True
            reports[mode]=report;summaries[mode][seed]=summary
            row=dict(seed=seed,mode=mode,summary=summary,per_leg_pad_release_fraction=release,process_receipt=receipt,
                     available_ram_before_gib=ram,report_sha256=sha(target/'report.json'),
                     observations_sha256=sha(target/'observations.npz'),adhesion_observations_sha256=sha(target/'adhesion-observations.npz'))
            result['cases'].append(row);write(out/'result.json',result)
            print('COMPLETE',seed,mode,'standing',summary['standing_fraction'],'heading',summary['heading_forward_distance_mm'],flush=True)
        for key in MATCHED_SETTINGS:
            if key=='body':
                a,b=dict(reports['legacy']['body']),dict(reports['corrected']['body'])
                assert a.pop('adhesion_mapping')=='legacy-positive-Femur'
                assert b.pop('adhesion_mapping')=='corrected-negative-Femur'
                assert a==b
            else:assert reports['legacy'][key]==reports['corrected'][key],key
        assert reports['legacy']['seeds']==reports['corrected']['seeds']==[seed]
        result['one_factor_context'][str(seed)]=True;write(out/'result.json',result)
    result.update(status='completed_A1',physical_screen=screen(summaries['legacy'],summaries['corrected'],seeds))
except BaseException as error:
    result.update(status='failed',error_type=type(error).__name__,error=str(error));raise
finally:
    result.update(seconds=time.monotonic()-started,parent_unchanged=sha(parent)==expected,
                  source_unchanged=sources()==hashes,data_unchanged=all(sha(data/f['path'])==f['sha256'] for f in manifest),
                  process_tree_empty_verified=empty)
    if empty and lock.exists() and json.loads(lock.read_text())['token']==token:lock.unlink()
    result['lease_released']=not lock.exists();write(out/'result.json',result)
    print('RESULT',result['status'],'seconds',result['seconds'],flush=True)
