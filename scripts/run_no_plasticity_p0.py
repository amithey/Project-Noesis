"""P0 same-drive active bridge and unchanged-weight physical exposure controls."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import uuid
import numpy as np
import psutil

root=Path(__file__).resolve().parents[1];out=root/'outputs/no-plasticity-p0-20261106'
s2=root/'outputs/standing-reacquisition-s2-20261106'
parent=root/'work/evidence/s1-restored/gpu-pilot/outputs/e4-exploration-20261029/train-control-1000/weights-last.npz'
expected='9fd7d41177bf2c745809401c3b142a832661d1a18d0b117b30cb02aea63fea3c'
pilot=root/'work/Noesis-Sensory/gpu-pilot/src';core=root/'work/Noesis-Core/src'
sys.path[:0]=[str(pilot),str(core)]
from noesis_gpu.process_watchdog import run_windows_job,WatchdogError
from noesis_gpu.retention_audit import compare_checkpoints


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while chunk:=f.read(1024*1024):h.update(chunk)
    return h.hexdigest()


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
def read(path):return json.loads(path.read_text())
def exact(a,b,prefix=None):
    with np.load(a,allow_pickle=False) as x,np.load(b,allow_pickle=False) as y:
        assert set(x.files)==set(y.files)
        for k in x.files:
            a,b=x[k],y[k]
            if prefix is not None:a,b=a[:prefix],b[:prefix]
            np.testing.assert_array_equal(a,b,err_msg=k)


def sources():
    return {str(folder.relative_to(root)): {p.relative_to(folder).as_posix():sha(p) for p in sorted(folder.rglob('*.py'))}
            for folder in (pilot,core)}|{'scripts':{n:sha(root/'work'/n) for n in
                ('p0_control.py','p0_worker.py','run_no_plasticity_p0.py','s2_worker.py','s2_contract.py','adhesion_mapping.py')}}


own={os.getpid()}|{p.pid for p in psutil.Process().parents()}
def resources():
    for p in psutil.process_iter(['name']):
        if p.pid not in own and (p.info['name'] or '').lower() in ('python.exe','pythonw.exe'):
            raise RuntimeError('Competing Python; no allocation')
    ram=psutil.virtual_memory().available/2**30
    if ram<2:raise RuntimeError('RAM below2GiB')
    return ram


assert read(s2/'result.json')['status']=='completed_S2'
old=read(s2/'protocol.json');assert sha(parent)==expected==old['parent_sha256']
for name,value in old['source_hashes']['scripts'].items():assert sha(root/'work'/name)==value
provenance=read(root/'outputs/thermo-pilot-tp2-20261106-source.json')
for label,folder in [('pilot',pilot),('core',core)]:
    assert len(list(folder.rglob('*.py')))==len(provenance[label])
    for f in provenance[label]:assert sha(folder/f['path'])==f['sha256']
manifest=read(root/'outputs/noesis-required-data-sha256.json')['files'];source=sources()
prior={m:{n:sha(s2/f'train-{m}-1004'/n) for n in
       ('report.json','weights-last.npz','training-observations.npz','exploration-log.npz','adhesion-observations.npz')} for m in ('legacy','corrected')}
resources();out.mkdir(exist_ok=False)
lock=root/'work/Project-Noesis/gpu-pilot/outputs/codex-training-host.lock';token=uuid.uuid4().hex
with os.fdopen(os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY),'w') as f:
    json.dump(dict(owner='codex',pid=os.getpid(),token=token,start_unix=time.time(),deadline_unix=time.time()+2100),f)
started=time.monotonic();empty=True
protocol=dict(experiment='P0 no-plasticity same-drive exposure control',date='2026-11-06',parent_sha256=expected,
              source_hashes=source,prior_S2_hashes=prior,seed=1004,seconds=25,chunks=5000,exploration_active_rows=4990,
              reward='curriculum-standing-v7',exploration_mv_ms=1.5,active_learning_rate=.2,
              jobs=['active-legacy','control-legacy','control-corrected'],aggregate_seconds=2100,worker_seconds=600,
              gates=['full active legacy bridge exactly replays S2 arrays/checkpoint before controls',
                     'full network weights read-only and unchanged in controls, zero updates',
                     'complete exploration streams match corresponding S2; initial11body/pad rows exact'],
              stop='technical/hash/horizon/stream/prefix/resource/timeout failure or insufficient full worker budget; no retry/retune/partial interpretation',
              model_promoted=False,walking_proven=False,independent_training_replications=1)
result=dict(status='running',jobs=[],controls={},model_promoted=False,walking_proven=False)
env=dict(os.environ,PYTHONUTF8='1',MPLCONFIGDIR=str(root/'work/matplotlib-cache'))


def unchanged():
    return sources()==source and sha(parent)==expected and all(sha(s2/f'train-{m}-1004'/n)==h for m,files in prior.items() for n,h in files.items())


try:
    write(out/'protocol.json',protocol)
    for arm,mode in [('active','legacy'),('control','legacy'),('control','corrected')]:
        assert unchanged()
        for f in manifest:assert sha(root/'work/NoesisData'/f['path'])==f['sha256']
        ram=resources()
        if time.monotonic()-started>1500:raise RuntimeError('Insufficient remaining600s worker budget')
        name=f'{arm}-{mode}';directory=out/name;print('START',name,flush=True);empty=False
        try:
            with (out/(name+'.log')).open('w',encoding='utf-8') as log:
                receipt=run_windows_job([sys.executable,str(root/'work/p0_worker.py'),'--arm',arm,'--mode',mode,
                    '--root',str(root),'--checkpoint',str(parent),'--output',str(directory)],timeout=600,stdout=log,cwd=root,env=env)
            empty=receipt['tree_empty']
        except WatchdogError as error:
            empty=error.cleanup_verified;raise
        write(out/(name+'-receipt.json'),receipt)
        result['jobs'].append(dict(name=name,arm=arm,mode=mode,receipt=receipt,available_ram_before_gib=ram));write(out/'result.json',result)
        assert empty and not receipt['timed_out'] and receipt['exit_code']==0
        report=read(directory/'report.json');assert report['completed_chunks']==5000 and report['simulated_seconds']==25 and report['seed']==1004
        assert report['adhesion_probe_mode']==mode and report['import_checkpoint_sha256']==expected
        for file in ('training-observations.npz','adhesion-observations.npz'):
            exact(directory/file,s2/f'train-{mode}-1004'/file,None if arm=='active' else 11)
        exact(directory/'exploration-log.npz',s2/f'train-{mode}-1004/exploration-log.npz')
        with np.load(directory/'training-observations.npz',allow_pickle=False) as z:
            assert all(len(z[k])==5000 and np.isfinite(z[k]).all() for k in z.files)
        if arm=='active':
            assert sha(directory/'weights-last.npz')==prior[mode]['weights-last.npz']
            result['active_bridge_exact']=True
        else:
            assert report['status']=='completed_no_plasticity_control' and report['learning_updates']==0
            assert report['control_steps']==4990 and report['full_network_weights_unchanged'] and report['network_weights_readonly']
            assert not report['learning'] and not report['weights_changed'] and not (directory/'weights-last.npz').exists()
            cp=directory/'weights-control.npz';check=compare_checkpoints(parent,cp)
            assert check['compatible'] and check['weights']['changed_count']==0 and check['parameters']['match'] and check['rule']['match']
            side=read(cp.with_suffix('.model.json'))
            for key,value in report.items():assert side[key]==value
            with np.load(directory/'training-observations.npz',allow_pickle=False) as z,np.load(s2/f'train-{mode}-1004/training-observations.npz',allow_pickle=False) as a:
                metrics={}
                for window,sel in [('all_after_birth',slice(10,None)),('last5s',slice(4000,None))]:
                    metrics[window]={k:dict(active=float(a[k][sel].mean()),control=float(z[k][sel].mean()),
                        active_minus_control=float(a[k][sel].mean()-z[k][sel].mean())) for k in ('standing','reward','stance_slip_mm_s')}
            result['controls'][mode]=dict(metrics=metrics,checkpoint_audit=check,report_sha256=sha(directory/'report.json'),
                  observations_sha256=sha(directory/'training-observations.npz'),exploration_sha256=sha(directory/'exploration-log.npz'),
                  checkpoint_sha256=sha(cp),prefix11_exact=True,drive_stream_exact=True)
        write(out/'result.json',result);print('COMPLETE',name,flush=True)
    result['status']='completed_P0'
except BaseException as error:
    result.update(status='failed',error_type=type(error).__name__,error=str(error));raise
finally:
    result.update(seconds=time.monotonic()-started,source_parent_prior_unchanged=unchanged(),
                  data_unchanged=all(sha(root/'work/NoesisData'/f['path'])==f['sha256'] for f in manifest),process_tree_empty_verified=empty)
    if empty and lock.exists() and read(lock)['token']==token:lock.unlink()
    result['lease_released']=not lock.exists();write(out/'result.json',result)
    print('RESULT',result['status'],result['seconds'],flush=True)
