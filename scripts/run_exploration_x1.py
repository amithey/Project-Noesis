"""Exclusive fixed-weight parent/child x current exposure diagnostic; never trains."""
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
from x1_drive import physical_summary,transfer_interaction

root=Path(__file__).resolve().parents[1];out=root/'outputs/exploration-transfer-x1-20261106'
pilot=root/'work/Noesis-Sensory/gpu-pilot/src';core=root/'work/Noesis-Core/src'
sys.path[:0]=[str(pilot),str(core)]
from noesis_gpu.process_watchdog import run_windows_job,WatchdogError
from noesis_gpu.evaluate import summarize,MATCHED_SETTINGS
from noesis_gpu.retention_audit import compare_checkpoints


def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while b:=f.read(1024*1024):h.update(b)
    return h.hexdigest()
def exact(a,b,body_prefix=False):
    with np.load(a,allow_pickle=False) as x,np.load(b,allow_pickle=False) as y:
        assert set(x.files)==set(y.files)
        for k in x.files:
            v,w=x[k],y[k]
            if body_prefix:w=w[:1000]
            np.testing.assert_array_equal(v,w,err_msg=k)
def sources():
    return {str(f.relative_to(root)):{p.relative_to(f).as_posix():sha(p) for p in sorted(f.rglob('*.py'))} for f in (pilot,core)}|{
        'scripts':{n:sha(root/'work'/n) for n in ('x1_drive.py','x1_worker.py','run_exploration_x1.py','p0_control.py','s2_worker.py','s2_contract.py','adhesion_mapping.py')}}
own={os.getpid()}|{p.pid for p in psutil.Process().parents()}
def resources():
    if any(p.pid not in own and (p.info['name'] or '').lower() in ('python.exe','pythonw.exe') for p in psutil.process_iter(['name'])):
        raise RuntimeError('Competing Python')
    ram=psutil.virtual_memory().available/2**30
    if ram<2:raise RuntimeError('RAM below2GiB')
    return ram


p0=root/'outputs/no-plasticity-p0-20261106';s2=root/'outputs/standing-reacquisition-s2-20261106'
assert read(p0/'result.json')['status']=='completed_P0' and read(s2/'result.json')['status']=='completed_S2'
for n,h in read(p0/'protocol.json')['source_hashes']['scripts'].items():assert sha(root/'work'/n)==h
provenance=read(root/'outputs/thermo-pilot-tp2-20261106-source.json')
for label,folder in [('pilot',pilot),('core',core)]:
    assert len(list(folder.rglob('*.py')))==len(provenance[label])
    for f in provenance[label]:assert sha(folder/f['path'])==f['sha256']
parent=root/'work/evidence/s1-restored/gpu-pilot/outputs/e4-exploration-20261029/train-control-1000/weights-last.npz'
checkpoints={('parent',m):parent for m in ('legacy','corrected')}|{('child',m):s2/f'train-{m}-1004/weights-last.npz' for m in ('legacy','corrected')}
assert sha(parent)==read(p0/'protocol.json')['parent_sha256']
for mode in ('legacy','corrected'):
    assert sha(checkpoints['child',mode])==read(s2/'result.json')['training'][mode]['checkpoint_sha256']
    assert compare_checkpoints(parent,checkpoints['child',mode])['compatible']
receipt=read(root/'outputs/no-plasticity-p0-20261106-artifact.json')
archive=root/'outputs'/receipt['path'];assert sha(archive)==receipt['sha256']
prior={}
with zipfile.ZipFile(archive) as z:
    mf=json.loads(z.read('manifest.json'))
    for mode in ('legacy','corrected'):
        for n in ('report.json','training-observations.npz','exploration-log.npz','adhesion-observations.npz'):
            f=p0/f'control-{mode}'/n;h=mf[f.relative_to(root/'outputs').as_posix()]['sha256'];assert sha(f)==h
            prior[str(f.relative_to(root))]=h
for cp in checkpoints.values():prior[str(cp.relative_to(root))]=sha(cp)
data=read(root/'outputs/noesis-required-data-sha256.json')['files'];source=sources()
resources();out.mkdir(exist_ok=False)
lock=root/'work/Project-Noesis/gpu-pilot/outputs/codex-training-host.lock';token=uuid.uuid4().hex
with os.fdopen(os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY),'w') as f:
    json.dump(dict(owner='codex',pid=os.getpid(),token=token,start_unix=time.time(),deadline_unix=time.time()+1800),f)
started=time.monotonic();empty=True
protocol=dict(experiment='X1 fixed weights x exploration current',date='2026-11-06',source_hashes=source,prior_hashes=prior,
    seed=1004,seconds=5,chunks=1000,drive_rows=990,drive_onset_chunk=10,actual_current_amplitude=dict(on=1.5,off=0),
    aggregate_seconds=1800,worker_seconds=300,maximum_jobs=9,learning_updates=0,
    stop='technical/hash/horizon/parity/log/context/resource/timeout failure; no retry/retune/extra probes/partial interpretation',
    gates=['both parent/on P0body/pad/drive prefixes exact','parent/off legacy native body/pad replay exact'],
    limitations='one training seed reused development seed1004; descriptive interaction, no training-seed replication',
    model_promoted=False,walking_proven=False)
result=dict(status='running',jobs=[],cases=[],gates={},interaction={},model_promoted=False,walking_proven=False)
env=dict(os.environ,PYTHONUTF8='1',MPLCONFIGDIR=str(root/'work/matplotlib-cache'))
schedule=[('native','parent','legacy','off'),('probe','parent','legacy','on'),('probe','parent','corrected','on'),
          ('probe','parent','legacy','off'),('probe','parent','corrected','off'),
          ('probe','child','legacy','on'),('probe','child','legacy','off'),('probe','child','corrected','on'),('probe','child','corrected','off')]
reference_seeds=None;reports={};cells={m:{} for m in ('legacy','corrected')}
def unchanged():return sources()==source and all(sha(root/n)==h for n,h in prior.items())


try:
    write(out/'protocol.json',protocol)
    for index,(kind,bank,mode,drive) in enumerate(schedule):
        assert unchanged()
        for f in data:assert sha(root/'work/NoesisData'/f['path'])==f['sha256']
        ram=resources()
        if time.monotonic()-started>1500:raise RuntimeError('Insufficient remaining300s worker budget')
        if index>=4:assert len(result['gates'])==3 and all(result['gates'].values())
        name=f'{kind}-{bank}-{mode}-{drive}';directory=out/name;cp=checkpoints[bank,mode]
        if kind=='native':
            command=[sys.executable,str(root/'work/s2_worker.py'),'--phase','frozen','--mode',mode,'--seed','1004']
        else:command=[sys.executable,str(root/'work/x1_worker.py'),'--mode',mode,'--drive',drive]
        command+=['--root',str(root),'--checkpoint',str(cp),'--output',str(directory)]
        print('START',name,flush=True);empty=False
        try:
            with (out/(name+'.log')).open('w',encoding='utf-8') as log:
                receipt=run_windows_job(command,timeout=300,stdout=log,cwd=root,env=env)
            empty=receipt['tree_empty']
        except WatchdogError as error:
            empty=error.cleanup_verified;raise
        write(out/(name+'-receipt.json'),receipt)
        result['jobs'].append(dict(name=name,kind=kind,bank=bank,mode=mode,drive=drive,receipt=receipt,available_ram_before_gib=ram));write(out/'result.json',result)
        assert empty and receipt['exit_code']==0 and not receipt['timed_out']
        report=read(directory/'report.json');assert report['status']=='completed_transport_pilot' and report['completed_control_chunks']==1000
        assert report['simulated_seconds_per_world']==5 and report['seeds']==[1004] and report['weights_unchanged'] and not report['learning']
        assert report['checkpoint_sha256']==sha(cp)
        summary=physical_summary(directory)
        if drive=='off':
            _,native_summary=summarize(directory)
            assert all(summary[k]==native_summary[k] for k in summary)
        else:
            try:summarize(directory)
            except ValueError:pass
            else:raise AssertionError('Current exposure silently treated as ordinary frozen probe')
        if kind=='native':reports[name]=report;continue
        assert report['control_steps']==990 and report['learning_updates']==0 and report['full_network_weights_unchanged'] and report['network_weights_readonly']
        with np.load(directory/'latent-and-seeds.npz',allow_pickle=False) as z:
            assert z['latent_drive'].shape[0]==990 and z['chunk_seeds'].shape==(1000,)
            if reference_seeds is None:reference_seeds=z['chunk_seeds'].copy()
            np.testing.assert_array_equal(z['chunk_seeds'],reference_seeds)
            with np.load(p0/f'control-{mode}/exploration-log.npz',allow_pickle=False) as prior_log:
                np.testing.assert_array_equal(z['latent_drive'],prior_log['drive'][:990])
        with np.load(directory/'exploration-log.npz',allow_pickle=False) as z,np.load(p0/f'control-{mode}/exploration-log.npz',allow_pickle=False) as old:
            for k in old.files:
                expected=old[k][:990] if k in ('drive','credit','control_chunk_index') else old[k]
                if k=='drive' and drive=='off':expected=np.zeros_like(expected)
                np.testing.assert_array_equal(z[k],expected,err_msg=k)
        if bank=='parent' and drive=='on':
            with np.load(directory/'observations.npz',allow_pickle=False) as x,np.load(p0/f'control-{mode}/training-observations.npz',allow_pickle=False) as y:
                for k in x.files:np.testing.assert_array_equal(x[k][:,0],y[k][:1000],err_msg=k)
            exact(directory/'adhesion-observations.npz',p0/f'control-{mode}/adhesion-observations.npz',True)
            result['gates'][f'parent_on_{mode}_P0prefix']=True
        if bank=='parent' and mode=='legacy' and drive=='off':
            for f in ('observations.npz','adhesion-observations.npz'):exact(directory/f,out/f'native-parent-legacy-off/{f}')
            result['gates']['parent_off_legacy_native']=True
        for oldname,oldreport in reports.items():
            if oldreport['body']['adhesion_mapping']==report['body']['adhesion_mapping']:
                assert all(oldreport[k]==report[k] for k in MATCHED_SETTINGS)
        reports[name]=report;cells[mode][bank,drive]=summary
        result['cases'].append(dict(name=name,bank=bank,mode=mode,drive=drive,summary=summary,checkpoint_sha256=sha(cp),
            report_sha256=sha(directory/'report.json'),observations_sha256=sha(directory/'observations.npz'),
            pads_sha256=sha(directory/'adhesion-observations.npz'),latent_sha256=sha(directory/'latent-and-seeds.npz'),
            exploration_sha256=sha(directory/'exploration-log.npz')))
        write(out/'result.json',result);print('COMPLETE',name,'standing',summary['standing_fraction'],flush=True)
    result['interaction']={m:transfer_interaction(cells[m]) for m in cells};result['status']='completed_X1'
except BaseException as error:
    result.update(status='failed',error_type=type(error).__name__,error=str(error));raise
finally:
    result.update(seconds=time.monotonic()-started,source_parent_prior_unchanged=unchanged(),
        data_unchanged=all(sha(root/'work/NoesisData'/f['path'])==f['sha256'] for f in data),process_tree_empty_verified=empty)
    if empty and lock.exists() and read(lock)['token']==token:lock.unlink()
    result['lease_released']=not lock.exists();write(out/'result.json',result)
    print('RESULT',result['status'],result['seconds'],flush=True)
