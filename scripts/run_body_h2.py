"""Three exclusive full frozen replays with exact parity before telemetry interpretation."""
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

root=Path(__file__).resolve().parents[1];out=root/'outputs/body-telemetry-h2b-20261106'
pilot=root/'work/Noesis-Sensory/gpu-pilot/src';core=root/'work/Noesis-Core/src'
sys.path[:0]=[str(pilot),str(core)]
from noesis_gpu.process_watchdog import run_windows_job,WatchdogError
from noesis_gpu.evaluate import summarize,compare


def read(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while b:=f.read(1024*1024):h.update(b)
    return h.hexdigest()
def exact(a,b):
    with np.load(a,allow_pickle=False) as x,np.load(b,allow_pickle=False) as y:
        assert set(x.files)==set(y.files)
        for k in x.files:np.testing.assert_array_equal(x[k],y[k],err_msg=k)
def sources():
    return {p.relative_to(root).as_posix():sha(p) for f in (pilot,core) for p in sorted(f.rglob('*.py'))}|{
        'work/'+n:sha(root/'work'/n) for n in ('h2_telemetry.py','h2_worker.py','run_body_h2.py','s2_worker.py','s2_contract.py','adhesion_mapping.py')}|{
        n:sha(root/n) for n in ('work/flygym/src/flygym/simulation.py','work/flygym/src/flygym/compose/world/flat_ground.py')}
me=psutil.Process();own={me.pid,*[p.pid for p in me.parents()]}
def resources():
    if any(p.pid not in own and (p.info['name'] or '').lower() in ('python.exe','pythonw.exe') for p in psutil.process_iter(['name'])):raise RuntimeError('Competing Python')
    ram=psutil.virtual_memory().available/2**30
    if ram<2:raise RuntimeError('RAM below2GiB')
    return ram


parent=root/'work/evidence/s1-restored/gpu-pilot/outputs/e4-exploration-20261029/train-control-1000/weights-last.npz'
cases=[dict(name='parent-10007',seed=10007,checkpoint=parent,baseline=root/'outputs/adhesion-sign-a1-20261106/legacy-10007',artifact='adhesion-sign-a1-20261106'),
    dict(name='child1005-10007',seed=10007,checkpoint=root/'outputs/training-seed-sr1-20261106/train-1005/weights-last.npz',baseline=root/'outputs/training-seed-sr1-20261106/child-1005-10007',artifact='training-seed-sr1-20261106'),
    dict(name='child1004-10006',seed=10006,checkpoint=root/'outputs/standing-reacquisition-s2-20261106/train-legacy-1004/weights-last.npz',baseline=root/'outputs/standing-reacquisition-s2-20261106/child-legacy-10006',artifact='standing-reacquisition-s2-20261106')]
prior={};baseline_summaries={}
for c in cases:
    artifact=read(root/'outputs'/f"{c['artifact']}-artifact.json");archive=root/'outputs'/artifact['path'];assert sha(archive)==artifact['sha256']
    with zipfile.ZipFile(archive) as z:
        manifest=json.loads(z.read('manifest.json'))
        for n in ('report.json','observations.npz','adhesion-observations.npz'):
            p=c['baseline']/n
            key=next(k for k in (p.relative_to(root).as_posix(),p.relative_to(root/'outputs').as_posix()) if k in manifest)
            h=manifest[key]['sha256'];assert sha(p)==h;prior[p.relative_to(root).as_posix()]=h
    report,summary=summarize(c['baseline']);baseline_summaries[c['name']]=summary
    assert sha(c['checkpoint'])==report['checkpoint_sha256'];prior[c['checkpoint'].relative_to(root).as_posix()]=sha(c['checkpoint'])
    c['checkpoint']=c['checkpoint'].relative_to(root).as_posix();c['baseline']=c['baseline'].relative_to(root).as_posix()
provenance=read(root/'outputs/thermo-pilot-tp2-20261106-source.json')
for label,folder in [('pilot',pilot),('core',core)]:
    assert len(list(folder.rglob('*.py')))==len(provenance[label])
    for f in provenance[label]:assert sha(folder/f['path'])==f['sha256']
for n,h in read(root/'outputs/standing-reacquisition-s2-20261106/protocol.json')['source_hashes']['scripts'].items():assert sha(root/'work'/n)==h
static_path=root/'outputs/body-telemetry-h2-20261106/static-body-map.json';static=read(static_path)
assert static['all_missing_names_restored_without_fusion'] and static['all_critical_observation_ids_valid'] and static['physical_integration_steps']==0
prior[static_path.relative_to(root).as_posix()]=sha(static_path)
static_check=root/'outputs/body-telemetry-h2b-static-20261106/native-static-measurement.json'
static_receipt=root/'outputs/body-telemetry-h2b-static-20261106/receipt.json'
proof=read(static_check);test_receipt=read(static_receipt)
assert proof['status']=='passed_native_static_measurement' and proof['physics_steps']==proof['neural_steps']==0 and proof['production_fusestatic']
assert test_receipt['exit_code']==0 and test_receipt['tree_empty'] and not test_receipt['timed_out']
for p in (static_check,static_receipt):prior[p.relative_to(root).as_posix()]=sha(p)
source=sources();data=read(root/'outputs/noesis-required-data-sha256.json')['files'];resources();out.mkdir(exist_ok=False)
lock=root/'work/Project-Noesis/gpu-pilot/outputs/codex-training-host.lock';token=uuid.uuid4().hex
with os.fdopen(os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY),'w') as f:json.dump(dict(owner='codex',pid=os.getpid(),token=token,deadline_unix=time.time()+1200),f)
started=time.monotonic();empty=True
protocol=dict(experiment='H2B corrected compiled-mass telemetry replay',revision='new allocation after zero-step A0 technical failure, not automatic retry',
    previous_failed_code='4bde8da979265e7fcfff04350d86c4c82a26cd08',date='2026-11-06',source_hashes=source,prior_hashes=prior,
    cases=cases,seconds=5,chunks=1000,samples_with_initial=1001,aggregate_seconds=1200,worker_seconds=300,
    learning_updates=0,policy='legacy-positive-Femur',
    stop='technical/hash/horizon/parity/context/mass/force/resource/timeout failure or insufficient full worker budget; no retry/retune/partial interpretation',
    limits='Three selected known failure types, not model selection/representative validation; no gate relaxation',model_promoted=False)
result=dict(status='running',jobs=[],cases=[],model_promoted=False,walking_proven=False)
env=dict(os.environ,PYTHONUTF8='1',MPLCONFIGDIR=str(root/'work/matplotlib-cache'))
def unchanged():return sources()==source and all(sha(root/n)==h for n,h in prior.items())


try:
    write(out/'protocol.json',protocol)
    for c in cases:
        assert unchanged()
        for f in data:assert sha(root/'work/NoesisData'/f['path'])==f['sha256']
        ram=resources()
        if time.monotonic()-started>900:raise RuntimeError('Insufficient300s remaining worker budget')
        name=c['name'];directory=out/name;print('START',name,flush=True);empty=False
        try:
            with (out/(name+'.log')).open('w',encoding='utf-8') as log:
                receipt=run_windows_job([sys.executable,str(root/'work/h2_worker.py'),'--seed',str(c['seed']),'--root',str(root),
                    '--checkpoint',str(root/c['checkpoint']),'--output',str(directory)],timeout=300,stdout=log,cwd=root,env=env)
            empty=receipt['tree_empty']
        except WatchdogError as error:
            empty=error.cleanup_verified;raise
        write(out/(name+'-receipt.json'),receipt)
        result['jobs'].append(dict(name=name,receipt=receipt,available_ram_before_gib=ram));write(out/'result.json',result)
        assert empty and receipt['exit_code']==0 and not receipt['timed_out']
        for n in ('observations.npz','adhesion-observations.npz'):exact(directory/n,root/c['baseline']/n)
        report,summary=summarize(directory);old=read(root/c['baseline']/'report.json')
        assert report['spikes_per_world']==old['spikes_per_world'] and report['checkpoint_sha256']==sha(root/c['checkpoint'])
        compare(root/c['baseline'],directory)
        for k,v in baseline_summaries[name].items():
            if k not in ('report_sha256','observations_sha256'):assert summary[k]==v
        model=read(directory/'telemetry-model.json')
        assert model['nbody']==static['compiled_nbody'] and model['unmapped_anatomical_count']==len(static['missing_fused_names'])
        assert model['critical_thorax_claw_ids_valid'] and model['production_static_fusion_unchanged']
        assert np.isclose(model['total_mass'],model['model_total_mass'],rtol=1e-12,atol=1e-15)
        weight=model['total_mass']*abs(model['gravity'][2])
        with np.load(directory/'body-telemetry.npz',allow_pickle=False) as z:
            assert all(len(z[k])==1001 and np.isfinite(z[k]).all() for k in z.files)
            np.testing.assert_allclose(z['time'],np.arange(1001)*.005,rtol=0,atol=1e-8)
            np.testing.assert_allclose(z['ground_force'],z['root_constraint_force'],rtol=1e-6,atol=1e-6*weight+1e-10)
            assert z['force_closure_max_error'].max()<=1e-6*weight+1e-10
            assert z['upstream_force_max_error'].max()<=1e-10
            mass=np.asarray(model['body_mass']);com=(z['body_xipos']*mass[None,:,None]).sum(axis=1)/mass.sum()
            np.testing.assert_allclose(com,z['com'],rtol=1e-12,atol=1e-12)
        with np.load(directory/'ground-contacts.npz',allow_pickle=False) as z:
            assert all(np.isfinite(z[k]).all() for k in z.files) and z['sample_index'].min()>=0 and z['sample_index'].max()<=1000
        result['cases'].append(dict(name=name,summary=summary,native_body_pad_replay_exact=True,
            metadata_sha256=sha(directory/'telemetry-model.json'),telemetry_sha256=sha(directory/'body-telemetry.npz'),
            contacts_sha256=sha(directory/'ground-contacts.npz'),report_sha256=sha(directory/'report.json'),
            mass_coverage_verified=True,root_and_upstream_force_closure_verified=True))
        write(out/'result.json',result);print('COMPLETE',name,'standing',summary['standing_fraction'],flush=True)
    result['status']='completed_H2B'
except BaseException as error:
    result.update(status='failed',error_type=type(error).__name__,error=str(error));raise
finally:
    result.update(seconds=time.monotonic()-started,source_prior_checkpoints_unchanged=unchanged(),
        data_unchanged=all(sha(root/'work/NoesisData'/f['path'])==f['sha256'] for f in data),process_tree_empty_verified=empty)
    if empty and lock.exists() and read(lock)['token']==token:lock.unlink()
    result['lease_released']=not lock.exists();write(out/'result.json',result)
    print('RESULT',result['status'],result['seconds'],flush=True)
