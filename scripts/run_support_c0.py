"""Bounded static/offline C0 calibration; no standing acceptance or learning change."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import uuid
import zipfile
import numpy as np
from support_descriptor_c0 import describe_support,static_fixture,REGIONS

root=Path(__file__).resolve().parents[1];out=root/'outputs/support-calibration-c0-20261106'
sys.path[:0]=[str(root/'work/Noesis-Sensory/gpu-pilot/src'),str(root/'work/Noesis-Core/src')]
def read(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def worker():
    from height_audit_h1 import masks,quantiles
    from noesis_gpu.evaluate import summarize
    started=time.monotonic();p=read(out/'protocol.json');h2=root/'outputs/body-telemetry-h2b-20261106'
    expected={'airborne':'unsupported','foot_only':'tarsal_only','body_only':'central_body_contact',
        'mixed_body_feet':'central_body_contact','proximal_leg':'proximal_leg_contact','other_tarsal':'tarsal_only'}
    fixtures=[]
    for name,label in expected.items():
        f=static_fixture(name);assert f['descriptor']['contact_mode']==label and f['model_time']==0
        fixtures.append(f)
    modes=('unsupported','central_body_contact','proximal_leg_contact','tarsal_only','indeterminate_low_force')
    cases=[];h2_summary=read(root/'outputs/body-telemetry-h2b-20261106-summary.json')
    for case in h2_summary['cases']:
        if time.monotonic()-started>170:raise RuntimeError('Light180s budget')
        name=case['name'];directory=h2/name;model=read(directory/'telemetry-model.json')
        report,legacy=summarize(directory)
        with np.load(directory/'ground-contacts.npz',allow_pickle=False) as z:contacts={k:z[k].copy() for k in z.files}
        with np.load(directory/'body-telemetry.npz',allow_pickle=False) as z:t={k:z[k].copy() for k in z.files}
        with np.load(directory/'observations.npz',allow_pickle=False) as z:obs={k:z[k][:,0].copy() for k in z.files}
        names=[None]*model['nbody']
        for i,n in zip(model['body_ids'],model['body_names']):names[i]=n
        code=[];amounts=[];max_error=0.
        for frame in range(1000):
            force=np.zeros((model['nbody'],3));use=contacts['sample_index']==frame+1
            np.add.at(force,contacts['body_id'][use],contacts['force_on_fly'][use])
            np.testing.assert_allclose(force.sum(0),t['ground_force'][frame+1],rtol=1e-12,atol=1e-10)
            d=describe_support(force,names);code.append(modes.index(d['contact_mode']))
            amounts.append([d['vertical_reaction_by_region'][region] for region in REGIONS])
            max_error=max(max_error,abs(d['total_vertical_reaction']-float(force[:,2].sum())))
            assert not d['physical_standing_validated'] and not d['dynamic_balance_validated']
        code=np.asarray(code,np.int8);amounts=np.asarray(amounts);standing,high=masks(obs);measured=np.arange(1000)>=10
        assert float(standing[10:].mean())==legacy['standing_fraction']==case['standing_fraction']
        groups={}
        for label,select in {'all_measured':measured,'standing':standing&measured,'high_only':high&measured,
                            'upright_failure':(obs['up_body'][:,2]<.7)&measured}.items():
            count=int(select.sum());counts={mode:int(np.count_nonzero(code[select]==i)) for i,mode in enumerate(modes)}
            assert sum(counts.values())==count
            previous=case['groups'][label]
            assert count==previous['samples']
            if count:assert np.isclose(counts['central_body_contact']/count,previous['central_body_ground_contact_fraction'],rtol=0,atol=1e-15)
            groups[label]=dict(samples=count,contact_mode_counts=counts,
                COM_abs_vertical_velocity_mm_s=quantiles(np.abs(t['com_velocity'][np.flatnonzero(select)+1,2])),
                region_vertical_reaction_sum={region:float(amounts[select,i].sum()) for i,region in enumerate(REGIONS)})
        np.savez_compressed(out/(name+'-descriptors.npz'),contact_mode_code=code,region_vertical_reaction=amounts,
            legacy_standing=standing,legacy_high_only=high)
        cases.append(dict(name=name,groups=groups,legacy_standing_fraction=legacy['standing_fraction'],
            legacy_regime=legacy['locomotion_regime'],reaction_partition_max_error=max_error,
            output_sha256=sha(out/(name+'-descriptors.npz')),no_new_acceptance_verdict=True))
    for n,h in p['input_hashes'].items():assert sha(root/n)==h
    for n,h in p['source_hashes'].items():assert sha(root/n)==h
    result=dict(status='completed_C0',seconds=time.monotonic()-started,fixtures=fixtures,cases=cases,
        contact_mode_codebook=list(modes),region_columns=list(REGIONS),force_tolerance=1e-5,
        unchanged_legacy_mask_and_verdicts=True,input_source_unchanged=True,physics_integration_steps=0,neural_steps=0,
        fixture_positives_are_not_fly_or_learning_validation=True,new_standing_gate=False,new_reward=False,
        model_promoted=False,walking_proven=False,
        limits=['Contact mode describes force-bearing anatomy, not equilibrium, gait, continuous support or retained learning.',
                'Flat horizontal ground/world vertical force convention only; loaded unknown anatomy or negative vertical force rejected.',
                'Other tarsal segments are separately identified; multiple loaded body segments are not counts of independent legs.',
                'Central compiled thorax includes fused head; no anatomically separate head-force claim.',
                'Three H2Bdiagnostic cases are not representative validation or policy selection.'])
    write(out/'result.json',result);print(json.dumps(dict(status=result['status'],seconds=result['seconds'],fixtures=len(fixtures),cases=len(cases))))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--worker',action='store_true');args=ap.parse_args()
    if args.worker:return worker()
    import psutil
    from noesis_gpu.process_watchdog import run_windows_job,WatchdogError
    me=psutil.Process();excluded={me.pid,*[p.pid for p in me.parents()]}
    if any(p.pid not in excluded and (p.info['name'] or '').lower() in ('python.exe','pythonw.exe') for p in psutil.process_iter(['name'])):raise RuntimeError('Competing Python')
    if psutil.virtual_memory().available<2*2**30:raise RuntimeError('RAM below2GiB')
    prior=read(root/'outputs/body-telemetry-h2b-20261106-artifact.json');archive=root/'outputs'/prior['path']
    assert sha(archive)==prior['sha256']=='f660b673da8d431873cd829a364d423633b01fe395ce971ecb6f3190984e3257'
    inputs={}
    with zipfile.ZipFile(archive) as z:
        manifest=json.loads(z.read('manifest.json'))
        for name,entry in manifest.items():
            if name.startswith('outputs/body-telemetry-h2b-20261106/') and name.endswith(('/observations.npz','/ground-contacts.npz','/body-telemetry.npz','/telemetry-model.json','/report.json')) or name=='outputs/body-telemetry-h2b-20261106-summary.json':
                assert sha(root/name)==entry['sha256'];inputs[name]=entry['sha256']
    h2=read(root/'outputs/body-telemetry-h2b-20261106/protocol.json');source=dict(h2['source_hashes'])
    h1=read(root/'outputs/height-audit-h1-20261106/protocol.json')
    source['work/height_audit_h1.py']=h1['source_hashes']['work/height_audit_h1.py']
    for n,h in source.items():assert sha(root/n)==h
    for n in ('support_descriptor_c0.py','run_support_c0.py'):source['work/'+n]=sha(root/'work'/n)
    out.mkdir(exist_ok=False)
    protocol=dict(experiment='C0 anatomical support descriptor calibration',input_hashes=inputs,source_hashes=source,
        parent_artifact_sha256=prior['sha256'],worker_seconds=180,static_fixtures=6,existing_cases=3,
        physics_integration_steps=0,neural_steps=0,force_tolerance=1e-5,
        unknown_loaded_anatomy_rejected=True,descriptor_not_standing_validator=True,
        stop='technical/hash/nonfinite/shape/force/fixture-label/partition/legacy-mask mismatch or timeout; no automatic retry/retune',
        new_standing_gate=False,new_reward=False,model_promoted=False)
    lock=root/'work/Project-Noesis/gpu-pilot/outputs/codex-training-host.lock';token=uuid.uuid4().hex;empty=True
    with os.fdopen(os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY),'w') as f:json.dump(dict(owner='codex',pid=os.getpid(),token=token,deadline_unix=time.time()+180),f)
    try:
        write(out/'protocol.json',protocol);empty=False
        with (out/'worker.log').open('w',encoding='utf-8') as log:
            receipt=run_windows_job([sys.executable,str(Path(__file__).resolve()),'--worker'],timeout=180,stdout=log,cwd=root,env=dict(os.environ,PYTHONUTF8='1'))
        empty=receipt['tree_empty'];write(out/'receipt.json',receipt)
        assert receipt['exit_code']==0 and not receipt['timed_out'] and empty
    except WatchdogError as error:
        empty=error.cleanup_verified;raise
    finally:
        if empty and lock.exists() and read(lock)['token']==token:lock.unlink()
        write(out/'cleanup.json',dict(tree_empty=empty,lease_released=not lock.exists()))


if __name__=='__main__':main()
