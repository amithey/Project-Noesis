"""WS1 preregistered frozen sensitivity experiment, not a training allocation."""
import argparse
import json
import os
import sys
import time
import uuid
import zipfile
import numpy as np
import psutil
import run_physical_standing_ps1 as history
from ws1_weights import checkpoint_variant

base=history.base
native=base.base
root=history.root
read,write,sha,arrays=history.read,history.write,history.sha,history.arrays
PARENT=history.PARENT
SEEDS=(1004,10006,10007,10010,10011)
out=root/'outputs/weight-sensitivity-ws1-20261110'
inputs=root/'outputs/weight-sensitivity-ws1-inputs-20261110'
SOURCE=('ws1_weights.py','test_ws1_weights.py','ws1_import.py','test_ws1_import.py',
        'ws1_worker.py','test_ws1_worker.py','run_weight_sensitivity_ws1.py','test_ws1_response.py','ps1_worker.py','ps1_objective.py',
        'run_physical_standing_ps1.py','audit_ap2.py','audit_ps1_objective.py','audit_weight_sensitivity_ws1.py')


def prospective():
    old=read(base.out/'protocol.json')
    source=dict(old['source_hashes']);prior=dict(old['prior_hashes']);artifacts=dict(old['artifact_hashes'])
    for group in (source,prior,artifacts):
        for n,h in group.items():assert sha(root/n)==h
    base.verify_native()
    archive=root/'outputs/anatomical-feedback-ap2-20261107-evidence.zip'
    assert sha(archive)=='42e20e37d1ee95c550d65d2cb491d69895ae75777b4f279788110d1468d1373d'
    artifacts[archive.relative_to(root).as_posix()]=sha(archive)
    with zipfile.ZipFile(archive) as z:manifest=json.loads(z.read('manifest.json'))
    for seed in SEEDS:
        d=history.reference('parent',seed)
        for path in d.iterdir():
            if not path.is_file():continue
            n=path.relative_to(root).as_posix()
            expected=manifest[n]['sha256'] if n in manifest else prior[n]
            assert sha(path)==expected;prior[n]=expected
    assert sha(PARENT)=='9fd7d41177bf2c745809401c3b142a832661d1a18d0b117b30cb02aea63fea3c'
    parent=arrays(PARENT);inputs.mkdir(exist_ok=True)
    variants={}
    for name,direction in (('minus',-1),('plus',1)):
        expected=checkpoint_variant(parent,direction);cp=inputs/(name+'.npz')
        if not cp.exists():np.savez_compressed(cp,**expected)
        actual=arrays(cp);assert set(actual)==set(expected)
        for k in expected:np.testing.assert_array_equal(actual[k],expected[k])
        variants[name]=sha(cp);prior[cp.relative_to(root).as_posix()]=sha(cp)
        sidecar=cp.with_suffix('.model.json')
        descriptor=dict(experiment='WS1',constructed_not_trained=True,variant=name,
            parent_sha256=sha(PARENT),checkpoint_sha256=sha(cp),offset_original_absolute=.05*direction,
            weights_only=True,training_updates=0,model_promoted=False,walking_proven=False)
        if sidecar.exists():assert read(sidecar)==descriptor
        else:write(sidecar,descriptor)
        prior[sidecar.relative_to(root).as_posix()]=sha(sidecar)
    for n in SOURCE:source['work/'+n]=sha(root/'work'/n)
    return dict(experiment='WS1 bounded existing-neural-weight sensitivity',date='2026-11-10',
        source_hashes=source,prior_hashes=prior,artifact_hashes=artifacts,
        common_parent_sha256=sha(PARENT),constructed_variant_sha256=variants,
        change='Existing selected edge weights plus/minus .05*abs(original); no clipping, other checkpoint fields identical, imported weights-only',
        constructed_not_trained=True,learning=False,seconds_frozen=5,maximum_new_jobs=11,
        aggregate_seconds=3300,worker_seconds=300,seeds=list(SEEDS),development_seeds=list(SEEDS[1:]),
        sequence='Exact own1004 baseline replay, then minus and plus each own1004 plus four development seeds; no performance-based skipping',
        controls='Signed same-seed parent references; one fresh baseline exact raw observations/pads/anatomy/total spikes; actual full-graph import verification',
        primary='Per seed report exact changed desired-angle and actual joint/body trajectory samples plus max absolute deltas; responsiveness only, no ability threshold or winner',
        secondary='Unchanged native strict retention versus same-seed parent, descriptive only; cannot qualify constructed variants as learned skills',
        invariants='Frozen existing neural-to-muscle execution, unchanged body/senses/RNG/controller/topology/signs/original-reference50%bound; final seeds untouched',
        stop='Any input/source/data/import/hash/horizon/exact replay/resource/timeout/containment failure, or insufficient full worker budget; no retry/retuning/partial science',
        model_promoted=False,standing_proven=False,walking_proven=False)


def response(reference,directory):
    result={}
    for filename,keys in [('adhesion-observations.npz',('desired','pads')),
                          ('observations.npz',('angles','position','up_body','omega'))]:
        a,b=arrays(reference/filename),arrays(directory/filename)
        for key in keys:
            x,y=a[key],b[key];assert x.shape==y.shape and np.isfinite(y).all()
            diff=y-x
            changed=np.any(diff!=0,axis=tuple(range(1,diff.ndim)))
            result[key]=dict(changed_samples=int(changed.sum()),max_abs_delta=float(np.max(np.abs(diff))),
                first_changed_sample=int(np.flatnonzero(changed)[0]) if changed.any() else None)
    return result


def execute(p):
    me=psutil.Process();own={me.pid,*[x.pid for x in me.parents()]}
    data=read(root/'outputs/noesis-required-data-sha256.json')['files']
    def resources():
        if any(x.pid not in own and (x.info['name'] or '').lower() in ('python.exe','pythonw.exe') for x in psutil.process_iter(['name'])):raise RuntimeError('Competing Python')
        ram=psutil.virtual_memory().available/2**30
        if ram<2 or psutil.disk_usage(str(root)).free<2*2**30:raise RuntimeError('RAM/disk below2GiB')
        return ram
    def unchanged():return all(sha(root/n)==h for group in ('source_hashes','prior_hashes','artifact_hashes') for n,h in p[group].items())
    def data_ok():return all(sha(root/'work/NoesisData'/f['path'])==f['sha256'] for f in data)
    resources();assert unchanged() and data_ok() and not out.exists()
    lock=root/'work/Project-Noesis/gpu-pilot/outputs/codex-training-host.lock';token=uuid.uuid4().hex
    with os.fdopen(os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY),'w') as f:
        json.dump(dict(owner='codex',pid=os.getpid(),token=token,deadline_unix=time.time()+3300),f)
    started=time.monotonic();empty=True
    result=dict(status='running',jobs=[],constructed_not_trained=True,model_promoted=False,walking_proven=False)
    jobs=[('baseline',1004)]+[(v,s) for v in ('minus','plus') for s in SEEDS]
    env=dict(os.environ,PYTHONUTF8='1',MPLCONFIGDIR=str(root/'work/matplotlib-cache'))
    try:
        out.mkdir(exist_ok=False);write(out/'protocol.json',p)
        for variant,seed in jobs:
            assert unchanged() and data_ok();base.verify_native();ram=resources()
            if time.monotonic()-started+300>3300:raise RuntimeError('Insufficient full worker budget')
            name=f'{variant}-{seed}';cp=PARENT if variant=='baseline' else inputs/(variant+'.npz')
            print('START',name,flush=True);empty=False
            try:
                with (out/(name+'.log')).open('w',encoding='utf-8') as log:
                    receipt=native.run_windows_job([sys.executable,str(root/'work/ws1_worker.py'),
                        '--variant',variant,'--seed',str(seed),'--root',str(root),'--output',str(out/name),
                        '--checkpoint',str(cp)],timeout=300,stdout=log,cwd=root,env=env)
                empty=receipt['tree_empty']
            except native.WatchdogError as error:
                empty=error.cleanup_verified;raise
            write(out/(name+'-receipt.json'),receipt)
            result['jobs'].append(dict(name=name,variant=variant,seed=seed,receipt=receipt,available_ram_before_gib=ram))
            write(out/'result.json',result)
            assert receipt['exit_code']==0 and not receipt['timed_out'] and empty
            report=read(out/name/'report.json');context=read(out/(name+'-ws1-context.json'))
            assert context==report['ws1_context'] and context['worker_completed'] and report['ws1_worker_completed']
            assert context['checkpoint_sha256']==sha(cp) and not report['learning'] and report['weights_unchanged']
            assert report['completed_control_chunks']==1000 and report['requested_simulated_seconds_per_world']==5
            assert report['status']=='completed_transport_pilot' and report['simulated_seconds_per_world']==5
            native.anatomy(out/name,'frozen')
            if variant=='baseline':
                reference=history.reference('parent',seed)
                for filename in ('observations.npz','adhesion-observations.npz','anatomy-observations.npz'):
                    native.exact(out/name/filename,reference/filename)
                original_report=read(reference/'report.json')
                assert report['spikes_per_world']==original_report['spikes_per_world']
            print('COMPLETE',name,flush=True)
        assert len(result['jobs'])==11
        result['sensitivity']={v:{s:response(history.reference('parent',s),out/f'{v}-{s}') for s in SEEDS} for v in ('minus','plus')}
        result['retention']={v:{s:native.compare(history.reference('parent',s),out/f'{v}-{s}') for s in SEEDS} for v in ('minus','plus')}
        parent_summaries={s:native.summarize(history.reference('parent',s))[1] for s in SEEDS[1:]}
        result['strict_development_retention']={v:history.standing_retention(parent_summaries,
            {s:native.summarize(out/f'{v}-{s}')[1] for s in SEEDS[1:]},SEEDS[1:]) for v in ('minus','plus')}
        result['status']='completed_WS1'
    except BaseException as error:
        result.update(status='failed',error_type=type(error).__name__,error=str(error));raise
    finally:
        result.update(seconds=time.monotonic()-started,source_prior_unchanged=unchanged(),data_unchanged=data_ok(),process_tree_empty_verified=empty)
        if empty and lock.exists() and read(lock)['token']==token:lock.unlink()
        result['lease_released']=not lock.exists()
        if out.exists():write(out/'result.json',result)
        print('RESULT',result['status'],result['seconds'],flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');a=ap.parse_args()
    p=prospective()
    if a.prepare:
        write(root/'outputs/weight-sensitivity-ws1-20261110-protocol.json',p)
        write(root/'outputs/weight-sensitivity-ws1-public-protocol-20261110.json',
              {k:v for k,v in p.items() if k not in ('source_hashes','prior_hashes','artifact_hashes')})
        print('WS1 signed preparation complete; no simulation allocated')
    else:execute(p)


if __name__=='__main__':main()
