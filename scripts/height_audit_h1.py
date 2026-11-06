"""H1 offline coordinate/contact/height audit; no model construction or dynamics."""
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


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def intervals(mask):
    mask=np.asarray(mask,bool)
    if mask.ndim!=1:raise ValueError('Require one dimensional mask')
    changes=np.diff(np.r_[False,mask,False].astype(int))
    return list(zip(np.flatnonzero(changes==1).tolist(),np.flatnonzero(changes==-1).tolist()))


def quantiles(values):
    a=np.asarray(values,float)
    if a.size==0:return None
    if not np.isfinite(a).all():raise ValueError('Nonfinite sample')
    return {k:float(v) for k,v in zip(('min','p25','median','p75','max'),np.quantile(a,[0,.25,.5,.75,1]))}


def masks(obs):
    z=obs['position'][:,2];up=obs['up_body'][:,2];support=(obs['foot_load']>1e-5).any(axis=-1)
    standing=(up>=.7)&(z>=.3)&(z<=1.5)&support
    high_only=(up>=.7)&(z>1.5)&support
    return standing,high_only


def score_row(obs,index,height=None):
    from noesis_gpu.learning import score_transition
    before={k:v[index-1].copy() for k,v in obs.items()}
    after={k:v[index].copy() for k,v in obs.items()}
    if height is not None:after['position'][2]=height
    return score_transition(before,after,5.,'curriculum-standing-v7')['reward']


def case_audit(obs,metadata):
    if any(len(a)!=1000 or not np.isfinite(a).all() for a in obs.values()):raise ValueError('Incomplete/nonfinite frozen trace')
    standing,high=masks(obs);measured=np.arange(1000)>=10;high&=measured
    z=obs['position'][:,2];up=obs['up_body'][:,2];support=(obs['foot_load']>1e-5).sum(axis=-1)
    vz=np.r_[0.,np.diff(z)/.005]
    rewards=np.array([score_row(obs,i) for i in range(10,1000)])
    ids=np.flatnonzero(high);below=np.array([score_row(obs,i,1.5-1e-6) for i in ids]);above=np.array([score_row(obs,i,1.5+1e-6) for i in ids])
    bouts=[]
    for start,end in intervals(high):
        bouts.append(dict(start_seconds=start*.005,end_seconds=end*.005,samples=end-start,duration_seconds=(end-start)*.005,
            height_mm=quantiles(z[start:end]),up_min=float(up[start:end].min()),
            supporting_claws_min=int(support[start:end].min()),supporting_claws_max=int(support[start:end].max()),
            vertical_speed_mm_s=quantiles(vz[start:end])))
    high_measured=high[10:]
    return dict(**metadata,standing_fraction=float(standing[10:].mean()),high_only_fraction=float(high[10:].mean()),
        height_mm=quantiles(z[10:]),high_only_excess_mm=quantiles(z[high]-1.5),
        high_only_uprightness=quantiles(up[high]),high_only_supporting_claws=quantiles(support[high]),
        high_only_ge3_claws_fraction=float((support[high]>=3).mean()) if len(ids) else None,
        high_only_signed_vertical_speed_mm_s=quantiles(vz[high]),high_only_abs_vertical_speed_mm_s=quantiles(np.abs(vz[high])),
        high_only_reward=quantiles(rewards[high_measured]),height_valid_standing_reward=quantiles(rewards[standing[10:]]),
        counterfactual_score_boundary=dict(epsilon_mm=1e-6,held_other_inputs_fixed=True,physical_simulation=False,
            below_minus_above_reward=quantiles(below-above),below_reward=quantiles(below),above_reward=quantiles(above)),
        high_only_bouts=bouts,high_only_bout_count=len(bouts),
        sampled_ground_contact_does_not_prove_continuous_support_or_balance=True)


def layout(root):
    pilot=root/'work/Noesis-Sensory/gpu-pilot/src';core=root/'work/Noesis-Core/src'
    sys.path[:0]=[str(pilot),str(core)]
    return pilot,core


def run_worker(root,out):
    started=time.monotonic();layout(root);protocol=read(out/'protocol.json')
    from noesis_gpu.learning import standing_mask,score_transition
    sr1=read(root/'outputs/training-seed-sr1-20261106-audit.json')
    cases=[];checks=[]
    for old in sr1['posthoc_standing_failure_breakdown']:
        if time.monotonic()-started>170:raise RuntimeError('Offline180s budget nearly exhausted')
        bank,train,seed=old['bank'],old['training_seed'],old['evaluation_seed']
        if bank=='parent':directory=root/f'outputs/adhesion-sign-a1-20261106/legacy-{seed}'
        elif bank=='child':directory=root/(f'outputs/standing-reacquisition-s2-20261106/child-legacy-{seed}' if train==1004 else f'outputs/training-seed-sr1-20261106/child-{train}-{seed}')
        elif train==1004:directory=root/f"outputs/exploration-transfer-x1-20261106/probe-{'parent' if bank=='parent-own' else 'child'}-legacy-off"
        else:directory=root/(f'outputs/training-seed-sr1-20261106/parent-own-{train}' if bank=='parent-own' else f'outputs/training-seed-sr1-20261106/child-{train}-{train}')
        with np.load(directory/'observations.npz',allow_pickle=False) as z:obs={k:z[k][:,0].copy() for k in z.files}
        standing,_=masks(obs);np.testing.assert_array_equal(standing,standing_mask(obs))
        item=case_audit(obs,dict(bank=bank,training_seed=train,evaluation_seed=seed,input_path=directory.relative_to(root).as_posix()))
        assert item['standing_fraction']==1-old['nonstanding_fraction'] or np.isclose(item['standing_fraction'],1-old['nonstanding_fraction'],atol=1e-15,rtol=0)
        assert item['high_only_fraction']==old['criterion_mask_counts'].get('4',0)/990
        cases.append(item)
    for seed in (1005,1006):
        directory=root/f'outputs/training-seed-sr1-20261106/train-{seed}'
        with np.load(directory/'training-observations.npz',allow_pickle=False) as z:obs={k:z[k].copy() for k in z.files}
        assert len(obs['reward'])==5000
        errors=[]
        for i in range(1,5000):
            if i%500==0 and time.monotonic()-started>170:raise RuntimeError('Offline budget')
            before={k:v[i-1].copy() for k,v in obs.items()};after={k:v[i].copy() for k,v in obs.items()}
            errors.append(abs(score_transition(before,after,5.,'curriculum-standing-v7')['reward']-obs['reward'][i]))
        assert max(errors)==0
        checks.append(dict(training_seed=seed,exact_reward_rows=4999,first_row_missing_initial_observation=True,max_error=max(errors)))
    for name,h in protocol['input_hashes'].items():assert sha(root/name)==h
    for name,h in protocol['source_hashes'].items():assert sha(root/name)==h
    result=dict(status='completed_H1',seconds=time.monotonic()-started,cases=cases,training_reward_replay=checks,
        coordinate_facts=dict(position='thorax body-frame origin xpos in global mm, not whole-body COM',
            floor='FlatGroundWorld infinite plane at z=0',height='global thorax z relative to that plane',
            support='tarsus5 XBODY-to-floor contact found and norm(force)>1e-5; not vertical force/COM stability',
            feet_position='tarsus5 body-frame origins, not contact points or claw-tip clearance',
            original_standing='up>=.7, .3<=thorax_z<=1.5, any claw force norm>1e-5',
            reward_boundary='same height gate in native standing-v7 reward; retained as-is'),
        input_and_source_unchanged=True,thresholds_changed=False,new_simulation=False,new_training=False,
        model_promoted=False,walking_proven=False,
        limits=['5ms samples cannot exclude flight/contact loss between samples.',
                'Recorded42active joint angles omit passive tarsus states; no exact physical pose replay claimed.',
                'No whole-body COM, contact normals/points or complete solver state stored; no certified balance/overextension conclusion.',
                'Boundary probes change only scorer inputs, not physics or the reward used in training.',
                'No rescoring to success, new gate, promotion or claim that height explains the sustained1004failure.'])
    write(out/'result.json',result);print(json.dumps(dict(status=result['status'],seconds=result['seconds'],cases=len(cases))))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,required=True);ap.add_argument('--worker',action='store_true')
    args=ap.parse_args();root=args.root.resolve();out=root/'outputs/height-audit-h1-20261106'
    if args.worker:return run_worker(root,out)
    pilot,core=layout(root)
    import psutil
    from noesis_gpu.process_watchdog import run_windows_job,WatchdogError
    me=psutil.Process();excluded={me.pid,*[p.pid for p in me.parents()]}
    if any(p.pid not in excluded and (p.info['name'] or '').lower() in ('python.exe','pythonw.exe') for p in psutil.process_iter(['name'])):raise RuntimeError('Competing Python')
    if psutil.virtual_memory().available<2*2**30:raise RuntimeError('RAM below2GiB')
    prior=read(root/'outputs/training-seed-sr1-20261106-artifact.json');archive=root/'outputs'/prior['path'];assert sha(archive)==prior['sha256']
    source={};inputs={}
    provenance=read(root/'outputs/thermo-pilot-tp2-20261106-source.json')
    for label,folder in [('pilot',pilot),('core',core)]:
        assert len(list(folder.rglob('*.py')))==len(provenance[label])
        for f in provenance[label]:
            p=folder/f['path'];assert sha(p)==f['sha256'];source[p.relative_to(root).as_posix()]=sha(p)
    for rel in ('work/flygym/src/flygym/simulation.py','work/flygym/src/flygym/compose/world/flat_ground.py','work/flygym/src/flygym/compose/world/base_world.py','work/height_audit_h1.py'):
        source[rel]=sha(root/rel)
    with zipfile.ZipFile(archive) as z:
        manifest=json.loads(z.read('manifest.json'))
        for rel,v in manifest.items():
            if rel.endswith(('/observations.npz','/training-observations.npz','/report.json')) or rel=='outputs/training-seed-sr1-20261106-audit.json':
                f=root/rel;assert sha(f)==v['sha256'];inputs[rel]=v['sha256']
    protocol=dict(experiment='H1 offline coordinate/height/contact audit',source_hashes=source,input_hashes=inputs,
        upstream_commit='ca65a510c2afe6ac61c51df4f274c8d190c2f95f',sr1_archive_sha256=prior['sha256'],
        offline_worker_budget_seconds=180,new_physics_or_neural_steps=0,threshold_changes=0,
        analysis='all22prior cases, high-only bouts/support/height/vertical finite differences, fixed-input scoring boundary and exact actual training-reward replay',
        stop='input/source/hash/nonfinite/shape/original-mask/reward-consistency/timeout failure; no retry/retune',model_promoted=False)
    out.mkdir(exist_ok=False)
    lock=root/'work/Project-Noesis/gpu-pilot/outputs/codex-training-host.lock';token=uuid.uuid4().hex
    with os.fdopen(os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY),'w') as f:json.dump(dict(owner='codex',pid=os.getpid(),token=token,deadline_unix=time.time()+180),f)
    empty=True
    try:
        write(out/'protocol.json',protocol);empty=False
        with (out/'worker.log').open('w',encoding='utf-8') as log:
            receipt=run_windows_job([sys.executable,str(Path(__file__).resolve()),'--root',str(root),'--worker'],timeout=180,stdout=log,cwd=root,env=dict(os.environ,PYTHONUTF8='1'))
        empty=receipt['tree_empty'];write(out/'receipt.json',receipt)
        assert receipt['exit_code']==0 and not receipt['timed_out'] and empty
    except WatchdogError as error:
        empty=error.cleanup_verified;raise
    finally:
        if empty and lock.exists() and read(lock)['token']==token:lock.unlink()
        write(out/'cleanup.json',dict(process_tree_empty_verified=empty,lease_released=not lock.exists()))


if __name__=='__main__':main()
