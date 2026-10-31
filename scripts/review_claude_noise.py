"""Read-only audit of Claude's three credit-noise pairs; no training or promotion."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
from compare_exploration import digest,write
from run_duration_study import PROBES,TRACE_FIELDS,LOG_FIELDS,PARENT,SOURCES,audit_bank
from noesis_gpu.evaluate import summarize,MATCHED_SETTINGS


def classify(real,control,strict=False):
    dh=[real[s]['heading_forward_distance_mm']-control[s]['heading_forward_distance_mm'] for s in PROBES]
    dn=[real[s]['net_forward_displacement_mm']-control[s]['net_forward_displacement_mm'] for s in PROBES]
    st=[real[s]['standing_fraction']>=control[s]['standing_fraction'] for s in PROBES]
    sl=[real[s]['mean_stance_slip_mm_s']<=control[s]['mean_stance_slip_mm_s'] for s in PROBES]
    rst=[control[s]['standing_fraction']>=real[s]['standing_fraction'] for s in PROBES]
    rsl=[control[s]['mean_stance_slip_mm_s']<=real[s]['mean_stance_slip_mm_s'] for s in PROBES]
    h,n=float(np.median(dh)),float(np.median(dn))
    better=h>=.25 and n>=.25 and (sum(a and b for a,b in zip(st,sl))>=3 if strict else sum(st)>=3 and sum(sl)>=3)
    worse=h<=-.25 and n<=-.25 and (sum(a and b for a,b in zip(rst,rsl))>=3 if strict else sum(st)<=1 and sum(sl)<=1)
    return dict(verdict='REAL_BETTER' if better else 'CONTROL_BETTER' if worse else 'INDISTINGUISHABLE',
        median_dheading=h,median_dnet=n,standing_ok=sum(st),slip_ok=sum(sl),
        same_probes_retained=sum(a and b for a,b in zip(st,sl)),
        reverse_standing_ok=sum(rst),reverse_slip_ok=sum(rsl)),dh,dn


def same(a,b):
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,list):return isinstance(b,list) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    if isinstance(a,(int,float)) and not isinstance(a,bool):return np.isclose(a,b,rtol=0,atol=1e-9)
    return a==b


def audit(root,official):
    out=root/'gpu-pilot/outputs';rep=out/'credit-noise-replication-20261031'
    ctrl=out/'credit-noise-control-20261031';t1b=out/'duration-t1b-guarded-20261031'
    parent=out/'e4-exploration-20261029/train-control-1000/weights-last.npz'
    if digest(parent)!=PARENT:raise ValueError('Parent changed')
    pairs,pooled_h,pooled_n={},[],[]
    raw_files={parent};receipt_notes=[];weight_rows={};bank_hashes={}
    for folder in (ctrl,rep):
        for name in ('receipts.json','result.json'):
            p=folder/name
            if p.exists():raw_files.add(p)
        receipt=json.loads((folder/'receipts.json').read_text(encoding='utf-8-sig'))
        receipt_notes.append(dict(directory=folder.name,jobs=len(receipt),
            jobs_without_exit_code=sum(j.get('exit_code') is None for j in receipt)))
    for seed in (1000,1001,1002):
        births={'real':t1b/'train25' if seed==1000 else rep/f'train25-real-{seed}',
                'control':ctrl/'train25-control' if seed==1000 else rep/f'train25-control-{seed}'}
        rows={a:{} for a in births};reports={a:{} for a in births};traces={};logs={};banks={}
        sidecar_zero=[]
        for arm,directory in births.items():
            report=json.loads((directory/'report.json').read_text())
            expected=dict(status='completed_training_birth',seed=seed,learning_rate=.2,
                noesis_source_sha256=SOURCES['core'],import_checkpoint_sha256=PARENT,
                completed_chunks=5000,learning_updates=4990,simulated_seconds=25,
                reward_objective='curriculum-gripped-cycle-v9',cycle_weight=.1,
                motor_exploration_mv_ms=1.5,control_ms=5.,physics_dt=.0001,birth_ms=50,eligibility_ms=100)
            if any(report.get(k)!=v for k,v in expected.items()):raise ValueError(f'Training context:{seed}/{arm}')
            bank=directory/'weights-last.npz';bank_sha=audit_bank(bank,parent)
            if report['saved_checkpoint_sha256']!=bank_sha:raise ValueError('Training bank hash mismatch')
            with np.load(bank,allow_pickle=False) as b,np.load(parent,allow_pickle=False) as p:
                if any(not np.array_equal(b[k],p[k]) for k in ('version','normalize_reward_error','bias_correct_reward_error')):
                    raise ValueError('Learning normalization/version changed')
                banks[arm]=b['weights'].copy()
            bank_hashes[f'{seed}-{arm}']=bank_sha
            with np.load(directory/'training-observations.npz',allow_pickle=False) as a:
                if set(a.files)!=TRACE_FIELDS or any(a[k].shape[0]!=5000 or not np.isfinite(a[k]).all() for k in a.files):
                    raise ValueError('Incomplete training arrays')
                traces[arm]={k:a[k][:10].copy() for k in a.files}
                if not np.array_equal(a['learning_active'],np.arange(5000)>=10):raise ValueError('Training activity alignment')
            with np.load(directory/'exploration-log.npz',allow_pickle=False) as a:
                names=LOG_FIELDS|({'applied_credit'} if arm=='control' else set())
                if set(a.files)!=names or any(not np.isfinite(a[k]).all() for k in names):raise ValueError('Credit log schema/values')
                if not np.array_equal(a['control_chunk_index'],np.arange(10,5000)) or a['control_ms']!=5:
                    raise ValueError('Log timing/alignment')
                if a['drive'].shape!=(4990,len(a['drive_idx'])) or a['credit'].shape!=(4990,len(a['motor_indices'])):
                    raise ValueError('Log shapes')
                logs[arm]={k:a[k].copy() for k in names}
            for filename in ('report.json','training-observations.npz','exploration-log.npz','weights-last.npz','weights-last.model.json','profile-run.json'):
                p=directory/filename
                if p.exists():raw_files.add(p)
            profile=directory/'profile-run.json'
            if profile.exists():
                p=json.loads(profile.read_text())
                if p.get('credit_control')!=('independent' if arm=='control' else None):raise ValueError('Control flag mismatch')
                if p.get('cycle_scoring',{}).get('steps')==0:sidecar_zero.append(arm)
        exact={k:np.array_equal(logs['real'][k],logs['control'][k]) for k in LOG_FIELDS}
        warmup={k:np.array_equal(traces['real'][k],traces['control'][k]) for k in TRACE_FIELDS}
        if not all(exact.values()) or not all(warmup.values()):raise ValueError('Body-side perturbation/warmup not matched')
        applied=logs['control']['applied_credit'];realcredit=logs['control']['credit']
        if applied.shape!=realcredit.shape or np.array_equal(applied,realcredit):raise ValueError('Independent credit not applied/logged')
        corr=float(np.corrcoef(applied.ravel(),realcredit.ravel())[0,1])
        for probe in PROBES:
            for arm in births:
                directory=(t1b/f'frozen-child25-{probe}' if arm=='real' else ctrl/f'frozen-control-child25-{probe}') if seed==1000 else rep/f'frozen-{arm}-{seed}-{probe}'
                r,s=summarize(directory)
                if r['seeds']!=[probe] or r['checkpoint_sha256']!=bank_hashes[f'{seed}-{arm}']:
                    raise ValueError('Frozen context/bank mismatch')
                reports[arm][probe]=r;rows[arm][probe]=s
                raw_files.update((directory/'report.json',directory/'observations.npz'))
            if any(reports['real'][probe][k]!=reports['control'][probe][k] for k in MATCHED_SETTINGS):
                raise ValueError('Frozen runtime settings mismatch')
        verdict,dh,dn=classify(rows['real'],rows['control']);strict,_,_=classify(rows['real'],rows['control'],strict=True)
        pooled_h+=dh;pooled_n+=dn
        published=json.loads(official.read_text())['pairs'][str(seed)]
        keys=('standing_fraction','mean_stance_slip_mm_s','heading_forward_distance_mm','net_forward_displacement_mm','locomotion_regime')
        table={a:{str(p):{k:rows[a][p][k] for k in keys} for p in PROBES} for a in rows}
        if not same({k:verdict[k] for k in ('verdict','median_dheading','median_dnet','standing_ok','slip_ok')},
                    {k:published[k] for k in ('verdict','median_dheading','median_dnet','standing_ok','slip_ok')}) or not same(table,published['table']):
            raise ValueError('Published result differs from raw recomputation')
        with np.load(parent,allow_pickle=False) as p:dr=banks['real']-p['weights'];dc=banks['control']-p['weights']
        weight_rows[str(seed)]=dict(norm_real=float(np.linalg.norm(dr)),norm_control=float(np.linalg.norm(dc)),
            cosine=float(dr@dc/np.linalg.norm(dr)/np.linalg.norm(dc)))
        pairs[str(seed)]=dict(implemented=verdict,strict_same_probe_symmetric=strict,
            exploration_exact=exact,prelearning_all19_exact=all(warmup.values()),
            applied_vs_real_credit_correlation=corr,profile_cycle_scoring_zero_steps=sidecar_zero,table=table)
    published=json.loads(official.read_text());h,n=float(np.median(pooled_h)),float(np.median(pooled_n))
    rb=sum(p['implemented']['verdict']=='REAL_BETTER' for p in pairs.values())
    cb=sum(p['implemented']['verdict']=='CONTROL_BETTER' for p in pairs.values())
    verdict='REAL_BETTER' if rb>=2 and h>=.25 and n>=.25 else 'CONTROL_BETTER' if cb>=2 and h<=-.25 and n<=-.25 else 'INDISTINGUISHABLE'
    if verdict!=published['overall_verdict'] or not same(weight_rows,published['weights']):raise ValueError('Pooled/weight summary mismatch')
    return dict(status='all6full_births_and24frozen_probes_recomputed',pairs=pairs,weights=weight_rows,
        overall_verdict=verdict,pooled_median_heading=h,pooled_median_net=n,independent_training_pairs=3,
        pooled_probe_differences=12,probe_differences_are_not_12_independent_training_replicates=True,
        receipt_limits=receipt_notes,bank_sha256=bank_hashes,
        raw_files=[dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(raw_files)],
        model_promoted=False,walking_proven=False,
        scope='Descriptive failure to meet superiority rule; not equivalence or absence-of-credit proof; no retrospective Codex resource/exit-code validation')


def archive(root,result,target):
    target.mkdir(parents=True,exist_ok=False)
    allowed={'report.json','receipts.json','result.json','weights-last.npz','weights-last.model.json',
             'profile-run.json','observations.npz','training-observations.npz','exploration-log.npz'}
    destination=target/'claude-credit-noise-evidence.zip'
    with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for entry in result['raw_files']:
            p=(root/entry['path']).resolve()
            if (not p.is_relative_to(root) or p.name not in allowed or 'brain-cache' in p.parts
                    or p.is_symlink() or digest(p)!=entry['sha256'] or p.stat().st_size!=entry['bytes']):
                raise ValueError('Unapproved or changed evidence path')
            before=p.stat();z.write(p,entry['path']);after=p.stat()
            if before.st_mtime_ns!=after.st_mtime_ns or digest(p)!=entry['sha256']:
                raise ValueError('Evidence changed during archive')
        z.writestr('evidence-manifest.json',json.dumps(result,indent=2))
    with zipfile.ZipFile(destination) as z:
        if z.testzip() is not None:raise ValueError('ZIP CRC failed')
        if set(z.namelist())!={e['path'] for e in result['raw_files']}|{'evidence-manifest.json'}:
            raise ValueError('ZIP file list differs')
        for entry in result['raw_files']:
            data=z.read(entry['path'])
            if len(data)!=entry['bytes'] or hashlib.sha256(data).hexdigest()!=entry['sha256']:
                raise ValueError('ZIP per-file SHA mismatch')
    write(target/'claude-credit-noise-manifest.json',dict(audit=result,archive=dict(
        name=destination.name,bytes=destination.stat().st_size,sha256=digest(destination))))
    return dict(files=len(result['raw_files']),archive_bytes=destination.stat().st_size,archive_sha256=digest(destination))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('shared-root','published-result','output'):ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--archive-directory',type=Path)
    args=ap.parse_args();result=audit(args.shared_root.resolve(),args.published_result.resolve())
    write(args.output,result)
    print(json.dumps({k:result[k] for k in ('status','overall_verdict','pooled_median_heading','pooled_median_net')}))
    if args.archive_directory:print(json.dumps(archive(args.shared_root.resolve(),result,args.archive_directory.resolve())))
