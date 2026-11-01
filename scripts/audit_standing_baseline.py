"""Recompute B1 from raw frozen observations and archive scoped evidence."""
import argparse
import json
from pathlib import Path
import numpy as np
from compare_exploration import digest,tree_digest,write
from run_standing_baseline import decision,PROBES,ANCHOR,PARENT,SOURCES,compare_traces,audit_bank
from audit_standing_entry import failures
from noesis_gpu.evaluate import summarize,MATCHED_SETTINGS
from review_claude_noise import archive


def audit(root,core,study):
    own=Path(__file__).resolve().parents[1]
    if dict(core=tree_digest(core),pilot=tree_digest(own/'gpu-pilot/src'))!=SOURCES:raise ValueError('Source changed')
    protocol=json.loads((study/'protocol.json').read_text());result=json.loads((study/'result.json').read_text())
    if (protocol['experiment']!='B1 frozen standing ancestor qualification' or result['status']!='completed_frozen_standing_baseline'
            or len(result['jobs'])!=8 or result['wall_seconds']>1200 or protocol['source_hashes']!=SOURCES):
        raise ValueError('Incomplete B1 or changed profile')
    banks=dict(parent=root/'gpu-pilot/outputs/e4-exploration-20261029/train-control-1000/weights-last.npz',
               anchor=root/'gpu-pilot/outputs/standing-001-train-11000/weights-last.npz')
    if digest(banks['parent'])!=PARENT or digest(banks['anchor'])!=ANCHOR:raise ValueError('Bank changed')
    audit_bank(banks['anchor'],banks['parent'])
    cases={a:{} for a in banks};rows=[];resources=[];inputs=set(study.glob('*.json'))|set(study.glob('*.log'))
    reference=root/'gpu-pilot/outputs/duration-t1b-guarded-20261031'
    for job in result['jobs']:
        name=job['name'];t=json.loads((study/f'{name}.telemetry.json').read_text())
        if (job['exit_code']!=0 or t['status']!='process_terminal' or t['final_exit_code']!=0
                or t['process_wall_seconds']>330 or not t['samples'] or min(s['host_available_gib'] for s in t['samples'])<1):
            raise ValueError('Job/telemetry integrity failure')
        pf=json.loads((study/f'{name}-resource-preflight.json').read_text())
        if not pf['decision']['passed'] or len(pf['samples'])!=10:raise ValueError('Admission failed')
        resources.append(dict(name=name,seconds=t['process_wall_seconds'],
            minimum_available_gib=min(s['host_available_gib'] for s in t['samples']),
            peak_rss_gib=max(s['child_rss_bytes'] for s in t['samples'])/2**30,
            max_host_cpu=max(s['host_cpu_percent'] for s in t['samples'])))
    for seed in PROBES:
        reports={}
        for arm,bank in banks.items():
            name=f'frozen-{arm}-{seed}';directory=study/name;r,s=summarize(directory)
            if r['checkpoint_sha256']!=digest(bank) or r['seeds']!=[seed]:raise ValueError('Frozen context mismatch')
            with np.load(directory/'observations.npz',allow_pickle=False) as a:
                if any(a[k].shape[0]!=1000 or not np.isfinite(a[k]).all() for k in a.files):raise ValueError('Truncated/nonfinite raw trace')
                s['standing_failure_detail']=failures({k:a[k] for k in ('position','up_body','foot_load')})
            if s!=json.loads((study/f'{name}-summary.json').read_text()):raise ValueError('Raw summary mismatch')
            reports[arm]=r;cases[arm][seed]=s
            rows.append(dict(arm=arm,seed=seed,**{k:s[k] for k in ('standing_fraction','mean_stance_slip_mm_s',
                'heading_forward_distance_mm','net_forward_displacement_mm','locomotion_regime','leg_step_counts','standing_failure_detail')}))
            inputs.update((directory/'report.json',directory/'observations.npz'))
            if arm=='parent':
                old=reference/f'frozen-parent-{seed}'
                if not all(v['exact_equal'] for v in compare_traces(directory/'observations.npz',old/'observations.npz').values()):
                    raise ValueError('Parent repeat mismatch')
                inputs.update((old/'report.json',old/'observations.npz'))
        if any(reports['parent'][k]!=reports['anchor'][k] for k in (*MATCHED_SETTINGS,'seeds')):raise ValueError('Unmatched arm settings')
    recomputed=decision(cases)
    if recomputed!=result['decision']:raise ValueError('Decision differs')
    for bank in banks.values():inputs.update(bank.parent/n for n in ('weights-last.npz','weights-last.model.json','report.json'))
    return dict(status='B1_all8full_frozen_raw_probes_recomputed',protocol_commit=protocol['commit'],
        source_hashes=SOURCES,banks_sha256=protocol['banks_sha256'],wall_seconds=result['wall_seconds'],
        decision=recomputed,probes=rows,resources=resources,
        parent_raw_replays_all4_exact=True,weights_unchanged=True,learning=False,
        model_promoted=False,walking_proven=False,
        raw_files=[dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(inputs)])


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('shared-root','core-source','study','output','archive-directory'):ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args();root=args.shared_root.resolve();study=args.study.resolve()
    result=audit(root,args.core_source.resolve(),study);write(args.output,result)
    extras=[p.name for p in study.glob('*.json')]+[p.name for p in study.glob('*.log')]
    print(json.dumps(archive(root,result,args.archive_directory.resolve(),prefix='standing-baseline-b1',extra_allowed=extras)))
