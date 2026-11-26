"""Supplementary MP1 technical replay check, not a new scientific success gate.

Prepared before negative-control results. It executes only after the fixed study
terminates: a parent fallback has identical weights and should reproduce matched
frozen observations under the declared deterministic seed schedule.
"""
import json
import numpy as np
import run_measured_mp1 as run
from mp1_select import TRAIN_SEEDS,DEV,FRESH


def array_match(left,right):
    if set(left)!=set(right):return dict(exact=False,reason='array_keys')
    differences=[]
    for key in left:
        a,b=left[key],right[key]
        if a.shape!=b.shape or not np.array_equal(a,b):differences.append(key)
    return dict(exact=not differences,different_arrays=differences)


def main():
    result=run.read(run.out/'result.json');p=run.read(run.out/'protocol.json')
    assert result['status']=='completed_MP1' and result['lease_released'] and result['process_tree_empty_verified']
    assert not (run.root/'work/Project-Noesis/gpu-pilot/outputs/codex-training-host.lock').exists()
    decision=run.read(run.out/'selection.json')['negative']
    if not decision['parent_fallback']:
        print('MP1 fallback diagnostic not applicable: negative selector did not return parent')
        return
    assert run.sha(run.out/'negative.npz')==run.sha(run.root/p['parent_checkpoint'])
    pins={n:h for n,h in result['created_hashes'].items()}
    rows=[]
    for seed in (*TRAIN_SEEDS,*DEV,*FRESH):
        parent=run.reference(p,'parent',seed);negative=run.reference(p,'negative',seed)
        run.native.compare(parent,negative)
        row=dict(seed=seed,reference_kind='current_MP1' if seed in (*TRAIN_SEEDS,*FRESH) else 'reused_signed_bank',files={})
        for filename in ('observations.npz','adhesion-observations.npz','anatomy-observations.npz'):
            if not (parent/filename).exists() or not (negative/filename).exists():continue
            for path in (parent/filename,negative/filename):
                n=path.relative_to(run.root).as_posix()
                h=pins.get(n,p['prior_hashes'].get(n));assert h is not None and run.sha(path)==h
            row['files'][filename]=array_match(run.arrays(parent/filename),run.arrays(negative/filename))
        row['spike_count_exact']=run.read(parent/'report.json')['spikes_per_world']==run.read(negative/'report.json')['spikes_per_world']
        row['all_comparable_exact']=all(v['exact'] for v in row['files'].values()) and row['spike_count_exact']
        rows.append(row)
    payload=dict(status='completed_supplementary_MP1_fallback_replay',rows=rows,
        exact_cases=sum(r['all_comparable_exact'] for r in rows),total_cases=len(rows),
        original_scientific_verdict_unchanged=True,model_promoted=False,
        scope='Technical equality check on parent/negative fallback; no learning efficacy, independent distinct-control count or full neural spike reconstruction')
    run.write(run.root/'outputs/measured-policy-mp1-fallback-replay-private-20261110.json',payload)
    print('Supplementary fallback replay check complete; detailed result remains local')


if __name__=='__main__':main()
