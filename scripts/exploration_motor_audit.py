"""Read-only anatomical exploration/command association, not causal validation."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np


def project(mapping,indices,values):
    indices=np.asarray(indices);values=np.asarray(values,float)
    if (indices.ndim!=1 or indices.dtype.kind not in 'iu' or len(np.unique(indices))!=len(indices)
            or np.any(indices<0) or np.any(indices>=mapping.shape[1])
            or values.ndim!=2 or values.shape[1]!=len(indices) or not np.isfinite(values).all()):
        raise ValueError('Invalid exploration indices/values')
    result=np.asarray((mapping[:,indices]@values.T).T)
    if not np.isfinite(result).all():raise ValueError('Nonfinite anatomical projection')
    return result


def corresponding_correlations(inputs,outputs,max_lag=20):
    x,y=np.asarray(inputs,float),np.asarray(outputs,float)
    if x.ndim!=2 or x.shape!=y.shape or len(x)<=4*max_lag or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Invalid paired channel arrays')
    n=len(x);a=x[max_lag:n-max_lag];a=a-a.mean(axis=0);anorm=np.sqrt(np.sum(a*a,axis=0))
    correlations=np.full((max_lag+1,x.shape[1]),np.nan)
    for lag in range(max_lag+1):
        b=y[max_lag+lag:n-max_lag+lag];b=b-b.mean(axis=0)
        denom=anorm*np.sqrt(np.sum(b*b,axis=0))
        np.divide(np.sum(a*b,axis=0),denom,out=correlations[lag],where=denom>0)
    return np.clip(correlations,-1,1)


def summarize_channels(correlations,names):
    rows=[]
    for col,name in enumerate(names):
        c=correlations[:,col];valid=np.isfinite(c)
        best=int(np.nanargmax(np.abs(c))) if valid.any() else None
        rows.append(dict(joint=name,fixed_lag_ms=25,fixed_signed_r=float(c[5]) if np.isfinite(c[5]) else None,
            zero_lag_r=float(c[0]) if np.isfinite(c[0]) else None,
            secondary_best_lag_ms=best*5 if best is not None else None,
            secondary_best_signed_r=float(c[best]) if best is not None else None,
            undefined_reason=None if valid.any() else 'No variation in projected input or command channel'))
    return rows


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--shared-root',type=Path,required=True);ap.add_argument('--noesis-source',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();root,source=args.shared_root.resolve(),args.noesis_source.resolve()
    from runtime_diagnostic import CORE,PARENT
    from compare_exploration import digest,tree_digest
    if subprocess.check_output(['git','-C',str(source.parent),'rev-parse','HEAD'],text=True).strip()!=CORE:
        raise ValueError('Pinned core required')
    import psutil
    if psutil.virtual_memory().available<.5*2**30:raise MemoryError('Offline metadata audit needs>=.5GiB available')
    sys.path.insert(0,str(source))
    import pandas as pd
    import noesis.motor
    if Path(noesis.motor.__file__).resolve()!=source/'noesis/motor.py':raise ValueError('Wrong imported muscle mapping')
    from noesis.motor import build_muscle_matrices
    from noesis_gpu.bridge import LEGACY_JOINTS
    from noesis_gpu.run import source_digest
    neurons_path=root/'data/processed/neurons.parquet';neurons=pd.read_parquet(neurons_path)
    pos,neg=build_muscle_matrices(neurons,list(LEGACY_JOINTS));mapping=pos-neg
    births={1000:root/'gpu-pilot/outputs/logging-parity-d3-20261030/logged',
            1001:root/'gpu-pilot/outputs/credit-replication-c2-20261030/train-1001',
            1002:root/'gpu-pilot/outputs/credit-replication-c2-20261030/train-1002'}
    result=dict(status='completed_descriptive_exploration_motor_audit',core_commit=CORE,
        core_source_sha256=source_digest(source),metadata_sha256=digest(neurons_path),
        pilot_source_sha256=tree_digest(Path(__file__).resolve().parents[1]/'gpu-pilot/src'),
        projection_scope='Existing muscle pool mean difference Gpos-Gneg applied to logged input currents/credits; NOT measured firing rates.',
        primary_lag_ms=25,secondary_lag_search_ms=[0,100],cases=[],model_promoted=False,walking_proven=False,
        limits='Closed-loop development association only; no randomized ablation, calibrated significance or physical-walking proof. Training reports lack historical metadata hashes; mapping uses current recorded-hash cache.')
    for seed,birth in births.items():
        report=json.loads((birth/'report.json').read_text(encoding='utf-8'))
        expected=dict(seed=seed,status='completed_training_birth',learning_updates=990,completed_chunks=1000,
            learning_rate=.2,motor_exploration_mv_ms=1.5,cycle_weight=.1,import_checkpoint_sha256=PARENT,
            noesis_source_sha256=source_digest(source),reward_objective='curriculum-gripped-cycle-v9')
        for key,value in expected.items():
            if report.get(key)!=value:raise ValueError(f'Mismatched context:{key}')
        if digest(birth/'weights-last.npz')!=report['saved_checkpoint_sha256']:raise ValueError('Checkpoint hash mismatch')
        with np.load(birth/'training-observations.npz',allow_pickle=False) as trace,np.load(birth/'exploration-log.npz',allow_pickle=False) as log:
            if trace['command'].shape!=(1000,42) or trace['learning_active'].shape!=(1000,):
                raise ValueError('Incomplete command/learning trace')
            active=np.flatnonzero(trace['learning_active'])
            if not np.array_equal(active,np.arange(10,1000)) or not np.array_equal(log['control_chunk_index'],active):
                raise ValueError('Signal/command row alignment failed')
            commands=trace['command'][active];delta=trace['command'][active]-trace['command'][active-1]
            projected=project(mapping,log['drive_idx'],log['drive'])
            innovation=project(mapping,log['motor_indices'],log['credit'])
        if np.max(np.abs(delta))>.0600000001:raise ValueError('Command exceeded existing slew cap')
        row=dict(seed=seed,input_sha256={name:digest(birth/name) for name in ['report.json','exploration-log.npz','training-observations.npz']},
            input_projection_to_command=summarize_channels(corresponding_correlations(projected,commands),LEGACY_JOINTS),
            innovation_projection_to_command_change=summarize_channels(corresponding_correlations(innovation,delta),LEGACY_JOINTS),
            command_slew_cap_fraction=float(np.mean(np.isclose(np.abs(delta),.06,rtol=0,atol=1e-12))))
        result['cases'].append(row)
    result['positive_in_all_three_at_fixed_lag']={key:[name for j,name in enumerate(LEGACY_JOINTS)
        if all(c[key][j]['fixed_signed_r'] is not None and c[key][j]['fixed_signed_r']>0 for c in result['cases'])]
        for key in ['input_projection_to_command','innovation_projection_to_command_change']}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps({'status':result['status'],'positive_in_all_three_counts':{k:len(v) for k,v in result['positive_in_all_three_at_fixed_lag'].items()},
        'cases':[{'seed':r['seed'],'command_slew_cap_fraction':r['command_slew_cap_fraction']} for r in result['cases']]},indent=2))


if __name__=='__main__':main()
