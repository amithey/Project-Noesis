"""C2: collect two unchanged training reproductions; test a fixed C1 association."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
import numpy as np
import psutil
from compare_exploration import digest,tree_digest,write
from runtime_diagnostic import CORE,PARENT,inventory
from check_logging_parity import check_alignment
from run_e6 import audit_bank
from training_power import keep_system_awake

EXPECTED={1001:'daece1dc1a3a799d8a0d33448f6bfdb0f7514a4fa4e6ee7e823b44d09caf5932',
          1002:'d7013c88af9aa5a20e150f2aad6e2d4395aa2c58603ada775016e470393d1dc1'}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--shared-root',type=Path,required=True)
    ap.add_argument('--core-source',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();root,core,out=args.shared_root.resolve(),args.core_source.resolve(),args.output.resolve()
    own=Path(__file__).resolve().parents[1];pilot=own/'gpu-pilot/src'
    parent=root/'gpu-pilot/outputs/e4-exploration-20261029/train-control-1000/weights-last.npz'
    if digest(parent)!=PARENT or subprocess.check_output(['git','-C',str(core.parent),'rev-parse','HEAD'],text=True).strip()!=CORE:
        raise ValueError('Parent/core mismatch')
    hashes=dict(core=tree_digest(core),pilot=tree_digest(pilot));token=uuid.uuid4().hex
    protocol=dict(owner='codex',pid=os.getpid(),token=token,host=os.environ.get('COMPUTERNAME'),
        commit=subprocess.check_output(['git','-C',str(own),'rev-parse','HEAD'],text=True).strip(),
        core_commit=CORE,source_hashes=hashes,parent_sha256=PARENT,expected_checkpoints=EXPECTED,
        training_seeds=[1001,1002],seconds=5,learning_rate=.2,cycle_weight=.1,exploration_mv_ms=1.5,
        primary_fixed_motor_index=117575,primary_fixed_lag_steps=36,C1_direction=1,
        budget_seconds=720,job_wall_seconds=300,process_wall_seconds=330,analysis_wall_seconds=30,
        minimum_available_gib=2,output=str(out),start_unix=time.time(),deadline_unix=time.time()+720)
    lock=root/'gpu-pilot/outputs/codex-training-host.lock'
    with os.fdopen(os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL),'w') as stream:json.dump(protocol,stream)
    started,created=time.monotonic(),False
    result=dict(status='running',jobs=[],model_promoted=False,walking_proven=False)
    try:
        protocol['excluded_dashboard_pids']=inventory()
        if psutil.virtual_memory().available<2*2**30:raise MemoryError('Require>=2GiB available')
        out.mkdir(parents=True,exist_ok=False);created=True;write(out/'protocol.json',protocol)
        def verify():
            if digest(parent)!=PARENT or tree_digest(core)!=hashes['core'] or tree_digest(pilot)!=hashes['pilot']:
                raise RuntimeError('Parent/source changed')
        sys.path.insert(0,str(core))
        from credit_diagnostic import replay_rewards,lag_correlations
        for seed,expected in EXPECTED.items():
            verify();inventory();remaining=720-(time.monotonic()-started)
            if remaining<360 or psutil.virtual_memory().available<2*2**30:raise RuntimeError('Insufficient budget/resources')
            birth=out/f'train-{seed}'
            command=[sys.executable,'-m','noesis_gpu.train','--backend','cpu','--seconds','5','--wall-seconds','300',
                '--seed',str(seed),'--profile','gripped-cycle','--cycle-weight','.1','--learning-rate','.2',
                '--exploration-mv-ms','1.5','--log-exploration','--checkpoint',str(parent),
                '--noesis-source',str(core),'--data',str(root/'data'),'--output',str(birth)]
            print(f'Starting C2 logged reproduction seed{seed}',flush=True);job_start=time.monotonic()
            with (out/f'train-{seed}.log').open('w',encoding='utf-8') as stream:
                child=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=330,check=False)
            elapsed=time.monotonic()-job_start;report=json.loads((birth/'report.json').read_text(encoding='utf-8'))
            receipt=dict(seed=seed,exit_code=child.returncode,process_wall_seconds=elapsed,status=report['status'])
            result['jobs'].append(receipt);write(out/'result.json',result)
            if child.returncode or elapsed>335 or report['status']!='completed_training_birth' or report['learning_updates']!=990 or report['completed_chunks']!=1000:
                raise RuntimeError('Incomplete/late training reproduction')
            receipt['checkpoint_sha256']=audit_bank(birth/'weights-last.npz',parent)
            if receipt['checkpoint_sha256']!=expected:raise ValueError('Historical checkpoint reproduction failed')
            receipt['alignment']=check_alignment(birth)
            previous=root/f'gpu-pilot/outputs/e5-continuation-retry1-20261029/train-v9-{seed}/training-observations.npz'
            with np.load(previous,allow_pickle=False) as a,np.load(birth/'training-observations.npz',allow_pickle=False) as b:
                if set(a.files)!=set(b.files) or not all(np.array_equal(a[k],b[k]) for k in a.files):
                    raise ValueError('Historical physical trace reproduction failed')
            diagnostic=out/f'diagnostic-{seed}.json';remaining=720-(time.monotonic()-started)
            subprocess.run([sys.executable,str(own/'scripts/credit_diagnostic.py'),'--noesis-source',str(core),
                '--birth',str(birth),'--output',str(diagnostic)],stdout=subprocess.DEVNULL,check=True,timeout=min(30,remaining))
            with np.load(birth/'training-observations.npz',allow_pickle=False) as trace,np.load(birth/'exploration-log.npz',allow_pickle=False) as log:
                metrics,_=replay_rewards(trace['reward'][log['control_chunk_index']]);corr,_,_=lag_correlations(log['credit'],metrics['learning_signal'])
                columns=np.flatnonzero(log['motor_indices']==117575)
                if len(columns)!=1:raise ValueError('Fixed C1 motor target absent/duplicated')
                fixed=float(corr[36,columns[0]])
            receipt.update(historical_trace_exact=True,fixed_motor_index=117575,fixed_lag_ms=180,
                fixed_signed_r=fixed,same_direction_as_C1=fixed>0,
                secondary_search=json.loads(diagnostic.read_text(encoding='utf-8'))['association'])
            write(out/'result.json',result);verify()
        result.update(status='completed_fixed_association_replication',
            positive_direction_in_both=all(j['same_direction_as_C1'] for j in result['jobs']),
            interpretation='Descriptive two-seed development association. No calibrated significance/causal claim or promotion; peaks are secondary.')
    except BaseException as error:
        result.update(status='stopped',error_type=type(error).__name__,error=str(error));raise
    finally:
        if created:result['wall_seconds']=time.monotonic()-started;write(out/'result.json',result)
        if json.loads(lock.read_text()).get('token')==token:lock.unlink()


if __name__=='__main__':
    with keep_system_awake():main()
