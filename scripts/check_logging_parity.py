"""D3: full physical training reproducibility, then read-only logging parity."""
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
from compare_exploration import digest, tree_digest, write
from runtime_diagnostic import CORE, PARENT, inventory
from run_e6 import audit_bank
from training_power import keep_system_awake

EXPECTED = '2074b1e2c12061a03e13d60daf93e07a6517d832bc1f31fe914a797784d70214'


def check_alignment(path):
    with np.load(path/'exploration-log.npz',allow_pickle=False) as log, np.load(path/'training-observations.npz',allow_pickle=False) as trace:
        active=np.flatnonzero(trace['learning_active'])
        if not np.array_equal(active,np.arange(10,1000)) or not np.array_equal(log['control_chunk_index'],active):
            raise ValueError('Exploration/reward row alignment failed')
        for key,index_key in [('drive','drive_idx'),('credit','motor_indices')]:
            values=log[key]
            if values.shape!=(990,len(log[index_key])) or not np.isfinite(values).all():
                raise ValueError(f'Invalid exploration log: {key}')
        if np.max(np.abs(log['drive']))>1.5 or np.max(np.abs(log['credit']))>1:
            raise ValueError('Exploration log exceeds original bounds')
    return dict(steps=990,first_control_chunk=10,last_control_chunk=999)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--shared-root',type=Path,required=True)
    ap.add_argument('--core-source',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    root,core,out=args.shared_root.resolve(),args.core_source.resolve(),args.output.resolve()
    own=Path(__file__).resolve().parents[1]; pilot=own/'gpu-pilot/src'
    parent=root/'gpu-pilot/outputs/e4-exploration-20261029/train-control-1000/weights-last.npz'
    if digest(parent)!=PARENT or subprocess.check_output(['git','-C',str(core.parent),'rev-parse','HEAD'],text=True).strip()!=CORE:
        raise ValueError('Parent/core mismatch')
    hashes=dict(core=tree_digest(core),pilot=tree_digest(pilot))
    token=uuid.uuid4().hex
    protocol=dict(owner='codex',pid=os.getpid(),token=token,host=os.environ.get('COMPUTERNAME'),
        commit=subprocess.check_output(['git','-C',str(own),'rev-parse','HEAD'],text=True).strip(),
        core_commit=CORE,source_hashes=hashes,parent_sha256=PARENT,expected_checkpoint_sha256=EXPECTED,
        seed=1000,seconds=5,learning_rate=.2,exploration_mv_ms=1.5,cycle_weight=.1,
        budget_seconds=720,job_wall_seconds=300,process_wall_seconds=330,minimum_available_gib=2,
        output=str(out),start_unix=time.time(),deadline_unix=time.time()+720)
    lock=root/'gpu-pilot/outputs/codex-training-host.lock'
    with os.fdopen(os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL),'w') as handle: json.dump(protocol,handle)
    started,created=time.monotonic(),False
    result=dict(status='running',jobs=[],model_promoted=False,walking_proven=False)
    try:
        protocol['excluded_dashboard_pids']=inventory()
        if psutil.virtual_memory().available<2*2**30: raise MemoryError('Require>=2GiB available')
        out.mkdir(parents=True,exist_ok=False); created=True
        write(out/'protocol.json',protocol)
        def verify():
            if digest(parent)!=PARENT or tree_digest(core)!=hashes['core'] or tree_digest(pilot)!=hashes['pilot']:
                raise RuntimeError('Parent/source changed')
        for name,logged in [('plain',False),('logged',True)]:
            verify(); inventory()
            available=psutil.virtual_memory().available/2**30
            if available<2 or 720-(time.monotonic()-started)<330:
                raise RuntimeError('Insufficient resource/time budget for another full job')
            command=[sys.executable,'-m','noesis_gpu.train','--backend','cpu','--seconds','5',
                '--wall-seconds','300','--seed','1000','--profile','gripped-cycle','--cycle-weight','.1',
                '--learning-rate','.2','--exploration-mv-ms','1.5','--checkpoint',str(parent),
                '--noesis-source',str(core),'--data',str(root/'data'),'--output',str(out/name)]
            if logged:command.append('--log-exploration')
            print(f'Starting D3 {name}:5s training reproduction',flush=True)
            job_start=time.monotonic()
            with (out/f'{name}.log').open('w',encoding='utf-8') as stream:
                child=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=330,check=False)
            elapsed=time.monotonic()-job_start
            report=json.loads((out/name/'report.json').read_text(encoding='utf-8'))
            receipt=dict(name=name,exit_code=child.returncode,process_wall_seconds=elapsed,available_gib_at_start=available,report=report)
            result['jobs'].append(receipt);write(out/'result.json',result)
            if child.returncode or elapsed>335 or report.get('status')!='completed_training_birth' or report.get('completed_chunks')!=1000 or report.get('learning_updates')!=990:
                raise RuntimeError('Incomplete/late D3 birth')
            sha=audit_bank(out/name/'weights-last.npz',parent)
            if sha!=EXPECTED:raise ValueError('Historical full training checkpoint not reproduced')
            receipt['checkpoint_sha256']=sha
            if logged:receipt['log_alignment']=check_alignment(out/name)
            verify()
        with np.load(out/'plain/training-observations.npz',allow_pickle=False) as a,np.load(out/'logged/training-observations.npz',allow_pickle=False) as b:
            if set(a.files)!=set(b.files):raise ValueError('Trace schema mismatch')
            result['exact_trace_fields']={k:bool(np.isfinite(a[k]).all() and np.isfinite(b[k]).all() and np.array_equal(a[k],b[k])) for k in a.files}
            if not all(result['exact_trace_fields'].values()):raise ValueError('Logging changed physical/learning observations')
        result['status']='completed_full_training_reproduction_and_logging_parity'
    except BaseException as error:
        result.update(status='stopped',error_type=type(error).__name__,error=str(error));raise
    finally:
        if created:
            result['wall_seconds']=time.monotonic()-started;write(out/'result.json',result)
        if json.loads(lock.read_text()).get('token')==token:lock.unlink()


if __name__=='__main__':
    with keep_system_awake():main()
