"""B1 frozen ancestor/descendant standing qualification; never promotes a bank."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
import psutil
from compare_exploration import digest,tree_digest,write
from run_resource_envelope import SOURCES,monitored_run
from run_duration_study import CORE,PARENT,PROBES,inventory,resource_decision,audit_bank
from runtime_diagnostic import compare_traces
from training_resources import sample_quiescence
from training_power import keep_system_awake
from audit_standing_entry import ANCHOR,failures
from noesis_gpu.evaluate import summarize,MATCHED_SETTINGS
import numpy as np


def decision(cases):
    if set(cases)!={'parent','anchor'} or any(set(c)!=set(PROBES) for c in cases.values()):
        raise ValueError('Require both banks and all four complete probes')
    rows=[]
    for seed in PROBES:
        a,p=cases['anchor'][seed],cases['parent'][seed]
        rows.append(dict(seed=seed,anchor_standing_pass=a['single_probe_standing_check'],
            parent_standing_pass=p['single_probe_standing_check'],
            standing_delta=a['standing_fraction']-p['standing_fraction'],
            slip_delta_mm_s=a['mean_stance_slip_mm_s']-p['mean_stance_slip_mm_s'],
            heading_delta_mm=a['heading_forward_distance_mm']-p['heading_forward_distance_mm'],
            net_delta_mm=a['net_forward_displacement_mm']-p['net_forward_displacement_mm']))
    return dict(four_probe_anchor_standing_gate_passed=all(r['anchor_standing_pass'] for r in rows),
        four_probe_parent_standing_gate_passed=all(r['parent_standing_pass'] for r in rows),
        anchor_standing_slip_retained_every_probe=all(r['standing_delta']>=0 and r['slip_delta_mm_s']<=0 for r in rows),
        per_probe=rows,model_promoted=False,stage_advanced=False,walking_proven=False,
        scope='Qualification of existing banks on four development probes, not newly learned ability or final validation')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for arg in ('shared-root','core-source','output'):ap.add_argument('--'+arg,type=Path,required=True)
    args=ap.parse_args();root,core,out=args.shared_root.resolve(),args.core_source.resolve(),args.output.resolve()
    own=Path(__file__).resolve().parents[1]
    banks=dict(parent=root/'gpu-pilot/outputs/e4-exploration-20261029/train-control-1000/weights-last.npz',
               anchor=root/'gpu-pilot/outputs/standing-001-train-11000/weights-last.npz')
    expected=dict(parent=PARENT,anchor=ANCHOR)
    reference=root/'gpu-pilot/outputs/duration-t1b-guarded-20261031'
    pinned={str(bank):expected[arm] for arm,bank in banks.items()}
    for arm,bank in banks.items():
        report=json.loads((bank.parent/'report.json').read_text())
        if (report['status']!='completed_training_birth' or report['simulated_seconds']!=5
                or report['completed_chunks']!=1000 or report['learning_updates']!=990
                or report['saved_checkpoint_sha256']!=expected[arm]):raise ValueError('Incomplete bank ancestry')
        for filename in ('report.json','weights-last.model.json'):
            p=bank.parent/filename;pinned[str(p)]=digest(p)
    for seed in PROBES:
        for name in ('observations.npz','report.json'):
            p=reference/f'frozen-parent-{seed}'/name;pinned[str(p)]=digest(p)
    def verify():
        if any(digest(Path(name))!=sha for name,sha in pinned.items()):raise ValueError('Bank/reference changed')
        if dict(core=tree_digest(core),pilot=tree_digest(own/'gpu-pilot/src'))!=SOURCES:raise ValueError('Source changed')
    verify()
    if subprocess.check_output(['git','-C',str(core.parent),'rev-parse','HEAD'],text=True).strip()!=CORE:raise ValueError('Wrong core commit')
    audit_bank(banks['anchor'],banks['parent'])
    other=[p for p in (root/'gpu-pilot/outputs').glob('*host.lock') if p.name!='codex-training-host.lock']
    if other:raise RuntimeError('Other host lease requires authoritative owner/process check; not deleted')
    token=uuid.uuid4().hex
    protocol=dict(experiment='B1 frozen standing ancestor qualification',owner='codex',pid=os.getpid(),token=token,
        host=os.environ.get('COMPUTERNAME'),commit=subprocess.check_output(['git','-C',str(own),'rev-parse','HEAD'],text=True).strip(),
        core_commit=CORE,source_hashes=SOURCES,banks_sha256=expected,probe_seeds=list(PROBES),seconds=5,
        jobs=8,learning=False,exploration=0,budget_seconds=1200,inner_seconds=300,process_seconds=330,
        minimum_available_gib=2,mean_cpu_limit=60,resource_samples=10,telemetry_seconds=2,memory_floor_gib=1,
        arm_order='parent,anchor on10006/10010; anchor,parent on10007/10011',
        output=str(out),deadline_unix=time.time()+1200,original_defaults_changed=False,scientific_gates_changed=False)
    lock=root/'gpu-pilot/outputs/codex-training-host.lock'
    with os.fdopen(os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY),'w') as f:json.dump(protocol,f)
    started=time.monotonic();created=False;result=dict(status='initializing',jobs=[],model_promoted=False,walking_proven=False)
    cases={a:{} for a in banks};reports={a:{} for a in banks}
    try:
        inventory();out.mkdir(parents=True,exist_ok=False);created=True;write(out/'protocol.json',protocol)
        for seed in PROBES:
            for arm in (('parent','anchor') if seed in (10006,10010) else ('anchor','parent')):
                verify();inventory();name=f'frozen-{arm}-{seed}'
                samples=sample_quiescence();resources=resource_decision(samples,True)
                write(out/f'{name}-resource-preflight.json',dict(samples=samples,decision=resources))
                if not resources['passed'] or psutil.virtual_memory().available/2**30<2:raise RuntimeError('B1 fresh resource gate refused')
                if 1200-(time.monotonic()-started)<335:raise TimeoutError('Insufficient full next-job budget')
                print('Starting '+name,flush=True)
                command=[sys.executable,'-m','noesis_gpu.run','--backend','cpu','--worlds','1','--noesis-source',str(core),
                    '--data',str(root/'data'),'--seconds','5','--wall-seconds','300','--seed',str(seed),
                    '--checkpoint',str(banks[arm]),'--output',str(out/name)]
                job=monitored_run(command,out/f'{name}.log')
                result['jobs'].append(dict(name=name,**{k:v for k,v in job.items() if k!='samples'}))
                write(out/'result.json',result)
                if job['exit_code']!=0:raise RuntimeError('Incomplete frozen job')
                r,s=summarize(out/name)
                if r['checkpoint_sha256']!=expected[arm] or r['seeds']!=[seed]:raise ValueError('Bank/seed mismatch')
                cases[arm][seed]=s;reports[arm][seed]=r
                with np.load(out/name/'observations.npz',allow_pickle=False) as data:
                    s['standing_failure_detail']=failures({k:data[k] for k in ('position','up_body','foot_load')})
                write(out/f'{name}-summary.json',s)
                if arm=='parent':
                    comparison=compare_traces(out/name/'observations.npz',reference/f'frozen-parent-{seed}/observations.npz')
                    if not all(row['exact_equal'] for row in comparison.values()):raise ValueError('Current parent replay mismatch')
                    write(out/f'{name}-replay.json',comparison)
                verify()
            if any(reports['parent'][seed][k]!=reports['anchor'][seed][k] for k in (*MATCHED_SETTINGS,'seeds')):
                raise ValueError('Unmatched frozen contexts')
            write(out/'cases.json',cases)
        result.update(status='completed_frozen_standing_baseline',decision=decision(cases))
    except BaseException as error:
        result.update(status='stopped',error_type=type(error).__name__,error=str(error));raise
    finally:
        if created:result['wall_seconds']=time.monotonic()-started;write(out/'result.json',result)
        if json.loads(lock.read_text())['token']==token:lock.unlink()


if __name__=='__main__':
    with keep_system_awake():main()
