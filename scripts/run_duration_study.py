"""One25s birth and12 frozen development probes; no automatic promotion."""
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
from check_logging_parity import EXPECTED as CONTROL
from run_e6 import audit_bank
from training_power import keep_system_awake
from training_resources import assess,sample_quiescence

PROBES=(10006,10007,10010,10011)
LEGS={'LF','LM','LH','RF','RM','RH'}


def duration_decision(cases):
    if set(cases)!={'parent','control5','child25'} or any(set(rows)!=set(PROBES) for rows in cases.values()):
        raise ValueError('Require every arm and all four matched probes')
    for rows in cases.values():
        for seed,s in rows.items():
            for key in ('standing_fraction','mean_stance_slip_mm_s','heading_forward_distance_mm','net_forward_displacement_mm'):
                if not np.isfinite(s[key]):raise ValueError('Nonfinite metric')
            if s['seed']!=seed or not 0<=s['standing_fraction']<=1 or s['mean_stance_slip_mm_s']<0:
                raise ValueError('Invalid metric/seed')
            counts=s['leg_step_counts']
            if set(counts)!=LEGS or any(isinstance(n,bool) or not isinstance(n,(int,np.integer)) or n<0 for n in counts.values()):
                raise ValueError('Invalid cycle counts')
    rows=[]
    for seed in PROBES:
        child=cases['child25'][seed]
        primary=(child['standing_fraction']>=.9 and child['heading_forward_distance_mm']>=.25
                 and child['mean_stance_slip_mm_s']<=10
                 and sum(n>=2 for n in child['leg_step_counts'].values())>=4)
        deltas={arm:{'standing_delta':child['standing_fraction']-cases[arm][seed]['standing_fraction'],
                     'slip_delta_mm_s':child['mean_stance_slip_mm_s']-cases[arm][seed]['mean_stance_slip_mm_s'],
                     'heading_delta_mm':child['heading_forward_distance_mm']-cases[arm][seed]['heading_forward_distance_mm'],
                     'net_delta_mm':child['net_forward_displacement_mm']-cases[arm][seed]['net_forward_displacement_mm']}
                for arm in ('parent','control5')}
        retained=all(d['standing_delta']>=0 and d['slip_delta_mm_s']<=0 for d in deltas.values())
        rows.append(dict(seed=seed,child_primary_pass=primary,standing_slip_retained_vs_both=retained,deltas=deltas))
    medians={arm:{key:float(np.median([s[key] for s in cases[arm].values()]))
                 for key in ('heading_forward_distance_mm','net_forward_displacement_mm')} for arm in cases}
    progress=all(medians['child25'][key]>=max(medians['parent'][key],medians['control5'][key])+.25 for key in medians['child25'])
    return dict(per_probe=rows,medians=medians,median_heading_and_net_gain_pass=progress,
                exploratory_method_rule_pass=bool(progress and all(r['child_primary_pass'] and r['standing_slip_retained_vs_both'] for r in rows)),
                model_promoted=False,stage_advanced=False,walking_proven=False,
                scope='One training seed, four development probes; descriptive duration/retention test only')


def prefix_check(long_birth,control_birth):
    with np.load(long_birth/'training-observations.npz',allow_pickle=False) as a,np.load(control_birth/'training-observations.npz',allow_pickle=False) as b:
        if set(a.files)!=set(b.files):raise ValueError('Training trace schema changed')
        for key in b.files:
            if a[key].shape[0]!=5000 or b[key].shape[0]!=1000 or not np.isfinite(a[key]).all() or not np.array_equal(a[key][:1000],b[key]):
                raise ValueError(f'Exact5s prefix failed:{key}')
        fields=list(b.files)
    with np.load(long_birth/'exploration-log.npz',allow_pickle=False) as log:
        if not np.array_equal(log['control_chunk_index'],np.arange(10,5000)):
            raise ValueError('Long log alignment failed')
        for key,index in [('drive','drive_idx'),('credit','motor_indices')]:
            if log[key].shape!=(4990,len(log[index])) or not np.isfinite(log[key]).all():
                raise ValueError('Long exploration log shape/values failed')
    return fields


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--shared-root',type=Path,required=True);ap.add_argument('--core-source',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--allow-reviewed-test-workers',type=Path)
    args=ap.parse_args();root,core,out=args.shared_root.resolve(),args.core_source.resolve(),args.output.resolve()
    own=Path(__file__).resolve().parents[1];pilot=own/'gpu-pilot/src'
    allowed_workers=json.loads(args.allow_reviewed_test_workers.read_text(encoding='utf-8')) if args.allow_reviewed_test_workers else []
    parent=root/'gpu-pilot/outputs/e4-exploration-20261029/train-control-1000/weights-last.npz'
    control_birth=root/'gpu-pilot/outputs/logging-parity-d3-20261030/logged';control=control_birth/'weights-last.npz'
    if digest(parent)!=PARENT or digest(control)!=CONTROL or subprocess.check_output(['git','-C',str(core.parent),'rev-parse','HEAD'],text=True).strip()!=CORE:
        raise ValueError('Parent/control/core mismatch')
    hashes=dict(core=tree_digest(core),pilot=tree_digest(pilot));token=uuid.uuid4().hex
    protocol=dict(owner='codex',pid=os.getpid(),token=token,host=os.environ.get('COMPUTERNAME'),
        commit=subprocess.check_output(['git','-C',str(own),'rev-parse','HEAD'],text=True).strip(),core_commit=CORE,source_hashes=hashes,
        parent_sha256=PARENT,control5_sha256=CONTROL,reference_trace_sha256=digest(control_birth/'training-observations.npz'),
        training_seed=1000,training_seconds=25,training_chunks=5000,learning_updates=4990,probe_seeds=list(PROBES),
        learning_rate=.2,cycle_weight=.1,exploration_mv_ms=1.5,budget_seconds=3600,training_inner_seconds=900,
        training_process_seconds=930,probe_inner_seconds=300,probe_process_seconds=330,minimum_available_gib=3,
        maximum_host_cpu_percent=30,resource_window_samples=10,
        output=str(out),start_unix=time.time(),deadline_unix=time.time()+3600)
    lock=root/'gpu-pilot/outputs/codex-training-host.lock'
    with os.fdopen(os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL),'w') as stream:json.dump(protocol,stream)
    started,created=time.monotonic(),False
    result=dict(status='running',jobs=[],model_promoted=False,walking_proven=False)
    cases={arm:{} for arm in ('parent','control5','child25')};reports={arm:{} for arm in cases}
    try:
        protocol['allowed_reviewed_test_workers']=allowed_workers
        protocol['excluded_dashboard_pids']=inventory(allowed_workers)
        samples=sample_quiescence();decision=assess(samples)
        preflight=out.with_name(out.name+'-resource-preflight.json')
        with preflight.open('x',encoding='utf-8') as stream:
            json.dump(dict(protocol=protocol,samples=samples,decision=decision,simulation_started=False),stream,indent=2)
        if not decision['passed']:raise MemoryError('Sustained resource gate refused; no simulation launched')
        protocol['initial_resource_preflight']=decision
        out.mkdir(parents=True,exist_ok=False);created=True;write(out/'protocol.json',protocol)
        def verify():
            if digest(parent)!=PARENT or digest(control)!=CONTROL or tree_digest(core)!=hashes['core'] or tree_digest(pilot)!=hashes['pilot']:
                raise RuntimeError('Parent/control/source changed')
        def execute(name,module,options,timeout,expected_status):
            verify();inventory(allowed_workers);samples=sample_quiescence();resources=assess(samples)
            write(out/f'{name}-resource-preflight.json',dict(samples=samples,decision=resources))
            remaining=3600-(time.monotonic()-started)
            available=psutil.virtual_memory().available/2**30
            if remaining<timeout+5 or available<3 or not resources['passed']:raise RuntimeError('Insufficient sustained resource/time budget for full next job')
            command=[sys.executable,'-m',module,'--backend','cpu','--noesis-source',str(core),'--data',str(root/'data'),*options,'--output',str(out/name)]
            print(f'Starting {name}',flush=True);job_start=time.monotonic()
            with (out/f'{name}.log').open('w',encoding='utf-8') as stream:
                child=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=timeout,check=False)
            elapsed=time.monotonic()-job_start;report=json.loads((out/name/'report.json').read_text(encoding='utf-8'))
            result['jobs'].append(dict(name=name,exit_code=child.returncode,status=report['status'],process_wall_seconds=elapsed,available_gib_at_start=available))
            write(out/'result.json',result)
            if child.returncode or elapsed>timeout+5 or report['status']!=expected_status:raise RuntimeError(f'Incomplete/late job:{name}')
            verify();return report
        trained=execute('train25','noesis_gpu.train',['--seconds','25','--wall-seconds','900','--seed','1000','--profile','gripped-cycle',
            '--cycle-weight','.1','--learning-rate','.2','--exploration-mv-ms','1.5','--log-exploration','--checkpoint',str(parent)],930,'completed_training_birth')
        if trained['completed_chunks']!=5000 or trained['learning_updates']!=4990 or trained['simulated_seconds']!=25:raise ValueError('Long duration/update count mismatch')
        child=out/'train25/weights-last.npz';child_sha=audit_bank(child,parent)
        result.update(child25_sha256=child_sha,exact_prefix_fields=prefix_check(out/'train25',control_birth),status='training_complete_evaluation_pending')
        write(out/'result.json',result)
        from noesis_gpu.evaluate import summarize,MATCHED_SETTINGS
        banks=dict(parent=parent,control5=control,child25=child)
        for seed in PROBES:
            for arm,bank in banks.items():
                name=f'frozen-{arm}-{seed}'
                report=execute(name,'noesis_gpu.run',['--seconds','5','--wall-seconds','300','--seed',str(seed),'--worlds','1','--checkpoint',str(bank)],330,'completed_transport_pilot')
                _,summary=summarize(out/name)
                if report['checkpoint_sha256']!=digest(bank) or (arm=='child25' and digest(bank)!=child_sha):raise ValueError('Frozen bank mismatch/mutation')
                cases[arm][seed]=summary;reports[arm][seed]=report;write(out/f'{name}-summary.json',summary)
            for arm in ('control5','child25'):
                if any(key not in reports['parent'][seed] or key not in reports[arm][seed]
                       or reports['parent'][seed][key]!=reports[arm][seed][key] for key in (*MATCHED_SETTINGS,'seeds')):
                    raise ValueError('Frozen evaluation context mismatch')
            write(out/'cases.json',cases)
        result.update(status='completed_duration_study',decision=duration_decision(cases))
    except BaseException as error:
        result.update(status='stopped',error_type=type(error).__name__,error=str(error));raise
    finally:
        if created:result['wall_seconds']=time.monotonic()-started;write(out/'result.json',result)
        if json.loads(lock.read_text()).get('token')==token:lock.unlink()


if __name__=='__main__':
    with keep_system_awake():main()
