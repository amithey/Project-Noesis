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
from run_resource_envelope import SOURCES,monitored_run
from run_training_resource_parity import admission_window,validate_training,exact_archive,TRACE_FIELDS
from check_logging_parity import check_alignment

PROBES=(10006,10007,10010,10011)
LEGS={'LF','LM','LH','RF','RM','RH'}
LOG_FIELDS={'drive','credit','drive_idx','motor_indices','control_chunk_index','control_ms'}
T1B_CHILD='ab4cf13f3a7908ae36e87826267ba8f9c13a2134562cdae1fd4d191c48db0641'


def lr_bank_audit(path,parent):
    with np.load(path,allow_pickle=False) as bank,np.load(parent,allow_pickle=False) as original:
        if set(bank.files)!=set(original.files):raise ValueError('L2 bank schema changed')
        for key in ('topology','motor_indices','initial_weights','rule','version','normalize_reward_error','bias_correct_reward_error'):
            if not np.array_equal(bank[key],original[key]):raise ValueError(f'L2 invariant changed:{key}')
        expected=original['parameters'].copy()
        if expected.shape!=(3,) or expected[0]!=.2:raise ValueError('L2 parent LR metadata invalid')
        expected[0]=.02
        if not np.array_equal(bank['parameters'],expected):raise ValueError('Only registered LR parameter may differ')
        w,ref=bank['weights'],bank['initial_weights']
        if (w.shape!=original['weights'].shape or not np.isfinite(w).all() or not np.array_equal(np.sign(w),np.sign(ref))
                or np.any(np.abs(w)>1.5*np.abs(ref)+1e-12) or np.any(np.abs(w)<.5*np.abs(ref)-1e-12)):
            raise ValueError('L2 original-reference bounds/sign violation')
    return digest(path)


def verify_lr_control(study,root,core):
    from audit_duration_result import audit
    audited=audit(root,core,study)
    if audited['child25_sha256']!=T1B_CHILD or audited['complete_frozen_probes']!=12:
        raise ValueError('Require complete known T1B control')
    return {p.relative_to(study).as_posix():digest(p) for p in study.rglob('*')
            if p.is_file() and 'brain-cache' not in p.parts and p.suffix in ('.json','.npz')}


def lr_trace_check(candidate,control):
    with np.load(candidate/'training-observations.npz',allow_pickle=False) as a,np.load(control/'training-observations.npz',allow_pickle=False) as b:
        if set(a.files)!=TRACE_FIELDS or set(b.files)!=TRACE_FIELDS:raise ValueError('L2 trace schema mismatch')
        for key in TRACE_FIELDS:
            if (a[key].shape!=b[key].shape or a[key].shape[0]!=5000 or not np.isfinite(a[key]).all()
                    or not np.array_equal(a[key][:10],b[key][:10])):
                raise ValueError(f'L2 warmup/shape mismatch:{key}')
        if not np.array_equal(a['learning_active'],np.arange(5000)>=10):raise ValueError('L2 learning window changed')
    with np.load(candidate/'exploration-log.npz',allow_pickle=False) as a,np.load(control/'exploration-log.npz',allow_pickle=False) as b:
        if set(a.files)!=LOG_FIELDS or set(b.files)!=LOG_FIELDS:raise ValueError('L2 log schema mismatch')
        for key in LOG_FIELDS:
            if not np.isfinite(a[key]).all() or not np.array_equal(a[key],b[key]):raise ValueError(f'L2 exploration changed:{key}')
        if not np.array_equal(a['control_chunk_index'],np.arange(10,5000)):raise ValueError('L2 log alignment changed')
    return sorted(TRACE_FIELDS)


def lr_report_check(report,control):
    for key in ('status','seed','worlds','backend','noesis_source_sha256','import_checkpoint_sha256',
                'reward_objective','eligibility_ms','motor_exploration_mv_ms','original_weight_bound',
                'control_ms','physics_dt','birth_ms','requested_seconds','completed_chunks',
                'simulated_seconds','learning_updates','cycle_weight','external_vision_and_chemical_senses'):
        if key not in report or key not in control or report[key]!=control[key]:raise ValueError(f'L2 training context changed:{key}')
    if report.get('learning_rate')!=.02 or control.get('learning_rate')!=.2:raise ValueError('Require explicit L2 learning rates')


def resource_decision(samples,guarded=False):
    original=assess(samples)
    if not guarded:return original
    decision=admission_window(samples,mean_cpu=True)
    decision['original_T1_rule_passed']=original['passed']
    return decision


def guarded_receipt_metadata(protocol,result,hashes):
    expected=dict(experiment='D4M guarded training replay',core_commit=CORE,source_hashes=hashes,
        parent_sha256=PARENT,expected_checkpoint_sha256=CONTROL,seed=1000,seconds=5,chunks=1000,
        learning_updates=990,learning_rate=.2,cycle_weight=.1,exploration_mv_ms=1.5,
        cpu_admission_statistic='mean',launch_mean_cpu_percent=60,launch_minimum_gib=2,
        launch_samples=10,in_run_stop_gib=1,telemetry_interval_seconds=2,
        aggregate_seconds=420,inner_seconds=300,process_seconds=330)
    if hashes!=SOURCES or any(protocol.get(k)!=v for k,v in expected.items()):
        raise ValueError('Require pinned complete D4M receipt/protocol')
    for key,value in dict(status='completed_guarded_training_replay',checkpoint_sha256=CONTROL,
            shared_training_last_updated=False,T1_gate_changed=False,scientific_gates_changed=False,
            model_promoted=False,walking_proven=False).items():
        if result.get(key)!=value:raise ValueError(f'D4M receipt invalid:{key}')
    for key,names in [('trace_exact_fields',TRACE_FIELDS),('exploration_exact_fields',LOG_FIELDS)]:
        fields=result.get(key,{})
        if set(fields)!=names or any(v is not True for v in fields.values()):raise ValueError('D4M receipt parity incomplete')
    wall,total,prior=result.get('wall_seconds'),result.get('aggregate_active_seconds'),protocol.get('prior_wall_seconds')
    if (any(isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) or v<0 for v in [wall,total,prior])
            or total>420 or not np.isclose(total,wall+prior,rtol=0,atol=1e-6)):
        raise ValueError('D4M receipt budget mismatch')
    jobs=result.get('jobs',[])
    if len(jobs)!=1 or jobs[0].get('exit_code')!=0 or jobs[0].get('final_exit_code')!=0:
        raise ValueError('D4M receipt process incomplete')


def verify_guarded_receipt(study,parent,control_birth,hashes):
    protocol=json.loads((study/'protocol.json').read_text(encoding='utf-8'))
    result=json.loads((study/'result.json').read_text(encoding='utf-8'))
    guarded_receipt_metadata(protocol,result,hashes)
    report=json.loads((study/'train/report.json').read_text(encoding='utf-8'));validate_training(report)
    if audit_bank(study/'train/weights-last.npz',parent)!=CONTROL or report.get('saved_checkpoint_sha256')!=CONTROL:
        raise ValueError('D4M prerequisite bank mismatch')
    exact_archive(study/'train/training-observations.npz',control_birth/'training-observations.npz',trace=True)
    exact_archive(study/'train/exploration-log.npz',control_birth/'exploration-log.npz');check_alignment(study/'train')
    names=['protocol.json','result.json','resource-preflight.json','train/report.json','train/weights-last.npz',
           'train/training-observations.npz','train/exploration-log.npz']
    return {name:digest(study/name) for name in names}


def duration_decision(cases,*,control_arm='control5'):
    if control_arm not in ('control5','control25'):raise ValueError('Unknown control arm')
    if set(cases)!={'parent',control_arm,'child25'} or any(set(rows)!=set(PROBES) for rows in cases.values()):
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
                for arm in ('parent',control_arm)}
        retained=all(d['standing_delta']>=0 and d['slip_delta_mm_s']<=0 for d in deltas.values())
        rows.append(dict(seed=seed,child_primary_pass=primary,standing_slip_retained_vs_both=retained,deltas=deltas))
    medians={arm:{key:float(np.median([s[key] for s in cases[arm].values()]))
                 for key in ('heading_forward_distance_mm','net_forward_displacement_mm')} for arm in cases}
    progress=all(medians['child25'][key]>=max(medians['parent'][key],medians[control_arm][key])+.25 for key in medians['child25'])
    return dict(per_probe=rows,medians=medians,median_heading_and_net_gain_pass=progress,
                exploratory_method_rule_pass=bool(progress and all(r['child_primary_pass'] and r['standing_slip_retained_vs_both'] for r in rows)),
                model_promoted=False,stage_advanced=False,walking_proven=False,
                scope='One training seed, four development probes; descriptive '+('learning-rate' if control_arm=='control25' else 'duration')+'/retention test only')


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
    ap.add_argument('--guarded-mean-cpu',action='store_true')
    ap.add_argument('--d4m-study',type=Path)
    ap.add_argument('--low-rate-control-study',type=Path)
    args=ap.parse_args();root,core,out=args.shared_root.resolve(),args.core_source.resolve(),args.output.resolve()
    own=Path(__file__).resolve().parents[1];pilot=own/'gpu-pilot/src'
    allowed_workers=json.loads(args.allow_reviewed_test_workers.read_text(encoding='utf-8')) if args.allow_reviewed_test_workers else []
    parent=root/'gpu-pilot/outputs/e4-exploration-20261029/train-control-1000/weights-last.npz'
    control_birth=root/'gpu-pilot/outputs/logging-parity-d3-20261030/logged';control=control_birth/'weights-last.npz'
    if digest(parent)!=PARENT or digest(control)!=CONTROL or subprocess.check_output(['git','-C',str(core.parent),'rev-parse','HEAD'],text=True).strip()!=CORE:
        raise ValueError('Parent/control/core mismatch')
    hashes=dict(core=tree_digest(core),pilot=tree_digest(pilot));token=uuid.uuid4().hex
    if args.guarded_mean_cpu!=bool(args.d4m_study):raise ValueError('Guarded opt-in and D4M receipt must be supplied together')
    receipt=args.d4m_study.resolve() if args.d4m_study else None
    receipt_files=verify_guarded_receipt(receipt,parent,control_birth,hashes) if receipt else {}
    lr_study=args.low_rate_control_study.resolve() if args.low_rate_control_study else None
    if lr_study and not receipt:raise ValueError('L2 requires explicit guarded mode and D4M receipt')
    lr_files=verify_lr_control(lr_study,root,core) if lr_study else {}
    expected_control,control_arm,learning_rate=(T1B_CHILD,'control25',.02) if lr_study else (CONTROL,'control5',.2)
    if lr_study:control_birth=lr_study/'train25';control=control_birth/'weights-last.npz'
    protocol=dict(owner='codex',pid=os.getpid(),token=token,host=os.environ.get('COMPUTERNAME'),
        commit=subprocess.check_output(['git','-C',str(own),'rev-parse','HEAD'],text=True).strip(),core_commit=CORE,source_hashes=hashes,
        parent_sha256=PARENT,control5_sha256=CONTROL,reference_trace_sha256=digest(control_birth/'training-observations.npz'),
        training_seed=1000,training_seconds=25,training_chunks=5000,learning_updates=4990,probe_seeds=list(PROBES),
        learning_rate=learning_rate,cycle_weight=.1,exploration_mv_ms=1.5,budget_seconds=3600,training_inner_seconds=900,
        training_process_seconds=930,probe_inner_seconds=300,probe_process_seconds=330,minimum_available_gib=3,
        maximum_host_cpu_percent=30,resource_window_samples=10,
        output=str(out),start_unix=time.time(),deadline_unix=time.time()+3600)
    protocol.update(experiment='T1B guarded meanCPU duration' if receipt else 'T1 original duration',
        guarded_mean_cpu=bool(receipt),D4M_receipt=str(receipt) if receipt else None,D4M_files=receipt_files,
        original_T1_default_changed=False,scientific_gates_changed=False)
    if receipt:
        protocol.update(minimum_available_gib=2,mean_host_cpu_percent_limit=60,in_run_memory_stop_gib=1,
            telemetry_interval_seconds=2,cpu_admission_statistic='mean')
        protocol.pop('maximum_host_cpu_percent')
    if lr_study:
        protocol.pop('control5_sha256')
        protocol.update(experiment='L2 matched25s learning-rate contrast',control25_sha256=T1B_CHILD,
            control_learning_rate=.2,control_training_seconds=25,L2_control_study=str(lr_study),L2_control_files=lr_files)
    lock=root/'gpu-pilot/outputs/codex-training-host.lock'
    with os.fdopen(os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL),'w') as stream:json.dump(protocol,stream)
    started,created=time.monotonic(),False
    result=dict(status='running',jobs=[],model_promoted=False,walking_proven=False)
    cases={arm:{} for arm in ('parent',control_arm,'child25')};reports={arm:{} for arm in cases}
    frozen_child={}
    try:
        protocol['allowed_reviewed_test_workers']=allowed_workers
        protocol['excluded_dashboard_pids']=inventory(allowed_workers)
        samples=sample_quiescence();decision=resource_decision(samples,bool(receipt))
        preflight=out.with_name(out.name+'-resource-preflight.json')
        with preflight.open('x',encoding='utf-8') as stream:
            json.dump(dict(protocol=protocol,samples=samples,decision=decision,simulation_started=False),stream,indent=2)
        if not decision['passed']:raise MemoryError('Sustained resource gate refused; no simulation launched')
        protocol['initial_resource_preflight']=decision
        out.mkdir(parents=True,exist_ok=False);created=True;write(out/'protocol.json',protocol)
        def verify():
            if digest(parent)!=PARENT or digest(control)!=expected_control or tree_digest(core)!=hashes['core'] or tree_digest(pilot)!=hashes['pilot']:
                raise RuntimeError('Parent/control/source changed')
            if receipt and any(digest(receipt/name)!=sha for name,sha in receipt_files.items()):raise RuntimeError('D4M prerequisite changed')
            if lr_study and any(digest(lr_study/name)!=sha for name,sha in lr_files.items()):raise RuntimeError('L2 control receipt changed')
            if frozen_child and digest(frozen_child['path'])!=frozen_child['sha256']:raise RuntimeError('Complete child changed')
        def execute(name,module,options,timeout,expected_status):
            verify();inventory(allowed_workers);samples=sample_quiescence();resources=resource_decision(samples,bool(receipt))
            write(out/f'{name}-resource-preflight.json',dict(samples=samples,decision=resources))
            remaining=3600-(time.monotonic()-started)
            available=psutil.virtual_memory().available/2**30
            if remaining<timeout+5 or available<protocol['minimum_available_gib'] or not resources['passed']:raise RuntimeError('Insufficient sustained resource/time budget for full next job')
            command=[sys.executable,'-m',module,'--backend','cpu','--noesis-source',str(core),'--data',str(root/'data'),*options,'--output',str(out/name)]
            print(f'Starting {name}',flush=True);job_start=time.monotonic()
            if receipt:
                monitored=monitored_run(command,out/f'{name}.log',process_seconds=timeout)
                returncode=monitored['exit_code']
            else:
                with (out/f'{name}.log').open('w',encoding='utf-8') as stream:
                    process=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=timeout,check=False)
                returncode=process.returncode
            elapsed=time.monotonic()-job_start;report=json.loads((out/name/'report.json').read_text(encoding='utf-8'))
            result['jobs'].append(dict(name=name,exit_code=returncode,status=report['status'],process_wall_seconds=elapsed,available_gib_at_start=available))
            write(out/'result.json',result)
            if returncode or elapsed>timeout+5 or report['status']!=expected_status:raise RuntimeError(f'Incomplete/late job:{name}')
            verify();return report
        trained=execute('train25','noesis_gpu.train',['--seconds','25','--wall-seconds','900','--seed','1000','--profile','gripped-cycle',
            '--cycle-weight','.1','--learning-rate',str(learning_rate),'--exploration-mv-ms','1.5','--log-exploration','--checkpoint',str(parent)],930,'completed_training_birth')
        if trained['completed_chunks']!=5000 or trained['learning_updates']!=4990 or trained['simulated_seconds']!=25:raise ValueError('Long duration/update count mismatch')
        child=out/'train25/weights-last.npz';child_sha=(lr_bank_audit if lr_study else audit_bank)(child,parent)
        if lr_study:
            lr_report_check(trained,json.loads((control_birth/'report.json').read_text()))
            sidecar=json.loads((out/'train25/weights-last.model.json').read_text())
            if sidecar.get('learning_rate')!=.02:raise ValueError('L2 sidecar LR mismatch')
            checked=lr_trace_check(out/'train25',control_birth)
            result.update(child25_sha256=child_sha,exact_prelearning_fields=checked,all_exploration_fields_exact=True,status='training_complete_evaluation_pending')
        else:
            result.update(child25_sha256=child_sha,exact_prefix_fields=prefix_check(out/'train25',control_birth),status='training_complete_evaluation_pending')
        frozen_child.update(path=child,sha256=child_sha)
        write(out/'result.json',result)
        from noesis_gpu.evaluate import summarize,MATCHED_SETTINGS
        banks={'parent':parent,control_arm:control,'child25':child}
        for seed in PROBES:
            for arm,bank in banks.items():
                name=f'frozen-{arm}-{seed}'
                report=execute(name,'noesis_gpu.run',['--seconds','5','--wall-seconds','300','--seed',str(seed),'--worlds','1','--checkpoint',str(bank)],330,'completed_transport_pilot')
                _,summary=summarize(out/name)
                if lr_study and arm!='child25':
                    old=lr_study/f'frozen-{"child25" if arm==control_arm else "parent"}-{seed}'
                    with np.load(out/name/'observations.npz',allow_pickle=False) as a,np.load(old/'observations.npz',allow_pickle=False) as b:
                        if set(a.files)!=set(b.files) or any(not np.array_equal(a[k],b[k]) for k in b.files):raise ValueError('Fresh L2 control replay differs from T1B')
                if report['checkpoint_sha256']!=digest(bank) or (arm=='child25' and digest(bank)!=child_sha):raise ValueError('Frozen bank mismatch/mutation')
                cases[arm][seed]=summary;reports[arm][seed]=report;write(out/f'{name}-summary.json',summary)
            for arm in (control_arm,'child25'):
                if any(key not in reports['parent'][seed] or key not in reports[arm][seed]
                       or reports['parent'][seed][key]!=reports[arm][seed][key] for key in (*MATCHED_SETTINGS,'seeds')):
                    raise ValueError('Frozen evaluation context mismatch')
            write(out/'cases.json',cases)
        result.update(status='completed_learning_rate_study' if lr_study else 'completed_duration_study',decision=duration_decision(cases,control_arm=control_arm))
    except BaseException as error:
        result.update(status='stopped',error_type=type(error).__name__,error=str(error));raise
    finally:
        if created:result['wall_seconds']=time.monotonic()-started;write(out/'result.json',result)
        if json.loads(lock.read_text()).get('token')==token:lock.unlink()


if __name__=='__main__':
    with keep_system_awake():main()
