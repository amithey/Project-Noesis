"""Prospective bounded P2: matched sensory-history controls, then pulse ablations."""
import argparse,json,os,sys,time,uuid,zipfile
import numpy as np
import psutil
import run_physical_p1 as base
from p2_sensory_history import FIELDS

root=base.root;out=root/'outputs/sensory-causality-p2-20261107'
read,write,sha,arrays=base.read,base.write,base.sha,base.arrays
MODELS=base.MODELS
REFERENCES={model:base.out/(model+'-baseline') for model in MODELS}


def prospective():
    p=base.prospective();source=dict(p['source_hashes']);prior=dict(p['prior_hashes']);artifacts=dict(p['artifact_hashes'])
    result=read(base.out/'result.json');assert result['status']=='completed_P1' and result['lease_released'] and result['process_tree_empty_verified']
    receipt=read(root/'outputs/physical-calibration-p1-20261107-artifact.json');archive=root/'outputs'/receipt['path']
    assert sha(archive)==receipt['sha256']=='ab1ab628cc1228d527599584e8f0660d59b2fb42d7bce5c1a7900b1d4c68d907'
    artifacts[archive.relative_to(root).as_posix()]=sha(archive)
    with zipfile.ZipFile(archive) as z:
        manifest=json.loads(z.read('manifest.json'))
        for n,entry in manifest.items():
            if n.startswith('outputs/physical-calibration-p1-20261107/'):
                assert sha(root/n)==entry['sha256'];prior[n]=entry['sha256']
    for n in ('p2_sensory_history.py','p2_worker.py','run_sensory_p2.py','test_p2_sensory_history.py'):source['work/'+n]=sha(root/'work'/n)
    return dict(experiment='P2 own-baseline sensory-history causality diagnostic',date='2026-11-07',source_hashes=source,prior_hashes=prior,artifact_hashes=artifacts,
        checkpoint_sha256=p['checkpoint_sha256'],seed=10011,worlds=1,seconds=5,chunks=1000,learning_updates=0,motor_exploration=0,
        sequence=['parent-baseline','candidate-baseline','parent-pulse','candidate-pulse'],maximum_new_jobs=4,aggregate_seconds=1200,worker_seconds=300,
        sensory_intervention=dict(fields=list(FIELDS),timing='chunk0 pristine live input; chunkk>0 own baseline post-step rowk-1',
            motor_replay=False,native_brain_rng_muscles_unchanged=True,live_body_observation_preserved=True),
        pulse=p['pulse'],source_reference='Published P1 matched closed-loop cases; original AP2 birth verdicts unchanged',
        analysis=dict(intervals='All post-pulse states t>=1.965, and separately final1s t>=4.0; endpoints inclusive',
            physical_metrics=['fraction up_body.z<0.7','tarsal-only contact fraction','mean non-tarsal vertical reaction fraction',
                'mean COM speed','mean absolute vertical COM speed','mean body angular speed','COM height range'],
            matched_comparison='Compare live P1 feedback versus P2 own-history input separately for each checkpoint; no pooled winner or binary recovery/ability pass',
            interpretation='Condition differences isolate this sensory-history intervention in one selected deterministic context, not general usefulness or learned balance'),
        controls='Both no-pulse workers must exactly reproduce P1 full physical traces and commands before pulse allocation; every pulse history must match its own baseline neural inputs, commands and spike total',
        stop='source/checkpoint/context/history/timing/force/mass/horizon/prefix/receipt/resource/timeout/remaining-full-worker-budget failure; no automatic retry or parameter tuning',
        physical_balance_validated=False,model_promoted=False,walking_proven=False)


def limited_commands(desired,rest):
    command=np.array(rest,copy=True);rows=[]
    for d in desired:
        command+=np.clip(d-command,-.06,.06);rows.append(command.copy())
    return np.stack(rows)


def validate(directory,model,enabled):
    physical=base.validate(directory,enabled);report=read(directory/'report.json');n=arrays(directory/'p2-neural-input-output.npz')
    assert report['p2_worker_completed'] and report['p2_context']['model']==model and report['p2_context']['condition']==('pulse' if enabled else 'baseline')
    assert report['p2_context']['history_sha256']==sha(REFERENCES[model]/'observations.npz')
    assert report['p2_context']==read(directory.with_name(directory.name+'-p2-context.json'))
    assert all(len(v)==1000 and np.isfinite(v).all() for v in n.values());np.testing.assert_array_equal(n['chunk'],np.arange(1000))
    live=arrays(directory/'observations.npz');history=arrays(REFERENCES[model]/'observations.npz')
    for k in FIELDS:
        np.testing.assert_array_equal(n['live_'+k][1:],live[k][:-1])
        np.testing.assert_array_equal(n['input_'+k][1:],history[k][:-1])
        np.testing.assert_array_equal(n['input_'+k][0],n['live_'+k][0])
        if not enabled:np.testing.assert_array_equal(n['input_'+k],n['live_'+k])
    refpads=arrays(REFERENCES[model]/'adhesion-observations.npz');pads=arrays(directory/'adhesion-observations.npz')
    for k in refpads:np.testing.assert_array_equal(pads[k],refpads[k],err_msg=k)
    expected=limited_commands(refpads['desired'],read(REFERENCES[model]/'report.json')['body']['neutral_pose'])
    np.testing.assert_array_equal(n['commands'][:,0,:],expected);np.testing.assert_array_equal(n['pads'][:,0,:],refpads['pads'])
    assert report['spikes_per_world']==read(REFERENCES[model]/'report.json')['spikes_per_world']
    # The intervention is at neural input, but identical sensory histories must
    # generate identical outputs from the unmodified deterministic frozen brain.
    if enabled:
        control=arrays(out/(model+'-baseline')/'p2-neural-input-output.npz')
        for k in FIELDS:np.testing.assert_array_equal(n['input_'+k],control['input_'+k])
        base.comparison(out/(model+'-baseline'),directory)
    else:
        for filename in ('observations.npz','adhesion-observations.npz','p1-telemetry.npz','p1-microforce.npz','anatomy-observations.npz'):
            left,right=arrays(directory/filename),arrays(REFERENCES[model]/filename);assert set(left)==set(right)
            for k in left:np.testing.assert_array_equal(left[k],right[k],err_msg=filename+':'+k)
    return dict(**physical,exact_history_neural_output_verified=True,exact_no_pulse_physical_replay=not enabled)


def describe(directory):
    a=arrays(directory/'p1-telemetry.npz');o=arrays(directory/'observations.npz');an=arrays(directory/'anatomy-observations.npz')
    # Native observation excludes the initial frame, whereas telemetry includes it.
    up=np.r_[np.nan,o['up_body'][:,0,2]];spin=np.r_[np.nan,np.linalg.norm(o['omega'][:,0],axis=1)]
    result={}
    for label,start in (('post_pulse',1.965),('final_1s',4.)):
        mask=a['time']>=start-1e-10;assert np.isfinite(up[mask]).all()
        result[label]=dict(samples=int(mask.sum()),fraction_up_below_point7=float((up[mask]<.7).mean()),
            tarsal_only_fraction=float((an['contact_mode_code'][mask]==3).mean()),
            mean_non_tarsal_vertical_fraction=float(an['non_tarsal_fraction'][mask].mean()),
            mean_COM_speed_mm_s=float(np.linalg.norm(a['com_velocity'][mask],axis=1).mean()),
            mean_abs_vertical_COM_speed_mm_s=float(np.abs(a['com_velocity'][mask,2]).mean()),
            mean_body_angular_speed_rad_s=float(spin[mask].mean()),
            COM_height_range_mm=[float(a['com'][mask,2].min()),float(a['com'][mask,2].max())])
    return result


def execute(p):
    me=psutil.Process();own={me.pid,*[x.pid for x in me.parents()]};files=read(root/'outputs/noesis-required-data-sha256.json')['files']
    def resources():
        if any(x.pid not in own and (x.info['name'] or '').lower() in ('python.exe','pythonw.exe') for x in psutil.process_iter(['name'])):raise RuntimeError('Competing Python')
        ram=psutil.virtual_memory().available/2**30
        if ram<2:raise RuntimeError('RAM below2GiB')
        return ram
    def unchanged():return all(sha(root/n)==h for group in ('source_hashes','prior_hashes','artifact_hashes') for n,h in p[group].items())
    def data_ok():return all(sha(root/'work/NoesisData'/f['path'])==f['sha256'] for f in files)
    resources();assert unchanged() and data_ok() and not out.exists()
    lock=root/'work/Project-Noesis/gpu-pilot/outputs/codex-training-host.lock';token=uuid.uuid4().hex
    with os.fdopen(os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY),'w') as f:json.dump(dict(owner='codex',pid=os.getpid(),token=token,deadline_unix=time.time()+1200),f)
    started=time.monotonic();empty=True;result=dict(status='running',jobs=[],cases={},matched_physical_descriptions={},model_promoted=False,walking_proven=False)
    env=dict(os.environ,PYTHONUTF8='1',MPLCONFIGDIR=str(root/'work/matplotlib-cache'))
    try:
        out.mkdir(exist_ok=False);write(out/'protocol.json',p)
        for name in p['sequence']:
            model,condition=name.split('-');assert unchanged() and data_ok();base.base.verify_native();ram=resources()
            if time.monotonic()-started>900:raise RuntimeError('Insufficient full300s worker budget')
            directory=out/name;print('START',name,flush=True);empty=False
            try:
                with (out/(name+'.log')).open('w',encoding='utf-8') as log:
                    receipt=base.base.base.run_windows_job([sys.executable,str(root/'work/p2_worker.py'),'--model',model,'--condition',condition,
                        '--root',str(root),'--output',str(directory),'--checkpoint',str(MODELS[model]),'--history',str(REFERENCES[model]/'observations.npz')],timeout=300,stdout=log,cwd=root,env=env)
                empty=receipt['tree_empty']
            except base.base.base.WatchdogError as error:
                empty=error.cleanup_verified;raise
            write(out/(name+'-receipt.json'),receipt);result['jobs'].append(dict(name=name,receipt=receipt,available_ram_before_gib=ram));write(out/'result.json',result)
            assert receipt['exit_code']==0 and not receipt['timed_out'] and empty
            result['cases'][name]=validate(directory,model,condition=='pulse');write(out/'result.json',result);print('COMPLETE',name,flush=True)
        for model in MODELS:
            result['matched_physical_descriptions'][model]=dict(live_feedback=describe(base.out/(model+'-pulse')),own_history_input=describe(out/(model+'-pulse')))
        result.update(status='completed_P2',physical_balance_validated=False,one_selected_diagnostic_context=True)
    except BaseException as error:
        result.update(status='failed',error_type=type(error).__name__,error=str(error));raise
    finally:
        result.update(seconds=time.monotonic()-started,source_prior_checkpoints_unchanged=unchanged(),data_unchanged=data_ok(),process_tree_empty_verified=empty)
        if empty and lock.exists() and read(lock)['token']==token:lock.unlink()
        result['lease_released']=not lock.exists()
        if out.exists():write(out/'result.json',result)
        print('RESULT',result['status'],result['seconds'],flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');a=ap.parse_args();p=prospective()
    if a.prepare:
        write(root/'outputs/sensory-causality-p2-20261107-protocol.json',p)
        write(root/'outputs/sensory-causality-p2-public-protocol-20261107.json',{k:v for k,v in p.items() if k not in ('source_hashes','prior_hashes','artifact_hashes')})
        print('P2 signatures verified and prospective protocol prepared; no simulation/lease')
    else:execute(p)


if __name__=='__main__':main()
