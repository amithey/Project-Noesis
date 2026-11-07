"""Fixed P1 calibration: exact neural baselines, then environmental-force pairs."""
import argparse,json,os,sys,time,uuid,zipfile
import numpy as np
import psutil
import run_anatomical_ap2 as base
from p1_telemetry import START,WIDTH,FRACTION,CONTROL_DT,PHYSICS_DT

root=base.root;out=root/'outputs/physical-calibration-p1-20261107'
read,write,sha,arrays=base.read,base.write,base.sha,base.arrays
MODELS={'parent':base.PARENT,'candidate':base.out/'train-candidate-1004/weights-last.npz'}
OLD={'parent':base.parent_dev(10011),'candidate':base.out/'frozen-candidate-1004-10011'}


def prospective():
    old=read(base.out/'protocol.json');source=dict(old['source_hashes']);prior=dict(old['prior_hashes']);archives={}
    for group in (source,prior,old['artifact_hashes']):
        for n,h in group.items():assert sha(root/n)==h
    base.verify_native()
    receipt=read(root/'outputs/anatomical-feedback-ap2-20261107-artifact.json');archive=root/'outputs'/receipt['path']
    assert sha(archive)==receipt['sha256']=='42e20e37d1ee95c550d65d2cb491d69895ae75777b4f279788110d1468d1373d'
    archives[archive.relative_to(root).as_posix()]=sha(archive)
    with zipfile.ZipFile(archive) as z:
        manifest=json.loads(z.read('manifest.json'))
        selected=[base.out/'result.json',root/'outputs/anatomical-feedback-ap2-20261107-summary.json',MODELS['candidate']]
        selected += [d/n for d in OLD.values() for n in ('report.json','observations.npz','adhesion-observations.npz','anatomy-observations.npz','anatomy-model.json')]
        for p in selected:
            n=p.relative_to(root).as_posix();h=manifest[n]['sha256'];assert sha(p)==h;prior[n]=h
    selected_path=root/'outputs/physical-calibration-p1-selected-case-20261107.json';selection=read(selected_path)
    with zipfile.ZipFile(root/'outputs/anatomical-feedback-ap2-failure-audit-20261107-evidence.zip') as z:
        m=json.loads(z.read('manifest.json'));n=selected_path.relative_to(root).as_posix();assert sha(selected_path)==m[n]['sha256'];prior[n]=sha(selected_path)
    assert selection['event_chunk']==START and selection['event_time_seconds']==START*CONTROL_DT
    obs=arrays(OLD['candidate']/'observations.npz');a=arrays(OLD['candidate']/'anatomy-observations.npz');i=START-1
    assert obs['position'][i,0,2]>1.5 and obs['up_body'][i,0,2]>=.7 and (obs['foot_load'][i,0]>1e-5).sum()>=3 and a['contact_mode_code'][START]==3
    assert sha(MODELS['parent'])=='9fd7d41177bf2c745809401c3b142a832661d1a18d0b117b30cb02aea63fea3c'
    assert sha(MODELS['candidate'])=='068b49a0d44f4dca6fd53bd0950048a4db1a3ebd891ac0e0727ff0a39ff33c10'
    for cp in MODELS.values():prior[cp.relative_to(root).as_posix()]=sha(cp)
    for n in ('p1_worker.py','p1_telemetry.py','run_physical_p1.py','test_p1_contract.py'):source['work/'+n]=sha(root/'work'/n)
    return dict(experiment='P1 frozen physical-disturbance calibration',date='2026-11-07',source_hashes=source,prior_hashes=prior,artifact_hashes=archives,
        checkpoint_sha256={k:sha(v) for k,v in MODELS.items()},seed=10011,seconds=5,chunks=1000,anatomy_samples=1001,microsteps=50000,
        learning_updates=0,motor_exploration=0,neural_controller_unchanged=True,
        sequence=['parent-baseline','candidate-baseline','parent-pulse','candidate-pulse'],maximum_new_jobs=4,aggregate_seconds=1200,worker_seconds=300,
        pulse=dict(world_axis='+X',target='compiled thorax body COM',start_chunk=START,start_seconds=START*CONTROL_DT,width_chunks=WIDTH,duration_seconds=WIDTH*CONTROL_DT,force_fraction_of_weight=FRACTION),
        force_epoch='Verified before/after EVERYmj_step. Snapshot force is preceding-chunk input held until next advance, not a new instantaneous force outside scheduled physics steps.',
        prefix=dict(native_body_rows=START,anatomy_full_telemetry_rows=START+1,neural_command_rows=START+1),
        analysis='Each model relative to its own exact baseline: COM/up/contact/command/adhesion deviations and actual impulse, no binary recovery/ranking/gait/retention verdict',
        selection_is_moving_known_case_not_standing_positive=True,new_gate=False,model_promoted=False,walking_proven=False,
        stop='hash/context/mass/force/horizon/prefix/resource/timeout/remaining-full-worker-budget failure; no automatic retry/retune or partial interpretation')


def validate(directory,enabled):
    report=read(directory/'report.json');a=arrays(directory/'p1-telemetry.npz');micro=arrays(directory/'p1-microforce.npz');m=read(directory/'p1-model.json')
    assert report['status']=='completed_transport_pilot' and report['p1_worker_completed'] and report['weights_unchanged'] and not report['learning'] and report['motor_exploration']==0 and report['seeds']==[10011]
    assert all(len(v)==1001 and np.isfinite(v).all() for v in a.values()) and micro['force'].shape==(50000,3)
    np.testing.assert_allclose(a['time'],np.arange(1001)*CONTROL_DT,rtol=0,atol=1e-10);np.testing.assert_allclose(micro['time'],np.arange(50000)*PHYSICS_DT,rtol=0,atol=1e-10)
    np.testing.assert_array_equal(a['completed_control_chunks'],np.arange(1001))
    assert np.isclose(m['total_mass'],m['model_total_mass'],rtol=1e-12,atol=1e-15) and m['pulse_enabled']==enabled and m['production_static_fusion']
    expected=np.zeros((50000,3))
    if enabled:expected[START*50:(START+WIDTH)*50,0]=FRACTION*m['whole_fly_weight']
    np.testing.assert_array_equal(micro['force'],expected)
    frame_force=np.zeros((1001,3))
    if enabled:frame_force[START+1:START+WIDTH+1,0]=FRACTION*m['whole_fly_weight']
    np.testing.assert_array_equal(a['force_used_preceding_chunk'],frame_force)
    np.testing.assert_array_equal(a['xfrc_applied'][:,m['pulse_target_body'],:3],frame_force)
    assert np.count_nonzero(a['xfrc_applied'])==(WIDTH if enabled else 0)
    dof=m['free_dof'];np.testing.assert_allclose(a['qfrc_xfrc'][:,dof:dof+3],frame_force,rtol=0,atol=1e-12)
    for k in ('smooth_force_max_error','dynamic_force_max_error'):assert np.all(a[k]<=1e-9*a['dynamic_force_scale']+1e-10)
    np.testing.assert_allclose(a['mass_times_qacc'],a['qfrc_smooth']+a['qfrc_constraint'],rtol=1e-9,atol=1e-8)
    np.testing.assert_allclose(a['qfrc_smooth'],a['qfrc_passive']-a['qfrc_bias']+a['qfrc_actuator']+a['qfrc_applied']+a['qfrc_xfrc'],rtol=1e-9,atol=1e-8)
    np.testing.assert_allclose(a['ground_force'],a['root_constraint_force'],rtol=1e-6,atol=1e-8)
    base.base.anatomy(directory,'frozen')
    return dict(impulse_world=(micro['force'].sum(0)*PHYSICS_DT).tolist(),active_microsteps=int(np.count_nonzero(micro['force'][:,0])),
        max_dynamic_force_error=float(a['dynamic_force_max_error'].max()),max_smooth_force_error=float(a['smooth_force_max_error'].max()),full_horizon=True,physical_balance_validated=False)


def comparison(baseline,pulse):
    b,p=arrays(baseline/'p1-telemetry.npz'),arrays(pulse/'p1-telemetry.npz');bo,po=arrays(baseline/'observations.npz'),arrays(pulse/'observations.npz')
    for k in b:np.testing.assert_array_equal(p[k][:START+1],b[k][:START+1],err_msg=k)
    for k in bo:np.testing.assert_array_equal(po[k][:START],bo[k][:START],err_msg=k)
    ba,pa=arrays(baseline/'anatomy-observations.npz'),arrays(pulse/'anatomy-observations.npz')
    for k in ba:np.testing.assert_array_equal(pa[k][:START+1],ba[k][:START+1],err_msg=k)
    bd,pd=arrays(baseline/'adhesion-observations.npz'),arrays(pulse/'adhesion-observations.npz')
    for k in bd:np.testing.assert_array_equal(pd[k][:START+1],bd[k][:START+1],err_msg=k)
    distance=np.linalg.norm(p['com']-b['com'],axis=1);speed=np.linalg.norm(p['com_velocity']-b['com_velocity'],axis=1)
    command=np.linalg.norm(pd['desired']-bd['desired'],axis=1);pads=(pd['pads']!=bd['pads']).any(axis=1)
    return dict(exact_prefix_verified=True,max_COM_deviation_mm=float(distance.max()),max_COM_velocity_deviation_mm_s=float(speed.max()),
        final_1s_RMS_COM_deviation_mm=float(np.sqrt(np.mean(distance[-200:]**2))),final_1s_mean_velocity_deviation_mm_s=float(speed[-200:].mean()),
        changed_neural_command_rows=int(np.count_nonzero(command)),maximum_neural_command_L2_difference_radians=float(command.max()),changed_adhesion_rows=int(pads.sum()),
        different_contact_mode_samples=int(np.count_nonzero(pa['contact_mode_code']!=ba['contact_mode_code'])),
        max_up_vector_difference=float(np.linalg.norm(po['up_body']-bo['up_body'],axis=2).max()),
        first_changed_command_chunk=int(np.flatnonzero(command)[0]) if np.any(command) else None,
        mechanical_response_is_not_neural_or_balance_qualification=True)


def execute(p):
    me=psutil.Process();own={me.pid,*[x.pid for x in me.parents()]};data=read(root/'outputs/noesis-required-data-sha256.json')['files']
    def resources():
        if any(x.pid not in own and (x.info['name'] or '').lower() in ('python.exe','pythonw.exe') for x in psutil.process_iter(['name'])):raise RuntimeError('Competing Python')
        ram=psutil.virtual_memory().available/2**30
        if ram<2:raise RuntimeError('RAM below2GiB')
        return ram
    def unchanged():return all(sha(root/n)==h for group in ('source_hashes','prior_hashes') for n,h in p[group].items())
    def data_ok():return all(sha(root/'work/NoesisData'/f['path'])==f['sha256'] for f in data)
    resources();assert unchanged() and data_ok() and not out.exists()
    lock=root/'work/Project-Noesis/gpu-pilot/outputs/codex-training-host.lock';token=uuid.uuid4().hex
    with os.fdopen(os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY),'w') as f:json.dump(dict(owner='codex',pid=os.getpid(),token=token,deadline_unix=time.time()+1200),f)
    started=time.monotonic();empty=True;result=dict(status='running',jobs=[],cases={},responses={},new_gate=False,model_promoted=False,walking_proven=False)
    env=dict(os.environ,PYTHONUTF8='1',MPLCONFIGDIR=str(root/'work/matplotlib-cache'))
    try:
        out.mkdir(exist_ok=False);write(out/'protocol.json',p)
        for name in p['sequence']:
            model,condition=name.split('-');enabled=condition=='pulse';assert unchanged() and data_ok();base.verify_native();ram=resources()
            if time.monotonic()-started>900:raise RuntimeError('Insufficient full300s worker budget')
            directory=out/name;print('START',name,flush=True);empty=False
            try:
                with (out/(name+'.log')).open('w',encoding='utf-8') as log:
                    receipt=base.base.run_windows_job([sys.executable,str(root/'work/p1_worker.py'),'--model',model,'--condition',condition,'--root',str(root),'--output',str(directory),'--checkpoint',str(MODELS[model])],timeout=300,stdout=log,cwd=root,env=env)
                empty=receipt['tree_empty']
            except base.base.WatchdogError as error:
                empty=error.cleanup_verified;raise
            write(out/(name+'-receipt.json'),receipt);result['jobs'].append(dict(name=name,receipt=receipt,available_ram_before_gib=ram));write(out/'result.json',result)
            assert receipt['exit_code']==0 and not receipt['timed_out'] and empty
            report=read(directory/'report.json');assert report['checkpoint_sha256']==sha(MODELS[model]) and report['p1_context']['model']==model and report['p1_context']['condition']==condition
            result['cases'][name]=validate(directory,enabled)
            if not enabled:
                for n in ('observations.npz','adhesion-observations.npz'):base.base.exact(directory/n,OLD[model]/n)
                assert report['spikes_per_world']==read(OLD[model]/'report.json')['spikes_per_world'];base.base.compare(OLD[model],directory)
                old_a,new_a=arrays(OLD[model]/'anatomy-observations.npz'),arrays(directory/'anatomy-observations.npz')
                for k in old_a:np.testing.assert_array_equal(new_a[k],old_a[k],err_msg=k)
                result['cases'][name]['exact_prior_neural_body_pad_spike_replay']=True
            else:
                base.base.compare(out/(model+'-baseline'),directory);result['responses'][model]=comparison(out/(model+'-baseline'),directory)
            write(out/'result.json',result);print('COMPLETE',name,flush=True)
        assert len(result['jobs'])==4;result.update(status='completed_P1',one_selected_diagnostic_context=True,physical_balance_validated=False,neural_feedback_qualified=False)
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
        write(root/'outputs/physical-calibration-p1-20261107-protocol.json',p)
        public={k:v for k,v in p.items() if k not in ('source_hashes','prior_hashes','artifact_hashes')};public['publication_scope']='Compact scientific protocol only; extensive private integrity maps local'
        write(root/'outputs/physical-calibration-p1-public-protocol-20261107.json',public);print('P1 prepared, signatures/selection verified, no simulation/lease')
    else:execute(p)


if __name__=='__main__':main()
