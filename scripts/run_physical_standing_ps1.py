"""PS1 fixed one-birth objective pilot; no automatic promotion or criterion relaxation."""
import argparse,json,os,sys,time,uuid,zipfile
import numpy as np
import psutil
import run_anatomical_ap2 as base
import audit_ap2 as prior
from ps1_worker import validate_training
from ps1_objective import physical_score,OBJECTIVE
from audit_ps1_objective import physical_data
from ap2_contract import DEV,anatomical_rule
from s2_contract import standing_retention

root=base.root;out=root/'outputs/physical-standing-ps1-20261107'
read,write,sha,arrays=base.read,base.write,base.sha,base.arrays
PARENT=base.PARENT
OLD_TRAIN=base.out/'train-candidate-1004'
SEEDS=(1004,*DEV)


def reference(bank,seed):
    return prior.directory(1004,{'parent':'parent','v7':'control','AP1':'candidate'}[bank],seed)


def prospective():
    previous=read(base.out/'protocol.json');source=dict(previous['source_hashes']);prior=dict(previous['prior_hashes']);artifacts=dict(previous['artifact_hashes'])
    for group in (source,prior,artifacts):
        for n,h in group.items():assert sha(root/n)==h
    base.verify_native();receipt=read(root/'outputs/anatomical-feedback-ap2-20261107-artifact.json');archive=root/'outputs'/receipt['path']
    assert sha(archive)==receipt['sha256']=='42e20e37d1ee95c550d65d2cb491d69895ae75777b4f279788110d1468d1373d'
    artifacts[archive.relative_to(root).as_posix()]=sha(archive)
    with zipfile.ZipFile(archive) as z:
        manifest=json.loads(z.read('manifest.json'));selected=[base.out/'result.json',OLD_TRAIN/'weights-last.npz',OLD_TRAIN/'report.json']
        selected += [OLD_TRAIN/n for n in ('training-observations.npz','adhesion-observations.npz','anatomy-observations.npz','exploration-log.npz','reward-accounting.npz')]
        for bank in ('parent','v7','AP1'):
            for seed in SEEDS:
                d=reference(bank,seed);selected += [d/n for n in ('report.json','observations.npz','adhesion-observations.npz')]
                selected += [d/n for n in (('anatomy-observations.npz','anatomy-model.json') if (d/'anatomy-observations.npz').exists() else ('body-telemetry.npz','ground-contacts.npz','telemetry-model.json'))]
        for path in selected:
            n=path.relative_to(root).as_posix();assert sha(path)==manifest[n]['sha256'];prior[n]=sha(path)
    audit_path=root/'outputs/physical-standing-ps1-objective-audit-private-20261107.json';audit=read(audit_path)
    assert audit['status']=='completed_readonly_PS1_objective_audit' and audit['unique_cases']==37 and audit['learning_updates']==0
    for group in ('source_hashes','input_hashes'):
        for n,h in audit[group].items():assert sha(root/n)==h
    prior[audit_path.relative_to(root).as_posix()]=sha(audit_path)
    for n in ('ps1_objective.py','test_ps1_objective.py','audit_ps1_objective.py','ps1_worker.py','run_physical_standing_ps1.py','audit_ap2.py','audit_ap1c.py'):
        source['work/'+n]=sha(root/'work'/n)
    assert sha(PARENT)=='9fd7d41177bf2c745809401c3b142a832661d1a18d0b117b30cb02aea63fea3c'
    assert sha(OLD_TRAIN/'weights-last.npz')=='068b49a0d44f4dca6fd53bd0950048a4db1a3ebd891ac0e0727ff0a39ff33c10'
    return dict(experiment='PS1 fixed physical-standing scalar-objective pilot',date='2026-11-07',source_hashes=source,prior_hashes=prior,artifact_hashes=artifacts,
        common_parent_sha256=sha(PARENT),control_training_sha256=sha(OLD_TRAIN/'weights-last.npz'),training_seed=1004,development_seeds=list(DEV),own_seed_descriptive=True,
        objective=OBJECTIVE,formula='2*upright*tarsal_share*vertical_load_quality*claw_support*motion_quality-1',
        components=dict(upright='clip((up.z-.7)/.3,0,1)',tarsal_share='(claw+other_tarsal vertical reaction)/all vertical reaction, zero if unsupported',
            vertical_load_quality='exp(-abs(total_vertical_reaction/whole_fly_weight-1))',claw_support='min(number_loaded_claws/3,1)',
            motion_quality='1/(1+(COM_speed/1mm_s)^2+(body_angular_speed/1rad_s)^2)'),
        scale_status='Fixed engineering modelling assumptions, not biological calibration or physical acceptance thresholds',
        factor='Only scalar objective: exact AP1 control versus PS1 candidate from same parent; no height term in PS1, original height/evaluation metrics retained unchanged',
        seconds_train=25,seconds_frozen=5,learning_updates_per_birth=4990,learning_rate=.2,eligibility_ms=100,original_reference_bound=.5,exploration_mv_ms=1.5,
        aggregate_seconds=2700,training_worker_seconds=600,frozen_worker_seconds=300,maximum_new_jobs=7,
        sequence=['exact AP1 control birth1004','PS1 candidate birth1004','candidate frozen own1004','candidate frozen10006','candidate frozen10007','candidate frozen10010','candidate frozen10011'],
        reused_frozen_reference_cells=15,reference_banks=['parent','v7','AP1'],
        primary='Exploratory all4dev: PS1 mean quality higher and mean COMspeed/body angular speed/non-tarsal reaction fraction lower versus each reference; no own-seed selection or aggregate rescue',
        retention='Report unchanged strict standing/slip/heading/netforward/cycles/tripod criteria against parent, v7 and AP1 separately; retain old anatomical comparison versus parent/AP1; none replaced by PS1 reward',
        limits='One learning birth pilot; no physical standing gate, generalized efficacy, retained ability, gait or validated-best claim. Even positive pilot requires independent births and physical validation.',
        stop='technical/hash/context/mass/force/accounting/sidecar/exactcontrol/prefix/horizon/resource/timeout/insufficient-full-worker-budget failure; no automatic retry, retuning or partial inference',
        model_promoted=False,physical_balance_validated=False,walking_proven=False)


def training_checks(directory,bank):
    report=read(directory/'report.json');validate_training(report,bank);assert report['ps1_context']==read(directory.with_name(directory.name+'-ps1-context.json'))
    assert report['import_checkpoint_sha256']==sha(PARENT)
    checkpoint=directory/'weights-last.npz';assert sha(checkpoint)==report['saved_checkpoint_sha256']
    rule=base.base.compare_checkpoints(PARENT,checkpoint);assert rule['compatible'] and rule['rule']['match'] and rule['parameters']['match']
    sidecar=read(checkpoint.with_suffix('.model.json'))
    for k,v in report.items():assert sidecar[k]==v
    a=base.base.anatomy(directory,'train');log=arrays(directory/'physical-reward-accounting.npz');o=arrays(directory/'training-observations.npz')
    assert all(len(x)==5000 and np.isfinite(x).all() for x in log.values());np.testing.assert_array_equal(log['control_chunk_index'],np.arange(5000))
    np.testing.assert_array_equal(log['time'],a['time'][1:]);np.testing.assert_array_equal(log['applied_reward'],o['reward'])
    m=read(directory/'anatomy-model.json');p=physical_score(a['region_vertical_reaction'][1:],a['com_velocity'][1:],o['omega'],o['up_body'][:,2],o['foot_load'],m['total_mass']*abs(m['gravity'][2]))
    # Scalar online versus vector offline reductions may differ by roundoff only.
    for k,v in p.items():np.testing.assert_allclose(log['PS1_'+k],v,rtol=1e-14,atol=1e-15)
    expected=np.clip(log['native_reward']-.5*a['non_tarsal_fraction'][1:],-1,1)
    np.testing.assert_array_equal(log['AP1_reference_reward'],expected)
    np.testing.assert_array_equal(log['applied_reward'],expected if bank=='control' else log['PS1_reward'])
    if bank=='control':
        assert sha(checkpoint)==sha(OLD_TRAIN/'weights-last.npz')
        for n in ('training-observations.npz','adhesion-observations.npz','anatomy-observations.npz','exploration-log.npz'):base.base.exact(directory/n,OLD_TRAIN/n)
    else:
        control=out/'train-control';base.base.exact(directory/'exploration-log.npz',control/'exploration-log.npz')
        for n in ('training-observations.npz','adhesion-observations.npz'):
            c,d=arrays(control/n),arrays(directory/n);assert set(c)==set(d)
            for k in c:
                if k!='reward':np.testing.assert_array_equal(c[k][:11],d[k][:11],err_msg=k)
    return dict(full_horizon=True,checkpoint_sha256=sha(checkpoint),weights_invariants=True,objective=report['reward_objective'],exact_control_replay=bank=='control')


def physical_summary(directory):
    o={k:v[:,0] for k,v in arrays(directory/'observations.npz').items()};r,v,weight=physical_data(directory)
    p=physical_score(r,v,o['omega'],o['up_body'][:,2],o['foot_load'],weight);q=np.divide(r[:,0]+r[:,3],r.sum(1),out=np.zeros(1000),where=r.sum(1)>1e-5)
    return dict(mean_PS1_quality=float(p['quality'][10:].mean()),mean_COM_speed_mm_s=float(np.linalg.norm(v[10:],axis=1).mean()),
        mean_angular_speed_rad_s=float(np.linalg.norm(o['omega'][10:],axis=1).mean()),mean_non_tarsal_fraction=float(q[10:].mean()),
        samples=990,physical_balance_validated=False)


def finish():
    banks={}
    for bank in ('parent','v7','AP1','candidate'):
        banks[bank]={}
        for seed in SEEDS:
            d=out/f'frozen-candidate-{seed}' if bank=='candidate' else reference(bank,seed)
            _,summary=base.base.summarize(d);banks[bank][seed]=dict(summary=summary,physical=physical_summary(d),anatomical=prior.previous.contact_data(d))
    child={s:banks['candidate'][s]['summary'] for s in DEV}
    retention={bank:standing_retention({s:banks[bank][s]['summary'] for s in DEV},child,DEV) for bank in ('parent','v7','AP1')}
    paired={};passed=True
    for seed in DEV:
        paired[str(seed)]={};c=banks['candidate'][seed]['physical']
        for bank in ('parent','v7','AP1'):
            ref=banks[bank][seed]['physical'];d={k:c[k]-ref[k] for k in ('mean_PS1_quality','mean_COM_speed_mm_s','mean_angular_speed_rad_s','mean_non_tarsal_fraction')}
            ok=d['mean_PS1_quality']>0 and all(d[k]<0 for k in d if k!='mean_PS1_quality');passed=passed and ok
            paired[str(seed)][bank]=dict(changes=d,all_physical_directions_improve=bool(ok))
    anatomy=anatomical_rule(*({s:banks[bank][s]['anatomical'] for s in DEV} for bank in ('parent','AP1','candidate')))
    return dict(banks=banks,retention=retention,anatomical=anatomy,paired_physical=paired,all_four_development_physical_directions_improve=bool(passed),
        combined_exploratory_pilot_pass=bool(passed and all(x['passed'] for x in retention.values()) and anatomy['passed']),
        physical_balance_validated=False,model_promoted=False,walking_proven=False)


def execute(p):
    me=psutil.Process();own={me.pid,*[x.pid for x in me.parents()]};data=read(root/'outputs/noesis-required-data-sha256.json')['files']
    def resources():
        if any(x.pid not in own and (x.info['name'] or '').lower() in ('python.exe','pythonw.exe') for x in psutil.process_iter(['name'])):raise RuntimeError('Competing Python')
        ram=psutil.virtual_memory().available/2**30
        if ram<2:raise RuntimeError('RAM below2GiB')
        return ram
    def unchanged():return all(sha(root/n)==h for group in ('source_hashes','prior_hashes','artifact_hashes') for n,h in p[group].items())
    def data_ok():return all(sha(root/'work/NoesisData'/f['path'])==f['sha256'] for f in data)
    resources();assert unchanged() and data_ok() and not out.exists()
    lock=root/'work/Project-Noesis/gpu-pilot/outputs/codex-training-host.lock';token=uuid.uuid4().hex
    with os.fdopen(os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY),'w') as f:json.dump(dict(owner='codex',pid=os.getpid(),token=token,deadline_unix=time.time()+2700),f)
    started=time.monotonic();empty=True;result=dict(status='running',jobs=[],training={},model_promoted=False,walking_proven=False)
    jobs=[('train','control',1004),('train','candidate',1004)]+[('frozen','candidate',s) for s in SEEDS]
    env=dict(os.environ,PYTHONUTF8='1',MPLCONFIGDIR=str(root/'work/matplotlib-cache'))
    try:
        out.mkdir(exist_ok=False);write(out/'protocol.json',p)
        for phase,bank,seed in jobs:
            assert unchanged() and data_ok();base.verify_native();ram=resources();timeout=600 if phase=='train' else 300
            if time.monotonic()-started+timeout>2700:raise RuntimeError('Insufficient full worker budget')
            name=f'train-{bank}' if phase=='train' else f'frozen-candidate-{seed}';cp=PARENT if phase=='train' else out/'train-candidate/weights-last.npz'
            print('START',name,flush=True);empty=False
            try:
                with (out/(name+'.log')).open('w',encoding='utf-8') as log:
                    receipt=base.base.run_windows_job([sys.executable,str(root/'work/ps1_worker.py'),'--phase',phase,'--bank',bank,'--seed',str(seed),'--root',str(root),
                        '--output',str(out/name),'--checkpoint',str(cp)],timeout=timeout,stdout=log,cwd=root,env=env)
                empty=receipt['tree_empty']
            except base.base.WatchdogError as error:
                empty=error.cleanup_verified;raise
            write(out/(name+'-receipt.json'),receipt);result['jobs'].append(dict(name=name,phase=phase,bank=bank,seed=seed,receipt=receipt,available_ram_before_gib=ram));write(out/'result.json',result)
            assert receipt['exit_code']==0 and not receipt['timed_out'] and empty
            if phase=='train':result['training'][bank]=training_checks(out/name,bank)
            else:
                report=read(out/name/'report.json');assert report['checkpoint_sha256']==sha(cp) and report['ps1_worker_completed'] and not report['learning'] and report['weights_unchanged']
                base.base.anatomy(out/name,'frozen');base.base.compare(reference('parent',seed),out/name)
            write(out/'result.json',result);print('COMPLETE',name,flush=True)
        assert len(result['jobs'])==7;result.update(status='completed_PS1',science=finish())
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
        write(root/'outputs/physical-standing-ps1-20261107-protocol.json',p)
        write(root/'outputs/physical-standing-ps1-public-protocol-20261107.json',{k:v for k,v in p.items() if k not in ('source_hashes','prior_hashes','artifact_hashes')})
        print('PS1 prepared, signed controls/audit verified, no simulation or lease')
    else:execute(p)


if __name__=='__main__':main()
