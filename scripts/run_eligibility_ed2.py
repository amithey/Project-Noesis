"""ED2 fixed-method replication; private input pins are supplied locally, never published."""
import argparse,json,os,sys,time,uuid,zipfile
import numpy as np
import psutil
import run_eligibility_ed1 as old
from ed2_worker import validate_training
from ed2_contract import NEW_BIRTHS,BIRTHS,replication_summary
from ap2_contract import DEV

root=old.root;out=root/'outputs/eligibility-replication-ed2-20261107'
read,write,sha,arrays=old.read,old.write,old.sha,old.arrays
PARENT=old.PARENT
native=old.previous.base.base


def directory(birth,bank,seed):
    if bank in ('v7','AP1','previous_PS1'):return old.reference(bank,seed)
    if birth==1005:
        return old.reference('parent',seed) if bank=='parent' else old.out/f'frozen-{bank}-{seed}'
    if bank=='parent':
        if seed in DEV:return old.reference('parent',seed)
        return old.previous.base.out/'frozen-parent-1006-1006' if seed==1006 else out/'frozen-parent-1007'
    return out/f'frozen-{bank}-{birth}-{seed}'


def prospective(pins_path):
    previous=read(old.out/'protocol.json');source=dict(previous['source_hashes']);prior=dict(previous['prior_hashes']);artifacts=dict(previous['artifact_hashes'])
    for group in (source,prior,artifacts):
        for n,h in group.items():assert sha(root/n)==h
    old.previous.base.verify_native();pins=read(pins_path);receipt_path=root/pins['receipt_path']
    assert sha(receipt_path)==pins['receipt_sha256'];receipt=read(receipt_path);archive=root/pins['artifact_folder']/receipt['path']
    assert sha(archive)==receipt['sha256']==pins['artifact_sha256']
    artifacts[archive.relative_to(root).as_posix()]=pins['artifact_sha256']
    prior[receipt_path.relative_to(root).as_posix()]=sha(receipt_path);prior[pins_path.relative_to(root).as_posix()]=sha(pins_path)
    with zipfile.ZipFile(archive) as z:manifest=json.loads(z.read('manifest.json'))
    for n,entry in manifest.items():assert sha(root/n)==entry['sha256'];prior[n]=entry['sha256']
    previous_result=read(old.out/'result.json');assert previous_result['status']=='completed_ED1' and previous_result['lease_released']
    assert not previous_result['science']['combined_exploratory_pilot_pass']
    # The extra own-seed reference is from the already signed public AP2 bank.
    extra=directory(1006,'parent',1006);known=root/'outputs/anatomical-feedback-ap2-20261107-evidence.zip'
    key=known.relative_to(root).as_posix();assert sha(known)==artifacts[key]
    with zipfile.ZipFile(known) as z:m=json.loads(z.read('manifest.json'))
    for name in ('report.json','observations.npz','adhesion-observations.npz','anatomy-observations.npz','anatomy-model.json'):
        p=extra/name;n=p.relative_to(root).as_posix();assert sha(p)==m[n]['sha256'];prior[n]=sha(p)
    for n in ('ed2_contract.py','test_ed2_contract.py','ed2_worker.py','run_eligibility_ed2.py'):
        source['work/'+n]=sha(root/'work'/n)
    assert sha(PARENT)=='9fd7d41177bf2c745809401c3b142a832661d1a18d0b117b30cb02aea63fea3c'
    return dict(experiment='ED2 fixed-method independent training-birth replication',date='2026-11-07',source_hashes=source,prior_hashes=prior,artifact_hashes=artifacts,
        common_parent_sha256=sha(PARENT),original_training_seed=1005,new_training_seeds=list(NEW_BIRTHS),development_seeds=list(DEV),own_seed_descriptive=True,
        fixed_method='Same PS1 scalar objective; native0 versus full-eligibility delay10chunks50ms; no coefficient/delay sweep',
        objective='physical-standing-ps1-v1',learning_rate=.2,eligibility_ms=100,original_reference_bound=.5,exploration_mv_ms=1.5,
        seconds_train=25,learning_attempts_per_birth=4990,seconds_frozen=5,maximum_new_jobs=26,aggregate_seconds=9300,training_worker_seconds=600,frozen_worker_seconds=300,
        sequence=['exact prior positive-delay1005 replay','new parent-own1007 frozen reference','paired0/50ms25s births1006 then10matched frozen probes','paired0/50ms25s births1007 then10matched frozen probes'],
        original_failed_birth_required=True,main_frozen_bank_positions=45,main_unique_cells=37,auxiliary_unique_cells=12,
        primary='Unchanged ED1 EACHdev physical directions versus parent/control, strict original retention versus parent/control/v7/AP1 and anatomical rule; every birth reported including failed1005',
        secondary='Predeclared descriptive four-development-seed mean physical changes versus concurrent control per birth; never rescue primary or select winner',
        proof='Exact positive-delay replay before new births; same reward/motor/native-run AST; every actual queue/weight-step/moment check, raw innovation distinguished from used eligibility; original bounds and source/context guards',
        final_reserved_seeds_unused=[10,11,200,201,202],
        stop='hash/context/mass/force/reward/queue/weight/checkpoint/sidecar/replay/prefix/horizon/resource/time/full-worker-budget failure; no automatic retry/retune/partial inference',
        limits='Three training births share development reference cells; not independent frozen replications or proof of physical ability. No model promotion or winner selection.',
        physical_balance_validated=False,model_promoted=False,walking_proven=False)


def training_checks(d,bank,birth):
    r=read(d/'report.json');validate_training(r,bank);assert r['ed2_context']==read(d.with_name(d.name+'-ed2-context.json'))
    assert r['seed']==birth and r['ed2_context']['training_seed']==birth and r['import_checkpoint_sha256']==sha(PARENT)
    cp=d/'weights-last.npz';assert sha(cp)==r['saved_checkpoint_sha256'];rule=native.compare_checkpoints(PARENT,cp)
    assert rule['compatible'] and rule['rule']['match'] and rule['parameters']['match']
    side=read(cp.with_suffix('.model.json'))
    for k,v in r.items():assert side[k]==v
    a=native.anatomy(d,'train');account=arrays(d/'physical-reward-accounting.npz');o=arrays(d/'training-observations.npz')
    assert all(len(v)==5000 and np.isfinite(v).all() for v in account.values())
    np.testing.assert_array_equal(account['control_chunk_index'],np.arange(5000));np.testing.assert_array_equal(account['time'],a['time'][1:])
    np.testing.assert_array_equal(account['applied_reward'],o['reward']);np.testing.assert_array_equal(o['reward'],account['PS1_reward'])
    m=read(d/'anatomy-model.json');physical=old.previous.physical_score(a['region_vertical_reaction'][1:],a['com_velocity'][1:],o['omega'],o['up_body'][:,2],o['foot_load'],m['total_mass']*abs(m['gravity'][2]))
    for k,v in physical.items():np.testing.assert_allclose(v,account['PS1_'+k],rtol=1e-14,atol=1e-15)
    eligibility=old.eligibility_checks(d,10 if bank in ('replay','candidate') else 0)
    if bank=='replay':
        original=old.out/'train-candidate';assert sha(cp)==sha(original/'weights-last.npz')
        for n in ('training-observations.npz','adhesion-observations.npz','anatomy-observations.npz','physical-reward-accounting.npz','exploration-log.npz','eligibility-audit.npz'):native.exact(d/n,original/n)
    elif bank=='candidate':
        control=out/f'train-control-{birth}';native.exact(d/'exploration-log.npz',control/'exploration-log.npz')
        for n in ('training-observations.npz','adhesion-observations.npz'):
            a,b=arrays(control/n),arrays(d/n);assert set(a)==set(b)
            for k in a:np.testing.assert_array_equal(a[k][:11],b[k][:11])
    return dict(checkpoint_sha256=sha(cp),full_horizon=True,original_weight_invariants=True,eligibility=eligibility,seed=birth,delay_chunks=10 if bank in ('candidate','replay') else 0,exact_prior_positive_replay=bank=='replay')


def finish():
    births={};positions=[];unique=set()
    for birth in BIRTHS:
        banks={}
        for bank in ('parent','control','candidate','v7','AP1','previous_PS1'):
            banks[bank]={}
            for seed in ((birth,*DEV) if bank in ('parent','control','candidate') else DEV):
                d=directory(birth,bank,seed);unique.add(d)
                _,summary=native.summarize(d);banks[bank][seed]=dict(summary=summary,physical=old.previous.physical_summary(d),anatomical=old.previous.prior.previous.contact_data(d))
                if bank in ('parent','control','candidate'):positions.append(dict(training_seed=birth,bank=bank,seed=seed))
        births[birth]=banks
    assert len(positions)==45 and len(unique)==49
    result=replication_summary(births);original=read(old.out/'result.json')['science'];first=result['births'][0]
    for k in ('retention','anatomical','paired_physical','all_four_development_physical_directions_improve','combined_exploratory_pilot_pass'):assert first[k]==original[k]
    assert not first['combined_exploratory_pilot_pass']
    return dict(replication=result,banks=births,main_positions=positions,unique_frozen_cells=49)


def execute(p):
    me=psutil.Process();own={me.pid,*[x.pid for x in me.parents()]};data=read(root/'outputs/noesis-required-data-sha256.json')['files'];children={}
    def resources():
        if any(x.pid not in own and (x.info['name'] or '').lower() in ('python.exe','pythonw.exe') for x in psutil.process_iter(['name'])):raise RuntimeError('Competing Python')
        ram=psutil.virtual_memory().available/2**30
        if ram<2 or psutil.disk_usage(str(root)).free<2*2**30:raise RuntimeError('RAM/disk below2GiB')
        return ram
    def unchanged():return all(sha(root/n)==h for group in ('source_hashes','prior_hashes','artifact_hashes') for n,h in p[group].items())
    def children_ok():return all(sha(path)==h for path,h in children.items())
    def data_ok():return all(sha(root/'work/NoesisData'/f['path'])==f['sha256'] for f in data)
    resources();assert unchanged() and data_ok() and not out.exists()
    lock=root/'work/Project-Noesis/gpu-pilot/outputs/codex-training-host.lock';token=uuid.uuid4().hex
    with os.fdopen(os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY),'w') as f:json.dump(dict(owner='codex',pid=os.getpid(),token=token,deadline_unix=time.time()+9300),f)
    started=time.monotonic();empty=True;result=dict(status='running',jobs=[],training={},model_promoted=False,walking_proven=False)
    jobs=[('train','replay',1005,1005),('frozen','parent',1007,1007)]
    for birth in NEW_BIRTHS:
        jobs += [('train','control',birth,birth),('train','candidate',birth,birth)]+[('frozen',b,birth,s) for b in ('control','candidate') for s in (birth,*DEV)]
    assert len(jobs)==26
    env=dict(os.environ,PYTHONUTF8='1',MPLCONFIGDIR=str(root/'work/matplotlib-cache'))
    try:
        out.mkdir(exist_ok=False);write(out/'protocol.json',p)
        for phase,bank,birth,seed in jobs:
            assert unchanged() and children_ok() and data_ok();old.previous.base.verify_native();ram=resources();timeout=600 if phase=='train' else 300
            if time.monotonic()-started+timeout>9300:raise RuntimeError('Insufficient full worker budget')
            name=f'train-{bank}-{birth}' if phase=='train' else f'frozen-{bank}-{birth}' if bank=='parent' else f'frozen-{bank}-{birth}-{seed}'
            cp=PARENT if phase=='train' or bank=='parent' else out/f'train-{bank}-{birth}/weights-last.npz'
            print('START',name,flush=True);empty=False
            try:
                with (out/(name+'.log')).open('w',encoding='utf-8') as log:
                    receipt=native.run_windows_job([sys.executable,str(root/'work/ed2_worker.py'),'--phase',phase,'--bank',bank,'--training-seed',str(birth),'--seed',str(seed),
                        '--root',str(root),'--output',str(out/name),'--checkpoint',str(cp)],timeout=timeout,stdout=log,cwd=root,env=env)
                empty=receipt['tree_empty']
            except native.WatchdogError as error:
                empty=error.cleanup_verified;raise
            write(out/(name+'-receipt.json'),receipt);result['jobs'].append(dict(name=name,phase=phase,bank=bank,training_seed=birth,seed=seed,receipt=receipt,available_ram_before_gib=ram));write(out/'result.json',result)
            assert receipt['exit_code']==0 and not receipt['timed_out'] and empty
            if phase=='train':
                result['training'][name]=training_checks(out/name,bank,birth);children[out/name/'weights-last.npz']=result['training'][name]['checkpoint_sha256']
            else:
                r=read(out/name/'report.json');assert r['ed2_worker_completed'] and not r['learning'] and r['weights_unchanged'] and r['checkpoint_sha256']==sha(cp)
                assert r['ed2_context']['training_seed']==birth and r['ed2_context']['active_delay_chunks']==0
                native.anatomy(out/name,'frozen')
                if bank!='parent':native.compare(directory(birth,'parent',seed),out/name)
            write(out/'result.json',result);print('COMPLETE',name,flush=True)
        assert len(result['jobs'])==26;result.update(status='completed_ED2',science=finish())
    except BaseException as error:
        result.update(status='failed',error_type=type(error).__name__,error=str(error));raise
    finally:
        result.update(seconds=time.monotonic()-started,source_prior_checkpoints_unchanged=unchanged(),completed_children_unchanged=children_ok(),data_unchanged=data_ok(),process_tree_empty_verified=empty)
        if empty and lock.exists() and read(lock)['token']==token:lock.unlink()
        result['lease_released']=not lock.exists()
        if out.exists():write(out/'result.json',result)
        print('RESULT',result['status'],result['seconds'],flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--private-pins',type=lambda x:root/x,required=True);ap.add_argument('--prepare',action='store_true');a=ap.parse_args();p=prospective(a.private_pins)
    if a.prepare:
        write(root/'outputs/eligibility-replication-ed2-20261107-protocol.json',p)
        write(root/'outputs/eligibility-replication-ed2-public-protocol-20261107.json',{k:v for k,v in p.items() if k not in ('source_hashes','prior_hashes','artifact_hashes')})
        print('ED2 signed inputs/prospective plan verified; private input metadata omitted; no simulation/lease')
    else:execute(p)


if __name__=='__main__':main()
