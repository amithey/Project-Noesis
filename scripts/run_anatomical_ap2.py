"""Fixed AP2 replication: paired disabled controls and two independent candidate births."""
import argparse,json,os,sys,time,uuid,zipfile
from pathlib import Path
import numpy as np
import psutil
import run_anatomical_ap1b as base
from ap1_feedback import corrected_reward,validate_ap1_training
from ap2_contract import DEV,replication_summary

root=base.root;out=root/'outputs/anatomical-feedback-ap2-20261107'
SR=root/'outputs/training-seed-sr1-20261106';S2=root/'outputs/standing-reacquisition-s2-20261106'
X1=root/'outputs/exploration-transfer-x1-20261106';B=root/'outputs/anatomical-feedback-ap1b-20261106'
C=root/'outputs/anatomical-feedback-ap1c-20261106';H2=root/'outputs/body-telemetry-h2b-20261106'
read,write,sha,arrays=base.read,base.write,base.sha,base.arrays
TRAIN={1004:S2/'train-legacy-1004',1006:SR/'train-1006'}
CONTROL_SHA={1004:'07aa4f8198c20dae779d1a836fffb4bf6ddef76f3d62c0af55a050a2b2293e43',1006:'4da00819274782cb197a00237efef052785c6b1421f7beb6ae62b9c5c937c880'}
PARENT=base.PARENT


def parent_dev(seed):return H2/'parent-10007' if seed==10007 else B/f'frozen-parent-{seed}'


def reference(birth,bank,seed):
    if bank=='parent':
        if seed in DEV:return parent_dev(seed)
        return X1/'probe-parent-legacy-off' if birth==1004 else SR/'parent-own-1006'
    if birth==1004:return X1/'probe-child-legacy-off' if seed==1004 else S2/f'child-legacy-{seed}'
    return SR/f'child-1006-{seed}'


def verify_native():
    p=read(root/'outputs/thermo-pilot-tp2-20261106-source.json')
    for name,folder in (('pilot',base.pilot),('core',base.core)):
        assert len(list(folder.rglob('*.py')))==len(p[name])
        for f in p[name]:assert sha(folder/f['path'])==f['sha256']


def prospective():
    previous=read(C/'protocol.json');source=dict(previous['source_hashes']);prior=dict(previous['prior_hashes'])
    for group in (source,prior,previous['artifact_hashes']):
        for n,h in group.items():assert sha(root/n)==h
    verify_native();archives={}
    def signed(artifact,expected,paths,outputs_relative=False):
        receipt=read(root/'outputs'/f'{artifact}-artifact.json');archive=root/'outputs'/receipt['path']
        assert sha(archive)==receipt['sha256']==expected;archives[archive.relative_to(root).as_posix()]=expected
        with zipfile.ZipFile(archive) as z:
            manifest=json.loads(z.read('manifest.json'))
            for p in paths:
                key=p.relative_to(root/'outputs' if outputs_relative else root).as_posix()
                h=manifest[key]['sha256'];assert sha(p)==h;prior[p.relative_to(root).as_posix()]=h
    basic=('report.json','observations.npz','adhesion-observations.npz')
    train_names=('report.json','weights-last.npz','training-observations.npz','adhesion-observations.npz','exploration-log.npz')
    signed('standing-reacquisition-s2-20261106','e9d3eb27a7d29f91615f61c4bad165d8da4a89e661dc07414fa75b8b2f51f905',
        [TRAIN[1004]/n for n in train_names]+[reference(1004,'control',s)/n for s in DEV for n in basic],True)
    signed('training-seed-sr1-20261106','e23d61e932062a5172391232b2d6e2513b77c6cd011c494c6a50168614352941',
        [TRAIN[1006]/n for n in train_names]+[reference(1006,b,s)/n for b in ('parent','control') for s in (1006,*DEV) if b=='control' or s==1006 for n in basic])
    signed('exploration-transfer-x1-20261106','38231241807c56c623d4b11dd7e9a6038ed37b503b843f0c760eabf47a84f084',
        [reference(1004,b,1004)/n for b in ('parent','control') for n in basic],True)
    telemetry=('report.json','observations.npz','adhesion-observations.npz','body-telemetry.npz','ground-contacts.npz','telemetry-model.json')
    signed('body-telemetry-h2b-20261106','f660b673da8d431873cd829a364d423633b01fe395ce971ecb6f3190984e3257',
        [H2/'child1004-10006'/n for n in telemetry])
    selected=[C/'result.json',root/'outputs/anatomical-feedback-ap1c-20261106-summary.json']
    for seed in DEV:
        folder=parent_dev(seed);names=telemetry if seed==10007 else (*basic,'anatomy-observations.npz','anatomy-model.json')
        selected.extend(folder/n for n in names)
    signed('anatomical-feedback-ap1c-20261106','2f60d305c87dcf08640476fbe541b6aaf3eb66aa69cfd563553cdcd52a676540',selected)
    old=read(C/'result.json');assert old['status']=='completed_AP1C' and old['complete_scientific_bank'] and not old['anatomical_dev_each_vs_both']
    assert not old['retention']['parent']['passed'] and not old['retention']['control']['passed']
    for birth in (1004,1006):assert sha(TRAIN[birth]/'weights-last.npz')==CONTROL_SHA[birth]
    for n in ('ap2_worker.py','ap2_contract.py','run_anatomical_ap2.py','test_ap2_contract.py'):
        source['work/'+n]=sha(root/'work'/n)
    return dict(experiment='AP2 fixed-method cross-birth replication',date='2026-11-07',source_hashes=source,prior_hashes=prior,artifact_hashes=archives,
        parent_sha256=sha(PARENT),native_control_sha256=CONTROL_SHA,new_training_seeds=[1004,1006],original_training_seed=1005,
        development_seeds=list(DEV),own_seed_descriptive=True,formula='clip(native_v7-.5*q_non_tarsal,-1,1)',
        measurement_revision='C1-native-trochanterfemur-proximal',training_seconds=25,learning_updates=4990,frozen_seconds=5,
        learning_rate=.2,eligibility_ms=100,original_reference_bound=.5,exploration_mv_ms=1.5,
        aggregate_seconds=5400,training_worker_seconds=600,frozen_worker_seconds=300,maximum_new_jobs=25,
        new_births=4,new_reference_probes=11,new_candidate_probes=10,reused_reference_cells=5,
        sequence='For birth1004then1006: disabled25s exact old replay; missing parent/control reference bank; sameparent candidate25s; five frozen candidate cells',
        primary='Original strict standing/slip/proxy retention versus both references AND increased tarsal-only/reduced mean non-tarsal fraction at EACH shareddev seed; all3births reported including failed1005',
        stop='technical/hash/context/horizon/mass/force/parity/resource/timeout/remaining-full-worker-budget failure; no automatic retry/retune or partial scientific interpretation',
        no_best_checkpoint_selection=True,no_weight_pooling=True,model_promoted=False,walking_proven=False,
        shared_parent_dev_cells_are_not_independent_replications=True,reserved_final_seeds_unused=[10,11,200,201,202])


def entry(directory,a):
    report,summary=base.summarize(directory)
    return dict(summary=summary,anatomical=base.contact_summary(a),directory=directory.relative_to(root).as_posix(),
        report_sha256=sha(directory/'report.json'),observations_sha256=sha(directory/'observations.npz'))


def execute(p):
    me=psutil.Process();own={me.pid,*[x.pid for x in me.parents()]}
    def resources():
        if any(x.pid not in own and (x.info['name'] or '').lower() in ('python.exe','pythonw.exe') for x in psutil.process_iter(['name'])):raise RuntimeError('Competing Python')
        ram=psutil.virtual_memory().available/2**30
        if ram<2:raise RuntimeError('RAM below2GiB')
        return ram
    children={};data=read(root/'outputs/noesis-required-data-sha256.json')['files']
    def unchanged():return all(sha(root/n)==h for group in ('source_hashes','prior_hashes') for n,h in p[group].items()) and all(sha(root/n)==h for n,h in children.items())
    def data_ok():return all(sha(root/'work/NoesisData'/f['path'])==f['sha256'] for f in data)
    resources();assert unchanged() and data_ok() and not out.exists()
    lock=root/'work/Project-Noesis/gpu-pilot/outputs/codex-training-host.lock';token=uuid.uuid4().hex
    with os.fdopen(os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY),'w') as f:json.dump(dict(owner='codex',pid=os.getpid(),token=token,deadline_unix=time.time()+5400),f)
    started=time.monotonic();empty=True;old=read(C/'result.json')
    banks={birth:{name:{} for name in ('parent','control','candidate')} for birth in (1004,1005,1006)}
    for name in banks[1005]:
        for seed in (1005,*DEV):banks[1005][name][seed]=old['cells'][f'{name}-{seed}']
    result=dict(status='running',jobs=[],training={},births=banks,replication=None,model_promoted=False,walking_proven=False)
    env=dict(os.environ,PYTHONUTF8='1',MPLCONFIGDIR=str(root/'work/matplotlib-cache'))
    def launch(phase,bank,birth,seed,cp,strength):
        nonlocal empty
        assert unchanged() and data_ok();verify_native();ram=resources();budget=600 if phase=='train' else 300
        if time.monotonic()-started>5400-budget:raise RuntimeError('Insufficient full worker budget')
        name=f'train-{bank}-{birth}' if phase=='train' else f'frozen-{bank}-{birth}-{seed}'
        directory=out/name;print('START',name,flush=True);empty=False
        try:
            with (out/(name+'.log')).open('w',encoding='utf-8') as log:
                receipt=base.run_windows_job([sys.executable,str(root/'work/ap2_worker.py'),'--phase',phase,'--bank',bank,'--training-seed',str(birth),
                    '--strength',str(strength),'--seed',str(seed),'--root',str(root),'--checkpoint',str(cp),'--output',str(directory)],timeout=budget,stdout=log,cwd=root,env=env)
            empty=receipt['tree_empty']
        except base.WatchdogError as error:
            empty=error.cleanup_verified;raise
        write(out/(name+'-receipt.json'),receipt);result['jobs'].append(dict(name=name,phase=phase,bank=bank,training_seed=birth,seed=seed,receipt=receipt,available_ram_before_gib=ram));write(out/'result.json',result)
        assert receipt['exit_code']==0 and not receipt['timed_out'] and empty
        report=read(directory/'report.json');ctx=report['ap1_context']
        assert report['ap1_worker_completed'] and ctx['experiment']=='AP2' and ctx['training_seed']==birth and ctx['seed']==seed
        assert ctx['effective_training_penalty']==strength and ctx['loaded_checkpoint_sha256']==sha(cp)
        assert report['adhesion_probe_mode']=='legacy' and report['s2_phase']==phase
        assert (report['seed']==seed if phase=='train' else report['seeds']==[seed])
        assert report['import_checkpoint_sha256' if phase=='train' else 'checkpoint_sha256']==sha(cp)
        a=base.anatomy(directory,phase);pads=arrays(directory/'adhesion-observations.npz');count=5000 if phase=='train' else 1000
        assert pads['desired'].shape==(count,42) and pads['pads'].shape==pads['Femur_delta'].shape==(count,6)
        np.testing.assert_array_equal(pads['pads'],(pads['Femur_delta']<=.08).astype(float))
        if phase=='train':
            validate_ap1_training(report,strength);r=arrays(directory/'reward-accounting.npz');obs=arrays(directory/'training-observations.npz')
            assert all(len(v)==5000 and np.isfinite(v).all() for v in r.values()) and all(len(v)==5000 and np.isfinite(v).all() for v in obs.values())
            np.testing.assert_array_equal(r['control_chunk_index'],np.arange(5000));np.testing.assert_array_equal(r['time'],a['time'][1:])
            np.testing.assert_array_equal(r['non_tarsal_fraction'],a['non_tarsal_fraction'][1:]);np.testing.assert_array_equal(r['effective_penalty'],r['native_reward']-r['applied_reward'])
            np.testing.assert_array_equal(r['applied_reward'],[corrected_reward(n,q,strength) for n,q in zip(r['native_reward'],r['non_tarsal_fraction'])])
            np.testing.assert_array_equal(obs['reward'],r['applied_reward']);np.testing.assert_array_equal(obs['learning_active'],np.arange(5000)>=10)
            newcp=directory/'weights-last.npz';assert sha(newcp)==report['saved_checkpoint_sha256']
            audit=base.compare_checkpoints(PARENT,newcp);assert audit['compatible'] and audit['parameters']['match'] and audit['rule']['match']
            original,new=arrays(PARENT),arrays(newcp)
            for k in ('version','normalize_reward_error','bias_correct_reward_error'):np.testing.assert_array_equal(original[k],new[k])
            sidecar=read(newcp.with_suffix('.model.json'))
            for k,v in report.items():assert sidecar[k]==v
            log=arrays(directory/'exploration-log.npz');np.testing.assert_array_equal(log['control_chunk_index'],np.arange(10,5000))
            assert log['drive'].shape[0]==log['credit'].shape[0]==4990
            children[newcp.relative_to(root).as_posix()]=sha(newcp)
            result['training'][f'{bank}-{birth}']=dict(checkpoint_sha256=sha(newcp),checkpoint_audit=audit,actual_objective=report['reward_objective'],
                mean_native_reward=float(r['native_reward'].mean()),mean_applied_reward=float(r['applied_reward'].mean()),mean_effective_penalty=float(r['effective_penalty'].mean()))
        else:
            assert report['status']=='completed_transport_pilot' and report['weights_unchanged'] and not report['learning'] and report['motor_exploration']==0 and report['seeds']==[seed]
        print('COMPLETE',name,flush=True);return directory,a
    def add(bank,birth,seed,directory,a,reused=False):
        banks[birth][bank][seed]=entry(directory,a)|{'reused':reused};write(out/'result.json',result)
    def paired_reference(bank,birth,seed,cp):
        directory,a=launch('frozen',bank,birth,seed,cp,0.);previous=reference(birth,bank,seed)
        for n in ('observations.npz','adhesion-observations.npz'):base.exact(directory/n,previous/n)
        base.compare(previous,directory);assert read(directory/'report.json')['spikes_per_world']==read(previous/'report.json')['spikes_per_world']
        add(bank,birth,seed,directory,a)
    try:
        out.mkdir(exist_ok=False);write(out/'protocol.json',p)
        for birth in (1004,1006):
            for seed in DEV:
                directory=parent_dev(seed);a=base.reused('parent-10007') if seed==10007 else base.anatomy(directory,'frozen')
                add('parent',birth,seed,directory,a,True)
            control,_=launch('train','control',birth,birth,PARENT,0.)
            assert sha(control/'weights-last.npz')==CONTROL_SHA[birth]
            for n in ('weights-last.npz','training-observations.npz','adhesion-observations.npz','exploration-log.npz'):base.exact(control/n,TRAIN[birth]/n)
            result['training'][f'control-{birth}']['exact_old_native_replay']=True;write(out/'result.json',result)
            paired_reference('parent',birth,birth,PARENT)
            for seed in (birth,*DEV):
                if birth==1004 and seed==10006:
                    directory=H2/'child1004-10006';previous=reference(birth,'control',seed)
                    for n in ('observations.npz','adhesion-observations.npz'):base.exact(directory/n,previous/n)
                    base.compare(previous,directory);add('control',birth,seed,directory,base.reused('child1004-10006'),True)
                else:paired_reference('control',birth,seed,TRAIN[birth]/'weights-last.npz')
            candidate,_=launch('train','candidate',birth,birth,PARENT,.5)
            base.exact(control/'exploration-log.npz',candidate/'exploration-log.npz')
            for n in ('training-observations.npz','adhesion-observations.npz'):
                a,b=arrays(control/n),arrays(candidate/n);assert set(a)==set(b)
                for k in a:
                    if k!='reward':np.testing.assert_array_equal(a[k][:11],b[k][:11])
            result['training'][f'candidate-{birth}']['current_credit_initial_body_exact_control']=True;write(out/'result.json',result)
            cp=candidate/'weights-last.npz'
            for seed in (birth,*DEV):
                directory,a=launch('frozen','candidate',birth,seed,cp,0.)
                for bank in ('parent','control'):base.compare(root/banks[birth][bank][seed]['directory'],directory)
                add('candidate',birth,seed,directory,a)
        result['replication']=replication_summary(banks)
        original=next(r for r in result['replication']['births'] if r['training_seed']==1005)
        assert original['retention']==old['retention'] and original['anatomical']['passed']==old['anatomical_dev_each_vs_both']
        assert len(result['jobs'])==25
        result.update(status='completed_AP2',all_three_births_including_original_failure_reported=True)
    except BaseException as error:
        result.update(status='failed',error_type=type(error).__name__,error=str(error));raise
    finally:
        result.update(seconds=time.monotonic()-started,source_prior_children_unchanged=unchanged(),data_unchanged=data_ok(),process_tree_empty_verified=empty)
        if empty and lock.exists() and read(lock)['token']==token:lock.unlink()
        result['lease_released']=not lock.exists()
        if out.exists():write(out/'result.json',result)
        print('RESULT',result['status'],result['seconds'],flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');args=ap.parse_args();p=prospective()
    if args.prepare:
        write(root/'outputs/anatomical-feedback-ap2-20261107-protocol.json',p)
        public={k:v for k,v in p.items() if k not in ('source_hashes','prior_hashes','artifact_hashes')}
        public['publication_scope']='Compact scientific protocol; extensive local integrity manifest retained locally'
        write(root/'outputs/anatomical-feedback-ap2-public-protocol-20261107.json',public);print('AP2 prepared, signatures verified, no simulation/lease')
    else:execute(p)


if __name__=='__main__':main()
