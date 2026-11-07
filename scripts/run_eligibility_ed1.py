"""ED1 fixed timing hypothesis: exact native replay, independent birth, frozen tests."""
import argparse,json,os,sys,time,uuid,zipfile
import numpy as np
import psutil
import run_physical_standing_ps1 as previous
from ed1_worker import validate_training
from ed1_contract import digest
from sc1_signal import normalized_signal
from s2_contract import standing_retention
from ap2_contract import DEV,anatomical_rule

root=previous.root;out=root/'outputs/eligibility-delay-ed1-20261107'
read,write,sha,arrays=previous.read,previous.write,previous.sha,previous.arrays
PARENT=previous.PARENT;SEEDS=(1005,*DEV)


def reference(bank,seed):
    if bank=='parent':return previous.prior.previous.cell('parent',1005) if seed==1005 else previous.reference('parent',seed)
    if bank=='previous_PS1':return previous.out/f'frozen-candidate-{seed}'
    return previous.reference(bank,seed)


def prospective():
    old=read(previous.out/'protocol.json');source=dict(old['source_hashes']);prior=dict(old['prior_hashes']);artifacts=dict(old['artifact_hashes'])
    for group in (source,prior,artifacts):
        for n,h in group.items():assert sha(root/n)==h
    previous.base.verify_native()
    def signed(label,expected,paths):
        receipt=read(root/'outputs'/f'{label}-artifact.json');archive=root/'outputs'/receipt['path'];assert sha(archive)==receipt['sha256']==expected
        artifacts[archive.relative_to(root).as_posix()]=expected
        with zipfile.ZipFile(archive) as z:m=json.loads(z.read('manifest.json'))
        for p in paths:
            n=p.relative_to(root).as_posix();assert sha(p)==m[n]['sha256'];prior[n]=sha(p)
    names=('report.json','observations.npz','adhesion-observations.npz','anatomy-observations.npz','anatomy-model.json')
    selected=[previous.out/'result.json',root/'outputs/physical-standing-ps1-20261107-summary.json']
    selected += [previous.out/'train-candidate'/n for n in ('report.json','weights-last.npz','training-observations.npz','physical-reward-accounting.npz','adhesion-observations.npz','anatomy-observations.npz','exploration-log.npz')]
    selected += [reference('previous_PS1',s)/n for s in DEV for n in names]
    signed('physical-standing-ps1-20261107','317b45fb3bcca26e65f1e56025607f630b38723e9ad6f9e1d390dd6de079cbd4',selected)
    signed('anatomical-feedback-ap1c-20261106','2f60d305c87dcf08640476fbe541b6aaf3eb66aa69cfd563553cdcd52a676540',[reference('parent',1005)/n for n in names])
    for n in ('ed1_eligibility.py','test_ed1_eligibility.py','ed1_contract.py','test_ed1_contract.py','ed1_worker.py','run_eligibility_ed1.py','sc1_signal.py','audit_physical_standing_ps1.py'):
        source['work/'+n]=sha(root/'work'/n)
    assert sha(PARENT)=='9fd7d41177bf2c745809401c3b142a832661d1a18d0b117b30cb02aea63fea3c'
    return dict(experiment='ED1 fixed full-eligibility delay pilot',date='2026-11-07',source_hashes=source,prior_hashes=prior,artifact_hashes=artifacts,
        common_parent_sha256=sha(PARENT),replay_seed=1004,replay_checkpoint_sha256=sha(previous.out/'train-candidate/weights-last.npz'),training_seed=1005,development_seeds=list(DEV),own_seed_descriptive=True,
        factor='Full eligibility trace used for weight step: native current trace versus trace10updates(50ms)old; same PS1 scalar objective, brain/body/senses/RNG/exploration/normalizer/original bounds',
        timing_hypothesis='Unvalidated fixed engineering candidate; observational correlations did not establish optimum, causality or efficacy',
        learning_rate=.2,eligibility_ms=100,original_reference_bound=.5,exploration_mv_ms=1.5,objective='physical-standing-ps1-v1',
        delay_chunks=dict(replay=0,control=0,candidate=10),control_ms=5,delayed_initial_zero_updates=10,checkpoint_semantics='Weights-only transfer; delayed queue not checkpointed; exact trace resume prohibited',
        seconds_train=25,learning_attempts_per_birth=4990,seconds_frozen=5,maximum_new_jobs=13,aggregate_seconds=4800,training_worker_seconds=600,frozen_worker_seconds=300,
        sequence=['zero-delay exact PS1seed1004 replay','zero-delay PS1seed1005 control','50ms-delay PS1seed1005 candidate','five frozen control probes own1005+fourdev','five frozen candidate probes own1005+fourdev'],
        reused_frozen_reference_positions=17,final_reserved_seeds_unused=[10,11,200,201,202],
        proof='Every live weight step independently recomputed; independent full-trace queue/hash/time chain; raw generated innovation is distinguished from actual used eligibility. Offline reward-moment and provenance audit; not full neural-spike reconstruction.',
        primary='At EACH4dev, mean PS1quality higher and mean COMspeed/angularspeed/non-tarsal fraction lower versus BOTH parent and concurrent zero-delay control; no own-seed or pooled rescue',
        retention='Unchanged strict standing/slip/heading/net/cycles/tripod versus parent, concurrent control, oldv7 and AP1; priorPS1 comparison additionally descriptive. Existing anatomical parent/control rule retained.',
        stop='hash/context/mass/force/accounting/queue/weight-step/checkpoint/sidecar/replay/prefix/horizon/resource/timeout/full-worker-remaining-budget failure; no automatic retry/retuning/partial inference',
        limits='One independent training seed for a timing hypothesis. Software fixtures or a positive pilot do not validate stable standing, retained ability, balance, walking or general efficacy.',
        model_promoted=False,walking_proven=False,physical_balance_validated=False)


def eligibility_checks(directory,delay):
    e=arrays(directory/'eligibility-audit.npz');o=arrays(directory/'training-observations.npz');cp=arrays(directory/'weights-last.npz')
    assert all(len(v)==4990 for v in e.values()) and all(np.isfinite(v).all() for v in e.values() if v.dtype.kind!='U')
    np.testing.assert_array_equal(e['update_index'],np.arange(4990));np.testing.assert_array_equal(e['control_chunk_index'],np.arange(10,5000))
    np.testing.assert_array_equal(e['used_source_update_index'],np.maximum(np.arange(4990)-delay,-1))
    assert np.all(e['independent_weight_step_exact']) and np.all(e['delay_chunks']==delay)
    if delay:
        np.testing.assert_array_equal(e['used_eligibility_sha256'][delay:],e['current_eligibility_sha256'][:-delay])
        assert np.all(e['used_eligibility_L2'][:delay]==0)
        np.testing.assert_array_equal(e['used_eligibility_sha256'][:delay],np.full(delay,digest(np.zeros(len(cp['weights'])))))
    else:np.testing.assert_array_equal(e['used_eligibility_sha256'],e['current_eligibility_sha256'])
    moments=normalized_signal(o['reward'][10:])
    for a,b in (('signal','learning_signal'),('raw_error','prediction_error'),('baseline','reward_baseline'),('variance','reward_variance'),('mass','reward_normalization_mass')):np.testing.assert_array_equal(moments[a],e[b])
    for a,b in (('baseline','reward_baseline'),('variance','reward_variance'),('mass','reward_normalization_mass')):assert moments[a][-1]==float(cp[b])
    return dict(rows=4990,delay_chunks=delay,independent_online_weight_steps_exact=True,queue_hash_chain_exact=True,all_native_reward_moments_exact=True)


def training_checks(directory,bank):
    r=read(directory/'report.json');validate_training(r,bank);assert r['ed1_context']==read(directory.with_name(directory.name+'-ed1-context.json'))
    assert r['import_checkpoint_sha256']==sha(PARENT);cp=directory/'weights-last.npz';assert sha(cp)==r['saved_checkpoint_sha256']
    rule=previous.base.base.compare_checkpoints(PARENT,cp);assert rule['compatible'] and rule['rule']['match'] and rule['parameters']['match']
    side=read(cp.with_suffix('.model.json'))
    for k,v in r.items():assert side[k]==v
    a=previous.base.base.anatomy(directory,'train');log=arrays(directory/'physical-reward-accounting.npz');o=arrays(directory/'training-observations.npz')
    assert all(len(v)==5000 and np.isfinite(v).all() for v in log.values());np.testing.assert_array_equal(log['control_chunk_index'],np.arange(5000));np.testing.assert_array_equal(log['time'],a['time'][1:])
    np.testing.assert_array_equal(log['applied_reward'],o['reward']);np.testing.assert_array_equal(log['applied_reward'],log['PS1_reward'])
    m=read(directory/'anatomy-model.json');p=previous.physical_score(a['region_vertical_reaction'][1:],a['com_velocity'][1:],o['omega'],o['up_body'][:,2],o['foot_load'],m['total_mass']*abs(m['gravity'][2]))
    for k,v in p.items():np.testing.assert_allclose(log['PS1_'+k],v,rtol=1e-14,atol=1e-15)
    checked=eligibility_checks(directory,10 if bank=='candidate' else 0)
    if bank=='replay':
        old=previous.out/'train-candidate';assert sha(cp)==sha(old/'weights-last.npz')
        for n in ('training-observations.npz','adhesion-observations.npz','anatomy-observations.npz','exploration-log.npz','physical-reward-accounting.npz'):previous.base.base.exact(directory/n,old/n)
    elif bank=='candidate':
        previous.base.base.exact(directory/'exploration-log.npz',out/'train-control/exploration-log.npz')
        for n in ('training-observations.npz','adhesion-observations.npz'):
            c,d=arrays(out/'train-control'/n),arrays(directory/n);assert set(c)==set(d)
            for k in c:np.testing.assert_array_equal(c[k][:11],d[k][:11])
    return dict(checkpoint_sha256=sha(cp),objective=r['reward_objective'],seed=r['seed'],full_horizon=True,original_weight_invariants=True,eligibility=checked,exact_prior_replay=bank=='replay')


def finish():
    banks={}
    for bank in ('parent','control','candidate','v7','AP1','previous_PS1'):
        banks[bank]={}
        for seed in (SEEDS if bank in ('parent','control','candidate') else DEV):
            d=out/f'frozen-{bank}-{seed}' if bank in ('control','candidate') else reference(bank,seed)
            _,summary=previous.base.base.summarize(d);banks[bank][seed]=dict(summary=summary,physical=previous.physical_summary(d),anatomical=previous.prior.previous.contact_data(d))
    child={s:banks['candidate'][s]['summary'] for s in DEV}
    retention={bank:standing_retention({s:banks[bank][s]['summary'] for s in DEV},child,DEV) for bank in ('parent','control','v7','AP1','previous_PS1')}
    anatomy=anatomical_rule(*({s:banks[b][s]['anatomical'] for s in DEV} for b in ('parent','control','candidate')))
    paired={};passed=True
    for seed in DEV:
        paired[str(seed)]={};c=banks['candidate'][seed]['physical']
        for b in ('parent','control'):
            r=banks[b][seed]['physical'];d={k:c[k]-r[k] for k in ('mean_PS1_quality','mean_COM_speed_mm_s','mean_angular_speed_rad_s','mean_non_tarsal_fraction')}
            ok=d['mean_PS1_quality']>0 and all(d[k]<0 for k in d if k!='mean_PS1_quality');passed=passed and ok
            paired[str(seed)][b]=dict(changes=d,all_directions_improve=bool(ok))
    return dict(banks=banks,retention=retention,anatomical=anatomy,paired_physical=paired,all_four_development_physical_directions_improve=bool(passed),
        combined_exploratory_pilot_pass=bool(passed and anatomy['passed'] and all(retention[b]['passed'] for b in ('parent','control','v7','AP1'))),
        physical_balance_validated=False,model_promoted=False,walking_proven=False)


def execute(p):
    me=psutil.Process();own={me.pid,*[x.pid for x in me.parents()]};data=read(root/'outputs/noesis-required-data-sha256.json')['files']
    def resources():
        if any(x.pid not in own and (x.info['name'] or '').lower() in ('python.exe','pythonw.exe') for x in psutil.process_iter(['name'])):raise RuntimeError('Competing Python')
        ram=psutil.virtual_memory().available/2**30
        if ram<2 or psutil.disk_usage(root).free<2*2**30:raise RuntimeError('RAM/disk below2GiB')
        return ram
    def unchanged():return all(sha(root/n)==h for group in ('source_hashes','prior_hashes','artifact_hashes') for n,h in p[group].items())
    def data_ok():return all(sha(root/'work/NoesisData'/f['path'])==f['sha256'] for f in data)
    resources();assert unchanged() and data_ok() and not out.exists()
    lock=root/'work/Project-Noesis/gpu-pilot/outputs/codex-training-host.lock';token=uuid.uuid4().hex
    with os.fdopen(os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY),'w') as f:json.dump(dict(owner='codex',pid=os.getpid(),token=token,deadline_unix=time.time()+4800),f)
    started=time.monotonic();empty=True;result=dict(status='running',jobs=[],training={},model_promoted=False,walking_proven=False)
    jobs=[('train','replay',1004),('train','control',1005),('train','candidate',1005)]+[('frozen',b,s) for b in ('control','candidate') for s in SEEDS]
    env=dict(os.environ,PYTHONUTF8='1',MPLCONFIGDIR=str(root/'work/matplotlib-cache'))
    try:
        out.mkdir(exist_ok=False);write(out/'protocol.json',p)
        for phase,bank,seed in jobs:
            assert unchanged() and data_ok();previous.base.verify_native();ram=resources();timeout=600 if phase=='train' else 300
            if time.monotonic()-started+timeout>4800:raise RuntimeError('Insufficient full worker budget')
            name=f'train-{bank}' if phase=='train' else f'frozen-{bank}-{seed}';cp=PARENT if phase=='train' else out/f'train-{bank}/weights-last.npz'
            print('START',name,flush=True);empty=False
            try:
                with (out/(name+'.log')).open('w',encoding='utf-8') as log:
                    receipt=previous.base.base.run_windows_job([sys.executable,str(root/'work/ed1_worker.py'),'--phase',phase,'--bank',bank,'--seed',str(seed),'--root',str(root),'--output',str(out/name),'--checkpoint',str(cp)],timeout=timeout,stdout=log,cwd=root,env=env)
                empty=receipt['tree_empty']
            except previous.base.base.WatchdogError as error:
                empty=error.cleanup_verified;raise
            write(out/(name+'-receipt.json'),receipt);result['jobs'].append(dict(name=name,phase=phase,bank=bank,seed=seed,receipt=receipt,available_ram_before_gib=ram));write(out/'result.json',result)
            assert receipt['exit_code']==0 and not receipt['timed_out'] and empty
            if phase=='train':result['training'][bank]=training_checks(out/name,bank)
            else:
                report=read(out/name/'report.json');assert report['ed1_worker_completed'] and not report['learning'] and report['weights_unchanged'] and report['checkpoint_sha256']==sha(cp)
                previous.base.base.anatomy(out/name,'frozen');previous.base.base.compare(reference('parent',seed),out/name)
            write(out/'result.json',result);print('COMPLETE',name,flush=True)
        assert len(result['jobs'])==13;result.update(status='completed_ED1',science=finish())
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
        write(root/'outputs/eligibility-delay-ed1-20261107-protocol.json',p)
        write(root/'outputs/eligibility-delay-ed1-public-protocol-20261107.json',{k:v for k,v in p.items() if k not in ('source_hashes','prior_hashes','artifact_hashes')})
        print('ED1 signed inputs and prospective protocol verified; no simulation/lease')
    else:execute(p)


if __name__=='__main__':main()
