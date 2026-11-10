"""MP1 measured-policy acceptance pilot; private provenance is local input."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import uuid
import zipfile
import shutil
import numpy as np
import psutil
import run_episodic_ew1 as previous
from ew1_update import apply_delta
from mp1_select import TRAIN_SEEDS,DIRECTIONS,STEP,DIRECTION_SEED,MIN_GAIN,DEV,FRESH,fixed_directions,select_policy,matched_gradient,training_panel
PROBE_STEP=UPDATE_STEP=STEP
from mp1_contract import validate_report,episode_rewards,return_record,comparison

root=previous.root
read,write,sha,arrays=previous.read,previous.write,previous.sha,previous.arrays
native=previous.native
out=root/'outputs/measured-policy-mp1-20261110'
inputs=root/'outputs/measured-policy-mp1-inputs-20261110'
SOURCE=('mp1_select.py','test_mp1_select.py','mp1_worker.py','test_mp1_worker.py','mp1_contract.py',
        'test_mp1_contract.py','run_measured_mp1.py','audit_measured_mp1.py','test_mp1_state.py','test_mp1_orchestration.py')


def local(name):
    path=(root/name).resolve()
    if not path.is_relative_to(root):raise ValueError('Private input must stay in workspace')
    return path


def reference(p,bank,seed):
    if bank=='parent':
        return out/f'parent-{seed}' if seed in (*TRAIN_SEEDS,*FRESH) else local(p['reference_folders'][str(seed)])
    return out/f'evaluation-{bank}-{seed}'


def save_new_checkpoint(path,expected):
    if path.exists():
        current=arrays(path);assert set(current)==set(expected)
        for k in current:np.testing.assert_array_equal(current[k],expected[k])
    else:np.savez_compressed(path,**expected)


def public_plan():
    return dict(experiment='MP1 measured-policy learning acceptance',date='2026-11-10',
        common_parent='Same original experimental parent, not any EW1 child or retrospectively selected probe',
        training_seeds=list(TRAIN_SEEDS),existing_development_seeds=list(DEV),fresh_frozen_seeds=list(FRESH),
        directions=DIRECTIONS,direction_rng_seed=DIRECTION_SEED,probe_step=STEP,matched_gradient_max_step=STEP,
        minimum_worst_training_return_gain=MIN_GAIN,
        algorithm='Choose actual measured whole policy maximizing worst training-context gain; strict positive margin acceptance, canonical first-index ties, otherwise parent fallback. No mixing candidate weights.',
        controls='No-update parent, reward-reversed measured selector, and matched-scale EW1-style gradient from the SAME fresh trial data',
        objective='physical-standing-ps1-v1',warmup_samples=10,return_samples=990,seconds_per_worker=5,
        maximum_new_workers=45,worker_seconds=300,aggregate_seconds=13500,
        sequence=['Exact baseline replay','Two training-parent controls','All16fresh symmetric probe trials',
            'Save/reload candidate, negative, matched-gradient and optimizer state','Two fresh frozen-parent controls AFTER selection',
            'All24candidate/negative/gradient evaluations at training+existing4+fresh2 contexts'],
        existing_rule='Original strict physical directions, strict standing/slip/locomotion retention and anatomical rule versus parent/negative/gradient across all four existing contexts',
        fresh_rule='No regression in physical metrics/anatomical support plus unchanged strict retention across two first-use contexts; no impossible strict gain beyond already perfect support',
        comparison='Candidate training acceptance is by construction, NOT independent efficacy; parent fallback cannot pass combined pilot',
        common_setting_change='Both methods use matched1percent probes/steps in this fresh cohort; this is not an exact EW1 scientific replication',
        persistence='Selected checkpoint byte-identical to measured probe or parent; independent state retains all returns/lineage and reproduces selection and gradient before evaluation',
        stop='Source/input/data/created/import/horizon/setting/replay/resource/timeout/containment or full remaining-worker budget failure; no retry, retune, skipping or partial promotion',
        limits='One learning round, four directions, fixed engineering objective and legacy proxies; fresh contexts are not final biological/robustness qualification',
        model_promoted=False,standing_proven=False,walking_proven=False)


def prospective(private_path):
    desc=read(private_path);old=read(local(desc['source_protocol']))
    source=dict(old['source_hashes']);prior=dict(old['prior_hashes']);artifacts=dict(old['artifact_hashes'])
    for group in (source,prior,artifacts):
        for n,h in group.items():assert sha(local(n))==h
    previous.previous.base.verify_native()
    receipt_path=local(desc['artifact_receipt']);assert sha(receipt_path)==desc['receipt_sha256']
    receipt=read(receipt_path);archive=local(desc['artifact_folder'])/receipt['path']
    assert archive.resolve().is_relative_to(root)
    assert sha(archive)==receipt['sha256']==desc['artifact_sha256']
    artifacts[archive.relative_to(root).as_posix()]=sha(archive)
    with zipfile.ZipFile(archive) as z:manifest=json.loads(z.read('manifest.json'))
    folders=[local(desc['baseline_folder']),*[local(v) for v in desc['reference_folders'].values()]]
    for d in folders:
        required={'report.json','observations.npz','adhesion-observations.npz'}
        assert required.issubset({p.name for p in d.iterdir() if p.is_file()})
        for path in d.iterdir():
            if not path.is_file():continue
            n=path.relative_to(root).as_posix();assert sha(path)==manifest[n]['sha256'];prior[n]=sha(path)
    assert set(desc['reference_folders'])=={str(s) for s in DEV}
    prior[private_path.relative_to(root).as_posix()]=sha(private_path)
    prior[receipt_path.relative_to(root).as_posix()]=sha(receipt_path)
    prior[local(desc['source_protocol']).relative_to(root).as_posix()]=sha(local(desc['source_protocol']))
    parent_path=local(desc['parent_checkpoint']);assert sha(parent_path)==old['parent_sha256']
    prior[parent_path.relative_to(root).as_posix()]=sha(parent_path)
    cp=arrays(parent_path);directions=fixed_directions(len(cp['weights']));inputs.mkdir(exist_ok=True)
    for d in range(DIRECTIONS):
        for sign in (-1,1):
            path=inputs/f'probe-{d}-{sign}.npz';save_new_checkpoint(path,apply_delta(cp,sign*PROBE_STEP*directions[d]))
            prior[path.relative_to(root).as_posix()]=sha(path)
    direction_file=inputs/'directions.npz';save_new_checkpoint(direction_file,{'directions':directions})
    prior[direction_file.relative_to(root).as_posix()]=sha(direction_file)
    for name in SOURCE:source['work/'+name]=sha(root/'work'/name)
    return dict(**public_plan(),source_hashes=source,prior_hashes=prior,artifact_hashes=artifacts,
        parent_checkpoint=desc['parent_checkpoint'],parent_sha256=sha(parent_path),baseline_folder=desc['baseline_folder'],
        reference_folders=desc['reference_folders'])


def measured(directory):
    report=read(directory/'report.json');validate_report(report)
    observations=arrays(directory/'observations.npz');a=native.anatomy(directory,'frozen');model=read(directory/'anatomy-model.json')
    rewards=episode_rewards(observations,a,model)
    return report,observations,a,model,rewards


def learn(p,records):
    returns=training_panel(records);directions=arrays(inputs/'directions.npz')['directions']
    baseline=np.array([float(measured(out/f'parent-{seed}')[-1][10:].mean()) for seed in TRAIN_SEEDS])
    decisions={b:select_policy(baseline,returns,sign) for b,sign in (('candidate',1),('negative',-1))}
    gradient=matched_gradient(directions,returns);parent_path=local(p['parent_checkpoint']);parent=arrays(parent_path)
    paths={}
    for bank in ('candidate','negative','gradient'):
        path=out/(bank+'.npz')
        if bank=='gradient':
            save_new_checkpoint(path,apply_delta(parent,gradient['relative_delta']))
            source_hash=None;accepted=False
        else:
            index=decisions[bank]['policy_index']
            direction,sign_index=divmod(index,2)
            source=parent_path if index<0 else inputs/f'probe-{direction}-{-1 if sign_index==0 else 1}.npz'
            if path.exists():assert sha(path)==sha(source)
            else:shutil.copyfile(source,path)
            assert sha(path)==sha(source)
            source_hash=sha(source);accepted=decisions[bank]['accepted_for_training_update']
        paths[bank]=path
        write(path.with_suffix('.model.json'),dict(experiment='MP1',status='completed_one_experimental_learning_round',
            bank=bank,accepted_for_training_update=accepted,measured_policy_source_sha256=source_hash,
            common_parent_sha256=p['parent_sha256'],checkpoint_sha256=sha(path),training_trial_count=16,
            completed_rounds=1,weights_only=True,native_online_learning=False,
            compatibility_trace_fields_are_not_optimizer_state=True,model_promoted=False,walking_proven=False))
    write(out/'selection.json',decisions)
    state=dict(directions=directions,policy_returns=returns,baseline_returns=baseline,
        gradient_coefficients=gradient['coefficients'],gradient=gradient['gradient'],gradient_delta=gradient['relative_delta'],
        completed_rounds=np.array(1),training_seeds=np.array(TRAIN_SEEDS),parent_sha256=np.array(p['parent_sha256']),
        step=np.array(STEP),minimum_gain=np.array(MIN_GAIN),selection_json=np.array(json.dumps(decisions,sort_keys=True)),
        return_records_json=np.array(json.dumps(records,sort_keys=True)))
    state_path=out/'optimizer-state.npz';save_new_checkpoint(state_path,state);loaded=arrays(state_path)
    for bank,sign in (('candidate',1),('negative',-1)):
        replay=select_policy(loaded['baseline_returns'],loaded['policy_returns'],sign)
        assert replay==decisions[bank]
        index=replay['policy_index'];d,k=divmod(index,2)
        source=parent_path if index<0 else inputs/f'probe-{d}-{-1 if k==0 else 1}.npz'
        assert sha(paths[bank])==sha(source)
    replay=matched_gradient(loaded['directions'],loaded['policy_returns'])
    np.testing.assert_array_equal(replay['relative_delta'],loaded['gradient_delta'])
    expected=apply_delta(parent,replay['relative_delta']);actual=arrays(paths['gradient'])
    for k in expected:np.testing.assert_array_equal(expected[k],actual[k])
    return dict(completed_rounds=1,training_trial_count=16,selection=decisions,
        candidate_update_applied=decisions['candidate']['accepted_for_training_update'],reload_reproduces_weights=True,
        gradient_zero_signal=gradient['zero_signal'],checkpoint_sha256={b:sha(path) for b,path in paths.items()},
        optimizer_state_sha256=sha(state_path))


def finish(p):
    banks={}
    for bank in ('parent','candidate','negative','gradient'):
        banks[bank]={}
        for seed in (*TRAIN_SEEDS,*DEV,*FRESH):
            d=reference(p,bank,seed);_,summary=native.summarize(d)
            if bank!='parent':native.compare(reference(p,'parent',seed),d)
            physical=previous.previous.history.physical_summary(d)
            if (d/'anatomy-observations.npz').exists():
                rewards=episode_rewards(arrays(d/'observations.npz'),native.anatomy(d,'frozen'),read(d/'anatomy-model.json'))
                value=float(np.mean(rewards[10:]))
            else:
                value=2*physical['mean_PS1_quality']-1
            banks[bank][seed]=dict(summary=summary,physical=physical,
                anatomical=previous.previous.history.prior.previous.contact_data(d),**{'return':value})
    return dict(banks=banks,comparison=comparison(banks,read(out/'selection.json')['candidate']['accepted_for_training_update']))


def execute(p):
    me=psutil.Process();own={me.pid,*[x.pid for x in me.parents()]};created={}
    data=read(root/'outputs/noesis-required-data-sha256.json')['files']
    def resources():
        if any(x.pid not in own and (x.info['name'] or '').lower() in ('python.exe','pythonw.exe') for x in psutil.process_iter(['name'])):raise RuntimeError('Competing Python')
        ram=psutil.virtual_memory().available/2**30
        if ram<2 or psutil.disk_usage(str(root)).free<2*2**30:raise RuntimeError('RAM/disk below2GiB')
        return ram
    def unchanged():
        try:return all(sha(local(n))==h for group in ('source_hashes','prior_hashes','artifact_hashes') for n,h in p[group].items()) and all(sha(path)==h for path,h in created.items())
        except OSError:return False
    def data_ok():
        try:return all(sha(root/'work/NoesisData'/f['path'])==f['sha256'] for f in data)
        except OSError:return False
    resources();assert unchanged() and data_ok() and not out.exists()
    lock=root/'work/Project-Noesis/gpu-pilot/outputs/codex-training-host.lock';token=uuid.uuid4().hex
    with os.fdopen(os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY),'w') as f:
        json.dump(dict(owner='codex',pid=os.getpid(),token=token,deadline_unix=time.time()+13500),f)
    started=time.monotonic();empty=True;records=[]
    result=dict(status='running',jobs=[],model_promoted=False,standing_proven=False,walking_proven=False)
    env=dict(os.environ,PYTHONUTF8='1',MPLCONFIGDIR=str(root/'work/matplotlib-cache'))
    def worker(name,phase,bank,seed,checkpoint,direction=None,sign=None):
        nonlocal empty
        assert unchanged() and data_ok();previous.previous.base.verify_native();ram=resources()
        if time.monotonic()-started+300>13500:raise RuntimeError('Insufficient full worker budget')
        command=[sys.executable,str(root/'work/mp1_worker.py'),'--root',str(root),'--output',str(out/name),
            '--checkpoint',str(checkpoint),'--phase',phase,'--bank',bank,'--seed',str(seed)]
        if phase=='trial':command+=['--direction',str(direction),'--sign',str(sign)]
        print('START',name,flush=True);empty=False
        try:
            with (out/(name+'.log')).open('w',encoding='utf-8') as log:
                receipt=native.run_windows_job(command,timeout=300,stdout=log,cwd=root,env=env)
            empty=receipt['tree_empty']
        except native.WatchdogError as error:
            empty=error.cleanup_verified;raise
        write(out/(name+'-receipt.json'),receipt)
        result['jobs'].append(dict(name=name,phase=phase,bank=bank,seed=seed,direction=direction,sign=sign,
            receipt=receipt,available_ram_before_gib=ram));write(out/'result.json',result)
        assert receipt['exit_code']==0 and empty and not receipt['timed_out']
        report,o,a,m,rewards=measured(out/name)
        context=report['mp1_context'];assert context==read(out/(name+'-mp1-context.json'))
        assert context['phase']==phase and context['bank']==bank and context['seed']==seed
        assert context['direction']==direction and context['sign']==sign and context['checkpoint_sha256']==sha(checkpoint)
        if phase=='trial':
            native.compare(reference(p,'parent',seed),out/name)
            sources={k:sha(out/name/n) for k,n in [('report','report.json'),('observations','observations.npz'),('anatomy','anatomy-observations.npz'),('model','anatomy-model.json')]}
            record=return_record(report,o,a,m,sources);records.append(record)
            write(out/'training-returns.json',records)
        if phase=='baseline':
            ref=local(p['baseline_folder'])
            for n in ('observations.npz','adhesion-observations.npz','anatomy-observations.npz'):native.exact(out/name/n,ref/n)
            assert report['spikes_per_world']==read(ref/'report.json')['spikes_per_world']
            native.compare(ref,out/name)
        for path in (out/name).iterdir():
            if path.is_file():created[path]=sha(path)
        for suffix in ('-receipt.json','-mp1-context.json'):created[out/(name+suffix)]=sha(out/(name+suffix))
        print('COMPLETE',name,flush=True)
    try:
        out.mkdir(exist_ok=False);write(out/'protocol.json',p);parent=local(p['parent_checkpoint'])
        worker('baseline','baseline','parent',1004,parent)
        for seed in TRAIN_SEEDS:worker(f'parent-{seed}','evaluation','parent',seed,parent)
        for d in range(DIRECTIONS):
            for seed in TRAIN_SEEDS:
                for sign in (-1,1):worker(f'trial-{d}-{seed}-{sign}','trial','probe',seed,inputs/f'probe-{d}-{sign}.npz',d,sign)
        result['learning']=learn(p,records)
        for name in ('candidate.npz','candidate.model.json','negative.npz','negative.model.json','gradient.npz','gradient.model.json','optimizer-state.npz','selection.json'):
            created[out/name]=sha(out/name)
        write(out/'result.json',result)
        for seed in FRESH:worker(f'parent-{seed}','evaluation','parent',seed,parent)
        for bank in ('candidate','negative','gradient'):
            for seed in (*TRAIN_SEEDS,*DEV,*FRESH):worker(f'evaluation-{bank}-{seed}','evaluation',bank,seed,out/(bank+'.npz'))
        assert len(result['jobs'])==45 and len(records)==16
        result.update(status='completed_MP1',science=finish(p))
    except BaseException as error:
        result.update(status='failed',error_type=type(error).__name__,error=str(error));raise
    finally:
        result.update(seconds=time.monotonic()-started,source_prior_and_created_unchanged=unchanged(),data_unchanged=data_ok(),process_tree_empty_verified=empty)
        result['created_hashes']={path.relative_to(root).as_posix():h for path,h in created.items()}
        if empty and lock.exists() and read(lock)['token']==token:lock.unlink()
        result['lease_released']=not lock.exists()
        if out.exists():write(out/'result.json',result)
        print('RESULT',result['status'],result['seconds'],flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--private-inputs',type=local,required=True);ap.add_argument('--prepare',action='store_true');a=ap.parse_args()
    p=prospective(a.private_inputs)
    if a.prepare:
        write(root/'outputs/measured-policy-mp1-20261110-protocol.json',p)
        write(root/'outputs/measured-policy-mp1-public-protocol-20261110.json',public_plan())
        print('MP1 signed inputs/preparation verified; no simulation allocated or private metadata printed')
    else:execute(p)


if __name__=='__main__':main()
