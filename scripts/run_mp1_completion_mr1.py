"""Explicit MP1 completion allocation after verified host sleep, never blind resume."""
import argparse
import json
import os
import sys
import time
import uuid
import psutil
import run_measured_mp1 as old
from awake_guard import AwakeRequest
from audit_mp1_interruption import verify,jobs
from mp1_select import TRAIN_SEEDS,DEV,FRESH
from mp1_contract import comparison

root=old.root;out=root/'outputs/mp1-completion-mr1-20261113'
read,write,sha,arrays=old.read,old.write,old.sha,old.arrays
SOURCE=('awake_guard.py','test_awake_guard.py','audit_mp1_interruption.py','run_mp1_completion_mr1.py',
        'audit_mp1_completion_mr1.py','test_mr1_completion.py')


def public_plan():
    return dict(experiment='MR1 explicit completion of technically interrupted MP1',date='2026-11-13',
        original_status='failed_incomplete',reused_completed_workers=29,original_failed_attempts_preserved=1,
        additional_workers=16,worker_seconds=300,aggregate_seconds=4800,seconds_per_worker=5,
        sequence='First missing negative1010 exact parent replay; remaining seven negative and all eight gradient cells in original order',
        unchanged='Original frozen worker, models, source, seeds, objective and acceptance/retention criteria; no training/selection repeated',
        preservation='Original failed result, timed-out attempt and29complete cells never overwritten; two allocations and46attempts remain explicit',
        power='Temporary thread-scoped system-idle-sleep request only during bounded allocation; restore on exit; explicit user sleep is still possible',
        stop='Request/resource/hash/source/data/import/full-horizon/matched-settings/exact-first-replay/timeout/containment/remaining-budget failure; no automatic retry',
        no_exact_episode_resume=True,model_promoted=False,standing_proven=False,walking_proven=False)


def prospective():
    prefix=verify()
    sources={'work/'+n:sha(root/'work'/n) for n in SOURCE}
    return dict(**public_plan(),original_hashes=prefix['original_hashes'],source_hashes=sources,
        original_protocol=old.read(old.out/'protocol.json'))


def reference(p,bank,seed):
    return out/f'evaluation-{bank}-{seed}' if bank in ('negative','gradient') else old.reference(p,bank,seed)


def finish(p):
    banks={}
    for bank in ('parent','candidate','negative','gradient'):
        banks[bank]={}
        for seed in (*TRAIN_SEEDS,*DEV,*FRESH):
            d=reference(p,bank,seed);_,summary=old.native.summarize(d)
            if bank!='parent':old.native.compare(reference(p,'parent',seed),d)
            physical=old.previous.previous.history.physical_summary(d)
            if (d/'anatomy-observations.npz').exists():
                rewards=old.episode_rewards(arrays(d/'observations.npz'),old.native.anatomy(d,'frozen'),read(d/'anatomy-model.json'))
                value=float(rewards[10:].mean())
            else:value=2*physical['mean_PS1_quality']-1
            banks[bank][seed]=dict(summary=summary,physical=physical,
                anatomical=old.previous.previous.history.prior.previous.contact_data(d),**{'return':value})
    accepted=read(old.out/'selection.json')['candidate']['accepted_for_training_update']
    return dict(banks=banks,comparison=comparison(banks,accepted))


def execute(protocol):
    p=protocol['original_protocol'];created={};me=psutil.Process();own={me.pid,*[x.pid for x in me.parents()]}
    data=read(root/'outputs/noesis-required-data-sha256.json')['files']
    def resources():
        if any(x.pid not in own and (x.info['name'] or '').lower() in ('python.exe','pythonw.exe') for x in psutil.process_iter(['name'])):raise RuntimeError('Competing Python')
        ram=psutil.virtual_memory().available/2**30
        if ram<2 or psutil.disk_usage(str(root)).free<2*2**30:raise RuntimeError('RAM/disk below2GiB')
        return ram
    def intact():
        try:return all(sha(root/n)==h for group in ('original_hashes','source_hashes') for n,h in protocol[group].items()) and all(sha(path)==h for path,h in created.items())
        except OSError:return False
    def data_ok():
        try:return all(sha(root/'work/NoesisData'/e['path'])==e['sha256'] for e in data)
        except OSError:return False
    resources();assert intact() and data_ok() and not out.exists()
    awake=AwakeRequest()
    lock=root/'work/Project-Noesis/gpu-pilot/outputs/codex-training-host.lock';token=uuid.uuid4().hex
    with os.fdopen(os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY),'w') as f:
        json.dump(dict(owner='codex',pid=os.getpid(),token=token,deadline_unix=time.time()+4800),f)
    started=time.monotonic();empty=True
    result=dict(status='running_MR1',jobs=[],original_status='failed_incomplete',reused_completed_workers=29,
        original_failed_attempts=1,no_training_update=True,model_promoted=False,standing_proven=False,walking_proven=False)
    env=dict(os.environ,PYTHONUTF8='1',MPLCONFIGDIR=str(root/'work/matplotlib-cache'))
    try:
        out.mkdir(exist_ok=False);write(out/'protocol.json',protocol)
        with awake:
            for index,(name,phase,bank,seed,d,sign) in enumerate(jobs()[29:]):
                assert phase=='evaluation' and d is None and sign is None
                assert intact() and data_ok();ram=resources()
                if time.monotonic()-started+300>4800:raise RuntimeError('Insufficient full worker budget')
                cp=old.out/(bank+'.npz');print('START',name,flush=True);empty=False
                try:
                    with (out/(name+'.log')).open('w',encoding='utf-8') as log:
                        receipt=old.native.run_windows_job([sys.executable,str(root/'work/mp1_worker.py'),'--root',str(root),
                            '--output',str(out/name),'--checkpoint',str(cp),'--phase',phase,'--bank',bank,'--seed',str(seed)],
                            timeout=300,stdout=log,cwd=root,env=env)
                    empty=receipt['tree_empty']
                except old.native.WatchdogError as error:
                    empty=error.cleanup_verified;raise
                write(out/(name+'-receipt.json'),receipt)
                result['jobs'].append(dict(name=name,phase=phase,bank=bank,seed=seed,receipt=receipt,available_ram_before_gib=ram))
                write(out/'result.json',result)
                assert receipt['exit_code']==0 and not receipt['timed_out'] and empty
                report,_,_,_,_=old.measured(out/name);c=report['mp1_context']
                assert c==read(out/(name+'-mp1-context.json'))
                assert c['phase']==phase and c['bank']==bank and c['seed']==seed and c['checkpoint_sha256']==sha(cp)
                assert c['worker_sha256']==p['source_hashes']['work/mp1_worker.py']
                old.native.compare(reference(p,'parent',seed),out/name)
                if index==0:
                    assert bank=='negative' and seed==1010
                    assert sha(cp)==sha(root/p['parent_checkpoint'])
                    parent=reference(p,'parent',seed)
                    for filename in ('observations.npz','adhesion-observations.npz','anatomy-observations.npz'):
                        old.native.exact(parent/filename,out/name/filename)
                    assert read(parent/'report.json')['spikes_per_world']==report['spikes_per_world']
                    result['first_fallback_replay_exact']=True
                for path in (out/name).iterdir():
                    if path.is_file():created[path]=sha(path)
                for suffix in ('-receipt.json','-mp1-context.json'):created[out/(name+suffix)]=sha(out/(name+suffix))
                print('COMPLETE',name,flush=True)
            assert len(result['jobs'])==16
            result.update(status='completed_MR1_cells',combined_distinct_workers=45,total_attempts=46,science=finish(p))
        assert awake.released and len(result['jobs'])==16
    except BaseException as error:
        result.update(status='failed_MR1',error_type=type(error).__name__,error=str(error));raise
    finally:
        result.update(seconds=time.monotonic()-started,old_source_created_unchanged=intact(),data_unchanged=data_ok(),
            process_tree_empty_verified=empty,awake_request_released=awake.released,
            created_hashes={path.relative_to(root).as_posix():h for path,h in created.items()})
        if empty and lock.exists() and read(lock)['token']==token:lock.unlink()
        result['lease_released']=not lock.exists()
        if out.exists():write(out/'result.json',result)
        print('RESULT',result['status'],result['seconds'],flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');a=ap.parse_args();p=prospective()
    if a.prepare:
        write(root/'outputs/mp1-completion-mr1-protocol-private-20261113.json',p)
        write(root/'outputs/mp1-completion-mr1-public-protocol-20261113.json',public_plan())
        print('MR1 verified completion plan prepared; no worker, lease or awake request allocated')
    else:execute(p)


if __name__=='__main__':main()
