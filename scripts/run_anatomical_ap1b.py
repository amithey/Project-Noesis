"""Fixed AP1 allocation: exact disabled replay, paired birth, complete frozen bank."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import uuid
import zipfile
import numpy as np
import psutil
from ap1_feedback import non_tarsal_fraction,corrected_reward,validate_ap1_training
from ap1_body import MODES
from support_descriptor_c0 import REGIONS,describe_support
from s2_contract import standing_retention

root=Path(__file__).resolve().parents[1]
out=root/'outputs/anatomical-feedback-ap1b-20261106'
pilot=root/'work/Noesis-Sensory/gpu-pilot/src';core=root/'work/Noesis-Core/src'
sys.path[:0]=[str(pilot),str(core)]
from noesis_gpu.evaluate import summarize,compare
from noesis_gpu.retention_audit import compare_checkpoints
from noesis_gpu.process_watchdog import run_windows_job,WatchdogError

DEV=(10006,10007,10010,10011);SEEDS=(1005,*DEV)
SR=root/'outputs/training-seed-sr1-20261106'
A1=root/'outputs/adhesion-sign-a1-20261106'
H2=root/'outputs/body-telemetry-h2b-20261106'
PARENT=root/'work/evidence/s1-restored/gpu-pilot/outputs/e4-exploration-20261029/train-control-1000/weights-last.npz'
CONTROL=SR/'train-1005/weights-last.npz'


def read(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while b:=f.read(1024*1024):h.update(b)
    return h.hexdigest()
def arrays(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def exact(a,b,prefix=None):
    a,b=arrays(a),arrays(b);assert set(a)==set(b)
    for k in a:np.testing.assert_array_equal(a[k] if prefix is None else a[k][:prefix],b[k] if prefix is None else b[k][:prefix],err_msg=k)
def old_dir(bank,seed):
    return (SR/'parent-own-1005' if seed==1005 else A1/f'legacy-{seed}') if bank=='parent' else SR/f'child-1005-{seed}'


def prospective():
    inherited=read(root/'outputs/support-calibration-c0-20261106/protocol.json')['source_hashes']
    for n,h in inherited.items():assert sha(root/n)==h
    source=dict(inherited)
    for n in ('ap1_feedback.py','ap1_body.py','ap1_worker.py','run_anatomical_ap1.py','test_ap1_feedback.py','run_anatomical_ap1b.py','test_ap1b_mass_gate.py','audit_ap1_a0.py'):
        source['work/'+n]=sha(root/'work'/n)
    prov=read(root/'outputs/thermo-pilot-tp2-20261106-source.json')
    for label,folder in (('pilot',pilot),('core',core)):
        assert len(list(folder.rglob('*.py')))==len(prov[label])
        for f in prov[label]:assert sha(folder/f['path'])==f['sha256']
    assert sha(PARENT)=='9fd7d41177bf2c745809401c3b142a832661d1a18d0b117b30cb02aea63fea3c'
    assert sha(CONTROL)=='46a6975dc7a1479f794034258abae67ff178d31772ac961d54617519a2b00853'
    selected={
        'adhesion-sign-a1-20261106':[old_dir('parent',s)/n for s in DEV for n in ('report.json','observations.npz','adhesion-observations.npz')],
        'training-seed-sr1-20261106':[SR/'train-1005'/n for n in ('report.json','weights-last.npz','training-observations.npz','adhesion-observations.npz','exploration-log.npz')]+
            [old_dir(b,s)/n for b in ('parent','control') for s in SEEDS if b=='control' or s==1005 for n in ('report.json','observations.npz','adhesion-observations.npz')],
        'body-telemetry-h2b-20261106':[H2/c/n for c in ('parent-10007','child1005-10007') for n in ('report.json','observations.npz','adhesion-observations.npz','ground-contacts.npz','body-telemetry.npz','telemetry-model.json')]}
    expected=('cee924bff17e1a72a2cb379c8f44e51aa5fab06fcaff6015a71f9df5ef9f92bf',
        'e23d61e932062a5172391232b2d6e2513b77c6cd011c494c6a50168614352941',
        'f660b673da8d431873cd829a364d423633b01fe395ce971ecb6f3190984e3257')
    prior={PARENT.relative_to(root).as_posix():sha(PARENT)};archives={}
    for (artifact,paths),expected_sha in zip(selected.items(),expected):
        receipt=read(root/'outputs'/f'{artifact}-artifact.json');archive=root/'outputs'/receipt['path']
        assert sha(archive)==receipt['sha256']==expected_sha;archives[archive.relative_to(root).as_posix()]=expected_sha
        with zipfile.ZipFile(archive) as z:
            manifest=json.loads(z.read('manifest.json'))
            for p in paths:
                key=p.relative_to(root/'outputs' if artifact.startswith('adhesion-') else root).as_posix()
                h=manifest[key]['sha256'];assert sha(p)==h;prior[p.relative_to(root).as_posix()]=h
    for bank,case in (('parent','parent-10007'),('control','child1005-10007')):
        for n in ('observations.npz','adhesion-observations.npz'):exact(H2/case/n,old_dir(bank,10007)/n)
        compare(H2/case,old_dir(bank,10007))
    audit_path=root/'outputs/anatomical-feedback-ap1-a0-audit-20261106.json'
    audit=read(audit_path)
    assert audit['status']=='audited_complete_disabled_replay_technical_gate_only' and audit['all_SR1_checkpoint_body_pad_exploration_arrays_exact']
    for n,h in audit['file_hashes'].items():assert sha(root/n)==h;prior[n]=h
    prior[audit_path.relative_to(root).as_posix()]=sha(audit_path)
    return dict(experiment='AP1B corrected technical mass gate; same AP1 scientific pilot',date='2026-11-06',source_hashes=source,
        prior_hashes=prior,artifact_hashes=archives,parent_sha256=sha(PARENT),control_checkpoint_sha256=sha(CONTROL),
        seed=1005,development_seeds=list(DEV),own_seed=1005,reserved_final_seeds_unused=[10,11,200,201,202],
        formula='clip(native_v7 - 0.5*q,-1,1); q=0 if total upward ground reaction<=1e-5 else (central+proximal)/total',
        region_columns=list(REGIONS),contact_mode_codebook=list(MODES),strengths=[0.,.5],
        policy='legacy-positive-Femur',training_seconds=25,frozen_seconds=5,learning_updates=4990,
        learning_rate=.2,eligibility_ms=100,original_reference_bound=.5,exploration_mv_ms=1.5,
        aggregate_seconds=3000,training_worker_seconds=600,frozen_worker_seconds=300,maximum_new_jobs=14,
        sequence=['reuse independently audited complete AP1A0 disabled control, no rerun','four missing parent probes','four missing v7child probes',
                  'candidate birth/same parent and exploration','five candidate probes'],
        reuse=['AP1A0 disabled25s complete control','H2B parent-10007','H2B child1005-10007'],
        primary='Each shared dev seed: increase tarsal-only fraction AND reduce mean non-tarsal reaction fraction versus BOTH parent/v7child; also unchanged strict standing/slip/proxy retention',
        interpretation='Anatomical improvement with capability loss is mixed, not pass; own seed descriptive; one pilot birth requires independent confirmation; no promotion',
        stop='Technical/hash/context/mass/force/horizon/log/sidecar/parity/resource/time failure or insufficient full worker budget; no retry/retune/partial scientific interpretation',
        new_balance_validator=False,model_promoted=False,walking_proven=False)


def anatomy(directory,phase):
    a=arrays(directory/'anatomy-observations.npz');model=read(directory/'anatomy-model.json')
    count=5001 if phase=='train' else 1001
    assert all(len(v)==count and np.isfinite(v).all() for v in a.values())
    np.testing.assert_allclose(a['time'],np.arange(count)*.005,rtol=0,atol=1e-10)
    assert a['region_vertical_reaction'].shape==(count,4) and np.all(a['region_vertical_reaction']>=0)
    np.testing.assert_array_equal(a['non_tarsal_fraction'],[non_tarsal_fraction(r) for r in a['region_vertical_reaction']])
    assert np.isin(a['contact_mode_code'],np.arange(5)).all()
    assert model['region_columns']==list(REGIONS) and model['contact_mode_codebook']==list(MODES)
    assert np.isclose(model['total_mass'],model['model_total_mass'],rtol=1e-12,atol=1e-15) and model['unmapped_anatomical_count']==1
    assert model['production_static_fusion'] and model['descriptor_is_not_standing_validator']
    assert np.max(a['force_closure_max_error'])<=1e-6*model['total_mass']*9810+1e-10
    assert np.max(a['upstream_claw_force_max_error'])<=1e-10
    return a


def reused(case):
    directory=H2/case;model=read(directory/'telemetry-model.json');t=arrays(directory/'body-telemetry.npz');c=arrays(directory/'ground-contacts.npz')
    names=[None]*model['nbody']
    for i,n in zip(model['body_ids'],model['body_names']):names[i]=n
    amounts=[];codes=[]
    for frame in range(1001):
        forces=np.zeros((model['nbody'],3));use=c['sample_index']==frame
        np.add.at(forces,c['body_id'][use],c['force_on_fly'][use])
        np.testing.assert_allclose(forces.sum(0),t['ground_force'][frame],rtol=1e-12,atol=1e-10)
        d=describe_support(forces,names);amounts.append([d['vertical_reaction_by_region'][r] for r in REGIONS]);codes.append(MODES.index(d['contact_mode']))
    return dict(region_vertical_reaction=np.asarray(amounts),contact_mode_code=np.asarray(codes),
        non_tarsal_fraction=np.asarray([non_tarsal_fraction(a) for a in amounts]),com_velocity=t['com_velocity'])


def contact_summary(a):
    # Match native frozen summary: discard initial sample and first ten chunks.
    select=slice(11,None);r=a['region_vertical_reaction'][select];total=float(r.sum())
    return dict(samples=len(r),mode_fractions={m:float(np.mean(a['contact_mode_code'][select]==i)) for i,m in enumerate(MODES)},
        mean_non_tarsal_fraction=float(np.mean(a['non_tarsal_fraction'][select])),
        summed_vertical_reaction_share={n:float(r[:,i].sum()/total) if total else None for i,n in enumerate(REGIONS)},
        median_abs_com_vertical_velocity_mm_s=float(np.median(np.abs(a['com_velocity'][select,2]))),descriptor_not_balance_proof=True)


def execute(protocol):
    me=psutil.Process();own={me.pid,*[p.pid for p in me.parents()]}
    def resources():
        if any(p.pid not in own and (p.info['name'] or '').lower() in ('python.exe','pythonw.exe') for p in psutil.process_iter(['name'])):raise RuntimeError('Competing Python')
        ram=psutil.virtual_memory().available/2**30
        if ram<2:raise RuntimeError('RAM below2GiB')
        return ram
    data=read(root/'outputs/noesis-required-data-sha256.json')['files'];child_hashes={}
    def unchanged():
        return all(sha(root/n)==h for group in ('source_hashes','prior_hashes') for n,h in protocol[group].items()) and all(sha(root/n)==h for n,h in child_hashes.items())
    def data_unchanged():return all(sha(root/'work/NoesisData'/f['path'])==f['sha256'] for f in data)
    resources();assert data_unchanged();out.mkdir(exist_ok=False)
    lock=root/'work/Project-Noesis/gpu-pilot/outputs/codex-training-host.lock';token=uuid.uuid4().hex
    with os.fdopen(os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY),'w') as f:json.dump(dict(owner='codex',pid=os.getpid(),token=token,deadline_unix=time.time()+3000),f)
    started=time.monotonic();empty=True
    result=dict(status='running',jobs=[],training={},cells={},retention={},model_promoted=False,walking_proven=False)
    env=dict(os.environ,PYTHONUTF8='1',MPLCONFIGDIR=str(root/'work/matplotlib-cache'))
    def launch(phase,bank,seed,cp,strength):
        nonlocal empty
        assert unchanged() and data_unchanged();ram=resources();budget=600 if phase=='train' else 300
        if time.monotonic()-started>3000-budget:raise RuntimeError('Insufficient full worker budget')
        name=f'{phase}-{bank}-{seed}';directory=out/name;print('START',name,flush=True);empty=False
        try:
            with (out/(name+'.log')).open('w',encoding='utf-8') as log:
                receipt=run_windows_job([sys.executable,str(root/'work/ap1_worker.py'),'--phase',phase,'--bank',bank,'--strength',str(strength),
                    '--seed',str(seed),'--root',str(root),'--checkpoint',str(cp),'--output',str(directory)],timeout=budget,stdout=log,cwd=root,env=env)
            empty=receipt['tree_empty']
        except WatchdogError as error:
            empty=error.cleanup_verified;raise
        write(out/(name+'-receipt.json'),receipt);result['jobs'].append(dict(name=name,phase=phase,bank=bank,seed=seed,receipt=receipt,available_ram_before_gib=ram));write(out/'result.json',result)
        assert empty and receipt['exit_code']==0 and not receipt['timed_out']
        report=read(directory/'report.json');assert report['ap1_worker_completed'] and report['adhesion_probe_mode']=='legacy'
        assert report['ap1_context']['effective_training_penalty']==strength
        assert report['ap1_context']['loaded_checkpoint_sha256']==sha(cp)
        assert report['seed']==seed if phase=='train' else report['seeds']==[seed]
        a=anatomy(directory,phase);pads=arrays(directory/'adhesion-observations.npz');count=5000 if phase=='train' else 1000
        assert pads['desired'].shape==(count,42) and pads['pads'].shape==pads['Femur_delta'].shape==(count,6)
        np.testing.assert_array_equal(pads['pads'],(pads['Femur_delta']<=.08).astype(float))
        if phase=='train':
            validate_ap1_training(report,strength);r=arrays(directory/'reward-accounting.npz');obs=arrays(directory/'training-observations.npz')
            assert all(len(v)==5000 and np.isfinite(v).all() for v in r.values())
            assert all(len(v)==5000 and np.isfinite(v).all() for v in obs.values())
            np.testing.assert_array_equal(r['control_chunk_index'],np.arange(5000));np.testing.assert_array_equal(r['time'],a['time'][1:])
            np.testing.assert_array_equal(r['non_tarsal_fraction'],a['non_tarsal_fraction'][1:])
            np.testing.assert_array_equal(r['applied_reward'],[corrected_reward(n,q,strength) for n,q in zip(r['native_reward'],r['non_tarsal_fraction'])])
            np.testing.assert_array_equal(obs['reward'],r['applied_reward']);np.testing.assert_array_equal(obs['learning_active'],np.arange(5000)>=10)
            np.testing.assert_array_equal(r['effective_penalty'],r['native_reward']-r['applied_reward'])
            checkpoint=directory/'weights-last.npz';assert sha(checkpoint)==report['saved_checkpoint_sha256']
            audit=compare_checkpoints(PARENT,checkpoint);assert audit['compatible'] and audit['parameters']['match'] and audit['rule']['match']
            for k in ('version','normalize_reward_error','bias_correct_reward_error'):np.testing.assert_array_equal(arrays(PARENT)[k],arrays(checkpoint)[k])
            sidecar=read(checkpoint.with_suffix('.model.json'))
            for k,v in report.items():assert sidecar[k]==v
            child_hashes[checkpoint.relative_to(root).as_posix()]=sha(checkpoint)
            result['training'][bank]=dict(checkpoint_sha256=sha(checkpoint),checkpoint_audit=audit,actual_objective=report['reward_objective'],
                mean_native_reward=float(r['native_reward'].mean()),mean_applied_reward=float(r['applied_reward'].mean()),mean_effective_penalty=float(r['effective_penalty'].mean()))
        else:assert report['weights_unchanged'] and not report['learning'] and report['status']=='completed_transport_pilot'
        print('COMPLETE',name,flush=True);return directory,a
    banks={b:{} for b in ('parent','control','candidate')};directories={b:{} for b in banks}
    def cell(bank,seed,directory,a,is_reused=False):
        _,summary=summarize(directory);banks[bank][seed]=summary;directories[bank][seed]=directory
        if bank!='parent':compare(directories['parent'][seed],directory)
        result['cells'][f'{bank}-{seed}']=dict(summary=summary,anatomical=contact_summary(a),reused=is_reused,report_sha256=sha(directory/'report.json'),observations_sha256=sha(directory/'observations.npz'))
        write(out/'result.json',result)
    try:
        write(out/'protocol.json',protocol)
        control_dir=root/'outputs/anatomical-feedback-ap1-20261106/train-control-1005'
        anatomy(control_dir,'train');control_report=read(control_dir/'report.json');validate_ap1_training(control_report,0.)
        result['training']['control']=dict(reused=True,checkpoint_sha256=sha(control_dir/'weights-last.npz'),actual_objective=control_report['reward_objective'])
        assert sha(control_dir/'weights-last.npz')==sha(CONTROL)
        for n in ('training-observations.npz','adhesion-observations.npz','exploration-log.npz'):exact(control_dir/n,SR/'train-1005'/n)
        result['disabled_control_exact_SR1_replay']=True;write(out/'result.json',result)
        for bank,cp,case in (('parent',PARENT,'parent-10007'),('control',CONTROL,'child1005-10007')):
            for seed in SEEDS:
                if seed==10007:cell(bank,seed,H2/case,reused(case),True);continue
                directory,a=launch('frozen',bank,seed,cp,0.)
                for n in ('observations.npz','adhesion-observations.npz'):exact(directory/n,old_dir(bank,seed)/n)
                compare(directory,old_dir(bank,seed));assert read(directory/'report.json')['spikes_per_world']==read(old_dir(bank,seed)/'report.json')['spikes_per_world']
                cell(bank,seed,directory,a)
        candidate_dir,_=launch('train','candidate',1005,PARENT,.5)
        # Same perturbation/current schedule. Credit deliberately need not match.
        base=arrays(control_dir/'exploration-log.npz');new=arrays(candidate_dir/'exploration-log.npz')
        assert set(base)==set(new)
        for k in base:
            if k!='credit':np.testing.assert_array_equal(base[k],new[k],err_msg=k)
        for n in ('training-observations.npz','adhesion-observations.npz'):
            a,b=arrays(control_dir/n),arrays(candidate_dir/n)
            for k in a:
                if k!='reward':np.testing.assert_array_equal(a[k][:11],b[k][:11],err_msg=k)
        result['candidate_drive_schedule_and_initial_body_exact']=True
        cp=candidate_dir/'weights-last.npz'
        for seed in SEEDS:
            directory,a=launch('frozen','candidate',seed,cp,0.);cell('candidate',seed,directory,a)
        for reference in ('parent','control'):
            result['retention'][reference]=standing_retention({s:banks[reference][s] for s in DEV},{s:banks['candidate'][s] for s in DEV},DEV)
        result['anatomical_comparisons']={}
        for s in SEEDS:
            c=result['cells'][f'candidate-{s}']['anatomical'];pair={}
            for b in ('parent','control'):
                p=result['cells'][f'{b}-{s}']['anatomical']
                pair[b]=dict(tarsal_only_fraction_gain=c['mode_fractions']['tarsal_only']-p['mode_fractions']['tarsal_only'],
                    mean_non_tarsal_fraction_change=c['mean_non_tarsal_fraction']-p['mean_non_tarsal_fraction'])
            result['anatomical_comparisons'][str(s)]=pair
        result['anatomical_dev_each_vs_both']=all(v['tarsal_only_fraction_gain']>0 and v['mean_non_tarsal_fraction_change']<0 for s in DEV for v in result['anatomical_comparisons'][str(s)].values())
        assert len(result['jobs'])==14
        result.update(status='completed_AP1B',pilot_births=1,independent_confirmation_required=True,old_gate_unchanged=True)
    except BaseException as error:
        result.update(status='failed',error_type=type(error).__name__,error=str(error));raise
    finally:
        result.update(seconds=time.monotonic()-started,source_prior_children_unchanged=unchanged(),data_unchanged=data_unchanged(),process_tree_empty_verified=empty)
        if empty and lock.exists() and read(lock)['token']==token:lock.unlink()
        result['lease_released']=not lock.exists();write(out/'result.json',result)
        print('RESULT',result['status'],result['seconds'],flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');args=ap.parse_args();p=prospective()
    if args.prepare:
        write(root/'outputs/anatomical-feedback-ap1b-20261106-protocol.json',p)
        print('AP1B preflight: hashes, signed inputs, two reused frozen bridges verified; no simulation or lease')
    else:execute(p)


if __name__=='__main__':main()
