"""Three frozen-only corrective probes; reuse completed AP1B training and bank."""
import argparse,json,os,sys,time,uuid
from pathlib import Path
import numpy as np
import psutil
import run_anatomical_ap1b as base

root=base.root;out=root/'outputs/anatomical-feedback-ap1c-20261106';old=root/'outputs/anatomical-feedback-ap1b-20261106'
read,write,sha,arrays=base.read,base.write,base.sha,base.arrays
CP=old/'train-candidate-1005/weights-last.npz';SEEDS=(10007,10010,10011)


def prospective():
    previous=read(old/'protocol.json');stop_path=root/'outputs/anatomical-feedback-ap1b-stop-audit-20261106.json';stop=read(stop_path)
    assert stop['status']=='audited_AP1B_technical_stop_no_scientific_verdict' and stop['all_5000_candidate_native_rewards_recomputed']
    assert stop['exploration_drive_AND_credit_exact_control'] and stop['source_data_priors_unchanged']
    source=dict(previous['source_hashes']);prior=dict(previous['prior_hashes'])
    for group in (source,prior,previous['artifact_hashes'],stop['file_hashes']):
        for n,h in group.items():assert sha(root/n)==h
    prior.update(stop['file_hashes']);prior[stop_path.relative_to(root).as_posix()]=sha(stop_path)
    static=root/'outputs/anatomical-coverage-ap1c-static-20261106.json';coverage=read(static)
    assert coverage['all_ground_geometry_classified'] and coverage['explicit_ground_pairs']==55 and coverage['compound_proximal_geoms']==6
    assert coverage['physical_integration_steps']==coverage['neural_steps']==coverage['model_time']==0
    prior[static.relative_to(root).as_posix()]=sha(static)
    for n in ('support_descriptor_c1.py','ap1c_body.py','ap1c_worker.py','run_anatomical_ap1c.py','test_ap1c_coverage.py','audit_ap1b_stop.py'):
        source['work/'+n]=sha(root/'work'/n)
    for n in ('anatomy.py','assets/model/neuromechfly/rigging.yaml'):
        path=root/'work/flygym/src/flygym'/n;source[path.relative_to(root).as_posix()]=sha(path)
    assert sha(CP)==stop['candidate_sha256']=='aced093428ea850e6f091f36fa13caa86afad0948ca0deb6cc5f3d85321b265c'
    return dict(experiment='AP1C frozen-only native compound-link coverage correction',date='2026-11-07',source_hashes=source,prior_hashes=prior,
        artifact_hashes=previous['artifact_hashes'],candidate_sha256=sha(CP),training_source='a05b0daa3c4416f1ffc0a2c59f8770f77069bd8b',
        original_training_seconds=25,original_training_updates=4990,new_training_jobs=0,learning_updates=0,
        seeds=list(SEEDS),seconds=5,chunks=1000,maximum_new_jobs=3,aggregate_seconds=900,worker_seconds=300,
        descriptor_change='Exactly six native *_trochanterfemur links -> proximal_leg; old C0 supported-domain identical, other unknown loaded anatomy still rejects',
        scientific_formula_and_gates_unchanged=True,completed_A0_B_banks_reused_without_rerun=True,
        failed_partial_B_seed10007_preserved=True,require_exact_partial_body_anatomy_prefix=True,
        stop='technical/hash/context/horizon/mass/force/prefix/resource/timeout/remaining-full-worker-budget failure; no automatic retry/retune/partial scientific verdict',
        model_promoted=False,walking_proven=False,one_training_birth_requires_independent_confirmation=True)


def execute(p):
    me=psutil.Process();own={me.pid,*[x.pid for x in me.parents()]}
    def resources():
        if any(x.pid not in own and (x.info['name'] or '').lower() in ('python.exe','pythonw.exe') for x in psutil.process_iter(['name'])):raise RuntimeError('Competing Python')
        ram=psutil.virtual_memory().available/2**30
        if ram<2:raise RuntimeError('RAM below2GiB')
        return ram
    def unchanged():return all(sha(root/n)==h for group in ('source_hashes','prior_hashes') for n,h in p[group].items())
    data=read(root/'outputs/noesis-required-data-sha256.json')['files']
    def data_ok():return all(sha(root/'work/NoesisData'/f['path'])==f['sha256'] for f in data)
    resources();assert unchanged() and data_ok();out.mkdir(exist_ok=False)
    lock=root/'work/Project-Noesis/gpu-pilot/outputs/codex-training-host.lock';token=uuid.uuid4().hex
    with os.fdopen(os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY),'w') as f:json.dump(dict(owner='codex',pid=os.getpid(),token=token,deadline_unix=time.time()+900),f)
    started=time.monotonic();empty=True;previous=read(old/'result.json')
    result=dict(status='running',jobs=[],cells=dict(previous['cells']),training=previous['training'],retention={},model_promoted=False,walking_proven=False)
    env=dict(os.environ,PYTHONUTF8='1',MPLCONFIGDIR=str(root/'work/matplotlib-cache'))
    try:
        write(out/'protocol.json',p)
        for seed in SEEDS:
            assert unchanged() and data_ok();ram=resources()
            if time.monotonic()-started>600:raise RuntimeError('Insufficient full300s worker budget')
            name=f'frozen-candidate-{seed}';directory=out/name;print('START',name,flush=True);empty=False
            try:
                with (out/(name+'.log')).open('w',encoding='utf-8') as log:
                    receipt=base.run_windows_job([sys.executable,str(root/'work/ap1c_worker.py'),'--phase','frozen','--bank','candidate','--strength','0',
                        '--seed',str(seed),'--root',str(root),'--checkpoint',str(CP),'--output',str(directory)],timeout=300,stdout=log,cwd=root,env=env)
                empty=receipt['tree_empty']
            except base.WatchdogError as error:
                empty=error.cleanup_verified;raise
            write(out/(name+'-receipt.json'),receipt);result['jobs'].append(dict(name=name,seed=seed,receipt=receipt,available_ram_before_gib=ram));write(out/'result.json',result)
            assert receipt['exit_code']==0 and not receipt['timed_out'] and empty
            report,summary=base.summarize(directory);a=base.anatomy(directory,'frozen')
            assert report['ap1_worker_completed'] and report['checkpoint_sha256']==sha(CP) and report['seeds']==[seed]
            assert report['weights_unchanged'] and not report['learning'] and report['motor_exploration']==0
            assert report['ap1_context']['measurement_revision']=='C1-native-trochanterfemur-proximal'
            pads=arrays(directory/'adhesion-observations.npz')
            assert pads['desired'].shape==(1000,42) and pads['pads'].shape==pads['Femur_delta'].shape==(1000,6)
            np.testing.assert_array_equal(pads['pads'],(pads['Femur_delta']<=.08).astype(float))
            base.compare(base.old_dir('parent',seed),directory);base.compare(base.old_dir('control',seed),directory)
            if seed==10007:
                partial=old/name;oldobs=arrays(partial/'observations.npz');newobs=arrays(directory/'observations.npz');old_a=arrays(partial/'anatomy-observations.npz')
                n=read(partial/'report.json')['completed_control_chunks'];assert n==366
                for k,v in oldobs.items():np.testing.assert_array_equal(newobs[k][:n],v)
                for k,v in old_a.items():np.testing.assert_array_equal(a[k][:n+1],v)
                np.testing.assert_array_equal(a['compound_proximal_vertical_reaction'][:n+1],np.zeros(n+1))
                assert a['compound_proximal_vertical_reaction'][n+1]>0
                assert np.all(a['compound_proximal_vertical_reaction']<=a['region_vertical_reaction'][:,3]+1e-10)
                result['failed_partial_prefix_exact']=dict(body_chunks=n,anatomy_samples=n+1,first_new_compound_force=float(a['compound_proximal_vertical_reaction'][n+1]),first_new_time=float(a['time'][n+1]))
            result['cells'][f'candidate-{seed}']=dict(summary=summary,anatomical=base.contact_summary(a),reused=False,report_sha256=sha(directory/'report.json'),observations_sha256=sha(directory/'observations.npz'))
            write(out/'result.json',result);print('COMPLETE',name,flush=True)
        for ref in ('parent','control'):
            result['retention'][ref]=base.standing_retention({s:result['cells'][f'{ref}-{s}']['summary'] for s in base.DEV},{s:result['cells'][f'candidate-{s}']['summary'] for s in base.DEV},base.DEV)
        comparison={}
        for seed in base.SEEDS:
            c=result['cells'][f'candidate-{seed}']['anatomical'];comparison[str(seed)]={}
            for ref in ('parent','control'):
                r=result['cells'][f'{ref}-{seed}']['anatomical']
                comparison[str(seed)][ref]=dict(tarsal_only_fraction_gain=c['mode_fractions']['tarsal_only']-r['mode_fractions']['tarsal_only'],mean_non_tarsal_fraction_change=c['mean_non_tarsal_fraction']-r['mean_non_tarsal_fraction'])
        result['anatomical_comparisons']=comparison
        result['anatomical_dev_each_vs_both']=all(v['tarsal_only_fraction_gain']>0 and v['mean_non_tarsal_fraction_change']<0 for seed in base.DEV for v in comparison[str(seed)].values())
        assert len(result['cells'])==15 and len(result['jobs'])==3
        result.update(status='completed_AP1C',complete_scientific_bank=True,one_training_birth=True,independent_confirmation_required=True,original_gates_unchanged=True)
    except BaseException as error:
        result.update(status='failed',error_type=type(error).__name__,error=str(error));raise
    finally:
        result.update(seconds=time.monotonic()-started,source_prior_children_unchanged=unchanged(),data_unchanged=data_ok(),process_tree_empty_verified=empty)
        if empty and lock.exists() and read(lock)['token']==token:lock.unlink()
        result['lease_released']=not lock.exists();write(out/'result.json',result);print('RESULT',result['status'],result['seconds'],flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');args=ap.parse_args();p=prospective()
    if args.prepare:write(root/'outputs/anatomical-feedback-ap1c-20261106-protocol.json',p);print('AP1C prepared, no simulation/lease')
    else:execute(p)


if __name__=='__main__':main()
