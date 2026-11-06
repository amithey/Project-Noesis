"""Audit all SR1/prior births without selecting a bank; archive explicit evidence only."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
from sr1_contract import replication_summary
from s2_contract import validate_training_report

root=Path(__file__).resolve().parents[1];out=root/'outputs/training-seed-sr1-20261106'
sys.path[:0]=[str(root/'work/Noesis-Sensory/gpu-pilot/src'),str(root/'work/Noesis-Core/src')]
from noesis_gpu.evaluate import summarize,compare
from noesis_gpu.retention_audit import compare_checkpoints


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


r=read(out/'result.json');p=read(out/'protocol.json')
assert r['status']=='completed_SR1' and len(r['jobs'])==14 and len(r['frozen'])==10 and len(r['training'])==2
for k in ('source_parent_prior_children_unchanged','data_unchanged','process_tree_empty_verified','lease_released'):assert r[k]
assert not (root/'work/Project-Noesis/gpu-pilot/outputs/codex-training-host.lock').exists()
for folder,files in p['source_hashes'].items():
    base=root/('work' if folder=='scripts' else folder)
    if folder!='scripts':assert len(list(base.rglob('*.py')))==len(files)
    for n,h in files.items():assert sha(base/n)==h
for n,h in p['prior_hashes'].items():assert sha(root/n)==h
for f in read(root/'outputs/noesis-required-data-sha256.json')['files']:
    assert sha(root/'work/NoesisData'/f['path'])==f['sha256']
parent=root/'work/evidence/s1-restored/gpu-pilot/outputs/e4-exploration-20261029/train-control-1000/weights-last.npz'
assert sha(parent)==p['parent_sha256']
a1=root/'outputs/adhesion-sign-a1-20261106';s2=root/'outputs/standing-reacquisition-s2-20261106';x1=root/'outputs/exploration-transfer-x1-20261106'
dev=p['development_seeds'];parent_dev={s:summarize(a1/f'legacy-{s}')[1] for s in dev}
children_dev={1004:{s:summarize(s2/f'child-legacy-{s}')[1] for s in dev}}
own_pairs={1004:(summarize(x1/'probe-parent-legacy-off')[1],summarize(x1/'probe-child-legacy-off')[1])}
births={1004:s2/'train-legacy-1004/weights-last.npz'};deltas={};logs={}
old_receipt=read(root/'outputs/standing-reacquisition-s2-20261106-artifact.json')
old_archive=root/'outputs'/old_receipt['path'];assert sha(old_archive)==old_receipt['sha256']
with zipfile.ZipFile(old_archive) as z:
    old_manifest=json.loads(z.read('manifest.json'))
    for f in (births[1004],s2/'train-legacy-1004/report.json'):
        assert sha(f)==old_manifest[f.relative_to(root/'outputs').as_posix()]['sha256']
validate_training_report(read(s2/'train-legacy-1004/report.json'))
assert sha(births[1004])==read(s2/'result.json')['training']['legacy']['checkpoint_sha256']
assert all(summarize(s2/f'child-legacy-{s}')[0]['checkpoint_sha256']==sha(births[1004]) for s in dev)
assert read(x1/'probe-child-legacy-off/report.json')['checkpoint_sha256']==sha(births[1004])
for job in r['jobs']:
    name=job['name'];directory=out/name;report=read(directory/'report.json');receipt=read(out/(name+'-receipt.json'))
    assert receipt==job['receipt'] and receipt['exit_code']==0 and receipt['tree_empty'] and not receipt['timed_out'] and job['available_ram_before_gib']>=2
    context=read(out/(name+'-experiment-context.json'));assert context['training_seed']==job['training_seed'] and context['seed']==job['seed']
    assert context['phase']==job['phase'] and context['worker_sha256']==p['source_hashes']['scripts']['s2_worker.py']
    assert report['body']['adhesion_mapping']=='legacy-positive-Femur' and report['s2_phase']==job['phase']
    count=5000 if job['phase']=='train' else 1000
    with np.load(directory/'adhesion-observations.npz',allow_pickle=False) as z:
        assert z['desired'].shape==(count,42) and z['pads'].shape==z['Femur_delta'].shape==(count,6)
        assert all(np.isfinite(z[k]).all() for k in z.files)
        np.testing.assert_array_equal((z['desired']-report['body']['neutral_pose']).reshape(count,6,7)[:,:,3],z['Femur_delta'])
        np.testing.assert_array_equal(z['pads'],(z['Femur_delta']<=.08).astype(float))
    if job['phase']=='train':
        seed=job['seed'];validate_training_report(report);assert report['import_checkpoint_sha256']==p['parent_sha256']
        cp=directory/'weights-last.npz';births[seed]=cp;record=r['training'][str(seed)]
        assert sha(cp)==record['checkpoint_sha256']==report['saved_checkpoint_sha256']
        assert sha(directory/'report.json')==record['report_sha256'] and sha(directory/'training-observations.npz')==record['observations_sha256']
        assert sha(directory/'exploration-log.npz')==record['exploration_sha256']
        check=compare_checkpoints(parent,cp);assert check==record['checkpoint_audit'] and check['compatible'] and check['parameters']['match'] and check['rule']['match']
        side=read(cp.with_suffix('.model.json'))
        for k,v in report.items():assert side[k]==v
        with np.load(directory/'training-observations.npz',allow_pickle=False) as z:
            assert all(len(z[k])==5000 and np.isfinite(z[k]).all() for k in z.files)
            np.testing.assert_array_equal(z['learning_active'],np.arange(5000)>=10)
        with np.load(directory/'exploration-log.npz',allow_pickle=False) as z:
            np.testing.assert_array_equal(z['control_chunk_index'],np.arange(10,5000))
            assert z['drive'].shape[0]==z['credit'].shape[0]==4990 and all(np.isfinite(z[k]).all() for k in z.files)
            logs[seed]=z['drive'].copy()
        children_dev[seed]={}
    else:
        _,summary=summarize(directory)
        if name.startswith('parent-own-'):assert report['checkpoint_sha256']==p['parent_sha256']
        else:
            train=job['training_seed'];seed=job['seed'];record=next(x for x in r['frozen'] if x['training_seed']==train and x['seed']==seed)
            baseline=out/f'parent-own-{train}' if seed==train else a1/f'legacy-{seed}'
            assert summary==record['summary'] and compare(baseline,directory)==record['paired']
            assert report['checkpoint_sha256']==sha(births[train])
            for file,key in [('report.json','report_sha256'),('observations.npz','observations_sha256'),('adhesion-observations.npz','pads_sha256')]:
                assert sha(directory/file)==record[key]
            if seed==train:own_pairs[train]=(summarize(baseline)[1],summary)
            else:children_dev[train][seed]=summary
replication=replication_summary(parent_dev,children_dev,own_pairs);assert replication==r['replication']
failure_breakdown=[]
diagnostic_cases=[('parent',None,s,a1/f'legacy-{s}',parent_dev[s]) for s in dev]
for train in (1004,1005,1006):
    for seed in dev:
        directory=s2/f'child-legacy-{seed}' if train==1004 else out/f'child-{train}-{seed}'
        diagnostic_cases.append(('child',train,seed,directory,children_dev[train][seed]))
    if train==1004:
        own_dirs=(x1/'probe-parent-legacy-off',x1/'probe-child-legacy-off')
    else:own_dirs=(out/f'parent-own-{train}',out/f'child-{train}-{train}')
    for bank,directory,summary in zip(('parent-own','child-own'),own_dirs,own_pairs[train]):
        diagnostic_cases.append((bank,train,train,directory,summary))
for bank,train,seed,directory,summary in diagnostic_cases:
    with np.load(directory/'observations.npz',allow_pickle=False) as z:
        height=z['position'][10:,0,2];up=z['up_body'][10:,0,2]
        support=(z['foot_load'][10:,0]>1e-5).any(axis=-1)
    code=(up<.7).astype(np.uint8)+2*(height<.3).astype(np.uint8)+4*(height>1.5).astype(np.uint8)+8*(~support).astype(np.uint8)
    failed=code!=0;assert float((~failed).mean())==summary['standing_fraction']
    changes=np.diff(np.r_[False,failed,False].astype(int))
    durations=np.flatnonzero(changes==-1)-np.flatnonzero(changes==1)
    failure_breakdown.append(dict(bank=bank,training_seed=train,evaluation_seed=seed,posthoc_descriptive=True,
        nonstanding_fraction=float(failed.mean()),upright_failure_fraction=float((up<.7).mean()),
        low_height_fraction=float((height<.3).mean()),high_height_fraction=float((height>1.5).mean()),
        no_claw_support_fraction=float((~support).mean()),nonstanding_bouts=len(durations),
        longest_nonstanding_seconds=float(durations.max(initial=0)*.005),
        criterion_mask_counts={str(int(k)):int((code==k).sum()) for k in np.unique(code)},
        mask_bits=dict(upright=1,low_height=2,high_height=4,no_claw_support=8)))
with np.load(s2/'train-legacy-1004/exploration-log.npz',allow_pickle=False) as z:logs[1004]=z['drive'].copy()
for a in (1004,1005,1006):
    with np.load(parent,allow_pickle=False) as before,np.load(births[a],allow_pickle=False) as after:
        for key in ('version','normalize_reward_error','bias_correct_reward_error','initial_weights','motor_indices','topology','parameters','rule'):
            np.testing.assert_array_equal(before[key],after[key])
        refs=np.abs(before['initial_weights']);d=after['weights']-before['weights']
        deltas[a]=np.divide(d,refs,out=np.zeros_like(d),where=refs>0)
cosines=[]
for a,b in ((1004,1005),(1004,1006),(1005,1006)):
    assert not np.array_equal(logs[a],logs[b])
    denom=np.linalg.norm(deltas[a])*np.linalg.norm(deltas[b])
    cosines.append(dict(birth_a=a,birth_b=b,signed_relative_weight_delta_cosine=float(deltas[a]@deltas[b]/denom) if denom else None,
                        posthoc_descriptive=True,not_gradient_signal_proof=True))
limits=['Three training births, not twelve independent training replications.',
        'Own-seed gains are training-seed checks; shared development seeds reused, final validation unused.',
        'Three births cannot fully estimate variability or establish overfitting, learning absence or causal mechanism.',
        'All banks reported, no selection/merge/default change/model promotion.',
        'Weight-delta cosine is posthoc descriptive, not proof of a correct/incorrect learning gradient.',
        'Standing/proxy retention screens do not establish balance recovery, stable walking or environmental responses.']
audit=dict(status='completed_audited_SR1',code_commit='f7660f106e260a2e3240fccf18b83196a1f5bf4b',seconds=r['seconds'],
           replication=replication,posthoc_weight_update_cosines=cosines,posthoc_standing_failure_breakdown=failure_breakdown,
           all_training_horizons_and_logs_complete=True,
           all_checkpoint_reference_sign_topology_bounds_flags_verified=True,all_context_pad_formulas_verified=True,
           source_parent_prior_children_data_unchanged=True,all_jobs_exit0_empty=True,lease_released=True,
           model_promoted=False,walking_proven=False,limits=limits)
ap=root/'outputs/training-seed-sr1-20261106-audit.json';ap.write_text(json.dumps(audit,indent=2,allow_nan=False),encoding='utf-8')
lines=['# SR1: independent training-seed sensitivity','',f"Fourteen new full jobs in{r['seconds']:.3f}s: two25s/4990update births plus twelve5s probes. All technical/hash/context/checkpoint gates pass; source/prior/data unchanged, exit0/empty, lease released. Existing1004birth included, no bank selection.",'',
       '|Training seed|Own parent standing %|Own child standing %|Own gain pp|Development mean gain pp|Worst dev gain pp|Dev >=90%|Strict screen|',
       '|---|---:|---:|---:|---:|---:|---:|---|']
for bank in replication['banks']:
    lines.append(f"|{bank['training_seed']}|{100*bank['own_seed_parent_standing']:.4f}|{100*bank['own_seed_child_standing']:.4f}|{100*bank['own_seed_gain']['standing_fraction']:.4f}|{100*bank['dev_standing_gain_mean']:.4f}|{100*bank['dev_standing_gain_min']:.4f}|{bank['dev_standing_ge90_count']}/4|{bank['strict_screen']['passed']}|")
lines+=['','All development standing values:','','|Training seed|10006 %|10007 %|10010 %|10011 %|','|---|---:|---:|---:|---:|']
for bank in replication['banks']:lines.append('|'+str(bank['training_seed'])+'|'+'|'.join(f"{100*bank['dev_child_standing'][str(s)]:.4f}" for s in dev)+'|')
lines+=['','Complete screens and metrics in audit JSON.','','Limits:','']+['- '+x for x in limits]+['']
mp=root/'outputs/training-seed-sr1-20261106-audit.md';mp.write_text('\n'.join(lines),encoding='utf-8')
files=[out/'protocol.json',out/'result.json',ap,mp]
for job in r['jobs']:
    directory=out/job['name'];name=job['name']
    files.extend(out/(name+s) for s in ('-receipt.json','-context.json','-experiment-context.json','.log'))
    files.extend(directory/n for n in ('report.json','adhesion-observations.npz'))
    files.append(directory/('training-observations.npz' if job['phase']=='train' else 'observations.npz'))
    if job['phase']=='train':files.extend(directory/n for n in ('exploration-log.npz','weights-last.npz','weights-last.model.json'))
entries={f.relative_to(root).as_posix():f.read_bytes() for f in files}
for n in p['prior_hashes']:
    f=root/n;entries[f.relative_to(root).as_posix()]=f.read_bytes()
entries['prior-training-1004/weights-last.npz']=births[1004].read_bytes()
entries['runtime-source-manifest.json']=(root/'outputs/thermo-pilot-tp2-20261106-source.json').read_bytes()
for n in ('sr1_contract.py','run_training_seed_sr1.py','analyse_training_seed_sr1.py','test_sr1_contract.py','s2_worker.py','s2_contract.py','adhesion_mapping.py'):
    entries['scripts/'+n]=(root/'work'/n).read_bytes()
manifest={n:dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest()) for n,b in entries.items()}
archive=root/'outputs/training-seed-sr1-20261106-evidence.zip';assert not archive.exists()
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for n,b in entries.items():z.writestr(n,b)
    z.writestr('manifest.json',json.dumps(manifest,indent=2))
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None and set(z.namelist())==set(entries)|{'manifest.json'}
    assert all(hashlib.sha256(z.read(n)).hexdigest()==v['sha256'] for n,v in manifest.items())
receipt=dict(path=archive.name,bytes=archive.stat().st_size,sha256=sha(archive),files=len(entries),
             crc_and_all_hashes_verified=True,raw_graph_cache_private_lease_excluded=True,release_uploaded=False)
(root/'outputs/training-seed-sr1-20261106-artifact.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
print(json.dumps(dict(replication=replication,posthoc_cosines=cosines,artifact=receipt),indent=2))
