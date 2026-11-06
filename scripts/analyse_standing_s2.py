"""Recompute S2 physical evidence and package explicitly selected signed artifacts."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
from s2_contract import standing_retention,validate_training_report

root=Path(__file__).resolve().parents[1];out=root/'outputs/standing-reacquisition-s2-20261106'
sys.path[:0]=[str(root/'work/Noesis-Sensory/gpu-pilot/src'),str(root/'work/Noesis-Core/src')]
from noesis_gpu.evaluate import summarize,compare,MATCHED_SETTINGS
from noesis_gpu.retention_audit import compare_checkpoints


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def exact(a,b):
    with np.load(a,allow_pickle=False) as x,np.load(b,allow_pickle=False) as y:
        return set(x.files)==set(y.files) and all(np.array_equal(x[k],y[k]) for k in x.files)


p=read(out/'protocol.json');r=read(out/'result.json')
assert r['status']=='completed_S2' and len(r['jobs'])==12 and len(r['frozen'])==8
for k in ('parent_unchanged','source_unchanged','data_unchanged','process_tree_empty_verified','lease_released','exploration_stream_gate'):
    assert r[k],k
assert not (root/'work/Project-Noesis/gpu-pilot/outputs/codex-training-host.lock').exists()
assert all(r['bridge_gate'].values()) and len(r['bridge_gate'])==2
for job in r['jobs']:
    receipt=read(out/(job['name']+'-receipt.json'));assert receipt==job['receipt']
    assert receipt['exit_code']==0 and receipt['tree_empty'] and not receipt['timed_out']
    assert job['available_ram_before_gib']>=2
for n,h in p['source_hashes']['scripts'].items():assert sha(root/'work'/n)==h
provenance=read(root/'outputs/thermo-pilot-tp2-20261106-source.json')
for label,folder in [('pilot',root/'work/Noesis-Sensory/gpu-pilot/src'),('core',root/'work/Noesis-Core/src')]:
    assert len(list(folder.rglob('*.py')))==len(provenance[label])
    for f in provenance[label]:assert sha(folder/f['path'])==f['sha256']
for f in read(root/'outputs/noesis-required-data-sha256.json')['files']:
    assert sha(root/'work/NoesisData'/f['path'])==f['sha256']
parent=root/'work/evidence/s1-restored/gpu-pilot/outputs/e4-exploration-20261029/train-control-1000/weights-last.npz'
assert sha(parent)==p['parent_sha256']
seeds=p['frozen_seeds'];modes=p['modes'];a1=root/'outputs/adhesion-sign-a1-20261106'
bases={m:{} for m in modes};children={m:{} for m in modes};rows=[];births={}
for mode in modes:
    bridge=out/f'bridge-{mode}-10006'
    for name in ('observations.npz','adhesion-observations.npz'):
        assert exact(bridge/name,a1/f'{mode}-10006'/name)
    compare(a1/f'{mode}-10006',bridge)
    directory=out/f'train-{mode}-1004';report=read(directory/'report.json')
    validate_training_report(report)
    cp=directory/'weights-last.npz';assert sha(cp)==r['training'][mode]['checkpoint_sha256']==report['saved_checkpoint_sha256']
    assert sha(directory/'report.json')==r['training'][mode]['report_sha256']
    assert report['import_checkpoint_sha256']==p['parent_sha256']
    check=compare_checkpoints(parent,cp);assert check==r['training'][mode]['checkpoint_audit']
    assert check['compatible'] and check['parameters']['match'] and check['rule']['match']
    with np.load(parent,allow_pickle=False) as old,np.load(cp,allow_pickle=False) as new:
        for key in ('version','normalize_reward_error','bias_correct_reward_error'):
            np.testing.assert_array_equal(old[key],new[key])
    sidecar=read(cp.with_suffix('.model.json'))
    for key,value in report.items():assert sidecar[key]==value
    with np.load(directory/'training-observations.npz',allow_pickle=False) as z:
        assert all(len(z[k])==5000 and np.isfinite(z[k]).all() for k in z.files)
        np.testing.assert_array_equal(z['learning_active'],np.arange(5000)>=10)
    with np.load(directory/'exploration-log.npz',allow_pickle=False) as z:
        np.testing.assert_array_equal(z['control_chunk_index'],np.arange(10,5000))
        assert all(len(z[k])==4990 for k in ('drive','credit','control_chunk_index'))
        assert all(np.isfinite(z[k]).all() for k in z.files)
    births[mode]=dict(checkpoint_sha256=sha(cp),completed_chunks=5000,learning_updates=4990,
                      training_standing_fraction=report['standing_fraction'],checkpoint_audit=check)
    for seed in seeds:
        directory=out/f'child-{mode}-{seed}';baseline=a1/f'{mode}-{seed}'
        _,base=summarize(baseline);frozen,summary=summarize(directory)
        record=next(x for x in r['frozen'] if x['mode']==mode and x['seed']==seed)
        assert summary==record['summary'] and compare(baseline,directory)==record['paired']
        assert frozen['checkpoint_sha256']==sha(cp)
        for name,key in [('report.json','report_sha256'),('observations.npz','observations_sha256'),('adhesion-observations.npz','adhesion_sha256')]:
            assert sha(directory/name)==record[key]
        bases[mode][seed]=base;children[mode][seed]=summary
        metrics=('standing_fraction','mean_stance_slip_mm_s','heading_forward_distance_mm','net_forward_displacement_mm','total_leg_steps')
        row=dict(mode=mode,seed=seed,parent={k:base[k] for k in metrics},child={k:summary[k] for k in metrics},
                 paired_gain={k:summary[k]-base[k] for k in metrics},locomotion_regime=summary['locomotion_regime'])
        rows.append(row)
assert exact(out/'train-legacy-1004/exploration-log.npz',out/'train-corrected-1004/exploration-log.npz')
for job in r['jobs']:
    directory=out/job['name'];report=read(directory/'report.json');count=5000 if job['phase']=='train' else 1000
    with np.load(directory/'adhesion-observations.npz',allow_pickle=False) as z:
        assert z['desired'].shape==(count,42) and z['pads'].shape==z['Femur_delta'].shape==(count,6)
        np.testing.assert_array_equal((z['desired']-report['body']['neutral_pose']).reshape(count,6,7)[:,:,3],z['Femur_delta'])
        np.testing.assert_array_equal(z['pads'],((z['Femur_delta'] if job['mode']=='legacy' else -z['Femur_delta'])<=.08).astype(float))
        assert all(np.isfinite(z[k]).all() for k in z.files)
gates={m:standing_retention(bases[m],children[m],seeds) for m in modes};assert gates==r['stage_gates']
limits=['One paired training birth, not four independent training replications.',
        'Four reused development seeds and five-second frozen horizon; final validation unused.',
        'Standing-stage screen is not walking, perturbation recovery, environmental response or lifetime retention.',
        'Pad mapping uses neural desired angles; biological adhesion-force control remains unvalidated.',
        'No automatic model promotion, weight merging, learner/reward change or duration/rate retune.',
        'A failed screen does not prove absence of learning or identify the cause of every earlier failure.']
audit=dict(status='completed_audited_S2',code_commit='1c3be20a2eb5f0623cff25de09f3e36ec64d4759',seconds=r['seconds'],
           births=births,cases=rows,stage_gates=gates,all_bridges_exact=True,all_pad_formulas_exact=True,
           exploration_streams_exact=True,all_worker_trees_empty=True,source_parent_data_unchanged=True,
           lease_released=True,model_promoted=False,walking_proven=False,limits=limits)
audit_path=root/'outputs/standing-reacquisition-s2-20261106-audit.json'
audit_path.write_text(json.dumps(audit,indent=2,allow_nan=False),encoding='utf-8')
lines=['# S2: paired standing reacquisition','',f"Twelve full jobs completed in {r['seconds']:.3f}s. Two exact parent bridges, two25s/4990update births, eight5s frozen probes. All technical gates passed; unchanged source/parent/data; all contained workers empty and lease released.",'',
       '|Mapping|Seed|Parent standing %|Child standing %|Standing change pp|Slip gain mm/s|Heading gain mm|Child regime|',
       '|---|---|---:|---:|---:|---:|---:|---|']
for row in rows:
    lines.append(f"|{row['mode']}|{row['seed']}|{100*row['parent']['standing_fraction']:.4f}|{100*row['child']['standing_fraction']:.4f}|{100*row['paired_gain']['standing_fraction']:.4f}|{row['paired_gain']['mean_stance_slip_mm_s']:.5f}|{row['paired_gain']['heading_forward_distance_mm']:.5f}|{row['locomotion_regime']}|")
lines+=['','Registered standing-stage screens:','','```json',json.dumps(gates,indent=2),'```','','Limits:','']+['- '+x for x in limits]+['']
report_path=root/'outputs/standing-reacquisition-s2-20261106-audit.md';report_path.write_text('\n'.join(lines),encoding='utf-8')
files=[out/'protocol.json',out/'result.json',audit_path,report_path]
for job in r['jobs']:
    directory=out/job['name'];files.extend([out/(job['name']+'-receipt.json'),out/(job['name']+'-context.json'),out/(job['name']+'.log')])
    for name in ('report.json','adhesion-observations.npz'):
        files.append(directory/name)
    if job['phase']=='train':
        files.extend(directory/name for name in ('training-observations.npz','exploration-log.npz','weights-last.npz','weights-last.model.json'))
    else:files.append(directory/'observations.npz')
entries={f.relative_to(root/'outputs').as_posix():f.read_bytes() for f in files}
entries['common-parent/weights.npz']=parent.read_bytes()
entries['runtime-source-manifest.json']=(root/'outputs/thermo-pilot-tp2-20261106-source.json').read_bytes()
for name in ('adhesion_mapping.py','s2_contract.py','s2_worker.py','run_standing_s2.py','analyse_standing_s2.py','test_s2_contract.py'):
    entries['scripts/'+name]=(root/'work'/name).read_bytes()
manifest={n:dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest()) for n,b in entries.items()}
archive=root/'outputs/standing-reacquisition-s2-20261106-evidence.zip';assert not archive.exists()
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for n,b in entries.items():z.writestr(n,b)
    z.writestr('manifest.json',json.dumps(manifest,indent=2))
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None and set(z.namelist())==set(entries)|{'manifest.json'}
    assert all(hashlib.sha256(z.read(n)).hexdigest()==v['sha256'] for n,v in manifest.items())
receipt=dict(path=archive.name,bytes=archive.stat().st_size,sha256=sha(archive),files=len(entries),
             crc_and_all_hashes_verified=True,raw_graph_cache_private_lease_excluded=True,release_uploaded=False)
(root/'outputs/standing-reacquisition-s2-20261106-artifact.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
print(json.dumps(dict(stage_gates=gates,artifact=receipt),indent=2))
