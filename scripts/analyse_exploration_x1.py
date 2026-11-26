"""Recompute X1 cells/interaction, independently audit transport, sign evidence archive."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
from x1_drive import physical_summary,transfer_interaction

root=Path(__file__).resolve().parents[1];out=root/'outputs/exploration-transfer-x1-20261106';p0=root/'outputs/no-plasticity-p0-20261106'
sys.path[:0]=[str(root/'work/Noesis-Sensory/gpu-pilot/src'),str(root/'work/Noesis-Core/src')]
from noesis_gpu.evaluate import summarize,MATCHED_SETTINGS


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def exact(a,b,prefix=None):
    with np.load(a,allow_pickle=False) as x,np.load(b,allow_pickle=False) as y:
        assert set(x.files)==set(y.files)
        for k in x.files:np.testing.assert_array_equal(x[k],y[k] if prefix is None else y[k][:prefix],err_msg=k)


p=read(out/'protocol.json');r=read(out/'result.json')
assert r['status']=='completed_X1' and len(r['jobs'])==9 and len(r['cases'])==8
for k in ('source_parent_prior_unchanged','data_unchanged','process_tree_empty_verified','lease_released'):assert r[k]
assert len(r['gates'])==3 and all(r['gates'].values())
assert not (root/'work/Project-Noesis/gpu-pilot/outputs/codex-training-host.lock').exists()
for folder,files in p['source_hashes'].items():
    base=root/('work' if folder=='scripts' else folder)
    if folder!='scripts':assert len(list(base.rglob('*.py')))==len(files)
    for n,h in files.items():assert sha(base/n)==h
for n,h in p['prior_hashes'].items():assert sha(root/n)==h
for f in read(root/'outputs/noesis-required-data-sha256.json')['files']:
    assert sha(root/'work/NoesisData'/f['path'])==f['sha256']
cells={m:{} for m in ('legacy','corrected')};rows=[];reports={};seed_stream=None
for job in r['jobs']:
    name=job['name'];directory=out/name;report=read(directory/'report.json');reports[name]=report
    receipt=read(out/(name+'-receipt.json'));assert receipt==job['receipt']
    assert receipt['exit_code']==0 and receipt['tree_empty'] and not receipt['timed_out'] and job['available_ram_before_gib']>=2
    assert report['status']=='completed_transport_pilot' and report['completed_control_chunks']==1000 and report['simulated_seconds_per_world']==5
    assert report['seeds']==[1004] and report['weights_unchanged'] and report['learning'] is False
    summary=physical_summary(directory)
    if job['drive']=='off':
        _,native=summarize(directory);assert all(summary[k]==native[k] for k in summary)
    else:
        try:summarize(directory)
        except ValueError:pass
        else:raise AssertionError('Current-on exposure accepted as ordinary frozen probe')
    if job['kind']=='native':continue
    bank,mode,drive=job['bank'],job['mode'],job['drive']
    record=next(c for c in r['cases'] if c['name']==name)
    assert summary==record['summary'] and report['checkpoint_sha256']==record['checkpoint_sha256']
    for file,key in [('report.json','report_sha256'),('observations.npz','observations_sha256'),('adhesion-observations.npz','pads_sha256'),
                     ('exploration-log.npz','exploration_sha256'),('latent-and-seeds.npz','latent_sha256')]:
        assert sha(directory/file)==record[key]
    assert report['control_steps']==990 and report['learning_updates']==0 and report['network_weights_readonly'] and report['full_network_weights_unchanged']
    assert report['motor_exploration']==(1.5 if drive=='on' else 0)
    with np.load(directory/'latent-and-seeds.npz',allow_pickle=False) as z,np.load(p0/f'control-{mode}/exploration-log.npz',allow_pickle=False) as old:
        if seed_stream is None:seed_stream=z['chunk_seeds'].copy()
        assert seed_stream.shape==(1000,)
        np.testing.assert_array_equal(z['chunk_seeds'],seed_stream)
        np.testing.assert_array_equal(z['latent_drive'],old['drive'][:990])
        with np.load(directory/'exploration-log.npz',allow_pickle=False) as actual:
            assert set(actual.files)==set(old.files)
            for k in old.files:
                expected=old[k][:990] if k in ('drive','credit','control_chunk_index') else old[k]
                if k=='drive' and drive=='off':expected=np.zeros_like(expected)
                np.testing.assert_array_equal(actual[k],expected,err_msg=k)
    with np.load(directory/'adhesion-observations.npz',allow_pickle=False) as z:
        assert z['desired'].shape==(1000,42) and z['pads'].shape==z['Femur_delta'].shape==(1000,6)
        np.testing.assert_array_equal((z['desired']-report['body']['neutral_pose']).reshape(1000,6,7)[:,:,3],z['Femur_delta'])
        np.testing.assert_array_equal(z['pads'],((z['Femur_delta'] if mode=='legacy' else -z['Femur_delta'])<=.08).astype(float))
    if bank=='parent' and drive=='on':
        with np.load(directory/'observations.npz',allow_pickle=False) as a,np.load(p0/f'control-{mode}/training-observations.npz',allow_pickle=False) as b:
            for k in a.files:np.testing.assert_array_equal(a[k][:,0],b[k][:1000])
        exact(directory/'adhesion-observations.npz',p0/f'control-{mode}/adhesion-observations.npz',1000)
    if bank=='parent' and mode=='legacy' and drive=='off':
        for f in ('observations.npz','adhesion-observations.npz'):exact(directory/f,out/f'native-parent-legacy-off/{f}')
    cells[mode][bank,drive]=summary;rows.append(dict(mode=mode,bank=bank,drive=drive,summary=summary))
for mode in cells:
    selected=[v for v in reports.values() if v['body']['adhesion_mapping']==('legacy-positive-Femur' if mode=='legacy' else 'corrected-negative-Femur')]
    assert all(all(x[k]==selected[0][k] for k in MATCHED_SETTINGS) for x in selected)
interaction={m:transfer_interaction(cells[m]) for m in cells};assert interaction==r['interaction']
onsets=[]
for mode in cells:
    for bank in ('parent','child'):
        on=out/f'probe-{bank}-{mode}-on';off=out/f'probe-{bank}-{mode}-off'
        onset=dict(mode=mode,bank=bank,posthoc_descriptive=True)
        for file,key,label in [('adhesion-observations.npz','desired','desired'),('observations.npz','angles','physical_angles')]:
            with np.load(on/file,allow_pickle=False) as a,np.load(off/file,allow_pickle=False) as b:
                np.testing.assert_array_equal(a[key][:10],b[key][:10])
                different=np.flatnonzero(np.any(a[key]!=b[key],axis=tuple(range(1,a[key].ndim))))
                onset['first_'+label+'_different_chunk']=int(different[0]) if len(different) else None
        onsets.append(onset)
limits=['One reused development seed1004 and one S2training birth per mapping; no training-seed replication.',
        'Current-on is fixed-weight exposure, not an ordinary frozen/no-exploration validation.',
        'Interaction is descriptive; neither current benefit nor difference of differences proves learned walking or generalization.',
        'S2retention screens remain failed; no model promoted, weight merge, new training or rule change.',
        'All inputs/core/body matched within a mapping except the declared weight/current interventions.',
        'First divergence chunks are posthoc transport descriptions, not a fitted learning-credit delay.']
audit=dict(status='completed_audited_X1',code_commit='89abec5894e0ccdd38c75aed0a9fa515b9b43a52',seconds=r['seconds'],
           cases=rows,interaction=interaction,posthoc_transport_onsets=onsets,all_three_bridges_exact=True,
           all_latent_and_neural_seed_streams_exact=True,all_pad_formulas_exact=True,all_weights_readonly_unchanged=True,
           ordinary_frozen_evaluator_rejects_current_on=True,source_parent_prior_data_unchanged=True,
           all_jobs_exit0_empty=True,lease_released=True,model_promoted=False,walking_proven=False,limits=limits)
ap=root/'outputs/exploration-transfer-x1-20261106-audit.json';ap.write_text(json.dumps(audit,indent=2,allow_nan=False),encoding='utf-8')
lines=['# X1: fixed-weight current withdrawal','',f"Nine full5s jobs complete in{r['seconds']:.3f}s. All bridges/latent drive/seeds/pad formulas/source/data/weight checks pass; exit0, empty trees, lease released.",'',
       '|Policy|Bank|Current|Standing %|Slip mm/s|Heading mm|Net-forward mm|Cycles|Regime|',
       '|---|---|---|---:|---:|---:|---:|---:|---|']
for row in rows:
    s=row['summary'];lines.append(f"|{row['mode']}|{row['bank']}|{row['drive']}|{100*s['standing_fraction']:.4f}|{s['mean_stance_slip_mm_s']:.5f}|{s['heading_forward_distance_mm']:.5f}|{s['net_forward_displacement_mm']:.5f}|{s['total_leg_steps']}|{s['locomotion_regime']}|")
lines+=['','Descriptive paired interactions:','','```json',json.dumps(interaction,indent=2),'```','','Limits:','']+['- '+x for x in limits]+['']
mp=root/'outputs/exploration-transfer-x1-20261106-audit.md';mp.write_text('\n'.join(lines),encoding='utf-8')
files=[out/'protocol.json',out/'result.json',ap,mp]
for job in r['jobs']:
    name=job['name'];directory=out/name
    files.extend(out/(name+suffix) for suffix in ('-receipt.json','-context.json','.log'))
    files.extend(directory/f for f in ('report.json','observations.npz','adhesion-observations.npz'))
    if job['kind']=='probe':files.extend(directory/f for f in ('exploration-log.npz','latent-and-seeds.npz'))
entries={f.relative_to(root/'outputs').as_posix():f.read_bytes() for f in files}
for mode in ('legacy','corrected'):
    cp=root/f'outputs/standing-reacquisition-s2-20261106/train-{mode}-1004/weights-last.npz'
    entries[f'checkpoints/child-{mode}.npz']=cp.read_bytes()
parent=root/'work/evidence/s1-restored/gpu-pilot/outputs/e4-exploration-20261029/train-control-1000/weights-last.npz'
entries['checkpoints/common-parent.npz']=parent.read_bytes()
entries['runtime-source-manifest.json']=(root/'outputs/thermo-pilot-tp2-20261106-source.json').read_bytes()
for name in ('x1_drive.py','x1_worker.py','run_exploration_x1.py','analyse_exploration_x1.py','p0_control.py','s2_worker.py','s2_contract.py','adhesion_mapping.py','test_x1_drive.py'):
    entries['scripts/'+name]=(root/'work'/name).read_bytes()
manifest={n:dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest()) for n,b in entries.items()}
archive=root/'outputs/exploration-transfer-x1-20261106-evidence.zip';assert not archive.exists()
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for n,b in entries.items():z.writestr(n,b)
    z.writestr('manifest.json',json.dumps(manifest,indent=2))
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None and set(z.namelist())==set(entries)|{'manifest.json'}
    assert all(hashlib.sha256(z.read(n)).hexdigest()==v['sha256'] for n,v in manifest.items())
receipt=dict(path=archive.name,bytes=archive.stat().st_size,sha256=sha(archive),files=len(entries),
             crc_and_all_hashes_verified=True,raw_graph_cache_private_lease_excluded=True,release_uploaded=False)
(root/'outputs/exploration-transfer-x1-20261106-artifact.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
print(json.dumps(dict(interaction=interaction,artifact=receipt),indent=2))
