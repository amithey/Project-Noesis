"""Independent raw P0 audit and selected, signed evidence packaging."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import numpy as np

root=Path(__file__).resolve().parents[1];out=root/'outputs/no-plasticity-p0-20261106'
s2=root/'outputs/standing-reacquisition-s2-20261106'
sys.path[:0]=[str(root/'work/Noesis-Sensory/gpu-pilot/src'),str(root/'work/Noesis-Core/src')]
from noesis_gpu.retention_audit import compare_checkpoints


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
def exact(a,b,prefix=None):
    with np.load(a,allow_pickle=False) as x,np.load(b,allow_pickle=False) as y:
        assert set(x.files)==set(y.files)
        for k in x.files:
            a,b=x[k],y[k]
            if prefix is not None:a,b=a[:prefix],b[:prefix]
            np.testing.assert_array_equal(a,b,err_msg=k)


r=read(out/'result.json');p=read(out/'protocol.json')
assert r['status']=='completed_P0' and len(r['jobs'])==3 and r['active_bridge_exact']
for key in ('source_parent_prior_unchanged','data_unchanged','process_tree_empty_verified','lease_released'):assert r[key]
assert not (root/'work/Project-Noesis/gpu-pilot/outputs/codex-training-host.lock').exists()
for folder,files in p['source_hashes'].items():
    if folder=='scripts':
        for name,value in files.items():assert sha(root/'work'/name)==value
    else:
        directory=root/folder;assert len(list(directory.rglob('*.py')))==len(files)
        for name,value in files.items():assert sha(directory/name)==value
for mode,files in p['prior_S2_hashes'].items():
    for name,value in files.items():assert sha(s2/f'train-{mode}-1004'/name)==value
for f in read(root/'outputs/noesis-required-data-sha256.json')['files']:
    assert sha(root/'work/NoesisData'/f['path'])==f['sha256']
parent=root/'work/evidence/s1-restored/gpu-pilot/outputs/e4-exploration-20261029/train-control-1000/weights-last.npz'
assert sha(parent)==p['parent_sha256']
rows=[]
for job in r['jobs']:
    name=job['name'];directory=out/name;mode=job['mode'];arm=job['arm']
    receipt=read(out/(name+'-receipt.json'));assert receipt==job['receipt']
    assert receipt['exit_code']==0 and receipt['tree_empty'] and not receipt['timed_out'] and job['available_ram_before_gib']>=2
    report=read(directory/'report.json');assert report['completed_chunks']==5000 and report['simulated_seconds']==25
    assert report['seed']==1004 and report['adhesion_probe_mode']==mode and report['import_checkpoint_sha256']==p['parent_sha256']
    with np.load(directory/'training-observations.npz',allow_pickle=False) as z:
        assert all(len(z[k])==5000 and np.isfinite(z[k]).all() for k in z.files)
        np.testing.assert_array_equal(z['learning_active'],np.arange(5000)>=10)
    for name in ('training-observations.npz','adhesion-observations.npz'):
        exact(directory/name,s2/f'train-{mode}-1004'/name,None if arm=='active' else 11)
    exact(directory/'exploration-log.npz',s2/f'train-{mode}-1004/exploration-log.npz')
    with np.load(directory/'exploration-log.npz',allow_pickle=False) as z:
        assert z['drive'].shape[0]==z['credit'].shape[0]==4990 and all(np.isfinite(z[k]).all() for k in z.files)
        np.testing.assert_array_equal(z['control_chunk_index'],np.arange(10,5000))
    with np.load(directory/'adhesion-observations.npz',allow_pickle=False) as z:
        assert z['desired'].shape==(5000,42) and z['pads'].shape==z['Femur_delta'].shape==(5000,6)
        np.testing.assert_array_equal((z['desired']-report['body']['neutral_pose']).reshape(5000,6,7)[:,:,3],z['Femur_delta'])
        np.testing.assert_array_equal(z['pads'],((z['Femur_delta'] if mode=='legacy' else -z['Femur_delta'])<=.08).astype(float))
    if arm=='active':
        assert sha(directory/'weights-last.npz')==p['prior_S2_hashes'][mode]['weights-last.npz']
        assert report['learning_updates']==4990 and report['status']=='completed_training_birth'
        continue
    assert report['status']=='completed_no_plasticity_control' and report['learning_updates']==0 and report['control_steps']==4990
    assert report['full_network_weights_unchanged'] and report['network_weights_readonly'] and not report['learning'] and not report['weights_changed']
    assert not (directory/'weights-last.npz').exists()
    cp=directory/'weights-control.npz';check=compare_checkpoints(parent,cp)
    assert check==r['controls'][mode]['checkpoint_audit'] and check['weights']['changed_count']==0 and check['compatible']
    with np.load(parent,allow_pickle=False) as a,np.load(cp,allow_pickle=False) as b:
        for key in ('version','normalize_reward_error','bias_correct_reward_error','weights','initial_weights','motor_indices','topology','parameters','rule'):
            np.testing.assert_array_equal(a[key],b[key])
    sidecar=read(cp.with_suffix('.model.json'))
    for key,value in report.items():assert sidecar[key]==value
    assert sha(cp)==report['control_checkpoint_sha256']==r['controls'][mode]['checkpoint_sha256']
    metrics={}
    with np.load(directory/'training-observations.npz',allow_pickle=False) as z,np.load(s2/f'train-{mode}-1004/training-observations.npz',allow_pickle=False) as a:
        for window,sel in [('all_after_birth',slice(10,None)),('last5s',slice(4000,None))]:
            metrics[window]={k:dict(active=float(a[k][sel].mean()),control=float(z[k][sel].mean()),
                active_minus_control=float(a[k][sel].mean()-z[k][sel].mean())) for k in ('standing','reward','stance_slip_mm_s')}
    assert metrics==r['controls'][mode]['metrics']
    rows.append(dict(mode=mode,metrics=metrics,checkpoint_sha256=sha(cp)))
limits=['One seed1004 exposure per policy; not independent training-seed replication.',
        'Standing/reward/slip here describe25s with exploration, not frozen retained ability.',
        'S2 frozen retention failures remain authoritative; no new candidate or promotion.',
        'Control raw learning_active row field is inherited exploration/call timing; actual plasticity updates are zero.',
        'No change in reward/learner/controller defaults or claim of stable walking/balance/environment response.',
        'Read-only weights and no learner.update also freeze learning traces; those traces only affect neural output through weight changes in this source.']
audit=dict(status='completed_audited_P0',code_commit='5fdcdb37ac4a80cecdcb1ef7b66932d99925a118',seconds=r['seconds'],
           active_bridge_exact=True,controls=rows,control_full_weights_unchanged=True,all_streams_exact=True,
           all_prefixes_exact=True,source_parent_prior_data_unchanged=True,all_jobs_exit0_empty=True,
           lease_released=True,walking_proven=False,model_promoted=False,limits=limits)
ap=root/'outputs/no-plasticity-p0-20261106-audit.json';ap.write_text(json.dumps(audit,indent=2,allow_nan=False),encoding='utf-8')
lines=['# P0: active learning versus unchanged-weight exposure','',
       f"Three full25s jobs complete in{r['seconds']:.3f}s. Active bridge checkpoint/body/pad/drive replay exact. Control weights read-only and unchanged,0updates,4990exploration rows; source/parent/data/S2 unchanged. All jobs exit0, trees empty and canonical lease released.",'',
       '|Policy|Window|Active standing %|Control standing %|Active minus control pp|Reward difference|Slip difference mm/s|',
       '|---|---|---:|---:|---:|---:|---:|']
for row in rows:
    for window,metrics in row['metrics'].items():
        standing=metrics['standing']
        lines.append(f"|{row['mode']}|{window}|{100*standing['active']:.4f}|{100*standing['control']:.4f}|{100*standing['active_minus_control']:.4f}|{metrics['reward']['active_minus_control']:.6f}|{metrics['stance_slip_mm_s']['active_minus_control']:.6f}|")
lines+=['','Limits:','']+['- '+x for x in limits]+['']
mp=root/'outputs/no-plasticity-p0-20261106-audit.md';mp.write_text('\n'.join(lines),encoding='utf-8')
files=[out/'protocol.json',out/'result.json',ap,mp]
for job in r['jobs']:
    directory=out/job['name'];name=job['name']
    files.extend(out/(name+suffix) for suffix in ('-receipt.json','-context.json','-p0-context.json','.log'))
    files.extend(directory/name for name in ('report.json','training-observations.npz','exploration-log.npz','adhesion-observations.npz'))
    cp='weights-last' if job['arm']=='active' else 'weights-control'
    files.extend([directory/(cp+'.npz'),directory/(cp+'.model.json')])
entries={f.relative_to(root/'outputs').as_posix():f.read_bytes() for f in files}
entries['common-parent/weights.npz']=parent.read_bytes()
entries['runtime-source-manifest.json']=(root/'outputs/thermo-pilot-tp2-20261106-source.json').read_bytes()
for name in ('p0_control.py','p0_worker.py','run_no_plasticity_p0.py','analyse_no_plasticity_p0.py','s2_worker.py','s2_contract.py','adhesion_mapping.py','test_p0_control.py'):
    entries['scripts/'+name]=(root/'work'/name).read_bytes()
manifest={n:dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest()) for n,b in entries.items()}
archive=root/'outputs/no-plasticity-p0-20261106-evidence.zip';assert not archive.exists()
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for n,b in entries.items():z.writestr(n,b)
    z.writestr('manifest.json',json.dumps(manifest,indent=2))
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None and set(z.namelist())==set(entries)|{'manifest.json'}
    assert all(hashlib.sha256(z.read(n)).hexdigest()==v['sha256'] for n,v in manifest.items())
receipt=dict(path=archive.name,bytes=archive.stat().st_size,sha256=sha(archive),files=len(entries),
             crc_and_all_hashes_verified=True,raw_graph_cache_private_lease_excluded=True,release_uploaded=False)
(root/'outputs/no-plasticity-p0-20261106-artifact.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
print(json.dumps(dict(controls=rows,artifact=receipt),indent=2))
