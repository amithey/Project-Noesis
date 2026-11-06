"""Recheck raw H1 aggregates and package explicit evidence with portable names."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
from height_audit_h1 import masks,intervals,quantiles

root=Path(__file__).resolve().parents[1];out=root/'outputs/height-audit-h1-20261106'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
p=read(out/'protocol.json');r=read(out/'result.json');receipt=read(out/'receipt.json');cleanup=read(out/'cleanup.json')
assert r['status']=='completed_H1' and len(r['cases'])==22
assert receipt['exit_code']==0 and not receipt['timed_out'] and receipt['tree_empty']
assert cleanup['lease_released'] and cleanup['process_tree_empty_verified']
assert not (root/'work/Project-Noesis/gpu-pilot/outputs/codex-training-host.lock').exists()
for name,h in p['input_hashes'].items():assert sha(root/name)==h
for name,h in p['source_hashes'].items():assert sha(root/name)==h
assert [x['exact_reward_rows'] for x in r['training_reward_replay']]==[4999,4999]
assert all(x['max_error']==0 for x in r['training_reward_replay'])
cases=[]
for c in r['cases']:
    with np.load(root/c['input_path']/'observations.npz',allow_pickle=False) as z:obs={k:z[k][:,0] for k in z.files}
    standing,high=masks(obs);high[:10]=False;ids=np.flatnonzero(high)
    assert float(standing[10:].mean())==c['standing_fraction'] and float(high[10:].mean())==c['high_only_fraction']
    height=obs['position'][:,2];claws=(obs['foot_load']>1e-5).sum(axis=-1);vz=np.r_[0.,np.diff(height)/.005]
    assert quantiles(height[high]-1.5)==c['high_only_excess_mm']
    assert quantiles(np.abs(vz[high]))==c['high_only_abs_vertical_speed_mm_s']
    assert quantiles(claws[high])==c['high_only_supporting_claws']
    bouts=intervals(high);assert len(bouts)==c['high_only_bout_count']
    cases.append(dict(bank=c['bank'],training_seed=c['training_seed'],evaluation_seed=c['evaluation_seed'],
        standing_fraction=c['standing_fraction'],high_only_fraction=c['high_only_fraction'],
        excess_median_mm=c['high_only_excess_mm']['median'] if len(ids) else None,
        excess_max_mm=c['high_only_excess_mm']['max'] if len(ids) else None,
        high_only_claws_median=c['high_only_supporting_claws']['median'] if len(ids) else None,
        high_only_ge3_claws_fraction=c['high_only_ge3_claws_fraction'],
        abs_vertical_speed_median_mm_s=c['high_only_abs_vertical_speed_mm_s']['median'] if len(ids) else None,
        longest_high_only_bout_seconds=max(((b-a)*.005 for a,b in bouts),default=0),
        high_only_reward_median=c['high_only_reward']['median'] if len(ids) else None,
        fixed_input_boundary_reward_jump=c['counterfactual_score_boundary']['below_minus_above_reward']))
compact=dict(status='completed_audited_H1',code_commit='f0437325ce1b4cf9e3f2df1276a6d10dc05908a4',seconds=r['seconds'],
    cases=cases,coordinate_facts=r['coordinate_facts'],training_reward_replay=r['training_reward_replay'],
    raw_aggregate_recheck_passed=True,input_and_source_unchanged=True,all_jobs_exit0_empty=True,lease_released=True,
    thresholds_changed=False,new_simulation=False,new_training=False,model_promoted=False,walking_proven=False,limits=r['limits'])
ap=root/'outputs/height-audit-h1-20261106-audit.json';ap.write_text(json.dumps(compact,indent=2,allow_nan=False),encoding='utf-8')
def fmt(x):return 'n/a' if x is None else f'{x:.4f}'
lines=['# H1: thorax height/contact/coordinate audit','',
       f"Offline worker{r['seconds']:.3f}s;22existing cases,9998stored training rewards replay exactly. Source/input/aggregate checks pass, exit0/empty, lease released. No model/dynamics/neural steps or threshold changes.",'',
       'Position is global thorax body-frame-origin z in mm; floor plane z=0. Contact norms are not whole-body COM/vertical support forces. Upper1.5mm gate is present in both physical mask and v7training reward. Fixed-input scorer boundary jumps1.5333..1.6 over2e-6mm; this is not a simulated trajectory. Existing pass/fail verdicts remain unchanged.','',
       '|Bank|Train seed|Eval seed|Standing %|High-only %|Excess median mm|>=3claws %|Median abs vz mm/s|Longest high-only s|',
       '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for c in cases:
    lines.append(f"|{c['bank']}|{c['training_seed']}|{c['evaluation_seed']}|{100*c['standing_fraction']:.4f}|{100*c['high_only_fraction']:.4f}|{fmt(c['excess_median_mm'])}|{fmt(None if c['high_only_ge3_claws_fraction'] is None else 100*c['high_only_ge3_claws_fraction'])}|{fmt(c['abs_vertical_speed_median_mm_s'])}|{c['longest_high_only_bout_seconds']:.4f}|")
lines+=['','Limits:','']+['- '+x for x in r['limits']]+['']
mp=root/'outputs/height-audit-h1-20261106-audit.md';mp.write_text('\n'.join(lines),encoding='utf-8')
files=[out/n for n in ('protocol.json','result.json','receipt.json','cleanup.json','worker.log')]+[ap,mp]
files.extend(root/n for n in p['input_hashes'])
entries={f.relative_to(root).as_posix():f.read_bytes() for f in files}
for name in ('height_audit_h1.py','publish_height_h1.py','test_height_audit_h1.py','h1-protocol.md'):
    entries['scripts/'+name]=(root/'work'/name).read_bytes()
entries['runtime-source-manifest.json']=(root/'outputs/thermo-pilot-tp2-20261106-source.json').read_bytes()
manifest={n:dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest()) for n,b in entries.items()}
archive=root/'outputs/height-audit-h1-20261106-evidence.zip';assert not archive.exists()
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for n,b in entries.items():z.writestr(n,b)
    z.writestr('manifest.json',json.dumps(manifest,indent=2))
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None and set(z.namelist())==set(entries)|{'manifest.json'}
    assert all(hashlib.sha256(z.read(n)).hexdigest()==v['sha256'] for n,v in manifest.items())
artifact=dict(path=archive.name,bytes=archive.stat().st_size,sha256=sha(archive),files=len(entries),
    crc_and_all_hashes_verified=True,raw_graph_cache_private_lease_excluded=True,release_uploaded=False)
(root/'outputs/height-audit-h1-20261106-artifact.json').write_text(json.dumps(artifact,indent=2),encoding='utf-8')
print(json.dumps(dict(cases=len(cases),reward_replayed_rows=9998,artifact=artifact),indent=2))
