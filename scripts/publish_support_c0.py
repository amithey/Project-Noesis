"""Audit C0 stored partitions and sign compact/static/offline calibration evidence."""
import hashlib,json,zipfile
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1];out=root/'outputs/support-calibration-c0-20261106'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
p=read(out/'protocol.json');r=read(out/'result.json');receipt=read(out/'receipt.json');cleanup=read(out/'cleanup.json')
assert r['status']=='completed_C0' and len(r['cases'])==3 and len(r['fixtures'])==6
assert receipt['exit_code']==0 and receipt['tree_empty'] and not receipt['timed_out'] and cleanup['lease_released'] and cleanup['tree_empty']
assert not (root/'work/Project-Noesis/gpu-pilot/outputs/codex-training-host.lock').exists()
for n,h in p['input_hashes'].items():assert sha(root/n)==h
for n,h in p['source_hashes'].items():assert sha(root/n)==h
expected={'airborne':'unsupported','foot_only':'tarsal_only','body_only':'central_body_contact','mixed_body_feet':'central_body_contact','proximal_leg':'proximal_leg_contact','other_tarsal':'tarsal_only'}
for f in r['fixtures']:
    assert f['descriptor']['contact_mode']==expected[f['kind']] and f['model_time']==0 and f['physics_integration_steps']==f['neural_steps']==0
    assert f['root_force_closure_max_error']<1e-9 and not f['descriptor']['physical_standing_validated']
for c in r['cases']:
    file=out/(c['name']+'-descriptors.npz');assert sha(file)==c['output_sha256']
    with np.load(file,allow_pickle=False) as z:
        assert z['contact_mode_code'].shape==(1000,) and z['region_vertical_reaction'].shape==(1000,4)
        assert np.isfinite(z['region_vertical_reaction']).all() and np.all(z['region_vertical_reaction']>=0)
        for group,mask in {'all_measured':np.arange(1000)>=10,'standing':z['legacy_standing']&(np.arange(1000)>=10),'high_only':z['legacy_high_only']&(np.arange(1000)>=10)}.items():
            counts={mode:int(np.count_nonzero(z['contact_mode_code'][mask]==i)) for i,mode in enumerate(r['contact_mode_codebook'])}
            assert counts==c['groups'][group]['contact_mode_counts']
        assert float(z['legacy_standing'][10:].mean())==c['legacy_standing_fraction']
    assert c['reaction_partition_max_error']<=1e-9
compact={k:r[k] for k in ('status','seconds','contact_mode_codebook','region_columns','force_tolerance','unchanged_legacy_mask_and_verdicts','input_source_unchanged','physics_integration_steps','neural_steps','fixture_positives_are_not_fly_or_learning_validation','new_standing_gate','new_reward','model_promoted','walking_proven','limits')}
compact.update(code_commit='5014f50d78771cfc3b180235ab8cd485b2f5910e',independent_stored_partition_audit_passed=True,
    fixtures=[dict(kind=f['kind'],mode=f['descriptor']['contact_mode'],time=f['model_time'],root_force_closure_error=f['root_force_closure_max_error']) for f in r['fixtures']],cases=r['cases'])
ap=root/'outputs/support-calibration-c0-20261106-audit.json';ap.write_text(json.dumps(compact,indent=2,allow_nan=False),encoding='utf-8')
lines=['# C0: anatomical support description','',f"Six static fixtures and all3H2Btraces checked in{r['seconds']:.3f}s worker time; eight unit checks passed. Source/input/force/partition/original-mask/count checks pass; exit0/empty, lease released. No neural/physics integration, standing gate or reward change.",'',
    '|Case|Group|Samples|Unsupported|Central-body contact|Proximal-leg contact|Tarsal-only|',
    '|---|---|---:|---:|---:|---:|---:|']
for c in r['cases']:
    for name in ('all_measured','standing','high_only'):
        g=c['groups'][name];x=g['contact_mode_counts']
        lines.append(f"|{c['name']}|{name}|{g['samples']}|{x['unsupported']}|{x['central_body_contact']}|{x['proximal_leg_contact']}|{x['tarsal_only']}|")
lines+=['','Contact modes are NOT standing/balance/learning validation. Tarsal-only inside old standing exists(383parent/447child1005samples); these are sampled states, not stable independent episodes. Alloldverdicts retained.','','Limits:','']+['- '+x for x in r['limits']]+['']
mp=root/'outputs/support-calibration-c0-20261106-audit.md';mp.write_text('\n'.join(lines),encoding='utf-8')
files=[out/n for n in ('protocol.json','result.json','receipt.json','cleanup.json','worker.log')]+[ap,mp]
files.extend(out/(c['name']+'-descriptors.npz') for c in r['cases']);files.extend(root/n for n in p['input_hashes'])
entries={f.relative_to(root).as_posix():f.read_bytes() for f in files}
for n in ('support_descriptor_c0.py','run_support_c0.py','publish_support_c0.py','test_support_c0.py','c0-protocol.md','h2_telemetry.py','height_audit_h1.py'):
    entries['scripts/'+n]=(root/'work'/n).read_bytes()
manifest={n:dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest()) for n,b in entries.items()}
archive=root/'outputs/support-calibration-c0-20261106-evidence.zip';assert not archive.exists()
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for n,b in entries.items():z.writestr(n,b)
    z.writestr('manifest.json',json.dumps(manifest,indent=2))
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None and set(z.namelist())==set(entries)|{'manifest.json'}
    assert all(hashlib.sha256(z.read(n)).hexdigest()==v['sha256'] for n,v in manifest.items())
artifact=dict(path=archive.name,bytes=archive.stat().st_size,sha256=sha(archive),files=len(entries),crc_and_all_hashes_verified=True,
    raw_graph_cache_private_lease_excluded=True,release_uploaded=False)
(root/'outputs/support-calibration-c0-20261106-artifact.json').write_text(json.dumps(artifact,indent=2),encoding='utf-8')
print(json.dumps(dict(cases=3,fixtures=6,artifact=artifact),indent=2))
