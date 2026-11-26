"""Preserve failed H2 allocation and static fusion diagnosis before corrective source edits."""
import hashlib
import json
from pathlib import Path
import zipfile
root=Path(__file__).resolve().parents[1];out=root/'outputs/body-telemetry-h2-20261106'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
r=read(out/'result.json');p=read(out/'protocol.json');static=read(out/'static-body-map.json')
assert r['status']=='failed' and not r['cases'] and len(r['jobs'])==1
assert r['jobs'][0]['receipt']['exit_code']==1 and r['jobs'][0]['receipt']['tree_empty']
assert r['source_prior_checkpoints_unchanged'] and r['data_unchanged'] and r['lease_released']
assert read(out/'parent-10007/report.json')['completed_control_chunks']==0
assert static['all_missing_names_restored_without_fusion'] and static['all_critical_observation_ids_valid'] and static['physical_integration_steps']==0
entries={}
for n in ('protocol.json','result.json','parent-10007.log','parent-10007-receipt.json','parent-10007-context.json','parent-10007-h2-context.json','parent-10007/report.json','static-body-map.json','static-body-map.log','static-body-map-receipt.json'):
    entries['a0/'+n]=(out/n).read_bytes()
for n in ('h2_telemetry.py','h2_worker.py','run_body_h2.py'):
    f=root/'work/h2-a0-source-20261106'/n;assert sha(f)==p['source_hashes']['work/'+n];entries['source-a0/'+n]=f.read_bytes()
entries['inspect_h2_body_map.py']=(root/'work/inspect_h2_body_map.py').read_bytes()
manifest={n:dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest()) for n,b in entries.items()}
archive=root/'outputs/body-telemetry-h2-a0-20261106-evidence.zip';assert not archive.exists()
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for n,b in entries.items():z.writestr(n,b)
    z.writestr('manifest.json',json.dumps(manifest,indent=2))
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None and set(z.namelist())==set(entries)|{'manifest.json'}
    assert all(hashlib.sha256(z.read(n)).hexdigest()==v['sha256'] for n,v in manifest.items())
report=dict(status='technical_failure_before_simulation',code_commit='4bde8da979265e7fcfff04350d86c4c82a26cd08',completed_cases=0,
    completed_physics_chunks=0,primary_error='metadata mass-layout rejected missing fused head ID -1',
    secondary_error='partial metadata writer assumed constructor completed',static_diagnosis=static,
    original_source_prior_checkpoint_data_unchanged=True,all_processes_empty=True,lease_released=True,
    scientific_interpretation=False,new_allocation_required=True,model_promoted=False,
    artifact=dict(path=archive.name,bytes=archive.stat().st_size,sha256=sha(archive),files=len(entries),crc_and_all_hashes_verified=True,release_uploaded=False))
(root/'outputs/body-telemetry-h2-a0-20261106.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
