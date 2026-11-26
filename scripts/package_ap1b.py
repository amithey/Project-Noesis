"""Signed selected AP1B evidence; never include data, graph caches or lease."""
import hashlib,json,zipfile
from pathlib import Path

root=Path(__file__).resolve().parents[1];out=root/'outputs/anatomical-feedback-ap1b-20261106'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

summary=read(root/'outputs/anatomical-feedback-ap1b-20261106-summary.json')
assert summary['status']=='audited_complete_AP1B' and summary['all_full_horizons_exit_zero_empty_tree']
selected={p for p in out.iterdir() if p.is_file() and p.suffix in ('.json','.log')}
for d in out.iterdir():
    if d.is_dir():selected|={p for p in d.iterdir() if p.is_file() and p.suffix in ('.npz','.json')}
for c in ('parent-10007','child1005-10007'):
    d=root/'outputs/body-telemetry-h2b-20261106'/c
    selected|={d/n for n in ('report.json','observations.npz','adhesion-observations.npz','body-telemetry.npz','ground-contacts.npz','telemetry-model.json')}
a0=read(root/'outputs/anatomical-feedback-ap1-a0-audit-20261106.json')
for n,h in a0['file_hashes'].items():assert sha(root/n)==h;selected.add(root/n)
selected|={root/'outputs'/n for n in ('anatomical-feedback-ap1b-20261106-summary.json','anatomical-feedback-ap1b-20261106-protocol.json',
    'anatomical-feedback-ap1-a0-audit-20261106.json','anatomical-feedback-ap1-a0-20261106-artifact.json','anatomical-feedback-ap1b-20261106.md')}
selected|={root/'work'/n for n in ('ap1_feedback.py','ap1_body.py','ap1_worker.py','run_anatomical_ap1.py','run_anatomical_ap1b.py',
    'test_ap1_feedback.py','test_ap1b_mass_gate.py','audit_ap1_a0.py','audit_ap1b.py','package_ap1b.py',
    's2_worker.py','s2_contract.py','adhesion_mapping.py','h2_telemetry.py','support_descriptor_c0.py')}
manifest={p.relative_to(root).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(selected)}
assert all('/brain-cache/' not in n and not n.endswith('.lock') and '/NoesisData/' not in n for n in manifest)
dest=root/'outputs/anatomical-feedback-ap1b-20261106-evidence.zip'
with zipfile.ZipFile(dest,'x',zipfile.ZIP_DEFLATED) as z:
    for n,entry in manifest.items():
        data=(root/n).read_bytes();assert hashlib.sha256(data).hexdigest()==entry['sha256'];z.writestr(n,data)
    z.writestr('manifest.json',json.dumps(manifest,indent=2))
with zipfile.ZipFile(dest) as z:
    assert z.testzip() is None
    for n,entry in manifest.items():assert hashlib.sha256(z.read(n)).hexdigest()==entry['sha256']
receipt=dict(path=dest.name,bytes=dest.stat().st_size,sha256=sha(dest),files=len(manifest),crc_and_all_hashes_verified=True,
    raw_graph_cache_private_lease_excluded=True,release_uploaded=False)
(root/'outputs/anatomical-feedback-ap1b-20261106-artifact.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
print(json.dumps(receipt))
