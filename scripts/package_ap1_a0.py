"""Package selected AP1A0 evidence with POSIX paths; excludes graph/cache/lease."""
import hashlib,json,zipfile
from pathlib import Path

root=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())

audit=root/'outputs/anatomical-feedback-ap1-a0-audit-20261106.json';a=read(audit)
selected={root/n for n,h in a['file_hashes'].items() if sha(root/n)==h}
assert len(selected)==len(a['file_hashes'])
selected|={audit,root/'outputs/anatomical-feedback-ap1-20261106.md',root/'outputs/anatomical-feedback-ap1-20261106-protocol.json'}
selected|={root/'work'/n for n in ('ap1_feedback.py','ap1_body.py','ap1_worker.py','run_anatomical_ap1.py','test_ap1_feedback.py','audit_ap1_a0.py','package_ap1_a0.py')}
manifest={p.relative_to(root).as_posix():dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(selected)}
dest=root/'outputs/anatomical-feedback-ap1-a0-20261106-evidence.zip'
with zipfile.ZipFile(dest,'x',zipfile.ZIP_DEFLATED) as z:
    for n,entry in manifest.items():
        data=(root/n).read_bytes();assert hashlib.sha256(data).hexdigest()==entry['sha256'];z.writestr(n,data)
    z.writestr('manifest.json',json.dumps(manifest,indent=2))
with zipfile.ZipFile(dest) as z:
    assert z.testzip() is None
    for n,entry in manifest.items():assert hashlib.sha256(z.read(n)).hexdigest()==entry['sha256']
receipt=dict(path=dest.name,bytes=dest.stat().st_size,sha256=sha(dest),files=len(manifest),crc_and_all_hashes_verified=True,
    raw_graph_cache_private_lease_excluded=True,release_uploaded=False)
(root/'outputs/anatomical-feedback-ap1-a0-20261106-artifact.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
print(json.dumps(receipt))
