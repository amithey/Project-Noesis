"""Package selected AP1B stop or completed AP1C evidence, with verified POSIX manifest."""
import argparse,hashlib,json,zipfile
from pathlib import Path

root=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=('stop','complete'),required=True);args=ap.parse_args()
    stop_path=root/'outputs/anatomical-feedback-ap1b-stop-audit-20261106.json';stop=read(stop_path)
    assert stop['status']=='audited_AP1B_technical_stop_no_scientific_verdict'
    selected=set()
    for n,h in stop['file_hashes'].items():assert sha(root/n)==h;selected.add(root/n)
    selected|={stop_path,root/'outputs/anatomical-coverage-ap1c-static-20261106.json',root/'outputs/anatomical-feedback-ap1c-20261106.md'}
    sources=('audit_ap1b_stop.py','support_descriptor_c0.py','support_descriptor_c1.py','ap1_body.py','ap1_worker.py','run_anatomical_ap1b.py',
        'ap1c_body.py','ap1c_worker.py','run_anatomical_ap1c.py','test_ap1c_coverage.py','package_ap1c.py','ap1_feedback.py','s2_worker.py','s2_contract.py','h2_telemetry.py')
    selected|={root/'work'/n for n in sources}
    label='anatomical-feedback-ap1b-stop-20261106'
    if args.phase=='complete':
        label='anatomical-feedback-ap1c-20261106';summary=root/'outputs'/f'{label}-summary.json'
        assert read(summary)['status']=='audited_complete_AP1C'
        selected|={summary,root/'work/audit_ap1c.py',root/'outputs/anatomical-feedback-ap1c-20261106-protocol.json'}
        folder=root/'outputs/anatomical-feedback-ap1c-20261106'
        selected|={p for p in folder.iterdir() if p.is_file() and p.suffix in ('.json','.log')}
        for d in folder.iterdir():
            if d.is_dir():selected|={p for p in d.iterdir() if p.is_file() and p.suffix in ('.npz','.json')}
        for case in ('parent-10007','child1005-10007'):
            d=root/'outputs/body-telemetry-h2b-20261106'/case
            selected|={d/n for n in ('report.json','observations.npz','adhesion-observations.npz','body-telemetry.npz','ground-contacts.npz','telemetry-model.json')}
        control=root/'outputs/anatomical-feedback-ap1-20261106/train-control-1005'
        selected|={p for p in control.iterdir() if p.is_file() and p.suffix in ('.npz','.json')}
    manifest={p.relative_to(root).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(selected)}
    assert all('/brain-cache/' not in n and '/NoesisData/' not in n and not n.endswith('.lock') for n in manifest)
    dest=root/'outputs'/f'{label}-evidence.zip'
    with zipfile.ZipFile(dest,'x',zipfile.ZIP_DEFLATED) as z:
        for n,entry in manifest.items():
            data=(root/n).read_bytes();assert hashlib.sha256(data).hexdigest()==entry['sha256'];z.writestr(n,data)
        z.writestr('manifest.json',json.dumps(manifest,indent=2))
    with zipfile.ZipFile(dest) as z:
        assert z.testzip() is None
        for n,entry in manifest.items():assert hashlib.sha256(z.read(n)).hexdigest()==entry['sha256']
    receipt=dict(path=dest.name,bytes=dest.stat().st_size,sha256=sha(dest),files=len(manifest),crc_and_all_hashes_verified=True,raw_graph_cache_private_lease_excluded=True,release_uploaded=False)
    (root/'outputs'/f'{label}-artifact.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(json.dumps(receipt))


if __name__=='__main__':main()
