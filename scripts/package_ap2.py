"""Selected signed local AP2 evidence; extensive integrity maps stay local."""
import json,zipfile
import run_anatomical_ap2 as study
import audit_ap2 as audit

root,out=study.root,study.out
read,write,sha=study.read,study.write,study.sha


def main():
    summary_path=root/'outputs/anatomical-feedback-ap2-20261107-summary.json';summary=read(summary_path)
    assert summary['status']=='audited_complete_AP2' and summary['all_full_horizons_receipts_exit_zero_trees_empty']
    selected={summary_path,root/'outputs/anatomical-feedback-ap2-20261107.md',root/'outputs/anatomical-feedback-ap2-public-protocol-20261107.json'}
    selected|={p for p in out.iterdir() if p.is_file() and p.suffix in ('.json','.log')}
    for d in out.iterdir():
        if d.is_dir():selected|={p for p in d.iterdir() if p.is_file() and p.suffix in ('.npz','.json')}
    directories={audit.directory(birth,bank,seed) for birth in (1004,1005,1006) for bank in ('parent','control','candidate') for seed in (birth,*study.DEV)}
    for d in directories:
        selected|={d/n for n in ('report.json','observations.npz','adhesion-observations.npz')}
        for n in ('anatomy-observations.npz','anatomy-model.json','body-telemetry.npz','ground-contacts.npz','telemetry-model.json'):
            if (d/n).exists():selected.add(d/n)
    for d in study.TRAIN.values():selected|={p for p in d.iterdir() if p.is_file() and p.suffix in ('.npz','.json')}
    selected|={study.C/'result.json',root/'outputs/anatomical-feedback-ap1c-20261106-summary.json'}
    for n in ('ap2_worker.py','ap2_contract.py','run_anatomical_ap2.py','test_ap2_contract.py','audit_ap2.py','package_ap2.py',
        'ap1c_body.py','support_descriptor_c1.py','ap1_feedback.py','run_anatomical_ap1b.py','audit_ap1c.py','s2_worker.py','s2_contract.py','h2_telemetry.py','adhesion_mapping.py'):
        selected.add(root/'work'/n)
    manifest={p.relative_to(root).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(selected)}
    assert all('/brain-cache/' not in n and '/NoesisData/' not in n and not n.endswith('.lock') for n in manifest)
    dest=root/'outputs/anatomical-feedback-ap2-20261107-evidence.zip'
    with zipfile.ZipFile(dest,'x',zipfile.ZIP_DEFLATED) as z:
        for n,entry in manifest.items():
            assert sha(root/n)==entry['sha256'];z.write(root/n,n)
        z.writestr('manifest.json',json.dumps(manifest,indent=2))
    with zipfile.ZipFile(dest) as z:
        assert z.testzip() is None
        import hashlib
        for n,entry in manifest.items():assert hashlib.sha256(z.read(n)).hexdigest()==entry['sha256']
    receipt=dict(path=dest.name,bytes=dest.stat().st_size,sha256=sha(dest),files=len(manifest),crc_and_all_hashes_verified=True,raw_graph_cache_private_lease_excluded=True,release_uploaded=False)
    write(root/'outputs/anatomical-feedback-ap2-20261107-artifact.json',receipt);print(json.dumps(receipt))


if __name__=='__main__':main()
