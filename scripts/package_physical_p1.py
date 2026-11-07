"""Seal selected P1 evidence locally. Does not upload private integrity maps."""
import hashlib,json,zipfile
import run_physical_p1 as study

root,out=study.root,study.out
read,write,sha=study.read,study.write,study.sha


def main():
    summary=root/'outputs/physical-calibration-p1-public-result-20261107.json'
    assert read(summary)['status']=='independently_audited_P1'
    selected={summary,root/'outputs/physical-calibration-p1-result-20261107.md',root/'outputs/physical-calibration-p1-public-protocol-20261107.json'}
    selected|={p for p in out.iterdir() if p.is_file() and p.suffix in ('.json','.log')}
    for d in out.iterdir():
        if d.is_dir():selected|={p for p in d.iterdir() if p.is_file() and p.suffix in ('.json','.npz')}
    for n in ('p1_worker.py','p1_telemetry.py','run_physical_p1.py','test_p1_contract.py','audit_physical_p1.py','package_physical_p1.py'):
        selected.add(root/'work'/n)
    for d in study.OLD.values():
        selected|={d/n for n in ('report.json','observations.npz','adhesion-observations.npz','anatomy-observations.npz','anatomy-model.json')}
    selected|=set(study.MODELS.values())
    manifest={p.relative_to(root).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(selected)}
    assert all('/NoesisData/' not in n and '/brain-cache/' not in n and not n.endswith('.lock') for n in manifest)
    dest=root/'outputs/physical-calibration-p1-20261107-evidence.zip'
    with zipfile.ZipFile(dest,'x',zipfile.ZIP_DEFLATED) as z:
        for n,entry in manifest.items():
            assert sha(root/n)==entry['sha256'];z.write(root/n,n)
        z.writestr('manifest.json',json.dumps(manifest,indent=2))
    with zipfile.ZipFile(dest) as z:
        assert z.testzip() is None
        for n,entry in manifest.items():assert hashlib.sha256(z.read(n)).hexdigest()==entry['sha256']
    receipt=dict(path=dest.name,bytes=dest.stat().st_size,sha256=sha(dest),files=len(manifest),crc_and_all_hashes_verified=True,
        raw_graph_cache_private_lease_excluded=True,release_uploaded=False,publication_scope='One archive receipt only; extensive private integrity manifest stays local')
    write(root/'outputs/physical-calibration-p1-20261107-artifact.json',receipt);print(json.dumps(receipt))


if __name__=='__main__':main()
