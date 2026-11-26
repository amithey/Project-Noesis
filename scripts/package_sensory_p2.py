"""Selected signed local P2 evidence; private maps are never uploaded here."""
import hashlib,json,zipfile
import run_sensory_p2 as study

root,out=study.root,study.out
read,write,sha=study.read,study.write,study.sha


def main():
    summary=root/'outputs/sensory-causality-p2-public-result-20261107.json'
    assert read(summary)['status']=='independently_audited_P2'
    selected={summary,root/'outputs/sensory-causality-p2-public-protocol-20261107.json'}
    selected|={p for p in out.iterdir() if p.is_file() and p.suffix in ('.json','.log')}
    for d in out.iterdir():
        if d.is_dir():selected|={p for p in d.iterdir() if p.is_file() and p.suffix in ('.json','.npz')}
    for model in study.MODELS:
        for condition in ('baseline','pulse'):
            d=study.base.out/(model+'-'+condition)
            selected|={p for p in d.iterdir() if p.is_file() and p.suffix in ('.json','.npz')}
    for n in ('p2_sensory_history.py','p2_worker.py','run_sensory_p2.py','test_p2_sensory_history.py','audit_sensory_p2.py','package_sensory_p2.py'):
        selected.add(root/'work'/n)
    selected|=set(study.MODELS.values())
    manifest={p.relative_to(root).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(selected)}
    assert all('/NoesisData/' not in n and '/brain-cache/' not in n and not n.endswith('.lock') for n in manifest)
    dest=root/'outputs/sensory-causality-p2-20261107-evidence.zip'
    with zipfile.ZipFile(dest,'x',zipfile.ZIP_DEFLATED) as z:
        for n,entry in manifest.items():
            assert sha(root/n)==entry['sha256'];z.write(root/n,n)
        z.writestr('manifest.json',json.dumps(manifest,indent=2))
    with zipfile.ZipFile(dest) as z:
        assert z.testzip() is None
        for n,entry in manifest.items():assert hashlib.sha256(z.read(n)).hexdigest()==entry['sha256']
    receipt=dict(path=dest.name,bytes=dest.stat().st_size,sha256=sha(dest),files=len(manifest),crc_and_all_hashes_verified=True,
        raw_graph_cache_private_lease_excluded=True,release_uploaded=False,publication_scope='One archive receipt only; extensive private integrity manifest stays local')
    write(root/'outputs/sensory-causality-p2-20261107-artifact.json',receipt);print(json.dumps(receipt))


if __name__=='__main__':main()
