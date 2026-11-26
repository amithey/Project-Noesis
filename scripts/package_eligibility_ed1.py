"""Seal selected ED1 evidence locally; private maps/pending payloads are not uploaded."""
import hashlib,json,zipfile
import run_eligibility_ed1 as study
from ap2_contract import DEV

root,out=study.root,study.out
read,write,sha=study.read,study.write,study.sha


def main():
    summary=root/'outputs/eligibility-delay-ed1-20261107-summary.json';assert read(summary)['status']=='independently_audited_ED1'
    selected={summary,root/'outputs/eligibility-delay-ed1-minimal-result-20261107.json',root/'outputs/eligibility-delay-ed1-public-protocol-20261107.json',study.PARENT}
    selected|={p for p in out.iterdir() if p.is_file() and p.suffix in ('.json','.log')}
    for d in out.iterdir():
        if d.is_dir():selected|={p for p in d.iterdir() if p.is_file() and p.suffix in ('.npz','.json')}
    for bank in ('parent','v7','AP1','previous_PS1'):
        for seed in (study.SEEDS if bank=='parent' else DEV):
            d=study.reference(bank,seed);selected|={p for p in d.iterdir() if p.is_file() and p.suffix in ('.npz','.json')}
    for n in ('ed1_eligibility.py','test_ed1_eligibility.py','ed1_contract.py','test_ed1_contract.py','ed1_worker.py','run_eligibility_ed1.py',
        'audit_eligibility_ed1.py','package_eligibility_ed1.py','ps1_objective.py','sc1_signal.py'):
        selected.add(root/'work'/n)
    manifest={p.relative_to(root).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(selected)}
    assert all('/NoesisData/' not in n and '/brain-cache/' not in n and not n.endswith('.lock') for n in manifest)
    dest=root/'outputs/eligibility-delay-ed1-20261107-evidence.zip'
    with zipfile.ZipFile(dest,'x',zipfile.ZIP_DEFLATED) as z:
        for n,v in manifest.items():
            assert sha(root/n)==v['sha256'];z.write(root/n,n)
        z.writestr('manifest.json',json.dumps(manifest,indent=2))
    with zipfile.ZipFile(dest) as z:
        assert z.testzip() is None
        for n,v in manifest.items():assert hashlib.sha256(z.read(n)).hexdigest()==v['sha256']
    receipt=dict(path=dest.name,bytes=dest.stat().st_size,sha256=sha(dest),files=len(manifest),crc_and_all_hashes_verified=True,
        raw_graph_cache_private_lease_excluded=True,release_uploaded=False,
        publication_scope='One archive receipt only. Archive includes private integrity maps and old performance data; no upload authorized by this script.')
    write(root/'outputs/eligibility-delay-ed1-20261107-artifact.json',receipt);print(json.dumps(receipt))


if __name__=='__main__':main()
