"""Local ED2 evidence sealing; this module does not publish private metadata."""
import hashlib,json,zipfile
import run_eligibility_ed2 as study
from ed2_contract import BIRTHS
from ap2_contract import DEV

root,out=study.root,study.out
read,write,sha=study.read,study.write,study.sha


def main():
    summary=root/'outputs/eligibility-replication-ed2-20261107-summary.json';assert read(summary)['status']=='independently_audited_ED2'
    selected={summary,root/'outputs/eligibility-replication-ed2-minimal-result-20261107.json',root/'outputs/eligibility-replication-ed2-public-protocol-20261107.json',study.PARENT}
    selected|={p for p in out.iterdir() if p.is_file() and p.suffix in ('.json','.log')}
    for d in out.iterdir():
        if d.is_dir():selected|={p for p in d.iterdir() if p.is_file() and p.suffix in ('.json','.npz')}
    for birth in BIRTHS:
        for bank in ('parent','control','candidate','v7','AP1','previous_PS1'):
            for seed in ((birth,*DEV) if bank in ('parent','control','candidate') else DEV):
                d=study.directory(birth,bank,seed);selected|={p for p in d.iterdir() if p.is_file() and p.suffix in ('.json','.npz')}
    for bank in ('control','candidate'):
        d=study.old.out/f'train-{bank}';selected|={p for p in d.iterdir() if p.is_file() and p.suffix in ('.json','.npz')}
    for n in ('ed2_contract.py','test_ed2_contract.py','ed2_worker.py','run_eligibility_ed2.py','audit_eligibility_ed2.py','package_eligibility_ed2.py',
        'ed1_eligibility.py','ed1_contract.py','sc1_signal.py','ps1_objective.py'):
        selected.add(root/'work'/n)
    manifest={p.relative_to(root).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(selected)}
    assert all('/NoesisData/' not in n and '/brain-cache/' not in n and not n.endswith('.lock') for n in manifest)
    dest=root/'outputs/eligibility-replication-ed2-20261107-evidence.zip'
    with zipfile.ZipFile(dest,'x',zipfile.ZIP_DEFLATED) as z:
        for n,v in manifest.items():
            assert sha(root/n)==v['sha256'];z.write(root/n,n)
        z.writestr('manifest.json',json.dumps(manifest,indent=2))
    with zipfile.ZipFile(dest) as z:
        assert z.testzip() is None
        for n,v in manifest.items():assert hashlib.sha256(z.read(n)).hexdigest()==v['sha256']
    receipt=dict(path=dest.name,bytes=dest.stat().st_size,sha256=sha(dest),files=len(manifest),crc_and_all_hashes_verified=True,
        raw_graph_cache_private_lease_excluded=True,release_uploaded=False,publication_scope='Private local evidence only; no publication of receipt or private metadata by this script')
    write(root/'outputs/eligibility-replication-ed2-20261107-artifact.json',receipt)
    print('Local ED2 evidence verified; private receipt was not published')


if __name__=='__main__':main()
