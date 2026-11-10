"""Seal completed WS1 evidence locally; no upload or public receipt output."""
import hashlib
import json
import zipfile
from pathlib import Path

root=Path(__file__).resolve().parents[1]


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while block:=f.read(1024*1024):h.update(block)
    return h.hexdigest()


def main():
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    folder=root/'outputs/weight-sensitivity-ws1-20261110'
    summary=root/'outputs/weight-sensitivity-ws1-20261110-summary.json'
    result=read(folder/'result.json');audit=read(summary)
    assert result['status']=='completed_WS1' and audit['status']=='completed_independent_WS1_audit'
    assert result['lease_released'] and result['process_tree_empty_verified']
    assert not (root/'work/Project-Noesis/gpu-pilot/outputs/codex-training-host.lock').exists()
    protocol=read(folder/'protocol.json')
    selected={p for p in folder.rglob('*') if p.is_file() and 'brain-cache' not in p.parts}
    selected.add(summary)
    selected.update((root/'outputs/weight-sensitivity-ws1-inputs-20261110').glob('*'))
    selected.update(root/n for n in protocol['source_hashes'])
    selected.update(root/'work'/n for n in ('package_weight_sensitivity_ws1.py','WS1-run-note.md'))
    selected.add(root/'outputs/weight-sensitivity-ws1-public-protocol-20261110.json')
    # Baseline raw evidence is included, not just an unverifiable path reference.
    import run_weight_sensitivity_ws1 as run
    for seed in run.SEEDS:
        selected.update(p for p in run.history.reference('parent',seed).iterdir() if p.is_file())
    selected.add(run.PARENT)
    selected={p.resolve() for p in selected}
    assert all(p.is_file() and p.is_relative_to(root) for p in selected)
    assert all('NoesisData' not in p.parts and 'brain-cache' not in p.parts and 'noesis-env' not in p.parts for p in selected)
    archive=root/'outputs/weight-sensitivity-ws1-20261110-evidence.zip'
    manifest={}
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(selected):
            name=path.relative_to(root).as_posix();before=digest(path)
            z.write(path,name)
            assert digest(path)==before
            manifest[name]=dict(sha256=before,bytes=path.stat().st_size)
        z.writestr('manifest.json',json.dumps(manifest,indent=2))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and set(z.namelist())==set(manifest)|{'manifest.json'}
        for name,entry in manifest.items():
            payload=z.read(name)
            assert len(payload)==entry['bytes'] and hashlib.sha256(payload).hexdigest()==entry['sha256']
            assert digest(root/name)==entry['sha256']
    receipt=dict(path=archive.name,sha256=digest(archive),bytes=archive.stat().st_size,
                 selected_files=len(manifest),crc_and_all_hashes_verified=True,local_only=True)
    (root/'outputs/weight-sensitivity-ws1-20261110-artifact.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print('WS1 evidence sealed and verified locally; private receipt not published')


if __name__=='__main__':main()
