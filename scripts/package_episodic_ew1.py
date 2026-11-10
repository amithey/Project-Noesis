"""Seal terminal EW1 selected evidence locally; never upload or print receipt."""
import hashlib
import json
from pathlib import Path
import zipfile

root=Path(__file__).resolve().parents[1]


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while block:=f.read(1024*1024):h.update(block)
    return h.hexdigest()


def main():
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    folder=root/'outputs/episodic-neural-ew1-20261110'
    summary_path=root/'outputs/episodic-neural-ew1-20261110-summary.json'
    result=read(folder/'result.json');summary=read(summary_path);p=read(folder/'protocol.json')
    assert result['status']=='completed_EW1' and summary['status']=='completed_independent_EW1_audit'
    assert summary['full_workers']==31 and summary['verified_training_trials']==16
    assert result['lease_released'] and result['process_tree_empty_verified']
    assert not (root/'work/Project-Noesis/gpu-pilot/outputs/codex-training-host.lock').exists()
    for name,expected in result['created_hashes'].items():assert digest(root/name)==expected
    for group in ('source_hashes','prior_hashes','artifact_hashes'):
        for name,expected in p[group].items():assert digest(root/name)==expected
    selected={path for path in folder.rglob('*') if path.is_file() and 'brain-cache' not in path.parts}
    selected.update((root/'outputs/episodic-neural-ew1-inputs-20261110').glob('*'))
    selected.update(root/name for name in p['source_hashes'])
    selected.add(root/p['parent_checkpoint']);selected.add(summary_path)
    selected.add(Path(__file__).resolve())
    selected.add(root/'outputs/episodic-neural-ew1-public-protocol-20261110.json')
    for d in [p['baseline_folder'],*p['reference_folders'].values()]:
        selected.update(path for path in (root/d).iterdir() if path.is_file())
    selected={path.resolve() for path in selected}
    assert all(path.is_file() and path.is_relative_to(root) for path in selected)
    assert all(not {'NoesisData','brain-cache','noesis-env'}&set(path.parts) for path in selected)
    archive=root/'outputs/episodic-neural-ew1-20261110-evidence.zip';manifest={}
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(selected):
            before=digest(path);name=path.relative_to(root).as_posix();z.write(path,name)
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
        selected_files=len(manifest),crc_and_all_hashes_verified=True,local_only=True,
        scope='Selected experiment evidence, not raw connectome/environment or exact mid-episode resume')
    (root/'outputs/episodic-neural-ew1-20261110-artifact.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print('EW1 selected evidence sealed and verified locally; private receipt not published')


if __name__=='__main__':main()
