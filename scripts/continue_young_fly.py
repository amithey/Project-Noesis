"""Continue a rejected fly candidate while reusing verified immutable quiet controls.

Invoke only after verifying the old training process is terminal. This changes the
experiment ledger, not the model source, candidate weights or acceptance criteria.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from noesis.experiments import learning_lab as lab
from noesis.experiments.train_validate import atomic_json, promotion_decision


def source_hash():
    base = Path(lab.__file__).resolve().parents[1]
    h = hashlib.sha256()
    for path in sorted(base.rglob('*.py')):
        h.update(path.relative_to(base).as_posix().encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def input_manifest(data):
    roots = [data/'raw', data/'processed', data/'manc', data.parent/'MANC']
    spec = importlib.util.find_spec('flygym')
    if spec and spec.origin:
        roots.append(Path(spec.origin).parent)
    extensions = {'.py', '.csv', '.parquet', '.pkl', '.npz', '.npy', '.xml',
                  '.yaml', '.json', '.obj', '.stl', '.ply', '.png'}
    files = {}
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix.lower() in extensions:
                stat = path.stat()
                files[str(path.resolve())] = [stat.st_size, stat.st_mtime_ns]
    versions = {}
    for name in ('numpy', 'scipy', 'numba', 'pandas', 'flygym', 'mujoco'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {'files': files, 'versions': versions}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def prepare_continuation(report_path, *, terminal_verified=False, dry_run=False):
    if not terminal_verified:
        raise ValueError('verify the old process is terminal before preparing a continuation')
    report_path = Path(report_path).resolve()
    report = json.loads(report_path.read_text(encoding='utf-8'))
    if report.get('schema') != 3 or report.get('status') != 'rejected_validation':
        raise ValueError('only a completed quiet-validation rejection can start a new cached cycle')
    if report['decision']['accepted'] or any(a['status'] == 'running' for a in report['attempts']):
        raise ValueError('cannot continue an active or accepted candidate this way')
    if report_path.parent != Path(report['settings']['output']).resolve():
        raise ValueError('report must remain in its own output directory')
    if report['settings']['source_sha256'] != source_hash():
        raise ValueError('model source changed; cached controls are not reusable')
    data = Path(report['settings']['data']).resolve()
    if report_path.parent == data or data in report_path.parents:
        raise ValueError('output cannot be inside source data')
    manifest = input_manifest(data)
    manifest_path = report_path.parent/'original-control-inputs.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding='utf-8')) != manifest:
        raise ValueError('source data or runtime changed; recompute the original controls')
    controls = [j for j in report['jobs'] if j['stage'] == 'validation'
                and j['arena'] == 'quiet' and j['model'] == 'original']
    if sorted(j['seed'] for j in controls) != sorted(report['settings']['validation_seeds']):
        raise ValueError('need every matched original quiet control')
    last = Path(report['last_training_checkpoint'])
    model_context = json.loads(last.with_suffix('.model.json').read_text(encoding='utf-8'))
    verified = []
    with np.load(last, allow_pickle=False) as trained:
        for job in controls:
            summary = job['summary']
            if (summary['status'] != 'completed' or summary['simulated_seconds'] != 5.
                    or summary['learning'] or summary['motor_exploration_mv_ms'] != 0
                    or summary['seed'] != job['seed']):
                raise ValueError('cannot reuse a partial, learning or exploratory control')
            path = Path(summary['checkpoint']).resolve()
            if json.loads(path.with_suffix('.json').read_text(encoding='utf-8')) != summary:
                raise ValueError('control summary no longer matches its artifact')
            if json.loads(path.with_suffix('.model.json').read_text(encoding='utf-8')) != model_context:
                raise ValueError('control body/neural/reward context differs')
            with np.load(path, allow_pickle=False) as original:
                for key in ('topology', 'motor_indices', 'initial_weights', 'parameters'):
                    if not np.array_equal(original[key], trained[key]):
                        raise ValueError('control uses a different original network or learning configuration')
                if not np.array_equal(original['weights'], original['initial_weights']):
                    raise ValueError('trained weights cannot be reused as an original control')
            with path.with_suffix('.csv').open(newline='', encoding='utf-8') as stream:
                rows = list(csv.DictReader(stream))
            if (len(rows) != 1000 or abs(float(rows[-1]['time_s'])-5.) > 1e-9
                    or sum(int(row['awake']) for row in rows) != summary['awake_samples']):
                raise ValueError('control does not contain its complete five-second trace')
            hashes = {p.name: digest(p) for p in sorted(path.parent.glob(path.stem+'.*')) if p.is_file()}
            if job.get('cache_artifact_hashes', hashes) != hashes:
                raise ValueError('cached control artifacts changed')
            item = copy.deepcopy(job)
            item.setdefault('original_execution_cycle', report['cycle'])
            item['cache_artifact_hashes'] = hashes
            item['reused_in_cycle'] = report['cycle']+1
            verified.append(item)
    failed = False
    for job in report['jobs']:
        if job['stage'] == 'validation' and job['arena'] == 'quiet' and job['model'] == 'candidate':
            original = next(j['summary'] for j in controls if j['seed'] == job['seed'])
            if not promotion_decision([original], [job['summary']])['accepted']:
                failed = True
    if not failed:
        raise ValueError('a completed mandatory quiet criterion must actually have failed')
    new = copy.deepcopy(report)
    new.setdefault('history', []).append({k: copy.deepcopy(report[k]) for k in
        ('cycle', 'jobs', 'decision', 'last_training_checkpoint', 'best_validated_checkpoint', 'settings')})
    seed = report['settings']['seed']+report['settings']['episodes']
    reserved = set(report['settings']['validation_seeds']+report['settings']['heldout_seeds'])
    while set(range(seed, seed+report['settings']['episodes'])) & reserved:
        seed += 1
    if seed+report['settings']['episodes'] > 2**32:
        raise ValueError('training seed space exhausted')
    audit = {'utc': datetime.now(UTC).isoformat(), 'from_cycle': report['cycle'],
             'to_cycle': report['cycle']+1, 'reused_original_jobs': [j['id'] for j in verified],
             'candidate_or_heldout_jobs_reused': [], 'source_sha256': source_hash(),
             'input_manifest_sha256': hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
             'next_training_seed': seed, 'training_start_checkpoint': report['last_training_checkpoint']}
    new['settings']['seed'] = seed
    new.update(cycle=report['cycle']+1, jobs=verified, validation_decision=None,
               decision=None, status='pending')
    new.setdefault('control_reuse_history', []).append(audit)
    if not dry_run:
        if not manifest_path.exists():
            atomic_json(manifest_path, manifest)
        atomic_json(report_path, new)
    return new, audit


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--report', type=Path, required=True)
    ap.add_argument('--terminal-verified', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    _, audit = prepare_continuation(args.report, terminal_verified=args.terminal_verified, dry_run=args.dry_run)
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == '__main__':
    main()
