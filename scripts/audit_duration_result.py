"""Recompute a completed T1B study and archive only its named evidence files."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
from compare_exploration import digest, tree_digest, write
from run_duration_study import (PROBES, PARENT, CONTROL, SOURCES, audit_bank,
    duration_decision, prefix_check, verify_guarded_receipt)
from noesis_gpu.evaluate import MATCHED_SETTINGS, summarize


def audit(root, core, study):
    own = Path(__file__).resolve().parents[1]
    protocol = json.loads((study/'protocol.json').read_text())
    result = json.loads((study/'result.json').read_text())
    if (result['status'] != 'completed_duration_study' or len(result['jobs']) != 13
            or result['wall_seconds'] > 3600 or result['model_promoted']
            or result['walking_proven'] or not protocol['guarded_mean_cpu']):
        raise ValueError('Require completed, unpromoted T1B within original budget')
    hashes = dict(core=tree_digest(core), pilot=tree_digest(own/'gpu-pilot/src'))
    if hashes != SOURCES or protocol['source_hashes'] != hashes:
        raise ValueError('Source changed')
    parent = root/'gpu-pilot/outputs/e4-exploration-20261029/train-control-1000/weights-last.npz'
    control = root/'gpu-pilot/outputs/logging-parity-d3-20261030/logged'
    if digest(parent) != PARENT or digest(control/'weights-last.npz') != CONTROL:
        raise ValueError('Parent/control changed')
    receipt = Path(protocol['D4M_receipt'])
    if verify_guarded_receipt(receipt, parent, control, hashes) != protocol['D4M_files']:
        raise ValueError('D4M receipt changed')
    trained = json.loads((study/'train25/report.json').read_text())
    if any(trained[k] != v for k, v in dict(status='completed_training_birth',
            completed_chunks=5000, learning_updates=4990, simulated_seconds=25,
            seed=1000, learning_rate=.2, motor_exploration_mv_ms=1.5,
            import_checkpoint_sha256=PARENT, noesis_source_sha256=hashes['core']).items()):
        raise ValueError('Training report mismatch')
    child = study/'train25/weights-last.npz'
    child_sha = audit_bank(child, parent)
    if child_sha != result['child25_sha256'] or child_sha != trained['saved_checkpoint_sha256']:
        raise ValueError('Child hash mismatch')
    fields = prefix_check(study/'train25', control)
    banks = dict(parent=parent, control5=control/'weights-last.npz', child25=child)
    cases = {arm: {} for arm in banks}
    rows, telemetry = [], []
    names = ['train25'] + [f'frozen-{a}-{s}' for s in PROBES for a in banks]
    if [j['name'] for j in result['jobs']] != names:
        raise ValueError('Job set/order changed')
    for job in result['jobs']:
        name = job['name']
        t = json.loads((study/f'{name}.telemetry.json').read_text())
        if (job['exit_code'] != 0 or t['exit_code'] != 0 or t['final_exit_code'] != 0
                or t['status'] != 'process_terminal' or not t['samples']
                or min(s['host_available_gib'] for s in t['samples']) < 1
                or t['process_wall_seconds'] > (930 if name == 'train25' else 330)):
            raise ValueError(f'Incomplete/resource-limited job: {name}')
        pf = json.loads((study/f'{name}-resource-preflight.json').read_text())
        if not pf['decision']['passed'] or len(pf['samples']) != 10:
            raise ValueError('Preflight not passed')
        telemetry.append(dict(name=name, process_seconds=t['process_wall_seconds'],
            minimum_available_gib=min(s['host_available_gib'] for s in t['samples']),
            peak_child_rss_gib=max(s['child_rss_bytes'] for s in t['samples'])/2**30,
            max_host_cpu_percent=max(s['host_cpu_percent'] for s in t['samples']),
            legacy_T1_gate_passed=pf['decision']['original_T1_rule_passed']))
    for seed in PROBES:
        reports = {}
        for arm, bank in banks.items():
            name = f'frozen-{arm}-{seed}'
            report, summary = summarize(study/name)
            cached = json.loads((study/f'{name}-summary.json').read_text())
            if summary != cached or report['seeds'] != [seed] or report['checkpoint_sha256'] != digest(bank):
                raise ValueError('Raw-summary/context/bank mismatch')
            with np.load(study/name/'observations.npz', allow_pickle=False) as data:
                if any(v.shape[0] != 1000 or not np.isfinite(v).all() for v in data.values()):
                    raise ValueError('Incomplete/nonfinite frozen observation field')
            reports[arm] = report
            cases[arm][seed] = summary
            rows.append(dict(arm=arm, seed=seed, **{k:summary[k] for k in
                ('standing_fraction','mean_stance_slip_mm_s','heading_forward_distance_mm',
                 'net_forward_displacement_mm','leg_step_counts','locomotion_regime')}))
        if any(reports[a][k] != reports['parent'][k] for a in ('control5','child25')
               for k in (*MATCHED_SETTINGS, 'seeds')):
            raise ValueError('Unmatched frozen settings')
    decision = duration_decision(cases)
    if decision != result['decision']:
        raise ValueError('Recomputed decision differs')
    return dict(status='independent_raw_recomputation_passed', protocol_commit=protocol['commit'],
        source_hashes=hashes, child25_sha256=child_sha, full_training_seconds=25,
        learning_updates=4990, exact_prefix_fields=fields, complete_frozen_probes=12,
        wall_seconds=result['wall_seconds'], decision=decision, probes=rows, telemetry=telemetry,
        model_promoted=False, walking_proven=False, retention_proven=False,
        scope='N1 training seed/four development probes; duration method failed; final held-outs unused')


def archive(root, study, target, summary):
    target.mkdir(parents=True, exist_ok=False)
    files = {p.relative_to(root).as_posix(): p for p in study.glob('*.json')}
    files.update({p.relative_to(root).as_posix(): p for p in study.glob('*.log')})
    initial=study.with_name(study.name+'-resource-preflight.json')
    files[initial.relative_to(root).as_posix()]=initial
    for name in ['train25']+[f'frozen-{a}-{s}' for s in PROBES for a in ('parent','control5','child25')]:
        allowed = ['report.json', 'training-observations.npz', 'exploration-log.npz',
                   'weights-last.npz', 'weights-last.model.json'] if name == 'train25' else ['report.json','observations.npz']
        for filename in allowed:
            p = study/name/filename
            if not p.is_file(): raise ValueError(f'Missing required evidence: {p}')
            files[p.relative_to(root).as_posix()] = p
    for relative in ['gpu-pilot/outputs/e4-exploration-20261029/train-control-1000/weights-last.npz',
                     'gpu-pilot/outputs/e4-exploration-20261029/train-control-1000/weights-last.model.json']:
        files[relative] = root/relative
    control = root/'gpu-pilot/outputs/logging-parity-d3-20261030/logged'
    for filename in ['report.json','weights-last.npz','weights-last.model.json','training-observations.npz','exploration-log.npz']:
        p=control/filename; files[p.relative_to(root).as_posix()]=p
    entries=[dict(path=n, bytes=p.stat().st_size, sha256=digest(p)) for n,p in sorted(files.items())]
    payload=dict(schema_version=1, files=entries, audit=summary)
    destination=target/'duration-t1b-evidence.zip'
    with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for entry in entries:
            p=files[entry['path']];z.write(p,entry['path'])
            if digest(p)!=entry['sha256']:raise ValueError('Evidence changed during archive')
        z.writestr('evidence-manifest.json',json.dumps(payload,indent=2))
    with zipfile.ZipFile(destination) as z:
        if z.testzip() is not None:raise ValueError('ZIP CRC failed')
        if set(z.namelist()) != set(files)|{'evidence-manifest.json'}:raise ValueError('ZIP entries differ')
        for entry in entries:
            data=z.read(entry['path'])
            if len(data)!=entry['bytes'] or hashlib.sha256(data).hexdigest()!=entry['sha256']:
                raise ValueError('ZIP SHA mismatch')
    write(target/'duration-t1b-manifest.json',dict(**payload,archive=dict(
        name=destination.name,bytes=destination.stat().st_size,sha256=digest(destination))))
    return dict(files=len(entries),archive_sha256=digest(destination),archive_bytes=destination.stat().st_size)


if __name__ == '__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for arg in ('shared-root','core-source','study','summary','archive-directory'):
        ap.add_argument('--'+arg,type=Path,required=True)
    args=ap.parse_args()
    summary=audit(args.shared_root.resolve(),args.core_source.resolve(),args.study.resolve())
    write(args.summary,summary)
    print(json.dumps(archive(args.shared_root.resolve(),args.study.resolve(),args.archive_directory.resolve(),summary)))
