"""Bounded development-only proprioception comparison; never promotes weights."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

from noesis.embodiment import load_training_brain
from noesis.experiments import learning_lab as lab
from noesis.experiments.train_validate import atomic_json
from noesis.proprioception import ENCODINGS
from noesis.wall_budget import WallBudget


def source_fingerprint():
    base = Path(lab.__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for source in sorted(base.rglob('*.py')):
        digest.update(source.relative_to(base).as_posix().encode())
        digest.update(source.read_bytes())
    return digest.hexdigest()


def main(argv=None):
    start = time.monotonic()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--wall-seconds', type=float, default=3600.)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--video', action='store_true')
    ap.add_argument('--seeds', type=int, nargs='+', default=[15, 16])
    ap.add_argument('--encodings', nargs='+', choices=ENCODINGS,
                    default=['legacy', 'typed-resistance', 'typed-reversed'])
    args = ap.parse_args(argv)
    if not math.isfinite(args.wall_seconds) or not 0 < args.wall_seconds <= 3600:
        ap.error('wall-seconds must be in (0, 3600]')
    if (not args.seeds or len(set(args.seeds)) != len(args.seeds)
            or any(seed < 0 or seed >= 2**32 for seed in args.seeds)):
        ap.error('development seeds must be unique unsigned 32-bit integers')
    if len(set(args.encodings)) != len(args.encodings):
        ap.error('development encodings must be unique')
    if set(args.seeds) & {10, 11, 200, 201, 202}:
        ap.error('validation and held-out seeds are reserved')
    if args.output.resolve() == args.data.resolve() or args.data.resolve() in args.output.resolve().parents:
        ap.error('output must be outside read-only source data')
    jobs = [{'id': f'{encoding}-{seed}', 'encoding': encoding, 'seed': seed}
            for index, seed in enumerate(args.seeds)
            for encoding in (args.encodings if index % 2 == 0 else args.encodings[::-1])]
    settings = {'data': str(args.data.resolve()), 'output': str(args.output.resolve()),
                'seeds': args.seeds, 'seconds': 5., 'source_sha256': source_fingerprint(),
                'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    report_path = args.output/'report.json'
    if args.resume:
        if not report_path.is_file():
            ap.error('resume requires an existing report')
        report = json.loads(report_path.read_text(encoding='utf-8'))
        if report.get('schema') != 1 or report.get('settings') != settings or report.get('planned_jobs') != jobs:
            ap.error('resume source/configuration mismatch; preserve prior outputs')
        if report['status'] == 'completed':
            ap.error('development screen already completed')
        for attempt in report['attempts']:
            if attempt['status'] == 'running':
                attempt['status'] = 'interrupted'
    else:
        if args.output.exists() and any(args.output.iterdir()):
            ap.error('output must be new or empty')
        args.output.mkdir(parents=True, exist_ok=True)
        report = {'schema': 1, 'settings': settings, 'planned_jobs': jobs, 'completed': [],
                  'attempts': [], 'invocations': [], 'status': 'pending',
                  'development_only': True, 'promotion_allowed': False}
    atomic_json(report_path, report)
    deadline = start + args.wall_seconds
    with WallBudget(deadline):
        cn, _scale = load_training_brain(args.data, args.output/'cache')
        done = {job['id'] for job in report['completed']}
        for job in jobs:
            if job['id'] in done:
                continue
            if time.monotonic() >= deadline - 5:
                break
            output = args.output/job['id']/f"attempt_{len(report['attempts']):04d}"
            case_argv = ['--data', str(args.data), '--output', str(output), '--seconds', '5',
                         '--seed', str(job['seed']), '--arena', 'quiet', '--motor-exploration', '0',
                         '--normalize-reward-error', '--proprioceptive-encoding', job['encoding']]
            if args.video:
                case_argv.append('--video')
            case = lab.parser().parse_args(case_argv)
            lab.validate_args(lab.parser(), case)
            output.mkdir(parents=True, exist_ok=True)
            attempt = dict(**job, output=str(output.resolve()), start_checkpoint=None, status='running')
            report['attempts'].append(attempt)
            atomic_json(report_path, report)
            try:
                summary, _checkpoint = lab.run_episode(cn, case, 0, None, deadline=deadline - 5)
            except Exception as exc:
                attempt.update(status='error', error=f'{type(exc).__name__}: {exc}')
                atomic_json(report_path, report)
                raise
            attempt.update(status=summary['status'], summary=summary)
            if summary['status'] == 'completed':
                if (summary['simulated_seconds'] < 5 or summary['learning']
                        or summary['motor_exploration_mv_ms'] != 0):
                    raise ValueError('development screen requires complete frozen original trials')
                report['completed'].append(dict(**job, summary=summary))
            atomic_json(report_path, report)
            print(json.dumps(dict(job=job['id'], **summary)), flush=True)
            if summary['status'] != 'completed':
                break
        report['status'] = 'completed' if len(report['completed']) == len(jobs) else 'pending'
        report['invocations'].append({'wall_seconds': time.monotonic()-start,
                                      'completed_jobs': len(report['completed']), 'total_jobs': len(jobs)})
        atomic_json(report_path, report)
        print(json.dumps({'status': report['status'], 'development_only': True,
                          'promotion_allowed': False}), flush=True)


if __name__ == '__main__':
    main()
