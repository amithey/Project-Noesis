"""Bounded, resumable brain-only training with frozen matched behavioral controls."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from noesis.embodied_metrics import GROUNDED_OBJECTIVES
from noesis.embodiment import load_training_brain
from noesis.experiments import learning_lab as lab
from noesis.proprioception import ENCODINGS
from noesis.wall_budget import WallBudget


def atomic_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def promotion_decision(original, candidate):
    """Every matched five-second trial must meet stance and walking criteria."""
    if not original or len(original) != len(candidate):
        raise ValueError('need equally sized nonempty matched evaluations')
    if [r['seed'] for r in original] != [r['seed'] for r in candidate]:
        raise ValueError('evaluation seeds must match')
    checks = []
    for baseline, trained in zip(original, candidate, strict=True):
        complete = all(r['status'] == 'completed' and r.get('simulated_seconds', 0) >= 5.
                       and not r.get('learning', True) and r.get('motor_exploration_mv_ms', 1) == 0
                       for r in (baseline, trained))
        finite = all(r.get(k) is not None and np.isfinite(r[k]) for r in (baseline, trained)
                     for k in ('mean_reward', 'mean_stance_slip_mm_s'))
        standing = trained.get('standing_fraction', 0) >= .9
        cycles = sum(v >= 2 for v in trained.get('leg_step_cycles', {}).values()) >= 4
        forward = trained.get('heading_forward_distance_mm', 0) >= 1.
        slip = finite and trained['mean_stance_slip_mm_s'] <= .8 * baseline['mean_stance_slip_mm_s']
        reward = finite and trained['mean_reward'] > baseline['mean_reward'] + .001
        checks.append({'seed': trained['seed'], 'complete': complete, 'standing': standing,
                           'cycles': cycles, 'forward': forward, 'slip': bool(slip), 'reward': bool(reward)})
    accepted = all(all(v for k, v in check.items() if k != 'seed') for check in checks)
    return {'accepted': accepted, 'reason': 'all behavioral criteria passed' if accepted
                else 'candidate did not pass all behavioral criteria', 'checks': checks}


def obstacle_decision(candidate, ablated):
    if [r['seed'] for r in candidate] != [r['seed'] for r in ablated] or not candidate:
        return False
    return all(a['status'] == b['status'] == 'completed'
               and a.get('simulated_seconds', 0) >= 5 and b.get('simulated_seconds', 0) >= 5
               and a.get('obstacle_observation_complete', False)
               and a.get('obstacle_response_seconds') is not None
               and a['obstacle_response_seconds'] <= 1.
               and a['obstacle_contact_seconds'] < b['obstacle_contact_seconds']
               and not a.get('learning', True) and not b.get('learning', True)
               and a.get('motor_exploration_mv_ms', 1) == b.get('motor_exploration_mv_ms', 1) == 0
               for a, b in zip(candidate, ablated, strict=True))


def build_jobs(episodes, seed, validation, heldout, training_arena='graded'):
    """Choose training scenes while retaining every frozen evaluation case."""
    if training_arena not in ('graded', 'quiet', 'obstacle'):
        raise ValueError('unknown training arena')
    jobs = [{'id': f'train-{i}', 'stage': 'training', 'model': 'candidate', 'seed': seed+i,
                 'arena': training_arena if training_arena != 'graded' else (
                     'quiet' if i < max(1, episodes//2) else 'obstacle')} for i in range(episodes)]
    for stage, seeds in [('validation', validation), ('heldout', heldout)]:
        for arena, models in [('quiet', ('original', 'candidate')),
                              ('obstacle', ('original', 'candidate', 'ablated'))]:
            jobs += [{'id': f'{stage}-{arena}-{model}-{s}', 'stage': stage, 'arena': arena,
                          'model': model, 'seed': s} for model in models for s in seeds]
    return jobs


def main():
    start = time.monotonic()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--import-weights', type=Path,
                    help='start a new protocol with compatible learned weights from an earlier model')
    ap.add_argument('--normalize-reward-error', action='store_true')
    ap.add_argument('--early-reject', action='store_true', help='reject failed quiet validation before consuming further controls')
    ap.add_argument('--episodes', type=int, default=3)
    ap.add_argument('--proprioceptive-encoding', choices=ENCODINGS, default='legacy')
    ap.add_argument('--grounded-objective', choices=GROUNDED_OBJECTIVES, default=GROUNDED_OBJECTIVES[0])
    ap.add_argument('--training-arena', choices=('graded', 'quiet', 'obstacle'), default='graded',
                    help='training scenes only; frozen quiet/obstacle controls remain mandatory')
    ap.add_argument('--seconds', type=float, default=5.)
    ap.add_argument('--training-fall-seconds', type=float, default=0.,
                    help='terminal sustained fall for training births only; evaluations stay full length')
    ap.add_argument('--evaluation-seconds', type=float, default=5.)
    ap.add_argument('--wall-seconds', type=float, default=3600.)
    ap.add_argument('--depression-u', type=float, choices=(0., .03, .1), default=0.)
    ap.add_argument('--dynamics-report', type=Path)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--video', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--validation-seeds', type=int, nargs='+', default=[10, 11])
    ap.add_argument('--heldout-seeds', type=int, nargs='+', default=[200, 201, 202])
    ap.add_argument('--learning-rate', type=float, default=.2)
    args = ap.parse_args()
    if not np.isfinite(args.wall_seconds) or not 0 < args.wall_seconds <= 3600:
        ap.error('wall-seconds must be in (0, 3600]')
    if args.evaluation_seconds != 5.:
        ap.error('behavioral evaluation must last five seconds')
    if args.depression_u:
        from noesis import lif
        if args.dynamics_report is None:
            ap.error('nonzero depression requires a completed excitatory screen')
        screen = json.loads(args.dynamics_report.read_text(encoding='utf-8'))
        if (screen.get('status') != 'completed' or screen['settings']['scope'] != 'excitatory-only'
                or screen['settings']['tau_ms'] != 300.
                or screen['settings']['data'] != str(args.data.resolve())
                or screen['settings']['lif_sha256'] != hashlib.sha256(Path(lif.__file__).read_bytes()).hexdigest()
                or not any(d['u'] == args.depression_u and d['accepted'] for d in screen['decisions'])):
            ap.error('depression has not passed the current sensory/recovery gate')
    if args.import_weights is not None and not args.import_weights.is_file():
        ap.error('import-weights checkpoint does not exist')
    train_seeds = set(range(args.seed, args.seed + args.episodes))
    validation, heldout = set(args.validation_seeds), set(args.heldout_seeds)
    if (args.episodes < 1 or not validation or not heldout or train_seeds & validation
            or train_seeds & heldout or validation & heldout):
        ap.error('training, validation and held-out seeds must be nonempty and disjoint')
    if len(validation) != len(args.validation_seeds) or len(heldout) != len(args.heldout_seeds):
        ap.error('duplicate evaluation seeds')
    if any(seed < 0 or seed >= 2**32 for seed in train_seeds | validation | heldout):
        ap.error('seeds must fit unsigned 32-bit integers')
    if args.output.resolve() == args.data.resolve() or args.data.resolve() in args.output.resolve().parents:
        ap.error('output must be outside read-only source data')
    template = lab.parser().parse_args(['--data', str(args.data), '--output', str(args.output/'probe'),
        '--seconds', str(args.seconds), '--learning-rate', str(args.learning_rate), '--arena', 'quiet', '--depression-u', str(args.depression_u), '--learn'])
    template.grounded_objective = args.grounded_objective
    template.training_fall_seconds = args.training_fall_seconds
    template.proprioceptive_encoding = args.proprioceptive_encoding
    template.normalize_reward_error = args.normalize_reward_error
    lab.validate_args(lab.parser(), template)
    # Reject resuming under changed source code, model parameters or evaluation protocol.
    digest = hashlib.sha256()
    for source in sorted(Path(lab.__file__).resolve().parents[1].rglob('*.py')):
        digest.update(source.relative_to(Path(lab.__file__).resolve().parents[1]).as_posix().encode())
        digest.update(source.read_bytes())
    settings = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()
                if k not in ('resume', 'wall_seconds', 'video')}
    settings['source_sha256'] = digest.hexdigest()
    report_path = args.output/'report.json'
    if args.resume:
        if not report_path.exists():
            ap.error('resume requires an existing report.json')
        report = json.loads(report_path.read_text(encoding='utf-8'))
        if report.get('schema') != 3 or report.get('settings') != settings:
            ap.error('resume configuration/source mismatch; preserve old run and use a new directory')
        for attempt in report['attempts']:
            if attempt['status'] == 'running':
                attempt['status'] = 'interrupted'
        if report['status'] in ('completed', 'rejected_validation'):
            report.setdefault('history', []).append({k: report[k] for k in
                ('cycle', 'jobs', 'decision', 'last_training_checkpoint', 'best_validated_checkpoint')})
            report.update(cycle=report['cycle']+1, jobs=[], decision=None, validation_decision=None, status='pending')
    else:
        if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
            ap.error('output must be new or empty; use --resume for this protocol')
        args.output.mkdir(parents=True, exist_ok=True)
        report = {'schema': 3, 'settings': settings, 'cycle': 0, 'jobs': [], 'attempts': [], 'cycles': [],
                      'last_training_checkpoint': None, 'best_validated_checkpoint': None,
                      'decision': None, 'status': 'pending'}
    deadline = start + args.wall_seconds
    atomic_json(report_path, report)
    with WallBudget(deadline):
        cn, scale = load_training_brain(args.data, args.output/'cache')
        report['region_scale'] = scale
        jobs = build_jobs(args.episodes, args.seed, args.validation_seeds, args.heldout_seeds,
                          args.training_arena)
        done = {j['id'] for j in report['jobs']}
        rejected = None
        for job in jobs:
            if job['id'] in done:
                continue
            # Reserve time for saving; episode checks the same monotonic deadline every 5 ms chunk.
            if time.monotonic() >= deadline-5:
                break
            case = argparse.Namespace(**vars(template))
            case.seed, case.arena = job['seed'], job['arena']
            case.learn = job['stage'] == 'training'
            case.seconds = args.seconds if case.learn else 5.
            case.training_fall_seconds = args.training_fall_seconds if case.learn else 0.
            case.motor_exploration = 1.5 if case.learn else 0.
            case.import_weights = (args.import_weights is not None
                                   and report['last_training_checkpoint'] is None and case.learn)
            case.no_environment_senses = job['model'] == 'ablated'
            case.video = args.video
            case.output = args.output/f"cycle_{report['cycle']:03d}"/job['id']/f"attempt_{len(report['attempts']):04d}"
            case.output.mkdir(parents=True, exist_ok=True)
            weights = None if job['model'] == 'original' else (
                report['last_training_checkpoint'] or args.import_weights)
            attempt = dict(**job, start_checkpoint=str(weights) if weights is not None else None, output=str(case.output.resolve()), status='running')
            report['attempts'].append(attempt)
            atomic_json(report_path, report)
            try:
                summary, checkpoint = lab.run_episode(cn, case, 0, weights, deadline=deadline-5)
            except Exception as exc:
                attempt.update(status='error', error=f'{type(exc).__name__}: {exc}')
                atomic_json(report_path, report)
                raise
            attempt.update(status=summary['status'], summary=summary)
            if summary['status'] == 'completed':
                report['jobs'].append(dict(**job, summary=summary))
                if case.learn:
                    report['last_training_checkpoint'] = str(checkpoint.resolve())
            validation_jobs = [j for j in jobs if j['stage'] == 'validation']
            if (report.get('validation_decision') is None and
                    all(j['id'] in {r['id'] for r in report['jobs']} for j in validation_jobs)):
                def validation_trials(arena, model):
                    return [r['summary'] for r in report['jobs'] if r['stage'] == 'validation'
                            and r['arena'] == arena and r['model'] == model]
                report['validation_decision'] = promotion_decision(
                    validation_trials('quiet', 'original'), validation_trials('quiet', 'candidate'))
                report['validation_decision']['obstacle_passed'] = obstacle_decision(
                    validation_trials('obstacle', 'candidate'), validation_trials('obstacle', 'ablated'))
                # Persist before any held-out trial is observed.
            if args.early_reject and job['stage'] == 'validation' and job['arena'] == 'quiet' and job['model'] == 'candidate' and summary['status'] == 'completed':
                original = next(j['summary'] for j in report['jobs'] if j['stage'] == 'validation'
                                and j['arena'] == 'quiet' and j['model'] == 'original' and j['seed'] == job['seed'])
                check = promotion_decision([original], [summary])
                if not check['accepted']:
                    rejected = check
                    # Persist rejection before a hard deadline can interrupt finalization.
                    report['status'] = 'rejected_validation'
                    report['decision'] = {'accepted': False, 'reason': 'completed quiet validation failed; remaining controls not claimed complete', 'quiet_check': check}
            atomic_json(report_path, report)
            print(json.dumps(dict(job=job['id'], **summary)), flush=True)
            if rejected is not None or summary['status'] != 'completed':
                break
        complete = len(report['jobs']) == len(jobs)
        if rejected is not None:
            report['decision'] = {'accepted': False, 'reason': 'completed quiet validation failed; remaining controls not claimed complete', 'quiet_check': rejected}
            report['status'] = 'rejected_validation'
        elif complete:
            decisions = {}
            for stage in ('validation', 'heldout'):
                def trials(arena, model, stage=stage):
                    return [j['summary'] for j in report['jobs'] if j['stage'] == stage
                            and j['arena'] == arena and j['model'] == model]
                decisions[stage] = promotion_decision(trials('quiet', 'original'), trials('quiet', 'candidate'))
                decisions[stage]['obstacle_passed'] = obstacle_decision(
                    trials('obstacle', 'candidate'), trials('obstacle', 'ablated'))
            decisions['validation'] = report['validation_decision']
            accepted = all(d['accepted'] and d['obstacle_passed'] for d in decisions.values())
            report['decision'] = {'accepted': accepted, 'stages': decisions}
            if accepted:
                report['best_validated_checkpoint'] = report['last_training_checkpoint']
            report['status'] = 'completed'
        else:
            report['status'] = 'pending'
            report['decision'] = {'accepted': False, 'reason': 'incomplete controls; no promotion'}
        report['cycles'].append({'cycle': report['cycle'], 'wall_seconds': time.monotonic()-start,
                                     'completed_jobs': len(report['jobs']), 'total_jobs': len(jobs)})
        atomic_json(report_path, report)
        print(json.dumps({'status': report['status'], 'decision': report['decision'],
                              'best_validated_checkpoint': report['best_validated_checkpoint']}), flush=True)


if __name__ == '__main__':
    main()
