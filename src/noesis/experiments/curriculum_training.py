"""Resumable training-only curriculum. Final validation and deployment stay separate."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

from noesis.curriculum import STAGES, phase_decision, phase_jobs


def atomic_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    # Windows scanners can briefly hold the destination; never discard the ledger.
    for attempt in range(9):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 8:
                raise
            time.sleep(.025 * (attempt + 1))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_digest(root):
    digest = hashlib.sha256()
    for source in sorted(root.rglob('*.py')):
        digest.update(source.relative_to(root).as_posix().encode())
        digest.update(source.read_bytes())
    return digest.hexdigest()


def episode_args(lab, data, output, job):
    """One shared model in both arms; only training reward differs."""
    args = lab.parser().parse_args(['--data', str(data), '--output', str(output),
        '--episodes', '1', '--seconds', '5', '--seed', str(job['seed']),
        '--arena', job['arena'], '--grounded-objective', job['objective'],
        '--bg-weight', '40', '--learning-rate', '.2', '--learning-scope', 'motor-circuit-v1',
        '--normalize-reward-error', '--bias-correct-reward-error'])
    args.learn = job['learning']
    args.motor_exploration = 1.5 if args.learn else 0.
    args.training_fall_seconds = .1 if args.learn else 0.
    args.no_environment_senses = job['model'] == 'ablated'
    # Carry only weights between births and across objective changes. This resets
    # eligibility and reward moments while preserving original synapse references.
    args.import_weights = job['model'] != 'original'
    lab.validate_args(lab.parser(), args)
    return args


def run_round(state, jobs, output, data, cn, lab, deadline):
    """Commit only completed episodes; interrupted attempts restart from their parent."""
    done = {j['id'] for j in state['round_jobs']}
    ledger = output / 'curriculum.json'
    for job in jobs:
        if job['id'] in done:
            continue
        if time.monotonic() >= deadline - 5:
            return False
        case_output = output / f"round_{state['round']:03d}" / job['id'] / f"attempt_{len(state['attempts']):04d}"
        args = episode_args(lab, data, case_output, job)
        parent = None if job['model'] == 'original' else state['latest_training_checkpoint']
        if parent is not None and sha(parent) != state['latest_training_sha256']:
            raise ValueError('latest checkpoint hash mismatch')
        attempt = dict(job, output=str(case_output), start_checkpoint=parent,
                       start_sha256=sha(parent) if parent else None, status='running')
        state['attempts'].append(attempt)
        atomic_json(ledger, state)
        case_output.mkdir(parents=True)
        try:
            summary, checkpoint = lab.run_episode(cn, args, 0, parent, deadline=deadline - 5)
        except Exception as exc:
            attempt.update(status='error', error=f'{type(exc).__name__}: {exc}')
            atomic_json(ledger, state)
            raise
        attempt.update(status=summary['status'], summary=summary)
        if summary['status'] == 'completed':
            state['round_jobs'].append(dict(job, summary=summary))
            if args.learn:
                state['latest_training_checkpoint'] = str(checkpoint.resolve())
                state['latest_training_sha256'] = sha(checkpoint)
                state['completed_training_seconds'] += summary['simulated_seconds']
        atomic_json(ledger, state)
        print(json.dumps(dict(round=state['round'], stage=STAGES[state['stage']][0],
                              job=job['id'], **summary)), flush=True)
        if summary['status'] != 'completed':
            return False
    return True


def finish_round(state):
    def summaries(model):
        return [j['summary'] for j in state['round_jobs']
                if not j['learning'] and j['model'] == model]
    decision = phase_decision(state['stage'], summaries('original'), summaries('candidate'),
                              summaries('ablated'))
    state['history'].append({'round': state['round'], 'stage': state['stage'],
                                 'jobs': state['round_jobs'], 'decision': decision})
    state['round'] += 1
    state['round_jobs'] = []
    state['stage_rounds'] += 1
    if decision['accepted']:
        state['stage_passed_checkpoint'] = state['latest_training_checkpoint']
        state['stage'] += 1
        state['stage_rounds'] = 0
        if state['stage'] == len(STAGES):
            state['status'] = 'ready_for_final_validation'
    elif state['stage_rounds'] == 3:
        state['status'] = 'stage_not_learned'
    return decision


def main():
    started = time.monotonic()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--import-weights', required=True, type=Path)
    ap.add_argument('--wall-seconds', type=float, default=3600.)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--fixed-objective', action='store_true',
                    help='comparison arm: same schedule/probes with v6 reward throughout')
    args = ap.parse_args()
    if not math.isfinite(args.wall_seconds) or not 0 < args.wall_seconds <= 3600:
        ap.error('wall-seconds must be in (0,3600]')
    args.data, args.output, args.import_weights = (p.resolve() for p in
                                                   (args.data, args.output, args.import_weights))
    if not args.import_weights.is_file():
        ap.error('parent checkpoint missing')
    if args.output == args.data or args.data in args.output.parents:
        ap.error('outputs must stay outside source data')
    settings = {'data': str(args.data), 'parent': str(args.import_weights),
                    'parent_sha256': sha(args.import_weights), 'fixed_objective': args.fixed_objective,
                    'source_sha256': source_digest(Path(__file__).resolve().parents[1]),
                    'episodes_per_round': 6, 'rounds_per_stage': 3, 'stages': STAGES,
                    'final_validation_seeds': [10, 11], 'final_heldout_seeds': [200, 201, 202]}
    # Normalize tuples before comparing against JSON on resume.
    settings = json.loads(json.dumps(settings))
    ledger = args.output / 'curriculum.json'
    if args.resume:
        state = json.loads(ledger.read_text(encoding='utf-8'))
        if state['schema'] != 1 or state['settings'] != settings:
            ap.error('resume settings/source mismatch')
        if state['status'] != 'pending':
            ap.error('terminal curriculum must be reviewed; use a new output for another experiment')
        for attempt in state['attempts']:
            if attempt['status'] == 'running':
                attempt['status'] = 'interrupted'
    else:
        if args.output.exists() and any(args.output.iterdir()):
            ap.error('new output required; use --resume for interrupted work')
        args.output.mkdir(parents=True, exist_ok=True)
        state = {'schema': 1, 'settings': settings, 'status': 'pending', 'stage': 0, 'stage_rounds': 0,
                     'round': 0, 'round_jobs': [], 'attempts': [], 'history': [], 'completed_training_seconds': 0.,
                     'latest_training_checkpoint': str(args.import_weights),
                     'latest_training_sha256': settings['parent_sha256'], 'stage_passed_checkpoint': None,
                     'best_validated_checkpoint': None, 'model_promoted': False, 'normal_startup_changed': False}
    atomic_json(ledger, state)
    # Lazy imports keep curriculum policy tests free from simulator/full-connectome setup.
    from noesis.embodiment import load_training_brain
    from noesis.experiments import learning_lab as lab
    from noesis.wall_budget import WallBudget
    deadline = started + args.wall_seconds
    with WallBudget(deadline):
        cn, _ = load_training_brain(args.data, args.output / 'cache')
        print(json.dumps({'event': 'curriculum_ready', 'neurons': cn.n, 'stage': state['stage'],
                              'source_sha256': settings['source_sha256'],
                              'parent_sha256': state['latest_training_sha256'],
                              'best_validated_checkpoint': None}), flush=True)
        while state['status'] == 'pending' and time.monotonic() < deadline - 5:
            jobs = phase_jobs(state['stage'], state['round'], fixed_objective=args.fixed_objective)
            if not run_round(state, jobs, args.output, args.data, cn, lab, deadline):
                break
            decision = finish_round(state)
            atomic_json(ledger, state)
            print(json.dumps({'event': 'training_stage_gate', 'decision': decision,
                                  'status': state['status'], 'stage': state['stage']}), flush=True)
    atomic_json(ledger, state)
    print(json.dumps({'status': state['status'], 'stage': state['stage'],
                          'latest_training_checkpoint': state['latest_training_checkpoint'],
                          'best_validated_checkpoint': None, 'model_promoted': False}), flush=True)


if __name__ == '__main__':
    main()
