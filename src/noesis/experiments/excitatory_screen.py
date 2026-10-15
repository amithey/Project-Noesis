"""Resumable excitatory-only depression screen; original P1-P8 thresholds unchanged."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from noesis import lif
from noesis.embodiment import load_training_brain
from noesis.experiments.response_battery import _groups, judge
from noesis.experiments.train_validate import atomic_json
from noesis.lif import LIFNetwork, LIFParams
from noesis.wall_budget import WallBudget

CONDITIONS = ('none', 'sugar', 'water', 'bitter', 'sugar+bitter', 'odour')


def trial(cn, groups, u, seed, condition, deadline):
    stim = (np.concatenate([groups['sugar'], groups['bitter']]) if condition == 'sugar+bitter'
            else groups.get(condition, np.empty(0, int)))
    net = LIFNetwork(cn.W, LIFParams(dt=.5, da=1., depression_u=u,
                                   depression_tau_ms=300., depression_excitatory_only=True), seed)
    windows = {}
    for name, duration in [('rest', 500), ('stim', 1000), ('recovery', 1000)]:
        counts = np.zeros(cn.n, dtype=np.int64)
        for _ in range(duration//50):
            if time.monotonic() >= deadline:
                return None
            _, spikes = net.run(50, stim if name == 'stim' else None,
                                 80. if condition == 'odour' else 100., 400.)
            counts += np.bincount(spikes, minlength=cn.n)
        windows[name] = counts
    record = {k: float(windows['stim'][groups[k]].mean())
              for k in ('mn9', 'pam', 'ppl1', 'pn_on', 'pn_off')}
    record.update({k+'_active': int(np.count_nonzero(v)) for k, v in windows.items()})
    record['recovery_mean_hz'] = float(windows['recovery'].mean())
    return record


def screen_decision(batteries):
    baseline = judge(batteries[0.])
    decisions = []
    for u in (.03, .1):
        verdicts = judge(batteries[u])
        preserved = all(not before[1] or after[1] for before, after in zip(baseline, verdicts, strict=True))
        before = np.mean([np.mean(batteries[0.][c]['recovery_mean_hz']) for c in CONDITIONS[1:]])
        after = np.mean([np.mean(batteries[u][c]['recovery_mean_hz']) for c in CONDITIONS[1:]])
        # Must pass original recovery test, improve offset activity, preserve every passing sensory assay.
        accepted = bool(preserved and verdicts[-1][1] and after < .9*before)
        decisions.append({'u': u, 'accepted': accepted, 'sensory_preserved': bool(preserved),
                          'recovery_before_hz': float(before), 'recovery_after_hz': float(after),
                          'verdicts': [(n, bool(ok), detail) for n, ok, detail in verdicts]})
    return decisions


def main():
    start = time.monotonic()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--seeds', type=int, nargs='+', default=[10, 11])
    ap.add_argument('--wall-seconds', type=float, default=3600.)
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()
    if not np.isfinite(args.wall_seconds) or not 0 < args.wall_seconds <= 3600:
        ap.error('wall-seconds must be in (0, 3600]')
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 or s >= 2**32 for s in args.seeds):
        ap.error('seeds must be unique unsigned 32-bit integers')
    if args.output.resolve() == args.data.resolve() or args.data.resolve() in args.output.resolve().parents:
        ap.error('output must be outside source data')
    settings = {'data': str(args.data.resolve()), 'seeds': args.seeds,
                'lif_sha256': hashlib.sha256(Path(lif.__file__).read_bytes()).hexdigest(),
                'scope': 'excitatory-only', 'tau_ms': 300.}
    path = args.output/'report.json'
    if args.resume:
        report = json.loads(path.read_text(encoding='utf-8'))
        if report['settings'] != settings:
            ap.error('resume configuration mismatch')
    else:
        if args.output.exists() and any(args.output.iterdir()):
            ap.error('use a new output directory or --resume')
        args.output.mkdir(parents=True, exist_ok=True)
        report = {'settings': settings, 'trials': [], 'status': 'pending', 'decisions': []}
    atomic_json(path, report)
    deadline = start+args.wall_seconds-5
    with WallBudget(deadline+5):
        cn, _ = load_training_brain(args.data, args.output/'cache')
        groups = _groups(cn, seed=0)
        done = {(r['u'], r['seed'], r['condition']) for r in report['trials']}
        for u in (0., .03, .1):
            for seed in args.seeds:
                for condition in CONDITIONS:
                    if (u, seed, condition) in done:
                        continue
                    metrics = trial(cn, groups, u, seed, condition, deadline)
                    if metrics is None:
                        atomic_json(path, report)
                        print('Budget reached; incomplete trial will restart on --resume', flush=True)
                        return
                    report['trials'].append({'u': u, 'seed': seed, 'condition': condition, 'metrics': metrics})
                    atomic_json(path, report)
                    print(json.dumps(report['trials'][-1]), flush=True)
        batteries = {}
        for u in (0., .03, .1):
            batteries[u] = {c: {k: [r['metrics'][k] for r in report['trials']
                                    if r['u'] == u and r['condition'] == c]
                                for k in report['trials'][0]['metrics']} for c in CONDITIONS}
        report.update(status='completed', decisions=screen_decision(batteries))
        atomic_json(path, report)
        print(json.dumps(report['decisions']), flush=True)


if __name__ == '__main__':
    main()
