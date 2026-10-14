"""Matched, resumable short-term-depression screen using PR #13's P1-P8 unchanged."""
from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from noesis.connectome import DEFAULT_NT_SIGN, NT_OVERRIDES, corrected_scale, load_cached
from noesis.experiments.response_battery import _groups, judge
from noesis.lif import LIFNetwork, LIFParams

_STATE = None
CONDITIONS = ('none', 'sugar', 'water', 'bitter', 'sugar+bitter', 'odour')


def _init(raw, cache, factors):
    global _STATE
    cn = load_cached(Path(raw), Path(cache), region_scale=factors)
    _STATE = cn, _groups(cn, seed=0)


def _trial(task):
    u, tau, seed, condition = task
    cn, groups = _STATE
    stim = (np.concatenate([groups['sugar'], groups['bitter']]) if condition == 'sugar+bitter'
            else groups.get(condition, np.zeros(0, int)))
    net = LIFNetwork(cn.W, LIFParams(dt=.5, da=1., depression_u=u, depression_tau_ms=tau), seed)
    counts = {}
    active = {}
    for name, ms, drive in [('rest', 500., None), ('stim', 1000., stim),
                            ('recovery_early', 500., None), ('recovery_late', 500., None)]:
        _, spikes = net.run(ms, drive, 80. if condition == 'odour' else 100., 400.)
        counts[name] = np.bincount(spikes, minlength=cn.n)
        active[name] = int(np.count_nonzero(counts[name]))
    counts['recovery'] = counts['recovery_early'] + counts['recovery_late']
    active['recovery'] = int(np.count_nonzero(counts['recovery']))
    rates = counts['stim']  # 1000 ms stimulus, so counts equal Hz.
    record = {k: float(rates[groups[k]].mean()) for k in ('mn9', 'pam', 'ppl1', 'pn_on', 'pn_off')}
    record.update({k + '_active': active[k] for k in active})
    record['recovery_late_mean_hz'] = float(counts['recovery_late'].mean() * 2)
    record['recovery_mean_hz'] = float(counts['recovery'].mean())
    return {'u': u, 'tau_ms': tau, 'seed': seed, 'condition': condition, 'metrics': record}


def trace_recovery(u, tau, seed, condition):
    """Resolve offset transients in 50 ms bins through five seconds of recovery."""
    cn, groups = _STATE
    stim = (np.concatenate([groups['sugar'], groups['bitter']]) if condition == 'sugar+bitter'
            else groups.get(condition, np.zeros(0, int)))
    net = LIFNetwork(cn.W, LIFParams(dt=.5, da=1., depression_u=u, depression_tau_ms=tau), seed)
    net.run(500)
    net.run(1000, stim, 80. if condition == 'odour' else 100., 400.)
    offset = net.time_ms
    bins, times, ids = [], [], []
    for k in range(100):
        t, i = net.run(50)
        times.extend(t.tolist())
        ids.extend(i.tolist())
        bins.append({'after_offset_ms': (k + 1) * 50, 'spikes': len(i), 'active': len(np.unique(i))})
    times, ids = np.asarray(times) - offset, np.asarray(ids)
    return {'u': u, 'tau_ms': tau, 'seed': seed, 'condition': condition, 'extended_ms': 5000,
            'last_spike_after_offset_ms': float(times.max()) if len(times) else None,
            'recovery_active': len(np.unique(ids[times < 1000])),
            'recovery_late_active': len(np.unique(ids[(times >= 500) & (times < 1000)])),
            'bins': bins}


def summarize(directory):
    trials = [json.loads(p.read_text()) for p in sorted(directory.glob('trial-*.json'))]
    result = []
    for u, tau in sorted({(r['u'], r['tau_ms']) for r in trials}):
        rows = [r for r in trials if r['u'] == u and r['tau_ms'] == tau]
        seeds = sorted({r['seed'] for r in rows})
        if len(rows) != len(seeds) * len(CONDITIONS):
            continue
        battery = {c: {k: [r['metrics'][k] for r in rows if r['condition'] == c]
                       for k in rows[0]['metrics']} for c in CONDITIONS}
        result.append({'u': u, 'tau_ms': tau, 'seeds': seeds, 'trials': battery,
                       'verdicts': [(n, bool(ok), d) for n, ok, d in judge(battery)]})
    (directory / 'summary.json').write_text(json.dumps(result, indent=2))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--output', type=Path, default=Path('outputs/depression'))
    ap.add_argument('--u', type=float, nargs='+', default=[.1, .3, .6])
    ap.add_argument('--tau-ms', type=float, nargs='+', default=[300., 1000.])
    ap.add_argument('--seeds', type=int, nargs='+', default=[0])
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--summarize-only', action='store_true')
    ap.add_argument('--trace-recovery', choices=CONDITIONS, help='one U/tau/seed, 5-second offset trace')
    a = ap.parse_args()
    if (any(not np.isfinite(u) or not 0 <= u <= 1 for u in a.u)
            or any(not np.isfinite(t) or t <= 0 for t in a.tau_ms)
            or a.workers < 1 or any(s < 0 or s >= 2**32 for s in a.seeds)):
        ap.error('finite U within [0,1], positive tau/workers, uint32 seeds required')
    a.output.mkdir(parents=True, exist_ok=True)
    if a.summarize_only:
        summarize(a.output)
        return
    factors = corrected_scale(a.data)
    if not any(k.startswith('VNC_') for k in factors):
        ap.error('MANC nerve-cord correction missing; this assay requires the fully corrected brain')
    inputs = {}
    for path in [a.data / 'raw/neurons.csv', a.data / 'raw/connections_princeton.csv']:
        digest = hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        inputs[path.name] = digest.hexdigest()
    protocol = {'factors': factors, 'input_sha256': inputs, 'dt_ms': .5, 'adaptation': 1.,
                'rest_ms': 500, 'stimulus_ms': 1000, 'recovery_ms': 1000,
                'drive_weight': 400, 'taste_hz': 100, 'odour_hz': 80,
                'sign_base': 'PR13 APL-only override', 'scope': 'all presynaptic rows',
                'nt_sign': DEFAULT_NT_SIGN, 'nt_overrides': NT_OVERRIDES}
    manifest = a.output / 'protocol.json'
    if manifest.exists() and json.loads(manifest.read_text()) != protocol:
        ap.error('existing protocol differs; use a fresh output directory')
    manifest.write_text(json.dumps(protocol, indent=2))
    cache = a.output / 'cache'
    # Create cache once before workers; avoid concurrent first-write races.
    load_cached(a.data / 'raw', cache, region_scale=factors)
    if a.trace_recovery:
        if any(len(set(values)) != 1 for values in (a.u, a.tau_ms, a.seeds)):
            ap.error('trace requires exactly one U, tau and seed')
        _init(str(a.data / 'raw'), str(cache), factors)
        record = trace_recovery(a.u[0], a.tau_ms[0], a.seeds[0], a.trace_recovery)
        previous = a.output / f'trial-u{a.u[0]:g}-t{a.tau_ms[0]:g}-s{a.seeds[0]}-{a.trace_recovery}.json'
        if previous.exists():
            old = json.loads(previous.read_text())['metrics']
            equal = all(record[k] == old[k] for k in ('recovery_active', 'recovery_late_active'))
            if not equal:
                raise RuntimeError('extended trace changed original recovery counts')
            record['original_window_counts_equal'] = equal
        (a.output / 'extended-recovery.json').write_text(json.dumps(record, indent=2))
        print(record['last_spike_after_offset_ms'], flush=True)
        return
    tasks = []
    for u, tau in [(0., 300.)] + [(u, t) for u in sorted(set(a.u)) if u > 0 for t in sorted(set(a.tau_ms))]:
        for seed in sorted(set(a.seeds)):
            for cond in CONDITIONS:
                name = f'trial-u{u:g}-t{tau:g}-s{seed}-{cond}.json'
                if not (a.output / name).exists():
                    tasks.append((name, (u, tau, seed, cond)))
    print(f'{len(tasks)} trials pending', flush=True)
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init,
                             initargs=(str(a.data / 'raw'), str(cache), factors)) as pool:
        futures = {pool.submit(_trial, task): name for name, task in tasks}
        for i, future in enumerate(as_completed(futures), 1):
            name = futures[future]
            record = future.result()
            (a.output / name).write_text(json.dumps(record))
            print(f'{i}/{len(tasks)} {name}: {record["metrics"]}', flush=True)
    for record in summarize(a.output):
        print(record['u'], record['tau_ms'], record['seeds'], record['verdicts'], flush=True)


if __name__ == '__main__':
    main()
