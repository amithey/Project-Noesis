"""Controlled olfactory-layer assays on the FAFB-region-scaled BANC brain.

Fresh network per odour; paired seeds, fixed disjoint glomerulus sets, no learning.
No KC class-pair rescaling or raw weak-edge restoration is applied. Results resume
at paired-trial boundaries and include onset-inclusive and late-window metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from noesis.connectome import (
    DEFAULT_NT_SIGN,
    UNKNOWN_NT_SIGN,
    Connectome,
    load_cached,
    neuropil_scale_factors,
)
from noesis.lif import LIFNetwork, LIFParams
from noesis.mushroom_body import GradedAPL, populations

BASELINE_NT_OVERRIDES = {"APL": "GABA"}  # Frozen PR #11 no-change control.
CHUNK_MS = 5.0
_STATE = None


def frozen_signs(neurons, overrides):
    """Signs independent of future global overrides, for reproducible controls."""
    nt = neurons['Predicted NT type'].copy()
    for cell_type, transmitter in overrides.items():
        nt.loc[neurons['Primary Cell Type'].eq(cell_type)] = transmitter
    return nt.map(DEFAULT_NT_SIGN).fillna(UNKNOWN_NT_SIGN).to_numpy(np.float32)


def resign_weights(weights, neurons, overrides):
    """Copy CSR magnitudes/topology, changing only requested presynaptic signs."""
    out = weights.copy().tocsr()
    rows = np.repeat(np.arange(out.shape[0]), np.diff(out.indptr))
    out.data = abs(out.data) * frozen_signs(neurons, overrides)[rows]
    return out


def orn_glomerulus(label: str) -> str | None:
    """Canonical labels, excluding generic ORN and preserving subglomeruli."""
    if label.startswith('ORN_') and len(label) > 4:
        return label[4:]
    if label.endswith('_ORN') and len(label) > 4:
        return label[:-4]
    return None


def pn_glomerulus(label: str, available: set[str]) -> str | None:
    """Only explicitly labelled uniglomerular PNs; not M_*, CB*, or '+' mixtures."""
    stem = label.split('_', 1)[0]
    return stem if '_' in label and stem in available else None


def make_pairs(available, count=5, size=5, seed=20261013):
    """Use globally disjoint sets if possible; otherwise fail rather than reuse."""
    available = sorted(set(available))
    if count < 1 or size < 1 or len(available) < 2 * count * size:
        raise ValueError('not enough glomeruli for globally disjoint odour pairs')
    order = np.random.default_rng(seed).permutation(available)
    return [{'A': sorted(order[2 * i * size:(2 * i + 1) * size].tolist()),
             'B': sorted(order[(2 * i + 1) * size:(2 * i + 2) * size].tolist())}
            for i in range(count)]


def rate_metrics(counts, duration_ms):
    counts = np.asarray(counts)
    if duration_ms <= 0 or not np.isfinite(duration_ms) or counts.ndim != 1:
        raise ValueError('finite positive duration and one-dimensional counts required')
    if not np.isfinite(counts).all() or np.any(counts < 0):
        raise ValueError('counts must be finite and nonnegative')
    if not len(counts):
        return {'n': 0, 'mean_hz': None, 'active_fraction': None, 'ge5hz_fraction': None}
    rates = counts / (duration_ms * 1e-3)
    return {'n': len(counts), 'mean_hz': float(rates.mean()),
            'active_fraction': float(np.mean(counts > 0)),
            'ge5hz_fraction': float(np.mean(rates >= 5))}


def overlap_metrics(a, b):
    """Jaccard plus chance expectation; silence is undefined, not perfect coding."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.ndim != 1 or not len(a):
        raise ValueError('equal nonempty count vectors required')
    if not np.isfinite(a).all() or not np.isfinite(b).all() or np.any(a < 0) or np.any(b < 0):
        raise ValueError('counts must be finite and nonnegative')
    active_a, active_b = a > 0, b > 0
    union = int(np.count_nonzero(active_a | active_b))
    intersection = int(np.count_nonzero(active_a & active_b))
    pa, pb = active_a.mean(), active_b.mean()
    expected_union = pa + pb - pa * pb
    denom = np.linalg.norm(a.astype(float)) * np.linalg.norm(b.astype(float))
    return {'intersection': intersection, 'union': union,
            'jaccard': intersection / union if union else None,
            'chance_jaccard': float(pa * pb / expected_union) if expected_union else None,
            'cosine': float(a.astype(float) @ b / denom) if denom else None}


def spread(values):
    """Descriptive distribution, not a CI treating repeated seeds as new odours."""
    values = np.asarray([v for v in values if v is not None], float)
    if not len(values):
        return {'n': 0, 'mean': None, 'sd': None, 'min': None, 'max': None}
    return {'n': len(values), 'mean': float(values.mean()),
            'sd': float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            'min': float(values.min()), 'max': float(values.max())}


def calibrate_gains(trials, window='full', expected_pairs=5, seeds=(0, 1, 2), treatment='control'):
    """Fit mean 5–10% sparseness on pairs 0–2; keep pairs 3+ as held-out odours.

    Never select an incomplete gain or call silence a calibrated representation.
    Seeds are repeated stochastic trials, not independent biological specimens.
    """
    gains = sorted({t['gain'] for t in trials if t['gain'] is not None and t['treatment'] == treatment})
    curve = []
    expected = {(p, s) for p in range(expected_pairs) for s in seeds}
    for gain in gains:
        subset = [t for t in trials if t['gain'] == gain and t['treatment'] == treatment]
        complete = {(t['pair'], t['seed']) for t in subset} == expected and len(subset) == len(expected)
        calibration = [t[o][window]['kc']['active_fraction'] for t in subset
                       if t['pair'] < 3 for o in ('A', 'B')]
        held_out = [t[o][window]['kc']['active_fraction'] for t in subset
                    if t['pair'] >= 3 for o in ('A', 'B')]
        curve.append({'gain': gain, 'complete': complete, 'calibration': spread(calibration),
                      'held_out': spread(held_out)})
    eligible = [r for r in curve if r['complete'] and r['calibration']['mean'] is not None
                and .05 <= r['calibration']['mean'] <= .10]
    chosen = min(eligible, key=lambda r: (abs(r['calibration']['mean'] - .075), r['gain'])) if eligible else None
    return {'window': window, 'selected_gain': chosen['gain'] if chosen else None, 'curve': curve}


def audit_local_neurons(cn, raw):
    """Raw synapse totals and actually retained corrected weights, by type/NT."""
    n = cn.neurons
    ln = np.flatnonzero(n['Class'].fillna('').str.contains('antennal_lobe_local').to_numpy())
    pn = populations(n).pn
    block = cn.W[ln][:, pn]
    table = n.iloc[ln][['Primary Cell Type', 'Predicted NT type', 'Hemilineage']].copy()
    table.columns = ['cell_type', 'predicted_nt', 'hemilineage']
    table = table.fillna('<unknown>')
    table['neurons'] = 1
    table['corrected_retained_weight'] = np.asarray(abs(block).sum(axis=1)).ravel()
    table['retained_pairs'] = block.getnnz(axis=1)
    table['effective_sign'] = frozen_signs(n, BASELINE_NT_OVERRIDES)[ln]
    conn = pd.read_csv(raw / 'connections_princeton.csv',
                       usecols=['pre_root_id', 'post_root_id', 'syn_count'])
    conn = conn[conn.pre_root_id.isin(n.iloc[ln].root_id) & conn.post_root_id.isin(n.iloc[pn].root_id)]
    raw_sums = conn.groupby('pre_root_id').syn_count.sum()
    table['raw_synapses'] = n.iloc[ln].root_id.map(raw_sums).fillna(0).to_numpy()
    return table.groupby(['cell_type', 'predicted_nt', 'effective_sign'], as_index=False).agg(
        neurons=('neurons', 'sum'), raw_synapses=('raw_synapses', 'sum'),
        corrected_retained_weight=('corrected_retained_weight', 'sum'),
        retained_pairs=('retained_pairs', 'sum'),
        hemilineages=('hemilineage', lambda x: ';'.join(sorted(set(x)))))


def apl_coverage_audit(cn, raw_connections, reference_connections, reference_types):
    """Distinguish absent raw APL edges from threshold loss, and compare FAFB."""
    pops = populations(cn.neurons)
    out = np.asarray(abs(cn.W[pops.apl][:, pops.kc]).sum(axis=0)).ravel()
    kc_ids = set(cn.neurons.iloc[pops.kc].root_id)
    apl_ids = set(cn.neurons.iloc[pops.apl].root_id)

    def incoming(path, pre_ids, post_ids):
        totals = pd.Series(dtype=float)
        for chunk in pd.read_csv(path, usecols=['pre_root_id', 'post_root_id', 'syn_count'], chunksize=250000):
            selected = chunk[chunk.pre_root_id.isin(pre_ids) & chunk.post_root_id.isin(post_ids)]
            totals = totals.add(selected.groupby('post_root_id').syn_count.sum(), fill_value=0)
        return totals

    raw = incoming(raw_connections, apl_ids, kc_ids)
    types = pd.read_csv(reference_types)
    ref_kc = set(types.loc[types.primary_type.fillna('').str.startswith('KC'), 'root_id'])
    ref_apl = set(types.loc[types.primary_type.eq('APL'), 'root_id'])
    reference = incoming(reference_connections, ref_apl, ref_kc)
    uncovered = pops.kc[out == 0]
    return {'banc_kcs': len(kc_ids), 'corrected_uncovered_kcs': len(uncovered),
            'raw_uncovered_kcs': len(kc_ids - set(raw.index)),
            'corrected_apl_weight_quantiles': np.quantile(out, [0, .1, .5, .9, 1]).tolist(),
            'uncovered_by_subclass': cn.neurons.iloc[uncovered]['Sub Class'].value_counts().to_dict(),
            'fafb_kcs': len(ref_kc), 'fafb_uncovered_kcs': len(ref_kc - set(reference.index)),
            'fafb_apl_synapses_per_kc_mean': float(reference.reindex(list(ref_kc), fill_value=0).mean()),
            'banc_apl_raw_synapses_per_kc_mean': float(raw.reindex(list(kc_ids), fill_value=0).mean())}


def _init_worker(raw, cache, factors, protocol):
    global _STATE
    cn = load_cached(Path(raw), Path(cache), region_scale=factors)
    cn = Connectome(cn.neurons, resign_weights(cn.W, cn.neurons, protocol['nt_overrides']))
    pops = populations(cn.neurons)
    ctype = cn.neurons['Primary Cell Type'].fillna('')
    orn_glom = np.array([orn_glomerulus(t) for t in ctype.iloc[pops.orn]], object)
    available = set(orn_glom) - {None}
    pn_glom = np.array([pn_glomerulus(t, available) for t in ctype.iloc[pops.pn]], object)
    ln = np.flatnonzero(cn.neurons['Class'].fillna('').str.contains('antennal_lobe_local').to_numpy())
    _STATE = cn, pops, orn_glom, pn_glom, ln, protocol


def _odour_trial(glomeruli, seed, gain, treatment):
    cn, pops, orn_glom, pn_glom, ln, protocol = _STATE
    W = cn.W
    # A diagnostic intervention, not a proposed biological NT reassignment.
    # Cholinergic LNs provide lateral excitation (Shang et al. 2007,
    # https://doi.org/10.1016/j.cell.2006.12.034). Removing their PN edges tests causality.
    if treatment == 'ln_excitation_off':
        W = W.copy()
        selected = ln[frozen_signs(cn.neurons, BASELINE_NT_OVERRIDES)[ln] > 0]
        for pre in selected:
            lo, hi = W.indptr[pre:pre + 2]
            W.data[lo:hi][np.isin(W.indices[lo:hi], pops.pn)] = 0
    if treatment == 'il3ln6_gaba':
        W = resign_weights(W, cn.neurons, {**BASELINE_NT_OVERRIDES, 'il3LN6': 'GABA'})
    net = LIFNetwork(W, LIFParams(dt=0.5, da=1.0), seed=seed)
    apl = GradedAPL(net, pops, gain=gain) if gain is not None else None
    stim = pops.orn[np.isin(orn_glom, glomeruli)]
    full = np.zeros(cn.n, np.int32)
    late = np.zeros(cn.n, np.int32)
    current = {}
    window_ms = protocol.get('feedback_ms', CHUNK_MS)
    n_chunks = round(protocol['duration_ms'] / window_ms)
    for k in range(n_chunks):
        _, spikes = net.run(window_ms, stim, protocol['odor_hz'], protocol['drive_weight'], **current)
        counts = np.bincount(spikes, minlength=cn.n)
        full += counts.astype(np.int32)
        if k * window_ms >= protocol['burn_ms']:
            late += counts.astype(np.int32)
        if apl:
            current = apl.current(spikes, window_ms)
    on = np.isin(pn_glom, glomeruli)
    known = np.array([g is not None for g in pn_glom])
    result = {}
    for window, counts, ms in [('full', full, protocol['duration_ms']),
                               ('late', late, protocol['duration_ms'] - protocol['burn_ms'])]:
        result[window] = {name: rate_metrics(counts[ids], ms) for name, ids in (
            ('orn_driven', stim), ('pn_on', pops.pn[on]), ('pn_off_all', pops.pn[~on]),
            ('pn_off_uniglomerular', pops.pn[known & ~on]), ('pn_unassigned', pops.pn[~known]),
            ('kc', pops.kc), ('ln', ln), ('apl', pops.apl), ('brain', np.arange(cn.n)))}
    return result, np.stack([full[pops.kc], late[pops.kc]]), np.stack([full[pops.pn], late[pops.pn]])


def _paired_trial(task):
    gain, treatment, pair_index, pair, seed = task
    start = time.monotonic()
    a, kc_a, pn_a = _odour_trial(pair['A'], seed, gain, treatment)
    b, kc_b, pn_b = _odour_trial(pair['B'], seed, gain, treatment)
    return {'gain': gain, 'treatment': treatment, 'pair': pair_index, 'seed': seed,
            'glomeruli': pair, 'A': a, 'B': b,
            'overlap': {name: overlap_metrics(kc_a[i], kc_b[i]) for i, name in enumerate(('full', 'late'))},
            'pn_overlap': {name: overlap_metrics(pn_a[i], pn_b[i]) for i, name in enumerate(('full', 'late'))},
            'wall_seconds': time.monotonic() - start}, kc_a, kc_b, pn_a, pn_b


def summarize(directory):
    trials = [json.loads(p.read_text()) for p in sorted(directory.glob('trial-*.json'))]
    rows = []
    for t in trials:
        for window in ('full', 'late'):
            for odour in ('A', 'B'):
                row = {'gain': t['gain'], 'treatment': t['treatment'], 'pair': t['pair'],
                       'seed': t['seed'], 'window': window, 'odour': odour}
                for layer, stats in t[odour][window].items():
                    row.update({f'{layer}_{key}': value for key, value in stats.items()})
                row.update({f'kc_{key}': value for key, value in t['overlap'][window].items()})
                row.update({f'pn_{key}': value for key, value in t['pn_overlap'][window].items()})
                rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(directory / 'measurements.csv', index=False)
    summaries = []
    if len(frame):
        for (treatment, gain, window), group in frame.groupby(['treatment', 'gain', 'window'], dropna=False):
            metrics = {}
            paired_fields = {f'{layer}_{metric}' for layer in ('kc', 'pn')
                             for metric in ('intersection', 'union', 'jaccard', 'chance_jaccard', 'cosine')}
            for column in frame.columns:
                if column in ('gain', 'treatment', 'pair', 'seed', 'window', 'odour'):
                    continue
                source = group.drop_duplicates(['pair', 'seed']) if column in paired_fields else group
                metrics[column] = spread(source[column].replace({np.nan: None}).tolist())
            summaries.append({'treatment': treatment, 'gain': None if pd.isna(gain) else float(gain),
                              'window': window, 'odour_trials': len(group), 'metrics': metrics})
    (directory / 'summary.json').write_text(json.dumps(summaries, indent=2, allow_nan=False))
    (directory / 'calibration.json').write_text(json.dumps(
        {t: {w: calibrate_gains(trials, window=w, treatment=t) for w in ('full', 'late')}
         for t in sorted({x['treatment'] for x in trials})}, indent=2, allow_nan=False))
    return summaries


def controls(pair, seeds, treatment='control'):
    """Zero-input silence and identical re-run, using fresh state and the same seed."""
    silence = []
    for seed in seeds:
        for gain in (None, 1024.0):
            result, _, _ = _odour_trial([], seed, gain, treatment)
            silence.append({'seed': seed, 'gain': gain, 'brain': result['full']['brain']})
    first = _odour_trial(pair['A'], seeds[0], None, treatment)
    repeat = _odour_trial(pair['A'], seeds[0], None, treatment)
    identical = first[0] == repeat[0] and all(np.array_equal(a, b) for a, b in zip(first[1:], repeat[1:]))
    if not identical or any(t['brain']['mean_hz'] != 0 for t in silence):
        raise RuntimeError('silence or no-change repeat control failed')
    return {'silence': silence, 'no_change_repeat_identical': identical}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--output', type=Path, default=Path('outputs/olfactory-specificity'))
    ap.add_argument('--gains', type=float, nargs='*', default=[1, 4, 16, 64, 256, 1024, 4096])
    ap.add_argument('--pairs', type=int, default=5)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--pair-seed', type=int, default=20261013)
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--cache', type=Path, help='optional private corrected cache from an earlier run')
    ap.add_argument('--skip-spiking', action='store_true', help='run only requested graded gains')
    ap.add_argument('--feedback-ms', type=float, default=CHUNK_MS,
                    help='graded feedback update interval (0.5..5 ms, multiples of neural dt)')
    ap.add_argument('--duration-ms', type=float, default=600.0)
    ap.add_argument('--burn-ms', type=float, default=100.0)
    ap.add_argument('--odor-hz', type=float, default=80.0)
    ap.add_argument('--drive-weight', type=float, default=400.0)
    ap.add_argument('--audit-only', action='store_true')
    ap.add_argument('--summarize-only', action='store_true')
    ap.add_argument('--il3ln6-gaba', action='store_true', help='source-backed il3LN6 sign correction')
    ap.add_argument('--ln-excitation-off', action='store_true', help='diagnostic ablation, not an NT override')
    a = ap.parse_args()
    if a.il3ln6_gaba and a.ln_excitation_off:
        ap.error('one intervention at a time')
    if a.workers < 1 or a.pairs < 1 or any(s < 0 or s >= 2**32 for s in a.seeds):
        ap.error('positive workers/pairs and uint32 seeds required')
    if any(not np.isfinite(g) or g < 0 for g in a.gains):
        ap.error('gains must be finite and nonnegative')
    if not np.isfinite(a.feedback_ms) or not 0.5 <= a.feedback_ms <= 5 or a.feedback_ms % 0.5:
        ap.error('--feedback-ms must be a multiple of 0.5 within [0.5, 5]')
    if (not 0 <= a.burn_ms < a.duration_ms or
            any(not np.isfinite(x) or x % CHUNK_MS for x in (a.duration_ms, a.burn_ms))):
        ap.error('duration/burn must be multiples of 5 ms, 0 <= burn < duration')
    if any(not np.isfinite(x) or x < 0 for x in (a.odor_hz, a.drive_weight)):
        ap.error('stimulus parameters must be finite and nonnegative')
    a.output.mkdir(parents=True, exist_ok=True)
    if a.summarize_only:
        summarize(a.output)
        return
    raw = a.data / 'raw'
    ref = a.data / 'fafb/connections_princeton.csv.gz'
    factors = neuropil_scale_factors(raw / 'connections_princeton.csv', ref)
    # A private cache avoids writing in the shared main checkout and avoids the
    # base loader's coarse _scaled cache key mixing different region factors.
    inputs = {}
    for path in (raw / 'neurons.csv', raw / 'connections_princeton.csv', ref):
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                digest.update(block)
        inputs[str(path.resolve())] = digest.hexdigest()
    fingerprint = hashlib.sha256(json.dumps([factors, inputs], sort_keys=True).encode()).hexdigest()[:16]
    cache = a.cache or a.output / ('cache-' + fingerprint)
    cn = load_cached(raw, cache, region_scale=factors)
    cn = Connectome(cn.neurons, resign_weights(cn.W, cn.neurons, BASELINE_NT_OVERRIDES))
    pops = populations(cn.neurons)
    ctype = cn.neurons['Primary Cell Type'].fillna('')
    orn_names = {orn_glomerulus(t) for t in ctype.iloc[pops.orn]} - {None}
    pn_names = {pn_glomerulus(t, orn_names) for t in ctype.iloc[pops.pn]} - {None}
    pairs = make_pairs(orn_names & pn_names, count=a.pairs, seed=a.pair_seed)
    protocol = {'duration_ms': a.duration_ms, 'burn_ms': a.burn_ms,
                'odor_hz': a.odor_hz, 'drive_weight': a.drive_weight,
                'dt_ms': 0.5, 'adaptation': 1.0, 'background': 0,
                'nt_overrides': BASELINE_NT_OVERRIDES, 'input_sha256': inputs,
                'pairs': pairs, 'region_scale': factors, 'edges': cn.W.nnz,
                'populations': {k: len(getattr(pops, k)) for k in ('orn', 'pn', 'kc', 'apl')},
                'data': str(a.data.resolve())}
    if a.feedback_ms != CHUNK_MS:
        protocol['feedback_ms'] = a.feedback_ms
    manifest = a.output / 'protocol.json'
    if manifest.exists() and json.loads(manifest.read_text()) != protocol:
        ap.error('output contains a different protocol; choose a fresh directory')
    manifest.write_text(json.dumps(protocol, indent=2))
    audit_local_neurons(cn, raw).to_csv(a.output / 'al_local_neuron_audit.csv', index=False)
    coverage_file = a.output / 'apl_coverage.json'
    if not coverage_file.exists():
        coverage = apl_coverage_audit(cn, raw / 'connections_princeton.csv', ref,
                                      a.data / 'fafb/consolidated_cell_types.csv.gz')
        coverage_file.write_text(json.dumps(coverage, indent=2))
    if a.audit_only:
        print(json.dumps(protocol, indent=2), flush=True)
        return
    if not (a.output / 'controls.json').exists():
        _init_worker(str(raw), str(cache), factors, protocol)
        (a.output / 'controls.json').write_text(json.dumps(controls(pairs[0], a.seeds), indent=2))
        print('Silence and no-change repeat controls passed', flush=True)
    tasks = []
    treatments = ['control']
    if a.ln_excitation_off:
        treatments.append('ln_excitation_off')
    if a.il3ln6_gaba:
        treatments.append('il3ln6_gaba')
    for treatment in treatments:
        control_file = a.output / ('controls.json' if treatment == 'control' else f'controls-{treatment}.json')
        if not control_file.exists():
            _init_worker(str(raw), str(cache), factors, protocol)
            control_file.write_text(json.dumps(controls(pairs[0], a.seeds, treatment), indent=2))
        for gain in ([] if a.skip_spiking else [None]) + sorted(set(a.gains)):
            for pair_index, pair in enumerate(pairs):
                for seed in sorted(set(a.seeds)):
                    label = 'spiking' if gain is None else f'g{gain:g}'
                    name = f'trial-{treatment}-{label}-p{pair_index}-s{seed}'
                    if not (a.output / (name + '.json')).exists():
                        tasks.append((name, (gain, treatment, pair_index, pair, seed)))
    print(f'{len(tasks)} paired jobs pending, {cn.W.nnz:,} corrected pairs', flush=True)
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init_worker,
                             initargs=(str(raw), str(cache), factors, protocol)) as pool:
        futures = {pool.submit(_paired_trial, task): name for name, task in tasks}
        for completed, future in enumerate(as_completed(futures), 1):
            name = futures[future]
            result, *counts = future.result()
            np.savez_compressed(a.output / (name + '.npz'), kc_a=counts[0], kc_b=counts[1],
                                pn_a=counts[2], pn_b=counts[3])
            (a.output / (name + '.json')).write_text(json.dumps(result, allow_nan=False))
            print(f'{completed}/{len(tasks)} {name} '
                  f'KC late {result["A"]["late"]["kc"]["active_fraction"]:.3f}/'
                  f'{result["B"]["late"]["kc"]["active_fraction"]:.3f}, '
                  f'overlap {result["overlap"]["late"]["jaccard"]}, '
                  f'{result["wall_seconds"]:.1f}s', flush=True)
    summarize(a.output)


if __name__ == '__main__':
    main()
