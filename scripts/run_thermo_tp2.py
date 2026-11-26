"""TP2 fixed25Hz thermo amplitude contrast with matched RNG and descendant watchdog."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import uuid

import numpy as np
import psutil

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1], help='Prepared second-host layout root')
root = ap.parse_args().root.resolve()
pilot = root / 'work/Noesis-Sensory'
core = root / 'work/Noesis-Core/src'
data = root / 'work/NoesisData'
parent = root / 'work/evidence/s1-restored/gpu-pilot/outputs/e4-exploration-20261029/train-control-1000/weights-last.npz'
baseline = root / 'outputs/frozen-baseline-b2-20261104'
out = root / 'outputs/thermo-pilot-tp2-20261106'
expected = '9fd7d41177bf2c745809401c3b142a832661d1a18d0b117b30cb02aea63fea3c'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree(path):
    digest = hashlib.sha256()
    for p in sorted(path.rglob('*.py')):
        digest.update(p.relative_to(path).as_posix().encode())
        digest.update(p.read_bytes())
    return digest.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def exact(a, b):
    with np.load(a, allow_pickle=False) as x, np.load(b, allow_pickle=False) as y:
        return set(x.files) == set(y.files) and all(np.array_equal(x[key], y[key]) for key in x.files)


ancestors = {p.pid for p in psutil.Process().parents()}


def resources():
    for p in psutil.process_iter(['pid', 'name']):
        if p.info['pid'] not in ancestors | {os.getpid()} and (p.info['name'] or '').lower() in ('python.exe', 'pythonw.exe'):
            raise RuntimeError('Competing Python process; no job launched')
    ram = psutil.virtual_memory().available / 2**30
    if ram < 2:
        raise RuntimeError('Available RAM below2GiB')
    return ram


sys.path[:0] = [str(pilot / 'gpu-pilot/src'), str(core)]
from noesis_gpu.evaluate import summarize, MATCHED_SETTINGS
from noesis_gpu.process_watchdog import run_windows_job, WatchdogError

assert sha(parent) == expected
for entry in json.loads((root / 'outputs/noesis-required-data-sha256.json').read_text())['files']:
    assert sha(data / entry['path']) == entry['sha256']
hashes = dict(pilot=tree(pilot / 'gpu-pilot/src'), core=tree(core))
protocol = dict(experiment='TP2 constant thermo amplitude, RNG-matched sham', date='2026-11-06',
                source_hashes=hashes, core_commit='42c20769376bda5694b4c1c66ca75e52cbd9f1f4',
                parent_sha256=expected, seeds=[10006, 10007], modes=['plain', 'sham', 'stimulus'],
                seconds_per_job=5, chunks_per_job=1000, warmup_ms=50,
                thermo_gain=50., thermo_signal=.5, thermo_rate_hz=25., event_weights=dict(sham=0., stimulus=400.),
                process_budget_seconds=300, aggregate_budget_seconds=1800,
                invariance_gate='plain exactly replays B2; paired sham/stimulus drive indices/order/rates and full625word MT RNG states identical for all1000chunks',
                neural_criterion='stimulus mean warm-afferent rate exceeds sham by>=5Hz/cell on BOTH reused development seeds',
                retention_criterion='stimulus standing>=.90 AND standing>=sham AND slip<=sham on BOTH seeds; descriptive frozen screen',
                stop='technical/invariance/context/hash/resource failure, timeout or remaining budget<300s; no retries, gain scan or extra seeds',
                watchdog='Windows Job Object assigned to suspended worker before resume; descendants contained, job emptiness verified before lease release',
                learning=False, model_promoted=False, walking_proven=False)
resources()
out.mkdir(exist_ok=False)
lock = root / 'work/Project-Noesis/gpu-pilot/outputs/codex-training-host.lock'
token = uuid.uuid4().hex
with os.fdopen(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY), 'w') as handle:
    json.dump(dict(owner='codex', pid=os.getpid(), token=token, start_unix=time.time(), deadline_unix=time.time()+1800), handle)
env = dict(os.environ, PYTHONPATH=str(pilot / 'gpu-pilot/src'), NOESIS_SOURCE=str(core), PYTHONUTF8='1',
           MPLCONFIGDIR=str(root / 'work/matplotlib-cache'))
result = dict(status='running', cases=[], rng_gate_by_seed={}, plain_gate_by_seed={}, model_promoted=False, walking_proven=False)
started = time.monotonic()
tree_empty = True
try:
    write(out / 'protocol.json', protocol)
    for seed in protocol['seeds']:
        rows, arrays, reports = {}, {}, {}
        for mode in protocol['modes']:
            if sha(parent) != expected or hashes != dict(pilot=tree(pilot / 'gpu-pilot/src'), core=tree(core)):
                raise RuntimeError('Source or parent changed')
            if time.monotonic() - started > 1500:
                raise RuntimeError('Remaining budget below300s')
            ram = resources()
            target = out / f'{mode}-{seed}'
            command = [sys.executable, '-m', 'noesis_gpu.tp2_probe', '--mode', mode, '--seed', str(seed),
                       '--core', str(core), '--data', str(data), '--checkpoint', str(parent), '--output', str(target)]
            print('START', seed, mode, flush=True)
            tree_empty = False
            try:
                with (out / f'{mode}-{seed}.log').open('w', encoding='utf-8') as log:
                    receipt = run_windows_job(command, timeout=300, stdout=log, cwd=pilot, env=env)
                tree_empty = receipt['tree_empty']
            except WatchdogError as error:
                tree_empty = error.cleanup_verified
                raise
            write(out / f'{mode}-{seed}-receipt.json', receipt)
            assert tree_empty and not receipt['timed_out'] and receipt['exit_code'] == 0
            report, summary = summarize(target)
            assert report['checkpoint_sha256'] == expected and report['tp2_mode'] == mode
            assert report['completed_control_chunks'] == 1000 and report['weights_unchanged']
            with np.load(target / 'tp2-observations.npz', allow_pickle=False) as archive:
                a = {key: archive[key].copy() for key in archive.files}
            assert all(len(v) == 1000 and np.isfinite(v).all() for v in a.values())
            assert a['warm_spikes'].shape == a['cool_spikes'].shape == (1000, 6)
            assert a['rng_state'].shape == (1000, 625) and a['rng_state'].dtype == np.uint32
            if mode == 'plain':
                assert exact(target / 'observations.npz', baseline / f'parent-{seed}/observations.npz'), 'B2 replay failed'
                assert not a['extra_rate_hz'].any()
                result['plain_gate_by_seed'][str(seed)] = True
            else:
                np.testing.assert_array_equal(a['extra_rate_hz'], np.full((1000, 6), 25.))
                np.testing.assert_array_equal(a['extra_weight'], np.full((1000, 6), 0. if mode == 'sham' else 400.))
                assert (a['extra_idx'] >= 0).all() and (a['drive_length'] == a['drive_length'][0]).all()
            row = dict(seed=seed, mode=mode, summary=summary, process_receipt=receipt,
                       warm_hz_per_cell=a['warm_spikes'][10:].sum(axis=0).astype(float).tolist(),
                       warm_rate_hz_per_neuron=float(a['warm_spikes'][10:].sum() / (6 * 4.95)),
                       cool_rate_hz_per_neuron=float(a['cool_spikes'][10:].sum() / (6 * 4.95)),
                       available_ram_before_gib=ram,
                       observations_sha256=sha(target / 'observations.npz'), observer_sha256=sha(target / 'tp2-observations.npz'),
                       report_sha256=sha(target / 'report.json'), afferents_sha256=sha(target / 'afferents.json'))
            row['warm_hz_per_cell'] = [x / 4.95 for x in row['warm_hz_per_cell']]
            rows[mode], arrays[mode], reports[mode] = row, a, report
            result['cases'].append(row)
            write(out / 'result.json', result)
            print('COMPLETE', seed, mode, 'standing', summary['standing_fraction'], 'warm_hz', row['warm_rate_hz_per_neuron'], flush=True)
        for key in ('extra_idx', 'extra_rate_hz', 'drive_length', 'drive_indices_sha256', 'rng_state'):
            np.testing.assert_array_equal(arrays['sham'][key], arrays['stimulus'][key], err_msg='Paired RNG/drive gate: '+key)
        assert (out / f'sham-{seed}/afferents.json').read_bytes() == (out / f'stimulus-{seed}/afferents.json').read_bytes()
        for key in MATCHED_SETTINGS:
            if key == 'senses':
                assert reports['sham'][key][:-1] == reports['stimulus'][key][:-1]
            else:
                assert reports['sham'][key] == reports['stimulus'][key], 'Context mismatch: '+key
        result['rng_gate_by_seed'][str(seed)] = True
        r, s = rows['stimulus'], rows['sham']
        r['warm_rate_gain_vs_sham'] = r['warm_rate_hz_per_neuron'] - s['warm_rate_hz_per_neuron']
        r['neural_criterion_pass'] = r['warm_rate_gain_vs_sham'] >= 5.
        r['retention_criterion_pass'] = bool(r['summary']['standing_fraction'] >= .9 and
            r['summary']['standing_fraction'] >= s['summary']['standing_fraction'] and
            r['summary']['mean_stance_slip_mm_s'] <= s['summary']['mean_stance_slip_mm_s'])
    stimulus = [r for r in result['cases'] if r['mode'] == 'stimulus']
    result.update(status='completed_TP2', all_neural_criteria_pass=all(r['neural_criterion_pass'] for r in stimulus),
                  all_retention_criteria_pass=all(r['retention_criterion_pass'] for r in stimulus))
except BaseException as error:
    result.update(status='failed', error_type=type(error).__name__, error=str(error))
    raise
finally:
    result.update(seconds=time.monotonic()-started, parent_unchanged=sha(parent)==expected,
                  source_unchanged=hashes==dict(pilot=tree(pilot / 'gpu-pilot/src'), core=tree(core)),
                  process_tree_empty_verified=tree_empty)
    if tree_empty and lock.exists() and json.loads(lock.read_text())['token'] == token:
        lock.unlink()
    result['lease_released'] = not lock.exists()
    write(out / 'result.json', result)
    print('RESULT', result['status'], 'seconds', result['seconds'], flush=True)
