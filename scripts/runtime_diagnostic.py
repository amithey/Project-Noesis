"""Two frozen physical repeats, with telemetry and a fail-closed shared lease."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

import numpy as np
import psutil

from compare_exploration import digest, tree_digest, write
from training_power import keep_system_awake

PARENT = '9fd7d41177bf2c745809401c3b142a832661d1a18d0b117b30cb02aea63fea3c'
CORE = '42c20769376bda5694b4c1c66ca75e52cbd9f1f4'


def validate_report(report):
    expected = {'status': 'completed_transport_pilot', 'learning': False,
                'motor_exploration': 0, 'weights_unchanged': True,
                'completed_control_chunks': 1000, 'simulated_seconds_per_world': 5,
                'checkpoint_sha256': PARENT, 'seeds': [10006], 'worlds': 1,
                'backend': 'cpu', 'controller': 'brain'}
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(f'Frozen repeat integrity mismatch: {key}')
    for key in ('setup_wall_seconds', 'loop_wall_seconds', 'total_wall_seconds'):
        value = report[key]
        if not np.isfinite(value) or value < 0:
            raise ValueError(f'Invalid timing: {key}')


def compare_traces(first, second):
    with np.load(first, allow_pickle=False) as a, np.load(second, allow_pickle=False) as b:
        if set(a.files) != set(b.files) or not a.files:
            raise ValueError('Observation schema mismatch')
        results = {}
        for key in a.files:
            x, y = a[key], b[key]
            if x.shape != y.shape or x.shape[0] != 1000:
                raise ValueError(f'Incomplete or mismatched trace: {key}')
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError(f'Nonfinite trace: {key}')
            results[key] = {'exact_equal': bool(np.array_equal(x, y)),
                            'max_absolute_difference': float(np.max(np.abs(x.astype(float)-y.astype(float))))}
        return results


def inventory():
    excluded = []
    for process in psutil.process_iter(['pid', 'name', 'cmdline']):
        if not (process.info['name'] or '').lower().startswith(('python.exe', 'pythonw.exe')):
            continue
        if process.pid == os.getpid():
            continue
        command = process.info['cmdline']
        if not command:
            raise RuntimeError('Cannot identify Python process')
        if any(command[i:i+3] == ['-m', 'streamlit', 'run'] for i in range(len(command)-2)):
            excluded.append(process.pid)
        else:
            raise RuntimeError(f'Other Python worker active: {process.pid}')
    return excluded


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--shared-root', type=Path, required=True)
    ap.add_argument('--core-source', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    root, core, output = args.shared_root.resolve(), args.core_source.resolve(), args.output.resolve()
    checkout = Path(__file__).resolve().parents[1]
    pilot = checkout/'gpu-pilot/src'
    parent = root/'gpu-pilot/outputs/e4-exploration-20261029/train-control-1000/weights-last.npz'
    if digest(parent) != PARENT:
        raise ValueError('Wrong parent')
    if subprocess.check_output(['git', '-C', str(core.parent), 'rev-parse', 'HEAD'], text=True).strip() != CORE:
        raise ValueError('Wrong core')
    sources = {'core': tree_digest(core), 'pilot': tree_digest(pilot)}
    token = uuid.uuid4().hex
    protocol = {'owner': 'codex', 'pid': os.getpid(), 'token': token,
                'host': os.environ.get('COMPUTERNAME'), 'parent_sha256': PARENT,
                'pilot_commit': subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip(),
                'core_commit': CORE, 'source_hashes': sources, 'output': str(output),
                'budget_seconds': 720, 'job_wall_seconds': 300, 'process_wall_seconds': 330,
                'seconds': 5, 'seed': 10006, 'learning': False, 'repeats': 2,
                'start_unix': time.time(), 'deadline_unix': time.time()+720}
    lock = root/'gpu-pilot/outputs/codex-training-host.lock'
    with os.fdopen(os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL), 'w') as stream:
        json.dump(protocol, stream)
    started, created = time.monotonic(), False
    outcome = {'status': 'running', 'model_promoted': False, 'walking_proven': False, 'jobs': []}
    try:
        protocol['excluded_dashboard_pids'] = inventory()
        output.mkdir(parents=True, exist_ok=False)
        created = True
        write(output/'protocol.json', protocol)
        def verify():
            if digest(parent) != PARENT or tree_digest(core) != sources['core'] or tree_digest(pilot) != sources['pilot']:
                raise RuntimeError('Parent/source changed')
        for index in (1, 2):
            verify()
            if 720-(time.monotonic()-started) < 330:
                raise TimeoutError('Insufficient budget for full repeat')
            inventory()
            target = output/f'repeat-{index}'
            command = [sys.executable, '-m', 'noesis_gpu.run', '--backend', 'cpu', '--worlds', '1',
                       '--seconds', '5', '--wall-seconds', '300', '--seed', '10006',
                       '--checkpoint', str(parent), '--noesis-source', str(core),
                       '--data', str(root/'data'), '--output', str(target)]
            print(f'Starting frozen repeat {index}', flush=True)
            samples, job_start = [], time.monotonic()
            with (output/f'repeat-{index}.log').open('w', encoding='utf-8') as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                process = psutil.Process(child.pid)
                try:
                    while child.poll() is None:
                        elapsed = time.monotonic()-job_start
                        if elapsed > 330:
                            raise TimeoutError('Process deadline exceeded')
                        try:
                            cpu = process.cpu_times()
                            mem = psutil.virtual_memory()
                            samples.append({'elapsed': elapsed, 'cpu_seconds': cpu.user+cpu.system,
                                            'rss_bytes': process.memory_info().rss,
                                            'host_available_bytes': mem.available,
                                            'host_cpu_percent': psutil.cpu_percent(interval=None)})
                        except psutil.NoSuchProcess:
                            break
                        try:
                            child.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            pass
                finally:
                    if child.poll() is None:
                        child.kill()
                    child.wait()
                    write(output/f'repeat-{index}-telemetry.json', samples)
            elapsed = time.monotonic()-job_start
            report = json.loads((target/'report.json').read_text(encoding='utf-8'))
            outcome['jobs'].append({'repeat': index, 'exit_code': child.returncode,
                                    'process_wall_seconds': elapsed, 'report': report})
            write(output/'result.json', outcome)
            if elapsed > 335 or child.returncode:
                raise RuntimeError('Repeat incomplete or late')
            validate_report(report)
            verify()
        comparison = compare_traces(output/'repeat-1/observations.npz', output/'repeat-2/observations.npz')
        write(output/'trace-comparison.json', comparison)
        outcome['status'] = 'completed_repeats' if all(x['exact_equal'] for x in comparison.values()) else 'repeat_mismatch'
        outcome['timing_scope'] = 'frozen runtime only; does not establish training throughput or causality of historical stalls'
    except BaseException as error:
        outcome.update(status='stopped', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        if created:
            outcome['wall_seconds'] = time.monotonic()-started
            write(output/'result.json', outcome)
        if json.loads(lock.read_text()).get('token') == token:
            lock.unlink()


if __name__ == '__main__':
    with keep_system_awake():
        main()
