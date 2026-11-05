"""D2: profile a frozen physical prefix without changing brain/body parameters."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import pstats
import subprocess
import sys
import time
import uuid

import numpy as np
import psutil
from compare_exploration import digest, tree_digest, write
from runtime_diagnostic import CORE, PARENT, inventory
from training_power import keep_system_awake


def validate_profile(report):
    expected = {'status': 'completed_transport_pilot', 'learning': False,
                'motor_exploration': 0, 'weights_unchanged': True,
                'completed_control_chunks': 100, 'simulated_seconds_per_world': .5,
                'seeds': [10006], 'checkpoint_sha256': PARENT, 'worlds': 1,
                'backend': 'cpu', 'controller': 'brain'}
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(f'Profile integrity mismatch: {key}')


def compare_prefix(reference, candidate):
    with np.load(reference, allow_pickle=False) as a, np.load(candidate, allow_pickle=False) as b:
        if not a.files or set(a.files) != set(b.files):
            raise ValueError('Trace schema mismatch')
        result = {}
        for key in a.files:
            x, y = a[key], b[key]
            if x.shape[0] != 1000 or y.shape[0] != 100 or x.shape[1:] != y.shape[1:]:
                raise ValueError(f'Trace duration/shape mismatch: {key}')
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError(f'Nonfinite trace: {key}')
            result[key] = bool(np.array_equal(x[:100], y))
        return result


def profile_rows(path):
    rows = []
    for (filename, line, function), (primitive, total, own, cumulative, _) in pstats.Stats(str(path)).stats.items():
        # Compact labels avoid publishing user-specific absolute library paths.
        label = Path(filename).name if not filename.startswith('<') else filename
        rows.append(dict(file=label, line=line, function=function, primitive_calls=primitive,
                         total_calls=total, own_seconds=own, cumulative_seconds=cumulative))
    return {'top_own': sorted(rows, key=lambda x:x['own_seconds'], reverse=True)[:25],
            'top_cumulative': sorted(rows, key=lambda x:x['cumulative_seconds'], reverse=True)[:25]}


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
    reference = root/'gpu-pilot/outputs/runtime-diagnostic-d1-20261030/repeat-1'
    if digest(parent) != PARENT:
        raise ValueError('Parent mismatch')
    ref_report = json.loads((reference/'report.json').read_text(encoding='utf-8'))
    from runtime_diagnostic import validate_report
    validate_report(ref_report)
    if subprocess.check_output(['git','-C',str(core.parent),'rev-parse','HEAD'],text=True).strip() != CORE:
        raise ValueError('Core pin mismatch')
    hashes = {'core': tree_digest(core), 'pilot': tree_digest(pilot)}
    token, started = uuid.uuid4().hex, time.monotonic()
    protocol = dict(owner='codex',pid=os.getpid(),token=token,host=os.environ.get('COMPUTERNAME'),
        commit=subprocess.check_output(['git','-C',str(checkout),'rev-parse','HEAD'],text=True).strip(),
        core_commit=CORE,source_hashes=hashes,parent_sha256=PARENT,
        reference_report_sha256=digest(reference/'report.json'),
        reference_trace_sha256=digest(reference/'observations.npz'),
        output=str(output),budget_seconds=180,job_wall_seconds=120,process_wall_seconds=150,
        simulated_seconds=.5,seed=10006,learning=False,minimum_available_gib=2,
        start_unix=time.time(),deadline_unix=time.time()+180)
    lock = root/'gpu-pilot/outputs/codex-training-host.lock'
    with os.fdopen(os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL),'w') as stream:
        json.dump(protocol,stream)
    created, samples = False, []
    result = dict(status='running',model_promoted=False,walking_proven=False)
    try:
        protocol['excluded_dashboard_pids'] = inventory()
        available = psutil.virtual_memory().available/2**30
        protocol['initial_available_gib'] = available
        if available < 2:
            raise MemoryError('D2 requires at least2GiB available; no run launched')
        output.mkdir(parents=True,exist_ok=False)
        created = True
        write(output/'protocol.json',protocol)
        command = [sys.executable,'-m','cProfile','-o',str(output/'profile.pstats'),'-m',
            'noesis_gpu.run','--backend','cpu','--worlds','1','--seconds','.5',
            '--wall-seconds','120','--seed','10006','--checkpoint',str(parent),
            '--noesis-source',str(core),'--data',str(root/'data'),'--output',str(output/'frozen')]
        print('Starting D2 frozen profile:0.5simulated seconds,150s process cap',flush=True)
        with (output/'profile.log').open('w',encoding='utf-8') as log:
            child = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            process = psutil.Process(child.pid)
            job_start = time.monotonic()
            try:
                while child.poll() is None:
                    elapsed = time.monotonic()-job_start
                    if elapsed > 150 or time.monotonic()-started > 180:
                        raise TimeoutError('D2 process/aggregate deadline')
                    try:
                        cpu=process.cpu_times()
                        samples.append(dict(elapsed=elapsed,cpu_seconds=cpu.user+cpu.system,
                            rss_bytes=process.memory_info().rss,
                            host_available_bytes=psutil.virtual_memory().available,
                            host_cpu_percent=psutil.cpu_percent(interval=None)))
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
        if child.returncode or time.monotonic()-job_start > 155:
            raise RuntimeError('Profile process failed/incomplete/late')
        report = json.loads((output/'frozen/report.json').read_text(encoding='utf-8'))
        validate_profile(report)
        if digest(parent)!=PARENT or tree_digest(core)!=hashes['core'] or tree_digest(pilot)!=hashes['pilot']:
            raise RuntimeError('Source/parent changed')
        comparison = compare_prefix(reference/'observations.npz',output/'frozen/observations.npz')
        if not all(comparison.values()):
            raise ValueError('Profiler changed physical trace; do not interpret as matched run')
        result.update(status='completed_profile_exact_prefix',report=report,
                      exact_prefix_fields=comparison,profile=profile_rows(output/'profile.pstats'))
    except BaseException as error:
        result.update(status='stopped',error_type=type(error).__name__,error=str(error))
        raise
    finally:
        if created:
            write(output/'telemetry.json',samples)
            result['wall_seconds']=time.monotonic()-started
            write(output/'result.json',result)
        if json.loads(lock.read_text()).get('token')==token:
            lock.unlink()


if __name__=='__main__':
    with keep_system_awake():
        main()
