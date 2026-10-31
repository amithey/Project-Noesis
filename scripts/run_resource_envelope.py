"""R4 frozen feasibility diagnostic; does not change T1 or scientific gates."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

import psutil
from compare_exploration import digest, tree_digest, write
from runtime_diagnostic import CORE, PARENT, inventory, validate_report, compare_traces
from training_power import keep_system_awake
from training_resources import assess, sample_quiescence

SOURCES = dict(core='f52c8d90f98ebbacfd5b4e1066aee1f938435455cf0ef9b3cd55778dbcd52888',
    pilot='80556cd09af112567f6963441beb3cd995842a68d9f37143bd99a24e7d5c87b0')


def preflight(samples):
    return assess(samples, minimum_gib=2., maximum_cpu_percent=60.)


def stop_reason(elapsed, available_gib, *, process_seconds=330):
    if isinstance(process_seconds, bool) or process_seconds not in (330, 930):
        raise ValueError('Only registered330s/930s process watchdogs are supported')
    if not math.isfinite(elapsed) or elapsed < 0 or not math.isfinite(available_gib) or available_gib < 0:
        raise ValueError('Unreadable elapsed/memory telemetry')
    if available_gib < 1.: return 'host_memory_below_1GiB'
    if elapsed >= process_seconds: return 'process_deadline'
    return None


def snapshot(child):
    row = dict(host_available_gib=psutil.virtual_memory().available / 2**30,
               host_cpu_percent=psutil.cpu_percent(interval=None))
    try:
        proc = psutil.Process(child.pid); cpu = proc.cpu_times()
        row.update(child_cpu_seconds=cpu.user + cpu.system, child_rss_bytes=proc.memory_info().rss)
    except psutil.NoSuchProcess:
        pass  # A just-exited child may disappear between poll and telemetry.
    return row


def monitored_run(command, log_path, *, spawn=None, clock=None, sample=None, process_seconds=330):
    """Own one child and always reap it, including telemetry errors and interrupts."""
    spawn, clock, sample = spawn or subprocess.Popen, clock or time.monotonic, sample or snapshot
    stop_reason(0., 1., process_seconds=process_seconds)  # Validate before spawning.
    result = dict(status='running', samples=[])
    started = clock()
    with log_path.open('w', encoding='utf-8') as log:
        child = spawn(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            while child.poll() is None:
                row = dict(elapsed_seconds=clock() - started, **sample(child))
                result['samples'].append(row)
                reason = stop_reason(row['elapsed_seconds'], row['host_available_gib'], process_seconds=process_seconds)
                if reason: raise RuntimeError(reason)
                try: child.wait(timeout=2)
                except subprocess.TimeoutExpired: pass
            result.update(status='process_terminal', exit_code=child.returncode)
        except BaseException as error:
            result.update(status='stopped', error_type=type(error).__name__, error=str(error))
            raise
        finally:
            if child.poll() is None: child.kill()
            child.wait()
            result.update(process_wall_seconds=clock() - started, final_exit_code=child.returncode)
            write(log_path.with_suffix('.telemetry.json'), result)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--shared-root', type=Path, required=True)
    ap.add_argument('--core-source', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args(); root, core, out = [p.resolve() for p in (args.shared_root, args.core_source, args.output)]
    own = Path(__file__).resolve().parents[1]; pilot = own / 'gpu-pilot/src'
    parent = root / 'gpu-pilot/outputs/e4-exploration-20261029/train-control-1000/weights-last.npz'
    reference = root / 'gpu-pilot/outputs/runtime-diagnostic-d1-20261030/repeat-1'
    pin = subprocess.check_output(['git', '-C', str(core.parent), 'rev-parse', 'HEAD'], text=True).strip()
    if pin != CORE or digest(parent) != PARENT or dict(core=tree_digest(core), pilot=tree_digest(pilot)) != SOURCES:
        raise ValueError('Pinned parent/core/pilot required')
    validate_report(json.loads((reference / 'report.json').read_text(encoding='utf-8')))
    reference_sha = digest(reference / 'observations.npz')
    token = uuid.uuid4().hex
    protocol = dict(experiment='R4 frozen operating-envelope feasibility', owner='codex', pid=os.getpid(), token=token,
        commit=subprocess.check_output(['git', '-C', str(own), 'rev-parse', 'HEAD'], text=True).strip(),
        core_commit=CORE, source_hashes=SOURCES, parent_sha256=PARENT, reference_sha256=reference_sha,
        seed=10006, seconds=5, repeats=2, learning=False, worlds=1, backend='cpu',
        aggregate_seconds=720, inner_seconds=300, process_seconds=330, next_job_margin_seconds=5,
        preflight_minimum_gib=2, preflight_maximum_cpu_percent=60, preflight_samples=10,
        in_run_memory_stop_gib=1, telemetry_interval_seconds=2, output=str(out), start_unix=time.time(),
        T1_gate_changed=False, scientific_gates_changed=False, training_throughput_verified=False)
    lock = root / 'gpu-pilot/outputs/codex-training-host.lock'
    with os.fdopen(os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL), 'w') as stream:
        json.dump(protocol, stream)
    started, created = time.monotonic(), False
    result = dict(status='initializing', jobs=[], model_promoted=False, walking_proven=False)
    try:
        # Run on the host-visible surface; fail closed on unidentified Python.
        protocol['excluded_dashboard_pids'] = inventory()
        out.mkdir(parents=True, exist_ok=False); created = True; write(out / 'protocol.json', protocol)
        def verify():
            if (digest(parent) != PARENT or dict(core=tree_digest(core), pilot=tree_digest(pilot)) != SOURCES
                    or digest(reference / 'observations.npz') != reference_sha):
                raise ValueError('Parent/source/reference mutation')
        for index in (1, 2):
            verify(); inventory()
            samples = sample_quiescence(); decision = preflight(samples)
            write(out / f'preflight-{index}.json', dict(samples=samples, decision=decision))
            if not decision['passed']: raise RuntimeError('R4 preflight refused; no next simulation')
            if 720 - (time.monotonic() - started) < 335: raise TimeoutError('Insufficient full next-job budget')
            target = out / f'repeat-{index}'
            command = [sys.executable, '-m', 'noesis_gpu.run', '--backend', 'cpu', '--worlds', '1',
                '--seconds', '5', '--wall-seconds', '300', '--seed', '10006', '--checkpoint', str(parent),
                '--noesis-source', str(core), '--data', str(root / 'data'), '--output', str(target)]
            print(f'Starting R4 frozen repeat{index}', flush=True)
            job = monitored_run(command, out / f'repeat-{index}.log')
            result['jobs'].append(dict(repeat=index, **{k: v for k, v in job.items() if k != 'samples'}))
            write(out / 'result.json', result)
            if job['exit_code'] or job['process_wall_seconds'] > 335: raise RuntimeError('Incomplete or late frozen repeat')
            report = json.loads((target / 'report.json').read_text(encoding='utf-8')); validate_report(report)
            from noesis_gpu.evaluate import MATCHED_SETTINGS
            old = json.loads((reference / 'report.json').read_text(encoding='utf-8'))
            if any(k not in report or k not in old or report[k] != old[k] for k in MATCHED_SETTINGS):
                raise ValueError('Reference context mismatch')
            verify()
            comparison = compare_traces(target / 'observations.npz', reference / 'observations.npz')
            write(out / f'reference-comparison-{index}.json', comparison)
            if not all(row['exact_equal'] for row in comparison.values()): raise ValueError('Frozen reference mismatch')
        comparison = compare_traces(out / 'repeat-1/observations.npz', out / 'repeat-2/observations.npz')
        write(out / 'repeat-comparison.json', comparison)
        if not all(row['exact_equal'] for row in comparison.values()): raise ValueError('Fresh repeat mismatch')
        result.update(status='completed_two_frozen_feasibility_repeats',
            scope='N2 same-seed frozen feasibility only; not causal load attribution, training throughput, a new default resource policy or walking proof')
    except BaseException as error:
        result.update(status='stopped', error_type=type(error).__name__, error=str(error)); raise
    finally:
        if created: result['wall_seconds'] = time.monotonic() - started; write(out / 'result.json', result)
        if json.loads(lock.read_text(encoding='utf-8')).get('token') == token: lock.unlink()


if __name__ == '__main__':
    with keep_system_awake(): main()
