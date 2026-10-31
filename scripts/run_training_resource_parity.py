"""D4: one bounded physical training replay, never a model promotion."""
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
import numpy as np

from check_logging_parity import EXPECTED, check_alignment
from compare_exploration import digest, tree_digest, write
from run_e6 import audit_bank
from run_resource_envelope import SOURCES, monitored_run, preflight
from runtime_diagnostic import CORE, PARENT, inventory
from training_power import keep_system_awake
from training_resources import sample_quiescence

TRACE_FIELDS = set('angles velocity position feet_position heading up_body omega ground_contact load foot_load command adhesion reward standing stance_slip_mm_s learning_active cycle_credit reward_unclipped actual_bonus_after_clipping'.split())


def admission_window(samples, *, mean_cpu=False):
    decision = preflight(samples)  # Validates full count, ranges and finite telemetry.
    decision['original_max_rule_passed'] = decision['passed']
    decision['cpu_decision_statistic'] = 'mean' if mean_cpu else 'maximum'
    decision['mean_host_cpu_percent'] = sum(s['host_cpu_percent'] for s in samples) / len(samples)
    if mean_cpu:
        decision['allowed_host_cpu_mean_percent'] = decision.pop('allowed_host_cpu_percent')
        decision['passed'] = decision['minimum_available_gib'] >= 2 and decision['mean_host_cpu_percent'] <= 60
        decision['note'] = 'D4M prospective mean-CPU contrast only; maximum is retained, no T1/default/scientific-policy adoption'
    return decision


def prior_refusal(directory, reference_files):
    protocol = json.loads((directory / 'protocol.json').read_text(encoding='utf-8'))
    result = json.loads((directory / 'result.json').read_text(encoding='utf-8'))
    resource = json.loads((directory / 'resource-preflight.json').read_text(encoding='utf-8'))
    required = dict(experiment='D4 guarded training replay', core_commit=CORE, source_hashes=SOURCES,
        parent_sha256=PARENT, expected_checkpoint_sha256=EXPECTED, reference_files=reference_files,
        seed=1000, seconds=5, chunks=1000, learning_updates=990, learning_rate=.2, cycle_weight=.1,
        exploration_mv_ms=1.5, aggregate_seconds=420, inner_seconds=300, process_seconds=330)
    if any(protocol.get(k) != v for k, v in required.items()):
        raise ValueError('Prior refusal provenance mismatch')
    if (result.get('status') != 'stopped' or result.get('jobs') != [] or (directory / 'train').exists()
            or result.get('error') != 'D4 resource preflight refused; no training launched'
            or preflight(resource['samples'])['passed'] or resource['decision'].get('passed') is not False):
        raise ValueError('Mean contrast requires an original max-guard refusal with ZERO training jobs')
    wall = result.get('wall_seconds')
    if isinstance(wall, bool) or not isinstance(wall, (int, float)) or not math.isfinite(wall) or not 0 <= wall <= 420:
        raise ValueError('Prior refusal wall accounting invalid')
    if (directory / 'mean-policy-allocation.json').exists():
        raise ValueError('Prior refusal already allocated a mean-policy contrast')
    return dict(wall_seconds=wall, files={name: digest(directory / name) for name in
        ['protocol.json', 'result.json', 'resource-preflight.json']})


def validate_training(report):
    expected = dict(status='completed_training_birth', seed=1000, worlds=1, backend='cpu',
        learning=True, requested_seconds=5, simulated_seconds=5, completed_chunks=1000,
        learning_updates=990, learning_rate=.2, motor_exploration_mv_ms=1.5, cycle_weight=.1,
        eligibility_ms=100, original_weight_bound=.5, control_ms=5., physics_dt=.0001,
        birth_ms=50, reward_objective='curriculum-gripped-cycle-v9',
        import_checkpoint_sha256=PARENT, noesis_source_sha256=SOURCES['core'])
    for key, value in expected.items():
        if report.get(key) != value: raise ValueError(f'Incomplete or mismatched training:{key}')


def exact_archive(candidate, reference, *, trace=False):
    with np.load(candidate, allow_pickle=False) as a, np.load(reference, allow_pickle=False) as b:
        if not a.files or set(a.files) != set(b.files) or (trace and set(a.files) != TRACE_FIELDS):
            raise ValueError('Archive schema mismatch')
        fields = {}
        for key in b.files:
            x, y = a[key], b[key]
            if (x.shape != y.shape or x.dtype != y.dtype
                    or (trace and (x.ndim < 1 or x.shape[0] != 1000))
                    or not np.isfinite(x).all() or not np.isfinite(y).all()
                    or not np.array_equal(x, y)):
                raise ValueError(f'Archive shape/dtype/value mismatch:{key}')
            fields[key] = True
    return fields


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--shared-root', type=Path, required=True)
    ap.add_argument('--core-source', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--mean-cpu-after-refusal', type=Path)
    args = ap.parse_args(); root, core, out = [p.resolve() for p in (args.shared_root, args.core_source, args.output)]
    own = Path(__file__).resolve().parents[1]; pilot = own / 'gpu-pilot/src'
    parent = root / 'gpu-pilot/outputs/e4-exploration-20261029/train-control-1000/weights-last.npz'
    reference = root / 'gpu-pilot/outputs/logging-parity-d3-20261030/logged'
    pin = subprocess.check_output(['git', '-C', str(core.parent), 'rev-parse', 'HEAD'], text=True).strip()
    if pin != CORE or digest(parent) != PARENT or digest(reference / 'weights-last.npz') != EXPECTED:
        raise ValueError('Pinned parent/core/reference bank required')
    validate_training(json.loads((reference / 'report.json').read_text(encoding='utf-8')))
    check_alignment(reference)
    reference_files = {name: digest(reference / name) for name in
        ['report.json', 'weights-last.npz', 'training-observations.npz', 'exploration-log.npz']}
    prior = args.mean_cpu_after_refusal.resolve() if args.mean_cpu_after_refusal else None
    inherited = prior_refusal(prior, reference_files) if prior else dict(wall_seconds=0., files={})
    token = uuid.uuid4().hex
    protocol = dict(experiment='D4 guarded training replay', owner='codex', pid=os.getpid(), token=token,
        commit=subprocess.check_output(['git', '-C', str(own), 'rev-parse', 'HEAD'], text=True).strip(),
        core_commit=CORE, source_hashes=SOURCES, parent_sha256=PARENT, reference_files=reference_files,
        expected_checkpoint_sha256=EXPECTED, seed=1000, seconds=5, chunks=1000, learning_updates=990,
        learning_rate=.2, cycle_weight=.1, exploration_mv_ms=1.5, aggregate_seconds=420,
        inner_seconds=300, process_seconds=330, launch_minimum_gib=2, launch_maximum_cpu_percent=60,
        launch_samples=10, in_run_stop_gib=1, telemetry_interval_seconds=2, output=str(out),
        T1_gate_changed=False, scientific_gates_changed=False, start_unix=time.time())
    protocol.update(cpu_admission_statistic='mean' if prior else 'maximum', prior_refusal=str(prior) if prior else None,
        prior_wall_seconds=inherited['wall_seconds'], prior_files=inherited['files'])
    if prior:
        protocol['experiment'] = 'D4M guarded training replay'
        protocol['launch_mean_cpu_percent'] = protocol.pop('launch_maximum_cpu_percent')
    def verify():
        if (digest(parent) != PARENT or dict(core=tree_digest(core), pilot=tree_digest(pilot)) != SOURCES
                or any(digest(reference / k) != v for k, v in reference_files.items())):
            raise ValueError('Parent/source/reference mutation')
        if prior and any(digest(prior / k) != v for k, v in inherited['files'].items()):
            raise ValueError('Prior refusal evidence changed')
    verify()
    lock = root / 'gpu-pilot/outputs/codex-training-host.lock'
    with os.fdopen(os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL), 'w') as stream:
        json.dump(protocol, stream)
    started, created = time.monotonic(), False
    result = dict(status='initializing', jobs=[], model_promoted=False, walking_proven=False)
    try:
        protocol['excluded_dashboard_pids'] = inventory()
        out.mkdir(parents=True, exist_ok=False); created = True; write(out / 'protocol.json', protocol)
        if prior:
            # One prospective followup only; no budget resets or fallbacks after a birth.
            with (prior / 'mean-policy-allocation.json').open('x', encoding='utf-8') as stream:
                json.dump(dict(output=str(out), token=token, prior_wall_seconds=inherited['wall_seconds']), stream)
        samples = sample_quiescence(); decision = admission_window(samples, mean_cpu=bool(prior))
        write(out / 'resource-preflight.json', dict(samples=samples, decision=decision))
        if not decision['passed']: raise RuntimeError('D4 resource preflight refused; no training launched')
        verify(); inventory()
        if 420 - inherited['wall_seconds'] - (time.monotonic() - started) < 335: raise TimeoutError('Insufficient full-job budget')
        birth = out / 'train'
        command = [sys.executable, '-m', 'noesis_gpu.train', '--backend', 'cpu', '--seconds', '5',
            '--wall-seconds', '300', '--seed', '1000', '--profile', 'gripped-cycle', '--cycle-weight', '.1',
            '--learning-rate', '.2', '--exploration-mv-ms', '1.5', '--log-exploration',
            '--checkpoint', str(parent), '--noesis-source', str(core), '--data', str(root / 'data'), '--output', str(birth)]
        print(f"Starting {'D4M' if prior else 'D4'} single5s training replay", flush=True)
        job = monitored_run(command, out / 'train.log')
        result['jobs'].append({k: v for k, v in job.items() if k != 'samples'}); write(out / 'result.json', result)
        if job['exit_code'] or job['process_wall_seconds'] > 335: raise RuntimeError('Incomplete/late training process')
        report = json.loads((birth / 'report.json').read_text(encoding='utf-8')); validate_training(report)
        bank = birth / 'weights-last.npz'; sha = audit_bank(bank, parent)
        if sha != EXPECTED or report.get('saved_checkpoint_sha256') != sha: raise ValueError('Training bank parity failed')
        if Path(report.get('latest_training_checkpoint', '')).resolve() != bank: raise ValueError('Local checkpoint path mismatch')
        result['trace_exact_fields'] = exact_archive(birth / 'training-observations.npz', reference / 'training-observations.npz', trace=True)
        result['exploration_exact_fields'] = exact_archive(birth / 'exploration-log.npz', reference / 'exploration-log.npz')
        result['log_alignment'] = check_alignment(birth); verify()
        if inherited['wall_seconds'] + time.monotonic() - started > 420: raise TimeoutError('Aggregate diagnostic budget exceeded')
        result.update(status='completed_guarded_training_replay', checkpoint_sha256=sha,
            scope='N1known-child5s replay; not improved learning, retention,25s throughput or adoption of a new default policy',
            shared_training_last_updated=False, T1_gate_changed=False, scientific_gates_changed=False)
    except BaseException as error:
        result.update(status='stopped', error_type=type(error).__name__, error=str(error)); raise
    finally:
        if created:
            result['wall_seconds'] = time.monotonic() - started
            result['aggregate_active_seconds'] = inherited['wall_seconds'] + result['wall_seconds']
            write(out / 'result.json', result)
        if json.loads(lock.read_text(encoding='utf-8')).get('token') == token: lock.unlink()


if __name__ == '__main__':
    with keep_system_awake(): main()
