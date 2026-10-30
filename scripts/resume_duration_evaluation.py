"""Continue frozen T1 controls from a COMPLETE birth; never resume neural/physics state."""
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

from compare_exploration import digest, tree_digest, write
from run_duration_study import CORE, PARENT, CONTROL, PROBES, audit_bank, prefix_check, duration_decision
from runtime_diagnostic import inventory
from training_power import keep_system_awake
from training_resources import assess, sample_quiescence

ARMS = ('parent', 'control5', 'child25')
BUDGET = 3600.


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def full_training(report, protocol, sources):
    expected = dict(status='completed_training_birth', seed=1000, requested_seconds=25,
        simulated_seconds=25, completed_chunks=5000, learning_updates=4990,
        learning_rate=.2, motor_exploration_mv_ms=1.5, cycle_weight=.1,
        import_checkpoint_sha256=PARENT, reward_objective='curriculum-gripped-cycle-v9',
        noesis_source_sha256=sources['core'], control_ms=5., physics_dt=.0001,
        eligibility_ms=100, birth_ms=50, original_weight_bound=.5, worlds=1,
        backend='cpu', learning=True)
    if any(report.get(k) != v for k, v in expected.items()):
        raise ValueError('Require complete original-context25s training; partial banks forbidden')
    required = dict(parent_sha256=PARENT, control5_sha256=CONTROL, core_commit=CORE,
        source_hashes=sources, training_seed=1000, training_seconds=25,
        training_chunks=5000, learning_updates=4990, probe_seeds=list(PROBES),
        budget_seconds=3600, learning_rate=.2, cycle_weight=.1, exploration_mv_ms=1.5,
        training_inner_seconds=900, training_process_seconds=930,
        probe_inner_seconds=300, probe_process_seconds=330, minimum_available_gib=3,
        maximum_host_cpu_percent=30, resource_window_samples=10)
    if any(protocol.get(k) != v for k, v in required.items()):
        raise ValueError('Original protocol/source context mismatch')
    if report.get('model_promoted') is not False or report.get('walking_proven') is not False:
        raise ValueError('Experimental scope mismatch')


def finite_spent(value):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= BUDGET:
        raise ValueError('Missing or invalid cumulative active-time accounting')
    return float(value)


def remaining_job_budget(spent, elapsed, reservation=0.):
    spent, elapsed, reservation = map(finite_spent, (spent, elapsed, reservation))
    remaining = BUDGET - spent - elapsed - reservation
    if remaining < 335:
        raise TimeoutError('Require full330s job plus5s within original3600s aggregate')
    return remaining


def pending(evidence):
    keys = [(row['arm'], row['seed']) for row in evidence]
    if len(set(keys)) != len(keys) or any(a not in ARMS or s not in PROBES for a, s in keys):
        raise ValueError('Duplicate or unregistered frozen evidence')
    return [(arm, seed) for seed in PROBES for arm in ARMS if (arm, seed) not in keys]


def receipt(directory, arm, seed, bank_sha):
    from noesis_gpu.evaluate import summarize
    report, summary = summarize(directory)
    saved = load(directory.parent / (directory.name + '-summary.json'))
    if (report.get('seeds') != [seed] or report.get('checkpoint_sha256') != bank_sha
            or saved != summary):
        raise ValueError('Frozen receipt/hash/summary mismatch')
    return dict(arm=arm, seed=seed, directory=str(directory.resolve()),
        report_sha256=summary['report_sha256'], observations_sha256=summary['observations_sha256']), report, summary


def check_receipts(rows, banks, core_hash):
    from noesis_gpu.evaluate import MATCHED_SETTINGS
    pending(rows)
    reports, cases = [], {arm: {} for arm in ARMS}
    for row in rows:
        actual, report, summary = receipt(Path(row['directory']), row['arm'], row['seed'], banks[row['arm']])
        if actual != row:
            raise ValueError('Stored receipt mutated')
        if (report.get('noesis_source_sha256') != core_hash or report.get('backend') != 'cpu'
                or report.get('control_ms') != 5 or report.get('physics_dt') != .0001):
            raise ValueError('Frozen receipt source/timing context changed')
        reports.append(report); cases[row['arm']][row['seed']] = summary
    if reports and any(any(k not in r or k not in reports[0] or r[k] != reports[0][k]
                           for k in MATCHED_SETTINGS) for r in reports):
        raise ValueError('Frozen source/body/senses/settings mismatch')
    return cases


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--shared-root', type=Path, required=True)
    ap.add_argument('--core-source', type=Path, required=True)
    ap.add_argument('--study', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    started = time.monotonic()
    root, core, study, out = [p.resolve() for p in (args.shared_root, args.core_source, args.study, args.output)]
    if out.exists() or out == study or study in out.parents:
        raise ValueError('Require unique sibling output; never overwrite original evidence')
    if out == root / 'data' or root / 'data' in out.parents:
        raise ValueError('Output must stay outside read-only data')
    own = Path(__file__).resolve().parents[1]
    sources = dict(core=tree_digest(core), pilot=tree_digest(own / 'gpu-pilot/src'))
    if subprocess.check_output(['git', '-C', str(core.parent), 'rev-parse', 'HEAD'], text=True).strip() != CORE:
        raise ValueError('Wrong core commit')
    protocol = load(study / 'protocol.json')
    report = load(study / 'train25/report.json')
    full_training(report, protocol, sources)
    control_birth = root / 'gpu-pilot/outputs/logging-parity-d3-20261030/logged'
    paths = dict(parent=root / 'gpu-pilot/outputs/e4-exploration-20261029/train-control-1000/weights-last.npz',
        control5=control_birth / 'weights-last.npz', child25=study / 'train25/weights-last.npz')
    banks = {arm: digest(path) for arm, path in paths.items()}
    if banks['parent'] != PARENT or banks['control5'] != CONTROL or banks['child25'] != report.get('saved_checkpoint_sha256'):
        raise ValueError('Parent/control/child bank hash mismatch')
    if digest(control_birth / 'training-observations.npz') != protocol.get('reference_trace_sha256'):
        raise ValueError('Reference trace changed')
    audit_bank(paths['child25'], paths['parent']); prefix_check(study / 'train25', control_birth)
    ledger_path = study / 'evaluation-continuation.json'
    token = uuid.uuid4().hex
    lock = root / 'gpu-pilot/outputs/codex-training-host.lock'
    with os.fdopen(os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL), 'w') as stream:
        json.dump(dict(owner='codex', pid=os.getpid(), token=token, output=str(out), study=str(study),
            operation='frozen_evaluation_only', budget_seconds=BUDGET), stream)
    ledger, created, child = None, False, None
    def verify():
        if sources != dict(core=tree_digest(core), pilot=tree_digest(own / 'gpu-pilot/src')) or any(digest(paths[a]) != banks[a] for a in ARMS):
            raise ValueError('Source/bank mutation during continuation')
    try:
        inventory()
        if ledger_path.exists():
            ledger = load(ledger_path)
            if ledger.get('banks') != banks or ledger.get('sources') != sources or ledger.get('study') != str(study):
                raise ValueError('Continuation provenance changed')
        else:
            result = load(study / 'result.json')
            # Missing final accounting after hard termination is not assumed to be zero.
            spent = finite_spent(result.get('wall_seconds'))
            if spent < finite_spent(report.get('wall_seconds')):
                raise ValueError('Original study undercounts training wall time')
            if result.get('child25_sha256') != banks['child25']:
                raise ValueError('Original study never committed a verified complete child')
            evidence = []
            for seed in PROBES:
                for arm in ARMS:
                    directory = study / f'frozen-{arm}-{seed}'
                    if directory.exists():
                        rpath = directory / 'report.json'
                        if rpath.exists() and load(rpath).get('status') == 'completed_transport_pilot':
                            evidence.append(receipt(directory, arm, seed, banks[arm])[0])
            ledger = dict(study=str(study), banks=banks, sources=sources, spent_seconds=spent,
                reserved_seconds=0., evidence=evidence, segments=[])
        check_receipts(ledger['evidence'], banks, sources['core'])
        base = finite_spent(ledger['spent_seconds']) + finite_spent(ledger['reserved_seconds'])
        finite_spent(base)
        remaining_job_budget(base, time.monotonic() - started)
        resources = assess(sample_quiescence())
        if not resources['passed']:
            raise MemoryError('Fresh sustained resource gate refused; no simulation')
        out.mkdir(parents=True, exist_ok=False); created = True
        write(out / 'protocol.json', dict(root_study=str(study), original_protocol_sha256=digest(study / 'protocol.json'),
            sources=sources, banks=banks, prior_spent_seconds=base, total_budget_seconds=BUDGET,
            resource_preflight=resources, training_resumed=False, model_promoted=False))
        for arm, seed in pending(ledger['evidence']):
            inventory()
            verify()
            resources = assess(sample_quiescence()); write(out / f'frozen-{arm}-{seed}-preflight.json', resources)
            if not resources['passed']:
                raise MemoryError('Sustained resource gate refused')
            remaining_job_budget(base, time.monotonic() - started)
            ledger.update(spent_seconds=base + time.monotonic() - started, reserved_seconds=330.)
            write(ledger_path, ledger)  # Charge worst-case process time before launch; hard kills cannot reset budget.
            target = out / f'frozen-{arm}-{seed}'
            command = [sys.executable, '-m', 'noesis_gpu.run', '--backend', 'cpu', '--seconds', '5',
                '--wall-seconds', '300', '--seed', str(seed), '--worlds', '1', '--checkpoint', str(paths[arm]),
                '--noesis-source', str(core), '--data', str(root / 'data'), '--output', str(target)]
            print(f'Starting frozen-only continuation {arm}/{seed}', flush=True)
            with (out / f'frozen-{arm}-{seed}.log').open('w', encoding='utf-8') as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                try:
                    code = child.wait(timeout=330)
                finally:
                    if child.poll() is None: child.kill()
                    child.wait()
            if code:
                raise RuntimeError('Frozen job failed; partial evidence excluded')
            verify()
            from noesis_gpu.evaluate import summarize
            _, summary = summarize(target)
            write(out / f'frozen-{arm}-{seed}-summary.json', summary)
            row = receipt(target, arm, seed, banks[arm])[0]
            ledger['evidence'].append(row)
            check_receipts(ledger['evidence'], banks, sources['core'])
            ledger.update(spent_seconds=base + time.monotonic() - started, reserved_seconds=0.)
            write(ledger_path, ledger)
        verify()
        cases = check_receipts(ledger['evidence'], banks, sources['core'])
        write(out / 'result.json', dict(status='completed_duration_evaluation_continuation',
            decision=duration_decision(cases), training_resumed=False, model_promoted=False, walking_proven=False))
    finally:
        if ledger is not None and created:
            ledger.update(spent_seconds=base + time.monotonic() - started, reserved_seconds=0.)
            ledger['segments'].append(dict(output=str(out), active_seconds=time.monotonic() - started))
            write(ledger_path, ledger)
        if load(lock).get('token') == token: lock.unlink()


if __name__ == '__main__':
    with keep_system_awake(): main()
