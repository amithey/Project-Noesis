"""Synthetic safety tests; none establish physical retention or walking."""
import sys
import json
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from resume_duration_evaluation import (ARMS, CORE, PARENT, CONTROL, PROBES,
    full_training, finite_spent, remaining_job_budget, pending, receipt, check_receipts)


def context():
    sources = {'core': 'corehash', 'pilot': 'pilothash'}
    report = dict(status='completed_training_birth', seed=1000, requested_seconds=25,
        simulated_seconds=25, completed_chunks=5000, learning_updates=4990,
        learning_rate=.2, motor_exploration_mv_ms=1.5, cycle_weight=.1,
        import_checkpoint_sha256=PARENT, reward_objective='curriculum-gripped-cycle-v9',
        model_promoted=False, walking_proven=False, noesis_source_sha256='corehash',
        control_ms=5., physics_dt=.0001, eligibility_ms=100, birth_ms=50,
        original_weight_bound=.5, worlds=1, backend='cpu', learning=True)
    protocol = dict(parent_sha256=PARENT, control5_sha256=CONTROL, core_commit=CORE,
        source_hashes=sources, training_seed=1000, training_seconds=25,
        training_chunks=5000, learning_updates=4990, probe_seeds=list(PROBES),
        budget_seconds=3600, learning_rate=.2, cycle_weight=.1, exploration_mv_ms=1.5,
        training_inner_seconds=900, training_process_seconds=930,
        probe_inner_seconds=300, probe_process_seconds=330, minimum_available_gib=3,
        maximum_host_cpu_percent=30, resource_window_samples=10)
    return report, protocol, sources


def test_partial_bank_never_qualifies_even_if_renamed():
    report, protocol, sources = context()
    full_training(report, protocol, sources)
    for field, value in [('status', 'incomplete_wall_budget'), ('completed_chunks', 4236),
                         ('learning_updates', 4226), ('simulated_seconds', 21.18)]:
        changed = {**report, field: value}
        with pytest.raises(ValueError, match='partial banks'):
            full_training(changed, protocol, sources)


def test_context_and_scientific_gates_cannot_change_on_resume():
    report, protocol, sources = context()
    for field, value in [('source_hashes', {'core': 'other', 'pilot': 'pilothash'}),
                         ('probe_seeds', [10, 11]), ('budget_seconds', 5000), ('learning_rate', .002),
                         ('probe_process_seconds', 400), ('minimum_available_gib', 1)]:
        with pytest.raises(ValueError, match='context'):
            full_training(report, {**protocol, field: value}, sources)
    with pytest.raises(ValueError):
        full_training({**report, 'walking_proven': True}, protocol, sources)


def test_interrupted_reservation_is_charged_without_fresh_budget():
    # A hard interruption after reservation consumes its full330s on restart.
    assert remaining_job_budget(2500, 10, 330) == 760
    with pytest.raises(TimeoutError): remaining_job_budget(3000, 10, 330)
    assert remaining_job_budget(3200, 65) == 335
    with pytest.raises(TimeoutError): remaining_job_budget(3200, 65.001)
    for bad in [None, True, float('nan'), float('inf'), -1, 3601]:
        with pytest.raises(ValueError): finite_spent(bad)


def test_resume_keeps_registered_order_and_skips_only_receipted_jobs():
    evidence = [dict(arm='parent', seed=10006), dict(arm='control5', seed=10006)]
    jobs = pending(evidence)
    assert len(jobs) == 10 and jobs[0] == ('child25', 10006)
    assert evidence == [dict(arm='parent', seed=10006), dict(arm='control5', seed=10006)]
    assert pending([dict(arm=a, seed=s) for s in PROBES for a in ARMS]) == []
    for rows in [evidence + evidence[:1], [dict(arm='child25', seed=10)], [dict(arm='averaged', seed=10006)]]:
        with pytest.raises(ValueError): pending(rows)


def frozen_fixture(tmp_path):
    """Artificial arrays for receipt validation; no actual fly was simulated."""
    from noesis_gpu.evaluate import summarize, MATCHED_SETTINGS
    directory = tmp_path / 'frozen-child25-10006'; directory.mkdir()
    report = {k: {} for k in MATCHED_SETTINGS}
    report.update(status='completed_transport_pilot', controller='brain', worlds=1,
        learning=False, motor_exploration=0, weights_unchanged=True, backend='cpu',
        control_ms=5., physics_dt=.0001, completed_control_chunks=1000,
        requested_simulated_seconds_per_world=5, simulated_seconds_per_world=5,
        seeds=[10006], checkpoint_sha256='childhash', noesis_source_sha256='corehash')
    (directory / 'report.json').write_text(json.dumps(report), encoding='utf-8')
    arrays = {k: np.zeros((1000, 1, *s)) for k, s in dict(position=(3,), up_body=(3,),
        heading=(3,), feet_position=(6, 3), foot_load=(6,)).items()}
    arrays['up_body'][:, :, 2] = 1
    arrays['heading'][:, :, 0] = 1
    np.savez(directory / 'observations.npz', **arrays)
    _, summary = summarize(directory)
    (tmp_path / (directory.name + '-summary.json')).write_text(json.dumps(summary), encoding='utf-8')
    return directory, report, arrays


def test_modified_observations_cannot_reuse_a_completed_receipt(tmp_path):
    directory, report, arrays = frozen_fixture(tmp_path)
    row, _, _ = receipt(directory, 'child25', 10006, 'childhash')
    assert check_receipts([row], {'child25': 'childhash'}, 'corehash')['child25'][10006]['walking_proven'] is False
    with pytest.raises(ValueError, match='source/timing'):
        check_receipts([row], {'child25': 'childhash'}, 'wrongcore')
    arrays['position'][999, 0, 0] = .1
    np.savez(directory / 'observations.npz', **arrays)
    with pytest.raises(ValueError, match='receipt/hash/summary'):
        receipt(directory, 'child25', 10006, 'childhash')


def test_partial_or_wrong_bank_frozen_probe_is_not_reused(tmp_path):
    directory, report, _ = frozen_fixture(tmp_path)
    with pytest.raises(ValueError, match='receipt/hash/summary'):
        receipt(directory, 'child25', 10006, 'differentchild')
    with pytest.raises(ValueError, match='receipt/hash/summary'):
        receipt(directory, 'child25', 10007, 'childhash')
    report.update(status='incomplete_wall_budget', completed_control_chunks=800)
    (directory / 'report.json').write_text(json.dumps(report), encoding='utf-8')
    with pytest.raises(ValueError, match='completed frozen'):
        receipt(directory, 'child25', 10006, 'childhash')


@pytest.mark.parametrize('timeout', [False, True])
def test_process_failure_reserves_budget_first_excludes_evidence_and_releases_lease(tmp_path, monkeypatch, timeout):
    """Exercise real journal/lease writes with a fake child; no simulation starts."""
    import resume_duration_evaluation as m
    root = tmp_path / 'root'; (root / 'gpu-pilot/outputs').mkdir(parents=True)
    study = root / 'study'; (study / 'train25').mkdir(parents=True)
    out = root / 'continuation'; core = root / 'core/src'; core.mkdir(parents=True)
    for path, value in [(study / 'protocol.json', {'reference_trace_sha256': 'trace'}),
                        (study / 'train25/report.json', {'saved_checkpoint_sha256': 'child', 'wall_seconds': 10}),
                        (study / 'result.json', {'child25_sha256': 'child', 'wall_seconds': 10})]:
        path.write_text(json.dumps(value), encoding='utf-8')
    def fake_digest(path):
        if 'train-control-1000' in str(path): return PARENT
        if 'logging-parity' in str(path): return 'trace' if path.name.endswith('observations.npz') else CONTROL
        return 'child'
    monkeypatch.setattr(m, 'digest', fake_digest)
    monkeypatch.setattr(m, 'tree_digest', lambda _: 'source')
    monkeypatch.setattr(m.subprocess, 'check_output', lambda *a, **k: CORE)
    monkeypatch.setattr(m, 'full_training', lambda *a: None)
    monkeypatch.setattr(m, 'audit_bank', lambda *a: None)
    monkeypatch.setattr(m, 'prefix_check', lambda *a: None)
    monkeypatch.setattr(m, 'check_receipts', lambda *a: {})
    monkeypatch.setattr(m, 'inventory', lambda: [])
    monkeypatch.setattr(m, 'sample_quiescence', lambda: [])
    monkeypatch.setattr(m, 'assess', lambda _: {'passed': True})
    class Child:
        killed = False
        def __init__(self, *args, **kwargs):
            saved = json.loads((study / 'evaluation-continuation.json').read_text())
            assert saved['reserved_seconds'] == 330 and saved['spent_seconds'] >= 10
        def wait(self, timeout=None):
            if timeout and globals_timeout: raise m.subprocess.TimeoutExpired('fake', 330)
            return -9 if self.killed else 1
        def poll(self): return None if globals_timeout and not self.killed else 1
        def kill(self): self.killed = True
    globals_timeout = timeout
    child = None
    def spawn(*a, **k):
        nonlocal child
        child = Child(*a, **k); return child
    monkeypatch.setattr(m.subprocess, 'Popen', spawn)
    monkeypatch.setattr(sys, 'argv', ['resume', '--shared-root', str(root), '--core-source', str(core),
        '--study', str(study), '--output', str(out)])
    with pytest.raises(m.subprocess.TimeoutExpired if timeout else RuntimeError): m.main()
    saved = json.loads((study / 'evaluation-continuation.json').read_text())
    assert saved['evidence'] == [] and saved['reserved_seconds'] == 0
    assert saved['spent_seconds'] >= 10 and len(saved['segments']) == 1
    assert not (root / 'gpu-pilot/outputs/codex-training-host.lock').exists()
    assert child.killed is timeout
