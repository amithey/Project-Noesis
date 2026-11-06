"""Audit and package completed TP1 numerical evidence; never changes runtime or weights."""
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

root = Path(__file__).resolve().parents[1]
out = root / 'outputs/thermo-pilot-tp1-20261106'
result = json.loads((out / 'result.json').read_text())
assert result['status'] == 'completed_thermo_integration_pilot'
assert result['parent_unchanged'] and result['source_unchanged'] and result['lease_released']
assert result['gate0_by_seed'] == {'10006': True, '10007': True}
assert len(result['cases']) == 10

def sha(data):
    return hashlib.sha256(data).hexdigest()

rows = []
for case in result['cases']:
    path = out / f"{case['mode']}-{case['seed']}"
    for filename, key in [('observations.npz', 'observations_sha256'),
                          ('sensory-observations.npz', 'sensory_sha256'), ('report.json', 'report_sha256')]:
        assert sha((path / filename).read_bytes()) == case[key]
    report = json.loads((path / 'report.json').read_text())
    assert report['weights_unchanged'] and report['completed_control_chunks'] == 1000
    assert report['simulated_seconds_per_world'] == 5 and report['learning'] is False
    summary = case['summary']
    row = {k: case[k] for k in ('seed', 'mode', 'warm_rate_hz_per_neuron', 'cool_rate_hz_per_neuron',
                                'requested_warm_hz_mean', 'requested_cool_hz_mean', 'u_min', 'u_max')}
    for key in ('standing_fraction', 'mean_stance_slip_mm_s', 'heading_forward_distance_mm',
                'net_forward_displacement_mm', 'total_leg_steps', 'leg_step_counts', 'tripod_coordination',
                'locomotion_regime'):
        row[key] = summary[key]
    with np.load(path / 'sensory-observations.npz', allow_pickle=False) as z:
        assert all(len(z[k]) == 1000 and np.isfinite(z[k]).all() for k in z.files)
        row['input_x_min_mm'] = float(z['input_position'][:, 0].min())
        row['input_x_max_mm'] = float(z['input_position'][:, 0].max())
    if case['mode'] in ('constant', 'active'):
        control = next(c for c in result['cases'] if c['seed'] == case['seed'] and c['mode'] == 'off')
        before = control['summary']
        row.update(warm_rate_gain_vs_off=case['warm_rate_gain_vs_off'],
                   standing_change_pp=100 * (summary['standing_fraction'] - before['standing_fraction']),
                   slip_change_percent=100 * (summary['mean_stance_slip_mm_s'] / before['mean_stance_slip_mm_s'] - 1),
                   heading_change_mm=summary['heading_forward_distance_mm'] - before['heading_forward_distance_mm'],
                   neural_criterion_pass=case['neural_criterion_pass'],
                   retention_criterion_pass=case['retention_criterion_pass'])
    rows.append(row)

audit = dict(status='completed_audited_TP1', source_head='137ca051281f1e7c60aaaad78c3057f4cc4f0142',
             seconds=result['seconds'], gate0_by_seed=result['gate0_by_seed'],
             all_neural_criteria_pass=result['all_neural_criteria_pass'],
             all_retention_criteria_pass=result['all_retention_criteria_pass'], cases=rows,
             parent_unchanged=True, source_unchanged=True, lease_released=True,
             learning=False, model_promoted=False, walking_proven=False,
             limitations=[
                 'Two reused development seeds; five seconds each, no independent final validation.',
                 'Analytic normalized field; no calibrated thermal physics or learned thermotaxis.',
                 'Constant matches active at x=0 only; differing exposure confounds pattern contrasts.',
                 'Extra drive indices consume random numbers before background draws in the shared core stream. Shared seeds do not hold background realization fixed across stimulus/off.',
                 'Afferent spikes are channel totals over six cells, not per-cell response estimates.',
                 'Frozen descriptive retention screen cannot prove retention across training or stable walking.'
             ])
compact = root / 'outputs/thermo-pilot-tp1-20261106-audit.json'
compact.write_text(json.dumps(audit, indent=2, allow_nan=False), encoding='utf-8')
lines = ['# TP1 frozen thermosensory integration: observed results', '',
         f"All ten full 5s/1000-chunk jobs completed in {result['seconds']:.3f}s. Plain exactly replayed B2 on both development seeds; off and zero exactly matched plain in every native and observer array. Source/parent unchanged; owned lease released.", '',
         f"Registered neural screen: {result['all_neural_criteria_pass']}. Strict standing/slip retention screen: {result['all_retention_criteria_pass']}. No learning, checkpoint, promotion, default activation or walking claim.", '',
         '|Seed|Mode|Warm spikes Hz/cell|Requested warm Hz|Standing %|Slip mm/s|Heading mm|Regime|',
         '|---|---|---:|---:|---:|---:|---:|---|']
for r in rows:
    lines.append(f"|{r['seed']}|{r['mode']}|{r['warm_rate_hz_per_neuron']:.4f}|{r['requested_warm_hz_mean']:.4f}|{100*r['standing_fraction']:.4f}|{r['mean_stance_slip_mm_s']:.5f}|{r['heading_forward_distance_mm']:.5f}|{r['locomotion_regime']}|")
lines += ['', 'Stimulus-arm deltas versus matched off:', '',
          '|Seed|Mode|Warm gain Hz/cell|Standing change pp|Slip change %|Neural pass|Retention pass|',
          '|---|---|---:|---:|---:|---|---|']
for r in rows:
    if r['mode'] in ('constant', 'active'):
        lines.append(f"|{r['seed']}|{r['mode']}|{r['warm_rate_gain_vs_off']:.4f}|{r['standing_change_pp']:.4f}|{r['slip_change_percent']:.4f}|{r['neural_criterion_pass']}|{r['retention_criterion_pass']}|")
lines += ['', 'Interpretation limits:', ''] + ['- ' + x for x in audit['limitations']]
lines += ['', 'Source/protocol: PR30, source137ca05, core42c2076, completed parent9fd7d411. Numerical artifact contains only explicit report/observer/body arrays and public scientific protocol/result; no raw connectome, derived graph cache, weights, private lease, token, PID or environment.', '']
(root / 'outputs/thermo-pilot-tp1-20261106-audit.md').write_text('\n'.join(lines), encoding='utf-8')

files = [out / 'protocol.json', out / 'result.json', compact,
         root / 'outputs/thermo-pilot-tp1-20261106-audit.md']
for r in rows:
    folder = out / f"{r['mode']}-{r['seed']}"
    files += [folder / n for n in ('report.json', 'observations.npz', 'sensory-observations.npz')]
entries = {p.relative_to(root / 'outputs').as_posix(): p.read_bytes() for p in files}
manifest = {name: dict(bytes=len(data), sha256=sha(data)) for name, data in entries.items()}
archive = root / 'outputs/thermo-pilot-tp1-20261106-evidence.zip'
assert not archive.exists()
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    for name, data in entries.items():
        z.writestr(name, data)
    z.writestr('manifest.json', json.dumps(manifest, indent=2))
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    assert set(z.namelist()) == set(entries) | {'manifest.json'}
    for name, entry in manifest.items():
        assert sha(z.read(name)) == entry['sha256']
receipt = dict(path=archive.name, bytes=archive.stat().st_size, sha256=sha(archive.read_bytes()),
               files=len(entries), manifest_verified=True, raw_data_excluded=True,
               release_uploaded=False, release_blocker='Local gh authentication was invalid at last check; no authenticated release uploader available.')
(root / 'outputs/thermo-pilot-tp1-20261106-artifact.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
print(json.dumps(dict(neural=audit['all_neural_criteria_pass'], retention=audit['all_retention_criteria_pass'],
                      seconds=result['seconds'], artifact=receipt), indent=2))
