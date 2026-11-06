"""Recheck completed TP2 raw controls, summarize deltas, package named numerical files."""
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

root = Path(__file__).resolve().parents[1]
out = root / 'outputs/thermo-pilot-tp2-20261106'
result = json.loads((out / 'result.json').read_text())
assert result['status'] == 'completed_TP2' and len(result['cases']) == 6
assert result['source_unchanged'] and result['parent_unchanged']
assert result['process_tree_empty_verified'] and result['lease_released']
assert result['plain_gate_by_seed'] == result['rng_gate_by_seed'] == {'10006': True, '10007': True}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def exact(a, b):
    with np.load(a, allow_pickle=False) as x, np.load(b, allow_pickle=False) as y:
        return set(x.files) == set(y.files) and all(np.array_equal(x[k], y[k]) for k in x.files)


rows, replay = [], {}
for seed in (10006, 10007):
    dirs = {mode: out / f'{mode}-{seed}' for mode in ('plain', 'sham', 'stimulus')}
    with np.load(dirs['sham'] / 'tp2-observations.npz', allow_pickle=False) as a, np.load(dirs['stimulus'] / 'tp2-observations.npz', allow_pickle=False) as b:
        for key in ('extra_idx', 'extra_rate_hz', 'drive_length', 'drive_indices_sha256', 'rng_state'):
            np.testing.assert_array_equal(a[key], b[key])
        assert a['rng_state'].shape == (1000, 625) and a['rng_state'].dtype == np.uint32
    replay[str(seed)] = exact(dirs['stimulus'] / 'observations.npz', root / f'outputs/thermo-pilot-tp1-20261106/constant-{seed}/observations.npz')
    assert replay[str(seed)], 'Prospective TP1 constant replay consistency failed'
    pair = {c['mode']: c for c in result['cases'] if c['seed'] == seed}
    for mode in ('plain', 'sham', 'stimulus'):
        case, path = pair[mode], dirs[mode]
        for name, key in [('report.json', 'report_sha256'), ('observations.npz', 'observations_sha256'),
                          ('tp2-observations.npz', 'observer_sha256'), ('afferents.json', 'afferents_sha256')]:
            assert sha((path / name).read_bytes()) == case[key]
        report = json.loads((path / 'report.json').read_text())
        assert report['weights_unchanged'] and report['completed_control_chunks'] == 1000
        receipt = case['process_receipt']
        assert receipt['tree_empty'] and not receipt['timed_out'] and receipt['exit_code'] == 0
        summary = case['summary']
        row = {key: case[key] for key in ('seed', 'mode', 'warm_rate_hz_per_neuron', 'warm_hz_per_cell', 'cool_rate_hz_per_neuron')}
        for key in ('standing_fraction', 'mean_stance_slip_mm_s', 'heading_forward_distance_mm',
                    'net_forward_displacement_mm', 'leg_step_counts', 'total_leg_steps', 'tripod_coordination', 'locomotion_regime'):
            row[key] = summary[key]
        if mode == 'stimulus':
            sham = pair['sham']
            row.update(warm_gain_hz_per_cell=case['warm_rate_gain_vs_sham'],
                       warm_gain_per_cell=(np.asarray(case['warm_hz_per_cell'])-sham['warm_hz_per_cell']).tolist(),
                       standing_change_pp=100*(summary['standing_fraction']-sham['summary']['standing_fraction']),
                       slip_change_percent=100*(summary['mean_stance_slip_mm_s']/sham['summary']['mean_stance_slip_mm_s']-1),
                       neural_criterion_pass=case['neural_criterion_pass'], retention_criterion_pass=case['retention_criterion_pass'])
        rows.append(row)

audit = dict(status='completed_audited_TP2', source_commit='e0cbd109bb6e018bd253c9f1b5e1698e0421b276',
             cases=rows, seconds=result['seconds'], full_MT_state_gate=result['rng_gate_by_seed'],
             plain_B2_replay=result['plain_gate_by_seed'], stimulus_TP1_constant_replay=replay,
             all_neural_criteria_pass=result['all_neural_criteria_pass'],
             all_retention_criteria_pass=result['all_retention_criteria_pass'],
             weights_unchanged=True, source_unchanged=True, lease_released=True,
             learning=False, model_promoted=False, walking_proven=False,
             limits=['Two reused development seeds, frozen5s probes; no independent final validation or retained-learning evidence.',
                     'Constant analytic warm input, not changing environmental temperature or thermotaxis.',
                     'Sham retains zero-weight event/active-set processing; it is not a no-input deployment baseline.',
                     'This amplitude contrast controls the identified RNG nuisance, not every possible biological modelling error.',
                     'Strict physical screen is descriptive; no promotion, default activation or gain retuning.'])
audit_path = root / 'outputs/thermo-pilot-tp2-20261106-audit.json'
audit_path.write_text(json.dumps(audit, indent=2, allow_nan=False), encoding='utf-8')
lines = ['# TP2: RNG-matched constant thermo amplitude', '',
         f"Six full5s jobs completed in{result['seconds']:.3f}s. All1000post-chunk full MT states and drive indices/rates match within each pair. Both plain probes replay B2, and both stimulus probes replay TP1 constant input exactly. All workers exit0 with entire Windows jobs empty; source/parent unchanged, lease released.", '',
         f"Registered neural screen:{result['all_neural_criteria_pass']}; strict physical screen:{result['all_retention_criteria_pass']}. No learning, checkpoint, model promotion or stable-walking claim.", '',
         '|Seed|Mode|Warm Hz/cell|Standing %|Slip mm/s|Heading mm|Regime|',
         '|---|---|---:|---:|---:|---:|---|']
for row in rows:
    lines.append(f"|{row['seed']}|{row['mode']}|{row['warm_rate_hz_per_neuron']:.4f}|{100*row['standing_fraction']:.4f}|{row['mean_stance_slip_mm_s']:.5f}|{row['heading_forward_distance_mm']:.5f}|{row['locomotion_regime']}|")
lines += ['', 'Stimulus minus matched sham:', '',
          '|Seed|Warm gain Hz/cell|Standing change pp|Slip change %|Neural pass|Physical pass|',
          '|---|---:|---:|---:|---|---|']
for row in rows:
    if row['mode'] == 'stimulus':
        lines.append(f"|{row['seed']}|{row['warm_gain_hz_per_cell']:.4f}|{row['standing_change_pp']:.4f}|{row['slip_change_percent']:.4f}|{row['neural_criterion_pass']}|{row['retention_criterion_pass']}|")
lines += ['', 'Interpretation limits:', ''] + ['- '+limit for limit in audit['limits']] + ['']
report_path = root / 'outputs/thermo-pilot-tp2-20261106-audit.md'
report_path.write_text('\n'.join(lines), encoding='utf-8')

files = [out / 'protocol.json', out / 'result.json', audit_path, report_path,
         root / 'outputs/thermo-pilot-tp2-20261106-source.json']
for row in rows:
    name = f"{row['mode']}-{row['seed']}"
    files.extend(out / name / f for f in ('report.json', 'observations.npz', 'tp2-observations.npz', 'afferents.json'))
    files.append(out / f'{name}-receipt.json')
entries = {p.relative_to(root / 'outputs').as_posix(): p.read_bytes() for p in files}
manifest = {name: dict(bytes=len(data), sha256=sha(data)) for name, data in entries.items()}
archive = root / 'outputs/thermo-pilot-tp2-20261106-evidence.zip'
assert not archive.exists()
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    for name, data in entries.items():
        z.writestr(name, data)
    z.writestr('manifest.json', json.dumps(manifest, indent=2))
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None and set(z.namelist()) == set(entries) | {'manifest.json'}
    assert all(sha(z.read(name)) == entry['sha256'] for name, entry in manifest.items())
receipt = dict(path=archive.name, bytes=archive.stat().st_size, sha256=sha(archive.read_bytes()),
               files=len(entries), crc_and_all_hashes_verified=True, raw_graph_cache_weights_private_lease_excluded=True,
               release_uploaded=False, release_blocker='CLI authentication invalid; browser logged out at last verified check.')
(root / 'outputs/thermo-pilot-tp2-20261106-artifact.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
print(json.dumps(dict(neural=audit['all_neural_criteria_pass'], physical=audit['all_retention_criteria_pass'],
                      stimulus_replay=replay, artifact=receipt), indent=2))
