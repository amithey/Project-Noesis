"""Re-audit complete ES1 native evidence and package named files only."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

from es1_proposal import assess

root = Path(__file__).resolve().parents[1]
out = root/'outputs/episodic-search-es1-20261106'
sys.path[:0] = [str(root/'work/Noesis-Sensory/gpu-pilot/src'),str(root/'work/Noesis-Core/src')]
from noesis_gpu.evaluate import summarize, compare

result = json.loads((out/'result.json').read_text())
protocol = json.loads((out/'protocol.json').read_text())
proposal = json.loads((out/'proposal.json').read_text())
assert result['status'] in ('completed_no_candidate','completed_fresh_rejection','completed_provisional_child')
assert result['parent_unchanged'] and result['source_unchanged'] and result['lease_released']
assert result['process_tree_empty_verified'] and result['parent_replay']=={str(s):True for s in protocol['dev_seeds']}
assert proposal['signs_topology_original_bounds_validated'] and proposal['nonplastic_weights_exact'] and proposal['no_clipping']


def sha(data):
    return hashlib.sha256(data).hexdigest()


parent = root/'work/evidence/s1-restored/gpu-pilot/outputs/e4-exploration-20261029/train-control-1000/weights-last.npz'
assert sha(parent.read_bytes())==protocol['parent_sha256']
with np.load(parent,allow_pickle=False) as z:
    original = {k:z[k].copy() for k in z.files}
banks = {}
for bank in ('plus','minus'):
    p = out/f'weights-proposed-{bank}.npz'
    assert sha(p.read_bytes())==proposal['bank_sha256'][bank]
    with np.load(p,allow_pickle=False) as z:
        assert set(z.files)==set(original)
        for key in z.files:
            if key=='weights':
                banks[bank]=z[key].copy()
            elif key in ('activity','eligibility','reward_baseline','reward_variance','reward_normalization_mass'):
                assert not z[key].any()
            else:
                np.testing.assert_array_equal(z[key],original[key])
    ratio=banks[bank]/original['initial_weights']
    assert np.isfinite(ratio).all() and ((ratio>=.5)&(ratio<=1.5)).all()
np.testing.assert_allclose((banks['plus']+banks['minus'])/2,original['weights'],rtol=0,
                           atol=2*np.finfo(float).eps*np.abs(original['weights']).max())
summaries={bank:{} for bank in ('parent','plus','minus')}
rows=[]
for case in result['cases']:
    bank,seed=case['bank'],case['seed']
    directory=out/f'{bank}-{seed}'
    assert sha((directory/'report.json').read_bytes())==case['report_sha256']
    assert sha((directory/'observations.npz').read_bytes())==case['observations_sha256']
    receipt=json.loads((out/f'{bank}-{seed}-receipt.json').read_text())
    assert receipt==case['process_receipt'] and receipt['exit_code']==0 and receipt['tree_empty'] and not receipt['timed_out']
    report,summary=summarize(directory)
    assert summary==case['summary'] and report['checkpoint_sha256']==proposal['bank_sha256'][bank]
    if bank!='parent':
        compare(out/f'parent-{seed}',directory)
    summaries[bank][seed]=summary
    row={key:summary[key] for key in ('standing_fraction','mean_stance_slip_mm_s','heading_forward_distance_mm',
                                     'net_forward_displacement_mm','total_leg_steps','leg_step_counts','tripod_coordination','locomotion_regime')}
    row.update(bank=bank,seed=seed)
    if bank!='parent':
        control=summaries['parent'][seed]
        row.update(standing_change_pp=100*(summary['standing_fraction']-control['standing_fraction']),
                   slip_change_percent=100*(summary['mean_stance_slip_mm_s']/control['mean_stance_slip_mm_s']-1),
                   heading_change_mm=summary['heading_forward_distance_mm']-control['heading_forward_distance_mm'],
                   net_forward_change_mm=summary['net_forward_displacement_mm']-control['net_forward_displacement_mm'])
    rows.append(row)
dev=protocol['dev_seeds']
for bank in ('plus','minus'):
    a={s:summaries['parent'][s] for s in dev}; b={s:summaries[bank][s] for s in dev}
    assert assess(a,b,dev)==result['development_screen'][bank]
eligible=[b for b in ('plus','minus') if result['development_screen'][b]['passed']]
if eligible:
    selected=max(eligible,key=lambda b:result['development_screen'][b]['ranking'])
    assert selected==result['selected_bank'] and len(rows)==16
    fresh=protocol['conditional_fresh_dev_seeds']
    assert assess({s:summaries['parent'][s] for s in fresh},{s:summaries[selected][s] for s in fresh},fresh)==result['fresh_screen']
else:
    assert result['status']=='completed_no_candidate' and result['selected_bank'] is None and len(rows)==12
audit=dict(status='completed_audited_ES1',runtime_commit=protocol['runtime_commit'],proposal_source_commit='0b4ca82321f0d97cb27b56f1746733d487943fe8',
           native_result_status=result['status'],seconds=result['seconds'],cases=rows,
           development_screen=result['development_screen'],fresh_screen=result.get('fresh_screen'),selected_bank=result['selected_bank'],
           exact_historical_parent_replay=result['parent_replay'],signs_original_bounds_topology_preserved=True,
           parent_unchanged=True,source_unchanged=True,all_worker_trees_empty=True,lease_released=True,
           model_promoted=False,walking_proven=False,
           limits=['ONE parameter direction/antithetic pair; not an optimizer-efficacy estimate.',
                   'Reused development probes select; conditional fresh development probes are not reserved final validation.',
                   'No within-episode learning or new gait/sensory policy; proposals are not completed native training births.',
                   'Strict screening does not establish stable walking/balance/environmental response or retained learning across generations.'])
audit_path=root/'outputs/episodic-search-es1-20261106-audit.json'
audit_path.write_text(json.dumps(audit,indent=2,allow_nan=False),encoding='utf-8')
lines=['# ES1 episodic synaptic search: observed results','',
       f"Status:{result['status']}; {len(rows)}full5s jobs, {result['seconds']:.3f}s. All four historical parent replays exact. Original-reference bounds/signs/topology/nonplastic weights preserved; all workers exit0/trees empty, source/parent unchanged, lease released.",'',
       'One direction60061,plus/minus2%original magnitude. No within-episode plasticity, external gait, active senses or model promotion.','',
       '|Seed|Bank|Standing %|Slip mm/s|Heading mm|Net-forward mm|Regime|',
       '|---|---|---:|---:|---:|---:|---|']
for r in rows:
    lines.append(f"|{r['seed']}|{r['bank']}|{100*r['standing_fraction']:.4f}|{r['mean_stance_slip_mm_s']:.5f}|{r['heading_forward_distance_mm']:.5f}|{r['net_forward_displacement_mm']:.5f}|{r['locomotion_regime']}|")
lines+=['','Development gates:','','```json',json.dumps(result['development_screen'],indent=2),'```','',
        'Limits:','']+['- '+x for x in audit['limits']]+['']
report_path=root/'outputs/episodic-search-es1-20261106-audit.md'
report_path.write_text('\n'.join(lines),encoding='utf-8')
files=[out/'protocol.json',out/'proposal.json',out/'result.json',audit_path,report_path]
for bank in ('plus','minus'):
    files.extend([out/f'weights-proposed-{bank}.npz',out/f'weights-proposed-{bank}.model.json'])
for case in result['cases']:
    name=f"{case['bank']}-{case['seed']}"
    files.extend([out/name/'report.json',out/name/'observations.npz',out/f'{name}-receipt.json'])
entries={p.relative_to(root/'outputs').as_posix():p.read_bytes() for p in files}
entries['common-parent/weights.npz']=parent.read_bytes()
entries['runtime-source-manifest.json']=(root/'outputs/thermo-pilot-tp2-20261106-source.json').read_bytes()
manifest={name:dict(bytes=len(data),sha256=sha(data)) for name,data in entries.items()}
archive=root/'outputs/episodic-search-es1-20261106-evidence.zip'
assert not archive.exists()
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for name,data in entries.items():z.writestr(name,data)
    z.writestr('manifest.json',json.dumps(manifest,indent=2))
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None and set(z.namelist())==set(entries)|{'manifest.json'}
    assert all(sha(z.read(name))==entry['sha256'] for name,entry in manifest.items())
receipt=dict(path=archive.name,bytes=archive.stat().st_size,sha256=sha(archive.read_bytes()),files=len(entries),
             all_hashes_and_crc_verified=True,raw_graph_cache_private_lease_excluded=True,release_uploaded=False)
(root/'outputs/episodic-search-es1-20261106-artifact.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
print(json.dumps(dict(status=audit['native_result_status'],gates=audit['development_screen'],artifact=receipt),indent=2))
