"""Independent A1 raw-evidence checks and explicit one-factor interpretation."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
from adhesion_mapping import screen

root=Path(__file__).resolve().parents[1]
out=root/'outputs/adhesion-sign-a1-20261106'
sys.path[:0]=[str(root/'work/Noesis-Sensory/gpu-pilot/src'),str(root/'work/Noesis-Core/src')]
from noesis_gpu.evaluate import summarize,compare,MATCHED_SETTINGS

result=json.loads((out/'result.json').read_text())
protocol=json.loads((out/'protocol.json').read_text())
assert result['status']=='completed_A1' and len(result['cases'])==8
assert result['parent_unchanged'] and result['source_unchanged'] and result['data_unchanged']
assert result['process_tree_empty_verified'] and result['lease_released']
assert result['legacy_replay']==result['one_factor_context']=={str(s):True for s in protocol['seeds']}


def sha(b):
    return hashlib.sha256(b).hexdigest()


def exact(a,b):
    with np.load(a,allow_pickle=False) as x,np.load(b,allow_pickle=False) as y:
        return set(x.files)==set(y.files) and all(np.array_equal(x[k],y[k]) for k in x.files)


summaries={m:{} for m in ('legacy','corrected')};reports={m:{} for m in summaries};rows=[]
for case in result['cases']:
    mode,seed=case['mode'],case['seed'];directory=out/f'{mode}-{seed}'
    for name,key in [('report.json','report_sha256'),('observations.npz','observations_sha256'),
                     ('adhesion-observations.npz','adhesion_observations_sha256')]:
        assert sha((directory/name).read_bytes())==case[key]
    receipt=json.loads((out/f'{mode}-{seed}-receipt.json').read_text())
    assert receipt==case['process_receipt'] and receipt['exit_code']==0 and not receipt['timed_out'] and receipt['tree_empty']
    report,summary=summarize(directory)
    assert summary==case['summary'] and report['checkpoint_sha256']==protocol['parent_sha256']
    with np.load(directory/'adhesion-observations.npz',allow_pickle=False) as z:
        assert z['desired'].shape==(1000,42) and z['pads'].shape==z['Femur_delta'].shape==(1000,6)
        assert all(np.isfinite(z[k]).all() for k in z.files)
        expected=((z['Femur_delta'] if mode=='legacy' else -z['Femur_delta'])<=.08).astype(float)
        np.testing.assert_array_equal(z['pads'],expected)
        np.testing.assert_array_equal((z['desired']-report['body']['neutral_pose']).reshape(1000,6,7)[:,:,3],z['Femur_delta'])
        release=(z['pads'][10:]==0).mean(axis=0)
        np.testing.assert_array_equal(release,case['per_leg_pad_release_fraction'])
    if mode=='legacy':
        assert exact(directory/'observations.npz',root/f'outputs/episodic-search-es1-20261106/parent-{seed}/observations.npz')
    summaries[mode][seed]=summary;reports[mode][seed]=report
    row={key:summary[key] for key in ('standing_fraction','mean_stance_slip_mm_s','heading_forward_distance_mm',
                  'net_forward_displacement_mm','total_leg_steps','leg_step_counts','tripod_coordination','locomotion_regime')}
    row.update(mode=mode,seed=seed,per_leg_pad_release_fraction=release.tolist(),mean_pad_release_fraction=float(release.mean()))
    if mode=='corrected':
        old=summaries['legacy'][seed]
        row.update(standing_change_pp=100*(summary['standing_fraction']-old['standing_fraction']),
                   slip_change_percent=100*(summary['mean_stance_slip_mm_s']/old['mean_stance_slip_mm_s']-1),
                   heading_change_mm=summary['heading_forward_distance_mm']-old['heading_forward_distance_mm'],
                   net_forward_change_mm=summary['net_forward_displacement_mm']-old['net_forward_displacement_mm'])
    rows.append(row)
for seed in protocol['seeds']:
    a,b=reports['legacy'][seed],reports['corrected'][seed]
    for key in MATCHED_SETTINGS:
        if key=='body':
            ab,bb=dict(a[key]),dict(b[key])
            assert ab.pop('adhesion_mapping')=='legacy-positive-Femur'
            assert bb.pop('adhesion_mapping')=='corrected-negative-Femur'
            assert ab==bb
        else:assert a[key]==b[key]
    try:
        compare(out/f'legacy-{seed}',out/f'corrected-{seed}')
    except ValueError:
        pass
    else:
        raise AssertionError('Ordinary compare silently equated different body mappings')
assert screen(summaries['legacy'],summaries['corrected'],protocol['seeds'])==result['physical_screen']
audit=dict(status='completed_audited_A1',source_commit='dd61cd9ec5663d8d452ee141f8f8009c1ad7dd38',seconds=result['seconds'],
           physical_screen=result['physical_screen'],cases=rows,all_legacy_replays_exact=True,all_pad_formulas_exact=True,
           one_factor_context_verified=True,ordinary_compare_rejects_different_mappings=True,
           source_parent_data_unchanged=True,all_worker_trees_empty=True,lease_released=True,
           learning=False,weights_unchanged=True,model_promoted=False,walking_proven=False,
           limits=['Four reused development seeds, same learned parent and5s horizon; no final validation.',
                   'FK1 static lift convention and this frozen dynamic performance question are distinct.',
                   'A bank trained under legacy adhesion can react differently to a sign-corrected interface; this test does not measure retraining.',
                   'Simulated adhesion is an assumed neural-command mapping, not validated biological force control.',
                   'A physical screen pass alone is not stable walking/balance recovery or retained learning; a failure does not undo static geometry.'])
audit_path=root/'outputs/adhesion-sign-a1-20261106-audit.json'
audit_path.write_text(json.dumps(audit,indent=2,allow_nan=False),encoding='utf-8')
lines=['# A1: frozen dynamic pad-sign comparison','',
       f"Eight full5s jobs completed in{result['seconds']:.3f}s. All legacy replays and pad formulas exact; only explicit body.adhesion_mapping differs. Ordinary compare rejects the changed mapping. Source/parent/data unchanged, all workers exit0/trees empty, lease released.",'',
       f"Registered strict physical screen:{result['physical_screen']['passed']}. No weights updated, training, model adoption or walking claim.",'',
       '|Seed|Mapping|Standing %|Slip mm/s|Heading mm|Net-forward mm|Released leg-frames %|Regime|',
       '|---|---|---:|---:|---:|---:|---:|---|']
for r in rows:
    lines.append(f"|{r['seed']}|{r['mode']}|{100*r['standing_fraction']:.4f}|{r['mean_stance_slip_mm_s']:.5f}|{r['heading_forward_distance_mm']:.5f}|{r['net_forward_displacement_mm']:.5f}|{100*r['mean_pad_release_fraction']:.3f}|{r['locomotion_regime']}|")
lines+=['','Registered screen:','','```json',json.dumps(result['physical_screen'],indent=2),'```','','Limits:','']+['- '+x for x in audit['limits']]+['']
report_path=root/'outputs/adhesion-sign-a1-20261106-audit.md'
report_path.write_text('\n'.join(lines),encoding='utf-8')
files=[out/'protocol.json',out/'result.json',audit_path,report_path]
for case in result['cases']:
    name=f"{case['mode']}-{case['seed']}"
    files.extend([out/name/'report.json',out/name/'observations.npz',out/name/'adhesion-observations.npz',out/f'{name}-receipt.json'])
entries={p.relative_to(root/'outputs').as_posix():p.read_bytes() for p in files}
entries['runtime-source-manifest.json']=(root/'outputs/thermo-pilot-tp2-20261106-source.json').read_bytes()
parent=root/'work/evidence/s1-restored/gpu-pilot/outputs/e4-exploration-20261029/train-control-1000/weights-last.npz'
assert sha(parent.read_bytes())==protocol['parent_sha256']
entries['common-parent/weights.npz']=parent.read_bytes()
manifest={name:dict(bytes=len(data),sha256=sha(data)) for name,data in entries.items()}
archive=root/'outputs/adhesion-sign-a1-20261106-evidence.zip';assert not archive.exists()
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for name,data in entries.items():z.writestr(name,data)
    z.writestr('manifest.json',json.dumps(manifest,indent=2))
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None and set(z.namelist())==set(entries)|{'manifest.json'}
    assert all(sha(z.read(name))==entry['sha256'] for name,entry in manifest.items())
receipt=dict(path=archive.name,bytes=archive.stat().st_size,sha256=sha(archive.read_bytes()),files=len(entries),
             crc_and_all_hashes_verified=True,raw_graph_cache_private_lease_excluded=True,release_uploaded=False)
(root/'outputs/adhesion-sign-a1-20261106-artifact.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
print(json.dumps(dict(screen=result['physical_screen'],artifact=receipt),indent=2))
