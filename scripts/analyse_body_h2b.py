"""Audit exact H2B replays and summarize measured COM/ground/actuator contributions."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
from scipy.spatial import ConvexHull,QhullError
from height_audit_h1 import masks,quantiles

root=Path(__file__).resolve().parents[1];out=root/'outputs/body-telemetry-h2b-20261106'
sys.path[:0]=[str(root/'work/Noesis-Sensory/gpu-pilot/src'),str(root/'work/Noesis-Core/src')]
from noesis_gpu.evaluate import summarize,compare
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def exact(a,b):
    with np.load(a,allow_pickle=False) as x,np.load(b,allow_pickle=False) as y:
        assert set(x.files)==set(y.files)
        for k in x.files:np.testing.assert_array_equal(x[k],y[k])


p=read(out/'protocol.json');r=read(out/'result.json')
assert r['status']=='completed_H2B' and len(r['jobs'])==len(r['cases'])==3
for k in ('source_prior_checkpoints_unchanged','data_unchanged','process_tree_empty_verified','lease_released'):assert r[k]
assert not (root/'work/Project-Noesis/gpu-pilot/outputs/codex-training-host.lock').exists()
for n,h in p['source_hashes'].items():assert sha(root/n)==h
for n,h in p['prior_hashes'].items():assert sha(root/n)==h
for f in read(root/'outputs/noesis-required-data-sha256.json')['files']:assert sha(root/'work/NoesisData'/f['path'])==f['sha256']
rows=[]
for c in p['cases']:
    name=c['name'];directory=out/name;record=next(x for x in r['cases'] if x['name']==name)
    job=next(x for x in r['jobs'] if x['name']==name);receipt=read(out/(name+'-receipt.json'))
    assert receipt==job['receipt'] and receipt['exit_code']==0 and receipt['tree_empty'] and not receipt['timed_out']
    assert job['available_ram_before_gib']>=2
    for file,key in [('telemetry-model.json','metadata_sha256'),('body-telemetry.npz','telemetry_sha256'),('ground-contacts.npz','contacts_sha256'),('report.json','report_sha256')]:assert sha(directory/file)==record[key]
    for n in ('observations.npz','adhesion-observations.npz'):exact(directory/n,root/c['baseline']/n)
    report,summary=summarize(directory);assert summary==record['summary'];compare(root/c['baseline'],directory)
    assert report['spikes_per_world']==read(root/c['baseline']/'report.json')['spikes_per_world']
    assert report['checkpoint_sha256']==sha(root/c['checkpoint']) and report['telemetry_samples']==1001 and report['read_only_telemetry']
    m=read(directory/'telemetry-model.json');assert m['critical_thorax_claw_ids_valid'] and m['production_static_fusion_unchanged']
    assert m['unmapped_anatomical_count']==1 and np.isclose(sum(m['body_mass']),m['model_total_mass'],rtol=1e-12,atol=1e-15)
    assert m['gravity'][2]<0 and np.linalg.norm(m['gravity'][:2])==0
    weight=m['total_mass']*abs(m['gravity'][2])
    with np.load(directory/'body-telemetry.npz',allow_pickle=False) as z:t={k:z[k].copy() for k in z.files}
    with np.load(directory/'ground-contacts.npz',allow_pickle=False) as z:contacts={k:z[k].copy() for k in z.files}
    with np.load(directory/'observations.npz',allow_pickle=False) as z:obs={k:z[k][:,0].copy() for k in z.files}
    assert all(len(a)==1001 and np.isfinite(a).all() for a in t.values())
    assert all(np.isfinite(a).all() for a in contacts.values())
    assert t['qpos'].shape==(1001,m['nq']) and t['qvel'].shape==(1001,m['nv'])
    np.testing.assert_allclose(t['time'],np.arange(1001)*.005,atol=1e-8,rtol=0)
    np.testing.assert_allclose((t['body_xipos']*np.asarray(m['body_mass'])[None,:,None]).sum(1)/m['total_mass'],t['com'],rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(t['ground_force'],t['root_constraint_force'],rtol=1e-6,atol=1e-6*weight+1e-10)
    assert t['force_closure_max_error'].max()<=1e-6*weight+1e-10 and t['upstream_force_max_error'].max()<=1e-10
    # Reconstruct aggregate forces independently from the ragged contact table.
    aggregate=np.zeros((1001,3));np.add.at(aggregate,contacts['sample_index'],contacts['force_on_fly'])
    np.testing.assert_allclose(aggregate,t['ground_force'],rtol=1e-12,atol=1e-10)
    measured_norm=np.linalg.norm(t['claw_ground_force'][1:],axis=2)
    sensor_error=float(np.max(np.abs(measured_norm-obs['foot_load'])))
    assert sensor_error<=1e-9
    assert not np.any(t['xfrc_applied']) and not np.any(t['root_applied_force'])
    # COM projection relative to actual loaded claw-contact points: descriptive
    # geometry only, NOT static-balance certification in the presence of adhesion.
    defined=np.zeros(1000,bool);margin=np.zeros(1000);normal=np.einsum('ij,ij->i',contacts['force_on_fly'],contacts['normal_on_fly'])
    isclaw=np.isin(contacts['body_id'],m['claw_body_ids'])&(normal>1e-5)
    body_names=dict(zip(m['body_ids'],m['body_names']))
    def region(body):
        name=body_names[int(body)].rsplit('/',1)[-1]
        if name.startswith('c_'):return 'central_body_including_fused_head'
        if name.endswith('_tarsus5'):return 'claw_tarsus5'
        if '_tarsus' in name:return 'other_tarsal'
        return 'proximal_leg_or_other'
    regions=np.array([region(b) for b in contacts['body_id']])
    central=np.zeros(1001)
    select_central=regions=='central_body_including_fused_head'
    np.add.at(central,contacts['sample_index'][select_central],contacts['force_on_fly'][select_central,2])
    for i in range(1000):
        use=isclaw&(contacts['sample_index']==i+1);pts=np.unique(contacts['position'][use,:2],axis=0)
        if len(pts)<3 or np.linalg.matrix_rank(pts-pts[0],tol=1e-12)<2:continue
        try:hull=ConvexHull(pts)
        except QhullError:continue
        values=hull.equations[:,:2]@t['com'][i+1,:2]+hull.equations[:,2]
        margin[i]=float(np.min(-values/np.linalg.norm(hull.equations[:,:2],axis=1)));defined[i]=True
    standing,high=masks(obs);measured=np.arange(1000)>=10
    groups={'all_measured':measured,'standing':standing&measured,'high_only':high&measured,
            'upright_failure':(obs['up_body'][:,2]<.7)&measured}
    summaries={}
    for label,mask in groups.items():
        samples=np.flatnonzero(mask)+1;poly=mask&defined;contact_use=np.isin(contacts['sample_index'],samples)
        ground=t['ground_force'][samples,2];claw_z=t['claw_ground_force'][samples,:,2].sum(1)
        ratio=np.divide(ground-claw_z,ground,out=np.zeros_like(ground),where=np.abs(ground)>1e-12)
        normalized_margin=((obs['angles']-np.asarray(report['body']['lower_limits']))/
            (np.asarray(report['body']['upper_limits'])-np.asarray(report['body']['lower_limits'])))
        near=np.minimum(normalized_margin,1-normalized_margin).min(1)
        total=float(contacts['force_on_fly'][contact_use,2].sum())
        shares={str(key):float(contacts['force_on_fly'][contact_use&(regions==key),2].sum()/total)
                for key in np.unique(regions) if np.any(contact_use&(regions==key))} if abs(total)>1e-12 else None
        if shares is not None:assert abs(sum(shares.values())-1)<1e-12
        summaries[label]=dict(samples=len(samples),thorax_height_mm=quantiles(obs['position'][mask,2]),
            whole_COM_height_mm=quantiles(t['com'][samples,2]),whole_COM_velocity_z_mm_s=quantiles(t['com_velocity'][samples,2]),
            whole_COM_abs_velocity_z_mm_s=quantiles(np.abs(t['com_velocity'][samples,2])),
            ground_vertical_force_over_weight=quantiles(ground/weight),
            root_actuator_vertical_force_over_weight=quantiles(t['root_actuator_force'][samples,2]/weight),
            recorded_contact_plus_actuator_vertical_over_weight=quantiles((ground+t['root_actuator_force'][samples,2])/weight),
            nonclaw_fraction_of_ground_vertical_force=quantiles(ratio[np.abs(ground)>1e-12]),
            actual_contact_position_z_mm=quantiles(contacts['position'][contact_use,2]),
            actual_contact_normal_z=quantiles(contacts['normal_on_fly'][contact_use,2]),
            claw_contact_polygon_defined_count=int(poly.sum()),
            COM_projection_inside_claw_polygon_fraction=float((margin[poly]>=0).mean()) if poly.any() else None,
            COM_claw_polygon_halfspace_margin_mm=quantiles(margin[poly]),
            minimum_active_joint_normalized_limit_margin=quantiles(near[mask]),
            sampled_vertical_reaction_share_by_region=shares,
            central_body_ground_contact_fraction=float((central[samples]>1e-5).mean()) if len(samples) else None,
            central_body_contact_threshold=1e-5)
    rows.append(dict(name=name,seed=c['seed'],standing_fraction=summary['standing_fraction'],locomotion_regime=summary['locomotion_regime'],
        total_mass_model_units=m['total_mass'],gravity=m['gravity'],fused_head_accounted_once=True,
        max_sensor_norm_vs_world_claw_force_error=sensor_error,max_root_force_closure_error=float(t['force_closure_max_error'].max()),
        max_upstream_claw_force_error=float(t['upstream_force_max_error'].max()),groups=summaries))
limits=['Three known failure types selected for diagnosis, not representative validation or checkpoint selection.',
    '5ms samples and geometric COM projection do not certify continuous/static balance; adhesion can support moments outside a claw-contact polygon.',
    'Contact+root-actuator force is a recorded contribution, not a fully audited Newton residual including every possible passive/aerodynamic term.',
    'Configured +/-0.7rad active ranges are pilot assumptions, not biologically validated extension limits.',
    'No threshold/reward/body/model changes, promotion, learned walking or retained-standing success claim.']
audit=dict(status='completed_audited_H2B',code_commit='90e41b03846a493aa1a6559f46b8cb8c8ce9a70d',seconds=r['seconds'],
    cases=rows,all_native_body_pad_spike_replays_exact=True,all_mass_root_upstream_raw_contact_force_checks_passed=True,
    source_prior_checkpoints_data_unchanged=True,all_jobs_exit0_empty=True,lease_released=True,thresholds_changed=False,
    model_promoted=False,walking_proven=False,prior_zero_step_allocation_preserved=True,limits=limits)
ap=root/'outputs/body-telemetry-h2b-20261106-audit.json';ap.write_text(json.dumps(audit,indent=2,allow_nan=False),encoding='utf-8')
compact=dict(status=audit['status'],code_commit=audit['code_commit'],seconds=audit['seconds'],
    all_native_body_pad_spike_replays_exact=True,all_mass_root_upstream_raw_contact_force_checks_passed=True,
    source_prior_checkpoints_data_unchanged=True,all_jobs_exit0_empty=True,lease_released=True,
    thresholds_changed=False,model_promoted=False,walking_proven=False,limits=limits,cases=[])
for case in rows:
    groups={}
    for name,g in case['groups'].items():
        groups[name]=dict(samples=g['samples'],COM_abs_vertical_velocity_median_mm_s=None if g['whole_COM_abs_velocity_z_mm_s'] is None else g['whole_COM_abs_velocity_z_mm_s']['median'],
            contact_Fz_over_weight_median=None if g['ground_vertical_force_over_weight'] is None else g['ground_vertical_force_over_weight']['median'],
            contact_plus_actuator_Fz_over_weight_median=None if g['recorded_contact_plus_actuator_vertical_over_weight'] is None else g['recorded_contact_plus_actuator_vertical_over_weight']['median'],
            polygon_defined_count=g['claw_contact_polygon_defined_count'],COM_projection_inside_polygon_fraction=g['COM_projection_inside_claw_polygon_fraction'],
            central_body_ground_contact_fraction=g['central_body_ground_contact_fraction'],
            sampled_vertical_reaction_share_by_region=g['sampled_vertical_reaction_share_by_region'])
    compact['cases'].append(dict(name=case['name'],standing_fraction=case['standing_fraction'],locomotion_regime=case['locomotion_regime'],groups=groups))
sp=root/'outputs/body-telemetry-h2b-20261106-summary.json';sp.write_text(json.dumps(compact,indent=2,allow_nan=False),encoding='utf-8')
lines=['# H2B: instrumented exact frozen replays','',f"Three full5s jobs in{r['seconds']:.3f}s. All native body/pad/spike replays exact. Whole mass/COM, root-force/upstream/raw-contact checks pass; source/prior/data unchanged, exit0/empty and lease released. A0zero-step failure remains archived.",'',
    '|Case|Group|Samples|Median COM abs vz mm/s|Median contact Fz/weight|Median actuator Fz/weight|Median sum/weight|COM inside claw polygon %|',
    '|---|---|---:|---:|---:|---:|---:|---:|']
def med(q):return 'n/a' if q is None else f"{q['median']:.5f}"
for case in rows:
    for group in ('standing','high_only','upright_failure'):
        g=case['groups'][group];inside=g['COM_projection_inside_claw_polygon_fraction']
        lines.append(f"|{case['name']}|{group}|{g['samples']}|{med(g['whole_COM_abs_velocity_z_mm_s'])}|{med(g['ground_vertical_force_over_weight'])}|{med(g['root_actuator_vertical_force_over_weight'])}|{med(g['recorded_contact_plus_actuator_vertical_over_weight'])}|{'n/a' if inside is None else f'{100*inside:.2f}'}|")
lines+=['','Polygon percentage is conditional on a defined noncollinear loaded-claw contact hull, not a balance gate. Median sum is computed per sample, not by adding separate medians.','','Limits:','']+['- '+x for x in limits]+['']
mp=root/'outputs/body-telemetry-h2b-20261106-audit.md';mp.write_text('\n'.join(lines),encoding='utf-8')
files=[out/'protocol.json',out/'result.json',ap,mp,sp]
for c in p['cases']:
    name=c['name'];files.extend(out/(name+n) for n in ('-receipt.json','-context.json','-h2-context.json','.log'))
    files.extend(out/name/n for n in ('report.json','observations.npz','adhesion-observations.npz','body-telemetry.npz','ground-contacts.npz','telemetry-model.json'))
files.extend(root/n for n in p['prior_hashes'])
entries={f.relative_to(root).as_posix():f.read_bytes() for f in files}
a0=read(root/'outputs/body-telemetry-h2-a0-20261106.json');a0zip=root/'outputs'/a0['artifact']['path'];assert sha(a0zip)==a0['artifact']['sha256']
entries['a0-failure-evidence.zip']=a0zip.read_bytes();entries['a0-failure.json']=(root/'outputs/body-telemetry-h2-a0-20261106.json').read_bytes()
for name in ('h2_telemetry.py','h2_worker.py','run_body_h2.py','analyse_body_h2b.py','test_h2_telemetry.py','height_audit_h1.py','s2_worker.py','s2_contract.py','adhesion_mapping.py','h2b-protocol.md'):
    entries['scripts/'+name]=(root/'work'/name).read_bytes()
entries['runtime-source-manifest.json']=(root/'outputs/thermo-pilot-tp2-20261106-source.json').read_bytes()
assert not any('brain-cache' in n or 'NoesisData' in n or n.endswith('.lock') for n in entries)
manifest={n:dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest()) for n,b in entries.items()}
archive=root/'outputs/body-telemetry-h2b-20261106-evidence.zip';assert not archive.exists()
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for n,b in entries.items():z.writestr(n,b)
    z.writestr('manifest.json',json.dumps(manifest,indent=2))
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None and set(z.namelist())==set(entries)|{'manifest.json'}
    assert all(hashlib.sha256(z.read(n)).hexdigest()==v['sha256'] for n,v in manifest.items())
artifact=dict(path=archive.name,bytes=archive.stat().st_size,sha256=sha(archive),files=len(entries),crc_and_all_hashes_verified=True,
    raw_graph_cache_private_lease_excluded=True,release_uploaded=False)
(root/'outputs/body-telemetry-h2b-20261106-artifact.json').write_text(json.dumps(artifact,indent=2),encoding='utf-8')
print(json.dumps(dict(cases=len(rows),artifact=artifact),indent=2))
