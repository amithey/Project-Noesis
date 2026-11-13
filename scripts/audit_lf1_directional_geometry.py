"""Signed offline LF1 TRAIN-only audit; no simulation or checkpoint writer."""
import argparse
import json
from pathlib import Path
import time
import zipfile
import numpy as np
import scipy
from lf1_directional_geometry import geometry,certificate,STEP,RADIUS,TOL,GAIN
import run_training_context_mp2 as mp2
from mp2_select import panel

ROOT=Path(__file__).resolve().parents[1]
INDEX=ROOT/'outputs/lf1-input-index-private-20261113.json'
OUT=ROOT/'outputs/training-context-mp2-20261113'
SOURCE=('lf1_directional_geometry.py','test_lf1_directional_geometry.py','audit_lf1_directional_geometry.py')


def prepare():
    if INDEX.exists():raise ValueError('Never overwrite LF1 inventory')
    p=mp2.read(OUT/'protocol.json');result=mp2.read(OUT/'result.json')
    if result['status']!='completed_MP2_training_panel' or len(result['jobs'])!=37:raise ValueError('MP2 incomplete')
    if not all(result[k] for k in ('baseline_exact','source_input_data_created_unchanged','process_tree_empty_verified','lease_released','awake_request_released')):
        raise ValueError('MP2 terminal guards failed')
    receipt_path=ROOT/'outputs/mp2-artifact-private-20261113.json';receipt=mp2.read(receipt_path)
    archive=ROOT/'outputs'/receipt['archive']
    if mp2.sha(archive)!=receipt['sha256']:raise ValueError('Sealed MP2 archive changed')
    with zipfile.ZipFile(archive) as z:
        if json.loads(z.read('manifest.json'))!=receipt['manifest']:raise ValueError('Sealed manifest mismatch')
    legacy=Path(p['runtime_root']);pins={}
    for name,entry in receipt['manifest'].items():
        prefix,relative=name.split('/',1)
        base=ROOT if prefix=='new' else legacy if prefix=='legacy' else None
        if base is None:raise ValueError('Invalid sealed prefix')
        path=(base/relative).resolve()
        if not path.is_relative_to(base.resolve()):raise ValueError('Escaped evidence scope')
        pins[str(path)]=entry['sha256']
    pins.update({**p['source_hashes'],**p['old_hashes'],**p['data_hashes'],**result['created_hashes']})
    for path in (OUT/'result.json',OUT/'protocol.json',receipt_path,archive):pins[str(path)]=mp2.sha(path)
    pins.update({str(ROOT/'work'/n):mp2.sha(ROOT/'work'/n) for n in SOURCE})
    mp2.check_pins(pins)
    mp2.write(INDEX,dict(pins=pins,mp2_output=str(OUT),runtime_root=str(legacy),
        offline_seconds=180,new_workers=0,probe_step=STEP,l1_radius=RADIUS,
        numeric_tolerance=TOL,surrogate_gain_threshold=GAIN,scipy_version=scipy.__version__))
    mp2.write(ROOT/'outputs/lf1-public-protocol-20261113.json',dict(study='LF1 TRAIN-only finite-probe geometry',
        training_contexts=6,directions=4,probe_step=STEP,l1_coefficient_radius=RADIUS,offline_seconds=180,
        numeric_tolerance=TOL,surrogate_gain_threshold=GAIN,zero_new_simulation=True,
        checkpoint_or_optimizer_write=False,held_out_scoring=False,independent_primal_dual_check=True,
        limits='Finite-probe surrogate diagnosis only; no calibrated derivative, actual policy, efficacy or physical validation.'))
    print('LF1 signed private inventory prepared; no simulation allocated')


def run():
    started=time.monotonic();index_hash=mp2.sha(INDEX);index=mp2.read(INDEX);legacy=Path(index['runtime_root'])
    if (legacy/'work/Project-Noesis/gpu-pilot/outputs/codex-training-host.lock').exists():raise ValueError('Heavy lease active')
    private=ROOT/'outputs/lf1-directional-geometry-private-20261113.json';public=ROOT/'outputs/lf1-directional-geometry-minimal-20261113.json'
    if private.exists() or public.exists():raise ValueError('Never overwrite LF1 outcome')
    def verify():
        if mp2.sha(INDEX)!=index_hash:raise ValueError('Inventory changed')
        mp2.check_pins(index['pins'])
        if time.monotonic()-started>180:raise TimeoutError('Offline budget exceeded')
    verify();p=mp2.read(OUT/'protocol.json');result=mp2.read(OUT/'result.json');_,old,_=mp2.legacy(legacy)
    records=mp2.reused_records(legacy,old)
    for actual,job in zip(result['jobs'],mp2.jobs()):
        if any(actual[k]!=v for k,v in job.items()):raise ValueError('Wrong sealed job')
        name=job['name'];folder=OUT/name;report=mp2.read(folder/'report.json');cp=mp2.checkpoint(p,job)
        mp2.validate_report(report,job,mp2.sha(cp),p['source_hashes'][str(ROOT/'work/mp2_worker.py')])
        receipt=mp2.read(OUT/(name+'-receipt.json'))
        if receipt!=actual['receipt'] or receipt['exit_code']!=0 or receipt['timed_out'] or not receipt['tree_empty']:
            raise ValueError('Incomplete sealed receipt')
        if job['phase']!='baseline':
            records.append(mp2.measured_record(folder,{k:job[k] for k in ('phase','bank','seed','direction','sign')},old))
    if records!=mp2.read(OUT/'training-returns.json'):raise ValueError('Raw TRAIN returns not reproduced')
    parent,probes=panel(records);state=mp2.arrays(OUT/'optimizer-state.npz')
    np.testing.assert_array_equal(parent,state['parent_returns']);np.testing.assert_array_equal(probes,state['policy_returns'])
    result_geometry=geometry(parent,probes)
    g=result_geometry['finite_direction_matrix'];mid=result_geometry['midpoint_departure'];odd=result_geometry['odd_amplitude']
    # Independent scalar reconstruction of all24 central differences/midpoints.
    for s in range(6):
        for d in range(4):
            minus,plus=float(probes[s,2*d]),float(probes[s,2*d+1])
            np.testing.assert_allclose(g[s,d],(plus-minus)/(2*STEP),rtol=1e-12,atol=1e-14)
            np.testing.assert_allclose(mid[s,d],(plus+minus)/2-float(parent[s]),rtol=1e-12,atol=1e-14)
    c=result_geometry['linear_certificate']
    checked=certificate(g,c['coefficients'],c['context_weights'])
    if checked['classification']!=c['classification']:raise ValueError('Certificate changed')
    verify()
    def serial(v):return v.tolist() if isinstance(v,np.ndarray) else v
    payload=dict(status='completed_readonly_LF1',geometry=result_geometry,pins=index['pins'],index_sha256=index_hash,
        raw_training_cases=54,elapsed_seconds=time.monotonic()-started,scipy_version=scipy.__version__)
    private.write_text(json.dumps(payload,indent=2,default=serial),encoding='utf-8')
    minimal=dict(status=payload['status'],training_contexts=6,directions=4,raw_training_cases=54,
        raw_returns_and_saved_panel_reproduced=True,independent_primal_dual_certificate_passed=True,
        linear_surrogate_classification=c['classification'],mixed_sign_directions=sum(result_geometry['mixed_sign_directions']),
        midpoint_departure_exceeds_odd_amplitude_cells=result_geometry['midpoint_exceeds_odd_cells'],direction_context_cells=24,
        new_simulation_workers=0,held_out_scoring=False,checkpoint_or_optimizer_written=False,
        actual_policy_selected=False,model_promoted=False,standing_proven=False,walking_proven=False,
        original_MP2_parent_fallback_preserved=True,limits=result_geometry['limits'])
    mp2.write(public,minimal);print(json.dumps(minimal))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=('prepare','run'));a=ap.parse_args()
    prepare() if a.mode=='prepare' else run()
