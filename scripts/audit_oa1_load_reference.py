"""Signed completed-telemetry sensitivity audit; no new simulation or objective adoption."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from oa1_load_reference import load_references,diagnostic_summary
from ps1_objective import physical_score


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while block:=f.read(1024*1024):h.update(block)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--index',type=Path,required=True)
    ap.add_argument('--private-output',type=Path,required=True);ap.add_argument('--public-output',type=Path,required=True);a=ap.parse_args()
    root=Path(__file__).resolve().parents[1];started=time.monotonic()
    index=json.loads(a.index.read_text());assert len(index['cases'])==24
    assert not (root/'work/Project-Noesis/gpu-pilot/outputs/codex-training-host.lock').exists()
    def path(name):
        p=(root/name).resolve()
        if not p.is_relative_to(root):raise ValueError('Input outside workspace')
        return p
    pins={**index['source_hashes']};rows=[];seen=set()
    for name,h in pins.items():assert sha(path(name))==h
    for case in index['cases']:
        assert case['id'] not in seen;seen.add(case['id'])
        for name,h in case['hashes'].items():assert sha(path(name))==h;pins[name]=h
        report=json.loads(path(case['files']['report']).read_text())
        assert report['status']=='completed_transport_pilot' and report['learning'] is False and report['weights_unchanged']
        assert report['worlds']==1 and report['completed_control_chunks']==1000 and report['simulated_seconds_per_world']==5
        def arrays(name):
            with np.load(path(case['files'][name]),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
        anatomy=arrays('anatomy');o=arrays('observations');model=json.loads(path(case['files']['model']).read_text())
        np.testing.assert_allclose(anatomy['time'],np.arange(1001)*.005,rtol=0,atol=1e-10)
        assert anatomy['region_vertical_reaction'].shape==(1001,4) and anatomy['root_actuator_force'].shape==(1001,3)
        assert anatomy['ground_force'].shape==(1001,3)
        weight=model['total_mass']*abs(model['gravity'][2]);use=slice(10,None)
        region=anatomy['region_vertical_reaction'][1:];ground=anatomy['ground_force'][1:,2]
        np.testing.assert_allclose(region.sum(1),ground,rtol=1e-12,atol=1e-12)
        actuator=anatomy['root_actuator_force'][1:,2]
        values=load_references(region.sum(1),actuator,weight)
        score=physical_score(region,anatomy['com_velocity'][1:],o['omega'][:,0],o['up_body'][:,0,2],o['foot_load'][:,0],weight)
        np.testing.assert_array_equal(values['raw_load_quality'],score['vertical_load_quality'])
        non_load=score['upright_quality']*score['tarsal_share']*score['claw_support_quality']*score['motion_quality']
        np.testing.assert_allclose(non_load*values['raw_load_quality'],score['quality'],rtol=1e-14,atol=1e-15)
        adjusted_quality=non_load*values['actuator_adjusted_load_quality']
        row=diagnostic_summary({k:v[use] for k,v in values.items()})
        row.update(case_id=case['id'],mean_original_quality=float(score['quality'][use].mean()),
            mean_counterfactual_quality=float(adjusted_quality[use].mean()),
            changed_counterfactual_quality_samples=int(np.count_nonzero(abs(adjusted_quality[use]-score['quality'][use])>1e-12)),
            region_ground_equivalence_checked=True)
        rows.append(row)
        assert time.monotonic()-started<180
    assert all(sha(path(name))==h for name,h in pins.items())
    source_names=['work/oa1_load_reference.py','work/test_oa1_load_reference.py','work/audit_oa1_load_reference.py']
    private=dict(status='completed_readonly_OA1',rows=rows,input_hashes=pins,
        audit_source_hashes={n:sha(path(n)) for n in source_names},index_sha256=sha(a.index),new_simulation_workers=0,
        limits='Posthoc sensitivity to a quasistatic force-reference assumption, not a Newton residual, physical balance validator or explanation of learning failure. No objective/gate/weights/policy ranking changed.')
    a.private_output.write_text(json.dumps(private,indent=2),encoding='utf-8')
    public=dict(status=private['status'],completed_cases=24,
        cases_with_recorded_actuator_contribution=sum(v['p95_absolute_actuator_weight_ratio']>1e-12 for v in rows),
        cases_with_changed_load_component=sum(v['changed_samples']>0 for v in rows),
        cases_with_changed_counterfactual_quality=sum(v['changed_counterfactual_quality_samples']>0 for v in rows),
        region_ground_equivalence_cases=sum(v['region_ground_equivalence_checked'] for v in rows),
        new_simulation_workers=0,original_failed_verdict_preserved=True,counterfactual_reference_adopted=False,
        no_policy_selected=True,model_promoted=False,standing_proven=False,walking_proven=False,
        limits=private['limits'])
    a.public_output.write_text(json.dumps(public,indent=2),encoding='utf-8')
    print('OA1 read-only signed sensitivity check complete; private case metrics/maps not printed')


if __name__=='__main__':main()
