"""Offline sensory transfer-function audit; does not simulate, learn, or modify any input."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np

started = time.monotonic()
root = Path(__file__).resolve().parents[1]
source = root / 'work/evidence/s1-restored/gpu-pilot/outputs/standing-profile-s1-20261101'
cases = []
def quantiles(a):
    return dict(zip(['min','p05','median','p95','max'], np.quantile(a,[0,.05,.5,.95,1]).tolist())) if a.size else None
for directory in sorted(source.glob('frozen-*')):
    if not directory.is_dir():
        continue
    p=directory/'observations.npz'
    with np.load(p,allow_pickle=False) as archive:
        load=archive['load'][:,0].copy()
        foot=archive['foot_load'][:,0].copy()
        up=archive['up_body'][:,0,2].copy()
        height=archive['position'][:,0,2].copy()
    assert load.shape == foot.shape == (1000,6)
    assert np.isfinite(load).all() and np.isfinite(foot).all() and (load>=0).all()
    load,foot,up,height=load[10:],foot[10:],up[10:],height[10:]
    contact=foot>1e-5
    upright=(up>=.7)&(height>=.3)&(height<=1.5)
    supported_upright=contact & upright[:,None]
    transfer={}
    for scale in (2.,45.):
        signal=np.tanh(load/scale)
        rate=2.+148.*signal
        normalized_slope=1.-signal**2
        transfer[str(scale)]={
            'all_leg_frames_saturated_fraction':float(np.mean(signal>=.99)),
            'foot_contact_saturated_fraction':float(np.mean(signal[contact]>=.99)) if contact.any() else None,
            'upright_foot_contact_saturated_fraction':float(np.mean(signal[supported_upright]>=.99)) if supported_upright.any() else None,
            'rate_hz_quantiles_on_contact':quantiles(rate[contact]),
            'relative_slope_quantiles_on_contact':quantiles(normalized_slope[contact]),
            'per_leg_contact_saturation_fraction':{leg:float(np.mean(signal[contact[:,i],i]>=.99)) if contact[:,i].any() else None for i,leg in enumerate(['LF','LM','LH','RF','RM','RH'])},
        }
    cases.append({'case':directory.name,'observation_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
                  'measured_control_frames':990,'foot_contact_leg_frames':int(contact.sum()),
                  'whole_leg_load_quantiles_all':quantiles(load),'whole_leg_load_quantiles_on_foot_contact':quantiles(load[contact]),
                  'foot_load_quantiles_on_contact':quantiles(foot[contact]),
                  'nonfoot_load_positive_when_foot_airborne_fraction':float(np.mean((load>1e-5)[~contact])),
                  'transfer_functions':transfer})
assert len(cases)==12
result={'status':'completed_offline_load_range_audit','cases':cases,
        'actual_formula':'rate=2+148*tanh(whole_leg_ground_contact_force_norm/2)',
        'saturation_definition':'tanh(load/load0)>=0.99; stated before analysis',
        'force_units':'native model force units; no biological-unit calibration is established here',
        'counterfactual_scale_45':'transfer function only; no neural or physical rerun, no improvement claim',
        'runtime_changes':False,'learning':False,'walking_proven':False,
        'seconds':time.monotonic()-started}
assert result['seconds']<60
(root/'outputs/load-sensory-range-audit-20261104.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps([{'case':r['case'],'contact_load_median':r['whole_leg_load_quantiles_on_foot_contact']['median'],
                   'actual_contact_saturation':r['transfer_functions']['2.0']['foot_contact_saturated_fraction'],
                   'counterfactual_45_contact_saturation':r['transfer_functions']['45.0']['foot_contact_saturated_fraction']} for r in cases],indent=2))

