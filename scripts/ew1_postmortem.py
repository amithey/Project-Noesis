"""Read-only, explicitly posthoc EW1 linearization/seed-consensus diagnosis."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np


def diagnose(state,parent_returns,candidate_returns):
    plus,minus=state['plus_returns'],state['minus_returns']
    directions,coefficients=state['directions'],state['coefficients']
    parent,candidate=np.asarray(parent_returns,float),np.asarray(candidate_returns,float)
    if plus.shape!=minus.shape or plus.shape!=(2,4) or parent.shape!=candidate.shape or parent.shape!=(2,):
        raise ValueError('Require exact two-context/four-direction EW1 panel')
    if not all(np.isfinite(x).all() for x in (plus,minus,parent,candidate,directions,coefficients)):
        raise ValueError('Nonfinite diagnostic input')
    epsilon=float(state['probe_step']);step=float(state['update_step'])
    if epsilon<=0 or step<=0:raise ValueError('Invalid fixed probe/update scale')
    derivative=(plus-minus)/(2*epsilon)
    np.testing.assert_allclose(coefficients,derivative.mean(0),rtol=1e-12,atol=1e-14)
    norm=float(abs(state['gradient']).max())
    basis=np.zeros(4) if norm<=1e-12 else step*coefficients/(4*norm)
    # This is a prediction from the measured four-direction local linear model,
    # not gradient.dot(delta), which would incorrectly introduce edge-count scale.
    np.testing.assert_allclose(directions.T@basis,state['relative_delta'],rtol=1e-12,atol=1e-14)
    predicted=derivative@basis;observed=candidate-parent
    product=(plus[0]-minus[0])*(plus[1]-minus[1])
    policies=np.stack((minus,plus),axis=-1).reshape(2,8)
    return dict(direction_count=4,training_contexts=2,
        concordant_nonzero_direction_signs=int(np.count_nonzero(product>0)),
        opposing_direction_signs=int(np.count_nonzero(product<0)),zero_direction_products=int(np.count_nonzero(product==0)),
        predicted_return_changes=predicted.tolist(),observed_return_changes=observed.tolist(),
        opposite_predicted_observed_signs=int(np.count_nonzero(predicted*observed<0)),
        measured_probe_policies_better_in_both_contexts=int(np.count_nonzero(np.all(policies>parent[:,None],axis=0))),
        measured_probe_policies=8,no_policy_selected_or_adopted=True,
        diagnosis_is_posthoc=True,original_failed_verdict_preserved=True,
        limits='Descriptive local linear-model check only, not a causal bottleneck proof, variance estimate, optimal step, effective method or independent confirmation')


def main():
    started=time.monotonic();root=Path(__file__).resolve().parents[1]
    read=lambda p:json.loads(p.read_text())
    def sha(path):
        h=hashlib.sha256()
        with path.open('rb') as f:
            while b:=f.read(1024*1024):h.update(b)
        return h.hexdigest()
    out=root/'outputs/episodic-neural-ew1-20261110'
    paths=[out/'result.json',out/'protocol.json',out/'training-returns.json',out/'optimizer-state.npz',
           root/'outputs/episodic-neural-ew1-20261110-summary.json']
    pins={str(path.relative_to(root)).replace('\\','/'):sha(path) for path in paths}
    result=read(paths[0]);summary=read(paths[-1])
    assert result['status']=='completed_EW1' and result['lease_released']
    assert summary['status']=='completed_independent_EW1_audit' and not summary['comparison']['combined_exploratory_pilot_pass']
    assert not (root/'work/Project-Noesis/gpu-pilot/outputs/codex-training-host.lock').exists()
    assert sha(paths[3])==result['learning']['optimizer_state_sha256']
    for n,h in result['created_hashes'].items():assert sha(root/n)==h
    with np.load(paths[3],allow_pickle=False) as z:state={k:z[k].copy() for k in z.files}
    banks=result['science']['banks']
    diagnostic=diagnose(state,*([banks[bank][str(seed)]['return'] for seed in (1008,1009)] for bank in ('parent','candidate')))
    assert all(sha(root/n)==h for n,h in pins.items()) and time.monotonic()-started<180
    private=dict(status='completed_readonly_posthoc_EW1_diagnosis',input_hashes=pins,source_sha256=sha(Path(__file__)),diagnostic=diagnostic,
        next_hypothesis='Fresh preregistered measured-policy acceptance versus an unverified mixture of policy directions; preserve controls and unchanged independent physical retention tests')
    private_path=root/'outputs/episodic-neural-ew1-postmortem-private-20261110.json'
    private_path.write_text(json.dumps(private,indent=2),encoding='utf-8')
    public={k:v for k,v in diagnostic.items() if k not in ('predicted_return_changes','observed_return_changes')}
    public.update(status=private['status'],new_simulation_workers=0,next_hypothesis=private['next_hypothesis'])
    (root/'outputs/episodic-neural-ew1-postmortem-minimal-20261110.json').write_text(json.dumps(public,indent=2),encoding='utf-8')
    print('Read-only posthoc EW1 diagnosis complete; no simulation, policy selection or original-verdict change')


if __name__=='__main__':main()
