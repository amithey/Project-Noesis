import numpy as np
from run_weight_sensitivity_ws1 import response


def test_sensitivity_separates_no_change_neural_only_and_body_change(tmp_path):
    reference=tmp_path/'reference';candidate=tmp_path/'candidate'
    reference.mkdir();candidate.mkdir()
    pad={k:np.zeros((1000,6)) for k in ('desired','pads')}
    body={k:np.zeros((1000,1,3)) for k in ('angles','position','up_body','omega')}
    for d in (reference,candidate):
        np.savez(d/'adhesion-observations.npz',**pad);np.savez(d/'observations.npz',**body)
    assert all(x['changed_samples']==0 for x in response(reference,candidate).values())
    pad['desired'][17,2]=.125
    np.savez(candidate/'adhesion-observations.npz',**pad)
    neural=response(reference,candidate)
    assert neural['desired']==dict(changed_samples=1,max_abs_delta=.125,first_changed_sample=17)
    assert neural['position']['changed_samples']==0
    body['position'][21,0,1]=-.25
    np.savez(candidate/'observations.npz',**body)
    assert response(reference,candidate)['position']==dict(changed_samples=1,max_abs_delta=.25,first_changed_sample=21)
