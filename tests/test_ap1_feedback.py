import numpy as np
import pytest
from ap1_feedback import non_tarsal_fraction,corrected_reward,apply_feedback,validate_ap1_training,OBJECTIVE


def test_anatomical_fraction_no_force_and_conservative_regions():
    assert non_tarsal_fraction([0,0,0,0])==0 and non_tarsal_fraction([0,2,3,0])==0
    assert non_tarsal_fraction([2,3,0,1])==.5
    for a in ([1,-1,0,0],[np.nan,0,0,0],[1,2]):
        with pytest.raises(ValueError):non_tarsal_fraction(a)


def test_disabled_exact_identity_and_fixed_monotone_bounded_penalty():
    for r in (-1.,-.8,-0.,0.,.7,1.):
        for q in (0.,.1,.5,1.):
            assert np.float64(corrected_reward(r,q,0)).tobytes()==np.float64(r).tobytes()
            shaped=corrected_reward(r,q,.5)
            assert -1<=shaped<=r and shaped==float(np.clip(r-.5*q,-1,1))
    assert corrected_reward(.8,0,.5)==.8 and corrected_reward(.8,1,.5)==pytest.approx(.3)
    for bad in (-.1,1.1,np.nan):
        with pytest.raises(ValueError):corrected_reward(.5,bad,.5)
    with pytest.raises(ValueError):corrected_reward(.5,.2,.2)


def test_actual_objective_sidecar_identity_cannot_be_silently_base_v7():
    from test_s2_contract import report
    r=report();r.update(base_reward_objective='curriculum-standing-v7',reward_objective=OBJECTIVE,
        ap1_context={'effective_training_penalty':.5},ap1_worker_completed=True)
    validate_ap1_training(r,.5);r['reward_objective']='curriculum-standing-v7'
    with pytest.raises(ValueError):validate_ap1_training(r,.5)


def test_scalar_only_correction_preserves_other_native_terms_and_disabled_object():
    native={'reward':.6,'stance_slip_mm_s':2.,'unchanged_array':np.arange(6)}
    assert apply_feedback(native,.8,0) is native
    shaped=apply_feedback(native,.8,.5)
    assert shaped['reward']==pytest.approx(.2) and native['reward']==.6
    assert set(shaped)==set(native)
    for k in native:
        if k!='reward':assert shaped[k] is native[k]


def test_contact_summary_uses_same_990_measured_chunks_as_native():
    from run_anatomical_ap1 import contact_summary
    a=dict(region_vertical_reaction=np.ones((1001,4)),non_tarsal_fraction=np.zeros(1001),
        contact_mode_code=np.full(1001,3),com_velocity=np.zeros((1001,3)))
    a['contact_mode_code'][:11]=1
    result=contact_summary(a)
    assert result['samples']==990 and result['mode_fractions']['tarsal_only']==1


def test_real_native_observer_zero_time_exact_inputs_and_no_state_mutation():
    from noesis_gpu.backend import FlyGymBatch
    from ap1_body import observing_body
    a=FlyGymBatch(1,'cpu');expected=a.observe()
    instances=[];b=observing_body(FlyGymBatch,instances)(1,'cpu');actual=b.observe()
    assert set(actual)==set(expected)
    for k in actual:np.testing.assert_array_equal(actual[k],expected[k])
    assert len(instances)==1 and len(b.ap1_records)==1 and b.ap1_records[0]['time']==0
    assert b.ap1_metadata['unmapped_anatomical_count']==1 and b.ap1_metadata['production_static_fusion']
