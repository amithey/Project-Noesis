import copy
import numpy as np
import pytest
from ew1_contract import validate_report,episode_rewards,return_record,comparison
from ew1_update import TRAIN_SEEDS
from ew1_worker import DEV
from support_descriptor_c1 import REGIONS


def report():
    context=dict(phase='trial',bank='probe',seed=1008,direction=0,sign=1,
        worker_completed=True,frozen_native_neural_motor_execution=True,online_learning=False,
        checkpoint_sha256='a'*64,import_records=[dict(actual_import_exact=True)])
    return dict(status='completed_transport_pilot',controller='brain',worlds=1,learning=False,
        motor_exploration=0,weights_unchanged=True,requested_simulated_seconds_per_world=5,
        simulated_seconds_per_world=5,completed_control_chunks=1000,control_ms=5.,physics_dt=.0001,
        ew1_worker_completed=True,seeds=[1008],checkpoint_sha256='a'*64,ew1_context=context)


def features():
    o=dict(omega=np.zeros((1000,1,3)),up_body=np.tile([0.,0.,1.],(1000,1,1)),
           foot_load=np.tile([1/3,1/3,1/3,0.,0.,0.],(1000,1,1)))
    a=dict(time=np.arange(1001)*.005,region_vertical_reaction=np.tile([0.,1.,0.,0.],(1001,1)),
           com_velocity=np.zeros((1001,3)))
    m=dict(region_columns=list(REGIONS),total_mass=1/9810,gravity=[0.,0.,-9810.])
    return o,a,m


def test_physical_return_alignment_and_warmup_are_exact():
    # Analytical feature fixture only, not a standing physics experiment.
    o,a,m=features();np.testing.assert_array_equal(episode_rewards(o,a,m),np.ones(1000))
    a['region_vertical_reaction'][1:11]=[1.,0.,0.,0.]
    r=return_record(report(),o,a,m,{k:'b'*64 for k in ('report','observations','anatomy','model')})
    assert r['return']==1 and r['measured_samples']==990 and r['warmup_samples']==10
    a['region_vertical_reaction'][11]=[1.,0.,0.,0.]
    r=return_record(report(),o,a,m,{k:'b'*64 for k in ('report','observations','anatomy','model')})
    assert r['return']==pytest.approx(988/990)


@pytest.mark.parametrize('fault',['partial','timeout_status','weights_changed','online_learning','wrong_seed','wrong_import','wrong_checkpoint','reserved_seed'])
def test_incomplete_or_mismatched_reports_cannot_supply_a_training_return(fault):
    r=report()
    if fault=='partial':r['completed_control_chunks']=999
    elif fault=='timeout_status':r['status']='wall_time_limit'
    elif fault=='weights_changed':r['weights_unchanged']=False
    elif fault=='online_learning':r['learning']=True
    elif fault=='wrong_seed':r['seeds']=[1009]
    elif fault=='wrong_import':r['ew1_context']['import_records'][0]['actual_import_exact']=False
    elif fault=='wrong_checkpoint':r['checkpoint_sha256']='c'*64
    else:r['seeds']=[10];r['ew1_context']['seed']=10
    with pytest.raises(ValueError):validate_report(r)


def test_frozen_development_evaluation_is_never_learning_data():
    r=report();r['ew1_context'].update(phase='evaluation',bank='candidate',seed=10006,direction=None,sign=None)
    r['seeds']=[10006];validate_report(r)
    with pytest.raises(ValueError):return_record(r,*features(),{k:'b'*64 for k in ('report','observations','anatomy','model')})


def banks():
    result={}
    for bank in ('parent','negative','candidate'):
        improved=bank=='candidate'
        summary=dict(standing_fraction=.96 if improved else .95,mean_stance_slip_mm_s=.1 if improved else .2,
            heading_forward_distance_mm=2. if improved else 1.,net_forward_displacement_mm=2. if improved else 1.,
            total_leg_steps=4 if improved else 2,tripod_coordination=dict(tripod_coordination_score=.7 if improved else .5))
        physical=dict(mean_PS1_quality=.6 if improved else .5,mean_COM_speed_mm_s=.1 if improved else .2,
            mean_angular_speed_rad_s=.1 if improved else .2,mean_non_tarsal_fraction=.1 if improved else .2)
        result[bank]={seed:dict(summary=copy.deepcopy(summary),physical=copy.deepcopy(physical),
                                anatomical=dict(mode_fractions=dict(tarsal_only=.7 if improved else .5),mean_non_tarsal_fraction=.1 if improved else .2),
                                **{'return':.2 if improved else 0.}) for seed in (*TRAIN_SEEDS,*DEV)}
    return result


@pytest.mark.parametrize('fault',['standing','slip','physical','anatomical','training','missing'])
def test_one_failed_cell_is_not_rescued_by_averaging_or_own_seed_selection(fault):
    b=banks();assert comparison(b)['combined_exploratory_pilot_pass']
    if fault=='standing':b['candidate'][DEV[0]]['summary']['standing_fraction']=.89
    elif fault=='slip':b['candidate'][DEV[0]]['summary']['mean_stance_slip_mm_s']=.3
    elif fault=='physical':b['candidate'][DEV[0]]['physical']['mean_non_tarsal_fraction']=.3
    elif fault=='anatomical':b['candidate'][DEV[0]]['anatomical']['mode_fractions']['tarsal_only']=.4
    elif fault=='training':b['candidate'][TRAIN_SEEDS[0]]['return']=-.1
    else:
        b['negative'].pop(DEV[0])
        with pytest.raises(ValueError):comparison(b)
        return
    assert not comparison(b)['combined_exploratory_pilot_pass']
