import copy
import pytest
from mp1_contract import comparison,validate_report,return_record
from mp1_select import TRAIN_SEEDS,DEV,FRESH
from test_ew1_contract import features,report as old_report


def report():
    r=old_report();c=r.pop('ew1_context');r.pop('ew1_worker_completed')
    c['seed']=1010;r.update(mp1_context=c,mp1_worker_completed=True,seeds=[1010])
    return r


def banks():
    result={}
    for bank in ('parent','candidate','negative','gradient'):
        improved=bank=='candidate';cells={}
        for seed in (*TRAIN_SEEDS,*DEV,*FRESH):
            cells[seed]=dict(summary=dict(seed=seed,standing_fraction=.96 if improved else .95,
                mean_stance_slip_mm_s=.1 if improved else .2,heading_forward_distance_mm=2. if improved else 1.,
                net_forward_displacement_mm=2. if improved else 1.,total_leg_steps=4 if improved else 2,
                tripod_coordination=dict(tripod_coordination_score=.7 if improved else .5)),
                physical=dict(mean_PS1_quality=.6 if improved else .5,mean_COM_speed_mm_s=.1 if improved else .2,
                    mean_angular_speed_rad_s=.1 if improved else .2,mean_non_tarsal_fraction=.1 if improved else .2),
                anatomical=dict(mode_fractions=dict(tarsal_only=.7 if improved else .5),mean_non_tarsal_fraction=.1 if improved else .2))
        result[bank]=cells
    return result


@pytest.mark.parametrize('fault',['fresh_regression','old_anatomy','gradient_retention','mislabeled','not_accepted'])
def test_no_frozen_failure_or_parent_fallback_is_rescued_by_other_cells(fault):
    b=banks();assert comparison(b,True)['combined_exploratory_pilot_pass']
    if fault=='fresh_regression':b['candidate'][FRESH[0]]['physical']['mean_non_tarsal_fraction']=.3
    elif fault=='old_anatomy':b['candidate'][DEV[0]]['anatomical']['mode_fractions']['tarsal_only']=.4
    elif fault=='gradient_retention':b['gradient'][DEV[0]]['summary']['mean_stance_slip_mm_s']=.01
    elif fault=='mislabeled':
        b['candidate'][FRESH[0]]['summary']['seed']=1010
        with pytest.raises(ValueError):comparison(b,True)
        return
    assert not comparison(b,fault!='not_accepted')['combined_exploratory_pilot_pass']


def test_fresh_no_regression_handles_perfect_support_without_impossible_strict_gain():
    b=banks()
    for bank in b:
        for seed in FRESH:
            b[bank][seed]['physical']['mean_non_tarsal_fraction']=0.
            b[bank][seed]['anatomical']=dict(mode_fractions=dict(tarsal_only=1.),mean_non_tarsal_fraction=0.)
    result=comparison(b,True)
    assert result['fresh_all_no_regression_pass'] and result['existing_all_physical_directions_pass']


@pytest.mark.parametrize('phase,seed',[('evaluation',11008),('evaluation',10006),('evaluation',1010)])
def test_only_actual_trial_phase_can_supply_training_data(phase,seed):
    r=report();r['mp1_context'].update(phase=phase,bank='candidate',seed=seed,direction=None,sign=None);r['seeds']=[seed]
    validate_report(r)
    with pytest.raises(ValueError):return_record(r,*features(),{k:'b'*64 for k in ('report','observations','anatomy','model')})


def test_complete_trial_has_exact_warmup_and_partial_trial_rejected():
    r=report();record=return_record(r,*features(),{k:'b'*64 for k in ('report','observations','anatomy','model')})
    assert record['seed']==1010 and record['measured_samples']==990 and record['return']==1.
    r['completed_control_chunks']=999
    with pytest.raises(ValueError):validate_report(r)
