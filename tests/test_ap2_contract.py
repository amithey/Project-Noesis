import copy
import pytest
from ap2_contract import DEV,BIRTHS,anatomical_rule,replication_summary


def entry(seed,standing=.95,tarsal=.7,q=.1):
    return dict(summary=dict(seed=seed,standing_fraction=standing,mean_stance_slip_mm_s=1.,heading_forward_distance_mm=1.,
        net_forward_displacement_mm=1.,total_leg_steps=10.,tripod_coordination={'tripod_coordination_score':.5}),
        anatomical=dict(mode_fractions={'tarsal_only':tarsal},mean_non_tarsal_fraction=q))


def banks():
    return {birth:{name:{seed:entry(seed,tarsal=.8 if name=='candidate' else .7,q=.05 if name=='candidate' else .1)
        for seed in (birth,*DEV)} for name in ('parent','control','candidate')} for birth in BIRTHS}


def test_original_failure_cannot_be_dropped_or_overridden_by_other_successes():
    b=banks();b[1005]['candidate'][10006]['summary']['standing_fraction']=.8
    result=replication_summary(b)
    assert result['combined_pilot_pass_count']==2 and not result['all_births_meet_original_rule']
    assert not result['model_promoted'] and result['no_best_checkpoint_selected']
    del b[1005]
    with pytest.raises(ValueError):replication_summary(b)


def test_mean_improvement_cannot_hide_one_anatomical_seed_or_reference_regression():
    p={s:entry(s)['anatomical'] for s in DEV};c=copy.deepcopy(p);a=copy.deepcopy(p)
    for s in DEV:a[s]['mode_fractions']['tarsal_only']=.9;a[s]['mean_non_tarsal_fraction']=.01
    a[10010]['mode_fractions']['tarsal_only']=.69
    assert not anatomical_rule(p,c,a)['passed']
    a[10010]['mode_fractions']['tarsal_only']=.9;c[10007]['mean_non_tarsal_fraction']=.001
    assert not anatomical_rule(p,c,a)['passed']


def test_missing_mislabeled_and_nonfinite_bank_data_fail_before_summary():
    b=banks();b[1004]['parent'][10006]['summary']['seed']=10007
    with pytest.raises(ValueError):replication_summary(b)
    b=banks();del b[1006]['control'][10011]
    with pytest.raises(ValueError):replication_summary(b)
    b=banks();b[1006]['candidate'][10006]['anatomical']['mean_non_tarsal_fraction']=float('nan')
    with pytest.raises(ValueError):replication_summary(b)


def test_equality_is_not_anatomical_improvement_and_own_seed_is_not_a_gate():
    p={s:entry(s)['anatomical'] for s in DEV}
    assert not anatomical_rule(p,p,p)['passed']
    b=banks()
    for birth in BIRTHS:b[birth]['candidate'][birth]['summary']['standing_fraction']=.1
    result=replication_summary(b)
    assert result['combined_pilot_pass_count']==3 and not result['walking_proven']


@pytest.mark.parametrize('phase,bank,strength,seed',[
    ('frozen','candidate','0','10'),('train','candidate','.5','1005'),
    ('train','control','.5','1004'),('frozen','candidate','.5','10006')])
def test_worker_rejects_reserved_or_undeclared_training_arms_before_loading(monkeypatch,phase,bank,strength,seed):
    import sys
    import ap2_worker
    monkeypatch.setattr(sys,'argv',['ap2_worker.py','--phase',phase,'--bank',bank,'--strength',strength,'--seed',seed,
        '--training-seed','1004','--root','.','--output','not-created','--checkpoint','not-read.npz'])
    with pytest.raises(SystemExit) as error:ap2_worker.main()
    assert error.value.code==2
