import ast,copy
from pathlib import Path
import pytest
from ap2_contract import DEV
from ed2_contract import configuration,replication_summary,BIRTHS


def fixture():
    def row(seed,candidate):
        return dict(summary=dict(seed=seed,standing_fraction=.95,mean_stance_slip_mm_s=.1,heading_forward_distance_mm=1.,net_forward_displacement_mm=1.,total_leg_steps=10,
            tripod_coordination=dict(tripod_coordination_score=.5)),
            anatomical=dict(mode_fractions=dict(tarsal_only=.6 if candidate else .5),mean_non_tarsal_fraction=.2 if candidate else .3),
            physical=dict(mean_PS1_quality=.05 if candidate else .02,mean_COM_speed_mm_s=1. if candidate else 2.,mean_angular_speed_rad_s=1. if candidate else 2.,mean_non_tarsal_fraction=.2 if candidate else .3))
    return {birth:{bank:{s:row(s,bank=='candidate') for s in ((birth,*DEV) if bank in ('parent','control','candidate') else DEV)}
        for bank in ('parent','control','candidate','v7','AP1','previous_PS1')} for birth in BIRTHS}


def test_original_failure_mandatory_and_mean_gain_cannot_rescue_single_seed_failure():
    banks=fixture();assert replication_summary(banks)['combined_pilot_pass_count']==3
    banks[1005]['candidate'][10010]['physical']['mean_PS1_quality']=.015
    r=replication_summary(banks);assert r['combined_pilot_pass_count']==2 and r['descriptive_mean_direction_birth_count']==3
    assert not r['births'][0]['combined_exploratory_pilot_pass']
    del banks[1005]
    with pytest.raises(ValueError):replication_summary(banks)


def test_reject_mislabeled_or_missing_bank_evidence():
    banks=fixture();banks[1006]['candidate'][10006]['summary']['seed']=10007
    with pytest.raises(ValueError):replication_summary(banks)
    banks=fixture();del banks[1007]['control'][10011]
    with pytest.raises(ValueError):replication_summary(banks)


def test_registered_replay_parent_and_replication_seeds_only():
    assert configuration('train','replay',1005,1005)==10
    assert configuration('frozen','parent',1007,1007)==0
    for birth in (1006,1007):
        assert configuration('train','control',birth,birth)==0
        assert configuration('train','candidate',birth,birth)==10
        for seed in (birth,*DEV):assert configuration('frozen','candidate',birth,seed)==10
    for phase,bank,birth,seed in [('train','control',1005,1005),('train','replay',1006,1006),('frozen','parent',1006,1006),('train','candidate',1006,10),('frozen','candidate',1007,201),('train','control',True,1006)]:
        with pytest.raises(ValueError):configuration(phase,bank,birth,seed)


def test_reward_motor_and_native_run_configuration_ast_unchanged():
    here=Path(__file__).parent
    old,new=(ast.parse((here/name).read_text()) for name in ('ed1_worker.py','ed2_worker.py'))
    for name in ('factory','observed_pad'):
        get=lambda tree:next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
        assert ast.dump(get(old),include_attributes=False)==ast.dump(get(new),include_attributes=False)
    for name in ('common','command'):
        get=lambda tree:next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))
        assert ast.dump(get(old),include_attributes=False)==ast.dump(get(new),include_attributes=False)
