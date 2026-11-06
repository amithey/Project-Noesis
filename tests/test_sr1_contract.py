import copy
import pytest
from sr1_contract import replication_summary


def inputs():
    row=dict(standing_fraction=.95,mean_stance_slip_mm_s=2.,heading_forward_distance_mm=.1,
             net_forward_displacement_mm=.1,total_leg_steps=10,tripod_coordination={'tripod_coordination_score':.2})
    parents={s:copy.deepcopy(row) for s in (10006,10007,10010,10011)}
    children={s:copy.deepcopy(parents) for s in (1004,1005,1006)}
    own={s:(copy.deepcopy(row),copy.deepcopy(row)) for s in children}
    return parents,children,own


def test_birth_count_is_not_probe_count_and_no_selection():
    p,c,o=inputs();c[1005][10006]['standing_fraction']=.4
    result=replication_summary(p,c,o)
    assert result['training_birth_count']==3 and len(result['banks'])==3 and result['strict_screen_pass_count']==2
    assert result['banks'][1]['dev_standing_ge90_count']==3 and not result['model_promoted']


def test_missing_birth_or_probe_is_not_complete_replication():
    p,c,o=inputs();c.pop(1006)
    with pytest.raises(ValueError):replication_summary(p,c,o)
    p,c,o=inputs();c[1006].pop(10011)
    with pytest.raises(ValueError):replication_summary(p,c,o)
