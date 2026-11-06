import numpy as np
import pytest
from support_descriptor_c0 import describe_support,static_fixture,anatomical_region


@pytest.mark.parametrize('kind,expected',[
    ('airborne','unsupported'),('foot_only','tarsal_only'),('body_only','central_body_contact'),
    ('mixed_body_feet','central_body_contact'),('proximal_leg','proximal_leg_contact'),('other_tarsal','tarsal_only')])
def test_known_physical_contact_geometry_and_no_behavior_claim(kind,expected):
    r=static_fixture(kind)
    assert r['descriptor']['contact_mode']==expected and r['model_time']==0 and r['physics_integration_steps']==0
    assert not r['descriptor']['physical_standing_validated'] and not r['descriptor']['dynamic_balance_validated']
    assert r['root_force_closure_max_error']<1e-9


def test_force_partition_is_conservative_and_does_not_hide_body_support():
    forces=np.array([[0.,0.,2.],[0.,0.,3.]])
    r=describe_support(forces,['nmf/c_abdomen6','nmf/lf_tarsus5'])
    assert r['contact_mode']=='central_body_contact' and r['reaction_share_by_region']['central_body']==.4
    assert sum(r['vertical_reaction_by_region'].values())==r['total_vertical_reaction']==5
    scaled=describe_support(10*forces,['nmf/c_abdomen6','nmf/lf_tarsus5'])
    assert scaled['reaction_share_by_region']==r['reaction_share_by_region']


def test_unknown_loaded_anatomy_nonfinite_and_wrong_force_convention_fail():
    with pytest.raises(ValueError):describe_support([[0,0,1]],['unknown'])
    with pytest.raises(ValueError):describe_support([[0,0,np.nan]],['nmf/lf_tarsus5'])
    with pytest.raises(ValueError):describe_support([[0,0,-1]],['nmf/lf_tarsus5'])
    assert anatomical_region('nmf/lm_tarsus3')=='other_tarsal'
