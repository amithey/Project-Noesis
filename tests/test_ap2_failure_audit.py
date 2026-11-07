import numpy as np
import pytest
from ap2_failure_audit import partition


def test_reason_partition_is_exhaustive_and_does_not_turn_high_foot_support_into_standing():
    obs=dict(position=np.array([[0,0,1.],[0,0,1.],[0,0,1.],[0,0,.2],[0,0,1.6],[0,0,.3],[0,0,1.5]]),
        up_body=np.array([[0,0,.9],[0,0,.6],[0,0,.9],[0,0,.9],[0,0,.9],[0,0,.7],[0,0,.7]]),foot_load=np.ones((7,6)))
    obs['foot_load'][2]=0
    groups=partition(obs)
    assert {k:np.flatnonzero(v).tolist() for k,v in groups.items()}==dict(standing=[0,5,6],upright_failure=[1],no_claw_upright=[2],low_height_upright_claw=[3],high_height_upright_claw=[4])
    assert not groups['standing'][4]


def test_nonfinite_state_and_invalid_force_inputs_are_not_silently_classified():
    obs=dict(position=np.zeros((1,3)),up_body=np.ones((1,3)),foot_load=np.zeros((1,6)))
    obs['position'][0,2]=np.nan
    with pytest.raises(ValueError):partition(obs)
    obs['position'][0,2]=1;obs['foot_load'][0,0]=-1
    with pytest.raises(ValueError):partition(obs)
