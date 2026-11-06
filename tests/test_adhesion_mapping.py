"""Pad-direction boundary/locality and strict physical screen checks."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from adhesion_mapping import negative_femur_adhesion,screen


def test_release_follows_negative_femur_only_and_has_no_leg_order_or_state():
    rest=np.zeros(42)
    for leg in range(6):
        desired=rest.copy();desired[leg*7+3]=-.1
        expected=np.ones(6);expected[leg]=0
        np.testing.assert_array_equal(negative_femur_adhesion(desired,rest),expected)
        desired[leg*7+3]=.1
        np.testing.assert_array_equal(negative_femur_adhesion(desired,rest),np.ones(6))
        desired[leg*7+3]=-.08
        np.testing.assert_array_equal(negative_femur_adhesion(desired,rest),np.ones(6))
    np.testing.assert_array_equal(rest,np.zeros(42))


def test_other_joint_commands_do_not_prescribe_a_pad_sequence():
    desired=np.full((6,7),.5);desired[:,3]=0
    np.testing.assert_array_equal(negative_femur_adhesion(desired.ravel(),np.zeros(42)),np.ones(6))


@pytest.mark.parametrize('values',[np.zeros(41),np.full(42,np.nan)])
def test_invalid_angles_fail(values):
    with pytest.raises(ValueError):negative_femur_adhesion(values,np.zeros(42))


def row(standing=.95,slip=2.,heading=0.,net=0.,cycles=10,tripod=.1):
    return dict(standing_fraction=standing,mean_stance_slip_mm_s=slip,heading_forward_distance_mm=heading,
                net_forward_displacement_mm=net,total_leg_steps=cycles,
                tripod_coordination=dict(tripod_coordination_score=tripod))


def test_progress_does_not_override_retention_or_nonfinite_evidence():
    before={s:row() for s in (10006,10007,10010,10011)}
    after={s:row(heading=.5,net=.5) for s in before}
    assert screen(before,after,list(before))['passed']
    after[10011]=row(standing=.89,heading=5.,net=5.)
    assert not screen(before,after,list(before))['passed']
    after[10011]=row(heading=float('nan'))
    with pytest.raises(ValueError):screen(before,after,list(before))
