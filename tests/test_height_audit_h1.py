import numpy as np
from height_audit_h1 import intervals,masks,score_row


def fixture():
    return dict(position=np.array([[0.,0.,1.],[0.,0.,1.5]]),up_body=np.tile([0.,0.,1.],(2,1)),
                heading=np.tile([1.,0.,0.],(2,1)),velocity=np.zeros((2,42)),foot_load=np.ones((2,6)),feet_position=np.zeros((2,6,3)))


def test_bout_edges_and_no_dropped_single_sample():
    assert intervals([1,1,0,1])==[(0,2),(3,4)]
    assert intervals([0,0])==[] and intervals([1])==[(0,1)]


def test_original_height_boundary_is_inclusive_and_support_required():
    obs=fixture();standing,high=masks(obs);np.testing.assert_array_equal(standing,[1,1]);assert not high.any()
    obs['position'][1,2]=1.5+1e-6;standing,high=masks(obs)
    assert not standing[1] and high[1]
    obs['foot_load'][1]=0;standing,high=masks(obs);assert not standing[1] and not high[1]


def test_real_native_reward_cliff_preserves_input_and_all_other_state():
    obs=fixture();before=obs['position'].copy()
    below=score_row(obs,1,1.5-1e-6);above=score_row(obs,1,1.5+1e-6)
    assert np.isclose(below,.8) and above==-.8 and np.isclose(below-above,1.6)
    np.testing.assert_array_equal(obs['position'],before)
