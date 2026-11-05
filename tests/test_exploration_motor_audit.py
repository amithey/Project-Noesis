import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from exploration_motor_audit import project,corresponding_correlations


def test_anatomical_opposite_pools_and_zero_channel():
    mapping=np.array([[1.,-1.,0.],[0.,0.,0.]])
    result=project(mapping,np.array([0,1]),np.array([[3.,1.],[-2.,4.]]))
    np.testing.assert_array_equal(result,[[2,0],[-6,0]])
    with pytest.raises(ValueError):project(mapping,np.array([0,0]),np.ones((2,2)))
    with pytest.raises(ValueError):project(mapping,np.array([0]),np.array([[float('nan')]]))


def test_known_lag_positive_negative_and_undefined_channel():
    x=np.random.default_rng(17).normal(size=(500,3));x[:,2]=0
    y=np.zeros_like(x);y[5:,:2]=x[:-5,:2];y[:,1]*=-1
    correlations=corresponding_correlations(x,y)
    assert correlations[5,0]==pytest.approx(1)
    assert correlations[5,1]==pytest.approx(-1)
    assert np.isnan(correlations[:,2]).all()
    with pytest.raises(ValueError):corresponding_correlations(x,y[:-1])
