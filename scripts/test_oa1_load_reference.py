import numpy as np
import pytest
from oa1_load_reference import load_references,diagnostic_summary


def test_zero_actuator_is_exact_original_load_reference():
    ground=np.array([0.,.5,1.,1.5,2.]);v=load_references(ground,np.zeros(5),1.)
    np.testing.assert_array_equal(v['raw_load_quality'],np.exp(-abs(ground-1)))
    np.testing.assert_array_equal(v['raw_load_quality'],v['actuator_adjusted_load_quality'])


def test_analytical_quasistatic_force_fixture_is_not_physics_validation():
    # Analytic algebra only: ground=weight+pull, actuator=-pull.
    pull=np.array([0.,.1,.3,1.]);v=load_references(1+pull,-pull,1.)
    np.testing.assert_allclose(v['actuator_adjusted_ratio_error'],0,atol=1e-15)
    np.testing.assert_allclose(v['raw_load_quality'],np.exp(-pull),atol=1e-15)
    assert diagnostic_summary(v)['not_a_standing_or_newton_residual']


def test_weight_and_force_unit_scaling_does_not_change_ratios():
    a=load_references(np.array([1.,1.2]),np.array([-.1,-.2]),1.)
    b=load_references(np.array([1000.,1200.]),np.array([-100.,-200.]),1000.)
    for k in a:np.testing.assert_allclose(a[k],b[k],atol=1e-15)


@pytest.mark.parametrize('ground,actuator,weight',[([1.],[0.,0.],1.),([np.nan],[0.],1.),([1.],[0.],0.),([],[],1.)])
def test_invalid_input_rejects(ground,actuator,weight):
    with pytest.raises(ValueError):load_references(ground,actuator,weight)
