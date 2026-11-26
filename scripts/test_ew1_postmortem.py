import numpy as np
import pytest
from ew1_postmortem import diagnose


def linear_fixture():
    h=np.array([[1.,1.,1.,1.],[1.,-1.,1.,-1.],[1.,1.,-1.,-1.],[1.,-1.,-1.,1.]])
    true=np.array([[.1,.2,.3,.4],[.2,.1,.4,.3]])
    epsilon=.02;step=.01;parent=np.array([.1,.2]);derivative=true@h.T
    coefficients=derivative.mean(0);gradient=np.mean(coefficients[:,None]*h,axis=0)
    delta=step*gradient/max(abs(gradient))
    state=dict(plus_returns=parent[:,None]+epsilon*derivative,minus_returns=parent[:,None]-epsilon*derivative,
        directions=h,coefficients=coefficients,gradient=gradient,relative_delta=delta,probe_step=np.array(epsilon),update_step=np.array(step))
    return state,parent,parent+true@delta


def test_linear_fixture_prediction_matches_true_policy_response_without_edge_count_factor():
    state,parent,candidate=linear_fixture();result=diagnose(state,parent,candidate)
    np.testing.assert_allclose(result['predicted_return_changes'],candidate-parent,rtol=0,atol=1e-15)
    assert result['opposite_predicted_observed_signs']==0 and result['no_policy_selected_or_adopted']


def test_an_observed_regression_is_not_hidden_by_mean_improvement():
    state,parent,candidate=linear_fixture();candidate[1]=parent[1]-.01
    result=diagnose(state,parent,candidate)
    assert result['opposite_predicted_observed_signs']==1 and result['original_failed_verdict_preserved']


def test_inconsistent_saved_step_rejected():
    state,parent,candidate=linear_fixture();state['relative_delta']*=2
    with pytest.raises(AssertionError):diagnose(state,parent,candidate)
