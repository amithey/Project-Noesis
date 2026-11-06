"""Control preserves drive/credit logging and forbids every weight/learning update."""
from types import SimpleNamespace
import numpy as np
import pytest
from p0_control import control_brain,validate_control_native
from test_s2_contract import report


class Base:
    def __init__(self):
        self.net=SimpleNamespace(data=np.array([1.,-2.]))
        self.learning_active=False;self.updates=0
        self._log={'credit':[]}
        def forbidden(*a,**kw):raise AssertionError('Plasticity must not run')
        self.learner=SimpleNamespace(update=forbidden,motor_indices=np.array([0,1]),metrics=lambda:{'unchanged':True})
        self.exploration=SimpleNamespace(credit=lambda ids:np.array([.1,-.2]))


def test_no_updates_or_weight_writes_with_exploration_credit():
    brain=control_brain(Base)()
    assert brain.learn(1,5)=={'unchanged':True} and brain.control_steps==0 and not brain._log['credit']
    brain.learning_active=True
    for _ in range(3):brain.learn(-1,5)
    assert brain.control_steps==3 and brain.updates==0 and len(brain._log['credit'])==3
    np.testing.assert_array_equal(brain.net.data,[1.,-2.])
    np.testing.assert_array_equal(brain._log['credit'][0],np.array([.1,-.2],np.float32))
    with pytest.raises(ValueError):brain.net.data[0]=2


def test_log_disabled_control_still_never_updates():
    brain=control_brain(Base)();brain._log=None;brain.learning_active=True;brain.learn(.5,5)
    assert brain.control_steps==1 and brain.updates==0


def test_only_full_unchanged_exposure_accepted():
    r=report();r.update(learning_updates=0,weights_changed=False);validate_control_native(r)
    for key,value in [('learning_updates',1),('weights_changed',True),('completed_chunks',4999)]:
        bad=dict(r);bad[key]=value
        with pytest.raises(ValueError):validate_control_native(bad)
