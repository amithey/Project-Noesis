import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from runtime_profile import compare_prefix, validate_profile
from runtime_diagnostic import PARENT


def test_actual_frozen_schema_and_reject_partial():
    r=dict(status='completed_transport_pilot',learning=False,motor_exploration=0,
           weights_unchanged=True,completed_control_chunks=100,simulated_seconds_per_world=.5,
           seeds=[10006],checkpoint_sha256=PARENT,worlds=1,backend='cpu',controller='brain')
    validate_profile(r)
    r['completed_control_chunks']=99
    with pytest.raises(ValueError): validate_profile(r)


def test_prefix_rejects_divergence_partial_and_nan(tmp_path):
    a,b=tmp_path/'a.npz',tmp_path/'b.npz'
    np.savez(a,position=np.zeros((1000,1,3)))
    x=np.zeros((100,1,3)); np.savez(b,position=x)
    assert all(compare_prefix(a,b).values())
    x[-1,0,0]=1; np.savez(b,position=x)
    assert not all(compare_prefix(a,b).values())
    x[-1,0,0]=np.nan; np.savez(b,position=x)
    with pytest.raises(ValueError,match='Nonfinite'): compare_prefix(a,b)
    np.savez(b,position=x[:99])
    with pytest.raises(ValueError,match='duration'): compare_prefix(a,b)
