import numpy as np
from mp1_fallback_replay import array_match


def test_identical_and_mismatched_arrays_are_distinguished():
    reference=dict(angles=np.array([[1.,2.],[3.,4.]]),position=np.zeros((2,3)))
    assert array_match(reference,{k:v.copy() for k,v in reference.items()})['exact']
    candidate={k:v.copy() for k,v in reference.items()};candidate['position'][1,1]=1.
    assert array_match(reference,candidate)==dict(exact=False,different_arrays=['position'])
    assert not array_match(reference,dict(angles=reference['angles']))['exact']
