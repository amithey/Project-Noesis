import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from noesis.learning_targets import select_premotor_targets


def network():
    rows=[]
    for side in ('left','right'):
        for segment in ('front','middle','hind'):
            rows.append({'Class':'leg_motor_neuron','Sub Class':segment+'_leg_motor_neuron',
                         'Primary Cell Type':'tibia_flexor','Nerve':side+'_leg','Soma side':side})
    for kind in ['single_leg_neuromere']*4+['sequential_leg_neuromeres','bristle_neuron']:
        rows.append({'Class':kind,'Sub Class':'','Primary Cell Type':'','Nerve':'','Soma side':''})
    n=pd.DataFrame(rows)
    W=sparse.lil_matrix((12,12))
    W[6,0]=5; W[7,0]=-5; W[8,3]=6; W[9,1]=4
    W[10,0]=2; W[10,3]=-3; W[11,0]=10000
    return W.tocsr(),n,np.arange(6)


def test_rank_absolute_weights_ties_by_index_and_anatomical_coverage():
    W,n,motors=network()
    out=select_premotor_targets(W,n,motors,local_per_leg=2,sequential_count=1)
    assert out['per_leg']['LF']['selected']==[6,7]
    assert out['per_leg']['RF']['selected']==[8]
    assert out['per_leg']['LM']['selected']==[9]
    assert out['indices']==[6,7,8,9,10]
    assert 11 not in out['indices']  # Large sensory projection cannot qualify as local CNS.


def test_only_mapped_motor_projections_count_and_inputs_unchanged():
    W,n,_=network(); before=(W.indptr.copy(),W.indices.copy(),W.data.copy()); ncopy=n.copy(deep=True)
    out=select_premotor_targets(W,n,np.array([3]),local_per_leg=1,sequential_count=1)
    assert out['local_indices']==[8] and out['sequential_indices']==[10]
    assert out['per_leg']['LF']['candidate_count']==0
    for a,b in zip(before,(W.indptr,W.indices,W.data),strict=True): np.testing.assert_array_equal(a,b)
    pd.testing.assert_frame_equal(n,ncopy)


def test_zero_counts_and_overlap_are_well_defined():
    W,n,m=network(); W=W.tolil(); W[6,3]=8; W=W.tocsr()
    out=select_premotor_targets(W,n,m,local_per_leg=1,sequential_count=0)
    assert out['indices']==[6,9]  # One cell selected for both LF/RF occurs once.
    assert select_premotor_targets(W,n,m,local_per_leg=0,sequential_count=0)['indices']==[]


@pytest.mark.parametrize('value',[-1,1.5,True,np.nan])
def test_bad_counts_rejected(value):
    W,n,m=network()
    with pytest.raises(ValueError): select_premotor_targets(W,n,m,local_per_leg=value)


@pytest.mark.parametrize('mapped',[np.array([-1]),np.array([12]),np.array([1.5]),np.array([[1]])])
def test_bad_motor_indices_rejected(mapped):
    W,n,_=network()
    with pytest.raises(ValueError): select_premotor_targets(W,n,mapped)


def test_misaligned_labels_and_nonfinite_weights_rejected():
    W,n,m=network(); shifted=n.copy(); shifted.index=np.arange(1,13)
    with pytest.raises(ValueError): select_premotor_targets(W,shifted,m)
    W.data[0]=np.nan
    with pytest.raises(ValueError): select_premotor_targets(W,n,m)
