import numpy as np
import pytest
from scipy import sparse
from noesis.lif import LIFNetwork
from noesis.plasticity import MotorPlasticity
from ws1_weights import bounded_offset,checkpoint_variant


def test_registered_symmetric_offsets_preserve_sign_zero_and_input():
    original=np.array([2.,-4.,0.]);current=np.array([2.2,-3.6,0.]);saved=current.copy()
    plus,minus=(bounded_offset(current,original,d) for d in (1,-1))
    np.testing.assert_allclose(plus-current,.05*np.abs(original),rtol=0,atol=1e-15)
    np.testing.assert_allclose(current-minus,.05*np.abs(original),rtol=0,atol=1e-15)
    for w in (plus,minus):
        np.testing.assert_array_equal(np.sign(w),np.sign(original));assert w[2]==0
        assert np.all(np.abs(w)>=.5*np.abs(original)) and np.all(np.abs(w)<=1.5*np.abs(original))
    np.testing.assert_array_equal(current,saved)


def test_actual_native_loader_accepts_only_weight_changes_on_existing_edges(tmp_path):
    def net():return LIFNetwork(sparse.csr_matrix(np.array([[0.,3.,4.],[0.,0.,-2.],[1.,0.,0.]])))
    a=MotorPlasticity(net(),[2],learning_rate=.2,tau_eligibility_ms=100,rule='perturbation',normalize_reward_error=True,bias_correct_reward_error=True)
    a.save(tmp_path/'parent.npz')
    with np.load(tmp_path/'parent.npz',allow_pickle=False) as z:parent={k:z[k].copy() for k in z.files}
    for direction in (-1,1):
        child=checkpoint_variant(parent,direction)
        for k in parent:
            if k!='weights':np.testing.assert_array_equal(child[k],parent[k])
        path=tmp_path/f'variant{direction}.npz';np.savez_compressed(path,**child)
        network=net();before=network.data.copy();indices=network.indices.copy();indptr=network.indptr.copy()
        loader=MotorPlasticity(network,[2],learning_rate=.2,tau_eligibility_ms=100,rule='perturbation',normalize_reward_error=True,bias_correct_reward_error=True)
        loader.load(path,weights_only=True)
        expected=before.copy();expected[loader.edge_indices]=child['weights'];np.testing.assert_array_equal(network.data,expected)
        np.testing.assert_array_equal(network.indices,indices);np.testing.assert_array_equal(network.indptr,indptr)


@pytest.mark.parametrize('current,original,direction',[
    ([1.5],[1.],1),([.5],[1.],-1),([-1.],[1.],1),([0.],[1.],1),([np.nan],[1.],1),([1.],[1.],True),([1.],[1.],0),([],[],1)])
def test_invalid_or_out_of_bounds_offsets_stop_without_clipping(current,original,direction):
    with pytest.raises(ValueError):bounded_offset(np.asarray(current,float),np.asarray(original,float),direction)
