import numpy as np
import pytest
from scipy import sparse
from noesis.lif import LIFNetwork
from noesis.plasticity import MotorPlasticity
from ws1_import import checked_import
from ws1_weights import checkpoint_variant


def network():
    return LIFNetwork(sparse.csr_matrix(np.array([[0.,3.,4.],[0.,0.,-2.],[1.,0.,0.]])))


def checkpoint(tmp_path):
    learner=MotorPlasticity(network(),[2])
    path=tmp_path/'parent.npz';learner.save(path)
    with np.load(path,allow_pickle=False) as cp:
        child=checkpoint_variant(cp,1)
    np.savez_compressed(path,**child)
    return path


def native(net,path):
    MotorPlasticity(net,[2]).load(path,weights_only=True)


def test_actual_import_and_full_graph_are_verified(tmp_path):
    result=checked_import(native,network(),checkpoint(tmp_path))
    assert result['actual_import_exact'] and result['existing_selected_edges']==2
    assert result['changed_selected_edges']==2 and result['topology_unchanged']


@pytest.mark.parametrize('fault',['skipped','nonselected','topology','wrong_reference'])
def test_import_verifier_detects_faults(tmp_path,fault):
    path=checkpoint(tmp_path);net=network()
    if fault=='wrong_reference':net.data[1]*=1.1
    def broken(n,p):
        if fault=='skipped':return
        native(n,p)
        if fault=='nonselected':n.data[0]*=1.01
        if fault=='topology':n.indices[0]=0
    with pytest.raises(AssertionError):checked_import(broken,net,path)
