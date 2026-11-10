import numpy as np
import pytest
from scipy import sparse
from noesis.lif import LIFNetwork
from noesis.plasticity import MotorPlasticity
import run_episodic_ew1 as run
from ew1_update import TRAIN_SEEDS,DIRECTIONS,PROBE_STEP,fixed_directions


def network():
    return LIFNetwork(sparse.csr_matrix(np.array([[0.,3.,4.],[0.,0.,-2.],[1.,0.,0.]])))


def setup(tmp_path,monkeypatch):
    monkeypatch.setattr(run,'root',tmp_path)
    monkeypatch.setattr(run,'out',tmp_path/'output');run.out.mkdir()
    monkeypatch.setattr(run,'inputs',tmp_path/'inputs');run.inputs.mkdir()
    cp=tmp_path/'parent.npz'
    MotorPlasticity(network(),[2]).save(cp)
    parent=run.arrays(cp);directions=fixed_directions(len(parent['weights']))
    np.savez_compressed(run.inputs/'directions.npz',directions=directions)
    records=[dict(seed=s,direction=d,sign=sign,**{'return':float(sign*PROBE_STEP*directions[d,0]*.2)},
        complete_horizon=True,actual_import_verified=True,checkpoint_sha256='a'*64,return_source_sha256='b'*64)
        for s in TRAIN_SEEDS for d in range(DIRECTIONS) for sign in (-1,1)]
    return dict(parent_checkpoint='parent.npz',parent_sha256=run.sha(cp)),parent,records


def test_saved_optimizer_reproduces_native_compatible_checkpoints(tmp_path,monkeypatch):
    p,parent,records=setup(tmp_path,monkeypatch)
    result=run.learn(p,records)
    assert result['completed_updates']==1 and result['training_trial_count']==16 and result['reload_reproduces_weights']
    state=run.arrays(run.out/'optimizer-state.npz')
    assert float(abs(state['relative_delta']).max())==pytest.approx(.01)
    for bank,sign in (('candidate',1),('negative',-1)):
        path=run.out/(bank+'.npz');child=run.arrays(path)
        expected=parent['weights']+sign*state['relative_delta']*abs(parent['initial_weights'])
        np.testing.assert_array_equal(child['weights'],expected)
        for k in parent:
            if k!='weights':np.testing.assert_array_equal(child[k],parent[k])
        net=network();loader=MotorPlasticity(net,[2]);loader.load(path,weights_only=True)
        np.testing.assert_array_equal(net.data[loader.edge_indices],expected)
        sidecar=run.read(path.with_suffix('.model.json'))
        assert sidecar['negative_control']==(bank=='negative') and sidecar['native_online_learning'] is False
    assert run.sha(tmp_path/'parent.npz')==p['parent_sha256']


def test_partial_training_panel_never_creates_learned_checkpoint(tmp_path,monkeypatch):
    p,parent,records=setup(tmp_path,monkeypatch)
    with pytest.raises(ValueError):run.learn(p,records[:-1])
    assert list(run.out.iterdir())==[]


def test_existing_file_is_checked_and_never_silently_overwritten(tmp_path):
    path=tmp_path/'immutable.npz';np.savez_compressed(path,x=np.array([1.]))
    before=run.sha(path)
    with pytest.raises(AssertionError):run.save_new_checkpoint(path,dict(x=np.array([2.])))
    assert run.sha(path)==before
