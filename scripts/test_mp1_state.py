import numpy as np
import pytest
from noesis.plasticity import MotorPlasticity
import run_measured_mp1 as run
from mp1_select import TRAIN_SEEDS,fixed_directions,STEP
from test_ew1_state import network


@pytest.mark.parametrize('fallback',[False,True])
def test_measured_policy_copy_and_state_reload_preserve_native_compatibility(tmp_path,monkeypatch,fallback):
    monkeypatch.setattr(run,'root',tmp_path);monkeypatch.setattr(run,'out',tmp_path/'out');run.out.mkdir()
    monkeypatch.setattr(run,'inputs',tmp_path/'inputs');run.inputs.mkdir()
    parent=tmp_path/'parent.npz';MotorPlasticity(network(),[2]).save(parent);cp=run.arrays(parent)
    directions=fixed_directions(len(cp['weights']));np.savez_compressed(run.inputs/'directions.npz',directions=directions)
    for d in range(4):
        for sign in (-1,1):np.savez_compressed(run.inputs/f'probe-{d}-{sign}.npz',**run.apply_delta(cp,sign*STEP*directions[d]))
    returns=np.zeros((2,8))
    if not fallback:returns[:,1]=[.11,.12];returns[:,2]=[-.2,-.3]
    records=[dict(seed=s,direction=d,sign=sign,**{'return':float(returns[row,d*2+(sign==1)])},
        complete_horizon=True,actual_import_verified=True,checkpoint_sha256='a'*64,return_source_sha256='b'*64)
        for row,s in enumerate(TRAIN_SEEDS) for d in range(4) for sign in (-1,1)]
    monkeypatch.setattr(run,'measured',lambda directory:(None,None,None,None,np.zeros(1000)))
    p=dict(parent_checkpoint='parent.npz',parent_sha256=run.sha(parent));result=run.learn(p,records)
    assert result['reload_reproduces_weights'] and result['candidate_update_applied']==(not fallback)
    assert result['completed_rounds']==1 and result['training_trial_count']==16
    for bank,index in (('candidate',-1 if fallback else 1),('negative',-1 if fallback else 2)):
        d,k=divmod(index,2);source=parent if index<0 else run.inputs/f'probe-{d}-{-1 if k==0 else 1}.npz'
        path=run.out/(bank+'.npz');assert run.sha(path)==run.sha(source)
        net=network();loader=MotorPlasticity(net,[2]);loader.load(path,weights_only=True)
        np.testing.assert_array_equal(net.data[loader.edge_indices],run.arrays(source)['weights'])
        for key in cp:
            if key!='weights':np.testing.assert_array_equal(run.arrays(path)[key],cp[key])
    assert run.sha(parent)==p['parent_sha256']
