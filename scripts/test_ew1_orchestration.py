"""Synthetic workflow controls; these never instantiate a fly or prove ability."""
import shutil
import numpy as np
import pytest
from noesis.plasticity import MotorPlasticity
import run_episodic_ew1 as run
from ew1_contract import validate_report,episode_rewards
from ew1_worker import parser
from ew1_update import TRAIN_SEEDS,DIRECTIONS,PROBE_STEP,fixed_directions,apply_delta
from test_ew1_state import network
from support_descriptor_c1 import REGIONS


@pytest.mark.parametrize('bad_replay',[False,True])
def test_allocation_has_exact_order_or_stops_without_retry_and_releases_lease(tmp_path,monkeypatch,bad_replay):
    monkeypatch.setattr(run,'root',tmp_path)
    monkeypatch.setattr(run,'out',tmp_path/'output')
    monkeypatch.setattr(run,'inputs',tmp_path/'inputs');run.inputs.mkdir()
    (tmp_path/'outputs').mkdir()
    run.write(tmp_path/'outputs/noesis-required-data-sha256.json',dict(files=[]))
    lease=tmp_path/'work/Project-Noesis/gpu-pilot/outputs/codex-training-host.lock';lease.parent.mkdir(parents=True)
    parent=tmp_path/'parent.npz';MotorPlasticity(network(),[2]).save(parent)
    cp=run.arrays(parent);directions=fixed_directions(len(cp['weights']))
    np.savez_compressed(run.inputs/'directions.npz',directions=directions)
    for d in range(DIRECTIONS):
        for sign in (-1,1):np.savez_compressed(run.inputs/f'probe-{d}-{sign}.npz',**apply_delta(cp,sign*PROBE_STEP*directions[d]))
    plan=dict(source_hashes={},prior_hashes={},artifact_hashes={},parent_checkpoint='parent.npz',parent_sha256=run.sha(parent),
              baseline_folder='baseline-reference',reference_folders={})
    calls=[]
    monkeypatch.setattr(run.previous.base,'verify_native',lambda:None)
    matched=[]
    monkeypatch.setattr(run.native,'compare',lambda a,b:matched.append((a,b)))
    monkeypatch.setattr(run,'finish',lambda p:dict(synthetic_workflow_fixture_only=True))
    def fake_job(command,**kwargs):
        a=parser().parse_args(command[2:]);calls.append((a.phase,a.bank,a.seed,a.direction,a.sign))
        a.output.mkdir()
        weights=run.arrays(a.checkpoint)['weights']
        quality=.5+.1*(weights[0]-cp['weights'][0])/abs(cp['initial_weights'][0])
        o=dict(omega=np.zeros((1000,1,3)),up_body=np.tile([0.,0.,.7+.3*quality],(1000,1,1)),
               foot_load=np.tile([1/3,1/3,1/3,0.,0.,0.],(1000,1,1)))
        anatomy=dict(time=np.arange(1001)*.005,region_vertical_reaction=np.tile([0.,1.,0.,0.],(1001,1)),com_velocity=np.zeros((1001,3)))
        model=dict(region_columns=list(REGIONS),total_mass=1/9810,gravity=[0.,0.,-9810.])
        context=dict(phase=a.phase,bank=a.bank,seed=a.seed,direction=a.direction,sign=a.sign,
            worker_completed=True,frozen_native_neural_motor_execution=True,online_learning=False,
            checkpoint_sha256=run.sha(a.checkpoint),import_records=[dict(actual_import_exact=True)])
        report=dict(status='completed_transport_pilot',controller='brain',worlds=1,learning=False,motor_exploration=0,
            weights_unchanged=True,requested_simulated_seconds_per_world=5,simulated_seconds_per_world=5,
            completed_control_chunks=1000,control_ms=5.,physics_dt=.0001,ew1_worker_completed=True,seeds=[a.seed],
            checkpoint_sha256=context['checkpoint_sha256'],ew1_context=context,spikes_per_world=[0])
        run.write(a.output/'report.json',report);run.write(a.output/'anatomy-model.json',model)
        run.write(a.output.with_name(a.output.name+'-ew1-context.json'),context)
        for name,data in [('observations.npz',o),('anatomy-observations.npz',anatomy),('adhesion-observations.npz',dict(desired=np.zeros((1000,42))))]:
            np.savez_compressed(a.output/name,**data)
        if a.phase=='baseline':
            reference=tmp_path/'baseline-reference';shutil.copytree(a.output,reference)
            if bad_replay:np.savez_compressed(reference/'adhesion-observations.npz',desired=np.ones((1000,42)))
        return dict(exit_code=0,timed_out=False,tree_empty=True,seconds=0.)
    def measured(path):
        r=run.read(path/'report.json');validate_report(r)
        o=run.arrays(path/'observations.npz');a=run.arrays(path/'anatomy-observations.npz');m=run.read(path/'anatomy-model.json')
        return r,o,a,m,episode_rewards(o,a,m)
    monkeypatch.setattr(run.native,'run_windows_job',fake_job)
    monkeypatch.setattr(run,'measured',measured)
    if bad_replay:
        with pytest.raises(AssertionError):run.execute(plan)
    else:run.execute(plan)
    result=run.read(run.out/'result.json')
    assert result['lease_released'] and not lease.exists() and result['process_tree_empty_verified']
    assert len(calls)==(1 if bad_replay else 31)
    assert result['status']==('failed' if bad_replay else 'completed_EW1')
    if not bad_replay:
        assert calls[0]==('baseline','parent',1004,None,None)
        assert [c[2] for c in calls[1:3]]==list(TRAIN_SEEDS)
        assert len([c for c in calls if c[0]=='trial'])==16
        assert all(c[2] in TRAIN_SEEDS for c in calls if c[0]=='trial')
        assert len(run.read(run.out/'training-returns.json'))==16 and result['learning']['completed_updates']==1
        assert len(matched)==17
        assert all(run.sha(tmp_path/n)==h for n,h in result['created_hashes'].items())
