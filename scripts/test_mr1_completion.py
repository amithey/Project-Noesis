"""Synthetic allocation/cleanup controls, never a physical success claim."""
import shutil
import numpy as np
import pytest
from noesis.plasticity import MotorPlasticity
from test_ew1_state import network
from test_ew1_contract import features
from mp1_worker import parser
from mp1_contract import validate_report,episode_rewards
from awake_guard import AwakeRequest,CONTINUOUS,SYSTEM_REQUIRED
import run_mp1_completion_mr1 as run


@pytest.mark.parametrize('fault',['none','first_replay','timeout'])
def test_completion_is_scoped_preserves_original_and_never_retries(tmp_path,monkeypatch,fault):
    monkeypatch.setattr(run,'root',tmp_path);monkeypatch.setattr(run,'out',tmp_path/'completion')
    monkeypatch.setattr(run.old,'root',tmp_path);monkeypatch.setattr(run.old,'out',tmp_path/'original');run.old.out.mkdir()
    (tmp_path/'outputs').mkdir();run.write(tmp_path/'outputs/noesis-required-data-sha256.json',dict(files=[]))
    lease=tmp_path/'work/Project-Noesis/gpu-pilot/outputs/codex-training-host.lock';lease.parent.mkdir(parents=True)
    parent=tmp_path/'parent.npz';MotorPlasticity(network(),[2]).save(parent)
    for bank in ('negative','gradient'):shutil.copyfile(parent,run.old.out/(bank+'.npz'))
    run.write(run.old.out/'result.json',dict(status='failed',fixture=True));old_result_hash=run.sha(run.old.out/'result.json')
    p=dict(parent_checkpoint='parent.npz',source_hashes={'work/mp1_worker.py':'a'*64},reference_folders={str(s):'original/unused-reference' for s in (10006,10007,10010,10011)})
    protocol=dict(original_protocol=p,original_hashes={},source_hashes={})
    original_parent=run.old.out/'parent-1010';original_parent.mkdir()
    o,a,m=features()
    for filename,data in [('observations.npz',o),('adhesion-observations.npz',dict(desired=np.zeros((1000,42)))),('anatomy-observations.npz',a)]:
        np.savez_compressed(original_parent/filename,**data)
    run.write(original_parent/'report.json',dict(spikes_per_world=[0]))
    power_calls=[]
    def api(flags):power_calls.append(flags);return CONTINUOUS
    guard=AwakeRequest(api);monkeypatch.setattr(run,'AwakeRequest',lambda:guard)
    monkeypatch.setattr(run.old.native,'compare',lambda a,b:None)
    monkeypatch.setattr(run,'finish',lambda p:dict(synthetic_fixture_only=True))
    calls=[]
    def job(command,**kwargs):
        args=parser().parse_args(command[2:]);calls.append(args)
        assert power_calls==[CONTINUOUS|SYSTEM_REQUIRED]
        assert kwargs['timeout']==300 and args.phase=='evaluation'
        if fault=='timeout':return dict(exit_code=124,timed_out=True,tree_empty=True,seconds=301.)
        args.output.mkdir();o,a,m=features()
        if fault=='first_replay':o['up_body'][0,0,0]=.1
        c=dict(phase='evaluation',bank=args.bank,seed=args.seed,direction=None,sign=None,worker_completed=True,
            online_learning=False,frozen_native_neural_motor_execution=True,checkpoint_sha256=run.sha(args.checkpoint),
            worker_sha256='a'*64,import_records=[dict(actual_import_exact=True)])
        report=dict(status='completed_transport_pilot',controller='brain',worlds=1,learning=False,motor_exploration=0,
            weights_unchanged=True,requested_simulated_seconds_per_world=5,simulated_seconds_per_world=5,
            completed_control_chunks=1000,control_ms=5.,physics_dt=.0001,mp1_worker_completed=True,
            seeds=[args.seed],checkpoint_sha256=c['checkpoint_sha256'],mp1_context=c,spikes_per_world=[0])
        run.write(args.output/'report.json',report);run.write(args.output/'anatomy-model.json',m)
        run.write(args.output.with_name(args.output.name+'-mp1-context.json'),c)
        for filename,data in [('observations.npz',o),('adhesion-observations.npz',dict(desired=np.zeros((1000,42)))),('anatomy-observations.npz',a)]:
            np.savez_compressed(args.output/filename,**data)
        return dict(exit_code=0,timed_out=False,tree_empty=True,seconds=0.)
    def measured(path):
        r=run.read(path/'report.json');validate_report(r)
        o=run.arrays(path/'observations.npz');a=run.arrays(path/'anatomy-observations.npz');m=run.read(path/'anatomy-model.json')
        return r,o,a,m,episode_rewards(o,a,m)
    monkeypatch.setattr(run.old.native,'run_windows_job',job);monkeypatch.setattr(run.old,'measured',measured)
    if fault=='none':run.execute(protocol)
    else:
        with pytest.raises(AssertionError):run.execute(protocol)
    result=run.read(run.out/'result.json')
    assert len(calls)==(16 if fault=='none' else 1)
    assert result['status']==('completed_MR1_cells' if fault=='none' else 'failed_MR1')
    assert result['lease_released'] and result['awake_request_released'] and not lease.exists()
    assert power_calls==[CONTINUOUS|SYSTEM_REQUIRED,CONTINUOUS]
    assert run.sha(run.old.out/'result.json')==old_result_hash
