import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
import mp1_worker
from noesis_gpu import controller,run


def test_native_motor_and_import_observation_code_is_unchanged_from_ew1():
    def functions(path):
        tree=ast.parse(Path(path).read_text())
        return {n.name:ast.dump(n,include_attributes=False) for n in ast.walk(tree) if isinstance(n,ast.FunctionDef)}
    old,new=functions('work/ew1_worker.py'),functions('work/mp1_worker.py')
    for name in ('observed_pad','observed_import'):assert old[name]==new[name]


@pytest.mark.parametrize('seed',[10,11,200,201,202,1008,1009,10006,11008,11009])
def test_trial_scope_excludes_old_training_frozen_and_final_seeds(seed):
    with pytest.raises(ValueError):mp1_worker.validate_arguments(SimpleNamespace(phase='trial',bank='probe',seed=seed,direction=0,sign=1))


def test_patch_cleanup_on_native_failure(tmp_path,monkeypatch):
    cp=tmp_path/'cp.npz';cp.write_bytes(b'test fixture only')
    original=(run.FlyGymBatch,controller.adhesion_from_command,controller.import_checkpoint)
    monkeypatch.setattr(sys,'argv',['mp1_worker','--root',str(tmp_path),'--output',str(tmp_path/'probe'),'--checkpoint',str(cp),
        '--phase','trial','--bank','probe','--seed','1010','--direction','0','--sign','-1'])
    def fail(args):
        assert args.seed==1010 and args.seconds==5 and args.worlds==1 and args.wall_seconds==300
        raise RuntimeError('fixture')
    monkeypatch.setattr(run,'run',fail)
    with pytest.raises(RuntimeError):mp1_worker.main()
    assert (run.FlyGymBatch,controller.adhesion_from_command,controller.import_checkpoint)==original
    assert not json.loads((tmp_path/'probe-mp1-context.json').read_text())['worker_completed']
