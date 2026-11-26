import json
import sys
from types import SimpleNamespace
import pytest
import ew1_worker
from noesis_gpu import run,controller


@pytest.mark.parametrize('seed',[10,11,200,201,202,10006])
def test_trial_registration_excludes_frozen_and_final_seeds(seed):
    with pytest.raises(ValueError):
        ew1_worker.validate_arguments(SimpleNamespace(phase='trial',bank='probe',seed=seed,direction=0,sign=1))


def test_observational_patches_restore_even_if_native_run_fails(tmp_path,monkeypatch):
    checkpoint=tmp_path/'cp.npz';checkpoint.write_bytes(b'unit-test only')
    original=(run.FlyGymBatch,controller.adhesion_from_command,controller.import_checkpoint)
    monkeypatch.setattr(sys,'argv',['ew1_worker','--root',str(tmp_path),'--output',str(tmp_path/'probe'),
        '--checkpoint',str(checkpoint),'--phase','trial','--bank','probe','--seed','1008','--direction','0','--sign','-1'])
    def fail(args):
        assert args.seed==1008 and args.seconds==5 and args.worlds==1 and args.backend=='cpu'
        assert args.wall_seconds==300
        raise RuntimeError('native run failure fixture')
    monkeypatch.setattr(run,'run',fail)
    with pytest.raises(RuntimeError):ew1_worker.main()
    assert (run.FlyGymBatch,controller.adhesion_from_command,controller.import_checkpoint)==original
    context=json.loads((tmp_path/'probe-ew1-context.json').read_text())
    assert not context['worker_completed'] and not context['online_learning']
