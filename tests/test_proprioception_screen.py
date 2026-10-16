import importlib.util
import json
from pathlib import Path

import pytest

spec=importlib.util.spec_from_file_location('proprioception_screen',Path(__file__).parents[1]/'scripts/screen_proprioception.py')
screen=importlib.util.module_from_spec(spec)
spec.loader.exec_module(screen)


def test_screen_replays_unfinished_original_trial_and_never_trains_or_promotes(monkeypatch,tmp_path):
    calls=[];interrupt=[True]
    def episode(cn,args,index,checkpoint,*,deadline):
        calls.append((args.seed,args.proprioceptive_encoding,args.learn,args.motor_exploration,
                      args.no_senses,args.no_balance,args.no_environment_senses,args.seconds,checkpoint))
        summary={'status':'completed','simulated_seconds':5.,'learning':False,'motor_exploration_mv_ms':0.}
        if len(calls)==2 and interrupt[0]:
            summary.update(status='wall_budget',simulated_seconds=2.)
            interrupt[0]=False
        saved=args.output/'episode_000.npz';saved.write_bytes(b'original')
        return summary,saved
    monkeypatch.setattr(screen,'load_training_brain',lambda *a:(None,{}))
    monkeypatch.setattr(screen.lab,'run_episode',episode)
    output=tmp_path/'screen';argv=['--data',str(tmp_path/'data'),'--output',str(output)]
    screen.main(argv)
    first=json.loads((output/'report.json').read_text())
    assert len(first['completed'])==1 and first['status']=='pending'
    screen.main(argv+['--resume'])
    final=json.loads((output/'report.json').read_text())
    assert calls[1]==calls[2]
    assert final['status']=='completed' and len(final['completed'])==6
    assert not final['promotion_allowed'] and final['development_only']
    assert all(not c[2] and c[3]==0 and not c[4] and not c[5] and not c[6] and c[7]==5 and c[8] is None for c in calls)
    assert {c[:2] for c in calls}=={(seed,encoding) for seed in (15,16) for encoding in ('legacy','typed-resistance','typed-reversed')}
    assert len(final['attempts'])==7


@pytest.mark.parametrize('seed',[10,11,200,201,202])
def test_screen_preserves_reserved_validation_and_heldout_seeds(tmp_path,seed):
    with pytest.raises(SystemExit):
        screen.main(['--data',str(tmp_path/'data'),'--output',str(tmp_path/'out'),'--seeds',str(seed)])
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('seconds',['0','3601','nan','inf'])
def test_screen_rejects_invalid_wall_budget(tmp_path,seconds):
    with pytest.raises(SystemExit):
        screen.main(['--data',str(tmp_path/'data'),'--output',str(tmp_path/'out'),'--wall-seconds',seconds])
    assert not (tmp_path/'out').exists()


def test_screen_custom_encodings_stay_frozen_and_resume_cannot_change_comparison(monkeypatch,tmp_path):
    calls=[]
    def episode(cn,args,index,checkpoint,*,deadline):
        calls.append((args.seed,args.proprioceptive_encoding,args.learn,args.motor_exploration,checkpoint))
        saved=args.output/'episode_000.npz';saved.write_bytes(b'original')
        return {'status':'completed','simulated_seconds':5.,'learning':False,'motor_exploration_mv_ms':0.},saved
    monkeypatch.setattr(screen,'load_training_brain',lambda *a:(None,{}))
    monkeypatch.setattr(screen.lab,'run_episode',episode)
    output=tmp_path/'custom'
    argv=['--data',str(tmp_path/'data'),'--output',str(output),'--encodings','typed-resistance','typed-consistent']
    screen.main(argv)
    assert calls==[(15,'typed-resistance',False,0.,None),(15,'typed-consistent',False,0.,None),
                   (16,'typed-consistent',False,0.,None),(16,'typed-resistance',False,0.,None)]
    report=json.loads((output/'report.json').read_text())
    assert len(report['completed'])==4 and not report['promotion_allowed']
    with pytest.raises(SystemExit):
        screen.main(argv+['--resume','--encodings','legacy'])


def test_screen_rejects_duplicate_encodings_before_creating_output(tmp_path):
    with pytest.raises(SystemExit):
        screen.main(['--data',str(tmp_path/'data'),'--output',str(tmp_path/'out'),
                     '--encodings','typed-consistent','typed-consistent'])
    assert not (tmp_path/'out').exists()
