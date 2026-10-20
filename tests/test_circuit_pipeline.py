"""Circuit scope through real episode/checkpoint and train-validate orchestration."""
import json
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from noesis.connectome import Connectome
from noesis.experiments import learning_lab as lab
from noesis.experiments import train_validate as tv
from test_learning_lab import lab_environment as lab_environment  # noqa: PLC0414


def configure(env):
    neurons=env.cn.neurons.copy()
    neurons.loc[0,'Class']='single_leg_neuromere'
    env.cn=Connectome(neurons,sparse.csr_matrix([[0.,50.],[-2.,0.]]))
    a=env.args
    a.learning_rule='perturbation'; a.learning_rate=.2
    a.normalize_reward_error=True; a.bias_correct_reward_error=True
    a.reward_mode='grounded'; a.grounded_objective='standing-gate-linear-slip-v5'
    return a


def test_old_subset_to_expanded_episode_then_frozen_has_no_injected_currents(lab_environment):
    env=lab_environment; a=configure(env)
    _,old=lab.run_episode(env.cn,a,0,None)
    previous=env.learners[-1]
    previous.net.data[previous.edge_indices]*=1.1
    previous.save(old)
    assert json.loads(old.with_suffix('.model.json').read_text())['schema']==8
    a.learning_scope='motor-circuit-v1'
    with pytest.raises(ValueError,match='context mismatch'):
        lab.run_episode(env.cn,a,1,old)
    a.import_weights=True; a.motor_exploration=1.5
    summary,expanded=lab.run_episode(env.cn,a,2,old)
    context=json.loads(expanded.with_suffix('.model.json').read_text())
    assert context['schema']==9 and context['additional_neural_targets']==[0]
    assert summary['plasticity_target_count']==2 and summary['additional_neural_target_count']==1
    assert set(env.networks[-1].last_drive['current_idx'])=={0,1}
    with np.load(expanded) as z:
        np.testing.assert_array_equal(z['initial_weights'],[50,-2])
        np.testing.assert_allclose(z['weights'],[55,-2])
        saved=z['weights'].copy()
        assert z['reward_normalization_mass']>0
    a.learn=False; a.motor_exploration=0; a.import_weights=False
    frozen,cp=lab.run_episode(env.cn,a,3,expanded)
    assert not frozen['learning'] and frozen['motor_exploration_mv_ms']==0
    assert 'current_idx' not in env.networks[-1].last_drive
    with np.load(cp) as z:
        np.testing.assert_array_equal(z['weights'],saved)
        for key in ('activity','eligibility','reward_baseline','reward_variance','reward_normalization_mass'):
            assert not z[key].any()
    assert all(sim.closed for sim in env.simulations)


def test_scope_rejected_before_body_creation_when_credit_configuration_invalid(lab_environment):
    env=lab_environment; env.args.learning_scope='motor-circuit-v1'
    with pytest.raises(SystemExit): lab.validate_args(lab.parser(),env.args)
    with pytest.raises(ValueError,match='requires'):
        lab.run_episode(env.cn,env.args,0,None)
    assert not env.simulations


def test_driver_propagates_scope_to_all_controls_and_replays_incomplete_training(monkeypatch,tmp_path):
    calls=[]; interrupt=[True]
    def episode(cn,args,index,checkpoint,*,deadline):
        calls.append(vars(args).copy())
        s={'seed':args.seed,'status':'completed','simulated_seconds':5.,'learning':args.learn,
           'motor_exploration_mv_ms':args.motor_exploration,'standing_fraction':.95,
           'mean_reward':.3 if checkpoint else .2,'mean_stance_slip_mm_s':.1 if checkpoint else .2,
           'heading_forward_distance_mm':2.,'leg_step_cycles':dict.fromkeys(('LF','LM','LH','RF','RM','RH'),3),
           'obstacle_observation_complete':True,'obstacle_response_seconds':.2,
           'obstacle_contact_seconds':.4 if args.no_environment_senses else .1}
        if interrupt[0]: s['status']='wall_budget'; interrupt[0]=False
        saved=args.output/'episode_000.npz'; saved.write_bytes(b'orchestration-test')
        return s,saved
    monkeypatch.setattr(tv,'load_training_brain',lambda *a:(None,{}))
    monkeypatch.setattr(tv.lab,'run_episode',episode)
    output=tmp_path/'run'
    base=['train_validate','--data',str(tmp_path/'data'),'--output',str(output),'--episodes','1',
          '--learning-scope','motor-circuit-v1','--normalize-reward-error','--bias-correct-reward-error']
    monkeypatch.setattr('sys.argv',base); tv.main()
    report=json.loads((output/'report.json').read_text())
    assert not report['jobs'] and report['best_validated_checkpoint'] is None
    monkeypatch.setattr('sys.argv',base+['--resume']); tv.main()
    report=json.loads((output/'report.json').read_text())
    assert report['status']=='completed' and len(report['jobs'])==26
    assert calls[0]['seed']==calls[1]['seed'] and calls[0]['learn']==calls[1]['learn']
    assert all(c['learning_scope']=='motor-circuit-v1' for c in calls)
    frozen=[c for c in calls if not c['learn']]
    assert len(frozen)==25 and all(c['motor_exploration']==c['training_fall_seconds']==0 for c in frozen)
    assert all(not c['no_senses'] and not c['no_balance'] for c in frozen)
    assert any(c['no_environment_senses'] for c in frozen)
    before=(output/'report.json').read_bytes()
    monkeypatch.setattr('sys.argv',[v for v in base if v!='--bias-correct-reward-error']+['--resume'])
    with pytest.raises(SystemExit): tv.main()
    assert (output/'report.json').read_bytes()==before


def test_default_model_context_stays_schema8_and_omits_scope(lab_environment):
    env=lab_environment; a=configure(env)
    s,cp=lab.run_episode(env.cn,a,0,None)
    context=json.loads(cp.with_suffix('.model.json').read_text())
    assert context['schema']==8 and 'learning_scope' not in context
    assert 'learning_scope' not in s
    assert not cp.with_suffix('.learning-targets.json').exists()
    assert Path(lab.__file__).is_relative_to(Path(__file__).resolve().parents[1]/'src')
