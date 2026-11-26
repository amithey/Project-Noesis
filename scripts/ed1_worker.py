"""ED1 native brain/body with one declared full-eligibility timing factor."""
import argparse,hashlib,json,sys
from pathlib import Path
from importlib.metadata import version,PackageNotFoundError
import numpy as np
from ap1c_body import observing_body
from ap1_feedback import corrected_reward
from ps1_objective import physical_score,OBJECTIVE
from s2_contract import validate_training_report
from ed1_contract import audited_learner


def validate_training(report,bank):
    expected=OBJECTIVE
    assert report['reward_objective']==expected and report['ed1_context']['bank']==bank and report['ed1_worker_completed']
    assert report['base_reward_objective']=='curriculum-standing-v7'
    assert report['eligibility_audit_rows']==4990 and report['ed1_context']['active_delay_chunks']==(10 if bank=='candidate' else 0)
    validate_training_report({**report,'reward_objective':'curriculum-standing-v7'})


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=('train','frozen'),required=True)
    ap.add_argument('--bank',choices=('replay','control','candidate'),required=True);ap.add_argument('--seed',type=int,required=True)
    for n in ('root','output','checkpoint'):ap.add_argument('--'+n,type=Path,required=True)
    a=ap.parse_args();root=a.root.resolve()
    if (a.phase=='train' and (a.seed!=(1004 if a.bank=='replay' else 1005))) or (a.phase=='frozen' and (a.bank=='replay' or a.seed not in (1005,10006,10007,10010,10011))):ap.error('Unregistered ED1 condition')
    sys.path[:0]=[str(root/'work/Noesis-Sensory/gpu-pilot/src'),str(root/'work/Noesis-Core/src')]
    from noesis_gpu import controller,run,train
    from noesis import plasticity
    module=train if a.phase=='train' else run;old_body=module.FlyGymBatch;old_factory=train.episode_scorer;old_pad=controller.adhesion_from_command
    instances=[];scores=[];pads=[];actual_objective=OBJECTIVE;learners=[]
    old_learner=plasticity.MotorPlasticity;delay=10 if a.bank=='candidate' else 0
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    context=dict(experiment='ED1',phase=a.phase,bank=a.bank,seed=a.seed,training_seed=1004 if a.bank=='replay' else 1005,
        actual_training_objective=actual_objective if a.phase=='train' else None,base_metric_objective='curriculum-standing-v7',
        checkpoint_sha256=digest(a.checkpoint),worker_sha256=digest(Path(__file__)),objective_sha256=digest(Path(__file__).with_name('ps1_objective.py')),
        eligibility_delay_chunks=delay,active_delay_chunks=delay if a.phase=='train' else 0,
        reward_objective_unchanged=True,neural_motor_policy_unchanged=True,neural_controller_unchanged=True,body_senses_live=True,new_acceptance_gate=False)
    a.output.with_name(a.output.name+'-ed1-context.json').write_text(json.dumps(context,indent=2),encoding='utf-8')

    def factory(objective,cycle_weight=None):
        assert objective=='curriculum-standing-v7' and cycle_weight is None
        native_scorer=old_factory(objective,cycle_weight)
        def scorer(before,after,dt_ms):
            assert dt_ms==5 and len(instances)==1
            body=instances[0];assert len(body.ap1_records)==len(scores)+2
            np.testing.assert_array_equal(after['position'],body.ap1_latest_position)
            row=body.ap1_records[-1];native=native_scorer(before,after,dt_ms);q=float(row['non_tarsal_fraction'])
            weight=body.ap1_metadata['total_mass']*abs(body.ap1_metadata['gravity'][2])
            components=physical_score(row['region_vertical_reaction'],row['com_velocity'],after['omega'],after['up_body'][2],after['foot_load'],weight)
            control=corrected_reward(native['reward'],q,.5);applied=components['reward']
            scores.append(dict(control_chunk_index=np.array(len(scores)),time=row['time'].copy(),native_reward=np.array(native['reward']),
                AP1_reference_reward=np.array(control),applied_reward=np.array(applied),**{'PS1_'+k:np.array(v) for k,v in components.items()}))
            return {**native,'reward':applied}
        return scorer

    def observed_pad(desired,rest):
        result=old_pad(desired,rest);pads.append((desired.copy(),(desired-rest).reshape(6,7)[:,3].copy(),result.copy()));return result

    common=['--backend','cpu','--seed',str(a.seed),'--noesis-source',str(root/'work/Noesis-Core/src'),'--data',str(root/'work/NoesisData'),
        '--checkpoint',str(a.checkpoint),'--output',str(a.output)]
    command=common+(['--seconds','25','--wall-seconds','600','--profile','standing','--learning-rate','.2','--exploration-mv-ms','1.5','--log-exploration'] if a.phase=='train' else ['--worlds','1','--seconds','5','--wall-seconds','300'])
    module.FlyGymBatch=observing_body(old_body,instances);controller.adhesion_from_command=observed_pad
    if a.phase=='train':
        train.episode_scorer=factory;plasticity.MotorPlasticity=audited_learner(old_learner,delay,learners)
    success=False
    try:
        report=module.run(module.parser().parse_args(command));count=5000 if a.phase=='train' else 1000
        assert len(instances)==1 and len(instances[0].ap1_records)==count+1 and len(pads)==count
        if a.phase=='train':assert len(scores)==5000 and report['status']=='completed_training_birth' and len(learners)==1 and len(learners[0].ed1_audit)==4990
        else:assert report['status']=='completed_transport_pilot' and report['weights_unchanged']
        success=True
    finally:
        module.FlyGymBatch=old_body;train.episode_scorer=old_factory;controller.adhesion_from_command=old_pad;plasticity.MotorPlasticity=old_learner
        path=a.output/'report.json'
        if instances and a.output.exists():
            body=instances[0]
            if body.ap1_records:np.savez_compressed(a.output/'anatomy-observations.npz',**{k:np.stack([r[k] for r in body.ap1_records]) for k in body.ap1_records[0]})
            (a.output/'anatomy-model.json').write_text(json.dumps(body.ap1_metadata,indent=2),encoding='utf-8')
            if pads:np.savez_compressed(a.output/'adhesion-observations.npz',**{k:np.stack([r[i] for r in pads]) for i,k in enumerate(('desired','Femur_delta','pads'))})
            if scores:np.savez_compressed(a.output/'physical-reward-accounting.npz',**{k:np.stack([r[k] for r in scores]) for k in scores[0]})
        if learners and a.output.exists() and learners[0].ed1_audit:
            rows=learners[0].ed1_audit
            np.savez_compressed(a.output/'eligibility-audit.npz',**{k:np.asarray([r[k] for r in rows]) for k in rows[0]})
        if path.exists():
            report=json.loads(path.read_text());report.update(ed1_context=context,ed1_worker_completed=success,adhesion_probe_mode='legacy',s2_phase=a.phase,
                neural_desired_angles_drive_pads=True,anatomical_measurement_not_balance_validation=True)
            if instances:
                body=instances[0];report['body']=dict(neutral_pose=body.rest.tolist(),lower_limits=body.lo.tolist(),upper_limits=body.hi.tolist(),
                    servo_kp=45.,force_range=[-65.,65.],damping=1.,noslip_iterations=0,axis_order='yaw_pitch_roll',adhesion_mapping='legacy-positive-Femur')
            if a.phase=='train':
                report.update(eligibility_audit_rows=len(learners[0].ed1_audit) if learners else 0,
                    actual_credit_provenance='Raw exploration packet archive is current innovation; eligibility-audit records actual used full-trace source, including delayed warmup',
                    base_reward_objective='curriculum-standing-v7',reward_objective=actual_objective,controller='brain',bg_hz=3.,bg_weight=40.,
                    senses=['joint_position','joint_velocity','ground_load','balance'],rng_schedule='per-world per-chunk reseeding; not legacy bitwise replay',packages={},seeds=[a.seed])
                for name in ('flygym','mujoco','mujoco-warp','warp-lang','numba','numpy'):
                    try:report['packages'][name]=version(name)
                    except PackageNotFoundError:report['packages'][name]=None
            path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
            if a.phase=='train':
                for name in ('weights-last.npz','weights-partial.npz'):
                    cp=a.output/name
                    if cp.exists():cp.with_suffix('.model.json').write_text(json.dumps({**report,'context':'ED1 experimental weights-only memory; delayed queue not checkpointed; no exact episode resume or validated-best'},indent=2,allow_nan=False),encoding='utf-8')
    if a.phase=='train':validate_training(report,a.bank)
    print(json.dumps(dict(experiment='ED1',phase=a.phase,bank=a.bank,completed=success)))


if __name__=='__main__':main()
