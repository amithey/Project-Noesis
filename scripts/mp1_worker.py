"""MP1 full native frozen trials; learning occurs only between complete episodes."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from ap1c_body import observing_body
from ws1_import import checked_import
from mp1_select import TRAIN_SEEDS,DIRECTIONS,DEV,FRESH



def parser():
    ap=argparse.ArgumentParser()
    for name in ('root','output','checkpoint'):ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--phase',choices=('baseline','trial','evaluation'),required=True)
    ap.add_argument('--bank',choices=('parent','probe','candidate','negative','gradient'),required=True)
    ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--direction',type=int)
    ap.add_argument('--sign',type=int)
    return ap


def validate_arguments(a):
    if a.phase=='baseline':
        valid=a.bank=='parent' and a.seed==1004 and a.direction is None and a.sign is None
    elif a.phase=='trial':
        valid=a.bank=='probe' and a.seed in TRAIN_SEEDS and a.direction in range(DIRECTIONS) and a.sign in (-1,1)
    else:
        valid=a.bank in ('parent','candidate','negative','gradient') and a.seed in (*TRAIN_SEEDS,*DEV,*FRESH) and a.direction is None and a.sign is None
    if not valid:raise ValueError('Unregistered MP1 trial/bank/seed/direction')


def main():
    ap=parser();a=ap.parse_args()
    try:validate_arguments(a)
    except ValueError as error:ap.error(str(error))
    root=a.root.resolve()
    sys.path[:0]=[str(root/'work/Noesis-Sensory/gpu-pilot/src'),str(root/'work/Noesis-Core/src')]
    from noesis_gpu import controller,run
    old_body,old_pad,old_import=run.FlyGymBatch,controller.adhesion_from_command,controller.import_checkpoint
    instances,pads,imports=[],[],[]
    context=dict(experiment='MP1',phase=a.phase,bank=a.bank,seed=a.seed,direction=a.direction,sign=a.sign,
        checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
        worker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        frozen_native_neural_motor_execution=True,online_learning=False,
        learning_role='Training-trial data for between-episode updates' if a.phase=='trial' else 'Frozen evaluation only',
        checkpoint_trace_fields='Historical native compatibility fields; discarded by weights-only import')

    def observed_import(net,path):
        assert Path(path).resolve()==a.checkpoint.resolve()
        imports.append(checked_import(old_import,net,path))

    def observed_pad(desired,rest):
        result=old_pad(desired,rest)
        pads.append((desired.copy(),(desired-rest).reshape(6,7)[:,3].copy(),result.copy()))
        return result

    command=['--backend','cpu','--seed',str(a.seed),'--noesis-source',str(root/'work/Noesis-Core/src'),
        '--data',str(root/'work/NoesisData'),'--checkpoint',str(a.checkpoint),'--output',str(a.output),
        '--worlds','1','--seconds','5','--wall-seconds','300']
    run.FlyGymBatch=observing_body(old_body,instances)
    controller.adhesion_from_command=observed_pad;controller.import_checkpoint=observed_import
    success=False
    try:
        report=run.run(run.parser().parse_args(command))
        assert len(instances)==len(imports)==1 and len(instances[0].ap1_records)==1001 and len(pads)==1000
        assert report['status']=='completed_transport_pilot' and report['completed_control_chunks']==1000
        assert report['simulated_seconds_per_world']==5 and not report['learning'] and report['weights_unchanged']
        success=True
    finally:
        run.FlyGymBatch=old_body;controller.adhesion_from_command=old_pad;controller.import_checkpoint=old_import
        context.update(worker_completed=success,import_records=imports)
        a.output.with_name(a.output.name+'-mp1-context.json').write_text(json.dumps(context,indent=2),encoding='utf-8')
        if instances and a.output.exists():
            body=instances[0]
            if body.ap1_records:np.savez_compressed(a.output/'anatomy-observations.npz',**{k:np.stack([r[k] for r in body.ap1_records]) for k in body.ap1_records[0]})
            (a.output/'anatomy-model.json').write_text(json.dumps(body.ap1_metadata,indent=2),encoding='utf-8')
            if pads:np.savez_compressed(a.output/'adhesion-observations.npz',**{k:np.stack([r[i] for r in pads]) for i,k in enumerate(('desired','Femur_delta','pads'))})
        path=a.output/'report.json'
        if path.exists():
            report=json.loads(path.read_text());report.update(mp1_context=context,mp1_worker_completed=success,
                adhesion_probe_mode='legacy',neural_desired_angles_drive_pads=True,anatomical_measurement_not_balance_validation=True)
            if instances:
                body=instances[0]
                report['body']=dict(neutral_pose=body.rest.tolist(),lower_limits=body.lo.tolist(),upper_limits=body.hi.tolist(),
                    servo_kp=45.,force_range=[-65.,65.],damping=1.,noslip_iterations=0,axis_order='yaw_pitch_roll',adhesion_mapping='legacy-positive-Femur')
            path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')


if __name__=='__main__':main()
