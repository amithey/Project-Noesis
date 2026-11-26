"""SS1 finite-scale native worker; explicit read-only runtime and writable output."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from ss1_scale import validate_worker


def validate(a):
    validate_worker(a)


def main():
    ap=argparse.ArgumentParser()
    for n in ('runtime-root','output','checkpoint'):ap.add_argument('--'+n,type=Path,required=True)
    ap.add_argument('--phase',choices=('baseline','trial'),required=True)
    ap.add_argument('--bank',choices=('parent','probe'),required=True)
    ap.add_argument('--amplitude',type=float);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--direction',type=int);ap.add_argument('--sign',type=int)
    a=ap.parse_args();validate(a);root=a.runtime_root.resolve()
    sys.path[:0]=[str(root/'work'),str(root/'work/Noesis-Sensory/gpu-pilot/src'),str(root/'work/Noesis-Core/src')]
    from ap1c_body import observing_body
    from ws1_import import checked_import
    from noesis_gpu import controller,run
    old_body,old_pad,old_import=run.FlyGymBatch,controller.adhesion_from_command,controller.import_checkpoint
    instances,pads,imports=[],[],[]
    context=dict(experiment='SS1',phase=a.phase,bank=a.bank,seed=a.seed,direction=a.direction,sign=a.sign,amplitude=a.amplitude,
        checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
        worker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        frozen_native_neural_motor_execution=True,online_learning=False,
        checkpoint_trace_fields='Historical compatibility fields discarded by weights-only import')
    def observed_import(net,path):
        if Path(path).resolve()!=a.checkpoint.resolve():raise ValueError('Checkpoint path mismatch')
        imports.append(checked_import(old_import,net,path))
    def observed_pad(desired,rest):
        result=old_pad(desired,rest);pads.append((desired.copy(),(desired-rest).reshape(6,7)[:,3].copy(),result.copy()))
        return result
    command=['--backend','cpu','--seed',str(a.seed),'--noesis-source',str(root/'work/Noesis-Core/src'),
        '--data',str(root/'work/NoesisData'),'--checkpoint',str(a.checkpoint),'--output',str(a.output),
        '--worlds','1','--seconds','5','--wall-seconds','300']
    run.FlyGymBatch=observing_body(old_body,instances)
    controller.adhesion_from_command=observed_pad;controller.import_checkpoint=observed_import;success=False
    try:
        report=run.run(run.parser().parse_args(command))
        assert len(instances)==len(imports)==1 and len(instances[0].ap1_records)==1001 and len(pads)==1000
        assert report['status']=='completed_transport_pilot' and report['completed_control_chunks']==1000
        assert report['simulated_seconds_per_world']==5 and not report['learning'] and report['weights_unchanged']
        success=True
    finally:
        run.FlyGymBatch=old_body;controller.adhesion_from_command=old_pad;controller.import_checkpoint=old_import
        context.update(worker_completed=success,import_records=imports)
        a.output.with_name(a.output.name+'-ss1-context.json').write_text(json.dumps(context,indent=2),encoding='utf-8')
        if instances and a.output.exists():
            body=instances[0]
            if body.ap1_records:np.savez_compressed(a.output/'anatomy-observations.npz',**{k:np.stack([r[k] for r in body.ap1_records]) for k in body.ap1_records[0]})
            (a.output/'anatomy-model.json').write_text(json.dumps(body.ap1_metadata,indent=2),encoding='utf-8')
            if pads:np.savez_compressed(a.output/'adhesion-observations.npz',**{k:np.stack([r[i] for r in pads]) for i,k in enumerate(('desired','Femur_delta','pads'))})
        path=a.output/'report.json'
        if path.exists():
            report=json.loads(path.read_text());report.update(ss1_context=context,ss1_worker_completed=success,
                adhesion_probe_mode='legacy',neural_desired_angles_drive_pads=True,anatomical_measurement_not_balance_validation=True)
            if instances:
                body=instances[0]
                report['body']=dict(neutral_pose=body.rest.tolist(),lower_limits=body.lo.tolist(),upper_limits=body.hi.tolist(),
                    servo_kp=45.,force_range=[-65.,65.],damping=1.,noslip_iterations=0,axis_order='yaw_pitch_roll',adhesion_mapping='legacy-positive-Femur')
            path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')


if __name__=='__main__':main()
