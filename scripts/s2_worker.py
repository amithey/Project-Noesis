"""Native standing birth or frozen probe with an explicitly labeled pad-mapping context."""
import argparse
import hashlib
from importlib.metadata import version,PackageNotFoundError
import json
from pathlib import Path
import sys

import numpy as np
from adhesion_mapping import negative_femur_adhesion
from s2_contract import validate_training_report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--phase',choices=('train','frozen'),required=True)
    ap.add_argument('--mode',choices=('legacy','corrected'),required=True)
    ap.add_argument('--seed',type=int,required=True)
    for name in ('root','output','checkpoint'):
        ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args()
    if args.seed in (10,11,200,201,202):ap.error('Final physical validation seed prohibited')
    root=args.root.resolve();pilot=root/'work/Noesis-Sensory';core=root/'work/Noesis-Core/src'
    sys.path[:0]=[str(pilot/'gpu-pilot/src'),str(core)]
    from noesis_gpu import controller,run,train
    module=train if args.phase=='train' else run
    old_pad=controller.adhesion_from_command;old_body=module.FlyGymBatch
    records=[];bodies=[]
    mapping='legacy-positive-Femur' if args.mode=='legacy' else 'corrected-negative-Femur'
    context=dict(phase=args.phase,mode=args.mode,seed=args.seed,adhesion_mapping=mapping,
                 loaded_checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                 native_runtime_commit='e0cbd109bb6e018bd253c9f1b5e1698e0421b276',
                 core_commit='42c20769376bda5694b4c1c66ca75e52cbd9f1f4',
                 worker_module_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 mapping_module_sha256=hashlib.sha256(Path(__file__).with_name('adhesion_mapping.py').read_bytes()).hexdigest())
    with args.output.with_name(args.output.name+'-context.json').open('x',encoding='utf-8') as f:
        json.dump(context,f,indent=2)

    class ObservedBody(old_body):
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw);bodies.append(self)

    def observed(desired,rest):
        pads=old_pad(desired,rest) if args.mode=='legacy' else negative_femur_adhesion(desired,rest)
        records.append((np.asarray(desired).copy(),(np.asarray(desired)-rest).reshape(6,7)[:,3].copy(),pads.copy()))
        return pads

    common=['--backend','cpu','--seed',str(args.seed),'--noesis-source',str(core),
            '--data',str(root/'work/NoesisData'),'--checkpoint',str(args.checkpoint),'--output',str(args.output)]
    if args.phase=='train':
        command=common+['--seconds','25','--wall-seconds','600','--profile','standing',
                        '--learning-rate','.2','--exploration-mv-ms','1.5','--log-exploration']
    else:command=common+['--worlds','1','--seconds','5','--wall-seconds','300']
    controller.adhesion_from_command=observed;module.FlyGymBatch=ObservedBody
    try:
        report=module.run(module.parser().parse_args(command))
    finally:
        controller.adhesion_from_command=old_pad;module.FlyGymBatch=old_body
    if len(bodies)!=1:raise RuntimeError('Wrong body population')
    body=bodies[0]
    body_context=dict(neutral_pose=body.rest.tolist(),lower_limits=body.lo.tolist(),upper_limits=body.hi.tolist(),
                      servo_kp=45.,force_range=[-65.,65.],damping=1.,noslip_iterations=0,axis_order='yaw_pitch_roll')
    if args.phase=='frozen':assert report['body']==body_context
    body_context['adhesion_mapping']=mapping
    report.update(body=body_context,adhesion_probe_mode=args.mode,s2_phase=args.phase,
                  neural_desired_angles_drive_pads=True,observer_module_sha256=context['worker_module_sha256'],
                  mapping_module_sha256=context['mapping_module_sha256'])
    if args.phase=='train':
        report.update(controller='brain',bg_hz=3.,bg_weight=40.,
                      senses=['joint_position','joint_velocity','ground_load','balance'],
                      rng_schedule='per-world per-chunk reseeding; not legacy bitwise replay',packages={},seeds=[args.seed])
        for name in ('flygym','mujoco','mujoco-warp','warp-lang','numba','numpy'):
            try:report['packages'][name]=version(name)
            except PackageNotFoundError:report['packages'][name]=None
    count=5000 if args.phase=='train' else 1000
    # Preserve partial native results/context if the worker completed a shorter episode.
    if records:
        arrays={key:np.stack([r[i] for r in records]) for i,key in enumerate(('desired','Femur_delta','pads'))}
        np.savez_compressed(args.output/'adhesion-observations.npz',**arrays)
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    if args.phase=='train':
        checkpoint=args.output/('weights-last.npz' if report['status']=='completed_training_birth' else 'weights-partial.npz')
        if checkpoint.exists():
            sidecar={**report,'context':'S2 experimental birth with explicit body policy; weights-only memory, not exact episode resume'}
            checkpoint.with_suffix('.model.json').write_text(json.dumps(sidecar,indent=2,allow_nan=False),encoding='utf-8')
        validate_training_report(report)
    else:assert report['status']=='completed_transport_pilot' and report['weights_unchanged']
    assert len(records)==count and all(np.isfinite(a).all() for a in arrays.values())
    assert arrays['desired'].shape==(count,42) and arrays['Femur_delta'].shape==arrays['pads'].shape==(count,6)
    print(json.dumps({k:report[k] for k in ('status','adhesion_probe_mode','s2_phase')}))


if __name__=='__main__':main()
