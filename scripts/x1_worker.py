"""Native frozen physics with current exposure on/off and read-only learned weights."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from p0_control import control_brain
from x1_drive import current_intervention


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--drive',choices=('on','off'),required=True)
    ap.add_argument('--mode',choices=('legacy','corrected'),required=True)
    for n in ('root','output','checkpoint'):ap.add_argument('--'+n,type=Path,required=True)
    args=ap.parse_args();root=args.root.resolve()
    sys.path[:0]=[str(root/'work/Noesis-Sensory/gpu-pilot/src'),str(root/'work/Noesis-Core/src')]
    from noesis_gpu import controller,run,learning
    from adhesion_mapping import negative_femur_adhesion
    base=control_brain(learning.LearningBrain);instances=[];pads=[]
    original_brain=controller.FrozenBrain;original_pad=controller.adhesion_from_command
    context=dict(experiment='X1',drive=args.drive,mode=args.mode,seed=1004,learning=False,
                 checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                 worker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with args.output.with_name(args.output.name+'-context.json').open('x',encoding='utf-8') as f:json.dump(context,f,indent=2)

    class Probe(base):
        def __init__(self,*a,**kw):
            super().__init__(*a,exploration_mv_ms=1.5,learning_rate=.2,record_exploration=True,**kw)
            self.chunks=0;self.latent=[];self.seed_stream=[];instances.append(self)

        def _current_drive(self,dt_ms):
            drive=super()._current_drive(dt_ms)
            if drive:self.latent.append(np.asarray(drive['current_mv_per_ms'],np.float32).copy())
            actual=current_intervention(drive,args.drive=='on')
            if actual:self._log['drive'][-1]=np.asarray(actual['current_mv_per_ms'],np.float32).copy()
            return actual

        def step(self,observation,dt_ms):
            self.learning_active=self.chunks>=10
            preview=np.random.default_rng(0);preview.bit_generator.state=self.rng.bit_generator.state
            self.seed_stream.append(int(preview.integers(0,2**31)))
            result=super().step(observation,dt_ms)
            self.learn(0.,dt_ms)  # control only logs credit; never updates weights/traces
            self.chunks+=1
            return result

    def observed(desired,rest):
        p=original_pad(desired,rest) if args.mode=='legacy' else negative_femur_adhesion(desired,rest)
        pads.append((np.asarray(desired).copy(),(np.asarray(desired)-rest).reshape(6,7)[:,3].copy(),p.copy()))
        return p

    controller.FrozenBrain=Probe;controller.adhesion_from_command=observed
    try:
        report=run.run(run.parser().parse_args(['--backend','cpu','--worlds','1','--seconds','5','--wall-seconds','300',
            '--seed','1004','--noesis-source',str(root/'work/Noesis-Core/src'),'--data',str(root/'work/NoesisData'),
            '--checkpoint',str(args.checkpoint),'--output',str(args.output)]))
    finally:
        controller.FrozenBrain=original_brain;controller.adhesion_from_command=original_pad
        if (args.output/'report.json').exists():
            partial=json.loads((args.output/'report.json').read_text())
            partial.update(x1_context=context,motor_exploration=1.5 if args.drive=='on' else 0,
                           plasticity_enabled=False,learning_updates=0,exploration_latent_generated=True,
                           drive_onset_chunk=10,latent_credit_not_used_for_learning=True)
            partial['body']['adhesion_mapping']='legacy-positive-Femur' if args.mode=='legacy' else 'corrected-negative-Femur'
            (args.output/'report.json').write_text(json.dumps(partial,indent=2,allow_nan=False),encoding='utf-8')
    report=json.loads((args.output/'report.json').read_text())
    assert report['status']=='completed_transport_pilot' and report['weights_unchanged'] and len(instances)==1
    b=instances[0];assert b.chunks==1000 and b.control_steps==990 and b.updates==0 and not b.net.data.flags.writeable and b.weights_unchanged()
    log=b.exploration_log();np.savez_compressed(args.output/'exploration-log.npz',**log,
        control_chunk_index=np.arange(10,1000),control_ms=np.array(5.))
    np.savez_compressed(args.output/'latent-and-seeds.npz',latent_drive=np.stack(b.latent),
                        chunk_seeds=np.asarray(b.seed_stream,np.uint32))
    arrays={k:np.stack([row[i] for row in pads]) for i,k in enumerate(('desired','Femur_delta','pads'))}
    np.savez_compressed(args.output/'adhesion-observations.npz',**arrays)
    report.update(network_weights_readonly=True,full_network_weights_unchanged=True,control_steps=990)
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(status=report['status'],mode=args.mode,drive=args.drive,weights_unchanged=True)))


if __name__=='__main__':main()
