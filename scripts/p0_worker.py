"""Delegate identical S2 exposure; freeze all network weights only in control arm."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from p0_control import control_brain,validate_control_native


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm',choices=('active','control'),required=True)
    parser.add_argument('--mode',choices=('legacy','corrected'),required=True)
    for name in ('root','output','checkpoint'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve()
    sys.path[:0]=[str(root/'work/Noesis-Sensory/gpu-pilot/src'),str(root/'work/Noesis-Core/src')]
    import s2_worker
    from noesis_gpu import learning
    previous=learning.LearningBrain;previous_validator=s2_worker.validate_training_report
    instances=[]
    if args.arm=='control':
        frozen=control_brain(previous)
        class Observed(frozen):
            def __init__(self,*a,**kw):
                super().__init__(*a,**kw);instances.append(self)
        learning.LearningBrain=Observed
        s2_worker.validate_training_report=validate_control_native
    context=dict(experiment='P0',arm=args.arm,mode=args.mode,seed=1004,
                 plasticity_enabled=args.arm=='active',exploration_enabled=True,
                 worker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 control_module_sha256=hashlib.sha256(Path(__file__).with_name('p0_control.py').read_bytes()).hexdigest())
    with args.output.with_name(args.output.name+'-p0-context.json').open('x',encoding='utf-8') as f:
        json.dump(context,f,indent=2)
    original_args=sys.argv.copy();success=False
    sys.argv=['s2_worker.py','--phase','train','--mode',args.mode,'--seed','1004',
              '--root',str(root),'--output',str(args.output),'--checkpoint',str(args.checkpoint)]
    try:
        s2_worker.main()
        if args.arm=='control':
            assert len(instances)==1 and instances[0].control_steps==4990 and instances[0].updates==0
            assert not instances[0].net.data.flags.writeable and instances[0].weights_unchanged()
        success=True
    finally:
        sys.argv=original_args;learning.LearningBrain=previous
        s2_worker.validate_training_report=previous_validator
        if args.arm=='control' and (args.output/'report.json').exists():
            path=args.output/'report.json';report=json.loads(path.read_text())
            report.update(p0_context=context,learning=False,plasticity_enabled=False,
                          exploration_enabled=True,exploration_active_rows=4990 if success else None,
                          learning_updates=0,network_weights_readonly=True,model_promoted=False,
                          native_status=report['status'],
                          status='completed_no_plasticity_control' if success else 'failed_no_plasticity_control')
            report.pop('latest_training_checkpoint',None)
            if success:
                source=args.output/'weights-last.npz';target=args.output/'weights-control.npz'
                assert not target.exists();source.rename(target)
                source.with_suffix('.model.json').unlink()
                report.update(control_checkpoint_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                              full_network_weights_unchanged=True,control_steps=instances[0].control_steps)
                target.with_suffix('.model.json').write_text(json.dumps({**report,
                    'context':'P0 unchanged-parent exposure evidence, NOT a learning birth/candidate/resume'},indent=2),encoding='utf-8')
            path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(experiment='P0',arm=args.arm,mode=args.mode,completed=success)))


if __name__=='__main__':main()
