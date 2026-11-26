"""Delegate unchanged neural frozen probe with P1world-force/telemetry wrapper."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
from p1_telemetry import instrumented_body,START,WIDTH,FRACTION,PHYSICS_DT


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--model',choices=('parent','candidate'),required=True)
    ap.add_argument('--condition',choices=('baseline','pulse'),required=True)
    for n in ('root','output','checkpoint'):ap.add_argument('--'+n,type=Path,required=True)
    a=ap.parse_args();root=a.root.resolve();enabled=a.condition=='pulse'
    sys.path[:0]=[str(root/'work/Noesis-Sensory/gpu-pilot/src'),str(root/'work/Noesis-Core/src')]
    import ap2_worker
    from noesis_gpu import run
    original=run.FlyGymBatch;instances=[];oldargs=sys.argv.copy()
    context=dict(experiment='P1',model=a.model,condition=a.condition,seed=10011,learning=False,motor_exploration=0,
        checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),worker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        environmental_pulse_only=True,neural_controller_unchanged=True,pulse_start_chunk=START,pulse_width_chunks=WIDTH,force_fraction=FRACTION)
    a.output.with_name(a.output.name+'-p1-context.json').write_text(json.dumps(context,indent=2),encoding='utf-8')
    run.FlyGymBatch=instrumented_body(original,instances,enabled)
    sys.argv=['ap2_worker.py','--phase','frozen','--bank',a.model,'--training-seed','1004','--strength','0','--seed','10011',
        '--root',str(root),'--output',str(a.output),'--checkpoint',str(a.checkpoint)]
    success=False
    try:
        ap2_worker.main();assert len(instances)==1;body=instances[0]
        assert body.p1_chunk==1000 and body.p1_count==50000 and len(body.p1_records)==1001
        force=body.p1_micro_force;expected=np.zeros_like(force)
        if enabled:expected[START*50:(START+WIDTH)*50,0]=FRACTION*body.p1_weight
        np.testing.assert_array_equal(force,expected)
        np.testing.assert_allclose(body.p1_micro_time,np.arange(50000)*PHYSICS_DT,rtol=0,atol=1e-10)
        success=True
    finally:
        run.FlyGymBatch=original;sys.argv=oldargs
        if instances and a.output.exists():
            body=instances[0]
            if body.p1_records:np.savez_compressed(a.output/'p1-telemetry.npz',**{k:np.stack([r[k] for r in body.p1_records]) for k in body.p1_records[0]})
            if body.p1_contacts:np.savez_compressed(a.output/'p1-ground-contacts.npz',**{k:np.asarray([c[k] for c in body.p1_contacts]) for k in body.p1_contacts[0]})
            np.savez_compressed(a.output/'p1-microforce.npz',time=body.p1_micro_time[:body.p1_count],force=body.p1_micro_force[:body.p1_count])
            (a.output/'p1-model.json').write_text(json.dumps(body.p1_meta,indent=2),encoding='utf-8')
        path=a.output/'report.json'
        if path.exists():
            report=json.loads(path.read_text());report.update(p1_context=context,p1_worker_completed=success,calibration_not_capability_validation=True)
            path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(experiment='P1',completed=success,model=a.model,condition=a.condition)))


if __name__=='__main__':main()
