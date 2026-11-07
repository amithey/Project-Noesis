"""Frozen P1 body/pulse worker with own-baseline sensory history at neural input."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from p2_sensory_history import history_population,FIELDS


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',choices=('parent','candidate'),required=True)
    ap.add_argument('--condition',choices=('baseline','pulse'),required=True)
    for n in ('root','output','checkpoint','history'):ap.add_argument('--'+n,type=Path,required=True)
    a=ap.parse_args();root=a.root.resolve();sys.path[:0]=[str(root/'work/Noesis-Sensory/gpu-pilot/src'),str(root/'work/Noesis-Core/src')]
    from noesis_gpu import controller
    import p1_worker
    from run_physical_p1 import sha
    original=controller.FrozenPopulation;instances=[];oldargs=sys.argv.copy()
    with np.load(a.history,allow_pickle=False) as z:history={k:z[k].copy() for k in FIELDS}
    context=dict(experiment='P2',model=a.model,condition=a.condition,history_sha256=sha(a.history),checkpoint_sha256=sha(a.checkpoint),
        source_sha256=sha(Path(__file__)),intervention='own unperturbed sensory history only',fields=list(FIELDS),
        input_timing='chunk0 live pristine input; chunkk>0 baseline post-step rowk-1',motor_output_replayed=False,
        live_body_measurement_preserved=True,learning=False,diagnostic_not_ability_validation=True)
    a.output.with_name(a.output.name+'-p2-context.json').write_text(json.dumps(context,indent=2),encoding='utf-8')
    controller.FrozenPopulation=history_population(original,history,instances)
    sys.argv=['p1_worker.py','--model',a.model,'--condition',a.condition,'--root',str(root),'--output',str(a.output),'--checkpoint',str(a.checkpoint)]
    success=False
    try:
        p1_worker.main();assert len(instances)==1 and len(instances[0].p2_records)==1000;success=True
    finally:
        controller.FrozenPopulation=original;sys.argv=oldargs
        if instances and a.output.exists() and instances[0].p2_records:
            rows=instances[0].p2_records
            np.savez_compressed(a.output/'p2-neural-input-output.npz',**{k:np.stack([r[k] for r in rows]) for k in rows[0]})
        path=a.output/'report.json'
        if path.exists():
            report=json.loads(path.read_text());report.update(p2_context=context,p2_worker_completed=success)
            path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(experiment='P2',completed=success,model=a.model,condition=a.condition)))


if __name__=='__main__':main()
