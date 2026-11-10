"""WS1 observes bounded-weight frozen neural/body sensitivity; no learning."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from ws1_import import checked_import


def main():
    ap = argparse.ArgumentParser()
    for name in ('root', 'output', 'checkpoint'):
        ap.add_argument('--'+name, type=Path, required=True)
    ap.add_argument('--seed', type=int, choices=(1004,10006,10007,10010,10011), required=True)
    ap.add_argument('--variant', choices=('baseline','plus','minus'), required=True)
    a = ap.parse_args()
    root = a.root.resolve()
    sys.path[:0] = [str(root/'work/Noesis-Sensory/gpu-pilot/src'), str(root/'work/Noesis-Core/src')]
    from noesis_gpu import controller
    import ps1_worker
    original_import, argv = controller.import_checkpoint, sys.argv
    records = []

    def observed_import(net, path):
        assert Path(path).resolve() == a.checkpoint.resolve()
        records.append(checked_import(original_import, net, path))

    context = dict(experiment='WS1', variant=a.variant, seed=a.seed,
                   constructed_not_trained=True, learning=False,
                   checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
                   frozen_native_neural_motor_execution=True,
                   ps1_context_role='Delegated frozen instrumentation only; WS1 context is authoritative')
    success = False
    controller.import_checkpoint = observed_import
    sys.argv = ['ps1_worker', '--phase','frozen','--bank','candidate','--seed',str(a.seed),
                '--root',str(root),'--output',str(a.output),'--checkpoint',str(a.checkpoint)]
    try:
        ps1_worker.main()
        assert len(records) == 1 and records[0]['actual_import_exact']
        report = json.loads((a.output/'report.json').read_text())
        assert not report['learning'] and report['weights_unchanged'] and report['ps1_worker_completed']
        assert report['checkpoint_sha256'] == context['checkpoint_sha256']
        success = True
    finally:
        controller.import_checkpoint, sys.argv = original_import, argv
        context.update(worker_completed=success, import_records=records)
        a.output.with_name(a.output.name+'-ws1-context.json').write_text(json.dumps(context,indent=2),encoding='utf-8')
        path = a.output/'report.json'
        if path.exists():
            report = json.loads(path.read_text())
            report.update(ws1_context=context, ws1_worker_completed=success)
            path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')


if __name__ == '__main__':
    main()
