"""Bounded static measurement checks, kept separate from all physical replay allocations."""
import json,os,sys,time,uuid
from pathlib import Path
root=Path(__file__).resolve().parents[1];out=root/'outputs/body-telemetry-h2b-static-20261106'
sys.path[:0]=[str(root/'work/Noesis-Sensory/gpu-pilot/src')]
from noesis_gpu.process_watchdog import run_windows_job,WatchdogError
out.mkdir(exist_ok=False);lock=root/'work/Project-Noesis/gpu-pilot/outputs/codex-training-host.lock';token=uuid.uuid4().hex;empty=True
with os.fdopen(os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY),'w') as f:json.dump(dict(owner='codex',pid=os.getpid(),token=token,deadline_unix=time.time()+180),f)
try:
    empty=False
    env=dict(os.environ,PYTHONUTF8='1',PYTHONPATH=os.pathsep.join(str(root/p) for p in ('work','work/Noesis-Sensory/gpu-pilot/src','work/Noesis-Core/src')),
        H2_STATIC_RESULT=str(out/'native-static-measurement.json'))
    with (out/'tests.log').open('w',encoding='utf-8') as log:
        receipt=run_windows_job([sys.executable,'-m','pytest',str(root/'work/test_h2_telemetry.py'),'-q'],timeout=180,stdout=log,cwd=root,env=env)
    empty=receipt['tree_empty'];(out/'receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    assert receipt['exit_code']==0 and empty and not receipt['timed_out']
except WatchdogError as error:
    empty=error.cleanup_verified;raise
finally:
    if empty and lock.exists() and json.loads(lock.read_text())['token']==token:lock.unlink()
    (out/'cleanup.json').write_text(json.dumps(dict(tree_empty=empty,lease_released=not lock.exists())),encoding='utf-8')
