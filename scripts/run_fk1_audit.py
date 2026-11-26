"""Exclusive120s Windows-contained geometry audit; never steps the physics clock."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import uuid

import psutil

root=Path(__file__).resolve().parents[1]
script=root/'work/audit_native_kinematics_fk1.py'
sys.path.insert(0,str(root/'work/Noesis-Sensory/gpu-pilot/src'))
from noesis_gpu.process_watchdog import run_windows_job, WatchdogError

me=psutil.Process();own={me.pid}|{p.pid for p in me.parents()}
for p in psutil.process_iter(['pid','name']):
    if p.info['pid'] not in own and (p.info['name'] or '').lower() in ('python.exe','pythonw.exe'):
        raise RuntimeError('Competing Python process; no audit started')
if psutil.virtual_memory().available<2*2**30:
    raise RuntimeError('Available RAM below2GiB')
assert not (root/'outputs/native-kinematics-fk1-20261106').exists()
lock=root/'work/Project-Noesis/gpu-pilot/outputs/codex-training-host.lock'
token=uuid.uuid4().hex
with os.fdopen(os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY),'w') as f:
    json.dump(dict(owner='codex',pid=os.getpid(),token=token,start_unix=time.time(),deadline_unix=time.time()+130),f)
env=dict(os.environ,PYTHONUTF8='1',MPLCONFIGDIR=str(root/'work/matplotlib-cache'))
receipt=dict(status='initializing',script_sha256=hashlib.sha256(script.read_bytes()).hexdigest())
empty=True
try:
    empty=False
    try:
        with (root/'outputs/native-kinematics-fk1-20261106.log').open('w',encoding='utf-8') as log:
            result=run_windows_job([sys.executable,str(script)],timeout=120,stdout=log,cwd=root,env=env)
        empty=result['tree_empty']
        receipt.update(result,status='completed_worker' if result['exit_code']==0 and not result['timed_out'] else 'failed_worker')
    except WatchdogError as error:
        empty=error.cleanup_verified
        receipt.update(status='watchdog_error',error=str(error))
        raise
finally:
    receipt['worker_tree_empty_verified']=empty
    if empty and lock.exists() and json.loads(lock.read_text())['token']==token:
        lock.unlink()
    receipt['lease_released']=not lock.exists()
    receipt['script_unchanged']=hashlib.sha256(script.read_bytes()).hexdigest()==receipt['script_sha256']
    (root/'outputs/native-kinematics-fk1-20261106-receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(receipt))
assert receipt['exit_code']==0 and not receipt['timed_out'] and empty and receipt['lease_released']
