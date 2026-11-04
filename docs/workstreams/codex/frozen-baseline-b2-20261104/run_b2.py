"""B2: two bounded sequential frozen parent runs, followed by historical raw-array audit."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
import numpy as np
import psutil

root = Path(__file__).resolve().parents[1]
pilot = root / "work/Project-Noesis"
core = root / "work/Noesis-Core/src"
data = root / "work/NoesisData"
parent = root / "work/evidence/s1-restored/gpu-pilot/outputs/e4-exploration-20261029/train-control-1000/weights-last.npz"
historical = root / "work/evidence/s1-restored/gpu-pilot/outputs/standing-profile-s1-20261101"
out = root / "outputs/frozen-baseline-b2-20261104"
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def tree_digest(path):
    h = hashlib.sha256()
    for p in sorted(path.rglob("*.py")):
        h.update(p.relative_to(path).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()
def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")
expected_parent = "9fd7d41177bf2c745809401c3b142a832661d1a18d0b117b30cb02aea63fea3c"
assert digest(parent) == expected_parent
sources = {"core":tree_digest(core), "pilot":tree_digest(pilot/"gpu-pilot/src")}
ancestors = {p.pid for p in psutil.Process().parents()}
for p in psutil.process_iter(["pid", "name"]):
    if p.info["pid"] not in ancestors | {os.getpid()} and (p.info["name"] or "").lower() in ("python.exe","pythonw.exe"):
        raise RuntimeError("Competing Python process; no simulation allocated")
out.mkdir(exist_ok=False)
lock = pilot / "gpu-pilot/outputs/codex-training-host.lock"
lock.parent.mkdir(parents=True,exist_ok=True)
token = uuid.uuid4().hex
started = time.monotonic()
protocol = dict(experiment="B2 second-host frozen baseline qualification", owner="codex",host=os.environ.get("COMPUTERNAME"),
                pid=os.getpid(), token=token, start_unix=time.time(), deadline_unix=time.time()+1200,
                seeds=[10006,10007], parent_sha256=expected_parent, source_hashes=sources,
                process_budget_seconds=600, aggregate_budget_seconds=1200,
                simulation_seconds=5, control_ms=5, backend="cpu", worlds=1, learning=False,
                motor_exploration=0, stop_on="technical failure, incomplete horizon, source/parent mutation, nonfinite arrays",
                no_model_or_stage_promotion=True)
with os.fdopen(os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL),"w") as f:
    json.dump(protocol,f)
write(out/"protocol.json",protocol)
env=dict(os.environ,PYTHONPATH=str(pilot/"gpu-pilot/src")+os.pathsep+str(core),
         NOESIS_SOURCE=str(core),PYTHONUTF8="1",MPLCONFIGDIR=str(root/"work/matplotlib-cache"))
sys.path[:0]=[str(pilot/"gpu-pilot/src"),str(core)]
from noesis_gpu.evaluate import summarize
result=dict(status="running", cases=[], walking_proven=False,model_promoted=False)
try:
    for seed in protocol["seeds"]:
        if digest(parent)!=expected_parent or sources!={"core":tree_digest(core),"pilot":tree_digest(pilot/"gpu-pilot/src")}:
            raise RuntimeError("Parent or source changed")
        ram=psutil.virtual_memory().available/2**30
        if ram<2: raise RuntimeError("Less than 2GiB free RAM")
        remaining=1200-(time.monotonic()-started)
        if remaining<600: raise RuntimeError("Insufficient remaining per-job budget; no restart")
        target=out/f"parent-{seed}"
        command=[sys.executable,"-m","noesis_gpu.run","--backend","cpu","--worlds","1","--seconds","5",
                 "--wall-seconds","600","--seed",str(seed),"--noesis-source",str(core),"--data",str(data),
                 "--checkpoint",str(parent),"--output",str(target)]
        print("START",seed,flush=True)
        job_start=time.monotonic()
        with (out/f"parent-{seed}.log").open("w",encoding="utf-8") as log:
            completed=subprocess.run(command,env=env,cwd=pilot,stdout=log,stderr=subprocess.STDOUT,timeout=600)
        if completed.returncode!=0: raise RuntimeError(f"Seed {seed} exit {completed.returncode}")
        report,summary=summarize(target)
        assert report["weights_unchanged"] and report["completed_control_chunks"]==1000
        raw_a=np.load(target/"observations.npz",allow_pickle=False)
        raw_b=np.load(historical/f"frozen-parent-{seed}"/"observations.npz",allow_pickle=False)
        arrays={}
        assert set(raw_a.files)==set(raw_b.files)
        for key in raw_a.files:
            a,b=raw_a[key],raw_b[key]
            if np.issubdtype(a.dtype,np.number):
                if not np.isfinite(a).all():raise RuntimeError("Nonfinite "+key)
                arrays[key]={"shape":list(a.shape),"identical":bool(np.array_equal(a,b)),
                             "max_abs_difference":float(np.abs(a-b).max())}
        old_report,old_summary=summarize(historical/f"frozen-parent-{seed}")
        row=dict(seed=seed,exit_code=completed.returncode,wall_seconds=time.monotonic()-job_start,
                 available_ram_before_gib=ram,summary=summary,historical_summary=old_summary,
                 raw_comparison=arrays,all_raw_arrays_identical=all(v["identical"] for v in arrays.values()),
                 packages=report["packages"],historical_packages=old_report["packages"],
                 report_sha256=digest(target/"report.json"),observations_sha256=digest(target/"observations.npz"))
        result["cases"].append(row)
        write(out/f"parent-{seed}-audit.json",row)
        print("COMPLETE",seed,"standing",summary["standing_fraction"],"historical_exact",row["all_raw_arrays_identical"],flush=True)
        write(out/"result.json",result)
    result["status"]="completed_frozen_baseline_qualification"
except Exception as error:
    result.update(status="failed",error_type=type(error).__name__,error=str(error))
    raise
finally:
    result["wall_seconds"]=time.monotonic()-started
    result["parent_unchanged"]=digest(parent)==expected_parent
    result["source_unchanged"]=sources=={"core":tree_digest(core),"pilot":tree_digest(pilot/"gpu-pilot/src")}
    if lock.exists() and json.loads(lock.read_text(encoding="utf-8"))["token"]==token:lock.unlink()
    result["lease_released"]=not lock.exists()
    write(out/"result.json",result)
    print("RESULT",result["status"],"seconds",result["wall_seconds"],flush=True)
