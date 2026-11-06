"""TP1: bounded frozen thermosensory integration pilot, gated before active stimuli."""
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

root=Path(__file__).resolve().parents[1]
pilot=root/"work/Noesis-Sensory"
core=root/"work/Noesis-Core/src"
data=root/"work/NoesisData"
parent=root/"work/evidence/s1-restored/gpu-pilot/outputs/e4-exploration-20261029/train-control-1000/weights-last.npz"
baseline=root/"outputs/frozen-baseline-b2-20261104"
out=root/"outputs/thermo-pilot-tp1-20261106"
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def tree(path):
    h=hashlib.sha256()
    for p in sorted(path.rglob("*.py")):
        h.update(p.relative_to(path).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()
def write(p,obj):
    p.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding="utf-8")
def exact(a,b):
    with np.load(a,allow_pickle=False) as x,np.load(b,allow_pickle=False) as y:
        return set(x.files)==set(y.files) and all(np.array_equal(x[k],y[k]) for k in x.files)
expected="9fd7d41177bf2c745809401c3b142a832661d1a18d0b117b30cb02aea63fea3c"
assert sha(parent)==expected
manifest=json.loads((root/"outputs/noesis-required-data-sha256.json").read_text(encoding="utf-8"))
for entry in manifest["files"]:
    assert sha(data/entry["path"])==entry["sha256"]
hashes={"pilot":tree(pilot/"gpu-pilot/src"),"core":tree(core)}
protocol=dict(experiment="TP1 frozen thermo integration pilot",date="2026-11-06",
              source_base="75cdc34aeeacb7684c73b6e91bd39fddf399cc9f",
              peer_source="d2160a9714dfe279d928c28aa323e78dbe700635",
              core_commit="42c20769376bda5694b4c1c66ca75e52cbd9f1f4",source_hashes=hashes,
              parent_sha256=expected,seeds=[10006,10007],modes=["plain","off","zero","constant","active"],
              seconds_per_job=5,control_ms=5,process_budget_seconds=300,aggregate_budget_seconds=1800,
              field=dict(axis=0,origin=-1.,length=2.,constant_level=.5,gain_hz_per_unit=50.),
              gate0="plain exactly replays B2; off and zero exactly match plain in native AND sensory arrays before stimuli",
              neural_criterion="constant and active warm-afferent mean rate exceed matched off by>=5Hz/neuron on BOTH seeds",
              retention_criterion="all constant/active probes standing>=.9, standing>=off and slip<=off; descriptive pilot only",
              stop_conditions="failure, timeout, incomplete/nonfinite evidence, invariance/source/parent mismatch, RAM<2GiB or insufficient remaining300s",
              learning=False,motor_exploration=0,model_promoted=False,walking_proven=False)
ancestors={p.pid for p in psutil.Process().parents()}
for p in psutil.process_iter(["pid","name"]):
    if p.info["pid"] not in ancestors|{os.getpid()} and (p.info["name"] or "").lower() in ("python.exe","pythonw.exe"):
        raise RuntimeError("Competing Python process; no simulation launched")
out.mkdir(exist_ok=False)
lock=root/"work/Project-Noesis/gpu-pilot/outputs/codex-training-host.lock"
token=uuid.uuid4().hex
lease=dict(owner="codex",host=os.environ.get("COMPUTERNAME"),pid=os.getpid(),token=token,
           start_unix=time.time(),deadline_unix=time.time()+1800)
with os.fdopen(os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL),"w") as f:
    json.dump(lease,f)
write(out/"protocol.json",protocol)  # public-safe scientific protocol; private lease stays separate
env=dict(os.environ,PYTHONPATH=str(pilot/"gpu-pilot/src"),NOESIS_SOURCE=str(core),
         PYTHONUTF8="1",MPLCONFIGDIR=str(root/"work/matplotlib-cache"))
sys.path[:0]=[str(pilot/"gpu-pilot/src"),str(core)]
from noesis_gpu.evaluate import summarize
started=time.monotonic()
result=dict(status="running",cases=[],gate0_by_seed={},model_promoted=False,walking_proven=False)
try:
    for seed in protocol["seeds"]:
        rows={}
        for mode in protocol["modes"]:
            if sha(parent)!=expected or hashes!={"pilot":tree(pilot/"gpu-pilot/src"),"core":tree(core)}:
                raise RuntimeError("Source or parent changed")
            if time.monotonic()-started>1500:
                raise RuntimeError("Remaining aggregate budget below300seconds; no job started")
            ram=psutil.virtual_memory().available/2**30
            if ram<2: raise RuntimeError("Insufficient available RAM")
            for p in psutil.process_iter(["pid","name"]):
                if p.info["pid"] not in ancestors|{os.getpid()} and (p.info["name"] or "").lower() in ("python.exe","pythonw.exe"):
                    raise RuntimeError("Competing Python process before job")
            target=out/f"{mode}-{seed}"
            command=[sys.executable,"-m","noesis_gpu.sensory_probe","--mode",mode,"--seed",str(seed),
                     "--core",str(core),"--data",str(data),"--checkpoint",str(parent),"--output",str(target)]
            print("START",seed,mode,flush=True)
            begin=time.monotonic()
            with (out/f"{mode}-{seed}.log").open("w",encoding="utf-8") as log:
                done=subprocess.run(command,cwd=pilot,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=300)
            if done.returncode!=0:raise RuntimeError(f"Worker {mode}/{seed} exit{done.returncode}")
            report,summary=summarize(target)
            assert report["weights_unchanged"] and report["completed_control_chunks"]==1000
            assert report["checkpoint_sha256"]==expected and report["sensory_mode"]==mode
            with np.load(target/"sensory-observations.npz",allow_pickle=False) as z:
                a={k:z[k].copy() for k in z.files}
            assert all(len(v)==1000 and np.isfinite(v).all() for v in a.values())
            if mode in ("plain","off","zero"):
                assert not a["u"].any() and not a["warm_requested_hz"].any() and not a["cool_requested_hz"].any()
            elif mode=="constant":
                np.testing.assert_array_equal(a["u"],np.full(1000,.5))
                np.testing.assert_array_equal(a["warm_requested_hz"],np.full(1000,25.))
                assert not a["cool_requested_hz"].any()
            else:
                np.testing.assert_allclose(a["u"],np.clip((a["input_position"][:,0]+1.)/2.,-1.,1.),rtol=0,atol=1e-14)
                np.testing.assert_allclose(a["warm_requested_hz"],50*np.maximum(a["u"],0),rtol=0,atol=1e-12)
                np.testing.assert_allclose(a["cool_requested_hz"],50*np.maximum(-a["u"],0),rtol=0,atol=1e-12)
            if mode=="plain":
                assert exact(target/"observations.npz",baseline/f"parent-{seed}"/"observations.npz"),"Observer/default changed B2"
            elif mode in ("off","zero"):
                plain=out/f"plain-{seed}"
                assert exact(target/"observations.npz",plain/"observations.npz"),"Native invariance failed"
                assert exact(target/"sensory-observations.npz",plain/"sensory-observations.npz"),"Observer invariance failed"
            if mode=="zero":
                result["gate0_by_seed"][str(seed)]=True
            row=dict(seed=seed,mode=mode,summary=summary,
                     warm_rate_hz_per_neuron=float(a["warm_spikes"][10:].sum()/(6*4.95)),
                     cool_rate_hz_per_neuron=float(a["cool_spikes"][10:].sum()/(6*4.95)),
                     requested_warm_hz_mean=float(a["warm_requested_hz"][10:].mean()),
                     requested_cool_hz_mean=float(a["cool_requested_hz"][10:].mean()),
                     u_min=float(a["u"].min()),u_max=float(a["u"].max()),
                     process_seconds=time.monotonic()-begin,available_ram_before_gib=ram,
                     observations_sha256=sha(target/"observations.npz"),
                     sensory_sha256=sha(target/"sensory-observations.npz"),report_sha256=sha(target/"report.json"))
            rows[mode]=row
            result["cases"].append(row)
            write(out/"result.json",result)
            print("COMPLETE",seed,mode,"standing",summary["standing_fraction"],"warm_hz",row["warm_rate_hz_per_neuron"],flush=True)
        for mode in ("constant","active"):
            r,o=rows[mode],rows["off"]
            r["warm_rate_gain_vs_off"]=r["warm_rate_hz_per_neuron"]-o["warm_rate_hz_per_neuron"]
            r["neural_criterion_pass"]=r["warm_rate_gain_vs_off"]>=5.
            r["retention_criterion_pass"]=bool(r["summary"]["standing_fraction"]>=.9 and
                r["summary"]["standing_fraction"]>=o["summary"]["standing_fraction"] and
                r["summary"]["mean_stance_slip_mm_s"]<=o["summary"]["mean_stance_slip_mm_s"])
    result["status"]="completed_thermo_integration_pilot"
    active=[r for r in result["cases"] if r["mode"] in ("constant","active")]
    result["all_neural_criteria_pass"]=all(r["neural_criterion_pass"] for r in active)
    result["all_retention_criteria_pass"]=all(r["retention_criterion_pass"] for r in active)
except Exception as error:
    result.update(status="failed",error_type=type(error).__name__,error=str(error))
    raise
finally:
    result["seconds"]=time.monotonic()-started
    result["parent_unchanged"]=sha(parent)==expected
    result["source_unchanged"]=hashes=={"pilot":tree(pilot/"gpu-pilot/src"),"core":tree(core)}
    if lock.exists() and json.loads(lock.read_text(encoding="utf-8"))["token"]==token:lock.unlink()
    result["lease_released"]=not lock.exists()
    write(out/"result.json",result)
    print("RESULT",result["status"],"seconds",result["seconds"],flush=True)
