"""Static compiled-body mapping inspection; no neural execution or integration."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import uuid
import numpy as np


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--worker',action='store_true');args=ap.parse_args()
    root=Path(__file__).resolve().parents[1];out=root/'outputs/body-telemetry-h2-20261106'
    sys.path[:0]=[str(root/'work/Noesis-Sensory/gpu-pilot/src'),str(root/'work/Noesis-Core/src')]
    if args.worker:
        import mujoco
        from noesis_gpu.backend import FlyGymBatch
        from h2_telemetry import ancestors
        body=FlyGymBatch(1,'cpu');m=body.sim.mj_model;d=body.sim.mj_data
        native=body.sim._internal_bodyids_by_fly[body.fly.name]
        mapping=list(body.fly.bodyseg_to_mjcfbody.items());missing=[]
        for (seg,element),idx in zip(mapping,native):
            if idx<0:missing.append(dict(segment=str(seg),name=element.name,compiled_id=int(idx)))
        valid=np.unique(native[native>0]);common=set(ancestors(m,int(valid[0])))
        for i in valid[1:]:common&=set(ancestors(m,int(i)))
        rootid=next(i for i in ancestors(m,int(valid[0])) if i in common)
        selected=[i for i in range(1,m.nbody) if rootid in ancestors(m,i)]
        # Compile a COPY with fusion disabled solely to verify why names vanished.
        spec=body.sim.world.mjcf_root.copy();original_fusion=bool(spec.compiler.fusestatic);spec.compiler.fusestatic=False
        unfused=spec.compile()
        restored=[mujoco.mj_name2id(unfused,mujoco.mjtObj.mjOBJ_BODY,x['name']) for x in missing]
        assert all(i>0 for i in restored) and original_fusion and d.time==0
        assert np.isclose(m.body_mass.sum(),unfused.body_mass.sum(),rtol=1e-12,atol=1e-15)
        assert np.isclose(m.body_mass[selected].sum(),m.body_mass.sum(),rtol=1e-12,atol=1e-15)
        critical=native[np.r_[body.thorax,body.feet]];assert np.all(critical>0) and len(set(critical.tolist()))==7
        result=dict(status='completed_static_body_mapping',original_fusestatic=original_fusion,
            native_body_mapping_count=len(native),missing_fused_names=missing,positive_unique_count=len(valid),
            critical_thorax_claw_ids=critical.tolist(),all_critical_observation_ids_valid=True,
            compiled_nbody=m.nbody,unfused_diagnostic_nbody=unfused.nbody,
            original_total_mass=float(m.body_mass.sum()),unfused_diagnostic_total_mass=float(unfused.body_mass.sum()),
            selected_subtree_mass=float(m.body_mass[selected].sum()),all_missing_names_restored_without_fusion=True,
            production_model_fusion_not_changed=bool(body.sim.world.mjcf_root.compiler.fusestatic),
            physical_integration_steps=0,neural_steps=0,original_model_time=float(d.time))
        (out/'static-body-map.json').write_text(json.dumps(result,indent=2),encoding='utf-8');return
    from noesis_gpu.process_watchdog import run_windows_job
    import psutil
    me=psutil.Process();excluded={me.pid,*[p.pid for p in me.parents()]}
    assert not any(p.pid not in excluded and (p.info['name'] or '').lower() in ('python.exe','pythonw.exe') for p in psutil.process_iter(['name']))
    lock=root/'work/Project-Noesis/gpu-pilot/outputs/codex-training-host.lock';token=uuid.uuid4().hex;empty=True
    with os.fdopen(os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY),'w') as f:json.dump(dict(owner='codex',pid=os.getpid(),token=token,deadline_unix=time.time()+180),f)
    try:
        empty=False
        with (out/'static-body-map.log').open('w',encoding='utf-8') as log:
            receipt=run_windows_job([sys.executable,str(Path(__file__).resolve()),'--worker'],timeout=180,stdout=log,cwd=root,env=dict(os.environ,PYTHONUTF8='1'))
        empty=receipt['tree_empty'];(out/'static-body-map-receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
        assert receipt['exit_code']==0 and not receipt['timed_out'] and empty
    finally:
        if empty and lock.exists() and json.loads(lock.read_text())['token']==token:lock.unlink()


if __name__=='__main__':main()
