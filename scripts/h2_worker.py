"""Delegate unchanged S2 frozen worker with read-only post-observation telemetry."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from h2_telemetry import mass_layout,snapshot


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--seed',type=int,required=True)
    for n in ('root','checkpoint','output'):ap.add_argument('--'+n,type=Path,required=True)
    args=ap.parse_args();root=args.root.resolve()
    if args.seed not in (10006,10007):ap.error('Only registered diagnostic seeds')
    sys.path[:0]=[str(root/'work/Noesis-Sensory/gpu-pilot/src'),str(root/'work/Noesis-Core/src')]
    import s2_worker
    from noesis_gpu import run
    import mujoco
    original=run.FlyGymBatch;instances=[]
    context=dict(experiment='H2',seed=args.seed,checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        worker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),read_only_instrumentation=True)
    args.output.with_name(args.output.name+'-h2-context.json').write_text(json.dumps(context,indent=2),encoding='utf-8')

    class Instrumented(original):
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw);self.h2_records=[];self.h2_contacts=[];instances.append(self)
            m=self.sim.mj_model;native=self.sim._internal_bodyids_by_fly[self.fly.name]
            self.h2_layout=mass_layout(m,native)
            self.h2_ground=self.sim._internal_ground_geom_ids.copy()
            self.h2_claws=native[self.feet].copy()
            self.h2_claw_segments=[self.fly.get_bodysegs_order()[i] for i in self.feet]
            ids=self.h2_layout['body_ids']
            self.h2_meta=dict(root_body=self.h2_layout['root'],body_ids=ids.tolist(),native_segment_body_ids=native.tolist(),
                body_names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,int(i)) for i in ids],
                body_mass=m.body_mass[ids].tolist(),total_mass=self.h2_layout['total_mass'],
                model_total_mass=float(m.body_mass.sum()),gravity=m.opt.gravity.tolist(),free_dof=self.h2_layout['free_dof'],
                nq=m.nq,nv=m.nv,nbody=m.nbody,claw_body_ids=self.h2_claws.tolist(),ground_geom_ids=self.h2_ground.tolist(),
                joint_names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,i) for i in range(m.njnt)],
                joint_types=m.jnt_type.tolist(),joint_qpos_addresses=m.jnt_qposadr.tolist(),joint_dof_addresses=m.jnt_dofadr.tolist(),
                actuator_names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_ACTUATOR,i) for i in range(m.nu)],
                actuator_transmission_types=m.actuator_trntype.tolist(),actuator_transmission_ids=m.actuator_trnid.tolist(),
                all_positive_model_mass_in_fly_subtree=True,contact_force_on_geom2='frame.T@contact_wrench[:3]',
                root_actuator_force='free-joint translation qfrc_actuator includes external adhesion; joint servos are internal',
                whole_com_velocity='subtree COM Jacobian times qvel, world frame',
                legacy_native_observations_not_modified=True)

        def observe(self):
            obs=super().observe()
            if not hasattr(self,'h2_records'):return obs
            m,d=self.sim.mj_model,self.sim.mj_data
            before=[d.qpos.copy(),d.qvel.copy(),d.qacc.copy(),d.ctrl.copy(),d.sensordata.copy()];clock=d.time
            state,contacts=snapshot(m,d,self.h2_layout,self.h2_ground,self.h2_claws)
            upstream=self.sim.get_bodysegment_contact_forces(self.fly.name,self.h2_claw_segments,ground_only=True)
            np.testing.assert_allclose(state['claw_ground_force'],upstream,rtol=1e-10,atol=1e-10)
            state['upstream_force_max_error']=np.array(np.max(np.abs(state['claw_ground_force']-upstream)))
            for actual,expected in zip((d.qpos,d.qvel,d.qacc,d.ctrl,d.sensordata),before):np.testing.assert_array_equal(actual,expected)
            assert d.time==clock
            for c in contacts:c['sample_index']=len(self.h2_records)
            self.h2_contacts.extend(contacts);self.h2_records.append(state)
            return obs

    run.FlyGymBatch=Instrumented;oldargs=sys.argv.copy()
    sys.argv=['s2_worker.py','--phase','frozen','--mode','legacy','--seed',str(args.seed),
        '--root',str(root),'--checkpoint',str(args.checkpoint),'--output',str(args.output)]
    success=False
    try:s2_worker.main();success=True
    finally:
        run.FlyGymBatch=original;sys.argv=oldargs
        if instances:
            body=instances[0]
            args.output.mkdir(exist_ok=True)
            if body.h2_records:np.savez_compressed(args.output/'body-telemetry.npz',**{k:np.stack([r[k] for r in body.h2_records]) for k in body.h2_records[0]})
            contacts=body.h2_contacts
            if contacts:np.savez_compressed(args.output/'ground-contacts.npz',**{k:np.asarray([c[k] for c in contacts]) for k in contacts[0]})
            (args.output/'telemetry-model.json').write_text(json.dumps(body.h2_meta,indent=2),encoding='utf-8')
    assert success and len(instances)==1 and len(instances[0].h2_records)==1001
    report=json.loads((args.output/'report.json').read_text());report.update(h2_context=context,telemetry_samples=1001,read_only_telemetry=True)
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(experiment='H2',completed=True,samples=1001)))


if __name__=='__main__':main()
