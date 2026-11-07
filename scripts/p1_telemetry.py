"""P1 environmental pulse and read-only full telemetry; no motor policy or learning."""
import numpy as np
from h2_telemetry import mass_layout,snapshot

START=393
WIDTH=4
CONTROL_DT=.005
PHYSICS_DT=.0001
FRACTION=.1


def pulse_force(chunk,weight,enabled):
    if isinstance(chunk,bool) or int(chunk)!=chunk or chunk<0 or not np.isfinite(weight) or weight<=0 or not isinstance(enabled,bool):raise ValueError('Invalid fixed pulse context')
    return np.array([FRACTION*weight if enabled and START<=chunk<START+WIDTH else 0.,0.,0.])


def dynamics(model,data):
    """Read-only generalized-force checks after native observe has called mj_forward."""
    import mujoco
    dense=np.empty((model.nv,model.nv));mujoco.mj_fullM(model,dense,data.qM)
    external=np.zeros(model.nv)
    for body in np.flatnonzero(np.any(data.xfrc_applied!=0,axis=1)):
        applied=data.xfrc_applied[body]
        mujoco.mj_applyFT(model,data,applied[:3],applied[3:],data.xipos[body],int(body),external)
    smooth=data.qfrc_passive-data.qfrc_bias+data.qfrc_actuator+data.qfrc_applied+external
    inertial=dense@data.qacc;rhs=data.qfrc_smooth+data.qfrc_constraint
    smooth_error=float(np.max(np.abs(smooth-data.qfrc_smooth)));dynamic_error=float(np.max(np.abs(inertial-rhs)))
    scale=max(1.,float(np.max(np.abs(inertial))),float(np.max(np.abs(rhs))),float(np.max(np.abs(smooth))))
    if max(smooth_error,dynamic_error)>1e-9*scale+1e-10:raise ValueError('Generalized force/mass closure mismatch')
    return dict(qfrc_bias=data.qfrc_bias.copy(),qfrc_passive=data.qfrc_passive.copy(),qfrc_smooth=data.qfrc_smooth.copy(),
        qfrc_applied=data.qfrc_applied.copy(),qfrc_constraint=data.qfrc_constraint.copy(),qfrc_xfrc=external,
        mass_times_qacc=inertial,smooth_force_max_error=np.array(smooth_error),dynamic_force_max_error=np.array(dynamic_error),dynamic_force_scale=np.array(scale))


def instrumented_body(original,instances,enabled):
    class Instrumented(original):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            if self.backend!='cpu' or self.n_worlds!=1:raise ValueError('P1one CPUworld only')
            import mujoco
            m,d=self.sim.mj_model,self.sim.mj_data;native=self.sim._internal_bodyids_by_fly[self.fly.name]
            self.p1_layout=mass_layout(m,native);self.p1_target=int(native[self.thorax]);self.p1_claws=native[self.feet].copy()
            assert self.p1_target>0 and np.all(self.p1_claws>0) and np.isclose(m.opt.timestep,PHYSICS_DT,rtol=0,atol=1e-15)
            assert d.time==0 and np.all(d.xfrc_applied==0)
            self.p1_ground=self.sim._internal_ground_geom_ids.copy();self.p1_weight=self.p1_layout['total_mass']*abs(float(m.opt.gravity[2]))
            self.p1_expected=np.zeros_like(d.xfrc_applied);self.p1_chunk=0;self.p1_count=0;self.p1_last_force=np.zeros(3)
            self.p1_micro_time=np.empty(50000);self.p1_micro_force=np.empty((50000,3));self.p1_records=[];self.p1_contacts=[]
            ids=self.p1_layout['body_ids']
            self.p1_meta=dict(body_ids=ids.tolist(),body_names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,int(i)) for i in ids],
                body_mass=m.body_mass[ids].tolist(),total_mass=self.p1_layout['total_mass'],model_total_mass=float(m.body_mass.sum()),
                root_body=self.p1_layout['root'],free_dof=self.p1_layout['free_dof'],nbody=m.nbody,nq=m.nq,nv=m.nv,
                gravity=m.opt.gravity.tolist(),pulse_enabled=enabled,pulse_target_body=self.p1_target,pulse_target_name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,self.p1_target),
                force_frame='world',force_application_point='target body center of mass',pulse_start_chunk=START,pulse_width_chunks=WIDTH,
                control_dt=CONTROL_DT,physics_dt=PHYSICS_DT,force_fraction=FRACTION,whole_fly_weight=self.p1_weight,
                production_static_fusion=bool(self.sim.world.mjcf_root.compiler.fusestatic),
                force_epoch='microforce sampled immediately BEFORE every mj_step; observed force is that used in preceding control chunk, held until next advance',
                generalized_force_equation='M*qacc=qfrc_smooth+qfrc_constraint; smooth=passive-bias+actuator+qfrc_applied+mapped_xfrc',
                contact_constraint_equation='ground force closes free-root TRANSLATIONAL qfrc_constraint; external pulse is separate, never added to this equality',
                no_new_gate=True,physical_balance_validated=False)
            step=self.sim.step
            def checked_step():
                if self.p1_count>=50000 or not np.array_equal(d.xfrc_applied,self.p1_expected):raise ValueError('Unexpected external force or physics count')
                self.p1_micro_time[self.p1_count]=d.time;self.p1_micro_force[self.p1_count]=d.xfrc_applied[self.p1_target,:3]
                step()
                if not np.array_equal(d.xfrc_applied,self.p1_expected):raise ValueError('External force overwritten during mj_step')
                self.p1_count+=1
            self.sim.step=checked_step;instances.append(self)

        def advance(self,commands,adhesion,steps):
            if steps!=50 or self.p1_count!=self.p1_chunk*50:raise ValueError('Wrong P1control/physics schedule')
            d=self.sim.mj_data
            if not np.array_equal(d.xfrc_applied,self.p1_expected):raise ValueError('Unexpected between-chunk external force')
            force=pulse_force(self.p1_chunk,self.p1_weight,enabled);self.p1_expected[:]=0;self.p1_expected[self.p1_target,:3]=force
            d.xfrc_applied[:]=self.p1_expected;self.p1_last_force=force.copy()
            super().advance(commands,adhesion,steps);self.p1_chunk+=1
            if self.p1_count!=self.p1_chunk*50:raise ValueError('Missing physics steps')

        def observe(self):
            obs=super().observe()
            if not hasattr(self,'p1_records'):return obs
            m,d=self.sim.mj_model,self.sim.mj_data;clock=d.time
            before=[x.copy() for x in (d.qpos,d.qvel,d.qacc,d.ctrl,d.sensordata,d.xfrc_applied,d.qfrc_applied)]
            state,contacts=snapshot(m,d,self.p1_layout,self.p1_ground,self.p1_claws)
            state.update(dynamics(m,d),force_used_preceding_chunk=self.p1_last_force.copy(),completed_control_chunks=np.array(self.p1_chunk))
            for actual,expected in zip((d.qpos,d.qvel,d.qacc,d.ctrl,d.sensordata,d.xfrc_applied,d.qfrc_applied),before):np.testing.assert_array_equal(actual,expected)
            assert d.time==clock
            for c in contacts:c['sample_index']=len(self.p1_records)
            self.p1_contacts.extend(contacts);self.p1_records.append(state)
            return obs
    return Instrumented
