"""Read-only native-body observer shared by disabled control/treatment/frozen probes."""
import numpy as np
from h2_telemetry import mass_layout,snapshot
from support_descriptor_c0 import describe_support,REGIONS
from ap1_feedback import non_tarsal_fraction

MODES=('unsupported','central_body_contact','proximal_leg_contact','tarsal_only','indeterminate_low_force')


def observing_body(original,instances):
    class AnatomyBody(original):
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw)
            m=self.sim.mj_model;native=self.sim._internal_bodyids_by_fly[self.fly.name]
            self.ap1_layout=mass_layout(m,native);critical=native[np.r_[self.thorax,self.feet]]
            assert np.all(critical>0) and len(set(critical.tolist()))==7
            import mujoco
            self.ap1_names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,i) for i in range(m.nbody)]
            # C0 classifies loaded contact bodies only. Unloaded eye/antenna
            # bodies still contribute to whole-fly mass/COM; a loaded unknown
            # remains a hard error in describe_support, never an omitted force.
            self.ap1_ground=self.sim._internal_ground_geom_ids.copy();self.ap1_claws=native[self.feet].copy()
            self.ap1_segments=[self.fly.get_bodysegs_order()[i] for i in self.feet]
            self.ap1_records=[];self.ap1_latest_position=None
            self.ap1_metadata=dict(body_names=self.ap1_names,total_mass=self.ap1_layout['total_mass'],
                model_total_mass=float(m.body_mass.sum()),gravity=m.opt.gravity.tolist(),nbody=m.nbody,
                unmapped_anatomical_count=self.ap1_layout['unmapped_anatomical_count'],
                region_columns=list(REGIONS),contact_mode_codebook=list(MODES),production_static_fusion=bool(self.sim.world.mjcf_root.compiler.fusestatic),
                descriptor_is_not_standing_validator=True)
            instances.append(self)

        def observe(self):
            obs=super().observe()
            if not hasattr(self,'ap1_records'):return obs
            m,d=self.sim.mj_model,self.sim.mj_data;clock=d.time
            before=[d.qpos.copy(),d.qvel.copy(),d.qacc.copy(),d.ctrl.copy(),d.sensordata.copy()]
            state,contacts=snapshot(m,d,self.ap1_layout,self.ap1_ground,self.ap1_claws)
            force=np.zeros((m.nbody,3))
            for c in contacts:force[c['body_id']]+=c['force_on_fly']
            descriptor=describe_support(force,self.ap1_names)
            reactions=np.array([descriptor['vertical_reaction_by_region'][n] for n in REGIONS])
            upstream=self.sim.get_bodysegment_contact_forces(self.fly.name,self.ap1_segments,ground_only=True)
            np.testing.assert_allclose(upstream,state['claw_ground_force'],rtol=1e-10,atol=1e-10)
            for actual,expected in zip((d.qpos,d.qvel,d.qacc,d.ctrl,d.sensordata),before):np.testing.assert_array_equal(actual,expected)
            assert d.time==clock
            self.ap1_records.append(dict(time=np.array(clock),region_vertical_reaction=reactions,
                non_tarsal_fraction=np.array(non_tarsal_fraction(reactions)),contact_mode_code=np.array(MODES.index(descriptor['contact_mode']),np.int8),
                com=state['com'],com_velocity=state['com_velocity'],ground_force=state['ground_force'],
                root_actuator_force=state['root_actuator_force'],force_closure_max_error=state['force_closure_max_error'],
                upstream_claw_force_max_error=np.array(np.max(np.abs(upstream-state['claw_ground_force'])))))
            self.ap1_latest_position=obs['position'][0].copy()
            return obs
    return AnatomyBody
