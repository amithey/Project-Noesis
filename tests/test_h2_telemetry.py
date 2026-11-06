import numpy as np
import mujoco
import pytest
from h2_telemetry import mass_layout,com_state,world_ground_contacts,snapshot


def fixture():
    model=mujoco.MjModel.from_xml_string('''<mujoco><option gravity="0 0 -9.81"/><worldbody>
    <geom name="floor" type="plane" size="5 5 .1"/>
    <body name="fly" pos="0 0 .08"><freejoint/><geom type="sphere" size=".1" mass="1"/>
    <body name="extra" pos="2 0 0"><geom type="sphere" size=".1" mass="3" contype="0" conaffinity="0"/></body>
    </body></worldbody><actuator><adhesion name="pad" body="fly" gain="1" ctrlrange="0 1"/></actuator></mujoco>''')
    data=mujoco.MjData(model);data.ctrl[0]=.3;mujoco.mj_forward(model,data)
    return model,data


def test_mass_weighted_com_uses_all_bodies_without_advancing_time():
    m,d=fixture();layout=mass_layout(m,[1,2]);before=d.qpos.copy();com,vel=com_state(m,d,layout)
    np.testing.assert_allclose(com,[1.5,0,.08],atol=1e-12)
    np.testing.assert_array_equal(vel,[0,0,0]);np.testing.assert_array_equal(d.qpos,before)
    assert layout['total_mass']==4 and d.time==0


def test_real_contact_world_sign_closes_root_and_records_adhesion_separately():
    m,d=fixture();layout=mass_layout(m,[1,2]);before=[d.qpos.copy(),d.qvel.copy(),d.ctrl.copy()]
    forces,rows=world_ground_contacts(m,d,layout['body_ids'],[0])
    assert rows and forces[:,2].sum()>0 and rows[0]['normal_on_fly'][2]>0
    state,_=snapshot(m,d,layout,[0],[1])
    np.testing.assert_allclose(state['ground_force'],state['root_constraint_force'],atol=1e-10)
    assert state['root_actuator_force'][2]<0
    for actual,expected in zip((d.qpos,d.qvel,d.ctrl),before):np.testing.assert_array_equal(actual,expected)
    assert d.time==0


def test_com_world_velocity_includes_mass_offset_under_root_rotation():
    m,d=fixture();d.qvel[5]=2.;mujoco.mj_forward(m,d)
    _,velocity=com_state(m,d,mass_layout(m,[1,2]))
    np.testing.assert_allclose(velocity,[0.,3.,0.],atol=1e-12)
    assert d.time==0


def test_multiple_free_roots_are_not_silently_called_one_fly():
    m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><freejoint/><geom type="sphere" size=".1"/></body><body><freejoint/><geom type="sphere" size=".1"/></body></worldbody></mujoco>')
    with pytest.raises(ValueError):mass_layout(m,[1])


def test_fused_anatomical_name_does_not_drop_or_double_count_compiled_mass():
    m=mujoco.MjModel.from_xml_string('<mujoco><compiler fusestatic="true"/><worldbody><body name="fly"><freejoint/><geom type="sphere" size=".1" mass="1"/><body name="head" pos="1 0 0"><geom type="sphere" size=".1" mass="2"/></body></body></worldbody></mujoco>')
    assert mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'head')==-1
    layout=mass_layout(m,[1,-1,1]);assert layout['total_mass']==3 and layout['unmapped_anatomical_count']==1
    assert layout['body_ids'].tolist()==[1]


def test_actual_native_fused_model_mass_contact_measurement_without_steps():
    import os,json
    from pathlib import Path
    from noesis_gpu.backend import FlyGymBatch
    body=FlyGymBatch(1,'cpu');body.observe();m,d=body.sim.mj_model,body.sim.mj_data
    native=body.sim._internal_bodyids_by_fly[body.fly.name];layout=mass_layout(m,native)
    critical=native[np.r_[body.thorax,body.feet]];assert np.all(critical>0) and len(set(critical.tolist()))==7
    before=(d.qpos.copy(),d.qvel.copy(),d.qacc.copy(),d.ctrl.copy(),d.sensordata.copy())
    state,_=snapshot(m,d,layout,body.sim._internal_ground_geom_ids,native[body.feet])
    segments=[body.fly.get_bodysegs_order()[i] for i in body.feet]
    upstream=body.sim.get_bodysegment_contact_forces(body.fly.name,segments,ground_only=True)
    np.testing.assert_allclose(state['claw_ground_force'],upstream,atol=1e-10,rtol=1e-10)
    for actual,expected in zip((d.qpos,d.qvel,d.qacc,d.ctrl,d.sensordata),before):np.testing.assert_array_equal(actual,expected)
    assert d.time==0 and layout['unmapped_anatomical_count']==1
    if 'H2_STATIC_RESULT' in os.environ:
        Path(os.environ['H2_STATIC_RESULT']).write_text(json.dumps(dict(status='passed_native_static_measurement',
            model_time=d.time,physics_steps=0,neural_steps=0,compiled_nbody=m.nbody,total_mass=layout['total_mass'],
            missing_fused_count=layout['unmapped_anatomical_count'],critical_ids=critical.tolist(),
            force_closure_max_error=float(state['force_closure_max_error']),
            upstream_claw_force_max_error=float(np.max(np.abs(state['claw_ground_force']-upstream))),
            whole_COM=state['com'].tolist(),production_fusestatic=bool(body.sim.world.mjcf_root.compiler.fusestatic)),indent=2),encoding='utf-8')
