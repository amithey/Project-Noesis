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
