import numpy as np
import pytest
from p1_telemetry import pulse_force,dynamics,instrumented_body,START,WIDTH,FRACTION,PHYSICS_DT


def test_fixed_chunk_force_and_impulse_without_leg_or_action_control():
    f=np.stack([pulse_force(i,10.,True) for i in range(1000)])
    np.testing.assert_array_equal(np.flatnonzero(f[:,0]),np.arange(393,397))
    assert np.all(f[:,1:]==0) and np.sum(f[:,0])*.005==pytest.approx(.02)
    assert np.all(np.stack([pulse_force(i,10.,False) for i in range(1000)])==0)
    for chunk,weight in ((True,10),(-1,10),(0,float('nan')),(0,-1)):
        with pytest.raises(ValueError):pulse_force(chunk,weight,True)


def test_installed_mujoco_cartesian_force_mapping_and_one_step_impulse():
    import mujoco
    xml='<mujoco><option timestep=".0001" gravity="0 0 0"/><worldbody><body name="fixture" pos="0 0 2"><freejoint/><geom type="sphere" size=".1" pos=".2 .1 0" mass="2"/></body></worldbody></mujoco>'
    m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m);d.xfrc_applied[1,:3]=[.2,.1,.3]
    mujoco.mj_forward(m,d);saved=[d.qpos.copy(),d.qvel.copy(),d.qacc.copy(),d.xfrc_applied.copy()]
    a=dynamics(m,d)
    assert a['dynamic_force_max_error']<1e-10 and a['smooth_force_max_error']<1e-10
    for x,y in zip((d.qpos,d.qvel,d.qacc,d.xfrc_applied),saved):np.testing.assert_array_equal(x,y)
    mujoco.mj_step(m,d)
    np.testing.assert_allclose(d.qvel[:3],np.array([.2,.1,.3])/2*PHYSICS_DT,rtol=0,atol=1e-12)
    # One synthetic primitive physics step, zero neural steps; no fly ability claim.


def test_native_zero_time_observer_preserves_inputs_and_real_force_balance():
    from noesis_gpu.backend import FlyGymBatch
    a=FlyGymBatch(1,'cpu');expected=a.observe();instances=[]
    b=instrumented_body(FlyGymBatch,instances,False)(1,'cpu');actual=b.observe()
    for k in expected:np.testing.assert_array_equal(actual[k],expected[k])
    assert b.p1_count==b.p1_chunk==0 and len(b.p1_records)==1 and b.sim.mj_data.time==0
    import mujoco
    b.sim.mj_data.xfrc_applied[b.p1_target,:3]=pulse_force(START,b.p1_weight,True)
    mujoco.mj_forward(b.sim.mj_model,b.sim.mj_data)
    a=dynamics(b.sim.mj_model,b.sim.mj_data)
    np.testing.assert_allclose(a['qfrc_xfrc'][b.p1_layout['free_dof']:b.p1_layout['free_dof']+3],[FRACTION*b.p1_weight,0,0],rtol=0,atol=1e-12)
    assert b.sim.mj_data.time==0 # Applied-force static calibration, no fly integration/neural success.
