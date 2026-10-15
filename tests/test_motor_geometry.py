"""Verify homologous motor commands against the actual symmetric leg skeleton."""
import numpy as np
import pandas as pd
import pytest

from noesis.body import JOINTS, LEGS, MuscleModel
from noesis.motor import build_motor_matrix, build_muscle_matrices


@pytest.fixture(scope='module')
def skeleton():
    mj = pytest.importorskip('mujoco')
    flygym = pytest.importorskip('flygym')
    from noesis.body import make_fly
    from noesis.world import floor_arena
    fly = make_fly(body_proxies=False)
    sim = flygym.SingleFlySimulation(fly=fly, cameras=[], timestep=1e-4, arena=floor_arena())
    sim.reset(seed=0)
    model = sim.physics.model.ptr
    scratch = mj.MjData(model)
    scratch.qpos[:] = sim.physics.data.qpos
    qadr = np.array([model.jnt_qposadr[mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT,
                      f'{fly.name}/{j}')] for j in JOINTS])
    feet = np.array([mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY,
                     f'{fly.name}/{leg}Tarsus5') for leg in LEGS])
    yield mj, model, scratch, qadr, feet
    sim.close()


@pytest.mark.parametrize('muscle', ['sternal_anterior_rotator', 'pleural_remotor_abductor',
                                   'sternal_adductor', 'femur_reductor', 'tibia_flexor'])
def test_homologous_motor_activity_produces_mirrored_physical_feet(skeleton, muscle):
    rows = []
    for leg in LEGS:
        side = 'left' if leg[0] == 'L' else 'right'
        segment = {'F': 'front', 'M': 'middle', 'H': 'hind'}[leg[1]]
        rows.append(('leg_motor_neuron', segment+'_leg_motor_neuron',
                     side+'_leg_nerve', side, muscle))
    neurons = pd.DataFrame(rows, columns=['Class', 'Sub Class', 'Nerve', 'Soma side', 'Primary Cell Type'])
    gp, gn = build_muscle_matrices(neurons, list(JOINTS))
    # Both public motor paths must describe the same anatomical direction.
    np.testing.assert_array_equal(build_motor_matrix(neurons, list(JOINTS)).toarray(), (gp-gn).toarray())
    muscles = MuscleModel(gp, gn, list(JOINTS))
    target = muscles.step(np.full(6, 50.), 500.)
    mj, model, scratch, qadr, feet = skeleton
    scratch.qpos[qadr] = target
    mj.mj_kinematics(model, scratch)
    positions = scratch.xpos[feet]
    reflected_right = positions[3:]*np.array([1., -1., 1.])
    np.testing.assert_allclose(positions[:3], reflected_right, rtol=0., atol=1e-8)
