"""Anatomical flexion/extension must agree with the actual body skeleton."""
import numpy as np
import pandas as pd
import pytest

from noesis.body import JOINTS, LEGS, MuscleModel, make_fly
from noesis.motor import build_motor_matrix, build_muscle_matrices
from noesis.world import floor_arena


@pytest.fixture(scope='module')
def anatomy_skeleton():
    mj = pytest.importorskip('mujoco')
    flygym = pytest.importorskip('flygym')
    fly = make_fly(body_proxies=False)
    sim = flygym.SingleFlySimulation(fly=fly, cameras=[], timestep=1e-4, arena=floor_arena())
    sim.reset(seed=0)
    model = sim.physics.model.ptr
    data = mj.MjData(model)
    data.qpos[:] = sim.physics.data.qpos
    qadr = np.array([model.jnt_qposadr[mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT,
                     f'{fly.name}/{joint}')] for joint in JOINTS])
    yield mj, model, data, qadr, fly.name
    sim.close()


def interior_angle(mj, model, data, fly_name, leg, dof):
    segments = {'Femur': ('Coxa', 'Femur', 'Tibia'),
                'Tibia': ('Femur', 'Tibia', 'Tarsus1')}[dof]
    ids = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, f'{fly_name}/{leg}{seg}') for seg in segments]
    assert min(ids) >= 0
    a, pivot, b = data.xpos[ids]
    u, v = a-pivot, b-pivot
    return np.arccos(np.clip(u@v / (np.linalg.norm(u)*np.linalg.norm(v)), -1., 1.))


@pytest.mark.parametrize(('muscle', 'dof', 'direction'), [
    ('trochanter_flexor', 'Femur', -1),
    ('accessory_trochanter_flexor', 'Femur', -1),
    ('trochanter_extensor', 'Femur', 1),
    ('tergotrochanter', 'Femur', 1),
    ('sternotrochanter', 'Femur', 1),
    ('tibia_flexor', 'Tibia', -1),
    ('tibia_extensor', 'Tibia', 1),
])
def test_named_muscle_moves_the_anatomical_joint_in_its_named_direction(
        anatomy_skeleton, muscle, dof, direction):
    # Anatomical flexion decreases the interior joint angle. This tests physical
    # segment positions, not a restatement of the signed motor mapping table.
    cells = []
    for leg in LEGS:
        side = 'left' if leg[0] == 'L' else 'right'
        segment = {'F': 'front', 'M': 'middle', 'H': 'hind'}[leg[1]]
        cells.append(('leg_motor_neuron', segment+'_leg_motor_neuron',
                      side+'_leg_nerve', side, muscle))
    neurons = pd.DataFrame(cells, columns=['Class', 'Sub Class', 'Nerve', 'Soma side', 'Primary Cell Type'])
    gp, gn = build_muscle_matrices(neurons, list(JOINTS))
    np.testing.assert_array_equal(build_motor_matrix(neurons, list(JOINTS)).toarray(), (gp-gn).toarray())
    muscles = MuscleModel(gp, gn, list(JOINTS))
    mj, model, data, qadr, fly_name = anatomy_skeleton
    data.qpos[qadr] = muscles.stand
    mj.mj_kinematics(model, data)
    before = np.array([interior_angle(mj, model, data, fly_name, leg, dof) for leg in LEGS])
    data.qpos[qadr] = muscles.step(np.full(6, 5.), 10.)
    mj.mj_kinematics(model, data)
    after = np.array([interior_angle(mj, model, data, fly_name, leg, dof) for leg in LEGS])
    assert np.all(direction*(after-before) > 1e-5), dict(zip(LEGS, after-before, strict=True))


@pytest.mark.parametrize(('muscle', 'direction', 'segment', 'axis'), [
    ('sternal_anterior_rotator', 1, 'Femur', 'forward'),
    ('sternal_posterior_rotator', -1, 'Femur', 'forward'),
    ('tergopleural_promotor', 1, 'Femur', 'forward'),
    ('pleural_remotor_abductor', -1, 'Femur', 'forward'),
    ('tarsus_depressor', -1, 'Tarsus5', 'vertical'),
    ('tarsus_levator', 1, 'Tarsus5', 'vertical'),
])
def test_named_muscle_moves_the_segment_in_its_named_direction(
        anatomy_skeleton, muscle, direction, segment, axis):
    cells = []
    for leg in LEGS:
        side = 'left' if leg[0] == 'L' else 'right'
        level = {'F': 'front', 'M': 'middle', 'H': 'hind'}[leg[1]]
        cells.append(('leg_motor_neuron', level+'_leg_motor_neuron',
                      side+'_leg_nerve', side, muscle))
    neurons = pd.DataFrame(cells, columns=['Class', 'Sub Class', 'Nerve', 'Soma side', 'Primary Cell Type'])
    gp, gn = build_muscle_matrices(neurons, list(JOINTS))
    np.testing.assert_array_equal(build_motor_matrix(neurons, list(JOINTS)).toarray(), (gp-gn).toarray())
    muscles = MuscleModel(gp, gn, list(JOINTS))
    mj, model, data, qadr, fly_name = anatomy_skeleton
    segment_ids = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, f'{fly_name}/{leg}{segment}') for leg in LEGS]
    assert min(segment_ids) >= 0
    data.qpos[qadr] = muscles.stand
    mj.mj_kinematics(model, data)
    before = data.xpos[segment_ids].copy()
    thorax = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, f'{fly_name}/Thorax')
    vector = data.xmat[thorax].reshape(3, 3)[:, 0].copy() if axis == 'forward' else np.array([0., 0., 1.])
    data.qpos[qadr] = muscles.step(np.full(6, 5.), 10.)
    mj.mj_kinematics(model, data)
    displacement = (data.xpos[segment_ids]-before)@vector
    assert np.all(direction*displacement > 1e-5), dict(zip(LEGS, displacement, strict=True))
