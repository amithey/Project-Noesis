"""FK1 native-joint geometry audit, no dynamics or neural execution."""
import hashlib
import json
from pathlib import Path
import sys
import time

import mujoco as mj
import numpy as np

root=Path(__file__).resolve().parents[1]
pilot=root/'work/Noesis-Sensory'
core=root/'work/Noesis-Core/src'
sys.path[:0]=[str(pilot/'gpu-pilot/src'),str(core)]
from noesis_gpu.backend import FlyGymBatch
from noesis_gpu.bridge import LEGS, LEGACY_JOINTS, adhesion_from_command
from noesis.motor import MUSCLE_JOINT

out=root/'outputs/native-kinematics-fk1-20261106'
out.mkdir(exist_ok=False)
started=time.monotonic()
paths=[pilot/'gpu-pilot/src/noesis_gpu/backend.py',pilot/'gpu-pilot/src/noesis_gpu/bridge.py',core/'noesis/motor.py']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


hashes={str(p.relative_to(root)):sha(p) for p in paths}
parent_path=root/'work/evidence/s1-restored/gpu-pilot/outputs/e4-exploration-20261029/train-control-1000/weights-last.npz'
parent_hash=sha(parent_path)
assert parent_hash=='9fd7d41177bf2c745809401c3b142a832661d1a18d0b117b30cb02aea63fea3c'
provenance=json.loads((root/'outputs/thermo-pilot-tp2-20261106-source.json').read_text())
for label,folder in [('pilot',pilot/'gpu-pilot/src'),('core',core)]:
    assert len(list(folder.rglob('*.py')))==len(provenance[label])
    for f in provenance[label]:
        assert sha(folder/f['path'])==f['sha256']

body=FlyGymBatch(1,'cpu')
model,data=body.sim.mj_model,body.sim.mj_data
base=data.qpos.copy()
time0=float(data.time)
ids=body.sim._internal_jointids_by_fly[body.fly.name][body.read_order]
qadr=model.jnt_qposadr[ids]
dadr=model.jnt_dofadr[ids]
assert len(ids)==42 and np.all(model.jnt_type[ids]==mj.mjtJoint.mjJNT_HINGE)
segments=body.fly.get_bodysegs_order()
bodyids=body.sim._internal_bodyids_by_fly[body.fly.name]
feet=bodyids[body.feet]
thorax=int(bodyids[body.thorax])
links={}
for leg in LEGS:
    links[leg]={name:int(bodyids[next(i for i,s in enumerate(segments) if s.pos==leg.lower() and s.link==name)])
                for name in ('coxa','trochanterfemur','tibia','tarsus1','tarsus5')}


def interior(a,b,c):
    u,v=a-b,c-b
    length=np.linalg.norm(u)*np.linalg.norm(v)
    if length<=1e-12:
        raise ValueError('Degenerate segment geometry')
    return float(np.arccos(np.clip(np.dot(u,v)/length,-1.,1.)))


assert np.isclose(interior(np.array([1.,0,0]),np.zeros(3),np.array([0.,1,0])),np.pi/2)


def geometry(angles):
    data.qpos[:]=base
    data.qpos[qadr]=angles
    mj.mj_forward(model,data)
    foot=data.xpos[feet].copy()
    angles_out=[]
    for leg in LEGS:
        p={k:data.xpos[v].copy() for k,v in links[leg].items()}
        angles_out.append([interior(p['coxa'],p['trochanterfemur'],p['tibia']),
                           interior(p['trochanterfemur'],p['tibia'],p['tarsus1'])])
    assert float(data.time)==time0
    return foot,np.asarray(angles_out)


poses=[body.rest.copy()]
labels=['native_neutral']
inputs=[]
es=json.loads((root/'outputs/episodic-search-es1-20261106/result.json').read_text())
for seed in (10006,10007,10010,10011):
    file=root/f'outputs/episodic-search-es1-20261106/parent-{seed}/observations.npz'
    expected=next(r['observations_sha256'] for r in es['cases'] if r['bank']=='parent' and r['seed']==seed)
    assert sha(file)==expected
    with np.load(file,allow_pickle=False) as z:
        active=z['angles'][:,0]
        assert active.shape==(1000,42) and np.isfinite(active).all()
        for frame in (10,250,500,750,999):
            poses.append(active[frame].copy());labels.append(f'ES1-parent-{seed}-frame-{frame}')
    inputs.append(dict(path=str(file.relative_to(root)),sha256=expected))
poses=np.stack(poses)
steps=(.001,.0005)
derivatives=[];analytic=[];joint_angle_derivatives=[];base_feet=[];boundary=[]
for angles in poses:
    positions,_=geometry(angles)
    base_feet.append(positions)
    expected=np.zeros((6,42))
    for leg,foot_id in enumerate(feet):
        jac=np.zeros((3,model.nv));rot=np.zeros_like(jac)
        mj.mj_jacBody(model,data,jac,rot,int(foot_id))
        expected[leg]=jac[2,dadr]
    analytic.append(expected)
    pose_derivatives=[];angle_derivatives=[]
    for h in steps:
        derivative=np.zeros((6,42));a_derivative=np.zeros((6,2,42))
        for channel in range(42):
            positive,negative=angles.copy(),angles.copy()
            positive[channel]+=h;negative[channel]-=h
            p,pa=geometry(positive);n,na=geometry(negative)
            derivative[:,channel]=(p[:,2]-n[:,2])/(2*h)
            a_derivative[:,:,channel]=(pa-na)/(2*h)
        pose_derivatives.append(derivative);angle_derivatives.append(a_derivative)
    derivatives.append(pose_derivatives);joint_angle_derivatives.append(angle_derivatives)
    boundary.append(((angles+.001)>body.hi)|((angles-.001)<body.lo))
derivatives=np.asarray(derivatives);analytic=np.asarray(analytic)
angle_derivatives=np.asarray(joint_angle_derivatives)
max_jac=float(np.abs(derivatives[:,1]-analytic).max())
max_convergence=float(np.abs(derivatives[:,0]-derivatives[:,1]).max())
assert max_jac<1e-5 and max_convergence<1e-5
other_leg=[]
for leg in range(6):
    other_leg.append(np.abs(analytic[:,leg,[j for j in range(42) if j//7!=leg]]).max())
assert max(other_leg)<1e-10

# Independent rigid-translation invariance, without advancing time.
original_positions,_=geometry(body.rest)
free=np.flatnonzero(model.jnt_type==mj.mjtJoint.mjJNT_FREE)
assert len(free)==1
root_q=int(model.jnt_qposadr[free[0]])
translation=np.array([.7,-1.3,2.])
data.qpos[root_q:root_q+3]+=translation
mj.mj_forward(model,data)
translation_error=float(np.abs(data.xpos[feet]-original_positions-translation).max())
assert translation_error<1e-10 and float(data.time)==time0
finite=[];neutral=[];representative=[]
for leg,name in enumerate(LEGS):
    femur=leg*7+3;tibia=leg*7+5
    dz=derivatives[:,1,leg,femur]
    neutral.append(dict(leg=name,foot_dz_per_positive_Femur_mm_rad=float(dz[0]),
                        apparent_hip_angle_derivative=float(angle_derivatives[0,1,leg,0,femur]),
                        tibia_interior_angle_derivative=float(angle_derivatives[0,1,leg,1,tibia]),
                        coded_trochanter_flexor_Femur_sign=-1,coded_trochanter_extensor_Femur_sign=1))
    representative.append(dict(leg=name,positive_Femur_raises=int((dz>1e-4).sum()),
                               positive_Femur_lowers=int((dz< -1e-4).sum()),near_zero=int((np.abs(dz)<=1e-4).sum()),
                               postures=len(dz),min_dz=float(dz.min()),max_dz=float(dz.max())))
    for offset in (-.1,.1):
        command=body.rest.copy();command[femur]+=offset
        pos,_=geometry(command)
        finite.append(dict(leg=name,Femur_offset_rad=offset,own_foot_dz_mm=float(pos[leg,2]-original_positions[leg,2]),
                           adhesion=float(adhesion_from_command(command,body.rest)[leg])))
geometry(body.rest)
assert float(data.time)==0 and time0==0
assert all(sha(p)==hashes[str(p.relative_to(root))] for p in paths) and sha(parent_path)==parent_hash
assert MUSCLE_JOINT['trochanter_flexor']==[('Femur',-1)] and MUSCLE_JOINT['trochanter_extensor']==[('Femur',1)]
np.savez_compressed(out/'numerical-evidence.npz',poses=poses,pose_labels=np.array(labels),
                    legacy_joint_names=np.array(LEGACY_JOINTS),height_derivative=derivatives,analytic_jacobian=analytic,
                    apparent_joint_angle_derivative=angle_derivatives,baseline_foot_positions=np.asarray(base_feet),
                    derivative_samples_cross_native_joint_bounds=np.asarray(boundary))
result=dict(status='completed_FK1_geometry_only',seconds=time.monotonic()-started,
            runtime_commit='e0cbd109bb6e018bd253c9f1b5e1698e0421b276',core_commit='42c20769376bda5694b4c1c66ca75e52cbd9f1f4',
            source_sha256=hashes,source_unchanged=True,parent_unchanged=True,input_body_arrays=inputs,
            postures=21,joints=42,derivative_steps_rad=list(steps),foot_point='native observed tarsus5 body origin, xpos, mm',
            max_finite_difference_jacobian_error_mm_rad=max_jac,max_step_halving_error_mm_rad=max_convergence,
            max_rigid_translation_error_mm=translation_error,max_other_leg_derivative_mm_rad=float(max(other_leg)),
            boundary_channels=int(np.asarray(boundary).sum()),neutral=neutral,pose_summary=representative,finite_offsets=finite,
            model_time_start=time0,model_time_end=float(data.time),dynamics_steps=0,neural_steps=0,
            controller_changed=False,weights_changed=False,walking_proven=False,model_promoted=False,
            limits=['Active-joint snapshots only; root and passive joints fixed neutral, not recorded-state replay.',
                    'Partial derivative holds all other joints fixed; simultaneous joint/body motion can alter real vertical velocity.',
                    'Desired-angle pad condition and actual-joint geometry differ under actuator lag.',
                    'Apparent 3D interior angles are geometric measurements, not biological torque or muscle validation.',
                    'Infinitesimal central derivatives near limits can sample outside the configured range; flags reported, no dynamics.',
                    'Geometry alone does not prove a cause of failed standing/walking or justify a sign/policy change.'])
(out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps({k:result[k] for k in ('status','seconds','max_finite_difference_jacobian_error_mm_rad','neutral','pose_summary','finite_offsets')},indent=2))
