"""Read-only mass and world-ground-force telemetry; no physics integration."""
import numpy as np


def ancestors(model,body):
    result=[int(body)]
    while result[-1]!=0:result.append(int(model.body_parentid[result[-1]]))
    return result


def mass_layout(model,native_ids):
    raw=np.asarray(native_ids,int)
    # Static fusion removes anatomical names (mj_name2id -> -1), while their
    # mass is transferred to compiled ancestors. Count COM over real compiled
    # bodies, guarded by complete mass coverage, never over anatomical aliases.
    if raw.ndim!=1 or np.any(raw<-1) or np.any(raw==0) or np.any(raw>=model.nbody):raise ValueError('Invalid fly body ids')
    ids=np.unique(raw[raw>0])
    if not len(ids):raise ValueError('No valid compiled fly bodies')
    common=set(ancestors(model,ids[0]))
    for i in ids[1:]:common&=set(ancestors(model,i))
    root=next(i for i in ancestors(model,ids[0]) if i in common)
    bodies=np.array([i for i in range(1,model.nbody) if root in ancestors(model,i)],int)
    mass=float(model.body_mass[bodies].sum());total=float(model.body_mass.sum())
    if mass<=0 or not np.isclose(mass,total,rtol=1e-12,atol=1e-15):raise ValueError('Fly subtree does not cover all positive model mass')
    if not np.isclose(model.body_subtreemass[root],mass,rtol=1e-12,atol=1e-15):raise ValueError('Subtree mass mismatch')
    import mujoco
    free=np.flatnonzero(model.jnt_type==mujoco.mjtJoint.mjJNT_FREE)
    if len(free)!=1:raise ValueError('Require exactly one free fly root')
    return dict(root=int(root),body_ids=bodies,total_mass=mass,free_joint=int(free[0]),free_dof=int(model.jnt_dofadr[free[0]]),
                unmapped_anatomical_count=int(np.count_nonzero(raw==-1)),unique_positive_native_count=len(ids))


def com_state(model,data,layout):
    import mujoco
    ids=layout['body_ids'];mass=layout['total_mass']
    com=(model.body_mass[ids,None]*data.xipos[ids]).sum(axis=0)/mass
    np.testing.assert_allclose(com,data.subtree_com[layout['root']],rtol=1e-12,atol=1e-12)
    jac=np.zeros((3,model.nv));mujoco.mj_jacSubtreeCom(model,data,jac,layout['root'])
    return com,jac@data.qvel


def world_ground_contacts(model,data,fly_bodies,ground_geoms):
    import mujoco
    fly=set(int(x) for x in fly_bodies);ground=set(int(x) for x in ground_geoms)
    per_body=np.zeros((model.nbody,3));rows=[]
    for i in range(data.ncon):
        c=data.contact[i]
        if c.exclude:continue
        g1,g2=int(c.geom1),int(c.geom2);b1,b2=int(model.geom_bodyid[g1]),int(model.geom_bodyid[g2])
        if g1 in ground and b2 in fly:body=b2;sign=1
        elif g2 in ground and b1 in fly:body=b1;sign=-1
        else:continue
        wrench=np.zeros(6);mujoco.mj_contactForce(model,data,i,wrench)
        frame=np.asarray(c.frame).reshape(3,3).copy();force=sign*(frame.T@wrench[:3])
        per_body[body]+=force
        rows.append(dict(contact_id=i,geom_ids=[g1,g2],body_id=body,position=np.asarray(c.pos).copy(),
            normal_on_fly=sign*frame[0],force_on_fly=force,wrench=wrench,frame=frame,distance=float(c.dist)))
    return per_body,rows


def snapshot(model,data,layout,ground_geoms,claw_ids):
    com,velocity=com_state(model,data,layout)
    force,contacts=world_ground_contacts(model,data,layout['body_ids'],ground_geoms)
    dof=layout['free_dof'];net=force.sum(axis=0)
    weight=layout['total_mass']*abs(float(model.opt.gravity[2]))
    root_constraint=data.qfrc_constraint[dof:dof+3].copy()
    error=float(np.max(np.abs(net-root_constraint)))
    if error>1e-6*weight+1e-10:raise ValueError('Ground-contact force does not close root translation constraint force')
    return dict(time=np.array(data.time),qpos=data.qpos.copy(),qvel=data.qvel.copy(),qacc=data.qacc.copy(),
        ctrl=data.ctrl.copy(),actuator_force=data.actuator_force.copy(),qfrc_actuator=data.qfrc_actuator.copy(),
        com=com,com_velocity=velocity,body_xpos=data.xpos[layout['body_ids']].copy(),
        body_xipos=data.xipos[layout['body_ids']].copy(),body_xquat=data.xquat[layout['body_ids']].copy(),
        ground_force=net,claw_ground_force=force[np.asarray(claw_ids,int)],
        root_constraint_force=root_constraint,root_actuator_force=data.qfrc_actuator[dof:dof+3].copy(),
        root_applied_force=data.qfrc_applied[dof:dof+3].copy(),xfrc_applied=data.xfrc_applied.copy(),
        force_closure_max_error=np.array(error)),contacts
