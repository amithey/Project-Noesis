"""PS1 proposed scalar standing feedback; no motor commands or acceptance gates.

Engineering scales are fixed before the offline audit:1mm/s COM speed,1rad/s
body rotation, up.z .7..1, three loaded claws, vertical load near body weight.
They are modelling choices, not biological calibration or validated balance.
"""
import numpy as np

OBJECTIVE='physical-standing-ps1-v1'
SPEED_SCALE=1.0
SPIN_SCALE=1.0
FORCE_TOLERANCE=1e-5


def physical_score(region_vertical_reaction,com_velocity,omega,up_z,claw_load,weight):
    r,v,w,u,f=(np.asarray(x,dtype=float) for x in (region_vertical_reaction,com_velocity,omega,up_z,claw_load))
    if r.shape[-1:]!=(4,) or v.shape[-1:]!=(3,) or w.shape[-1:]!=(3,) or f.shape[-1:]!=(6,):raise ValueError('Wrong physical feature shape')
    leading=r.shape[:-1]
    if v.shape[:-1]!=leading or w.shape[:-1]!=leading or f.shape[:-1]!=leading or u.shape!=leading:raise ValueError('Mismatched physical sample axes')
    if not all(np.isfinite(x).all() for x in (r,v,w,u,f)) or np.any(r<0) or np.any(f<0) or np.any(np.abs(u)>1+1e-12):raise ValueError('Invalid physical feature values')
    if not np.isscalar(weight) or not np.isfinite(weight) or weight<=0:raise ValueError('Require positive finite whole-fly weight')
    total=r.sum(axis=-1)
    if not np.isfinite(total).all():raise ValueError('Nonfinite total reaction')
    upright=np.clip((u-.7)/.3,0,1)
    tarsal=np.divide(r[...,1]+r[...,2],total,out=np.zeros_like(total),where=total>FORCE_TOLERANCE)
    load=np.exp(-np.abs(total/weight-1))
    claws=np.minimum(np.sum(f>FORCE_TOLERANCE,axis=-1)/3,1)
    speed=np.hypot.reduce(v,axis=-1)/SPEED_SCALE;spin=np.hypot.reduce(w,axis=-1)/SPIN_SCALE
    # This product reduces feedback for motion, impact-only support and body
    # propping without modifying the underlying neural or mechanical system.
    motion=1/(1+speed*speed+spin*spin)
    quality=np.clip(upright*tarsal*load*claws*motion,0,1);reward=2*quality-1
    result=dict(upright_quality=upright,tarsal_share=tarsal,vertical_load_quality=load,
        claw_support_quality=claws,motion_quality=motion,quality=quality,reward=reward)
    if not leading:return {k:float(x) for k,x in result.items()}
    return result
