"""Flat-ground anatomical force descriptor; never a standing/balance validator."""
import numpy as np

REGIONS=('central_body','claw_tarsus5','other_tarsal','proximal_leg')


def anatomical_region(name):
    if not isinstance(name,str):raise ValueError('Missing loaded body name')
    base=name.rsplit('/',1)[-1]
    if base.startswith('c_'):return 'central_body'  # Compiled thorax can include fused head.
    parts=base.split('_',1)
    if len(parts)!=2 or parts[0] not in ('lf','lm','lh','rf','rm','rh'):raise ValueError('Unknown loaded anatomy: '+name)
    link=parts[1]
    if link=='tarsus5':return 'claw_tarsus5'
    if link in ('tarsus1','tarsus2','tarsus3','tarsus4'):return 'other_tarsal'
    if link in ('coxa','trochanter','femur','tibia'):return 'proximal_leg'
    raise ValueError('Unknown loaded anatomy: '+name)


def describe_support(forces,body_names,force_tolerance=1e-5):
    forces=np.asarray(forces,float)
    if forces.ndim!=2 or forces.shape[1]!=3 or len(forces)!=len(body_names) or not np.isfinite(forces).all():raise ValueError('Invalid world force/name data')
    if not np.isfinite(force_tolerance) or force_tolerance<=0:raise ValueError('Require positive explicit force tolerance')
    if np.any(forces[:,2]<-1e-9):raise ValueError('Negative vertical reaction: require flat-ground force convention')
    vertical=np.maximum(forces[:,2],0.)
    amounts={k:0. for k in REGIONS};counts={k:0 for k in REGIONS}
    for i,value in enumerate(vertical):
        if value==0:continue
        region=anatomical_region(body_names[i]);amounts[region]+=float(value)
        counts[region]+=int(value>force_tolerance)
    total=float(vertical.sum());loaded=[k for k in REGIONS if amounts[k]>force_tolerance]
    if total<=force_tolerance:mode='unsupported'
    elif amounts['central_body']>force_tolerance:mode='central_body_contact'
    elif amounts['proximal_leg']>force_tolerance:mode='proximal_leg_contact'
    elif amounts['claw_tarsus5']+amounts['other_tarsal']>force_tolerance:mode='tarsal_only'
    else:mode='indeterminate_low_force'
    return dict(contact_mode=mode,total_vertical_reaction=total,vertical_reaction_by_region=amounts,
        reaction_share_by_region={k:v/total for k,v in amounts.items()} if total>force_tolerance else None,
        loaded_regions=loaded,loaded_body_count_by_region=counts,force_tolerance=force_tolerance,
        central_body_contact=amounts['central_body']>force_tolerance,
        physical_standing_validated=False,dynamic_balance_validated=False,model_promoted=False,
        description_only=True)


def static_fixture(kind):
    """Role-labelled sphere geometry controls, not a fly or learned-behavior test."""
    import mujoco
    from h2_telemetry import mass_layout,world_ground_contacts,snapshot
    config={
        'airborne':(2.,-.31,'tarsus5',False),
        'foot_only':(.4,-.31,'tarsus5',False),
        'body_only':(.09,.3,'tarsus5',False),
        'mixed_body_feet':(.09,0.,'tarsus5',False),
        'proximal_leg':(.4,.3,'tarsus5',True),
        'other_tarsal':(.4,-.31,'tarsus3',False)}
    if kind not in config:raise ValueError('Unknown fixture')
    height,offset,link,prox=config[kind]
    feet=''.join(f'<body name="nmf/{leg}_{link}" pos="{x} {y} {offset}"><geom type="sphere" size=".1" mass=".01"/></body>'
        for leg,x,y in [('lf',-.25,-.2),('rf',.25,-.2),('lh',0,.25)])
    extra='<body name="nmf/lf_femur" pos="0 0 -.31"><geom type="sphere" size=".1" mass=".02"/></body>' if prox else ''
    xml=f'<mujoco><compiler fusestatic="false"/><option gravity="0 0 -9.81"/><worldbody><geom name="floor" type="plane" size="5 5 .1"/><body name="nmf/c_thorax" pos="0 0 {height}"><freejoint/><geom type="sphere" size=".1" mass="1"/>{feet}{extra}</body></worldbody></mujoco>'
    m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    before=(d.qpos.copy(),d.qvel.copy(),d.qacc.copy(),d.ctrl.copy());ids=np.arange(1,m.nbody);layout=mass_layout(m,ids)
    names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,i) for i in range(m.nbody)]
    forces,contacts=world_ground_contacts(m,d,ids,[0]);state,_=snapshot(m,d,layout,[0],[i for i,n in enumerate(names) if n and n.endswith('tarsus5')])
    result=describe_support(forces,names)
    for actual,expected in zip((d.qpos,d.qvel,d.qacc,d.ctrl),before):np.testing.assert_array_equal(actual,expected)
    assert d.time==0
    return dict(kind=kind,descriptor=result,model_time=float(d.time),physics_integration_steps=0,neural_steps=0,
        fixture_is_not_fly=True,units='MuJoCo fixture model units, not biological fly calibration',
        body_names=names,world_ground_forces=forces.tolist(),COM=state['com'].tolist(),
        root_force_closure_max_error=float(state['force_closure_max_error']),contact_count=len(contacts))
