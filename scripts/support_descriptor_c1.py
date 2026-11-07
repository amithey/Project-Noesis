"""C1 explicit native compound proximal-leg coverage; no acceptance gate."""
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
    if link in ('coxa','trochanter','femur','trochanterfemur','tibia'):return 'proximal_leg'
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
