"""Opt-in negative-Femur pad release for a frozen one-factor diagnostic only."""
import numpy as np


def negative_femur_adhesion(desired, rest):
    desired,rest=np.asarray(desired,float),np.asarray(rest,float)
    if desired.shape!=(42,) or rest.shape!=(42,) or not np.isfinite(desired).all() or not np.isfinite(rest).all():
        raise ValueError('Require finite42channel desired and rest angles')
    lift=(rest-desired).reshape(6,7)[:,3]
    return (lift<=.08).astype(float)


def screen(legacy,corrected,seeds):
    if set(legacy)!=set(seeds) or set(corrected)!=set(seeds):
        raise ValueError('Missing/extra seed evidence')
    fields=('standing_fraction','mean_stance_slip_mm_s','heading_forward_distance_mm',
            'net_forward_displacement_mm','total_leg_steps')
    if any(not np.isfinite(row[k]) for bank in (legacy,corrected) for row in bank.values() for k in fields):
        raise ValueError('Nonfinite physical summary')
    standing=all(corrected[s]['standing_fraction']>=.9 and corrected[s]['standing_fraction']>=legacy[s]['standing_fraction'] for s in seeds)
    slip=all(corrected[s]['mean_stance_slip_mm_s']<=legacy[s]['mean_stance_slip_mm_s'] for s in seeds)
    heading=float(np.median([corrected[s]['heading_forward_distance_mm']-legacy[s]['heading_forward_distance_mm'] for s in seeds]))
    net=float(np.median([corrected[s]['net_forward_displacement_mm']-legacy[s]['net_forward_displacement_mm'] for s in seeds]))
    positive=np.median([corrected[s]['heading_forward_distance_mm'] for s in seeds])>0 and np.median([corrected[s]['net_forward_displacement_mm'] for s in seeds])>0
    cycles=np.median([corrected[s]['total_leg_steps'] for s in seeds])>=np.median([legacy[s]['total_leg_steps'] for s in seeds])
    tripod=np.median([corrected[s]['tripod_coordination']['tripod_coordination_score'] for s in seeds])>=np.median([legacy[s]['tripod_coordination']['tripod_coordination_score'] for s in seeds])
    return dict(passed=bool(standing and slip and heading>=.25 and net>=.25 and positive and cycles and tripod),
                standing_retained=standing,slip_retained=slip,median_heading_gain_mm=heading,
                median_net_forward_gain_mm=net,positive_forward_progress=bool(positive),
                cycles_retained=bool(cycles),tripod_retained=bool(tripod),model_promoted=False,walking_proven=False)
