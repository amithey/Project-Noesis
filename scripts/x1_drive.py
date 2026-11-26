"""Explicit current-only intervention and descriptive fixed-weight physical metrics."""
from pathlib import Path
import numpy as np


def current_intervention(drive,enabled):
    if not drive:return {}
    result=dict(drive)
    result['current_mv_per_ms']=np.asarray(drive['current_mv_per_ms']).copy()
    if not enabled:result['current_mv_per_ms'].fill(0)
    return result


def physical_summary(directory):
    from noesis_gpu.learning import standing_mask
    from noesis_gpu.gait_metrics import compute_gait_summary
    with np.load(Path(directory)/'observations.npz',allow_pickle=False) as z:
        obs={k:z[k][:,0].copy() for k in ('position','up_body','heading','feet_position','foot_load')}
        assert all(len(z[k])==1000 and np.isfinite(z[k]).all() for k in z.files)
    standing=standing_mask(obs)
    support=obs['foot_load']>1e-5
    speed=np.linalg.norm(np.diff(obs['feet_position'],axis=0)[...,:2],axis=-1)/.005
    count=support[1:].sum(axis=-1)
    slip=np.divide((speed*support[1:]).sum(axis=-1),count,out=np.zeros(999),where=count>0)
    gait=compute_gait_summary(obs,5.,warmup_samples=10)
    return dict(standing_fraction=float(standing[10:].mean()),mean_stance_slip_mm_s=float(slip[9:].mean()),
                **{k:gait[k] for k in ('heading_forward_distance_mm','net_forward_displacement_mm',
                                      'total_leg_steps','tripod_coordination','locomotion_regime')})


def transfer_interaction(cells):
    fields=('standing_fraction','mean_stance_slip_mm_s','heading_forward_distance_mm',
            'net_forward_displacement_mm','total_leg_steps')
    if set(cells)!={('parent','on'),('parent','off'),('child','on'),('child','off')}:
        raise ValueError('Require all four fixed-weight cells')
    if any(not np.isfinite(v[k]) for v in cells.values() for k in fields):raise ValueError('Nonfinite evidence')
    on={k:cells['child','on'][k]-cells['parent','on'][k] for k in fields}
    off={k:cells['child','off'][k]-cells['parent','off'][k] for k in fields}
    return dict(child_minus_parent_on=on,child_minus_parent_off=off,
                difference_of_differences_on_minus_off={k:on[k]-off[k] for k in fields},
                one_seed_only=True,model_promoted=False,walking_proven=False)
