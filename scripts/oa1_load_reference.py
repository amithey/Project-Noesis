"""OA1 descriptive load-reference audit, never an objective replacement.

The alternative is only an actuator-adjusted quasi-static engineering reference.
It omits acceleration, other applied forces and calibration. Neither reference
alone validates support, balance, learning or biological standing.
MuJoCo3.9 API: https://mujoco.readthedocs.io/en/3.9.0/APIreference/APItypes.html
"""
import numpy as np


def load_references(ground_z,actuator_z,weight):
    ground,actuator=np.asarray(ground_z,float),np.asarray(actuator_z,float)
    if ground.shape!=actuator.shape or not ground.size or not np.isfinite(ground).all() or not np.isfinite(actuator).all():
        raise ValueError('Require matching nonempty finite measured force arrays')
    if not np.isscalar(weight) or not np.isfinite(weight) or weight<=0:raise ValueError('Require positive finite weight')
    raw=ground/weight-1
    adjusted=(ground+actuator)/weight-1
    return dict(raw_ratio_error=raw,actuator_adjusted_ratio_error=adjusted,
        raw_load_quality=np.exp(-abs(raw)),actuator_adjusted_load_quality=np.exp(-abs(adjusted)),
        actuator_weight_ratio=actuator/weight)


def diagnostic_summary(values):
    raw,adjusted=values['raw_load_quality'],values['actuator_adjusted_load_quality']
    delta=adjusted-raw
    return dict(samples=int(raw.size),mean_raw_load_quality=float(raw.mean()),
        mean_adjusted_load_quality=float(adjusted.mean()),mean_change=float(delta.mean()),
        changed_samples=int(np.count_nonzero(abs(delta)>1e-12)),
        median_actuator_weight_ratio=float(np.median(values['actuator_weight_ratio'])),
        p95_absolute_actuator_weight_ratio=float(np.quantile(abs(values['actuator_weight_ratio']),.95)),
        numerical_change_threshold=1e-12,not_a_standing_or_newton_residual=True)
