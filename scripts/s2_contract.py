"""S2 stage/horizon gates; never overrides the native learning or physical scorer."""
import numpy as np


def validate_training_report(report):
    expected=dict(status='completed_training_birth',requested_seconds=25,simulated_seconds=25,
                  completed_chunks=5000,learning_updates=4990,reward_objective='curriculum-standing-v7',
                  learning_rate=.2,eligibility_ms=100,motor_exploration_mv_ms=1.5,original_weight_bound=.5,
                  control_ms=5,physics_dt=.0001,birth_ms=50,learning=True,worlds=1)
    if any(report.get(k)!=v for k,v in expected.items()):
        raise ValueError('Require full native25s/4990update standing birth with fixed settings')
    log=report.get('exploration_log',{})
    if log.get('steps')!=4990 or log.get('first_control_chunk')!=10 or report.get('exploration_log_error'):
        raise ValueError('Missing/misaligned exploration evidence')


def standing_retention(parent,child,seeds):
    if set(parent)!=set(seeds) or set(child)!=set(seeds):
        raise ValueError('Missing/extra frozen seed evidence')
    fields=('standing_fraction','mean_stance_slip_mm_s','heading_forward_distance_mm',
            'net_forward_displacement_mm','total_leg_steps')
    if any(not np.isfinite(r[k]) for bank in (parent,child) for r in bank.values() for k in fields):
        raise ValueError('Nonfinite frozen evidence')
    standing=all(child[s]['standing_fraction']>=.9 and child[s]['standing_fraction']>=parent[s]['standing_fraction'] for s in seeds)
    slip=all(child[s]['mean_stance_slip_mm_s']<=parent[s]['mean_stance_slip_mm_s'] for s in seeds)
    gains={k:float(np.median([child[s][k]-parent[s][k] for s in seeds])) for k in
           ('heading_forward_distance_mm','net_forward_displacement_mm','total_leg_steps')}
    tripod=float(np.median([child[s]['tripod_coordination']['tripod_coordination_score']-
                            parent[s]['tripod_coordination']['tripod_coordination_score'] for s in seeds]))
    return dict(passed=bool(standing and slip and all(v>=0 for v in gains.values()) and tripod>=0),
                standing_retained=standing,slip_retained=slip,median_paired_gains=gains,
                median_tripod_gain=tripod,standing_stage_only=True,walking_proven=False,model_promoted=False)
