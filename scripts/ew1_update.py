"""EW1 between-episode neural-weight arithmetic, not an external motor policy.

The frozen native brain generates every motor command during each trial. Only
existing checkpoint-selected weights change between complete trials. This
finite-direction pilot is not full-space optimization or a learned gait claim.
"""
import numpy as np

TRAIN_SEEDS=(1008,1009)
DIRECTIONS=4
PROBE_STEP=.02
UPDATE_STEP=.01
DIRECTION_SEED=20261110
SIGNAL_FLOOR=1e-12


def fixed_directions(edge_count):
    if isinstance(edge_count,bool) or not isinstance(edge_count,(int,np.integer)) or edge_count<1:
        raise ValueError('Require positive existing-edge count')
    # Separate generator: no consumption of native neural/physics RNG state.
    return np.random.default_rng(DIRECTION_SEED).choice(np.array([-1.,1.]),size=(DIRECTIONS,edge_count))


def apply_delta(checkpoint,relative_delta):
    result={k:np.asarray(v).copy() for k,v in checkpoint.items()}
    w,o=result['weights'],result['initial_weights']
    d=np.asarray(relative_delta)
    if (w.ndim!=1 or not len(w) or o.shape!=w.shape or d.shape!=w.shape
            or any(x.dtype.kind!='f' for x in (w,o,d))):
        raise ValueError('Require equal nonempty floating edge vectors')
    if not all(np.isfinite(x).all() for x in (w,o,d)):
        raise ValueError('Nonfinite neural-weight input')
    magnitude=np.abs(o)
    def valid(x):
        return (np.array_equal(np.sign(x),np.sign(o)) and np.isfinite(x).all()
                and np.all(np.abs(x)>=.5*magnitude) and np.all(np.abs(x)<=1.5*magnitude))
    if not valid(w):raise ValueError('Invalid original-reference parent')
    proposed=(w+d*magnitude).astype(w.dtype)
    if not valid(proposed):raise ValueError('Reject rather than clip original bounds/sign violations')
    result['weights']=proposed
    return result


def estimate_update(directions,plus_returns,minus_returns):
    directions=np.asarray(directions,float)
    plus,minus=np.asarray(plus_returns,float),np.asarray(minus_returns,float)
    if (directions.ndim!=2 or directions.shape[0]!=DIRECTIONS or directions.shape[1]<1
            or not np.isin(directions,[-1.,1.]).all()):
        raise ValueError('Require four fixed Rademacher edge directions')
    if plus.shape!=minus.shape or plus.shape!=(len(TRAIN_SEEDS),DIRECTIONS):
        raise ValueError('Require complete training-seed by direction return panel')
    if not all(np.isfinite(x).all() and np.all(np.abs(x)<=1+1e-12) for x in (plus,minus)):
        raise ValueError('Invalid bounded physical reward returns')
    coefficients=np.mean(plus-minus,axis=0)/(2*PROBE_STEP)
    gradient=np.mean(coefficients[:,None]*directions,axis=0)
    norm=float(np.max(np.abs(gradient)))
    delta=np.zeros_like(gradient) if norm<=SIGNAL_FLOOR else UPDATE_STEP*gradient/norm
    return dict(coefficients=coefficients,gradient=gradient,relative_delta=delta,
                gradient_max_abs=norm,zero_signal=norm<=SIGNAL_FLOOR)


def training_panel(records):
    """Reject missing/duplicate/out-of-scope trials before any learned update."""
    expected={(seed,direction,sign) for seed in TRAIN_SEEDS for direction in range(DIRECTIONS) for sign in (-1,1)}
    cells={}
    for r in records:
        key=(r['seed'],r['direction'],r['sign'])
        if key not in expected or key in cells:raise ValueError('Unexpected/duplicate training trial')
        value=r['return']
        if (r.get('complete_horizon') is not True or r.get('actual_import_verified') is not True
                or not isinstance(r.get('checkpoint_sha256'),str) or len(r['checkpoint_sha256'])!=64
                or not isinstance(r.get('return_source_sha256'),str) or len(r['return_source_sha256'])!=64
                or isinstance(value,bool) or not np.isfinite(value) or abs(value)>1+1e-12):
            raise ValueError('Incomplete/unverified training return')
        cells[key]=float(value)
    if set(cells)!=expected:raise ValueError('Incomplete paired training panel')
    return tuple(np.array([[cells[seed,d,sign] for d in range(DIRECTIONS)] for seed in TRAIN_SEEDS]) for sign in (1,-1))
