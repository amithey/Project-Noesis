"""MP1 measured whole-policy acceptance; no motor policy or weight averaging."""
import numpy as np

TRAIN_SEEDS=(1010,1011)
FRESH=(11008,11009)
DEV=(10006,10007,10010,10011)
DIRECTIONS=4
STEP=.01
DIRECTION_SEED=2026111001
MIN_GAIN=1e-6


def fixed_directions(edge_count):
    if isinstance(edge_count,bool) or not isinstance(edge_count,(int,np.integer)) or edge_count<1:
        raise ValueError('Require existing-edge count')
    return np.random.default_rng(DIRECTION_SEED).choice(np.array([-1.,1.]),size=(DIRECTIONS,edge_count))


def select_policy(parent_returns,policy_returns,reward_sign=1):
    """Policy columns in canonical order: d0minus,d0plus,...,d3plus.

    Acceptance is an in-sample training condition, never independent efficacy.
    A fallback to the parent is a zero update, not a successful learned ability.
    """
    parent,policies=np.asarray(parent_returns,float),np.asarray(policy_returns,float)
    if parent.shape!=(2,) or policies.shape!=(2,8):raise ValueError('Require complete two-context/eight-policy returns')
    if isinstance(reward_sign,bool) or reward_sign not in (-1,1):raise ValueError('Only positive/reversed reward control')
    if not all(np.isfinite(x).all() and np.all(abs(x)<=1+1e-12) for x in (parent,policies)):
        raise ValueError('Invalid bounded physical returns')
    margins=(reward_sign*(policies-parent[:,None])).min(axis=0)
    best=int(np.argmax(margins)) # first canonical id resolves exact ties
    accepted=bool(margins[best]>MIN_GAIN)
    return dict(accepted_for_training_update=accepted,policy_index=best if accepted else -1,
        worst_training_gain=float(margins[best]),reward_sign=reward_sign,parent_fallback=not accepted,
        training_condition_not_independent_efficacy=True,model_promoted=False)


def matched_gradient(directions,policy_returns):
    directions=np.asarray(directions,float);returns=np.asarray(policy_returns,float)
    if directions.ndim!=2 or directions.shape[0]!=4 or not directions.shape[1] or not np.isin(directions,[-1.,1.]).all():
        raise ValueError('Require registered four Rademacher directions')
    if returns.shape!=(2,8) or not np.isfinite(returns).all() or np.any(abs(returns)>1+1e-12):
        raise ValueError('Require complete bounded return panel')
    coefficients=(returns[:,1::2]-returns[:,::2]).mean(0)/(2*STEP)
    gradient=np.mean(coefficients[:,None]*directions,axis=0)
    norm=float(abs(gradient).max())
    delta=np.zeros_like(gradient) if norm<=1e-12 else STEP*gradient/norm
    return dict(coefficients=coefficients,gradient=gradient,relative_delta=delta,zero_signal=norm<=1e-12)


def training_panel(records):
    expected={(s,d,sign) for s in TRAIN_SEEDS for d in range(4) for sign in (-1,1)}
    cells={}
    for r in records:
        key=(r['seed'],r['direction'],r['sign'])
        if key not in expected or key in cells:raise ValueError('Unexpected/duplicate training trial')
        value=r['return']
        if (r.get('complete_horizon') is not True or r.get('actual_import_verified') is not True
            or not isinstance(r.get('checkpoint_sha256'),str) or len(r['checkpoint_sha256'])!=64
            or not isinstance(r.get('return_source_sha256'),str) or len(r['return_source_sha256'])!=64
            or isinstance(value,bool) or not np.isfinite(value) or abs(value)>1+1e-12):
            raise ValueError('Incomplete/unverified training data')
        cells[key]=float(value)
    if set(cells)!=expected:raise ValueError('Missing complete training trials')
    return np.array([[cells[s,d,sign] for d in range(4) for sign in (-1,1)] for s in TRAIN_SEEDS])
