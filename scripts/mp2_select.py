"""MP2 six-context acceptance of an existing whole policy, never a motor controller."""
import numpy as np

OLD=(1010,1011)
NEW=(1012,1013,1014,1015)
TRAIN=OLD+NEW
MIN_GAIN=1e-6


def key(context,baseline=False):
    seed,d,sign=(context.get(k) for k in ('seed','direction','sign'))
    if baseline:
        valid=(seed==1010 and type(seed) is int and context.get('phase')=='baseline'
            and context.get('bank')=='parent' and d is None and sign is None)
        if not valid:raise ValueError('Unregistered baseline')
        return seed,-1,0
    if type(seed) is not int or seed not in TRAIN:raise ValueError('Unregistered training context')
    if context.get('phase')=='parent' and context.get('bank')=='parent' and d is None and sign is None:
        return seed,-1,0
    if (context.get('phase')=='trial' and context.get('bank')=='probe' and type(d) is int
            and d in range(4) and type(sign) is int and sign in (-1,1)):
        return seed,d,sign
    raise ValueError('Evaluation/baseline/wrong-role data cannot enter training')


def select(parent,probes,reward_sign=1):
    parent,probes=np.asarray(parent,float),np.asarray(probes,float)
    if parent.shape!=(6,) or probes.shape!=(6,8):raise ValueError('Require six-context/eight-policy panel')
    if type(reward_sign) is not int or reward_sign not in (-1,1):raise ValueError('Invalid reward sign')
    if not all(np.isfinite(a).all() and np.all(abs(a)<=1+1e-12) for a in (parent,probes)):
        raise ValueError('Require finite bounded returns')
    margins=(reward_sign*(probes-parent[:,None])).min(axis=0)
    best=int(np.argmax(margins));accepted=bool(margins[best]>MIN_GAIN)
    return dict(policy_index=best if accepted else -1,accepted_for_training_update=accepted,
        parent_fallback=not accepted,worst_training_gain=float(margins[best]),reward_sign=reward_sign,
        training_condition_not_independent_efficacy=True,model_promoted=False)


def panel(records):
    expected={(s,-1,0) for s in TRAIN}|{(s,d,k) for s in TRAIN for d in range(4) for k in (-1,1)}
    cells={}
    for r in records:
        cell=key(r['context']);value=r['return']
        if cell in cells:raise ValueError('Duplicate training case')
        if (r.get('complete_horizon') is not True or r.get('actual_import_verified') is not True
                or r.get('warmup_samples')!=10 or r.get('measured_samples')!=990
                or r.get('return_objective')!='physical-standing-ps1-v1'
                or isinstance(value,bool) or not np.isscalar(value) or not np.isfinite(value) or abs(value)>1+1e-12):
            raise ValueError('Incomplete or invalid original-objective return')
        for field in ('checkpoint_sha256','return_source_sha256'):
            h=r.get(field)
            if not isinstance(h,str) or len(h)!=64 or any(c not in '0123456789abcdef' for c in h):
                raise ValueError('Missing provenance')
        cells[cell]=float(value)
    if set(cells)!=expected:raise ValueError('Require all54complete training cases')
    parent=np.array([cells[s,-1,0] for s in TRAIN])
    probes=np.array([[cells[s,d,k] for d in range(4) for k in (-1,1)] for s in TRAIN])
    return parent,probes
