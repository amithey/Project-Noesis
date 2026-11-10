"""WS1 fixed bounded neural-weight counterfactuals, not a motor policy or learned skill.

Only existing checkpoint-selected edge values change. The two registered offsets
are +/-0.05*abs(original reference); all other checkpoint fields are preserved.
There is no clipping, search, leg ordering, rate target or actuator command.
"""
import numpy as np

STEP=.05
BOUND=.5


def bounded_offset(weights,original,direction):
    w,o=np.asarray(weights),np.asarray(original)
    if isinstance(direction,bool) or direction not in (-1,1):raise ValueError('Only fixed minus/plus offsets')
    if w.ndim!=1 or not len(w) or w.shape!=o.shape or w.dtype.kind!='f' or o.dtype.kind!='f':raise ValueError('Require equal nonempty floating edge vectors')
    if not np.isfinite(w).all() or not np.isfinite(o).all() or not np.array_equal(np.sign(w),np.sign(o)):raise ValueError('Invalid current/original edge values')
    magnitude=np.abs(o)
    if np.any(np.abs(w)<(1-BOUND)*magnitude) or np.any(np.abs(w)>(1+BOUND)*magnitude):raise ValueError('Parent violates original reference bounds')
    shifted=w+direction*STEP*magnitude
    if not np.array_equal(np.sign(shifted),np.sign(o)) or np.any(np.abs(shifted)<(1-BOUND)*magnitude) or np.any(np.abs(shifted)>(1+BOUND)*magnitude):raise ValueError('Registered offset does not fit; reject rather than clip')
    return shifted.astype(w.dtype,copy=False)


def checkpoint_variant(checkpoint,direction):
    result={k:np.asarray(v).copy() for k,v in checkpoint.items()}
    result['weights']=bounded_offset(result['weights'],result['initial_weights'],direction)
    return result
