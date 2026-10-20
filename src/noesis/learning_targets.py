"""Anatomical candidates for plasticity, selected from existing motor projections.

Class labels and absolute original synaptic weights choose a fixed target set.
This selection is not a gait generator, rhythm reference or motor command.
"""
import numpy as np

from noesis.motor import leg_motor_table
from noesis.sensory import LEGS


def select_premotor_targets(W, neurons, mapped_motors, *, local_per_leg=8, sequential_count=16):
    """Return deterministic local/cross-leg target identities without mutation.

    Only annotated local or sequential leg-neuromere cells with existing direct
    projections to mapped leg motor neurons qualify. A fixed per-leg count
    improves anatomical coverage; it prescribes no phase, joint angle or action.
    Use original connectome W, before loading learned weights. Neurotransmitter
    signs are not inferred here: absolute projection weights rank candidates.
    """
    for value in (local_per_leg,sequential_count):
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or value<0:
            raise ValueError('target counts must be nonnegative integers')
    n=len(neurons)
    if W.shape!=(n,n) or not np.array_equal(neurons.index.to_numpy(),np.arange(n)):
        raise ValueError('neuron rows must be aligned to square connectivity indices')
    W=W.tocsr(copy=False)
    if not np.isfinite(W.data).all():
        raise ValueError('connectivity must be finite')
    mapped=np.asarray(mapped_motors)
    if mapped.ndim!=1 or mapped.dtype.kind not in 'iu' or np.any(mapped<0) or np.any(mapped>=n):
        raise ValueError('mapped motor indices must be valid integers')
    table=leg_motor_table(neurons)
    table=table[table.idx.isin(mapped)]
    edges=np.flatnonzero(np.isin(W.indices,table.idx.to_numpy()))
    pre=np.searchsorted(W.indptr,edges,side='right')-1
    post=W.indices[edges]
    weight=np.abs(W.data[edges])
    labels=neurons['Class'].fillna('').to_numpy()
    local_mask=labels=='single_leg_neuromere'
    sequential_mask=labels=='sequential_leg_neuromeres'
    selected_local=set()
    per_leg={}
    for leg in LEGS:
        ids=table.loc[table.leg.eq(leg),'idx'].to_numpy()
        take=np.isin(post,ids)
        score=np.bincount(pre[take],weights=weight[take],minlength=n)
        candidates=np.flatnonzero(local_mask&(score>0))
        chosen=sorted(candidates.tolist(),key=lambda i:(-score[i],i))[:local_per_leg]
        selected_local.update(chosen)
        per_leg[leg]={'candidate_count':len(candidates),'selected':chosen,
                      'selected_original_abs_projection':[float(score[i]) for i in chosen]}
    score=np.bincount(pre,weights=weight,minlength=n)
    candidates=np.flatnonzero(sequential_mask&(score>0))
    cross=sorted(candidates.tolist(),key=lambda i:(-score[i],i))[:sequential_count]
    local=sorted(selected_local)
    return {'indices':sorted(selected_local|set(cross)),'local_indices':local,
            'sequential_indices':sorted(cross),'per_leg':per_leg,
            'sequential_candidate_count':len(candidates),'mapped_leg_motor_count':len(table),
            'selection_uses_original_connectome_only':True}
