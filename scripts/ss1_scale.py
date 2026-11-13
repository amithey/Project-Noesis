"""SS1 finite-probe scale compatibility; no learner or motor policy."""
from pathlib import Path
import numpy as np
from mp2_select import TRAIN

SCALES=(.01,.005,.0025)
NEW_SCALES=SCALES[1:]
SIGNAL=1e-6
COMPATIBILITY=.25


def key(c,baseline=False):
    s,d,k,h=(c.get(n) for n in ('seed','direction','sign','amplitude'))
    if type(s) is not int or s not in TRAIN:raise ValueError('Only registered TRAIN contexts')
    if baseline:
        if s!=1010 or c.get('phase')!='baseline' or c.get('bank')!='parent' or any(v is not None for v in (d,k,h)):
            raise ValueError('Unregistered baseline')
        return s,0.,0
    if c.get('phase')=='parent' and c.get('bank')=='parent' and all(v is None for v in (d,k,h)):return s,0.,0
    if (c.get('phase')=='trial' and c.get('bank')=='probe' and type(d) is int and d==0
            and type(k) is int and k in (-1,1) and not isinstance(h,bool) and h in SCALES):return s,float(h),k
    raise ValueError('Wrong role/direction/scale')


def validate_worker(a):
    key(vars(a),baseline=a.phase=='baseline')
    if a.phase!='baseline' and a.amplitude not in NEW_SCALES:raise ValueError('Reuse sealed original scale; no parent rerun')
    if a.output.resolve().is_relative_to(a.runtime_root.resolve()) or a.output.exists():
        raise ValueError('Require new output outside immutable runtime/evidence root')


def panel(records):
    expected={(s,0.,0) for s in TRAIN}|{(s,h,k) for s in TRAIN for h in SCALES for k in (-1,1)};cells={}
    for r in records:
        cell=key(r['context']);value=r['return']
        if cell in cells:raise ValueError('Duplicate case')
        if (r.get('complete_horizon') is not True or r.get('actual_import_verified') is not True
                or r.get('warmup_samples')!=10 or r.get('measured_samples')!=990
                or r.get('return_objective')!='physical-standing-ps1-v1' or isinstance(value,bool)
                or not np.isscalar(value) or not np.isfinite(value) or abs(value)>1+1e-12):raise ValueError('Invalid/incomplete return')
        for field in ('checkpoint_sha256','return_source_sha256'):
            h=r.get(field)
            if not isinstance(h,str) or len(h)!=64 or any(x not in '0123456789abcdef' for x in h):raise ValueError('Invalid provenance')
        cells[cell]=float(value)
    if set(cells)!=expected:raise ValueError('Require all42TRAIN cases')
    return np.array([cells[s,0.,0] for s in TRAIN]),np.array([[[cells[s,h,k] for k in (-1,1)] for h in SCALES] for s in TRAIN])


def analyze(parent,returns):
    p,r=np.asarray(parent,float),np.asarray(returns,float)
    if p.shape!=(6,) or r.shape!=(6,3,2) or not all(np.isfinite(x).all() and np.all(abs(x)<=1+1e-12) for x in (p,r)):
        raise ValueError('Require bounded six-context/three-scale symmetric returns')
    odd=(r[:,:,1]-r[:,:,0])/2;even=(r[:,:,1]+r[:,:,0])/2-p[:,None]
    slopes=odd/np.array(SCALES)[None,:]
    signal=np.all(abs(odd[:,1:])>SIGNAL,axis=1)
    same_sign=np.sign(slopes[:,1])==np.sign(slopes[:,2])
    discrepancy=abs(slopes[:,1]-slopes[:,2])/np.maximum(np.maximum(abs(slopes[:,1]),abs(slopes[:,2])),1e-12)
    ratios=abs(even[:,1:])/np.maximum(abs(odd[:,1:]),SIGNAL)
    compatible=signal&same_sign&(discrepancy<=COMPATIBILITY)&np.all(ratios<=COMPATIBILITY,axis=1)
    cross=bool(np.all(abs(odd[:,-1])>SIGNAL) and np.all(np.sign(slopes[:,-1])==np.sign(slopes[0,-1])))
    return dict(odd_amplitude=odd,midpoint_departure=even,central_slopes=slopes,signal_present=signal,
        same_small_scale_sign=same_sign,relative_slope_discrepancy=discrepancy,small_scale_midpoint_odd_ratios=ratios,
        context_compatible=compatible,compatible_contexts=int(compatible.sum()),all_contexts_compatible=bool(compatible.all()),
        smallest_scale_cross_context_sign_consistent=cross,
        limits='One prechosen direction and finite scales on existing TRAIN contexts. Engineering compatibility thresholds, not calibrated gradients, a smaller-step remedy, efficacy or physical skill.')
