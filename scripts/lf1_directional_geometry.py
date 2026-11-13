"""TRAIN-only finite-probe surrogate geometry; no actual weight/policy update.

LP interface: https://docs.scipy.org/doc/scipy/reference/optimize.linprog-highs.html
Surrogate feasibility is not a physical gain, calibrated derivative or efficacy.
"""
import numpy as np
from scipy.optimize import linprog

STEP=.01
RADIUS=.01
TOL=1e-8
GAIN=1e-6


def certificate(g,coefficients,context_weights):
    g,c,w=(np.asarray(x,float) for x in (g,coefficients,context_weights))
    if g.shape!=(6,4) or c.shape!=(4,) or w.shape!=(6,) or not all(np.isfinite(x).all() for x in (g,c,w)):
        raise ValueError('Invalid certificate arrays')
    norm=float(abs(c).sum())
    if norm>RADIUS+TOL or np.any(w<-TOL) or abs(w.sum()-1)>TOL:raise ValueError('Infeasible certificate')
    # Enforce exact feasible witnesses rather than trusting solver tolerances.
    c=c*min(1.,RADIUS/norm) if norm else c.copy()
    w=np.maximum(w,0);w=w/w.sum()
    lower=float((g@c).min());upper=float(RADIUS*abs(g.T@w).max());gap=upper-lower
    if gap < -TOL or gap>TOL:raise ValueError('Primal/dual certificate gap exceeds tolerance')
    classification='positive_surrogate_margin' if lower>GAIN+TOL else 'no_margin_above_threshold' if upper<GAIN-TOL else 'threshold_ambiguous'
    return dict(coefficients=c,context_weights=w,lower_bound=lower,upper_bound=upper,duality_gap=gap,
        classification=classification,numerical_tolerance=TOL,not_an_actual_policy=True)


def robust_linear(g,solver=linprog):
    g=np.asarray(g,float)
    if g.shape!=(6,4) or not np.isfinite(g).all():raise ValueError('Require six-context/four-direction matrix')
    # gamma=p-n; sum(p+n)<=radius; maximize t with G gamma >= t.
    aub=np.vstack((np.column_stack((-g,g,np.ones(6))),np.r_[np.ones(8),0.]))
    primal=solver(np.r_[np.zeros(8),-1.],A_ub=aub,b_ub=np.r_[np.zeros(6),RADIUS],
        bounds=[(0,None)]*8+[(None,None)],method='highs',options=dict(time_limit=20.,primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9))
    # Independent dual: minimize radius*u over simplex weights, |G.T w|<=u.
    dual=solver(np.r_[np.zeros(6),RADIUS],A_ub=np.vstack((np.column_stack((g.T,-np.ones(4))),
        np.column_stack((-g.T,-np.ones(4))))),b_ub=np.zeros(8),A_eq=np.array([np.r_[np.ones(6),0.]]),
        b_eq=np.ones(1),bounds=[(0,None)]*7,method='highs',
        options=dict(time_limit=20.,primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9))
    if not all(r.success and r.status==0 for r in (primal,dual)):raise ValueError('LP solver failed; no geometry verdict')
    px,dx=np.asarray(primal.x,float),np.asarray(dual.x,float)
    if px.shape!=(9,) or dx.shape!=(7,) or not all(np.isfinite(x).all() for x in (px,dx)):
        raise ValueError('Invalid solver output')
    if (np.any(px[:8]<-TOL) or np.any(aub@px-np.r_[np.zeros(6),RADIUS]>TOL)
            or np.any(dx<-TOL) or abs(dx[:6].sum()-1)>TOL
            or np.any(abs(g.T@dx[:6])-dx[-1]>TOL)):
        raise ValueError('Solver returned infeasible solution')
    result=certificate(g,px[:4]-px[4:8],dx[:6])
    if abs(result['lower_bound']-px[-1])>TOL or abs(result['upper_bound']-RADIUS*dx[-1])>TOL:
        raise ValueError('LP objective and independent certificate disagree')
    return result


def geometry(parent,probes,solver=linprog):
    parent,probes=np.asarray(parent,float),np.asarray(probes,float)
    if parent.shape!=(6,) or probes.shape!=(6,8):raise ValueError('Require complete registered training panel')
    if not all(np.isfinite(x).all() and np.all(abs(x)<=1+1e-12) for x in (parent,probes)):
        raise ValueError('Invalid bounded returns')
    minus,plus=probes[:,::2],probes[:,1::2]
    odd=(plus-minus)/2
    midpoint=(plus+minus)/2-parent[:,None]
    g=odd/STEP
    result=robust_linear(g,solver)
    mixed=[bool(np.any(g[:,d]>TOL) and np.any(g[:,d]<-TOL)) for d in range(4)]
    return dict(finite_direction_matrix=g,midpoint_departure=midpoint,odd_amplitude=odd,
        mixed_sign_directions=mixed,midpoint_exceeds_odd_cells=int(np.count_nonzero(abs(midpoint)>abs(odd)+TOL)),
        linear_certificate=result,limits='Posthoc finite-probe linear surrogate only; no validated derivative, actual weight update, causal mechanism or physical improvement.')
