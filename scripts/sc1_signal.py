"""Read-only reward normalization and descriptive lag correlation, not a learner."""
import numpy as np

LAGS=(-40,-20,0,5,10,20,40)
SHIFTS=(500,1000,1500,2000,2500,3000,3500,4000)
SKIP_UPDATES=400


def normalized_signal(rewards):
    rewards=np.asarray(rewards,float)
    if rewards.ndim!=1 or not len(rewards) or not np.isfinite(rewards).all() or np.any(np.abs(rewards)>1):raise ValueError('Require bounded finite reward stream')
    baseline=variance=mass=0.;alpha=-np.expm1(-5./2000.);rows=[]
    for reward in rewards:
        raw=float(reward-baseline)
        if mass==0:baseline=float(reward);raw=0.
        mass+=alpha*(1-mass);effective=alpha/mass
        variance+=effective*(raw*raw-variance)
        scale=max(.01,float(np.sqrt(variance)));signal=float(np.clip(raw/scale,-1,1))
        baseline+=effective*raw
        rows.append((baseline,raw,variance,mass,scale,signal))
    return {k:np.array([r[i] for r in rows]) for i,k in enumerate(('baseline','raw_error','variance','mass','scale','signal'))}


def lag_correlation(credit,signal,lag):
    x,y=np.asarray(credit,float),np.asarray(signal,float)
    if x.ndim!=2 or y.shape!=(len(x),) or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Invalid correlation inputs')
    if isinstance(lag,bool) or int(lag)!=lag or len(y)-abs(lag)<4:raise ValueError('Invalid signed lag')
    if lag>0:x,y=x[:-lag],y[lag:]
    elif lag<0:x,y=x[-lag:],y[:lag]
    x=x-x.mean(0);y=y-y.mean();denominator=np.sqrt((x*x).sum(0)*(y*y).sum())
    valid=denominator>0
    return np.divide(y@x,denominator,out=np.zeros(x.shape[1]),where=valid),valid


def lag_diagnostics(credit,signal):
    credit=np.asarray(credit,float)[SKIP_UPDATES:];signal=np.asarray(signal,float)[SKIP_UPDATES:]
    rows=[]
    for lag in LAGS:
        correlation,valid=lag_correlation(credit,signal,lag)
        rms=float(np.sqrt(np.mean(correlation[valid]**2))) if valid.any() else 0.
        null=[]
        for shift in SHIFTS:
            r,use=lag_correlation(credit,np.roll(signal,shift),lag)
            null.append(float(np.sqrt(np.mean(r[use]**2))) if use.any() else 0.)
        rows.append(dict(lag_control_chunks=lag,lag_ms=lag*5,valid_credit_columns=int(valid.sum()),observed_RMS_correlation=rms,
            circular_shift_RMS_range=[min(null),max(null)],above_all_fixed_shift_values=bool(rms>max(null))))
    return rows
