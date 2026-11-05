"""Offline development diagnostic; exact core normalization, temporal reference ranks."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import numpy as np

STATE_KEYS = ('reward_baseline', 'reward_variance', 'reward_normalization_mass')


def replay_rewards(rewards, dt_ms=5.):
    """Run the actual normalizer; no physical/neural dynamics or learned weights."""
    from noesis.plasticity import MotorPlasticity
    rewards=np.asarray(rewards,dtype=float)
    if rewards.ndim!=1 or not len(rewards) or not np.isfinite(rewards).all():
        raise ValueError('Require finite one-dimensional rewards')
    # MotorPlasticity updates normalization independently of spikes/eligibility.
    net=SimpleNamespace(n=1,indptr=np.array([0,1]),indices=np.array([0]),data=np.array([1.]))
    learner=MotorPlasticity(net,[0],learning_rate=.2,tau_eligibility_ms=100,max_change=.5,
        rule='perturbation',normalize_reward_error=True,bias_correct_reward_error=True)
    names=('reward','prediction_error','learning_signal','reward_baseline','reward_variance')
    rows=[learner.update(np.array([],dtype=np.int64),dt_ms,float(reward)) for reward in rewards]
    return ({key:np.array([row[key] for row in rows]) for key in names},
            {key:float(getattr(learner,key)) for key in STATE_KEYS})


def lag_correlations(credit, signal, max_lag=100):
    """Pearson coefficients with a common central X window; signed lags are valid.

    All circular offsets are returned for reference shifts. Observed +/-max_lag
    coefficients contain NO wrapped samples. Circular references assume temporal
    exchangeability, which a short developing closed-loop episode need not satisfy.
    """
    x,y=np.asarray(credit,dtype=float),np.asarray(signal,dtype=float)
    if (x.ndim!=2 or y.ndim!=1 or len(x)!=len(y) or not np.isfinite(x).all()
            or not np.isfinite(y).all() or max_lag<1 or len(y)<=4*max_lag):
        raise ValueError('Invalid credit/signal shape, values or lag window')
    n=len(y); start=max_lag; width=n-2*max_lag
    middle=x[start:start+width]; middle=middle-middle.mean(axis=0)
    xnorm=np.sqrt(np.sum(middle*middle,axis=0)); active=xnorm>0
    if not active.any():raise ValueError('No varying credit channels')
    padded=np.zeros_like(x); padded[start:start+width]=middle
    products=np.conj(np.fft.rfft(padded,axis=0))*np.fft.rfft(y)[:,None]
    dots=np.fft.irfft(products,n=n,axis=0)
    yy=np.concatenate((y,y)); prefix=np.r_[0.,np.cumsum(yy)]; squared=np.r_[0.,np.cumsum(yy*yy)]
    positions=(start+np.arange(n))%n
    sums=prefix[positions+width]-prefix[positions]
    variance=np.maximum(squared[positions+width]-squared[positions]-sums*sums/width,0)
    denominators=np.sqrt(variance[:,None])*xnorm[None,:]
    correlations=np.divide(dots,denominators,out=np.zeros_like(dots),where=denominators>0)
    return np.clip(correlations,-1,1),active,width


def association_reference(credit,signal,max_lag=100,draws=499,seed=271828):
    corr,active,width=lag_correlations(credit,signal,max_lag)
    n=len(signal); family=corr[:max_lag+1,active]
    best=np.unravel_index(np.argmax(np.abs(family)),family.shape)
    observed=float(np.abs(family[best]))
    # Exclude offsets whose search window overlaps the observed +/-lag family.
    choices=np.arange(2*max_lag+1,n-2*max_lag)
    if not len(choices):raise ValueError('Insufficient temporal separation for reference shifts')
    shifts=np.random.default_rng(seed).choice(choices,size=draws,replace=True)
    maxima=np.array([np.max(np.abs(corr[(offset+np.arange(max_lag+1))%n][:,active])) for offset in shifts])
    return dict(observed_max_abs_r=observed,best_signed_r=float(family[best]),
        best_lag_steps=int(best[0]),best_column=int(np.flatnonzero(active)[best[1]]),
        varying_columns=int(active.sum()),constant_columns=int((~active).sum()),
        common_window_samples=width,lag_search_steps=[0,max_lag],reference_draws=draws,
        reference_seed=seed,reference_99th_percentile=float(np.quantile(maxima,.99)),
        circular_shift_reference_rank=float((1+np.count_nonzero(maxima>=observed))/(draws+1)),
        negative_lag_max_abs_r=float(np.max(np.abs(corr[-max_lag:,active]))),
        rank_is_calibrated_p_value=False,
        interpretation='Development association only. Stationarity/exchangeability not established; no causal or absent-learning conclusion.')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--noesis-source',type=Path,required=True)
    ap.add_argument('--birth',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args(); source=args.noesis_source.resolve(); birth=args.birth.resolve()
    from runtime_diagnostic import CORE,PARENT
    from compare_exploration import digest
    if subprocess.check_output(['git','-C',str(source.parent),'rev-parse','HEAD'],text=True).strip()!=CORE:
        raise ValueError('Pinned core required')
    sys.path.insert(0,str(source))
    import noesis.plasticity
    if Path(noesis.plasticity.__file__).resolve()!=source/'noesis/plasticity.py':
        raise ValueError('Wrong imported core')
    from noesis_gpu.run import source_digest
    report=json.loads((birth/'report.json').read_text(encoding='utf-8'))
    expected={'status':'completed_training_birth','completed_chunks':1000,'learning_updates':990,
              'simulated_seconds':5,'learning':True,'learning_rate':.2,
              'import_checkpoint_sha256':PARENT,'noesis_source_sha256':source_digest(source)}
    for key,value in expected.items():
        if report.get(key)!=value:raise ValueError(f'Birth provenance/completion mismatch:{key}')
    if digest(birth/'weights-last.npz')!=report['saved_checkpoint_sha256']:raise ValueError('Bank hash mismatch')
    with np.load(birth/'training-observations.npz',allow_pickle=False) as trace,np.load(birth/'exploration-log.npz',allow_pickle=False) as log,np.load(birth/'weights-last.npz',allow_pickle=False) as bank:
        active=np.flatnonzero(trace['learning_active'])
        if not np.array_equal(active,np.arange(10,1000)) or not np.array_equal(log['control_chunk_index'],active):
            raise ValueError('Log alignment mismatch')
        if not np.array_equal(log['motor_indices'],bank['motor_indices']) or float(log['control_ms'])!=5:
            raise ValueError('Column/time metadata mismatch')
        if not bool(bank['normalize_reward_error']) or not bool(bank['bias_correct_reward_error']):
            raise ValueError('Unexpected normalizer configuration')
        metrics,state=replay_rewards(trace['reward'][active])
        if not all(state[key]==float(bank[key]) for key in STATE_KEYS):
            raise ValueError('Exact final reward-state replay failed')
        credit=log['credit'].copy(); motor_indices=log['motor_indices'].copy()
    if credit.shape!=(990,len(motor_indices)) or np.any(np.abs(credit)>1):raise ValueError('Invalid credit log')
    association=association_reference(credit,metrics['learning_signal'])
    association['best_motor_index']=int(motor_indices[association['best_column']])
    result=dict(status='completed_development_credit_diagnostic',seed=report['seed'],
        core_commit=CORE,core_source_sha256=source_digest(source),
        source_plasticity_sha256=hashlib.sha256(Path(noesis.plasticity.__file__).read_bytes()).hexdigest(),
        input_sha256={name:digest(birth/name) for name in ['report.json','weights-last.npz','training-observations.npz','exploration-log.npz']},
        exact_final_reward_state_match=True,final_state=state,association=association,
        model_promoted=False,walking_proven=False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
