"""Read-only standing failure and native reward audit; no simulation or learning."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from compare_exploration import digest,tree_digest,write
from run_duration_study import PARENT,SOURCES,PROBES
from noesis_gpu.learning import standing_mask,score_transition
from noesis_gpu.evaluate import summarize
from credit_diagnostic import replay_rewards

ANCHOR='fb80cfcc3018329209784751570d7ffb1ed33aa9ef48fceaa85588862f7626a8'


def failures(observation,start=10,dt_ms=5):
    height=observation['position'][...,2];up=observation['up_body'][...,2]
    support=(observation['foot_load']>1e-5).any(axis=-1)
    if height.ndim==2:
        if height.shape[1]!=1:raise ValueError('Require one world')
        height,up,support=height[:,0],up[:,0],support[:,0]
    if not 0<=start<len(height):raise ValueError('Invalid measured window')
    reasons={'height_below':height<.3,'height_above':height>1.5,'upright_below':up<.7,'no_tarsus_support':~support}
    invalid=np.logical_or.reduce(list(reasons.values()))
    reference=np.asarray(standing_mask(observation)).reshape(-1)
    if not np.array_equal(~invalid,reference):raise ValueError('Standing definition mismatch')
    measured=invalid[start:];edges=np.diff(np.r_[False,measured,False].astype(int));beg=np.flatnonzero(edges==1);end=np.flatnonzero(edges==-1)
    episodes=[dict(start_ms=int((a+start)*dt_ms),duration_ms=int((b-a)*dt_ms),
        min_height_mm=float(height[start+a:start+b].min()),min_upright=float(up[start+a:start+b].min()),
        left_censored=bool(a==0),right_censored=bool(b==len(measured))) for a,b in zip(beg,end)]
    codes=sum((mask[start:].astype(int)<<bit) for bit,mask in enumerate(reasons.values()))
    return dict(standing_fraction=float((~measured).mean()),failed_samples=int(measured.sum()),
        reasons={key:int(mask[start:].sum()) for key,mask in reasons.items()},
        exclusive_reason_bitmasks={str(c):int((codes==c).sum()) for c in np.unique(codes) if c},
        reason_bit_order=list(reasons),episodes=len(episodes),
        failures_at_least100ms=sum(e['duration_ms']>=100 for e in episodes),
        max_failure_ms=max((e['duration_ms'] for e in episodes),default=0),
        longest_episodes=sorted(episodes,key=lambda e:-e['duration_ms'])[:8],
        diagnostic_duration_cutoff_is_not_a_new_success_gate=True)


def reward_audit(directory):
    report=json.loads((directory/'report.json').read_text())
    if report['status']!='completed_training_birth' or report['completed_chunks']!=5000 or report['learning_updates']!=4990:
        raise ValueError('Require full25s training')
    with np.load(directory/'training-observations.npz',allow_pickle=False) as data:
        d={k:data[k].copy() for k in data.files}
    if any(len(v)!=5000 or not np.isfinite(v).all() for v in d.values()):raise ValueError('Nonfinite/incomplete trace')
    terms={k:[] for k in ('standing','support','progress','effort','slip','cycle')}
    reconstructed=[];v7=[];unclipped=[]
    for i in range(1,5000):
        before={k:d[k][i-1] for k in d};after={k:d[k][i] for k in d}
        score=score_transition(before,after,5.,'gripped-cycle',cycle_credit=float(d['cycle_credit'][i]),cycle_weight=.1)
        stable=bool(standing_mask(after));count=score['supporting_feet']
        component=dict(standing=.4*stable-.8*(1-stable),support=.1*stable*min(count/3,1),
            progress=.6*score['grounded_progress'],effort=-.05*score['effort'],
            slip=-.03*score['stance_slip_mm_s'],cycle=.1*d['cycle_credit'][i]*score['grip_factor']*stable)
        for key,value in component.items():terms[key].append(float(value))
        total=sum(component.values());unclipped.append(total);reconstructed.append(score['reward'])
        if not np.isclose(total,score['reward_unclipped'],rtol=0,atol=1e-12):raise ValueError('Component sum mismatch')
        v7.append(score_transition(before,after,5.,'standing')['reward'])
    error=float(np.max(np.abs(np.asarray(reconstructed)-d['reward'][1:])))
    if error>1e-12:raise ValueError(f'Reward replay mismatch:{error}')
    active=np.arange(1,5000)>=10;measured=np.asarray(reconstructed)[active]
    standing=np.asarray(d['standing'][1:])[active].astype(bool)
    normalized,terminal=replay_rewards(measured)
    with np.load(directory/'weights-last.npz',allow_pickle=False) as bank:
        if any(not np.isclose(value,float(bank[key]),rtol=0,atol=1e-12) for key,value in terminal.items()):
            raise ValueError('Normalizer terminal state mismatch')
    signal=normalized['learning_signal']
    return dict(max_absolute_reward_replay_error=error,transitions_recomputed=4999,
        first_transition_excluded='Initial pre-step body state not stored; no invented reconstruction',
        learning_transitions=4990,training_standing=failures(d),
        reward_mean=float(measured.mean()),positive_reward_while_nonstanding=int(((measured>0)&~standing).sum()),
        nonstanding_learning_samples=int((~standing).sum()),
        positive_learning_signal_while_nonstanding=int(((signal>0)&~standing).sum()),
        negative_learning_signal_while_standing=int(((signal<0)&standing).sum()),
        normalizer_terminal_state_matches_bank=True,
        components={key:dict(mean=float(np.asarray(values)[active].mean()),
            std=float(np.asarray(values)[active].std())) for key,values in terms.items()},
        v7_rescore_mean=float(np.asarray(v7)[active].mean()),
        v7_rescore_scope='OFFLINE reward rescore of SAME v9 trajectory; not a v7 physical rollout or learning outcome',
        inputs={name:digest(directory/name) for name in ('report.json','training-observations.npz')})


def audit(root,core):
    own=Path(__file__).resolve().parents[1]
    if dict(core=tree_digest(core),pilot=tree_digest(own/'gpu-pilot/src'))!=SOURCES:raise ValueError('Source changed')
    parent=root/'gpu-pilot/outputs/e4-exploration-20261029/train-control-1000'
    ancestry=json.loads((parent/'report.json').read_text())
    if digest(parent/'weights-last.npz')!=PARENT or ancestry['import_checkpoint_sha256']!=ANCHOR:
        raise ValueError('Parent ancestry mismatch')
    rows={}
    for seed in PROBES:
        directory=root/f'gpu-pilot/outputs/duration-t1b-guarded-20261031/frozen-parent-{seed}'
        report,summary=summarize(directory)
        if report['checkpoint_sha256']!=PARENT or report['seeds']!=[seed]:raise ValueError('Frozen bank/seed mismatch')
        with np.load(directory/'observations.npz',allow_pickle=False) as data:
            obs={k:data[k].copy() for k in ('position','up_body','foot_load')}
        detail=failures(obs)
        if detail['standing_fraction']!=summary['standing_fraction']:raise ValueError('Standing metric mismatch')
        rows[str(seed)]=dict(**detail,inputs={n:digest(directory/n) for n in ('report.json','observations.npz')})
    trainings={name:reward_audit(root/f'gpu-pilot/outputs/{folder}/train25') for name,folder in
        [('T1B','duration-t1b-guarded-20261031'),('L2','learning-rate-l2-20261031')]}
    return dict(status='offline_standing_and_native_reward_replay_complete',source_hashes=SOURCES,
        current_parent_sha256=PARENT,standing_ancestor_sha256=ANCHOR,
        parent_ancestry=dict(seed=ancestry['seed'],profile=ancestry['reward_objective'],seconds=ancestry['simulated_seconds'],
            learning_rate=ancestry['learning_rate'],report_sha256=digest(parent/'report.json')),
        frozen_parent=rows,trainings=trainings,thresholds_changed=False,model_promoted=False,
        new_simulation_started=False,scope='Existing observations only; ancestral four-seed standing not inferred from descendant failure or historical two-seed gate')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('shared-root','core-source','output'):ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args();result=audit(args.shared_root.resolve(),args.core_source.resolve())
    write(args.output,result);print(json.dumps(dict(status=result['status'],parent_probes=len(result['frozen_parent']),training_replays=len(result['trainings']))))
