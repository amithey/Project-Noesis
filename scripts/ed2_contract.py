"""Fixed ED2 cohort and original pilot criteria; no winner/threshold selection."""
import numpy as np
from ap2_contract import DEV,anatomical_rule
from s2_contract import standing_retention

BIRTHS=(1005,1006,1007)
NEW_BIRTHS=(1006,1007)


def configuration(phase,bank,training_seed,seed):
    if any(isinstance(x,bool) or not isinstance(x,(int,np.integer)) for x in (training_seed,seed)):raise ValueError('Require integer registered seeds')
    if phase=='train':
        valid=(bank=='replay' and training_seed==seed==1005) or (bank in ('control','candidate') and training_seed in NEW_BIRTHS and seed==training_seed)
    elif phase=='frozen':
        valid=(bank=='parent' and training_seed==seed==1007) or (bank in ('control','candidate') and training_seed in NEW_BIRTHS and seed in (training_seed,*DEV))
    else:valid=False
    if not valid:raise ValueError('Unregistered ED2 phase/bank/training/probe seed')
    return 10 if bank in ('replay','candidate') else 0


def replication_summary(births):
    if set(births)!=set(BIRTHS):raise ValueError('All three births including original failed1005 required')
    rows=[]
    for birth in BIRTHS:
        banks=births[birth]
        if set(banks)!=set(('parent','control','candidate','v7','AP1','previous_PS1')):raise ValueError('Missing reference bank')
        for bank in ('parent','control','candidate'):
            if set(banks[bank])!=set((birth,*DEV)):raise ValueError('Incomplete main frozen bank')
        for bank in ('v7','AP1','previous_PS1'):
            if set(banks[bank])!=set(DEV):raise ValueError('Incomplete auxiliary bank')
        for bank in banks.values():
            for seed,entry in bank.items():
                if entry['summary']['seed']!=seed:raise ValueError('Mislabeled frozen seed')
        child={s:banks['candidate'][s]['summary'] for s in DEV}
        retention={b:standing_retention({s:banks[b][s]['summary'] for s in DEV},child,DEV) for b in ('parent','control','v7','AP1','previous_PS1')}
        anatomy=anatomical_rule(*({s:banks[b][s]['anatomical'] for s in DEV} for b in ('parent','control','candidate')))
        paired={};passed=True
        fields=('mean_PS1_quality','mean_COM_speed_mm_s','mean_angular_speed_rad_s','mean_non_tarsal_fraction')
        for seed in DEV:
            paired[str(seed)]={};c=banks['candidate'][seed]['physical']
            for b in ('parent','control'):
                r=banks[b][seed]['physical'];delta={k:c[k]-r[k] for k in fields}
                if not np.isfinite(list(delta.values())).all():raise ValueError('Nonfinite physical comparison')
                ok=delta[fields[0]]>0 and all(delta[k]<0 for k in fields[1:]);passed=passed and ok
                paired[str(seed)][b]=dict(changes=delta,all_directions_improve=bool(ok))
        mean_delta={k:float(np.mean([banks['candidate'][s]['physical'][k]-banks['control'][s]['physical'][k] for s in DEV])) for k in fields}
        descriptive=mean_delta[fields[0]]>0 and all(mean_delta[k]<0 for k in fields[1:])
        rows.append(dict(training_seed=birth,retention=retention,anatomical=anatomy,paired_physical=paired,
            all_four_development_physical_directions_improve=bool(passed),combined_exploratory_pilot_pass=bool(passed and anatomy['passed'] and all(retention[b]['passed'] for b in ('parent','control','v7','AP1'))),
            descriptive_mean_changes_vs_control=mean_delta,descriptive_all_mean_directions_improve=bool(descriptive),
            own_seed_descriptive={b:banks[b][birth] for b in ('parent','control','candidate')}))
    return dict(training_birth_count=3,new_training_birth_count=2,births=rows,combined_pilot_pass_count=sum(r['combined_exploratory_pilot_pass'] for r in rows),
        descriptive_mean_direction_birth_count=sum(r['descriptive_all_mean_directions_improve'] for r in rows),
        descriptive_means_never_rescue_primary_failure=True,original1005_required=True,no_winner_selected=True,
        physical_balance_validated=False,model_promoted=False,walking_proven=False,
        limits='Three training births sharing development references, not independent frozen replicates or validated physical ability. Original failure retained; no mean-based rescue.')
