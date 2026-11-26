"""Fixed cross-birth criteria; never discard an old failure or replace a gate by a mean."""
import numpy as np
from s2_contract import standing_retention

DEV=(10006,10007,10010,10011)
BIRTHS=(1004,1005,1006)


def anatomical_rule(parent,control,candidate):
    if any(set(bank)!=set(DEV) for bank in (parent,control,candidate)):raise ValueError('Require exact four-seed anatomical banks')
    def values(row):
        a=float(row['mode_fractions']['tarsal_only']);q=float(row['mean_non_tarsal_fraction'])
        if not np.isfinite([a,q]).all() or not 0<=a<=1 or not 0<=q<=1:raise ValueError('Invalid anatomical fractions')
        return a,q
    rows={}
    for seed in DEV:
        c,cq=values(candidate[seed]);rows[str(seed)]={}
        for name,bank in (('parent',parent),('control',control)):
            p,pq=values(bank[seed]);rows[str(seed)][name]=dict(tarsal_only_fraction_gain=c-p,mean_non_tarsal_fraction_change=cq-pq)
    passed=all(v['tarsal_only_fraction_gain']>0 and v['mean_non_tarsal_fraction_change']<0 for row in rows.values() for v in row.values())
    return dict(passed=bool(passed),paired=rows,descriptor_is_not_balance_validator=True)


def replication_summary(births):
    if set(births)!=set(BIRTHS):raise ValueError('All three births, including the original1005 failure, are mandatory')
    records=[]
    for birth in BIRTHS:
        banks=births[birth]
        if set(banks)!=set(('parent','control','candidate')) or any(set(bank)!=set((birth,*DEV)) for bank in banks.values()):raise ValueError('Incomplete or extra bank cells')
        for bank in banks.values():
            for seed,entry in bank.items():
                if entry['summary']['seed']!=seed:raise ValueError('Mislabeled frozen seed')
        candidate={s:banks['candidate'][s]['summary'] for s in DEV}
        retention={name:standing_retention({s:banks[name][s]['summary'] for s in DEV},candidate,DEV) for name in ('parent','control')}
        anatomy=anatomical_rule(*({s:banks[name][s]['anatomical'] for s in DEV} for name in ('parent','control','candidate')))
        for name in banks:
            # Validate all own-seed descriptors too, without treating own seed as held out.
            row=banks[name][birth]['anatomical']
            if not np.isfinite(row['mean_non_tarsal_fraction']) or not 0<=row['mean_non_tarsal_fraction']<=1:raise ValueError('Invalid own-seed descriptor')
        records.append(dict(training_seed=birth,retention=retention,anatomical=anatomy,
            combined_pilot_rule_pass=bool(anatomy['passed'] and all(v['passed'] for v in retention.values())),
            absolute_legacy_standing_ge90_count=sum(candidate[s]['standing_fraction']>=.9 for s in DEV),
            minimum_legacy_standing=float(min(candidate[s]['standing_fraction'] for s in DEV)),
            minimum_tarsal_only_fraction=float(min(banks['candidate'][s]['anatomical']['mode_fractions']['tarsal_only'] for s in DEV)),
            own_seed_descriptive={name:dict(summary=banks[name][birth]['summary'],anatomical=banks[name][birth]['anatomical']) for name in banks}))
    return dict(training_birth_count=3,new_training_birth_count=2,births=records,
        combined_pilot_pass_count=sum(r['combined_pilot_rule_pass'] for r in records),
        all_births_meet_original_rule=all(r['combined_pilot_rule_pass'] for r in records),
        shared_development_parent_cells_are_not_independent_replications=True,
        no_best_checkpoint_selected=True,model_promoted=False,walking_proven=False,
        limits='Three declared training seeds with shared development probes; not a statistical efficacy claim or physical ability qualification.')
