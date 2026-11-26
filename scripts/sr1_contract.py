"""Summarize independent births without counting repeated probes as replications."""
import numpy as np
from s2_contract import standing_retention


def replication_summary(parent_dev,children_dev,own_pairs):
    births={1004,1005,1006};dev={10006,10007,10010,10011}
    if set(parent_dev)!=dev or set(children_dev)!=births or set(own_pairs)!=births:
        raise ValueError('Require all three declared births and all four development parents')
    rows=[]
    metrics=('standing_fraction','mean_stance_slip_mm_s','heading_forward_distance_mm','net_forward_displacement_mm','total_leg_steps')
    for seed in sorted(births):
        child=children_dev[seed];before,after=own_pairs[seed]
        if set(child)!=dev:raise ValueError('Incomplete birth probe bank')
        if any(not np.isfinite(r[k]) for r in (before,after) for k in metrics):raise ValueError('Nonfinite own-seed evidence')
        gains=[child[s]['standing_fraction']-parent_dev[s]['standing_fraction'] for s in sorted(dev)]
        rows.append(dict(training_seed=seed,own_seed_gain={k:after[k]-before[k] for k in metrics},
            own_seed_parent_standing=before['standing_fraction'],own_seed_child_standing=after['standing_fraction'],
            dev_standing_gain_mean=float(np.mean(gains)),dev_standing_gain_median=float(np.median(gains)),
            dev_standing_gain_min=float(np.min(gains)),dev_standing_ge90_count=sum(child[s]['standing_fraction']>=.9 for s in dev),
            dev_standing_improve_count=sum(child[s]['standing_fraction']>parent_dev[s]['standing_fraction'] for s in dev),
            dev_child_standing={str(s):child[s]['standing_fraction'] for s in sorted(dev)},
            strict_screen=standing_retention(parent_dev,child,sorted(dev))))
    return dict(training_birth_count=3,new_training_birth_count=2,probes_per_dev_bank=4,
        banks=rows,strict_screen_pass_count=sum(r['strict_screen']['passed'] for r in rows),
        descriptive_only=True,training_seed_variability_not_fully_estimated=True,model_promoted=False,walking_proven=False)
