"""OA2 hypothetical training-buffer decisions; never writes/adopts a policy."""
import numpy as np
from mp1_select import TRAIN_SEEDS,select_policy
from oa1_load_reference import load_references
from ps1_objective import physical_score
from ew1_contract import episode_rewards


def case_key(context):
    seed,d,sign=(context.get(k) for k in ('seed','direction','sign'))
    if type(seed) is not int or seed not in TRAIN_SEEDS:
        raise ValueError('Only original training contexts allowed')
    phase,bank=context.get('phase'),context.get('bank')
    if phase=='evaluation' and bank=='parent' and d is None and sign is None:
        return seed,-1,0
    if (phase=='trial' and bank=='probe' and type(d) is int and d in range(4)
            and type(sign) is int and sign in (-1,1)):
        return seed,d,sign
    raise ValueError('Only training probes and training-parent controls allowed')


def counterfactual_returns(observations,anatomy,model):
    original=episode_rewards(observations,anatomy,model)
    for name in ('root_actuator_force','ground_force'):
        a=np.asarray(anatomy[name])
        if a.shape!=(1001,3) or not np.isfinite(a).all():
            raise ValueError('Require complete measured force arrays')
    region=anatomy['region_vertical_reaction'][1:]
    np.testing.assert_allclose(region.sum(1),anatomy['ground_force'][1:,2],rtol=1e-12,atol=1e-12)
    weight=model['total_mass']*abs(model['gravity'][2])
    scores=physical_score(region,anatomy['com_velocity'][1:],observations['omega'][:,0],
        observations['up_body'][:,0,2],observations['foot_load'][:,0],weight)
    refs=load_references(region.sum(1),anatomy['root_actuator_force'][1:,2],weight)
    np.testing.assert_array_equal(refs['raw_load_quality'],scores['vertical_load_quality'])
    other=scores['upright_quality']*scores['tarsal_share']*scores['claw_support_quality']*scores['motion_quality']
    # Keep the same bounded-product rule, warmup and averaging; only load changes.
    alternative=2*np.clip(other*refs['actuator_adjusted_load_quality'],0,1)-1
    return float(original[10:].mean()),float(alternative[10:].mean())


def decision_diagnostic(rows):
    expected={(s,-1,0) for s in TRAIN_SEEDS}|{(s,d,k) for s in TRAIN_SEEDS for d in range(4) for k in (-1,1)}
    cells={}
    for row in rows:
        key=case_key(row['context'])
        if key in cells:raise ValueError('Duplicate case')
        vals=[row[k] for k in ('original_return','alternative_return')]
        if any(isinstance(v,bool) or not np.isscalar(v) or not np.isfinite(v) or abs(v)>1+1e-12 for v in vals):
            raise ValueError('Require bounded finite episode returns')
        cells[key]=vals
    if set(cells)!=expected:raise ValueError('Require all18original training cases')
    result={}
    for i,name in enumerate(('original','alternative')):
        parent=np.array([cells[s,-1,0][i] for s in TRAIN_SEEDS])
        probes=np.array([[cells[s,d,k][i] for d in range(4) for k in (-1,1)] for s in TRAIN_SEEDS])
        result[name]=dict(parent_returns=parent,probe_returns=probes,
            decisions={b:select_policy(parent,probes,sign) for b,sign in (('candidate',1),('negative',-1))})
    result['policy_identity_changed']={b:result['original']['decisions'][b]['policy_index']!=result['alternative']['decisions'][b]['policy_index']
        for b in ('candidate','negative')}
    return result
