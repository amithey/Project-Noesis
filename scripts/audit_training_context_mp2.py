"""Independent terminal raw-return/selection/receipt audit; no model adoption."""
import argparse
import json
from pathlib import Path
import numpy as np
import run_training_context_mp2 as run
from mp2_select import TRAIN,MIN_GAIN


def main(out):
    out=out.resolve();p=run.read(out/'protocol.json');result=run.read(out/'result.json')
    if result['status']!='completed_MP2_training_panel':raise ValueError('Require complete terminal allocation')
    if not all(result[k] for k in ('source_input_data_created_unchanged','process_tree_empty_verified',
            'lease_released','awake_request_released','baseline_exact')):raise ValueError('Failed terminal guard')
    if result['seconds']>run.LIMIT:raise ValueError('Aggregate budget exceeded')
    run.check_pins({**p['source_hashes'],**p['old_hashes'],**p['data_hashes'],**result['created_hashes']})
    root=Path(p['runtime_root']);_,old,_=run.legacy(root)
    if (root/'work/Project-Noesis/gpu-pilot/outputs/codex-training-host.lock').exists():raise ValueError('Heavy lease active')
    records=run.reused_records(root,old)
    if len(result['jobs'])!=37:raise ValueError('Require all37jobs')
    for actual,expected in zip(result['jobs'],run.jobs()):
        if any(actual[k]!=v for k,v in expected.items()):raise ValueError('Unexpected/misordered worker')
        name=expected['name'];receipt=run.read(out/(name+'-receipt.json'))
        if (receipt!=actual['receipt'] or receipt['exit_code']!=0 or receipt['timed_out']
                or not receipt['tree_empty'] or receipt['seconds']>300):raise ValueError('Incomplete worker receipt')
        folder=out/name;report=run.read(folder/'report.json');cp=run.checkpoint(p,expected)
        run.validate_report(report,expected,run.sha(cp),p['source_hashes'][str(run.HERE/'mp2_worker.py')])
        if report['mp2_context']!=run.read(out/(name+'-mp2-context.json')):raise ValueError('Context mismatch')
        reference=old.out/'parent-1010' if expected['phase']=='baseline' else out/f"parent-{expected['seed']}"
        old.native.compare(reference,folder)
        if expected['phase']=='baseline':
            for n in ('observations.npz','adhesion-observations.npz','anatomy-observations.npz'):
                old.native.exact(reference/n,folder/n)
            if report['spikes_per_world']!=run.read(reference/'report.json')['spikes_per_world']:raise ValueError('Baseline spikes')
        else:records.append(run.measured_record(folder,{k:expected[k] for k in ('phase','bank','seed','direction','sign')},old))
    if records!=run.read(out/'training-returns.json'):raise ValueError('Raw returns not reproduced')
    # Independent loops reproduce panel/decision, including strict threshold and tie.
    cells={}
    for r in records:
        c=r['context'];s=c['seed'];i=-1 if c['phase']=='parent' else 2*c['direction']+(c['sign']==1)
        if (s,i) in cells:raise ValueError('Duplicate return')
        cells[s,i]=r['return']
    if set(cells)!={(s,i) for s in TRAIN for i in range(-1,8)}:raise ValueError('Wrong training panel')
    parent=np.array([cells[s,-1] for s in TRAIN]);probes=np.array([[cells[s,i] for i in range(8)] for s in TRAIN])
    state=run.arrays(out/'optimizer-state.npz');selection=run.read(out/'selection.json')
    np.testing.assert_array_equal(parent,state['parent_returns']);np.testing.assert_array_equal(probes,state['policy_returns'])
    np.testing.assert_array_equal(state['training_seeds'],TRAIN)
    if (json.loads(str(state['selection_json']))!=selection or json.loads(str(state['return_records_json']))!=records
            or str(state['parent_sha256'])!=p['parent_sha256']):raise ValueError('State/provenance mismatch')
    original=run.read(old.out/'selection.json')
    for bank,sign in (('candidate',1),('negative',-1)):
        if old.select_policy(parent[:2],probes[:2],sign)!=original[bank]:raise ValueError('Original decision mismatch')
        margins=[min(sign*(cells[s,i]-cells[s,-1]) for s in TRAIN) for i in range(8)]
        best=max(range(8),key=lambda i:margins[i]);accepted=margins[best]>MIN_GAIN;index=best if accepted else -1
        got=selection[bank]
        if (got['policy_index']!=index or got['parent_fallback']!=(index<0)
                or got['accepted_for_training_update']!=accepted):raise ValueError('Independent decision mismatch')
        np.testing.assert_allclose(got['worst_training_gain'],margins[best],rtol=1e-12,atol=1e-14)
        d,k=divmod(index,2);source=run.checkpoint(p,dict(bank='parent' if index<0 else 'probe',direction=d,sign=-1 if k==0 else 1))
        cp=out/(bank+'.npz');sidecar=run.read(cp.with_suffix('.model.json'))
        if (run.sha(cp)!=run.sha(source) or sidecar['checkpoint_sha256']!=run.sha(cp)
                or sidecar['common_parent_sha256']!=p['parent_sha256'] or sidecar['selection']!=got
                or sidecar['model_promoted']):raise ValueError('Selected source/sidecar mismatch')
    i=selection['candidate']['policy_index']
    science=dict(candidate_parent_fallback=i<0,candidate_identity_changed=i!=original['candidate']['policy_index'],
        independent_physical_evaluation_required=i>=0 and i!=original['candidate']['policy_index'],
        training_acceptance_not_efficacy=True,model_promoted=False,standing_proven=False,walking_proven=False)
    if science!=result['science']:raise ValueError('Decision endpoint mismatch')
    run.check_pins({**p['source_hashes'],**p['old_hashes'],**p['data_hashes'],**result['created_hashes']})
    public=dict(status='completed_independent_MP2_training_audit',full_new_workers=37,training_cases=54,
        training_contexts=6,raw_return_state_and_selection_reproduced=True,exact_baseline=True,**science,
        limits='One finite fixed-policy training panel; neural-seed diversity only. No new independent physical qualification, useful learning, standing, walking or environment response established.')
    target=out.with_name(out.name+'-minimal-result.json')
    if target.exists():raise ValueError('Never overwrite audit result')
    run.write(target,public);print(json.dumps(public))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();main(a.output)
