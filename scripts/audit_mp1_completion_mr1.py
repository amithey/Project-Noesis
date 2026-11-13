"""MR1 combined-bank audit; original failed MP1 allocation is never relabeled."""
import json
import numpy as np
import run_mp1_completion_mr1 as run
from audit_mp1_interruption import verify,jobs
from audit_measured_mp1 import equivalent
from mp1_select import TRAIN_SEEDS,DEV,FRESH,STEP,MIN_GAIN,fixed_directions,training_panel


def main():
    prefix=verify();r=run.read(run.out/'result.json');protocol=run.read(run.out/'protocol.json');p=protocol['original_protocol']
    assert r['status']=='completed_MR1_cells' and len(r['jobs'])==16
    assert r['old_source_created_unchanged'] and r['data_unchanged'] and r['process_tree_empty_verified'] and r['lease_released']
    assert r['awake_request_released'] and r['first_fallback_replay_exact']
    assert run.read(run.old.out/'result.json')['status']=='failed'
    for group in ('original_hashes','source_hashes'):
        for n,h in protocol[group].items():assert run.sha(run.root/n)==h
    for n,h in r['created_hashes'].items():assert run.sha(run.root/n)==h
    for actual,(name,phase,bank,seed,d,sign) in zip(r['jobs'],jobs()[29:]):
        assert [actual[k] for k in ('name','phase','bank','seed')]==[name,phase,bank,seed]
        receipt=run.read(run.out/(name+'-receipt.json'));assert receipt==actual['receipt']
        assert receipt['exit_code']==0 and receipt['tree_empty'] and not receipt['timed_out'] and actual['available_ram_before_gib']>=2
        report,_,_,_,_=run.old.measured(run.out/name);c=report['mp1_context']
        assert c==run.read(run.out/(name+'-mp1-context.json'))
        assert [c[k] for k in ('phase','bank','seed','direction','sign')]==[phase,bank,seed,None,None]
        assert c['worker_sha256']==p['source_hashes']['work/mp1_worker.py']
        cp=run.old.out/(bank+'.npz');assert c['checkpoint_sha256']==run.sha(cp)
        imported=c['import_records'][0];weights=run.arrays(cp)
        assert imported['actual_import_exact'] and imported['topology_unchanged'] and imported['nonselected_weights_unchanged']
        assert imported['existing_selected_edges']==len(weights['weights'])
        assert imported['changed_selected_edges']==int(np.count_nonzero(weights['weights']!=weights['initial_weights']))
        run.old.native.compare(run.reference(p,'parent',seed),run.out/name)
    # Independent arithmetic confirmation of the saved gradient comparator.
    state=run.arrays(run.old.out/'optimizer-state.npz');records=run.read(run.old.out/'training-returns.json');returns=training_panel(records)
    parent=run.arrays(run.root/p['parent_checkpoint']);directions=fixed_directions(len(parent['weights']))
    np.testing.assert_array_equal(state['directions'],directions)
    coeff=np.array([sum(float(returns[row,2*d+1]-returns[row,2*d]) for row in range(2))/(4*STEP) for d in range(4)])
    gradient=sum((coeff[d]*directions[d]/4 for d in range(4)),start=np.zeros(len(parent['weights'])))
    norm=float(abs(gradient).max());delta=np.zeros_like(gradient) if norm<=1e-12 else STEP*gradient/norm
    np.testing.assert_allclose(state['gradient'],gradient,rtol=1e-12,atol=1e-14)
    np.testing.assert_allclose(state['gradient_delta'],delta,rtol=1e-12,atol=1e-14)
    # Exact fallback replay over all eight contexts is supplementary technical evidence.
    replay=[]
    for seed in (*TRAIN_SEEDS,*DEV,*FRESH):
        parent_dir=run.reference(p,'parent',seed);negative=run.reference(p,'negative',seed)
        exact=True
        for filename in ('observations.npz','adhesion-observations.npz','anatomy-observations.npz'):
            if not (parent_dir/filename).exists():continue
            a,b=run.arrays(parent_dir/filename),run.arrays(negative/filename)
            exact=exact and set(a)==set(b) and all(np.array_equal(a[k],b[k]) for k in a)
        exact=exact and run.read(parent_dir/'report.json')['spikes_per_world']==run.read(negative/'report.json')['spikes_per_world']
        replay.append(dict(seed=seed,exact=bool(exact)))
    science=run.finish(p);equivalent(json.loads(json.dumps(science)),r['science'])
    summary=dict(status='completed_independent_MR1_combined_bank_audit',original_allocation_status='failed_incomplete',
        completed_distinct_workers=45,allocations=2,attempts=46,preserved_timeout_attempts=1,
        old_completed_workers=29,completion_workers=16,no_retraining_or_weight_update=True,
        candidate_training_replay_exact=prefix['candidate_training_replay_exact'],fallback_replay=replay,
        first_fallback_replay_exact=True,comparison=science['comparison'],model_promoted=False,standing_proven=False,walking_proven=False,
        limits='Combined frozen bank from two documented allocations; original timeout preserved. In-sample selection is not independent efficacy or biological/robustness qualification.')
    run.write(run.root/'outputs/mp1-completion-mr1-summary-private-20261113.json',summary)
    print('MR1 combined-bank audit complete; original failed allocation retained, private details not published')


if __name__=='__main__':main()
