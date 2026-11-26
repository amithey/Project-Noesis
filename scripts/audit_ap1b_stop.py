"""Audit technical AP1B stop; do not infer a scientific verdict from partial bank."""
import json
import numpy as np
from audit_ap1b import root,OUT,read,sha,arrays,exact,contact_data
from noesis_gpu.backend import FlyGymBatch
from noesis_gpu.learning import score_transition
from noesis_gpu.evaluate import summarize
from noesis_gpu.retention_audit import compare_checkpoints
from ap1_feedback import corrected_reward,non_tarsal_fraction,validate_ap1_training


def main():
    result=read(OUT/'result.json');p=read(OUT/'protocol.json')
    assert result['status']=='failed' and len(result['jobs'])==12 and len(result['cells'])==12
    assert result['source_prior_children_unchanged'] and result['data_unchanged'] and result['lease_released'] and result['process_tree_empty_verified']
    for group in ('source_hashes','prior_hashes','artifact_hashes'):
        for n,h in p[group].items():assert sha(root/n)==h
    for f in read(root/'outputs/noesis-required-data-sha256.json')['files']:assert sha(root/'work/NoesisData'/f['path'])==f['sha256']
    for i,job in enumerate(result['jobs']):
        receipt=read(OUT/(job['name']+'-receipt.json'));assert receipt==job['receipt']
        assert receipt['tree_empty'] and not receipt['timed_out']
        assert receipt['exit_code']==(1 if i==11 else 0)
    candidate=OUT/'train-candidate-1005';report=read(candidate/'report.json');validate_ap1_training(report,.5)
    cp=candidate/'weights-last.npz';assert sha(cp)==report['saved_checkpoint_sha256']=='aced093428ea850e6f091f36fa13caa86afad0948ca0deb6cc5f3d85321b265c'
    parent=root/'work/evidence/s1-restored/gpu-pilot/outputs/e4-exploration-20261029/train-control-1000/weights-last.npz'
    audit=compare_checkpoints(parent,cp);assert audit['compatible'] and audit['rule']['match'] and audit['parameters']['match']
    before_cp,after_cp=arrays(parent),arrays(cp)
    for k in ('version','normalize_reward_error','bias_correct_reward_error'):np.testing.assert_array_equal(before_cp[k],after_cp[k])
    sidecar=read(cp.with_suffix('.model.json'))
    for k,v in report.items():assert sidecar[k]==v
    a=arrays(candidate/'anatomy-observations.npz');r=arrays(candidate/'reward-accounting.npz');obs=arrays(candidate/'training-observations.npz')
    assert all(len(v)==5001 and np.isfinite(v).all() for v in a.values())
    assert all(len(v)==5000 and np.isfinite(v).all() for v in r.values())
    np.testing.assert_array_equal(r['non_tarsal_fraction'],[non_tarsal_fraction(x) for x in a['region_vertical_reaction'][1:]])
    np.testing.assert_array_equal(r['time'],a['time'][1:]);np.testing.assert_array_equal(r['control_chunk_index'],np.arange(5000))
    np.testing.assert_array_equal(r['applied_reward'],[corrected_reward(n,q,.5) for n,q in zip(r['native_reward'],r['non_tarsal_fraction'])])
    np.testing.assert_array_equal(r['applied_reward'],obs['reward']);np.testing.assert_array_equal(r['effective_penalty'],r['native_reward']-r['applied_reward'])
    initial={k:v[0].copy() for k,v in FlyGymBatch(1,'cpu').observe().items()}
    for i in range(5000):
        before=initial if i==0 else {k:v[i-1] for k,v in obs.items()}
        after={k:v[i] for k,v in obs.items()}
        assert score_transition(before,after,5.,objective='curriculum-standing-v7')['reward']==r['native_reward'][i]
    control=root/'outputs/anatomical-feedback-ap1-20261106/train-control-1005'
    exact(control/'exploration-log.npz',candidate/'exploration-log.npz')
    for n in ('training-observations.npz','adhesion-observations.npz'):exact(control/n,candidate/n,prefix=11,skip=('reward',))
    failed=OUT/'frozen-candidate-10007';failed_report=read(failed/'report.json')
    assert failed_report['status']=='failed' and failed_report['error']=='Unknown loaded anatomy: nmf/rh_trochanterfemur'
    assert not failed_report['ap1_worker_completed']
    partial=arrays(failed/'observations.npz');partial_a=arrays(failed/'anatomy-observations.npz');n=failed_report['completed_control_chunks']
    assert all(len(v)==n and np.isfinite(v).all() for v in partial.values())
    assert all(len(v)==n+1 and np.isfinite(v).all() for v in partial_a.values())
    assert len(result['retention'])==0 and not result['model_promoted'] and not result['walking_proven']
    for name,entry in result['cells'].items():
        bank,seed=name.rsplit('-',1)
        directory=(root/'outputs/body-telemetry-h2b-20261106'/('parent-10007' if bank=='parent' else 'child1005-10007')) if seed=='10007' else OUT/f'frozen-{bank}-{seed}'
        assert summarize(directory)[1]==entry['summary'] and contact_data(directory)==entry['anatomical']
    selected={f for f in OUT.iterdir() if f.is_file() and f.suffix in ('.json','.log')}
    for d in OUT.iterdir():
        if d.is_dir():selected|={f for f in d.iterdir() if f.is_file() and f.suffix in ('.json','.npz')}
    verdict=dict(status='audited_AP1B_technical_stop_no_scientific_verdict',evaluated_source='a05b0daa3c4416f1ffc0a2c59f8770f77069bd8b',
        seconds=result['seconds'],worker_starts=12,completed_workers=11,failed_workers=1,candidate_training_seconds=25,candidate_updates=4990,
        candidate_sha256=sha(cp),all_5000_candidate_native_rewards_recomputed=True,all_reward_accounting_exact=True,
        exploration_drive_AND_credit_exact_control=True,initial_11_body_pad_rows_exact=True,original_checkpoint_invariants=True,
        complete_reference_cells=10,complete_candidate_cells=2,missing_complete_candidate_seeds=[10007,10010,10011],
        failed_seed=10007,failed_error=failed_report['error'],failed_completed_chunks=n,failed_completed_seconds=n*.005,
        failed_anatomy_samples=n+1,source_data_priors_unchanged=True,all_process_trees_empty=True,lease_released=True,
        scientific_bank_complete=False,retention_verdict_available=False,model_promoted=False,walking_proven=False,
        file_hashes={f.relative_to(root).as_posix():sha(f) for f in sorted(selected)})
    dest=root/'outputs/anatomical-feedback-ap1b-stop-audit-20261106.json';dest.write_text(json.dumps(verdict,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in verdict.items() if k!='file_hashes'}))


if __name__=='__main__':main()
