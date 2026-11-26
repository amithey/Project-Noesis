"""Independent terminal AP2 audit of raw banks, paired controls and all three births."""
import json
import numpy as np
import run_anatomical_ap2 as study
import audit_ap1c as previous
from noesis_gpu.backend import FlyGymBatch
from noesis_gpu.learning import score_transition
from ap1_feedback import corrected_reward,non_tarsal_fraction,validate_ap1_training
from ap2_contract import DEV,replication_summary

root,out=study.root,study.out
read,write,sha,arrays=study.read,study.write,study.sha,study.arrays


def directory(birth,bank,seed):
    if birth==1005:return previous.cell(bank,seed)
    if bank=='parent' and seed in DEV:return study.parent_dev(seed)
    if birth==1004 and bank=='control' and seed==10006:return study.H2/'child1004-10006'
    return out/f'frozen-{bank}-{birth}-{seed}'


def main():
    result=read(out/'result.json');p=read(out/'protocol.json')
    assert result['status']=='completed_AP2' and len(result['jobs'])==25
    assert result['source_prior_children_unchanged'] and result['data_unchanged'] and result['process_tree_empty_verified'] and result['lease_released']
    assert not (root/'work/Project-Noesis/gpu-pilot/outputs/codex-training-host.lock').exists()
    study.verify_native()
    for group in ('source_hashes','prior_hashes','artifact_hashes'):
        for n,h in p[group].items():assert sha(root/n)==h
    for f in read(root/'outputs/noesis-required-data-sha256.json')['files']:assert sha(root/'work/NoesisData'/f['path'])==f['sha256']
    assert sum(j['phase']=='train' for j in result['jobs'])==4 and sum(j['phase']=='frozen' for j in result['jobs'])==21
    for job in result['jobs']:
        receipt=read(out/(job['name']+'-receipt.json'));assert receipt==job['receipt']
        assert receipt['exit_code']==0 and receipt['tree_empty'] and not receipt['timed_out'] and job['available_ram_before_gib']>=2
    initial={k:v[0].copy() for k,v in FlyGymBatch(1,'cpu').observe().items()};training={}
    for birth in (1004,1006):
        for bank,strength in (('control',0.),('candidate',.5)):
            d=out/f'train-{bank}-{birth}';report=read(d/'report.json');validate_ap1_training(report,strength)
            assert report['seed']==birth and report['ap1_context']['training_seed']==birth and report['ap1_context']['experiment']=='AP2'
            assert report['import_checkpoint_sha256']==sha(study.PARENT)
            cp=d/'weights-last.npz';assert sha(cp)==report['saved_checkpoint_sha256']
            check=study.base.compare_checkpoints(study.PARENT,cp);assert check['compatible'] and check['rule']['match'] and check['parameters']['match']
            original,current=arrays(study.PARENT),arrays(cp)
            for k in ('version','normalize_reward_error','bias_correct_reward_error'):np.testing.assert_array_equal(original[k],current[k])
            sidecar=read(cp.with_suffix('.model.json'))
            for k,v in report.items():assert sidecar[k]==v
            a=study.base.anatomy(d,'train');r=arrays(d/'reward-accounting.npz');o=arrays(d/'training-observations.npz')
            assert all(len(v)==5000 and np.isfinite(v).all() for v in r.values()) and all(len(v)==5000 and np.isfinite(v).all() for v in o.values())
            np.testing.assert_array_equal(r['control_chunk_index'],np.arange(5000));np.testing.assert_array_equal(r['time'],a['time'][1:])
            np.testing.assert_array_equal(r['non_tarsal_fraction'],[non_tarsal_fraction(x) for x in a['region_vertical_reaction'][1:]])
            np.testing.assert_array_equal(r['applied_reward'],[corrected_reward(n,q,strength) for n,q in zip(r['native_reward'],r['non_tarsal_fraction'])])
            np.testing.assert_array_equal(r['applied_reward'],o['reward']);np.testing.assert_array_equal(r['effective_penalty'],r['native_reward']-r['applied_reward'])
            np.testing.assert_array_equal(o['learning_active'],np.arange(5000)>=10)
            for i in range(5000):
                before=initial if i==0 else {k:v[i-1] for k,v in o.items()};after={k:v[i] for k,v in o.items()}
                assert score_transition(before,after,5.,objective='curriculum-standing-v7')['reward']==r['native_reward'][i]
            training[f'{bank}-{birth}']=dict(checkpoint_sha256=sha(cp),recomputed_native_transitions=5000,actual_objective=report['reward_objective'],
                mean_native_reward=float(r['native_reward'].mean()),mean_applied_reward=float(r['applied_reward'].mean()),mean_effective_penalty=float(r['effective_penalty'].mean()),original_weight_invariants=True)
        control=out/f'train-control-{birth}';candidate=out/f'train-candidate-{birth}'
        assert sha(control/'weights-last.npz')==study.CONTROL_SHA[birth]
        for n in ('weights-last.npz','training-observations.npz','adhesion-observations.npz','exploration-log.npz'):study.base.exact(control/n,study.TRAIN[birth]/n)
        study.base.exact(control/'exploration-log.npz',candidate/'exploration-log.npz')
        for n in ('training-observations.npz','adhesion-observations.npz'):
            a,b=arrays(control/n),arrays(candidate/n);assert set(a)==set(b)
            for k in a:
                if k!='reward':np.testing.assert_array_equal(a[k][:11],b[k][:11])
    banks={birth:{bank:{} for bank in ('parent','control','candidate')} for birth in (1004,1005,1006)};rows=[];unique=set()
    for birth in banks:
        for bank in banks[birth]:
            for seed in (birth,*DEV):
                d=directory(birth,bank,seed);unique.add(d);report,summary=study.base.summarize(d);anatomy=previous.contact_data(d)
                recorded=result['births'][str(birth)][bank][str(seed)]
                assert recorded['summary']==summary and recorded['anatomical']==anatomy
                assert not report['learning'] and report['motor_exploration']==0 and report['weights_unchanged']
                if bank!='parent':study.base.compare(directory(birth,'parent',seed),d)
                if birth!=1005 and bank!='candidate':
                    ref=(root/'outputs/adhesion-sign-a1-20261106'/f'legacy-{seed}') if bank=='parent' and seed in DEV else study.reference(birth,bank,seed)
                    for n in ('observations.npz','adhesion-observations.npz'):study.base.exact(d/n,ref/n)
                    assert report['spikes_per_world']==read(ref/'report.json')['spikes_per_world']
                banks[birth][bank][seed]=dict(summary=summary,anatomical=anatomy)
                rows.append(dict(training_seed=birth,bank=bank,seed=seed,own_seed=seed==birth,summary=summary,anatomical=anatomy))
    assert len(rows)==45 and len(unique)==37
    summary=replication_summary(banks);assert summary==result['replication']
    old=read(study.C/'result.json');birth1005=next(r for r in summary['births'] if r['training_seed']==1005)
    assert birth1005['retention']==old['retention'] and birth1005['anatomical']['passed']==old['anatomical_dev_each_vs_both']
    assert not birth1005['combined_pilot_rule_pass']
    verdict=dict(status='audited_complete_AP2',evaluated_source='7692865891a05a939c983ce8ee1946b9f768fff4',seconds=result['seconds'],
        new_complete_workers=25,new_native_reward_transitions_recomputed=20000,bank_positions=45,unique_frozen_cells=37,
        training=training,rows=rows,replication=summary,source_data_priors_unchanged=True,all_full_horizons_receipts_exit_zero_trees_empty=True,
        original1005failure_preserved=True,model_promoted=False,walking_proven=False,lease_released=True,
        limits='Fixed three training births, shared development data; no statistical efficacy, retained physical ability, gait or environment-response qualification.')
    write(root/'outputs/anatomical-feedback-ap2-20261107-summary.json',verdict)
    print(json.dumps({k:v for k,v in verdict.items() if k not in ('training','rows','replication')}))
    print(json.dumps([dict(training_seed=x['training_seed'],combined_pilot_rule_pass=x['combined_pilot_rule_pass'],anatomical_pass=x['anatomical']['passed'],retention={k:v['passed'] for k,v in x['retention'].items()},legacy_standing_ge90=x['absolute_legacy_standing_ge90_count']) for x in summary['births']]))


if __name__=='__main__':main()
