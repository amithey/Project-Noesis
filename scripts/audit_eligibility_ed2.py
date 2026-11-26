"""Independent terminal ED2 audit; private performance/reference maps remain local."""
import json
import numpy as np
import run_eligibility_ed2 as study
from audit_physical_standing_ps1 import independent_components
from sc1_signal import normalized_signal
from noesis_gpu.backend import FlyGymBatch
from noesis_gpu.learning import score_transition
from ed2_contract import BIRTHS,NEW_BIRTHS,replication_summary
from ap2_contract import DEV

root,out=study.root,study.out
read,write,sha,arrays=study.read,study.write,study.sha,study.arrays


def equivalent(actual,expected):
    if isinstance(expected,dict):
        assert set(actual)==set(expected)
        for k in expected:equivalent(actual[k],expected[k])
    elif isinstance(expected,list):
        assert len(actual)==len(expected)
        for a,b in zip(actual,expected):equivalent(a,b)
    elif isinstance(expected,bool) or expected is None or isinstance(expected,str):assert actual==expected
    else:np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-14)


def main():
    result,p=read(out/'result.json'),read(out/'protocol.json')
    assert result['status']=='completed_ED2' and len(result['jobs'])==26
    for k in ('source_prior_checkpoints_unchanged','completed_children_unchanged','data_unchanged','process_tree_empty_verified','lease_released'):assert result[k]
    assert not (root/'work/Project-Noesis/gpu-pilot/outputs/codex-training-host.lock').exists()
    for group in ('source_hashes','prior_hashes','artifact_hashes'):
        for n,h in p[group].items():assert sha(root/n)==h
    study.old.previous.base.verify_native()
    for f in read(root/'outputs/noesis-required-data-sha256.json')['files']:assert sha(root/'work/NoesisData'/f['path'])==f['sha256']
    for job in result['jobs']:
        receipt=read(out/(job['name']+'-receipt.json'));assert receipt==job['receipt']
        assert receipt['exit_code']==0 and not receipt['timed_out'] and receipt['tree_empty'] and job['available_ram_before_gib']>=2
    initial={k:v[0].copy() for k,v in FlyGymBatch(1,'cpu').observe().items()};training={}
    jobs=[('replay',1005)]+[(b,s) for s in NEW_BIRTHS for b in ('control','candidate')]
    for bank,birth in jobs:
        name=f'train-{bank}-{birth}';d=out/name;r=read(d/'report.json');study.validate_training(r,bank)
        assert r['ed2_context']==read(out/(name+'-ed2-context.json')) and r['seed']==r['ed2_context']['training_seed']==birth
        assert r['import_checkpoint_sha256']==sha(study.PARENT)
        cp=d/'weights-last.npz';assert sha(cp)==r['saved_checkpoint_sha256'];side=read(cp.with_suffix('.model.json'))
        for k,v in r.items():assert side[k]==v
        rule=study.native.compare_checkpoints(study.PARENT,cp);assert rule['compatible'] and rule['rule']['match'] and rule['parameters']['match']
        original,current=arrays(study.PARENT),arrays(cp)
        for k in ('version','normalize_reward_error','bias_correct_reward_error'):np.testing.assert_array_equal(original[k],current[k])
        anatomy=study.native.anatomy(d,'train');o=arrays(d/'training-observations.npz');account=arrays(d/'physical-reward-accounting.npz');e=arrays(d/'eligibility-audit.npz')
        assert all(len(v)==5000 and np.isfinite(v).all() for v in o.values()) and all(len(v)==5000 and np.isfinite(v).all() for v in account.values())
        np.testing.assert_array_equal(o['learning_active'],np.arange(5000)>=10)
        np.testing.assert_array_equal(account['control_chunk_index'],np.arange(5000));np.testing.assert_array_equal(account['time'],anatomy['time'][1:])
        np.testing.assert_array_equal(o['reward'],account['applied_reward']);np.testing.assert_array_equal(o['reward'],account['PS1_reward'])
        for i in range(5000):
            before=initial if i==0 else {k:v[i-1] for k,v in o.items()};after={k:v[i] for k,v in o.items()}
            assert score_transition(before,after,5.,objective='curriculum-standing-v7')['reward']==account['native_reward'][i]
        m=read(d/'anatomy-model.json');c=independent_components(anatomy['region_vertical_reaction'][1:],anatomy['com_velocity'][1:],o['omega'],o['up_body'][:,2],o['foot_load'],m['total_mass']*abs(m['gravity'][2]))
        for k,v in c.items():np.testing.assert_allclose(v,account['PS1_'+k],rtol=1e-13,atol=1e-14)
        assert all(len(v)==4990 for v in e.values());delay=0 if bank=='control' else 10
        np.testing.assert_array_equal(e['update_index'],np.arange(4990));np.testing.assert_array_equal(e['control_chunk_index'],np.arange(10,5000))
        np.testing.assert_array_equal(e['used_source_update_index'],np.where(np.arange(4990)>=delay,np.arange(4990)-delay,-1))
        assert np.all(e['independent_weight_step_exact']) and np.all(e['delay_chunks']==delay)
        if delay:
            np.testing.assert_array_equal(e['used_eligibility_sha256'][10:],e['current_eligibility_sha256'][:-10]);assert np.all(e['used_eligibility_L2'][:10]==0)
            np.testing.assert_array_equal(e['used_eligibility_sha256'][:10],np.full(10,study.old.digest(np.zeros(len(current['weights'])))))
        else:np.testing.assert_array_equal(e['used_eligibility_sha256'],e['current_eligibility_sha256'])
        moments=normalized_signal(o['reward'][10:])
        for a,b in (('signal','learning_signal'),('raw_error','prediction_error'),('baseline','reward_baseline'),('variance','reward_variance'),('mass','reward_normalization_mass')):np.testing.assert_array_equal(moments[a],e[b])
        for a,b in (('baseline','reward_baseline'),('variance','reward_variance'),('mass','reward_normalization_mass')):assert moments[a][-1]==float(current[b])
        if bank=='replay':
            old=study.old.out/'train-candidate';assert sha(cp)==sha(old/'weights-last.npz')
            for n in ('training-observations.npz','adhesion-observations.npz','anatomy-observations.npz','physical-reward-accounting.npz','exploration-log.npz','eligibility-audit.npz'):study.native.exact(d/n,old/n)
        elif bank=='candidate':
            control=out/f'train-control-{birth}';study.native.exact(d/'exploration-log.npz',control/'exploration-log.npz')
            for n in ('training-observations.npz','adhesion-observations.npz'):
                a,b=arrays(control/n),arrays(d/n);assert set(a)==set(b)
                for k in a:np.testing.assert_array_equal(a[k][:11],b[k][:11])
        training[name]=dict(checkpoint_sha256=sha(cp),training_seed=birth,delay_chunks=delay,full_25s=True,learning_attempts=4990,
            native_and_physical_rewards_recomputed=5000,normalizer_and_queue_updates_verified=4990,original_bound_rule_preserved=True)
    cache={};births={};positions=0
    for birth in BIRTHS:
        banks={}
        for bank in ('parent','control','candidate','v7','AP1','previous_PS1'):
            banks[bank]={}
            for seed in ((birth,*DEV) if bank in ('parent','control','candidate') else DEV):
                d=study.directory(birth,bank,seed)
                if d not in cache:
                    r,summary=study.native.summarize(d);assert not r['learning'] and r['motor_exploration']==0 and r['weights_unchanged'] and summary['seed']==seed
                    if d.is_relative_to(out):
                        assert r['ed2_worker_completed'] and r['ed2_context']['training_seed']==birth and r['ed2_context']['active_delay_chunks']==0
                        expected=sha(study.PARENT) if bank=='parent' else training[f'train-{bank}-{birth}']['checkpoint_sha256']
                        assert r['checkpoint_sha256']==expected;study.native.anatomy(d,'frozen')
                        if bank!='parent':study.native.compare(study.directory(birth,'parent',seed),d)
                    anatomy=study.old.previous.prior.previous.contact_data(d)
                    o={k:v[:,0] for k,v in arrays(d/'observations.npz').items()};reaction,v,weight=study.old.previous.physical_data(d)
                    score=independent_components(reaction,v,o['omega'],o['up_body'][:,2],o['foot_load'],weight);total=reaction.sum(-1)
                    q=np.divide(reaction[:,0]+reaction[:,3],total,out=np.zeros_like(total),where=total>1e-5)
                    physical=dict(mean_PS1_quality=float(score['quality'][10:].mean()),mean_COM_speed_mm_s=float(np.linalg.norm(v[10:],axis=1).mean()),
                        mean_angular_speed_rad_s=float(np.linalg.norm(o['omega'][10:],axis=1).mean()),mean_non_tarsal_fraction=float(q[10:].mean()),samples=990,physical_balance_validated=False)
                    cache[d]=dict(summary=summary,physical=physical,anatomical=anatomy)
                banks[bank][seed]=cache[d];equivalent(cache[d],result['science']['banks'][str(birth)][bank][str(seed)])
                if bank in ('parent','control','candidate'):positions+=1
        births[birth]=banks
    assert positions==45 and len(cache)==49
    cohort=replication_summary(births);equivalent(cohort,result['science']['replication'])
    old=read(study.old.out/'result.json')['science'];first=cohort['births'][0]
    for k in ('retention','anatomical','paired_physical','all_four_development_physical_directions_improve','combined_exploratory_pilot_pass'):equivalent(first[k],old[k])
    assert not first['combined_exploratory_pilot_pass']
    full=dict(status='independently_audited_ED2',runtime_source_commit='9604f188a7d36b0bb7967fba77ce951dbf8593a5',seconds=result['seconds'],new_complete_workers=26,
        training=training,main_bank_positions=45,unique_frozen_cells=49,replication=cohort,source_prior_children_data_unchanged=True,
        all_full_horizons_receipts_exit_zero_trees_empty=True,lease_released=True,physical_balance_validated=False,model_promoted=False,walking_proven=False,
        limits='Independent reward/statistic/moment/provenance audit; live independent weight-step checks, not offline full neural-spike reconstruction. Three births share reference cells; no statistical efficacy or physical-ability proof.')
    write(root/'outputs/eligibility-replication-ed2-20261107-summary.json',full)
    minimal={k:v for k,v in full.items() if k not in ('training','replication')}
    minimal.update(training_birth_count=3,original_failed_birth_preserved=True,combined_pilot_pass_count=cohort['combined_pilot_pass_count'],
        descriptive_mean_direction_birth_count=cohort['descriptive_mean_direction_birth_count'],descriptive_means_never_rescue_primary_failure=True,
        per_birth=[dict(training_seed=x['training_seed'],combined_pilot_pass=x['combined_exploratory_pilot_pass'],retention_pass={b:v['passed'] for b,v in x['retention'].items()}) for x in cohort['births']],
        publication_scope='Minimal new replication verdict only; no rejected receipt metadata, old reference performance tables or private integrity maps')
    write(root/'outputs/eligibility-replication-ed2-minimal-result-20261107.json',minimal);print(json.dumps(minimal,indent=2))


if __name__=='__main__':main()
