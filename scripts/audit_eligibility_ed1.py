"""Terminal ED1 raw reward, eligibility-provenance and frozen-retention audit."""
import json
import numpy as np
import run_eligibility_ed1 as study
from audit_physical_standing_ps1 import independent_components
from sc1_signal import normalized_signal
from noesis_gpu.backend import FlyGymBatch
from noesis_gpu.learning import score_transition
from s2_contract import standing_retention
from ap2_contract import DEV,anatomical_rule

root,out=study.root,study.out
read,write,sha,arrays=study.read,study.write,study.sha,study.arrays


def main():
    result,p=read(out/'result.json'),read(out/'protocol.json')
    assert result['status']=='completed_ED1' and len(result['jobs'])==13
    assert result['source_prior_checkpoints_unchanged'] and result['data_unchanged'] and result['process_tree_empty_verified'] and result['lease_released']
    assert not (root/'work/Project-Noesis/gpu-pilot/outputs/codex-training-host.lock').exists()
    for group in ('source_hashes','prior_hashes','artifact_hashes'):
        for n,h in p[group].items():assert sha(root/n)==h
    for f in read(root/'outputs/noesis-required-data-sha256.json')['files']:assert sha(root/'work/NoesisData'/f['path'])==f['sha256']
    study.previous.base.verify_native()
    for j in result['jobs']:
        receipt=read(out/(j['name']+'-receipt.json'));assert receipt==j['receipt']
        assert receipt['exit_code']==0 and receipt['tree_empty'] and not receipt['timed_out'] and j['available_ram_before_gib']>=2
    initial={k:v[0].copy() for k,v in FlyGymBatch(1,'cpu').observe().items()};training={}
    for bank in ('replay','control','candidate'):
        d=out/f'train-{bank}';r=read(d/'report.json');study.validate_training(r,bank)
        assert r['ed1_context']==read(out/(d.name+'-ed1-context.json')) and r['import_checkpoint_sha256']==sha(study.PARENT)
        cp=d/'weights-last.npz';assert sha(cp)==r['saved_checkpoint_sha256'];side=read(cp.with_suffix('.model.json'))
        for k,v in r.items():assert side[k]==v
        rule=study.previous.base.base.compare_checkpoints(study.PARENT,cp);assert rule['compatible'] and rule['rule']['match'] and rule['parameters']['match']
        original,current=arrays(study.PARENT),arrays(cp)
        for k in ('version','normalize_reward_error','bias_correct_reward_error'):np.testing.assert_array_equal(current[k],original[k])
        a=study.previous.base.base.anatomy(d,'train');o=arrays(d/'training-observations.npz');account=arrays(d/'physical-reward-accounting.npz');e=arrays(d/'eligibility-audit.npz')
        assert all(len(v)==5000 and np.isfinite(v).all() for v in o.values()) and all(len(v)==5000 and np.isfinite(v).all() for v in account.values())
        np.testing.assert_array_equal(o['learning_active'],np.arange(5000)>=10);np.testing.assert_array_equal(account['control_chunk_index'],np.arange(5000))
        np.testing.assert_array_equal(account['time'],a['time'][1:]);np.testing.assert_array_equal(account['applied_reward'],o['reward'])
        for i in range(5000):
            before=initial if i==0 else {k:v[i-1] for k,v in o.items()};after={k:v[i] for k,v in o.items()}
            assert score_transition(before,after,5.,objective='curriculum-standing-v7')['reward']==account['native_reward'][i]
        m=read(d/'anatomy-model.json');components=independent_components(a['region_vertical_reaction'][1:],a['com_velocity'][1:],o['omega'],o['up_body'][:,2],o['foot_load'],m['total_mass']*abs(m['gravity'][2]))
        for k,v in components.items():np.testing.assert_allclose(v,account['PS1_'+k],rtol=1e-13,atol=1e-14)
        np.testing.assert_array_equal(o['reward'],account['PS1_reward'])
        assert all(len(v)==4990 for v in e.values());delay=10 if bank=='candidate' else 0
        np.testing.assert_array_equal(e['control_chunk_index'],np.arange(10,5000));np.testing.assert_array_equal(e['update_index'],np.arange(4990))
        np.testing.assert_array_equal(e['used_source_update_index'],np.where(np.arange(4990)>=delay,np.arange(4990)-delay,-1))
        assert np.all(e['independent_weight_step_exact']) and np.all(e['delay_chunks']==delay)
        if delay:
            np.testing.assert_array_equal(e['used_eligibility_sha256'][10:],e['current_eligibility_sha256'][:-10])
            assert np.all(e['used_eligibility_L2'][:10]==0)
            np.testing.assert_array_equal(e['used_eligibility_sha256'][:10],np.full(10,study.digest(np.zeros(len(current['weights'])))))
        else:np.testing.assert_array_equal(e['used_eligibility_sha256'],e['current_eligibility_sha256'])
        moments=normalized_signal(o['reward'][10:])
        for x,y in (('signal','learning_signal'),('raw_error','prediction_error'),('baseline','reward_baseline'),('variance','reward_variance'),('mass','reward_normalization_mass')):np.testing.assert_array_equal(moments[x],e[y])
        for x,y in (('baseline','reward_baseline'),('variance','reward_variance'),('mass','reward_normalization_mass')):assert moments[x][-1]==float(current[y])
        if bank=='replay':
            old=study.previous.out/'train-candidate';assert sha(cp)==sha(old/'weights-last.npz')
            for name in ('training-observations.npz','adhesion-observations.npz','anatomy-observations.npz','physical-reward-accounting.npz','exploration-log.npz'):study.previous.base.base.exact(d/name,old/name)
        elif bank=='candidate':
            study.previous.base.base.exact(d/'exploration-log.npz',out/'train-control/exploration-log.npz')
            for name in ('training-observations.npz','adhesion-observations.npz'):
                c,v=arrays(out/'train-control'/name),arrays(d/name);assert set(c)==set(v)
                for k in c:np.testing.assert_array_equal(c[k][:11],v[k][:11])
        training[bank]=dict(checkpoint_sha256=sha(cp),objective=r['reward_objective'],seed=r['seed'],delay_chunks=delay,
            native_and_physical_rewards_recomputed=5000,normalizer_updates_verified=4990,online_weight_and_queue_checks_passed=True,original_bounds_preserved=True)
    banks={};rows=[]
    for bank in ('parent','control','candidate','v7','AP1','previous_PS1'):
        banks[bank]={}
        for seed in (study.SEEDS if bank in ('parent','control','candidate') else DEV):
            d=out/f'frozen-{bank}-{seed}' if bank in ('control','candidate') else study.reference(bank,seed)
            r,s=study.previous.base.base.summarize(d);stored=result['science']['banks'][bank][str(seed)]
            assert s==stored['summary'] and not r['learning'] and r['motor_exploration']==0 and r['weights_unchanged']
            if bank in ('control','candidate'):
                assert r['ed1_worker_completed'] and r['checkpoint_sha256']==training[bank]['checkpoint_sha256'] and r['ed1_context']['active_delay_chunks']==0
                study.previous.base.base.anatomy(d,'frozen');study.previous.base.base.compare(study.reference('parent',seed),d)
            anatomical=study.previous.prior.previous.contact_data(d);assert anatomical==stored['anatomical']
            o={k:v[:,0] for k,v in arrays(d/'observations.npz').items()};reaction,v,weight=study.previous.physical_data(d)
            score=independent_components(reaction,v,o['omega'],o['up_body'][:,2],o['foot_load'],weight);total=reaction.sum(-1)
            q=np.divide(reaction[:,0]+reaction[:,3],total,out=np.zeros_like(total),where=total>1e-5)
            physical=dict(mean_PS1_quality=float(score['quality'][10:].mean()),mean_COM_speed_mm_s=float(np.linalg.norm(v[10:],axis=1).mean()),
                mean_angular_speed_rad_s=float(np.linalg.norm(o['omega'][10:],axis=1).mean()),mean_non_tarsal_fraction=float(q[10:].mean()),samples=990,physical_balance_validated=False)
            for k,v in physical.items():np.testing.assert_allclose(v,stored['physical'][k],rtol=1e-12,atol=1e-14)
            banks[bank][seed]=dict(summary=s,physical=physical,anatomical=anatomical);rows.append(dict(bank=bank,seed=seed,own_seed=seed==1005,physical=physical))
    assert len(rows)==27
    retention={b:standing_retention({s:banks[b][s]['summary'] for s in DEV},{s:banks['candidate'][s]['summary'] for s in DEV},DEV) for b in ('parent','control','v7','AP1','previous_PS1')}
    assert retention==result['science']['retention']
    anatomy=anatomical_rule(*({s:banks[b][s]['anatomical'] for s in DEV} for b in ('parent','control','candidate')));assert anatomy==result['science']['anatomical']
    directions=[]
    for s in DEV:
        for b in ('parent','control'):
            c,r=banks['candidate'][s]['physical'],banks[b][s]['physical'];directions.append(c['mean_PS1_quality']>r['mean_PS1_quality'] and all(c[k]<r[k] for k in ('mean_COM_speed_mm_s','mean_angular_speed_rad_s','mean_non_tarsal_fraction')))
    physical=all(directions);assert physical==result['science']['all_four_development_physical_directions_improve']
    combined=bool(physical and anatomy['passed'] and all(retention[b]['passed'] for b in ('parent','control','v7','AP1')));assert combined==result['science']['combined_exploratory_pilot_pass']
    full=dict(status='independently_audited_ED1',runtime_source_commit='5dbdf9637adb862b90ba8b949acfb6025bf93bcf',seconds=result['seconds'],new_workers=13,
        training=training,frozen_positions=27,rows=rows,retention=retention,anatomical=anatomy,all_four_development_physical_directions_improve=bool(physical),
        combined_exploratory_pilot_pass=combined,source_prior_data_unchanged=True,all_full_horizons_receipts_exit_zero_trees_empty=True,lease_released=True,
        limits='Weight steps/full eligibility equality checked independently online; offline verifies recorded moments/hash provenance, not full neural-spike reconstruction. One seed, engineering objective, no general efficacy or ability proof.',
        physical_balance_validated=False,model_promoted=False,walking_proven=False)
    write(root/'outputs/eligibility-delay-ed1-20261107-summary.json',full)
    minimal={k:v for k,v in full.items() if k not in ('training','rows','retention','anatomical')}
    minimal.update(candidate_checkpoint_sha256=training['candidate']['checkpoint_sha256'],zero_delay_replay_exact=True,
        original_retention_pass={k:v['passed'] for k,v in retention.items()},anatomical_rule_pass=anatomy['passed'],publication_scope='Minimal NEW ED1 verdict only; no pending-approval old PS1 performance metrics or private maps')
    write(root/'outputs/eligibility-delay-ed1-minimal-result-20261107.json',minimal);print(json.dumps(minimal,indent=2))


if __name__=='__main__':main()
