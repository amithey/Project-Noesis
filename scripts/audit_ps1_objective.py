"""Read-only signed AP2 bank audit of a fixed PS1 objective; no learning."""
import json,time,zipfile
import numpy as np
import audit_ap2 as ap2
from ps1_objective import physical_score,OBJECTIVE,SPEED_SCALE,SPIN_SCALE
from support_descriptor_c1 import describe_support,REGIONS
from noesis_gpu.learning import score_transition,standing_mask

root=ap2.root
read,write,sha,arrays=ap2.read,ap2.write,ap2.sha,ap2.arrays


def physical_data(directory):
    if (directory/'anatomy-observations.npz').exists():
        a=arrays(directory/'anatomy-observations.npz');m=read(directory/'anatomy-model.json')
        assert m['region_columns']==list(REGIONS)
        return a['region_vertical_reaction'][1:],a['com_velocity'][1:],m['total_mass']*abs(m['gravity'][2])
    a=arrays(directory/'body-telemetry.npz');m=read(directory/'telemetry-model.json');c=arrays(directory/'ground-contacts.npz')
    names=[None]*m['nbody']
    for i,name in zip(m['body_ids'],m['body_names']):names[i]=name
    reaction=[]
    for i in range(1,1001):
        force=np.zeros((m['nbody'],3));select=c['sample_index']==i;np.add.at(force,c['body_id'][select],c['force_on_fly'][select])
        np.testing.assert_allclose(force.sum(0),a['ground_force'][i],rtol=1e-12,atol=1e-10)
        description=describe_support(force,names);reaction.append([description['vertical_reaction_by_region'][n] for n in REGIONS])
    return np.asarray(reaction),a['com_velocity'][1:],m['total_mass']*abs(m['gravity'][2])


def main():
    started=time.monotonic();source={str(p.relative_to(root)).replace('\\','/'):sha(p) for p in
        (root/'work/ps1_objective.py',root/'work/test_ps1_objective.py',root/'work/audit_ps1_objective.py',root/'work/support_descriptor_c1.py')}
    inputs={};summary_path=root/'outputs/anatomical-feedback-ap2-20261107-summary.json'
    artifact=read(root/'outputs/anatomical-feedback-ap2-20261107-artifact.json');archive=root/'outputs'/artifact['path']
    assert sha(archive)==artifact['sha256']=='42e20e37d1ee95c550d65d2cb491d69895ae75777b4f279788110d1468d1373d'
    with zipfile.ZipFile(archive) as z:manifest=json.loads(z.read('manifest.json'))
    assert sha(summary_path)==manifest[summary_path.relative_to(root).as_posix()]['sha256'];summary=read(summary_path)
    assert summary['status']=='audited_complete_AP2' and summary['replication']['combined_pilot_pass_count']==0
    ap2.study.verify_native();cases={};positions=0
    for row in summary['rows']:
        positions+=1;birth,bank,seed=row['training_seed'],row['bank'],row['seed'];d=ap2.directory(birth,bank,seed)
        key=f'parent-dev-{seed}' if bank=='parent' and not row['own_seed'] else f'{bank}-{birth}-{seed}'
        if key in cases:continue
        if time.monotonic()-started>180:raise RuntimeError('Offline180s budget exceeded')
        names=['report.json','observations.npz']
        names+=['anatomy-observations.npz','anatomy-model.json'] if (d/'anatomy-observations.npz').exists() else ['body-telemetry.npz','ground-contacts.npz','telemetry-model.json']
        for n in names:
            p=d/n;h=manifest[p.relative_to(root).as_posix()]['sha256'];assert sha(p)==h;inputs[p.relative_to(root).as_posix()]=h
        report=read(d/'report.json');assert report['status']=='completed_transport_pilot' and report['weights_unchanged'] and not report['learning']
        o={k:v[:,0] for k,v in arrays(d/'observations.npz').items()};r,v,weight=physical_data(d)
        assert r.shape==(1000,4) and v.shape==(1000,3)
        components=physical_score(r,v,o['omega'],o['up_body'][:,2],o['foot_load'],weight)
        use=slice(10,None);new=components['reward'][use];legacy=standing_mask(o)[use]
        assert float(legacy.mean())==row['summary']['standing_fraction']
        native=np.array([score_transition({k:x[i-1] for k,x in o.items()},{k:x[i] for k,x in o.items()},5.,objective='curriculum-standing-v7')['reward'] for i in range(10,1000)])
        total=r.sum(1);q=np.divide(r[:,0]+r[:,3],total,out=np.zeros(1000),where=total>1e-5);ap1=np.clip(native-.5*q[use],-1,1)
        supported=(r[use,1]+r[use,2]>1e-5)&(q[use]<1e-12)
        split={}
        for label,mask in (('legacy_standing',legacy),('legacy_standing_with_body_or_proximal_load',legacy&(q[use]>0)),
            ('tarsal_only',supported),('outside_legacy_height_with_tarsal_support',supported&~legacy)):
            split[label]=dict(samples=int(mask.sum()),mean_PS1_reward=float(new[mask].mean()) if mask.any() else None,
                mean_AP1_reward=float(ap1[mask].mean()) if mask.any() else None)
        cases[key]=dict(case_id=key,bank=bank,training_seed=None if bank=='parent' and not row['own_seed'] else birth,probe_seed=seed,
            samples=990,PS1_mean=float(new.mean()),PS1_std=float(new.std()),PS1_min=float(new.min()),PS1_max=float(new.max()),
            PS1_positive_samples=int((new>0).sum()),PS1_exact_floor_samples=int((new==-1).sum()),AP1_mean=float(ap1.mean()),
            mean_components={k:float(x[use].mean()) for k,x in components.items() if k not in ('quality','reward')},groups=split,
            physical_balance_validated=False)
    assert len(cases)==37 and positions==45
    for n,h in inputs.items():assert sha(root/n)==h
    for n,h in source.items():assert sha(root/n)==h
    result=dict(status='completed_readonly_PS1_objective_audit',objective=OBJECTIVE,seconds=time.monotonic()-started,unique_cases=37,bank_positions=45,
        samples=37*990,physics_steps=0,neural_steps=0,learning_updates=0,engineering_scales=dict(COM_speed_mm_s=SPEED_SCALE,body_spin_rad_s=SPIN_SCALE),
        cases=list(cases.values()),original_AP2_combined_pass_count=0,new_gate=False,model_promoted=False,physical_balance_validated=False,walking_proven=False,
        limits='Fixed proposed feedback evaluated post hoc on existing development trajectories. Distribution and software counterexamples do not validate standing, predict learning efficacy or establish useful stability thresholds.')
    write(root/'outputs/physical-standing-ps1-objective-audit-private-20261107.json',dict(**result,input_hashes=inputs,source_hashes=source,archive_sha256=artifact['sha256']))
    write(root/'outputs/physical-standing-ps1-objective-audit-20261107.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2))
    print(json.dumps(dict(positive_samples=sum(x['PS1_positive_samples'] for x in cases.values()),exact_floor_samples=sum(x['PS1_exact_floor_samples'] for x in cases.values()),
        reward_std_range=[min(x['PS1_std'] for x in cases.values()),max(x['PS1_std'] for x in cases.values())],reward_max_range=[min(x['PS1_max'] for x in cases.values()),max(x['PS1_max'] for x in cases.values())])))


if __name__=='__main__':main()
