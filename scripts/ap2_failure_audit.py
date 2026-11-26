"""Exploratory OFFLINE failure partition; never a new acceptance or balance gate."""
import hashlib,json,time,zipfile
from pathlib import Path
import numpy as np
from height_audit_h1 import masks,intervals,quantiles
from support_descriptor_c1 import describe_support,REGIONS

MODES=('unsupported','central_body_contact','proximal_leg_contact','tarsal_only','indeterminate_low_force')


def partition(obs):
    position=np.asarray(obs['position']);up=np.asarray(obs['up_body']);load=np.asarray(obs['foot_load'])
    n=len(position)
    if position.shape!=(n,3) or up.shape!=(n,3) or load.shape!=(n,6) or not all(np.isfinite(x).all() for x in (position,up,load)) or (load<0).any():raise ValueError('Invalid native frozen state')
    standing,_=masks(obs);upright=up[:,2]>=.7;claw=(load>1e-5).any(axis=1);z=position[:,2]
    groups=dict(standing=standing,upright_failure=~upright,no_claw_upright=upright&~claw,
        low_height_upright_claw=upright&claw&(z<.3),high_height_upright_claw=upright&claw&(z>1.5))
    np.testing.assert_array_equal(np.sum(np.stack(list(groups.values())),axis=0),np.ones(n,int))
    return groups


def support(directory):
    def arrays(path):
        with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
    if (directory/'anatomy-observations.npz').exists():
        a=arrays(directory/'anatomy-observations.npz');return a['contact_mode_code'][1:],a['com_velocity'][1:]
    m=json.loads((directory/'telemetry-model.json').read_text());a=arrays(directory/'body-telemetry.npz');c=arrays(directory/'ground-contacts.npz')
    names=[None]*m['nbody']
    for i,n in zip(m['body_ids'],m['body_names']):names[i]=n
    code=[]
    for i in range(1,1001):
        force=np.zeros((m['nbody'],3));use=c['sample_index']==i
        np.add.at(force,c['body_id'][use],c['force_on_fly'][use])
        np.testing.assert_allclose(force.sum(0),a['ground_force'][i],rtol=1e-12,atol=1e-10)
        code.append(MODES.index(describe_support(force,names)['contact_mode']))
    return np.asarray(code),a['com_velocity'][1:]


def main():
    root=Path(__file__).resolve().parents[1]
    def read(p):return json.loads(p.read_text())
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    before=sha(Path(__file__));started=time.monotonic();inputs={};artifact=read(root/'outputs/anatomical-feedback-ap2-20261107-artifact.json')
    archive=root/'outputs'/artifact['path'];assert sha(archive)==artifact['sha256']=='42e20e37d1ee95c550d65d2cb491d69895ae75777b4f279788110d1468d1373d'
    with zipfile.ZipFile(archive) as z:manifest=json.loads(z.read('manifest.json'))
    summary_path=root/'outputs/anatomical-feedback-ap2-20261107-summary.json'
    assert sha(summary_path)==manifest[summary_path.relative_to(root).as_posix()]['sha256']
    summary=read(summary_path);assert summary['status']=='audited_complete_AP2' and summary['replication']['combined_pilot_pass_count']==0
    import audit_ap2 as source
    cases={};positions=[]
    for row in summary['rows']:
        birth,bank,seed=row['training_seed'],row['bank'],row['seed'];d=source.directory(birth,bank,seed)
        key=f'parent-dev-{seed}' if bank=='parent' and not row['own_seed'] else f'{bank}-{birth}-{seed}'
        positions.append(dict(training_seed=birth,bank=bank,seed=seed,case_id=key))
        if key in cases:continue
        if time.monotonic()-started>180:raise RuntimeError('Offline180s budget')
        names=['report.json','observations.npz']
        names+=['anatomy-observations.npz','anatomy-model.json'] if (d/'anatomy-observations.npz').exists() else ['body-telemetry.npz','ground-contacts.npz','telemetry-model.json']
        for n in names:
            p=d/n;h=manifest[p.relative_to(root).as_posix()]['sha256'];assert sha(p)==h;inputs[p.relative_to(root).as_posix()]=h
        with np.load(d/'observations.npz',allow_pickle=False) as z:obs={k:z[k][:,0].copy() for k in ('position','up_body','foot_load')}
        assert len(obs['position'])==1000
        groups=partition(obs);codes,velocity=support(d);assert codes.shape==(1000,) and velocity.shape==(1000,3)
        assert np.isin(codes,np.arange(5)).all() and np.isfinite(velocity).all()
        measured=np.arange(1000)>=10;native_standing=float(groups['standing'][measured].mean());assert native_standing==row['summary']['standing_fraction']
        grouped={}
        for label,mask in groups.items():
            use=mask&measured;count=int(use.sum())
            counts={name:int(np.count_nonzero(codes[use]==i)) for i,name in enumerate(MODES)};assert sum(counts.values())==count
            grouped[label]=dict(samples=count,contact_mode_counts=counts,abs_COM_vertical_velocity_mm_s=quantiles(np.abs(velocity[use,2])))
        assert sum(x['samples'] for x in grouped.values())==990
        foot=(codes==3)&measured;inside=foot&groups['standing']
        longest=lambda mask:max((end-start for start,end in intervals(mask)),default=0)*.005
        cases[key]=dict(case_id=key,bank=bank,training_seed=None if bank=='parent' and not row['own_seed'] else birth,probe_seed=seed,own_seed=row['own_seed'],
            legacy_standing_fraction=native_standing,groups=grouped,
            longest_tarsal_only_sample_run_seconds=longest(foot),longest_legacy_standing_AND_tarsal_sample_run_seconds=longest(inside),
            physical_standing_validated=False,dynamic_balance_validated=False)
    assert len(cases)==37 and len(positions)==45
    for n,h in inputs.items():assert sha(root/n)==h
    assert sha(Path(__file__))==before and sha(summary_path)==manifest[summary_path.relative_to(root).as_posix()]['sha256']
    result=dict(status='completed_exploratory_AP2_failure_partition',seconds=time.monotonic()-started,unique_cases=37,bank_positions=45,
        cases=list(cases.values()),positions=positions,input_hashes=inputs,source_sha256=before,physics_steps=0,neural_steps=0,learning_updates=0,
        original_standing_masks_exact=True,reason_and_mode_counts_conservative=True,original_combined_pass_count=0,new_gate=False,new_reward=False,model_promoted=False,walking_proven=False,
        limits='Post-hoc descriptive partition of the old proxy, not ground-truth falls/standing. High/upright/claw/tarsal-only and COMspeed do not prove balance. Sample runs are end-of-step snapshots, not continuous physical stability.')
    p=root/'outputs/anatomical-feedback-ap2-failure-audit-20261107.json';p.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    for r in cases.values():
        if r['bank']=='candidate' and not r['own_seed']:
            print(json.dumps(dict(case=r['case_id'],standing=r['groups']['standing']['samples'],up_fail=r['groups']['upright_failure']['samples'],
                high=r['groups']['high_height_upright_claw']['samples'],high_tarsal=r['groups']['high_height_upright_claw']['contact_mode_counts']['tarsal_only'],
                standing_tarsal=r['groups']['standing']['contact_mode_counts']['tarsal_only'],longest_both=r['longest_legacy_standing_AND_tarsal_sample_run_seconds'])))
    print(json.dumps({k:v for k,v in result.items() if k not in ('cases','positions','input_hashes','limits')}))


if __name__=='__main__':main()
