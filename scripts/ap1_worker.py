"""Delegate unmodified native/S2 episode, opt-in scalar reward difference only."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
from ap1_body import observing_body
from ap1_feedback import apply_feedback,validate_ap1_training,OBJECTIVE


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--phase',choices=('train','frozen'),required=True);ap.add_argument('--bank',choices=('parent','control','candidate'),required=True)
    ap.add_argument('--strength',type=float,choices=(0.,.5),required=True);ap.add_argument('--seed',type=int,required=True)
    for n in ('root','output','checkpoint'):ap.add_argument('--'+n,type=Path,required=True)
    args=ap.parse_args();root=args.root.resolve()
    if args.seed not in (1005,10006,10007,10010,10011):ap.error('Unregistered seed')
    if args.phase=='train' and (args.seed!=1005 or args.bank=='parent' or args.strength!=(.5 if args.bank=='candidate' else 0)):ap.error('Wrong training arm')
    if args.phase=='frozen' and args.strength!=0:ap.error('No frozen correction/learning')
    sys.path[:0]=[str(root/'work/Noesis-Sensory/gpu-pilot/src'),str(root/'work/Noesis-Core/src')]
    import s2_worker
    from noesis_gpu import train,run
    module=train if args.phase=='train' else run;original_body=module.FlyGymBatch;original_factory=train.episode_scorer
    instances=[];scores=[];context=dict(experiment='AP1',phase=args.phase,bank=args.bank,seed=args.seed,
        effective_training_penalty=args.strength if args.phase=='train' else 0.,source_bank_penalty=.5 if args.bank=='candidate' else 0.,
        worker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),base_objective='curriculum-standing-v7',
        actual_training_objective=OBJECTIVE if args.phase=='train' and args.strength else 'curriculum-standing-v7',
        loaded_checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),learning_rule_unchanged=True)
    args.output.with_name(args.output.name+'-ap1-context.json').write_text(json.dumps(context,indent=2),encoding='utf-8')

    def factory(objective,cycle_weight=None):
        if objective!='curriculum-standing-v7' or cycle_weight is not None:raise ValueError('Only declared base objective')
        base=original_factory(objective,cycle_weight)
        def scorer(before,after,dt_ms):
            if dt_ms!=5 or len(instances)!=1:raise ValueError('Wrong scorer/body context')
            body=instances[0]
            assert len(body.ap1_records)==len(scores)+2
            np.testing.assert_array_equal(after['position'],body.ap1_latest_position)
            row=body.ap1_records[-1];result=base(before,after,dt_ms);native=result['reward']
            applied=apply_feedback(result,float(row['non_tarsal_fraction']),args.strength);shaped=applied['reward']
            scores.append(dict(control_chunk_index=np.array(len(scores)),time=row['time'].copy(),native_reward=np.array(native),
                non_tarsal_fraction=row['non_tarsal_fraction'].copy(),applied_reward=np.array(shaped),effective_penalty=np.array(native-shaped)))
            return applied
        return scorer

    module.FlyGymBatch=observing_body(original_body,instances)
    if args.phase=='train':train.episode_scorer=factory
    oldargs=sys.argv.copy();sys.argv=['s2_worker.py','--phase',args.phase,'--mode','legacy','--seed',str(args.seed),
        '--root',str(root),'--output',str(args.output),'--checkpoint',str(args.checkpoint)]
    success=False
    try:
        s2_worker.main()
        assert len(instances)==1 and len(instances[0].ap1_records)==(5001 if args.phase=='train' else 1001)
        if args.phase=='train':assert len(scores)==5000
        success=True
    finally:
        module.FlyGymBatch=original_body;train.episode_scorer=original_factory;sys.argv=oldargs
        if instances and args.output.exists():
            body=instances[0]
            if body.ap1_records:np.savez_compressed(args.output/'anatomy-observations.npz',**{k:np.stack([r[k] for r in body.ap1_records]) for k in body.ap1_records[0]})
            if scores:np.savez_compressed(args.output/'reward-accounting.npz',**{k:np.stack([r[k] for r in scores]) for k in scores[0]})
            (args.output/'anatomy-model.json').write_text(json.dumps(body.ap1_metadata,indent=2),encoding='utf-8')
        path=args.output/'report.json'
        if path.exists():
            report=json.loads(path.read_text());report.update(ap1_context=context,ap1_worker_completed=success,anatomical_measurement_not_balance_validation=True)
            if args.phase=='train':
                report.update(base_reward_objective='curriculum-standing-v7',reward_objective=OBJECTIVE if args.strength else 'curriculum-standing-v7')
            path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
            if args.phase=='train':
                cp=args.output/('weights-last.npz' if report['status']=='completed_training_birth' else 'weights-partial.npz')
                if cp.exists():cp.with_suffix('.model.json').write_text(json.dumps({**report,'context':'AP1 opt-in curriculum candidate/control; weights-only memory, not exact episode resume or validated bank'},indent=2),encoding='utf-8')
    if args.phase=='train':validate_ap1_training(report,args.strength)
    print(json.dumps(dict(experiment='AP1',phase=args.phase,bank=args.bank,completed=success)))


if __name__=='__main__':main()
