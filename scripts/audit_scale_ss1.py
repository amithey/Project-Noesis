"""Terminal SS1 raw-evidence and independent engineering-criterion audit."""
import argparse
from pathlib import Path
import numpy as np
import run_scale_ss1 as run
from ss1_scale import SCALES,SIGNAL,COMPATIBILITY,analyze,panel


def main(out):
    out=out.resolve();p=run.read(out/'protocol.json');r=run.read(out/'result.json')
    if r['status']!='completed_SS1' or len(r['jobs'])!=25 or r['seconds']>run.LIMIT:raise ValueError('Incomplete/budget-failed allocation')
    if not all(r[k] for k in ('baseline_exact','source_input_data_created_unchanged','process_tree_empty_verified','lease_released','awake_request_released')):
        raise ValueError('Failed terminal guards')
    pins={**p['prior_hashes'],**p['source_hashes'],**p['input_hashes'],**r['created_hashes']};run.base.check_pins(pins)
    runtime=Path(p['runtime_root']);_,old,_=run.base.legacy(runtime)
    if (runtime/'work/Project-Noesis/gpu-pilot/outputs/codex-training-host.lock').exists():raise ValueError('Heavy lease active')
    parent_cp=run.arrays(runtime/p['parent_checkpoint']);directions=run.arrays(old.inputs/'directions.npz')['directions']
    np.testing.assert_array_equal(directions,old.fixed_directions(len(parent_cp['weights'])))
    for h in SCALES[1:]:
        for k in (-1,1):
            expected=old.apply_delta(parent_cp,k*h*directions[0]);actual=run.arrays(Path(p['inputs'])/f'probe-{h}-{k}.npz')
            if set(actual)!=set(expected):raise ValueError('Diagnostic checkpoint schema changed')
            for field in expected:np.testing.assert_array_equal(actual[field],expected[field])
    records=run.reused(p,old)
    for actual,j in zip(r['jobs'],run.jobs()):
        if any(actual[k]!=v for k,v in j.items()):raise ValueError('Wrong/misordered registered cell')
        name=j['name'];receipt=run.read(out/(name+'-receipt.json'))
        if (receipt!=actual['receipt'] or receipt['exit_code']!=0 or receipt['timed_out'] or not receipt['tree_empty']
                or receipt['seconds']>300):raise ValueError('Failed receipt')
        folder=out/name;report=run.read(folder/'report.json');cp=run.checkpoint(p,j)
        run.validate_report(report,j,run.sha(cp),p['source_hashes'][str(run.HERE/'ss1_worker.py')])
        if report['ss1_context']!=run.read(out/(name+'-ss1-context.json')):raise ValueError('Context mismatch')
        ref=run.reference(p,j['seed']);old.native.compare(ref,folder)
        if j['phase']=='baseline':
            for n in ('observations.npz','adhesion-observations.npz','anatomy-observations.npz'):old.native.exact(ref/n,folder/n)
            if report['spikes_per_world']!=run.read(ref/'report.json')['spikes_per_world']:raise ValueError('Baseline spike mismatch')
        else:records.append(run.base.measured_record(folder,{k:j[k] for k in ('phase','bank','seed','direction','sign','amplitude')},old))
    if records!=run.read(out/'training-returns.json'):raise ValueError('Raw TRAIN returns changed')
    parent,returns=panel(records);analysis=analyze(parent,returns);saved=run.read(out/'scale-analysis-private.json')
    for k,v in analysis.items():
        if isinstance(v,np.ndarray):np.testing.assert_allclose(v,saved[k],rtol=1e-12,atol=1e-14)
        elif v!=saved[k]:raise ValueError('Saved analysis mismatch')
    independently_passed=[]
    for s in range(6):
        odds=[(float(returns[s,i,1])-float(returns[s,i,0]))/2 for i in range(3)]
        slopes=[odds[i]/SCALES[i] for i in range(3)]
        midpoint=[(float(returns[s,i,1])+float(returns[s,i,0]))/2-float(parent[s]) for i in range(3)]
        ratios=[abs(midpoint[i])/max(abs(odds[i]),SIGNAL) for i in (1,2)]
        discrepancy=abs(slopes[1]-slopes[2])/max(abs(slopes[1]),abs(slopes[2]),1e-12)
        passed=all(abs(odds[i])>SIGNAL for i in (1,2)) and np.sign(slopes[1])==np.sign(slopes[2]) and discrepancy<=COMPATIBILITY and all(v<=COMPATIBILITY for v in ratios)
        independently_passed.append(bool(passed))
    if independently_passed!=analysis['context_compatible'].tolist():raise ValueError('Independent criteria mismatch')
    for k in ('compatible_contexts','all_contexts_compatible','smallest_scale_cross_context_sign_consistent'):
        if r[k]!=analysis[k]:raise ValueError('Result endpoint mismatch')
    run.base.check_pins(pins)
    public=dict(status='completed_independent_SS1_audit',full_new_workers=25,training_contexts=6,canonical_direction=0,
        complete_training_cases=42,raw_returns_and_scale_criteria_reproduced=True,actual_probe_import_and_reference_bounds_verified=True,
        exact_baseline=True,compatible_contexts=analysis['compatible_contexts'],all_contexts_compatible=analysis['all_contexts_compatible'],
        smallest_scale_cross_context_sign_consistent=analysis['smallest_scale_cross_context_sign_consistent'],
        model_selected=False,model_promoted=False,standing_proven=False,walking_proven=False,limits=analysis['limits'])
    target=out.with_name(out.name+'-minimal-result.json')
    if target.exists():raise ValueError('Never overwrite final audit')
    run.write(target,public);print(__import__('json').dumps(public))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();main(a.output)
