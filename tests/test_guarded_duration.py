"""T1B admission/receipt tests are synthetic, not physical learning evidence."""
import copy
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_duration_study import (resource_decision,guarded_receipt_metadata,CORE,PARENT,CONTROL,
    SOURCES,TRACE_FIELDS,LOG_FIELDS)


def metadata():
    protocol=dict(experiment='D4M guarded training replay',core_commit=CORE,source_hashes=SOURCES,
        parent_sha256=PARENT,expected_checkpoint_sha256=CONTROL,seed=1000,seconds=5,chunks=1000,
        learning_updates=990,learning_rate=.2,cycle_weight=.1,exploration_mv_ms=1.5,
        cpu_admission_statistic='mean',launch_mean_cpu_percent=60,launch_minimum_gib=2,
        launch_samples=10,in_run_stop_gib=1,telemetry_interval_seconds=2,
        aggregate_seconds=420,inner_seconds=300,process_seconds=330,prior_wall_seconds=10.)
    result=dict(status='completed_guarded_training_replay',checkpoint_sha256=CONTROL,
        shared_training_last_updated=False,T1_gate_changed=False,scientific_gates_changed=False,
        model_promoted=False,walking_proven=False,trace_exact_fields={k:True for k in TRACE_FIELDS},
        exploration_exact_fields={k:True for k in LOG_FIELDS},wall_seconds=100.,aggregate_active_seconds=110.,
        jobs=[dict(exit_code=0,final_exit_code=0)])
    return protocol,result


def test_guarded_admission_is_opt_in_and_records_legacy_decision():
    samples=[dict(available_gib=2.5,host_cpu_percent=40.) for _ in range(10)]
    saved=copy.deepcopy(samples)
    assert not resource_decision(samples)['passed']
    guarded=resource_decision(samples,True)
    assert guarded['passed'] and guarded['original_T1_rule_passed'] is False
    assert guarded['cpu_decision_statistic']=='mean' and samples==saved
    samples[0]['available_gib']=1.99
    assert not resource_decision(samples,True)['passed']


@pytest.mark.parametrize('change',['partial','bank','trace','log','budget','policy','source','process','promoted',None])
def test_only_complete_unchanged_d4m_prerequisite_can_enable_guarded_mode(change):
    protocol,result=metadata()
    if change=='partial':result['status']='stopped'
    if change=='bank':result['checkpoint_sha256']='partial'
    if change=='trace':del result['trace_exact_fields']['command']
    if change=='log':result['exploration_exact_fields']['credit']=False
    if change=='budget':result['aggregate_active_seconds']=421
    if change=='policy':protocol['launch_mean_cpu_percent']=90
    if change=='source':protocol['source_hashes']={'core':'other','pilot':'other'}
    if change=='process':result['jobs'][0]['final_exit_code']=-9
    if change=='promoted':result['model_promoted']=True
    if change:
        with pytest.raises(ValueError):guarded_receipt_metadata(protocol,result,SOURCES)
    else:guarded_receipt_metadata(protocol,result,SOURCES)
