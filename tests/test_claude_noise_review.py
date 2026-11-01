import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from review_claude_noise import classify,PROBES


def rows(heading):
    return {seed:dict(standing_fraction=.95,mean_stance_slip_mm_s=2.,
        heading_forward_distance_mm=heading,net_forward_displacement_mm=heading) for seed in PROBES}


def test_three_separate_metric_passes_can_hide_only_two_joint_passes():
    real,control=rows(1.3),rows(.3)
    real[PROBES[0]]['mean_stance_slip_mm_s']=2.1
    real[PROBES[3]]['standing_fraction']=.94
    assert classify(real,control)[0]['verdict']=='REAL_BETTER'
    strict=classify(real,control,strict=True)[0]
    assert strict['same_probes_retained']==2 and strict['verdict']=='INDISTINGUISHABLE'


def test_tied_standing_and_slip_must_count_as_noninferior_in_both_directions():
    real,control=rows(.3),rows(1.3)
    assert classify(real,control)[0]['verdict']=='INDISTINGUISHABLE'
    assert classify(real,control,strict=True)[0]['verdict']=='CONTROL_BETTER'


def test_real_gain_with_every_probe_retained_agrees_under_both_rules():
    for strict in (False,True):assert classify(rows(1.3),rows(.3),strict)[0]['verdict']=='REAL_BETTER'
