import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_standing_baseline import decision,PROBES


def cases():
    return {arm:{s:dict(single_probe_standing_check=True,standing_fraction=.95,mean_stance_slip_mm_s=2.,
        heading_forward_distance_mm=0.,net_forward_displacement_mm=0.) for s in PROBES} for arm in ('parent','anchor')}


def test_qualification_never_implies_walking_promotion_or_stage_advancement():
    r=decision(cases());assert r['four_probe_anchor_standing_gate_passed']
    assert not r['walking_proven'] and not r['model_promoted'] and not r['stage_advanced']


def test_one_failed_or_missing_probe_cannot_qualify_the_anchor():
    c=cases();c['anchor'][10011]['single_probe_standing_check']=False
    assert not decision(c)['four_probe_anchor_standing_gate_passed']
    del c['anchor'][10011]
    with pytest.raises(ValueError):decision(c)


def test_absolute_gate_cannot_hide_a_paired_standing_or_slip_regression():
    c=cases();c['anchor'][10010]['standing_fraction']=.94
    r=decision(c);assert r['four_probe_anchor_standing_gate_passed']
    assert not r['anchor_standing_slip_retained_every_probe']
