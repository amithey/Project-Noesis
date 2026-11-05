import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_standing_entry import failures


def observation(n=100):
    p=np.zeros((n,3));p[:,2]=.5
    u=np.zeros((n,3));u[:,2]=1
    return dict(position=p,up_body=u,foot_load=np.ones((n,6)))


def test_overlapping_failure_causes_are_not_counted_as_separate_episodes():
    o=observation();o['position'][20:40,2]=.2;o['up_body'][30:50,2]=.6
    r=failures(o)
    assert r['failed_samples']==30 and r['episodes']==1 and r['max_failure_ms']==150
    assert r['reasons']['height_below']==20 and r['reasons']['upright_below']==20
    assert r['failures_at_least100ms']==1


def test_original_thresholds_remain_inclusive_and_warmup_excluded():
    o=observation();o['position'][:10,2]=0
    o['position'][10:,2]=.3;o['up_body'][10:,2]=.7
    assert failures(o)['standing_fraction']==1
    o['foot_load'][50:]=0
    r=failures(o);assert r['failed_samples']==50 and r['longest_episodes'][0]['right_censored']
