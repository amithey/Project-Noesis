import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from training_resources import assess,window_due


def samples():
    return [dict(available_gib=3.5,host_cpu_percent=20.) for _ in range(10)]


def test_transient_good_snapshot_cannot_hide_pressure():
    rows=samples();assert assess(rows)['passed']
    rows[3]['available_gib']=.62
    assert not assess(rows)['passed']
    rows=samples();rows[6]['host_cpu_percent']=100
    assert not assess(rows)['passed']


def test_missing_and_invalid_observations_fail_closed():
    with pytest.raises(ValueError):assess(samples()[:1])
    rows=samples();rows[1]['available_gib']=float('nan')
    with pytest.raises(ValueError):assess(rows)


def test_watch_windows_fit_budget_and_do_not_busy_retry():
    assert window_due(0,None)
    assert not window_due(30,0)
    assert window_due(60,0)
    assert not window_due(891,840)
    assert not window_due(901,840)
