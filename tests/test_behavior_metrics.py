import pytest

from noesis.behavior_metrics import behavior_summary
from noesis.sensory import LEGS


def row(t, stance=True, lift=False):
    r = {'time_s': t, 'awake': True, 'uprightness': 1., 'height_mm': .8, 'obstacle_force': 0.,
             'forward_speed_mm_s': 1., 'yaw_rad': 0.}
    for leg in LEGS:
        r[f'{leg}_ground_load'] = 1. if stance else 0.
        r[f'{leg}_foot_z_mm'] = .2 if lift else .01
    return r


def test_two_real_stance_lift_stance_cycles_and_no_credit_for_contact_chatter():
    rows = [row(i*.005, stance=(i//10)%2 == 0, lift=(i//10)%2 == 1) for i in range(50)]
    assert set(behavior_summary(rows)['leg_step_cycles'].values()) == {2}
    chatter = [row(i*.005, stance=i%2 == 0, lift=i%2 == 1) for i in range(100)]
    assert set(behavior_summary(chatter)['leg_step_cycles'].values()) == {0}
    slipping = [row(i*.005) for i in range(50)]
    assert set(behavior_summary(slipping)['leg_step_cycles'].values()) == {0}


def test_obstacle_stop_requires_previous_motion_and_sustained_stop():
    rows = [row(i*.005) for i in range(260)]
    for r in rows[20:]:
        r['obstacle_force'] = 1.
    for r in rows[30:]:
        r['forward_speed_mm_s'] = 0.
    summary = behavior_summary(rows)
    assert summary['obstacle_observation_complete']
    assert .1 <= summary['obstacle_response_seconds'] <= .16
    assert summary['obstacle_contact_seconds'] == pytest.approx(1.2)
    for r in rows:
        r['forward_speed_mm_s'] = 0.
    assert behavior_summary(rows)['obstacle_response_seconds'] is None
