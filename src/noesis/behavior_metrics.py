"""Body criteria measured independently of reward and neural rhythm."""
import numpy as np

from noesis.sensory import LEGS


def behavior_summary(rows):
    rows = [r for r in rows if r['awake']]
    if not rows:
        return {'standing_fraction': 0., 'leg_step_cycles': {leg: 0 for leg in LEGS},
                    'obstacle_contact_seconds': 0., 'obstacle_response_seconds': None,
                    'obstacle_observation_complete': False}
    times = np.array([r['time_s'] for r in rows])
    dt = float(np.median(np.diff(times))) if len(times) > 1 else .005
    standing = np.mean([r['uprightness'] >= .7 and .3 <= r['height_mm'] <= 1.5 for r in rows])
    cycles = {}
    # Debounce each state for 20 ms. Require physical clearance, not just adhesion release.
    for leg in LEGS:
        state, pending, count, seen_stance, lifted, steps = False, None, 0, False, False, 0
        for r in rows:
            next_state = 'stance' if r[f'{leg}_ground_load'] > 1e-5 else (
                'lift' if r[f'{leg}_foot_z_mm'] > .05 else 'other')
            count = count + 1 if next_state == pending else 1
            pending = next_state
            if count * dt < .02 or next_state == state:
                continue
            state = next_state
            if state == 'stance':
                if seen_stance and lifted:
                    steps += 1
                seen_stance, lifted = True, False
            elif state == 'lift' and seen_stance:
                lifted = True
        cycles[leg] = steps
    contacts = np.array([r['obstacle_force'] > 1e-5 for r in rows])
    response = None
    complete = False
    if contacts.any():
        i = int(np.flatnonzero(contacts)[0])
        complete = times[-1] - times[i] >= 1.
        # A sustained 100 ms stop relative to pre-contact speed, or >=30 degree turn.
        pre = [abs(r['forward_speed_mm_s']) for r in rows[max(0, i-20):i]]
        prior = float(np.mean(pre)) if pre else 0.
        stop_count = 0
        for j in range(i+1, len(rows)):
            if times[j] - times[i] > 1.:
                break
            turn = abs(np.arctan2(np.sin(rows[j]['yaw_rad']-rows[i]['yaw_rad']),
                                  np.cos(rows[j]['yaw_rad']-rows[i]['yaw_rad']))) >= np.pi/6
            stop = prior > .1 and abs(rows[j]['forward_speed_mm_s']) < .2*prior
            stop_count = stop_count + 1 if stop else 0
            if turn or stop_count*dt >= .1:
                response = float(times[j]-times[i])
                break
    return {'standing_fraction': float(standing), 'leg_step_cycles': cycles,
                'obstacle_contact_seconds': float(contacts.sum()*dt),
                'obstacle_response_seconds': response, 'obstacle_observation_complete': bool(complete)}
