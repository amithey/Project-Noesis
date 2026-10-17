"""Artificial training reward for upright walking; this is not a biological objective.

Distances are in mm, time in seconds, and joint velocities in rad/s. World +x is
the experiment's forward direction. Height and speed gates are heuristic guards,
not a substitute for measuring actual ground contact.
"""

from __future__ import annotations

import numpy as np

GROUNDED_OBJECTIVES = ("standing-first-linear-slip-v3", "standing-gate-linear-slip-v4")


def body_reward(
    previous_position,
    position,
    uprightness: float,
    joint_velocity,
    dt_seconds: float,
) -> dict[str, float]:
    """Return a bounded reward and body metrics without changing any input.

    Progress is rewarded only with uprightness >= 0.7 and height 0.3..1.5 mm.
    Forward speed saturates at 10 mm/s; speeds beyond 30 mm/s receive no progress
    credit to discourage launch artifacts. Low posture/height counts as a fall.
    ``effort`` is tanh(mean squared joint velocity / 100), a bounded motion-cost
    proxy, not mechanical work or metabolic energy.
    """
    previous = np.asarray(previous_position, dtype=float)
    current = np.asarray(position, dtype=float)
    velocity = np.asarray(joint_velocity, dtype=float)
    if previous.shape != (3,) or current.shape != (3,):
        raise ValueError("positions must each have shape (3,)")
    if velocity.ndim != 1 or velocity.size == 0:
        raise ValueError("joint_velocity must be a nonempty one-dimensional array")
    if not all(np.all(np.isfinite(a)) for a in (previous, current, velocity)):
        raise ValueError("positions and joint velocities must be finite")
    dt = float(dt_seconds)
    up = float(uprightness)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt_seconds must be finite and positive")
    if not np.isfinite(up) or not -1.0 <= up <= 1.0:
        raise ValueError("uprightness must be finite and within [-1, 1]")

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        speed = float((current[0] - previous[0]) / dt)
    if not np.isfinite(speed):
        raise ValueError("position change and dt_seconds produce nonfinite speed")
    height = float(current[2])
    # Clipping before squaring avoids overflow for finite but extreme velocities.
    effort = float(np.tanh(np.mean(np.square(np.clip(velocity / 10.0, -10, 10)))))
    grounded_posture = 0.3 <= height <= 1.5 and up >= 0.7
    progress = float(np.clip(speed / 10.0, -1.0, 1.0)) if (
        grounded_posture and abs(speed) <= 30.0
    ) else 0.0
    fallen = up < 0.5 or height < 0.3
    reward = float(np.clip(0.1 * max(up, 0.0) + 0.75 * progress
                           - 0.6 * fallen - 0.05 * effort, -1.0, 1.0))
    return {
        "reward": reward,
        "forward_speed_mm_s": speed,
        "uprightness": up,
        "height_mm": height,
        "effort": effort,
    }


def locomotion_reward(previous_position, position, uprightness, joint_velocity, dt_seconds,
                      heading, leg_loads, foot_velocity, obstacle_contact_seconds=0., *,
                      objective="standing-first-linear-slip-v3"):
    """Artificial grounded-forward objective with a measured stance-slip penalty.

    Uses horizontal thorax heading, >=3 force-bearing legs, and leg joint effort.
    The 1e-5 force threshold and 5 mm/s slip scale are explicit assumptions.
    This contact gate rejects airborne translation; it does not certify gait.
    The opt-in v4 objective credits actual upright/height-valid standing with at
    least one force-bearing foot; v3 remains the exact default objective.
    """
    if objective not in GROUNDED_OBJECTIVES:
        raise ValueError("unknown grounded reward objective")
    base = body_reward(previous_position, position, uprightness, joint_velocity, dt_seconds)
    heading = np.asarray(heading, float)
    loads = np.asarray(leg_loads, float)
    feet = np.asarray(foot_velocity, float)
    if (heading.shape != (3,) or loads.shape != (6,) or feet.shape != (6, 3)
            or not all(np.isfinite(x).all() for x in (heading, loads, feet))
            or np.any(loads < 0)):
        raise ValueError("heading, nonnegative leg loads and foot velocities must be finite")
    norm = np.linalg.norm(heading[:2])
    direction = heading[:2] / norm if norm > 1e-12 else np.zeros(2)
    speed = float((np.asarray(position)[:2] - np.asarray(previous_position)[:2]) @ direction
                  / dt_seconds)
    if not np.isfinite(speed):
        raise ValueError("nonfinite forward speed")
    support = loads > 1e-5
    grounded = support.sum() >= 3 and .3 <= base['height_mm'] <= 1.5 and uprightness >= .7
    progress = float(np.clip(speed / 10, -1, 1)) if grounded and abs(speed) <= 30 else 0.
    slip = float(np.mean(np.linalg.norm(feet[support, :2], axis=1))) if support.any() else 0.
    if not np.isfinite(obstacle_contact_seconds) or obstacle_contact_seconds < 0:
        raise ValueError("obstacle contact duration must be finite and nonnegative")
    grip = float(np.clip(1 - slip / (abs(speed) + .1), 0, 1))
    progress *= grip
    obstacle_cost = float(np.clip(obstacle_contact_seconds, 0, 1))
    # Match the mean-slip criterion: keep this cost linear instead of rewarding
    # a few large slip bursts through a concave/saturating penalty. Only the
    # final total reward is bounded, so high sliding continues to cost credit.
    slip_cost = slip / 5
    posture_quality = float(np.clip((uprightness - .5) / .4, 0, 1))
    support_quality = float(min(support.sum() / 3, 1))
    standing_quality = posture_quality * support_quality if .3 <= base['height_mm'] <= 1.5 else 0.
    # Standing-first candidate: preserve a linear measured slip cost while
    # reducing its scale tenfold so typical upright sliding cannot dominate
    # the support/posture signal. Continuous support loss also costs reward.
    # Physical standing, gait, mean-slip and obstacle acceptance stay separate.
    reward = float(np.clip(.75 * standing_quality + .5 * progress - .6 * (1 - standing_quality)
                           - .05 * base['effort'] - .03 * slip_cost - .2 * obstacle_cost, -1, 1))
    if objective == "standing-gate-linear-slip-v4":
        # Explicit standing credit cannot be replaced by a third loaded foot.
        # This is scalar learning feedback; it never produces a motor action.
        standing = bool(support.any() and .3 <= base['height_mm'] <= 1.5 and uprightness >= .7)
        reward = float(np.clip(.7 * standing - .8 * (1 - standing)
                               + .1 * standing * support_quality + .5 * progress
                               - .05 * base['effort'] - .03 * slip_cost - .2 * obstacle_cost, -1, 1))
    return {**base, 'reward': reward, 'forward_speed_mm_s': speed,
            'supporting_feet': int(support.sum()), 'stance_slip_mm_s': slip,
            'grounded_progress': progress, 'grip_factor': grip, 'obstacle_cost': obstacle_cost,
            'standing_quality': standing_quality}
