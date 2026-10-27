"""Artificial training reward for upright walking; this is not a biological objective.

Distances are in mm, time in seconds, and joint velocities in rad/s. World +x is
the experiment's forward direction. Height and speed gates are heuristic guards,
not a substitute for measuring actual ground contact.
"""

from __future__ import annotations

import numpy as np


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
