"""Offline reward prototype; not used by the running simulator."""

import math


def reward(row):
    keys = (
        "uprightness",
        "height_mm",
        "supporting_feet",
        "grounded_progress",
        "effort",
        "stance_slip_mm_s",
        "obstacle_cost",
    )
    v = {k: float(row[k]) for k in keys}
    if not all(math.isfinite(x) for x in v.values()):
        raise ValueError("reward observations must be finite")
    if not (
        -1 <= v["uprightness"] <= 1
        and 0 <= v["supporting_feet"] <= 6
        and v["supporting_feet"].is_integer()
        and -1 <= v["grounded_progress"] <= 1
        and 0 <= v["effort"] <= 1
        and v["stance_slip_mm_s"] >= 0
        and 0 <= v["obstacle_cost"] <= 1
    ):
        raise ValueError("invalid bounded reward observations")
    standing = (
        v["uprightness"] >= 0.7 and 0.3 <= v["height_mm"] <= 1.5 and v["supporting_feet"] >= 1
    )
    progress = v["grounded_progress"] if standing and v["supporting_feet"] >= 3 else 0.0
    quality = min(v["supporting_feet"] / 3, 1)
    value = (
        0.7 * standing
        - 0.8 * (1 - standing)
        + 0.1 * standing * quality
        + 0.5 * progress
        - 0.05 * v["effort"]
        - 0.006 * v["stance_slip_mm_s"]
        - 0.2 * v["obstacle_cost"]
    )
    return max(-1.0, min(1.0, value))
