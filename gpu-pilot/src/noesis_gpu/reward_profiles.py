"""Curriculum stage -> grounded reward objective mapping for the FlyGym 2 pilot.

Names mirror ``noesis.curriculum.STAGES`` in the pinned core. Reward formulas stay
in the core's ``locomotion_reward``; this module only names them. Standing is the
default, and exact core objective ids are accepted unchanged.
"""

from __future__ import annotations

DEFAULT_PROFILE = "standing"

PROFILES = {
    "standing": "curriculum-standing-v7",
    "gripped-steps": "curriculum-gripped-step-v8",
    "continuous-walking": "standing-gate-smooth-grip-v6",
    "obstacles": "standing-gate-smooth-grip-v6",
}


def resolve_objective(name=None):
    """Return the core objective id for a stage name or an exact objective id."""
    if name is None:
        name = DEFAULT_PROFILE
    if name in PROFILES:
        return PROFILES[name]
    if name in PROFILES.values():
        return name
    raise ValueError(f"unknown reward profile or objective: {name!r}")
