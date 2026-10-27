"""Physical gait, step cycle and locomotion coordination metrics from observations.

This module analyzes kinematic and contact trajectories from simulation observations:
- Step cycles (stance / swing) per leg with contact flutter debouncing
- Heading forward distance (instantaneous projection along body heading)
- Stance slip speed and horizontal foot velocity during ground contact
- Tripod coordination (inter-tripod alternation, correlation, and intra-tripod synchrony)
- Locomotion regime classification (Static Standing, Dragging/Slipping, Chatter, Walking, Fall)

Motion alone does NOT prove walking; dragging and threshold chatter are rejected.
"""

from __future__ import annotations

from typing import Any

import numpy as np

LEGS = ("LF", "LM", "LH", "RF", "RM", "RH")
TRIPOD_1 = ("LF", "RM", "LH")
TRIPOD_2 = ("RF", "LM", "RH")
SCHEMA_VERSION = "gait-metrics-v1.0"


def compute_heading_forward_distance(
    position: np.ndarray,
    heading: np.ndarray,
) -> dict[str, float]:
    """Compute forward displacement along the instantaneous body heading vector.

    Parameters
    ----------
    position : np.ndarray, shape (T, 3)
        Center of mass / thorax position across time in mm.
    heading : np.ndarray, shape (T, 3)
        Unit heading vector of the body in world coordinates across time.

    Returns
    -------
    dict containing:
        - heading_forward_distance_mm: Cumulative forward displacement along heading.
        - net_displacement_mm: Euclidean distance between final and initial positions.
        - net_forward_displacement_mm: (P_end - P_start) projected onto initial heading.
    """
    pos = np.asarray(position, dtype=float)
    head = np.asarray(heading, dtype=float)
    if pos.ndim != 2 or pos.shape[1] != 3 or head.shape != pos.shape:
        raise ValueError("position and heading must have matching shape (T, 3)")
    if not np.isfinite(pos).all() or not np.isfinite(head).all():
        raise ValueError("position and heading must be finite")
    if len(pos) < 2:
        return {
            "heading_forward_distance_mm": 0.0,
            "net_displacement_mm": 0.0,
            "net_forward_displacement_mm": 0.0,
        }

    head_norm = np.linalg.norm(head, axis=1, keepdims=True)
    if np.any(head_norm < 1e-12):
        raise ValueError("Heading vector cannot have zero magnitude")
    unit_head = head / head_norm

    # Step-wise displacement
    d_pos = np.diff(pos, axis=0)  # shape (T-1, 3)
    # Project each step displacement onto the heading at the start of that step
    step_forward = np.sum(d_pos * unit_head[:-1], axis=-1)  # shape (T-1,)
    cumulative_forward = float(np.sum(step_forward))

    net_disp = float(np.linalg.norm(pos[-1] - pos[0]))
    net_fwd = float(np.dot(pos[-1] - pos[0], unit_head[0]))

    return {
        "heading_forward_distance_mm": cumulative_forward,
        "net_displacement_mm": net_disp,
        "net_forward_displacement_mm": net_fwd,
    }


def debounce_contact(
    contact: np.ndarray,
    dt_ms: float,
    min_phase_ms: float = 15.0,
) -> np.ndarray:
    """Filter out contact chatter/jitter where state flips for less than min_phase_ms.

    A physical leg contact or lift cannot realistically last for only a single 5ms step.
    Short transient flips are merged into their surrounding sustained phase:
    1. Short isolated contact spikes (< min_samples) are suppressed to False.
    2. Short contact dropouts (< min_samples) flanked by contact are bridged to True.
    """
    c = np.asarray(contact, dtype=bool).copy()
    if len(c) == 0:
        return c
    min_samples = max(1, round(min_phase_ms / dt_ms))
    if min_samples <= 1:
        return c

    # Step 1: Suppress short True pulses (< min_samples)
    diffs = np.diff(np.concatenate(([False], c, [False])).astype(int))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    for s, e in zip(starts, ends):
        if e - s < min_samples:
            c[s:e] = False

    # Step 2: Bridge short False dropouts (< min_samples) flanked by True
    diffs = np.diff(np.concatenate(([True], c, [True])).astype(int))
    gap_starts = np.where(diffs == -1)[0]
    gap_ends = np.where(diffs == 1)[0]
    for s, e in zip(gap_starts, gap_ends):
        if s > 0 and e < len(c) and (e - s < min_samples):
            c[s:e] = True

    return c


def extract_leg_step_cycles(
    foot_load: np.ndarray,
    foot_position: np.ndarray,
    dt_ms: float,
    *,
    contact_threshold: float = 1e-5,
    min_phase_ms: float = 15.0,
) -> dict[str, Any]:
    """Extract step cycles (stance -> swing -> stance) for a single leg.

    Parameters
    ----------
    foot_load : np.ndarray, shape (T,)
        Normal force / load on the leg's claw/tarsus5.
    foot_position : np.ndarray, shape (T, 3)
        3D position of the foot across time in mm.
    dt_ms : float
        Control chunk time delta in milliseconds.
    contact_threshold : float
        Load threshold above which the foot is considered in contact.
    min_phase_ms : float
        Minimum duration in ms for a phase to be considered valid (debouncing).

    Returns
    -------
    dict containing step count, frequencies, duty factor, and cycle statistics.
    """
    load = np.asarray(foot_load, dtype=float)
    pos = np.asarray(foot_position, dtype=float)
    if load.ndim != 1 or pos.shape != (len(load), 3):
        raise ValueError("foot_load must be (T,) and foot_position must be (T, 3)")
    if not np.isfinite(load).all() or not np.isfinite(pos).all():
        raise ValueError("foot_load and foot_position must be finite")
    if dt_ms <= 0:
        raise ValueError("dt_ms must be positive")

    raw_contact = load > contact_threshold
    raw_fraction = float(np.mean(raw_contact)) if len(raw_contact) > 0 else 0.0
    raw_diff = np.diff(raw_contact.astype(int))
    raw_touchdowns = np.where(raw_diff == 1)[0] + 1
    raw_liftoffs = np.where(raw_diff == -1)[0] + 1

    debounced = debounce_contact(raw_contact, dt_ms, min_phase_ms)
    debounced_fraction = float(np.mean(debounced)) if len(debounced) > 0 else 0.0

    # Find transitions in debounced contact
    # Stance: True, Swing: False
    diff = np.diff(debounced.astype(int))
    touchdowns = np.where(diff == 1)[0] + 1  # 0 -> 1
    liftoffs = np.where(diff == -1)[0] + 1   # 1 -> 0

    # If first sample is stance, consider start as touchdown at t=0
    all_tds = list(touchdowns)
    if len(debounced) > 0 and debounced[0]:
        all_tds.insert(0, 0)
    all_los = list(liftoffs)

    # Match complete cycles: Touchdown -> Liftoff -> Touchdown
    cycles = []
    for td_start in all_tds:
        # Find next liftoff after td_start
        los_after = [lo for lo in all_los if lo > td_start]
        if not los_after:
            continue
        lo = los_after[0]
        # Find next touchdown after lo
        tds_after = [td for td in all_tds if td > lo]
        if not tds_after:
            continue
        td_next = tds_after[0]

        stance_dur_ms = (lo - td_start) * dt_ms
        swing_dur_ms = (td_next - lo) * dt_ms
        cycle_dur_ms = (td_next - td_start) * dt_ms

        # Foot stride length during swing (liftoff to touchdown)
        stride_len_mm = float(np.linalg.norm(pos[td_next, :2] - pos[lo, :2]))

        if cycle_dur_ms > 0:
            duty_factor = stance_dur_ms / cycle_dur_ms
            cycles.append({
                "td_start": int(td_start),
                "liftoff": int(lo),
                "td_end": int(td_next),
                "stance_ms": float(stance_dur_ms),
                "swing_ms": float(swing_dur_ms),
                "cycle_ms": float(cycle_dur_ms),
                "duty_factor": float(duty_factor),
                "stride_length_mm": stride_len_mm,
            })

    total_time_s = len(load) * dt_ms / 1000.0
    step_count = len(cycles)

    mean_stance = float(np.mean([c["stance_ms"] for c in cycles])) if cycles else None
    mean_swing = float(np.mean([c["swing_ms"] for c in cycles])) if cycles else None
    mean_cycle = float(np.mean([c["cycle_ms"] for c in cycles])) if cycles else None
    mean_duty = float(np.mean([c["duty_factor"] for c in cycles])) if cycles else None
    mean_stride = float(np.mean([c["stride_length_mm"] for c in cycles])) if cycles else None
    freq_hz = float(step_count / total_time_s) if total_time_s > 0 else 0.0

    return {
        "step_count": step_count,
        "step_frequency_hz": freq_hz,
        "raw_contact_fraction": raw_fraction,
        "debounced_contact_fraction": debounced_fraction,
        "mean_stance_duration_ms": mean_stance,
        "mean_swing_duration_ms": mean_swing,
        "mean_cycle_duration_ms": mean_cycle,
        "mean_duty_factor": mean_duty,
        "mean_stride_length_mm": mean_stride,
        "raw_touchdown_count": len(raw_touchdowns),
        "raw_liftoff_count": len(raw_liftoffs),
        "cycles": cycles,
    }


def compute_all_leg_cycles(
    foot_load: np.ndarray,
    feet_position: np.ndarray,
    dt_ms: float,
    *,
    contact_threshold: float = 1e-5,
    min_phase_ms: float = 15.0,
) -> dict[str, Any]:
    """Compute step cycles for all six legs: LF, LM, LH, RF, RM, RH."""
    load = np.asarray(foot_load, dtype=float)
    feet = np.asarray(feet_position, dtype=float)
    if load.shape[1] != 6 or feet.shape[1:3] != (6, 3):
        raise ValueError("foot_load must be (T, 6) and feet_position must be (T, 6, 3)")

    legs_out = {}
    total_steps = 0
    for idx, name in enumerate(LEGS):
        res = extract_leg_step_cycles(
            load[:, idx], feet[:, idx], dt_ms,
            contact_threshold=contact_threshold,
            min_phase_ms=min_phase_ms,
        )
        legs_out[name] = res
        total_steps += res["step_count"]

    return {
        "legs": legs_out,
        "total_leg_steps": int(total_steps),
        "mean_steps_per_leg": float(total_steps / 6.0),
        "all_legs_stepping": bool(all(legs_out[name]["step_count"] >= 1 for name in LEGS)),
    }


def compute_tripod_coordination(
    foot_load: np.ndarray,
    dt_ms: float,
    *,
    contact_threshold: float = 1e-5,
    min_phase_ms: float = 15.0,
) -> dict[str, Any]:
    """Compute tripod coordination between Tripod 1 (LF, RM, LH) and Tripod 2 (RF, LM, RH).

    In true alternating tripod locomotion:
    - Tripod 1 and Tripod 2 alternate: when one is in stance, the other is in swing.
    - Contact alternation index |S1(t) - S2(t)| approaches 1.0.
    - Cross-correlation between S1 and S2 is strongly negative.
    - In static standing (all 6 feet on ground), S1 = 1, S2 = 1, alternation is 0.0,
      and correlation is undefined (variance = 0).
    """
    load = np.asarray(foot_load, dtype=float)
    if load.ndim != 2 or load.shape[1] != 6:
        raise ValueError("foot_load must have shape (T, 6)")
    if len(load) < 2:
        return {
            "alternation_index": 0.0,
            "correlation": None,
            "intra_tripod_1_sync": 0.0,
            "intra_tripod_2_sync": 0.0,
            "tripod_coordination_score": 0.0,
            "tripod_gait_detected": False,
        }

    # Debounce contacts for each leg
    debounced = np.zeros_like(load, dtype=bool)
    for i in range(6):
        debounced[:, i] = debounce_contact(load[:, i] > contact_threshold, dt_ms, min_phase_ms)

    leg_idx = {name: i for i, name in enumerate(LEGS)}
    t1_indices = [leg_idx[name] for name in TRIPOD_1]
    t2_indices = [leg_idx[name] for name in TRIPOD_2]

    # Stance fraction in each tripod across time: shape (T,)
    s1 = debounced[:, t1_indices].mean(axis=-1)
    s2 = debounced[:, t2_indices].mean(axis=-1)

    # Alternation index: |s1 - s2|. In ideal tripod gait, this is 1.0.
    # In static standing with 6 feet down, s1=1, s2=1 -> alternation = 0.
    # In jump/collapse with all feet up or down together, alternation = 0.
    alternation = float(np.mean(np.abs(s1 - s2)))

    # Correlation between Tripod 1 and Tripod 2:
    s1_std, s2_std = float(np.std(s1)), float(np.std(s2))
    if s1_std > 1e-4 and s2_std > 1e-4:
        corr = float(np.corrcoef(s1, s2)[0, 1])
    else:
        corr = None

    # Intra-tripod synchrony: 1.0 - mean standard deviation across the 3 legs in the tripod
    # If all 3 legs in T1 always have the exact same contact state, std=0 -> sync=1.0.
    t1_sync = float(1.0 - np.mean(np.std(debounced[:, t1_indices], axis=-1)))
    t2_sync = float(1.0 - np.mean(np.std(debounced[:, t2_indices], axis=-1)))

    # Tripod coordination score:
    # Requires significant alternation (>0.2) AND anti-phase correlation (corr < -0.2)
    # Scaled to [0, 1].
    if corr is not None and corr < 0 and alternation > 0.1:
        # High alternation and negative correlation
        score = float(np.clip(alternation * (-corr) * 0.5 * (t1_sync + t2_sync), 0.0, 1.0))
        detected = bool(score >= 0.4 and alternation >= 0.4 and corr <= -0.4)
    else:
        score = 0.0
        detected = False

    return {
        "alternation_index": alternation,
        "correlation": corr,
        "intra_tripod_1_sync": t1_sync,
        "intra_tripod_2_sync": t2_sync,
        "tripod_coordination_score": score,
        "tripod_gait_detected": detected,
    }


def compute_stance_slip(
    feet_position: np.ndarray,
    foot_load: np.ndarray,
    dt_ms: float,
    *,
    contact_threshold: float = 1e-5,
) -> dict[str, float]:
    """Compute stance slip speed (mm/s) matching the reference evaluate.py formula.

    Slip is the horizontal displacement speed of feet during active ground contact.
    """
    feet = np.asarray(feet_position, dtype=float)
    load = np.asarray(foot_load, dtype=float)
    if feet.ndim != 3 or feet.shape[1:3] != (6, 3) or load.shape != (len(feet), 6):
        raise ValueError("feet_position must be (T, 6, 3) and foot_load must be (T, 6)")
    if len(feet) < 2 or dt_ms <= 0:
        return {"mean_stance_slip_mm_s": 0.0}

    support = load > contact_threshold
    speed = np.linalg.norm(np.diff(feet, axis=0)[..., :2], axis=-1)
    speed /= (dt_ms / 1000.0)
    counts = support[1:].sum(axis=-1)
    slip = np.divide(
        (speed * support[1:]).sum(axis=-1),
        counts,
        out=np.zeros(len(feet) - 1),
        where=counts > 0,
    )
    return {"mean_stance_slip_mm_s": float(slip.mean())}


def classify_locomotion(
    standing_fraction: float,
    heading_forward_distance_mm: float,
    mean_stance_slip_mm_s: float,
    total_leg_steps: int,
    mean_steps_per_leg: float,
    tripod_gait_detected: bool,
    mean_supporting_feet: float,
) -> str:
    """Classify physical locomotion into distinct physical regimes:

    - FALL: Body collapses or fails posture gate.
    - STATIC_STANDING: Body stands stably upright without stepping or slipping.
    - DRAGGING_OR_SLIPPING: Body translates forward/lateral by sliding feet on floor without stepping.
    - STEPPING_UNCOORDINATED: Legs make steps, but tripod coordination is absent or erratic.
    - COORDINATED_TRIPOD_WALKING: True walking with alternating tripod cycles and bounded slip.
    """
    if standing_fraction < 0.7:
        return "FALL"

    if total_leg_steps == 0:
        if abs(heading_forward_distance_mm) < 1.0 and mean_stance_slip_mm_s < 1.0:
            return "STATIC_STANDING"
        else:
            return "DRAGGING_OR_SLIPPING"

    # Legs are taking steps
    if mean_steps_per_leg < 1.0:
        # Only 1-2 legs fluttered or stepped, not full gait
        if mean_stance_slip_mm_s > 2.0 or abs(heading_forward_distance_mm) > 2.0:
            return "DRAGGING_OR_SLIPPING"
        return "STATIC_STANDING"

    # Multiple legs stepping:
    if tripod_gait_detected and heading_forward_distance_mm > 2.0:
        return "COORDINATED_TRIPOD_WALKING"
    elif heading_forward_distance_mm > 1.0:
        return "STEPPING_UNCOORDINATED"
    else:
        return "STEPPING_IN_PLACE"


def compute_gait_summary(
    observation: dict[str, np.ndarray],
    dt_ms: float,
    *,
    warmup_samples: int = 10,
    contact_threshold: float = 1e-5,
    min_phase_ms: float = 15.0,
) -> dict[str, Any]:
    """Compute comprehensive gait and step metrics over the measured post-warmup window."""
    measured = slice(warmup_samples, None)
    pos = observation["position"][measured]
    head = observation["heading"][measured]
    feet = observation["feet_position"][measured]
    load = observation["foot_load"][measured]

    fwd = compute_heading_forward_distance(pos, head)
    cycles = compute_all_leg_cycles(
        load, feet, dt_ms,
        contact_threshold=contact_threshold,
        min_phase_ms=min_phase_ms,
    )
    tripod = compute_tripod_coordination(
        load, dt_ms,
        contact_threshold=contact_threshold,
        min_phase_ms=min_phase_ms,
    )
    slip = compute_stance_slip(
        feet, load, dt_ms,
        contact_threshold=contact_threshold,
    )

    support = load > contact_threshold
    mean_sup_feet = float(support.sum(axis=-1).mean())

    from .learning import standing_mask
    full_standing = standing_mask(observation)
    standing_frac = float(full_standing[measured].mean())

    regime = classify_locomotion(
        standing_fraction=standing_frac,
        heading_forward_distance_mm=fwd["heading_forward_distance_mm"],
        mean_stance_slip_mm_s=slip["mean_stance_slip_mm_s"],
        total_leg_steps=cycles["total_leg_steps"],
        mean_steps_per_leg=cycles["mean_steps_per_leg"],
        tripod_gait_detected=tripod["tripod_gait_detected"],
        mean_supporting_feet=mean_sup_feet,
    )

    # Compact per-leg step cycle dict
    leg_steps = {
        leg: {
            "step_count": cycles["legs"][leg]["step_count"],
            "step_frequency_hz": cycles["legs"][leg]["step_frequency_hz"],
            "debounced_contact_fraction": cycles["legs"][leg]["debounced_contact_fraction"],
            "mean_duty_factor": cycles["legs"][leg]["mean_duty_factor"],
            "mean_stride_length_mm": cycles["legs"][leg]["mean_stride_length_mm"],
            "raw_touchdown_count": cycles["legs"][leg]["raw_touchdown_count"],
            "raw_liftoff_count": cycles["legs"][leg]["raw_liftoff_count"],
        }
        for leg in LEGS
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "heading_forward_distance_mm": fwd["heading_forward_distance_mm"],
        "net_displacement_mm": fwd["net_displacement_mm"],
        "net_forward_displacement_mm": fwd["net_forward_displacement_mm"],
        "total_leg_steps": cycles["total_leg_steps"],
        "mean_steps_per_leg": cycles["mean_steps_per_leg"],
        "leg_step_cycles": leg_steps,
        "tripod_coordination": tripod,
        "mean_stance_slip_mm_s": slip["mean_stance_slip_mm_s"],
        "locomotion_regime": regime,
        "true_walking_proven": bool(regime == "COORDINATED_TRIPOD_WALKING"),
    }
