"""Comprehensive unit tests for physical gait metrics using synthetic data.

Guards against:
- Claiming walking for static standing
- Claiming walking for dragging or sliding
- False steps from contact threshold chatter / micro-flutter
- False success during falls
- Uncoordinated or hopping motion masquerading as tripod gait
"""

import numpy as np
import pytest

from noesis_gpu.gait_metrics import (
    LEGS,
    compute_gait_summary,
    compute_heading_forward_distance,
    debounce_contact,
    extract_leg_step_cycles,
)


def test_compute_heading_forward_distance_straight():
    # Body moving at 10 mm/s forward along world +x
    t = 100
    pos = np.zeros((t, 3))
    pos[:, 0] = np.linspace(0, 10, t)  # 10 mm forward
    heading = np.tile([1.0, 0.0, 0.0], (t, 1))

    fwd = compute_heading_forward_distance(pos, heading)
    assert fwd["heading_forward_distance_mm"] == pytest.approx(10.0)
    assert fwd["net_displacement_mm"] == pytest.approx(10.0)
    assert fwd["net_forward_displacement_mm"] == pytest.approx(10.0)


def test_compute_heading_forward_distance_backward():
    # Body sliding backward
    t = 100
    pos = np.zeros((t, 3))
    pos[:, 0] = np.linspace(0, -5, t)
    heading = np.tile([1.0, 0.0, 0.0], (t, 1))

    fwd = compute_heading_forward_distance(pos, heading)
    assert fwd["heading_forward_distance_mm"] == pytest.approx(-5.0)
    assert fwd["net_displacement_mm"] == pytest.approx(5.0)
    assert fwd["net_forward_displacement_mm"] == pytest.approx(-5.0)


def test_compute_heading_forward_distance_turning():
    # Body heading turns 90 degrees while path follows heading
    # 2 steps: first step forward along +x, second step along +y with heading +y
    pos = np.array([
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],  # moved +1 in x with heading +x
        [1.0, 2.0, 1.0],  # moved +2 in y with heading +y
    ])
    heading = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    fwd = compute_heading_forward_distance(pos, heading)
    assert fwd["heading_forward_distance_mm"] == pytest.approx(3.0)


def test_compute_heading_forward_distance_guards():
    with pytest.raises(ValueError, match="matching shape"):
        compute_heading_forward_distance(np.zeros((5, 3)), np.zeros((4, 3)))
    with pytest.raises(ValueError, match="finite"):
        compute_heading_forward_distance(np.array([[np.nan, 0, 0], [0, 0, 0]]), np.ones((2, 3)))
    with pytest.raises(ValueError, match="zero magnitude"):
        compute_heading_forward_distance(np.zeros((2, 3)), np.zeros((2, 3)))


def test_debounce_contact_eliminates_chatter():
    dt_ms = 5.0
    # True contact with 1-sample dropouts (5ms lift, which is chatter)
    contact = np.ones(50, dtype=bool)
    contact[10] = False  # single sample dropout
    contact[20] = False  # single sample dropout

    debounced = debounce_contact(contact, dt_ms, min_phase_ms=15.0)
    # The 5ms dropouts should be debounced to True
    assert debounced.all()

    # Ground with single-sample touches (5ms spikes)
    floor = np.zeros(50, dtype=bool)
    floor[15] = True  # single sample spike
    debounced_floor = debounce_contact(floor, dt_ms, min_phase_ms=15.0)
    assert not debounced_floor.any()

    # Alternating contact chatter (T, F, T, F, T) within sustained stance
    # Should not artificially turn into a multi-sample false swing
    chatter = np.ones(50, dtype=bool)
    chatter[15] = False
    chatter[17] = False
    debounced_chatter = debounce_contact(chatter, dt_ms, min_phase_ms=15.0)
    assert debounced_chatter.all()


def test_synthetic_static_standing():
    """Static standing: 6 feet firmly on ground, zero motion, zero steps."""
    t = 200
    dt_ms = 5.0
    load = np.full((t, 6), 2.0)  # constant 2 mN load on all feet
    feet = np.zeros((t, 6, 3))
    pos = np.zeros((t, 3))
    pos[:, 2] = 1.0  # height 1 mm
    head = np.tile([1.0, 0.0, 0.0], (t, 1))
    up = np.tile([0.0, 0.0, 1.0], (t, 1))

    obs = {
        "position": pos,
        "heading": head,
        "feet_position": feet,
        "foot_load": load,
        "up_body": up,
    }

    summary = compute_gait_summary(obs, dt_ms, warmup_samples=10)
    assert summary["heading_forward_distance_mm"] == pytest.approx(0.0)
    assert summary["mean_stance_slip_mm_s"] == pytest.approx(0.0)
    assert summary["total_leg_steps"] == 0
    assert summary["tripod_coordination"]["tripod_gait_detected"] is False
    assert summary["locomotion_regime"] == "STATIC_STANDING"
    assert summary["diagnostic_tripod_walking"] is False


def test_synthetic_dragging_and_sliding():
    """Dragging/sliding: feet never leave ground, slide at 5 mm/s, zero steps.
    
    Must NOT be classified as walking despite forward motion!
    """
    t = 200
    dt_ms = 5.0
    load = np.full((t, 6), 2.0)  # all feet in continuous ground contact
    feet = np.zeros((t, 6, 3))
    # Feet and body sliding forward at 5 mm/s
    speed = 5.0  # mm/s
    dt_s = dt_ms / 1000.0
    for i in range(t):
        feet[i, :, 0] = i * dt_s * speed
    pos = np.zeros((t, 3))
    pos[:, 0] = np.arange(t) * dt_s * speed
    pos[:, 2] = 1.0
    head = np.tile([1.0, 0.0, 0.0], (t, 1))
    up = np.tile([0.0, 0.0, 1.0], (t, 1))

    obs = {
        "position": pos,
        "heading": head,
        "feet_position": feet,
        "foot_load": load,
        "up_body": up,
    }

    summary = compute_gait_summary(obs, dt_ms, warmup_samples=10)
    # Forward distance is positive
    assert summary["heading_forward_distance_mm"] > 4.0
    # But slip is high (~5 mm/s)
    assert summary["mean_stance_slip_mm_s"] == pytest.approx(5.0, rel=1e-3)
    # Zero legitimate step cycles (feet never lifted)
    assert summary["total_leg_steps"] == 0
    assert summary["tripod_coordination"]["tripod_gait_detected"] is False
    assert summary["locomotion_regime"] == "DRAGGING_OR_SLIPPING"
    assert summary["diagnostic_tripod_walking"] is False


def test_synthetic_contact_chatter_threshold():
    """Contact chattering around threshold (0, 2e-5, 0, 2e-5...).
    
    Debouncing must prevent registering spurious step cycles.
    """
    t = 200
    dt_ms = 5.0
    load = np.zeros((t, 6))
    # Leg 0 chatters every sample
    load[::2, 0] = 2e-5
    feet = np.zeros((t, 6, 3))

    res = extract_leg_step_cycles(load[:, 0], feet[:, 0], dt_ms, contact_threshold=1e-5, min_phase_ms=15.0)
    assert res["step_count"] == 0
    assert res["raw_touchdown_count"] > 40  # Raw signal had dozens of transitions


def test_synthetic_fall():
    """Falling body: height collapses or body tilts, fails standing gate."""
    t = 100
    dt_ms = 5.0
    pos = np.zeros((t, 3))
    pos[:, 2] = 0.1  # collapsed height (0.1 mm < 0.3 mm)
    up = np.tile([1.0, 0.0, 0.0], (t, 1))  # rolled onto side (up_z = 0.0 < 0.7)
    head = np.tile([1.0, 0.0, 0.0], (t, 1))
    load = np.zeros((t, 6))
    feet = np.zeros((t, 6, 3))

    obs = {
        "position": pos,
        "heading": head,
        "feet_position": feet,
        "foot_load": load,
        "up_body": up,
    }

    summary = compute_gait_summary(obs, dt_ms, warmup_samples=10)
    assert summary["locomotion_regime"] == "FALL"
    assert summary["diagnostic_tripod_walking"] is False


def test_synthetic_true_tripod_walking():
    """True alternating tripod walking:
    
    - 10 Hz stepping (100 ms period = 20 samples at 5ms dt)
    - Stance: 60 ms (12 samples), Swing: 40 ms (8 samples)
    - Tripod 1 (LF, RM, LH) alternates with Tripod 2 (RF, LM, RH)
    - In stance, foot is stationary on floor (zero slip)
    - In swing, foot moves forward
    - Body moves forward continuously at 10 mm/s
    """
    t = 300  # 1.5 seconds = 15 complete 100ms cycles
    dt_ms = 5.0
    dt_s = dt_ms / 1000.0
    period_samples = 20  # 100 ms
    stance_samples = 12  # 60 ms

    load = np.zeros((t, 6))
    feet = np.zeros((t, 6, 3))
    pos = np.zeros((t, 3))
    head = np.tile([1.0, 0.0, 0.0], (t, 1))
    up = np.tile([0.0, 0.0, 1.0], (t, 1))

    # Body moves forward at 10 mm/s
    speed = 10.0
    pos[:, 0] = np.arange(t) * dt_s * speed
    pos[:, 2] = 1.0

    t1_legs = [0, 4, 2]  # LF, RM, LH
    t2_legs = [3, 1, 5]  # RF, LM, RH

    for step in range(t):
        phase_t1 = step % period_samples
        # T1 is in stance for first 12 samples of cycle, swing for next 8
        t1_stance = phase_t1 < stance_samples
        # T2 is shifted by half period (10 samples)
        phase_t2 = (step + 10) % period_samples
        t2_stance = phase_t2 < stance_samples

        for leg in t1_legs:
            load[step, leg] = 2.0 if t1_stance else 0.0
            # During stance, foot x remains fixed at its touchdown position
            td_step = (step // period_samples) * period_samples
            if t1_stance:
                feet[step, leg, 0] = td_step * dt_s * speed
            else:
                # During swing, foot progresses forward
                feet[step, leg, 0] = step * dt_s * speed

        for leg in t2_legs:
            load[step, leg] = 2.0 if t2_stance else 0.0
            td_step = ((step + 10) // period_samples) * period_samples - 10
            if t2_stance:
                feet[step, leg, 0] = td_step * dt_s * speed
            else:
                feet[step, leg, 0] = step * dt_s * speed

    # Ensure touchdown transition does not introduce single-frame velocity artifact:
    # at the step where stance begins, the previous frame foot position was already at touchdown
    for step in range(1, t):
        for leg in range(6):
            if load[step, leg] > 0 and load[step - 1, leg] == 0:
                # Touchdown event: foot arrived at contact location
                feet[step - 1, leg, 0] = feet[step, leg, 0]

    obs = {
        "position": pos,
        "heading": head,
        "feet_position": feet,
        "foot_load": load,
        "up_body": up,
    }

    summary = compute_gait_summary(obs, dt_ms, warmup_samples=10)

    # 1. Forward distance is rewarded
    assert summary["heading_forward_distance_mm"] > 10.0
    # 2. Stance slip is virtually 0 because foot was stationary during contact
    assert summary["mean_stance_slip_mm_s"] < 0.1
    # 3. All 6 legs completed multiple step cycles
    assert summary["total_leg_steps"] >= 60  # ~14 cycles * 6 legs
    assert summary["mean_steps_per_leg"] >= 10
    for leg in LEGS:
        assert summary["leg_step_cycles"][leg]["step_count"] >= 10
        assert summary["leg_step_cycles"][leg]["mean_duty_factor"] == pytest.approx(0.6, abs=0.05)

    # 4. Tripod coordination detected
    tripod = summary["tripod_coordination"]
    assert tripod["tripod_gait_detected"] is True
    assert tripod["alternation_index"] > 0.8
    assert tripod["correlation"] < -0.5
    assert tripod["tripod_coordination_score"] > 0.4

    # 5. Regime classified as COORDINATED_TRIPOD_WALKING
    assert summary["locomotion_regime"] == "COORDINATED_TRIPOD_WALKING"
    assert summary["diagnostic_tripod_walking"] is True


def test_pure_vertical_with_pitch_heading_does_not_count_as_forward():
    """Regression test (PR #22 review finding):
    Pure vertical motion (dz = 1 mm) with a pitched heading [0.6, 0, 0.8]
    must yield zero horizontal forward progress and zero net forward displacement.
    """
    pos = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    heading = np.array([
        [0.6, 0.0, 0.8],
        [0.6, 0.0, 0.8],
    ])
    fwd = compute_heading_forward_distance(pos, heading)
    assert fwd["heading_forward_distance_mm"] == pytest.approx(0.0)
    assert fwd["net_displacement_mm"] == pytest.approx(0.0)
    assert fwd["net_forward_displacement_mm"] == pytest.approx(0.0)


def test_degenerate_horizontal_heading_handling():
    """When the body points purely vertically (norm_xy < 1e-6), horizontal heading
    is undefined and displacements must safely yield zero forward progress without crash.
    """
    pos = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ])
    # Heading points purely along +z (no horizontal component)
    heading = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ])
    fwd = compute_heading_forward_distance(pos, heading)
    assert fwd["heading_forward_distance_mm"] == pytest.approx(0.0)
    assert fwd["net_displacement_mm"] == pytest.approx(1.0)
    assert fwd["net_forward_displacement_mm"] == pytest.approx(0.0)


def test_extract_leg_step_counts_adapter():
    """Integration contract test:
    Verify extract_leg_step_counts extracts a flat leg -> int mapping for core phase_decision.
    """
    from noesis_gpu.gait_metrics import extract_leg_step_counts

    mock_cycles = {
        "LF": {"step_count": 15, "duty_factor": 0.6},
        "LM": {"step_count": 12, "duty_factor": 0.5},
        "LH": {"step_count": 18, "duty_factor": 0.7},
        "RF": {"step_count": 14, "duty_factor": 0.6},
        "RM": {"step_count": 16, "duty_factor": 0.55},
        "RH": {"step_count": 11, "duty_factor": 0.65},
    }
    counts = extract_leg_step_counts({"leg_step_cycles": mock_cycles})
    assert counts == {"LF": 15, "LM": 12, "LH": 18, "RF": 14, "RM": 16, "RH": 11}
    for leg in LEGS:
        assert isinstance(counts[leg], int)

