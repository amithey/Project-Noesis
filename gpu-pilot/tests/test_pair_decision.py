import pytest

from noesis_gpu.pair_decision import classify_pair

PROBES = ("10006", "10007", "10010", "10011")


def rows(heading, standing=.95, slip=2.):
    return {p: dict(standing_fraction=standing, mean_stance_slip_mm_s=slip, heading_forward_distance_mm=heading,
                    net_forward_displacement_mm=heading) for p in PROBES}


def test_separate_counts_cannot_hide_a_two_probe_conjunction():
    real, control = rows(1.3), rows(.3)
    real["10006"]["mean_stance_slip_mm_s"] = 2.1  # slip regresses on probe 1
    real["10011"]["standing_fraction"] = .94  # standing regresses on probe 4
    out = classify_pair(real, control)
    assert out["real_noninferior_probes"] == 2 and out["verdict"] == "INDISTINGUISHABLE"


def test_ties_are_noninferior_in_both_directions():
    out = classify_pair(rows(.3), rows(1.3))  # control is better in gain; every metric ties
    assert out["control_noninferior_probes"] == 4 and out["real_noninferior_probes"] == 4
    assert out["verdict"] == "CONTROL_BETTER"


def test_real_gain_with_all_probes_retained_and_the_gain_threshold_edge():
    assert classify_pair(rows(1.3), rows(.3))["verdict"] == "REAL_BETTER"
    assert classify_pair(rows(.549), rows(.3))["verdict"] == "INDISTINGUISHABLE"  # gain 0.249 < 0.25
    assert classify_pair(rows(.55), rows(.3))["verdict"] == "REAL_BETTER"


def test_gain_without_retention_is_not_a_win_and_vice_versa():
    real, control = rows(1.3, standing=.80), rows(.3)
    assert classify_pair(real, control)["verdict"] == "INDISTINGUISHABLE"
    assert classify_pair(control, real)["verdict"] == "INDISTINGUISHABLE"  # control gains nothing to retain: real worse in gain too


def test_probe_sets_must_match():
    with pytest.raises(ValueError):
        classify_pair(rows(1.), {p: v for p, v in rows(1.).items() if p != "10006"})
