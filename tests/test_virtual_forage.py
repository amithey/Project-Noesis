import pytest

from noesis.experiments.virtual_forage import CONDITIONS, paired_diff, summarise


def rows():
    out = []
    for seed, (b, c) in enumerate([(5.0, 15.0), (8.0, 16.0), (20.0, 18.0)]):
        out.append({"condition": "brain_hungry", "seed": seed, "reached": b < 6, "final_dist_mm": b,
                    "min_dist_mm": b, "efficiency": 0.8})
        out.append({"condition": "diffusion_only", "seed": seed, "reached": False, "final_dist_mm": c,
                    "min_dist_mm": c, "efficiency": 0.3})
    return out


def test_summarise_counts_and_medians():
    s = summarise(rows())
    assert s["brain_hungry"]["reached"] == 1 and s["brain_hungry"]["median_final_mm"] == 8.0
    assert s["diffusion_only"]["median_efficiency"] == 0.3


def test_paired_diff_negative_means_closer():
    d = paired_diff(rows(), "brain_hungry", "diffusion_only")
    assert d["n"] == 3 and d["negative_in"] == 2
    assert d["mean"] == pytest.approx((-10 - 8 + 2) / 3)


def test_conditions_cover_the_planned_controls():
    assert {"shuffled_sides", "reversed_sign", "random_dn", "no_gradient", "diffusion_only"} <= set(CONDITIONS)
