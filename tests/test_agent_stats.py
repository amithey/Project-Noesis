import pytest

from noesis.experiments.agent_stats import analyse, fisher_greater, holm, sign_test_less


def test_fisher_known_values():
    assert fisher_greater(4, 4, 0, 4) == pytest.approx(1 / 70)
    assert fisher_greater(2, 4, 0, 4) == pytest.approx(0.2142857, rel=1e-4)
    assert fisher_greater(2, 4, 1, 4) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        fisher_greater(5, 4, 0, 4)


def test_sign_test():
    assert sign_test_less([-1, -2, -3, -4]) == pytest.approx(1 / 16)
    assert sign_test_less([1, 2, -1]) == pytest.approx(7 / 8)
    assert sign_test_less([0, 0]) == 1.0


def test_holm_monotone_and_capped():
    out = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert out["a"] == pytest.approx(0.03) and out["c"] == pytest.approx(0.06) and out["b"] == pytest.approx(0.06)
    assert all(v <= 1 for v in out.values())


def test_analyse_counts_and_pairing():
    rows = []
    for s in range(4):
        rows.append({"condition": "decoder_hungry", "seed": s, "reached": True, "final_dist_mm": 3.0})
        rows.append({"condition": "diffusion_only", "seed": s, "reached": False, "final_dist_mm": 60.0})
    rep = analyse(rows)
    e = rep["conditions"]["diffusion_only"]
    assert e["reached"] == 0 and e["closer_in"] == 4 and e["fisher_p"] == pytest.approx(1 / 70)
    assert e["fisher_p_holm"] == pytest.approx(1 / 70)
