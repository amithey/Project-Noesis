import pytest

from noesis.experiments.mb_foraging_conditioning import learning_index, paired_effect


def test_learning_index_negative_when_paired_odour_weakens():
    before, after = {"A": 10.0, "B": 10.0}, {"A": 6.0, "B": 9.0}
    assert learning_index(before, after) == pytest.approx(-0.3)


def test_learning_index_none_without_baseline_response():
    assert learning_index({"A": 0.0, "B": 3.0}, {"A": 0.0, "B": 3.0}) is None


def test_paired_effect_drops_missing_seeds():
    e = paired_effect([-0.4, None, -0.2], [0.0, 0.1, 0.1])
    assert e["n"] == 2 and e["mean"] == pytest.approx(-0.35) and e["negative_in"] == 2


def test_paired_effect_empty():
    assert paired_effect([None], [0.1])["n"] == 0
