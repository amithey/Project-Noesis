import pytest

from noesis.experiments.train_validate import training_exploration


def test_default_is_registered_amplitude():
    assert training_exploration("train-3", None) == 1.5
    assert training_exploration("train-0", []) == 1.5


def test_schedule_is_indexed_by_birth():
    schedule = [1.5, 1.0, 0.6, 0.3]
    assert [training_exploration(f"train-{i}", schedule) for i in range(4)] == schedule


def test_out_of_range_birth_raises():
    with pytest.raises(IndexError):
        training_exploration("train-5", [1.5, 1.0])
