import math

import pytest

from noesis.drives import DriveParams, HomeostaticState


def test_decay_without_intake_gives_no_reward():
    s = HomeostaticState()
    r = s.step(10.0)
    assert s.energy == pytest.approx(0.9) and s.water == pytest.approx(0.85)
    assert r.reward == pytest.approx(0.0) and r.punishment == 0.0 and r.alive


def test_sugar_raises_energy_and_is_rewarding():
    s = HomeostaticState(energy=0.5)
    r = s.step(1.0, sugar=1.0)
    assert s.energy == pytest.approx(0.5 - 0.01 + 0.2)
    assert r.reward == pytest.approx(0.2)


def test_satiation_caps_reward():
    s = HomeostaticState(energy=1.0, water=1.0)
    r = s.step(1.0, sugar=5.0)
    assert s.energy == 1.0
    assert r.reward == pytest.approx(0.01)  # only the replaced decay counts


def test_death_when_energy_runs_out():
    s = HomeostaticState(DriveParams(energy_decay_per_s=0.5), energy=0.2)
    assert not s.step(1.0).alive and s.energy == 0.0


def test_hunger_raises_food_gain_and_lowers_bitter_gain():
    full, hungry = HomeostaticState(), HomeostaticState(energy=0.2)
    assert hungry.sensory_gain("fruity_volatile") > full.sensory_gain("fruity_volatile")
    assert hungry.sensory_gain("aversive_volatile") < full.sensory_gain("aversive_volatile")
    assert hungry.sensory_gain("plant_matter_volatile") == 1.0


def test_bitter_is_punishment_not_reward():
    s = HomeostaticState()
    r = s.step(2.0, bitter=1.5)
    assert r.punishment == pytest.approx(3.0) and r.reward == 0.0


@pytest.mark.parametrize("bad", [math.nan, math.inf, -1.0])
def test_rejects_invalid_inputs(bad):
    s = HomeostaticState()
    with pytest.raises(ValueError):
        s.step(bad)
    with pytest.raises(ValueError):
        s.step(1.0, sugar=bad)
    with pytest.raises(ValueError):
        DriveParams(hunger_gain=bad)
    with pytest.raises(ValueError):
        HomeostaticState(energy=1.5)
