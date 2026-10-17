import pytest

from noesis.experiments.motor_g_inh import reflex_gain, standing_state
from noesis.sensory import LEGS


def test_reflex_gain_sign_and_zero():
    assert reflex_gain(10.0, 2.0, 4.0, 6.0) == pytest.approx(10.0)
    assert reflex_gain(5.0, 5.0, 5.0, 5.0) == 0.0


def test_standing_state_only_flexes_the_left_front_leg():
    s = standing_state(tibia_lf=0.6)
    assert set(s) == set(LEGS)
    assert s["LF"].tibia == 0.6 and all(s[leg].tibia == 0.0 for leg in LEGS if leg != "LF")
    assert all(s[leg].load == 1.0 for leg in LEGS)
