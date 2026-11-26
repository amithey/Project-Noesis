from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from noesis_gpu.bridge import (
    DOFS,
    LEGACY_JOINTS,
    LEGS,
    Muscles,
    adhesion_from_command,
    joint_permutation,
    up_in_body,
)


def anatomical_dofs():
    segments = {"Coxa": "coxa", "Femur": "trochanterfemur",
                "Tibia": "tibia", "Tarsus1": "tarsus1"}
    for leg in LEGS:
        for dof in DOFS:
            joint, _, axis = dof.partition("_")
            yield SimpleNamespace(child=SimpleNamespace(pos=leg.lower(), link=segments[joint]),
                                  axis=SimpleNamespace(value=axis or "pitch"))


def test_anatomical_mapping_survives_native_reordering():
    dofs = list(anatomical_dofs())
    order = np.random.default_rng(6).permutation(42)
    permutation = joint_permutation([dofs[i] for i in order])
    np.testing.assert_array_equal(order[permutation], np.arange(42))
    commands = np.arange(42)
    native = commands[np.argsort(permutation)]
    np.testing.assert_array_equal(native[permutation], commands)


def test_mapping_rejects_missing_duplicate_and_unknown_channels():
    dofs = list(anatomical_dofs())
    for bad in (dofs[:-1], dofs[:-1] + [dofs[0]], dofs + [dofs[0]]):
        with pytest.raises(ValueError):
            joint_permutation(bad)
    dofs[0].axis.value = "unknown"
    with pytest.raises(ValueError):
        joint_permutation(dofs)


def test_muscle_equations_match_existing_body_over_time(monkeypatch):
    import noesis.body as legacy

    rng = np.random.default_rng(7)
    rest = rng.uniform(-0.2, 0.2, 42)
    lo, hi = rest - 0.7, rest + 0.7
    monkeypatch.setattr(legacy, "standing_pose", lambda: dict(zip(LEGACY_JOINTS, rest)))
    monkeypatch.setattr(legacy, "joint_limits", lambda _: dict(zip(LEGACY_JOINTS, zip(lo, hi))))
    gp = sparse.random(42, 15, density=0.2, random_state=8, format="csr")
    gn = sparse.random(42, 15, density=0.2, random_state=9, format="csr")
    params = legacy.BodyParams()
    original = legacy.MuscleModel(gp, gn, list(LEGACY_JOINTS), params)
    migrated = Muscles(gp, gn, rest, lo, hi, params)
    for _ in range(100):
        rates = rng.uniform(0, 200, 15)
        np.testing.assert_array_equal(original.step(rates, 5), migrated.step(rates, 5))


def test_adhesion_is_command_driven_and_has_no_leg_sequence():
    rest = np.zeros(42)
    np.testing.assert_array_equal(adhesion_from_command(rest, rest), np.ones(6))
    desired = rest.copy().reshape(6, 7)
    desired[[0, 4], 3] = 0.1
    np.testing.assert_array_equal(adhesion_from_command(desired.ravel(), rest), [0, 1, 1, 1, 0, 1])


def test_balance_frame_for_upright_tilt_and_inverted_body():
    q = np.array([[1, 0, 0, 0], [np.sqrt(0.5), np.sqrt(0.5), 0, 0], [0, 1, 0, 0]])
    np.testing.assert_allclose(up_in_body(q), [[0, 0, 1], [0, 1, 0], [0, 0, -1]], atol=1e-14)
    with pytest.raises(ValueError):
        up_in_body(np.zeros((1, 4)))
