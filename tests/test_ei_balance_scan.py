import numpy as np
import pytest
from scipy import sparse

from noesis.experiments.ei_balance_scan import response_label, scale_inhibition, widest_good_run


def test_scale_inhibition_only_touches_negative_entries_and_copies():
    W = sparse.csr_matrix(np.array([[0.0, 2.0, -3.0], [-1.0, 0.0, 4.0], [0.0, 0.0, 0.0]]))
    out = scale_inhibition(W, 2.0)
    assert out.toarray().tolist() == [[0.0, 2.0, -6.0], [-2.0, 0.0, 4.0], [0.0, 0.0, 0.0]]
    assert W.toarray()[0, 2] == -3.0
    with pytest.raises(ValueError):
        scale_inhibition(W, -1.0)
    with pytest.raises(ValueError):
        scale_inhibition(W, float("nan"))


def test_response_label_boundaries():
    assert response_label(0.5, 0.1) == "silent"
    assert response_label(10.0, 0.2) == "graded"
    assert response_label(10.0, 0.5) == "ignites"
    assert response_label(90.0, 0.2) == "ignites"


def test_widest_good_run_is_consecutive_in_background_order():
    pts = [{"bg_hz": h, "good": g} for h, g in [(3, True), (1, True), (2, False), (5, True), (8, True), (12, True)]]
    assert widest_good_run(pts) == 3
