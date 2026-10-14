"""Layout, spike brightness, and the activity file that feeds the 3D viewer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from noesis.viz.brain_map import (
    REGION_XY,
    read_activity_frame,
    read_activity_full,
    spike_histogram,
    write_activity,
)
from noesis.viz.brain_view3d import parse_window_geometry
from noesis.viz.layout3d import (
    BACKBONE_EDGES,
    FLASH_TAU_S,
    LIVE_EDGES,
    NEUROPILS,
    Glow,
    backbone_from_csr,
    base_colors,
    format_hud,
    glowing_index,
    live_synapse_edges,
    neuron_positions_3d,
    neuropil_center,
    point_sizes,
    quiet_display_index,
    rgba_from_glow,
    shell_wireframe,
)


def _census(per_side: int = 24) -> pd.DataFrame:
    rows = []
    for name in NEUROPILS:
        for side in ("left", "right"):
            rows.extend((name, side, "ACH") for _ in range(per_side))
    rows.extend(("ME.LO", "right", "GABA") for _ in range(60))
    rows.append(("NOT_A_REGION", "left", ""))
    return pd.DataFrame(rows, columns=["Top in/out region", "Soma side", "Predicted NT type"])


def _mask(neurons: pd.DataFrame, region: str, side: str) -> np.ndarray:
    return (neurons["Top in/out region"].eq(region) & neurons["Soma side"].eq(side)).to_numpy()


def test_layout_covers_every_schematic_region_and_stays_in_volume():
    assert set(REGION_XY) <= set(NEUROPILS)
    neurons = _census()
    pos = neuron_positions_3d(neurons, seed=0)
    assert pos.shape == (len(neurons), 3)
    assert np.isfinite(pos).all()
    # FAFB nanometre coordinates sit far outside this layout.
    assert np.abs(pos).max() < 2000

    again = neuron_positions_3d(neurons, seed=0)
    other = neuron_positions_3d(neurons, seed=1)
    assert np.allclose(pos, again)
    assert not np.allclose(pos, other)

    def mean(region, side):
        return pos[_mask(neurons, region, side)].mean(axis=0)

    # Brain above the neck, nerve cord below, segments in anatomical order.
    assert mean("ME", "right")[1] > mean("GNG", "right")[1] > mean("T1_PRONM", "right")[1]
    assert mean("T1_PRONM", "right")[1] > mean("T2_MESONM", "right")[1]
    assert mean("T2_MESONM", "right")[1] > mean("T3_METANM", "right")[1]
    assert mean("T3_METANM", "right")[1] > mean("ABDNM", "right")[1]
    # Antennal lobe is anterior of the mushroom-body calyx; optic lobe is more lateral.
    assert mean("AL", "right")[2] > mean("MB_CA", "right")[2] + 40
    assert abs(mean("ME", "right")[0]) > abs(mean("AL", "right")[0]) + 40
    assert mean("ME", "left")[0] < 0 < mean("ME", "right")[0]
    assert abs(pos[neurons["Top in/out region"].eq("PB").to_numpy(), 0].mean()) < 15
    # Unconnected fragments sit outside the optic lobe, not inside a fake neuropil.
    assert abs(mean("NO_CONS", "right")[0]) > abs(mean("ME", "right")[0])

    right_me = pos[_mask(neurons, "ME", "right")]
    radii = np.array(NEUROPILS["ME"].radii)
    scaled = (right_me - neuropil_center("ME", "right")) / radii
    assert np.sqrt((scaled ** 2).sum(axis=1)).max() <= 1.02
    between = pos[_mask(neurons, "ME.LO", "right")][:, 0].mean()
    assert neuropil_center("LO", "right")[0] < between < neuropil_center("ME", "right")[0]


def test_anchor_order_is_brain_above_cord_and_left_mirrors_right():
    assert neuropil_center("ME", "right")[1] > neuropil_center("GNG", "right")[1]
    assert neuropil_center("GNG", "right")[1] > neuropil_center("T1_PRONM", "right")[1]
    assert neuropil_center("T1_PRONM", "right")[1] > neuropil_center("T2_MESONM", "right")[1]
    assert neuropil_center("T2_MESONM", "right")[1] > neuropil_center("T3_METANM", "right")[1]
    assert neuropil_center("T3_METANM", "right")[1] > neuropil_center("ABDNM", "right")[1]
    assert neuropil_center("AL", "right")[2] > neuropil_center("MB_CA", "right")[2]
    assert abs(neuropil_center("ME", "right")[0]) > abs(neuropil_center("AL", "right")[0])
    assert neuropil_center("ME", "left")[0] < 0 < neuropil_center("ME", "right")[0]
    assert abs(neuropil_center("PB", "right")[0]) < abs(neuropil_center("ME", "right")[0]) / 3


def test_colors_follow_transmitter_or_region_and_spikes_light_up():
    neurons = pd.DataFrame({
        "Top in/out region": ["ME", "AL", "GNG"],
        "Soma side": ["right", "right", "left"],
        "Predicted NT type": ["ACH", "", "GABA"],
    })
    rgb = base_colors(neurons)
    assert rgb[0, 2] > rgb[0, 0]  # acetylcholine is cyan
    assert rgb[1, 0] > rgb[1, 2]  # missing transmitter uses the antennal-lobe color
    assert not np.allclose(rgb[0], rgb[2])

    glow = Glow(3)
    shown = glow.on_frame(np.array([0.0, 0.0, 0.0]), np.array([1, 0, 0]), 0.0)
    assert shown[0] == pytest.approx(1.0)
    assert shown[1] == pytest.approx(0.0)
    quiet = rgba_from_glow(rgb, shown)[1]
    hot = rgba_from_glow(rgb, shown)[0]
    assert quiet[:3].max() > 0.1  # dim, not invisible
    assert quiet[3] > 0.2
    assert hot[:3].max() > quiet[:3].max() * 3

    glow.on_tick(0.1)
    assert glow.flash[0] == pytest.approx(np.exp(-0.1 / FLASH_TAU_S))
    assert glow.shown()[0] > 0.7
    glow.on_tick(3.0)
    assert glow.shown()[0] < 0.05

    rated = Glow(1)
    plateau = rated.on_frame(np.array([50.0]), np.array([0]), 0.0)
    assert 0.4 < plateau[0] < 0.9


def test_activity_file_feeds_spike_brightness(tmp_path):
    rates = np.array([0.0, 4.0, 50.0], np.float32)
    spikes = np.array([0, 1, 0], np.uint16)
    path = tmp_path / "live.npz"
    triggers = {"odor_food": 0.4, "vision_mean": 0.2, "hemolymph_sugar": 0.5, "body_z_mm": 1.1}
    assert write_activity(
        path, rates, 1.5, awake=True, note="BANC corrected", triggers=triggers, spikes=spikes
    )
    frame = read_activity_frame(path)
    assert frame.time_s == pytest.approx(1.5)
    assert frame.awake is True
    assert frame.note == "BANC corrected"
    assert frame.triggers["odor_food"] == pytest.approx(0.4)
    np.testing.assert_array_equal(frame.spikes, spikes)
    # The 2D reader still returns the original 5-tuple when spike counts are present.
    rates_out, time_s, awake, note, trig = read_activity_full(path)
    assert awake is True and note == "BANC corrected"
    assert time_s == pytest.approx(1.5)
    assert trig["hemolymph_sugar"] == pytest.approx(0.5)
    assert np.allclose(rates_out, rates, rtol=1e-2)

    glow = Glow(3)
    shown = glow.on_frame(frame.rates, frame.spikes, 0.0)
    assert shown[1] > 0.9  # the spike, not the smoothed rate, is what flashes
    assert shown[0] < 0.05
    hud = format_hud(
        frame.time_s, 3, int(frame.spikes.sum()), int((shown > 0.45).sum()),
        frame.awake, frame.note, frame.triggers,
    )
    assert "neuropil layout" in hud
    assert "spiked 1" in hud
    assert "world  |" in hud
    assert "inside |" in hud
    assert "body   |" in hud
    assert spike_histogram([np.array([1, 1, 2]), np.array([2])], 4).tolist() == [0, 2, 2, 0]


def test_edges_are_capped_and_live_edges_stay_inside_the_active_set():
    rows = np.array([0, 0, 1, 3, 3, 0])
    cols = np.array([1, 2, 2, 4, 5, 3])
    data = np.array([10.0, 9.0, 8.0, 10.0, 7.0, 12.0])
    weights = sparse.csr_matrix((data, (rows, cols)), shape=(6, 6))
    region = np.array([0, 0, 0, 1, 1, 1])
    pre, post = backbone_from_csr(
        weights.indptr, weights.indices, weights.data, region, max_edges=4, seed=0
    )
    assert 1 <= len(pre) <= 4
    assert np.all(pre != post)
    same = region[pre] == region[post]
    assert same.any() and (~same).any()

    pre, post = live_synapse_edges(
        weights.indptr, weights.indices, weights.data, np.array([0, 1]), max_edges=10
    )
    assert set(zip(pre.tolist(), post.tolist())) == {(0, 1)}

    rng = np.random.default_rng(0)
    src = rng.integers(0, 12, 80)
    dst = rng.integers(0, 12, 80)
    dense = sparse.csr_matrix((np.ones(80), (src, dst)), shape=(12, 12))
    pre, post = live_synapse_edges(
        dense.indptr, dense.indices, dense.data, np.arange(12), max_edges=3
    )
    assert len(pre) <= 3
    assert np.isin(pre, np.arange(12)).all()
    assert np.isin(post, np.arange(12)).all()


def test_long_chords_drop_spikes_outsize_quiet_dots_and_shells_span_the_cns():
    positions = np.array([[0, 0, 0], [12, 0, 0], [390, 0, 0], [400, 0, 0]], np.float32)
    weights = sparse.csr_matrix(
        (np.array([5.0, 50.0, 4.0]), (np.array([0, 1, 2]), np.array([1, 3, 3]))),
        shape=(4, 4),
    )
    region = np.array([0, 0, 1, 1])
    pre, post = backbone_from_csr(
        weights.indptr, weights.indices, weights.data, region,
        max_edges=4, seed=0, positions=positions, max_length=52.0, bridge_length=80.0,
    )
    pairs = set(zip(pre.tolist(), post.tolist(), strict=True))
    assert (0, 1) in pairs
    assert (1, 3) not in pairs

    pre, post = live_synapse_edges(
        weights.indptr, weights.indices, weights.data, np.array([0, 1, 2, 3]),
        max_edges=4, positions=positions, max_length=52.0, max_long=0, long_length=80.0,
    )
    assert set(zip(pre.tolist(), post.tolist(), strict=True)) <= {(0, 1), (2, 3)}

    assert point_sizes(np.array([1.0]))[0] >= point_sizes(np.array([0.0]))[0] * 4

    seg, col = shell_wireframe()
    assert np.isfinite(seg).all() and np.isfinite(col).all()
    assert seg[:, 0].min() < -200 and seg[:, 0].max() > 200
    assert seg[:, 1].max() > 80 and seg[:, 1].min() < -280
    assert col[:, 3].max() < 0.8


def test_window_geometry_and_launchers_default_to_the_3d_view():
    assert parse_window_geometry("760x920+940+30") == ((760, 920), (940, 30))
    size, origin = parse_window_geometry(None)
    assert origin is None and size[0] > 0 and size[1] > 0
    root = Path(__file__).resolve().parents[1]
    run = (root / "Run-Fly.ps1").read_text(encoding="utf-8")
    assert "noesis.viz.brain_view3d" in run
    assert "Flat2D" in run
    muscles = (root / "src" / "noesis" / "experiments" / "newborn_muscles.py").read_text(encoding="utf-8")
    assert "spike_histogram(pending_spikes, cn.n)" in muscles
    assert "write_activity(" in muscles


def test_quiet_cloud_is_capped_and_every_spike_stays_drawable():
    neurons = _census(per_side=30)
    names = neurons["Top in/out region"].to_numpy()
    idx = quiet_display_index(names, cap=2000)
    assert len(idx) <= 2000
    drawn = set(names[idx])
    assert set(names) <= drawn
    again = quiet_display_index(names, cap=2000)
    np.testing.assert_array_equal(idx, again)
    # The live defaults stay small enough that the window is not an edge hairball.
    assert BACKBONE_EDGES <= 400
    assert LIVE_EDGES <= 120
    glow = np.zeros(8, np.float32)
    glow[1] = 1.0
    glow[3] = 0.2
    glow[5] = 0.01
    hot = glowing_index(glow)
    assert set(hot.tolist()) == {1, 3}
