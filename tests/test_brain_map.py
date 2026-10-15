import numpy as np
import pandas as pd
from scipy import sparse

from noesis.viz.brain_map import neuron_positions_3d, strongest_synapses


def test_brain_volume_renders_activity_with_installed_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    from noesis.viz.brain_map import BrainVolume

    volume = BrainVolume(np.array([[0., 0., 0.], [1., 1., 1.]]),
                         np.array([0]), np.array([1]), np.array([1.]))
    try:
        volume.update(np.array([15., 30.]), time_s=0.1)
        volume.fig.canvas.draw()
        assert "0.100" in volume.text.get_text()
    finally:
        import matplotlib.pyplot as plt
        plt.close(volume.fig)


def test_connections_remain_visible_when_silent_and_can_be_isolated():
    import matplotlib
    matplotlib.use("Agg")
    from noesis.viz.brain_map import BrainVolume
    from types import SimpleNamespace

    volume = BrainVolume(np.array([[0., 0., 0.], [1., 0., 0.], [2., 1., 0.]]),
                         np.array([0, 1]), np.array([1, 2]), np.array([1., -1.]))
    try:
        baseline = volume.wires.get_colors().copy()
        assert np.all((baseline[:, 3] > 0) & (baseline[:, 3] < .2))
        volume.update(np.array([80., 100., 0.]))
        np.testing.assert_array_equal(volume.wires.get_colors(), baseline)
        volume._pick(SimpleNamespace(artist=volume.node_dots, ind=[0]))
        assert volume.selected == 0
        assert volume.wires.get_colors()[0, 3] > .5
        assert volume.wires.get_colors()[1, 3] < .03
        assert "Displayed outgoing: 1" in volume.info.get_text()
        assert volume.focus_arrows is not None
        volume._key(SimpleNamespace(key="escape"))
        assert volume.selected is None
        assert volume.focus_arrows is None
        np.testing.assert_array_equal(volume.wires.get_colors(), baseline)
    finally:
        import matplotlib.pyplot as plt
        plt.close(volume.fig)


def test_firing_glow_tracks_activity_including_neurons_outside_displayed_edges():
    import matplotlib
    matplotlib.use("Agg")
    from noesis.viz.brain_map import BrainVolume

    volume = BrainVolume(np.array([[0., 0., 0.], [1., 0., 0.], [2., 1., 0.]]),
                         np.array([0]), np.array([1]), np.array([1.]))
    try:
        volume.update(np.array([0., 0., 40.]))
        assert len(volume.lit.get_sizes()) == 1
        assert volume.lit._offsets3d[0][0] == 2.
        low_size = volume.glow_outer.get_sizes()[0]
        volume.update(np.array([0., 0., 100.]))
        assert volume.glow_outer.get_sizes()[0] > low_size
        volume.update(np.zeros(3))
        assert len(volume.glow_outer.get_sizes()) == 0
        assert len(volume.glow_inner.get_sizes()) == 0
        assert len(volume.lit.get_sizes()) == 0
        volume.update(np.array([0., 0., 100.]), awake=False)
        assert len(volume.lit.get_sizes()) == 0
    finally:
        import matplotlib.pyplot as plt
        plt.close(volume.fig)


def test_schematic_layout_is_three_dimensional_and_sided():
    neurons = pd.DataFrame({
        "root_id": [1, 2, 3],
        "Top in/out region": ["ME", "AL", "GNG"],
        "Soma side": ["left", "right", "left"],
    })
    pos = neuron_positions_3d(neurons, seed=0)
    assert pos.shape == (3, 3)
    assert pos[0, 0] < 0
    assert pos[1, 0] > 0
    assert pos[2, 2] < pos[0, 2]


def test_compact_anatomy_separates_brain_and_nerve_cord():
    from noesis.viz.brain_map import compact_anatomy_positions
    neurons = pd.DataFrame({"Top in/out region": ["ME", "ME", "T1_PRONM", "ABDNM"],
                            "Soma side": ["left", "right", "left", "right"]})
    pos = compact_anatomy_positions(neurons)
    assert pos.shape == (4, 3)
    assert pos[0, 0] < -2 and pos[1, 0] > 2
    assert np.all(pos[:2, 2] > 0)
    assert np.all(pos[2:, 2] < -1)
    np.testing.assert_array_equal(pos, compact_anatomy_positions(neurons))


def test_strongest_synapses_keep_the_real_signed_edges():
    weights = sparse.csr_matrix(np.array([[0, 10, 1], [0, 0, -4], [0, 0, 0]], float))
    pre, post, sign = strongest_synapses(weights, limit=2)
    pairs = set(zip(pre.tolist(), post.tolist(), sign.tolist()))
    assert pairs == {(0, 1, 1.0), (1, 2, -1.0)}
