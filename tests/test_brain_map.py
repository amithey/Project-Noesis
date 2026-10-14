import numpy as np
import pandas as pd
from scipy import sparse

from noesis.viz.brain_map import neuron_positions_3d, strongest_synapses


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


def test_strongest_synapses_keep_the_real_signed_edges():
    weights = sparse.csr_matrix(np.array([[0, 10, 1], [0, 0, -4], [0, 0, 0]], float))
    pre, post, sign = strongest_synapses(weights, limit=2)
    pairs = set(zip(pre.tolist(), post.tolist(), sign.tolist()))
    assert pairs == {(0, 1, 1.0), (1, 2, -1.0)}
