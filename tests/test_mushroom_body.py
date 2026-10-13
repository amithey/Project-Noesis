import numpy as np
import pandas as pd
from scipy import sparse

from noesis.connectome import Connectome
from noesis.lif import LIFNetwork
from noesis.mushroom_body import (
    DopamineGatedPlasticity,
    MBPopulations,
    populations,
    with_weak_pathway_edges,
)

EMPTY = np.zeros(0, int)


def _net():
    # KC 0 and KC 1 -> MBON 2 (equal, excitatory); DAN 3; neuron 4 is an unrelated target of KC 0
    W = sparse.csr_matrix(np.array([
        [0, 0, 10.0, 0, 7.0],
        [0, 0, 10.0, 0, 0.0],
        [0, 0, 0, 0, 0.0],
        [0, 0, 0, 0, 0.0],
        [0, 0, 0, 0, 0.0],
    ]))
    pops = MBPopulations(orn=EMPTY, pn=EMPTY, kc=np.array([0, 1]), mbon=np.array([2]), dan=np.array([3]))
    net = LIFNetwork(W)
    return net, pops, DopamineGatedPlasticity(net, pops, eta=0.05, tau_baseline_ms=1e9)


def _weight(net, pre, post):
    row = slice(net.indptr[pre], net.indptr[pre + 1])
    return net.data[row][net.indices[row] == post][0]


def _train(net, plast, kc_spikes, dan_spikes, steps=200):
    for _ in range(steps):
        spikes = np.array([0] * kc_spikes + [3] * dan_spikes, dtype=int)
        plast.step(spikes, 5.0)


def test_only_kc_to_mbon_synapses_are_plastic():
    _, _, plast = _net()
    assert len(plast.pos) == 2 and set(plast.pre) == {0, 1}


def test_without_dopamine_nothing_changes_even_when_kcs_fire():
    net, _, plast = _net()
    _train(net, plast, kc_spikes=2, dan_spikes=0)
    assert plast.mean_weight_fraction == 1.0


def test_dopamine_paired_with_a_kc_depresses_only_that_kcs_synapse():
    net, _, plast = _net()
    _train(net, plast, kc_spikes=2, dan_spikes=3)  # KC 0 active, KC 1 silent, DAN firing
    assert _weight(net, 0, 2) < 0.7 * 10.0       # depressed
    assert _weight(net, 1, 2) == 10.0            # silent KC untouched
    assert _weight(net, 0, 4) == 7.0             # a non-MBON target of the same KC untouched


def test_depression_never_flips_the_sign_or_exceeds_the_connectome_value():
    net, _, plast = _net()
    _train(net, plast, kc_spikes=5, dan_spikes=10, steps=2000)
    w = _weight(net, 0, 2)
    assert 0.0 <= w <= 10.0


def test_baseline_dopamine_below_the_gate_is_not_reward():
    net, _, plast = _net()
    _train(net, plast, kc_spikes=2, dan_spikes=0)  # sets baseline at the (zero) resting level
    plast.baseline = 0.0
    # one DAN spike per 5 ms window over 1 DAN neuron is 200 Hz: above any 5 Hz gate; none is not
    plast.step(np.array([0, 0]), 5.0)
    assert plast.mean_weight_fraction == 1.0


def test_weights_recover_toward_the_connectome_value_when_learning_stops():
    net, _, plast = _net()
    _train(net, plast, kc_spikes=2, dan_spikes=3)
    low = plast.mean_weight_fraction
    plast.tau_rec = 200.0
    for _ in range(400):
        plast.step(np.array([], dtype=int), 5.0)
    assert plast.mean_weight_fraction > low


def test_populations_pick_up_kenyon_cells_mbons_and_dopamine_neurons():
    rows = [
        ("KCg", "intrinsic", "KCg", "x"), ("KCab", "intrinsic", "KCab", "x"),
        ("", "mushroom_body_output_neuron", "MBON01", "x"),
        ("", "mushroom_body_dopaminergic_neuron", "PAM05", "x"),
        ("antenna_olfactory_receptor_neuron", "sensory", "ORN_DA1", "x"),
        ("", "antennal_lobe_projection_neuron", "DA1_lPN", "x"),
    ]
    df = pd.DataFrame(rows, columns=["Sub Class", "Class", "Primary Cell Type", "Flow"])
    p = populations(df)
    assert list(p.kc) == [0, 1] and list(p.mbon) == [2] and list(p.dan) == [3]
    assert list(p.orn) == [4] and list(p.pn) == [5]


def test_weak_edges_are_added_only_inside_the_olfactory_mushroom_body_pathway():
    neurons = pd.DataFrame({"Predicted NT type": ["ACH", "ACH", "ACH", "GABA"]})
    W = sparse.csr_matrix((4, 4))
    cn = Connectome(neurons=neurons, W=W)
    pairs = pd.DataFrame({"pre": [0, 0, 3], "post": [1, 3, 1], "syn": [2, 2, 3]})
    pops = MBPopulations(orn=EMPTY, pn=EMPTY, kc=np.array([0]), mbon=np.array([1]), dan=EMPTY)
    out = with_weak_pathway_edges(cn, pairs, pops).toarray()
    assert out[0, 1] == 2.0   # KC -> MBON, 2 synapses, ACh excitatory
    assert out[0, 3] == 0.0   # target outside the pathway
    assert out[3, 1] == 0.0   # presynaptic neuron outside the pathway
