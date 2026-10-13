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


def test_graded_apl_removes_spiking_output_and_inhibits_in_proportion_to_kc_activity():
    from noesis.mushroom_body import GradedAPL
    # KC 0, KC 1 -> APL 2 ; APL 2 -> KC 0 (6 syn), KC 1 (3 syn)
    W = sparse.csr_matrix(np.array([[0, 0, 4.0], [0, 0, 4.0], [-6.0, -3.0, 0]]))
    pops = MBPopulations(orn=EMPTY, pn=EMPTY, kc=np.array([0, 1]), mbon=EMPTY, dan=EMPTY, apl=np.array([2]))
    net = LIFNetwork(W)
    apl = GradedAPL(net, pops, gain=1.0, tau_ms=1e-6)
    assert not net.data[net.indptr[2]:net.indptr[3]].any()   # spiking output removed
    quiet = apl.current(np.array([], dtype=int), 5.0)["current_mv_per_ms"]
    busy = apl.current(np.array([0, 0, 1, 1]), 5.0)["current_mv_per_ms"]
    assert np.all(quiet == 0) and np.all(busy < 0)
    assert np.isclose(busy[0] / busy[1], 2.0)                 # follows APL->KC synapse counts


def test_compartments_reweight_axonal_input_onto_mbons_only():
    from noesis.mushroom_body import compartmentalize_mbons
    neurons = pd.DataFrame({"Predicted NT type": ["ACH", "GABA", "ACH"], "Primary Cell Type": ["KCg", "x", "MBON01"]})
    # 0 -> 2 (KC onto MBON), 1 -> 2 (inhibitory, mostly axonal), 0 -> 1 (not an MBON target)
    W = sparse.csr_matrix(np.array([[0, 8.0, 10.0], [0, 0, -12.0], [0, 0, 0]]))
    comp = pd.DataFrame({"pre": [0, 1], "post": [2, 2], "dend": [10.0, 2.0], "axon": [0.0, 10.0]})
    out = compartmentalize_mbons(W, comp, neurons, axonal_weight=0.1).toarray()
    assert out[0, 2] == 10.0                    # dendritic KC input unchanged
    assert np.isclose(out[1, 2], -(2.0 + 1.0))  # 2 dendritic + 0.1 * 10 axonal, sign kept
    assert out[0, 1] == 8.0                     # non-MBON synapse untouched


def test_mean_weight_fraction_ignores_zero_weight_synapses():
    _, _, plast = _net()
    plast.w0[0] = 0.0
    assert np.isfinite(plast.mean_weight_fraction)


def _fafb_example():
    """BANC: PN 0, KCs 1-3 (KCg-m left, KCg left, KCab right), MBON 4, DAN 5, other 6.
    FAFB: one PN, four KCs, one MBON, one DAN (ids 100+)."""
    from noesis.connectome import Connectome
    neurons = pd.DataFrame({
        "Primary Cell Type": ["DA1_lPN", "KCg-m", "KCg", "KCab", "MBON01", "PAM05", "other"],
        "Sub Class": ["", "KCg", "KCg", "KCab", "", "", ""],
        "Class": ["antennal_lobe_projection_neuron", "", "", "", "", "", ""],
        "Soma side": ["left", "left", "left", "right", "left", "left", "left"],
        "Predicted NT type": ["ACH", "ACH", "ACH", "ACH", "GLUT", "DA", "ACH"],
    })
    pops = MBPopulations(orn=EMPTY, pn=np.array([0]), kc=np.array([1, 2, 3]), mbon=np.array([4]),
                         dan=np.array([5]))
    # BANC edges: PN->KC1 (5), KC1->MBON (2), KC1->other (9), other->MBON (7), MBON->DAN (6)
    W = sparse.csr_matrix(((5.0, 2.0, 9.0, 7.0, -6.0), ((0, 1, 1, 6, 4), (1, 4, 6, 4, 5))), shape=(7, 7))
    fafb_neurons = pd.DataFrame({
        "root_id": [100, 101, 102, 103, 104, 105, 106],
        "primary_type": ["DA1_lPN", "KCg-m", "KCg-m", "KCab", "KCapbp-m", "MBON01", "PAM05"],
        "class": ["ALPN", "Kenyon_Cell", "Kenyon_Cell", "Kenyon_Cell", "Kenyon_Cell", "MBON", "DAN"],
        "side": ["left", "left", "left", "right", "right", "left", "left"],
    })
    fafb_pairs = pd.DataFrame({
        "pre_root_id": [100, 101, 102, 103, 104, 105],
        "post_root_id": [101, 105, 105, 105, 105, 106],
        "syn": [20, 40, 30, 50, 60, 8],
        "mb": [0, 40, 30, 45, 60, 0],
    })
    return Connectome(neurons=neurons, W=W), pops, fafb_neurons, fafb_pairs


def test_fafb_mushroom_body_matches_cell_types_and_kcs_one_to_one():
    from noesis.mushroom_body import with_fafb_mushroom_body
    cn, pops, fn, fp = _fafb_example()
    r = with_fafb_mushroom_body(cn.W, cn.neurons, pops, fn, fp, seed=0)
    b2f = r.banc_to_fafb.to_dict()
    assert b2f[0] == 100 and b2f[4] == 105 and b2f[5] == 106      # PN, MBON, DAN by type
    assert b2f[3] == 103                                            # KCab right -> the KCab
    assert {b2f[1], b2f[2]} == {101, 102}                           # KCg-m, then KCg by lobe class
    assert 104 not in b2f.values()                                  # FAFB surplus KC is subsampled away
    assert r.report.loc["kc", "matched"] == 3 and r.report.loc["kc", "FAFB neurons"] == 4


def test_fafb_mushroom_body_replaces_only_edges_among_matched_neurons():
    from noesis.mushroom_body import with_fafb_mushroom_body
    cn, pops, fn, fp = _fafb_example()
    r = with_fafb_mushroom_body(cn.W, cn.neurons, pops, fn, fp, seed=0)
    out = r.W.toarray()
    kc_of = {v: k for k, v in r.banc_to_fafb.items()}
    assert out[0, kc_of[101]] == 20.0                   # PN -> KC from FAFB
    assert out[0, 1] == (20.0 if kc_of[101] == 1 else 0.0)  # BANC PN->KC1 (5) replaced
    assert out[3, 4] == 50.0                            # KCab -> MBON from FAFB
    assert out[1, 4] in (40.0, 30.0)                    # BANC 2 synapses replaced by FAFB count
    assert out[4, 5] == -8.0                            # MBON -> DAN, signed by BANC NT (GLUT)
    assert out[1, 6] == 9.0 and out[6, 4] == 7.0        # edges to/from the rest of BANC stay
    assert r.report.loc["kc", "FAFB syn"] == 180 and r.report.loc["kc", "FAFB syn mapped"] == 120


def test_fafb_mushroom_body_keeps_compartments_and_is_seeded():
    from noesis.mushroom_body import with_fafb_mushroom_body
    cn, pops, fn, fp = _fafb_example()
    r = with_fafb_mushroom_body(cn.W, cn.neurons, pops, fn, fp, seed=0)
    kcab = r.compartments[r.compartments["pre"] == 3].iloc[0]
    assert kcab["post"] == 4 and kcab["dend"] == 45 and kcab["axon"] == 5
    again = with_fafb_mushroom_body(cn.W, cn.neurons, pops, fn, fp, seed=0)
    assert again.banc_to_fafb.equals(r.banc_to_fafb)
