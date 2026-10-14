import pandas as pd
import pytest

from noesis.connectome import build_connectome


@pytest.fixture
def tiny(tmp_path):
    pd.DataFrame(
        {"Root ID": [10, 20, 30], "Predicted NT type": ["ACH", "GABA", None]}
    ).to_csv(tmp_path / "n.csv", index=False)
    pd.DataFrame(
        {
            "pre_root_id": [10, 10, 20, 30, 10],
            "post_root_id": [20, 20, 30, 10, 30],
            "neuropil": ["A", "B", "A", "A", "A"],
            "syn_count": [3, 4, 9, 6, 2],
            "nt_type": [None] * 5,
        }
    ).to_csv(tmp_path / "c.csv", index=False)
    return tmp_path


def test_aggregates_neuropils_and_signs_by_presynaptic_nt(tiny):
    cn = build_connectome(tiny / "n.csv", tiny / "c.csv", min_syn=5)
    W = cn.W.toarray()
    assert W[0, 1] == 7    # 3 + 4 synapses across two neuropils, ACh -> excitatory
    assert W[1, 2] == -9   # GABA -> inhibitory
    assert W[2, 0] == 6    # unknown NT -> default excitatory
    assert W[0, 2] == 0    # 2 synapses < min_syn, dropped


def test_rejects_connections_to_unknown_neurons(tiny):
    pd.DataFrame(
        {"pre_root_id": [10], "post_root_id": [99], "neuropil": ["A"], "syn_count": [8],
         "nt_type": [None]}
    ).to_csv(tiny / "c.csv", index=False)
    with pytest.raises(ValueError):
        build_connectome(tiny / "n.csv", tiny / "c.csv")


def test_nt_override_makes_apl_inhibitory_despite_its_prediction(tmp_path):
    pd.DataFrame({"Root ID": [1, 2], "Predicted NT type": ["ACH", "ACH"],
                  "Primary Cell Type": ["APL", "KCg"]}).to_csv(tmp_path / "n.csv", index=False)
    pd.DataFrame({"pre_root_id": [1, 2], "post_root_id": [2, 1], "neuropil": ["MB", "MB"],
                  "syn_count": [9, 9], "nt_type": [None, None]}).to_csv(tmp_path / "c.csv", index=False)
    W = build_connectome(tmp_path / "n.csv", tmp_path / "c.csv").W.toarray()
    assert W[0, 1] == -9   # APL -> KC inhibitory (GABAergic), whatever the table predicts
    assert W[1, 0] == 9    # KC -> APL stays excitatory


def test_region_scale_multiplies_synapses_by_region_before_the_threshold(tmp_path):
    from noesis.connectome import neuropil_scale_factors

    pd.DataFrame({"Root ID": [1, 2], "Predicted NT type": ["ACH", "ACH"]}).to_csv(tmp_path / "n.csv", index=False)
    pd.DataFrame({"pre_root_id": [1, 1], "post_root_id": [2, 2], "neuropil": ["MB_ML_L", "VNC_T1"],
                  "syn_count": [2, 1], "nt_type": [None, None]}).to_csv(tmp_path / "c.csv", index=False)
    pd.DataFrame({"pre_root_id": [9, 9], "post_root_id": [8, 8], "neuropil": ["MB_ML_L", "MB_ML_R"],
                  "syn_count": [30, 10], "nt_type": [None, None]}).to_csv(tmp_path / "ref.csv", index=False)
    factors = neuropil_scale_factors(tmp_path / "c.csv", tmp_path / "ref.csv")
    assert factors == {"MB_ML": 20.0}  # (30 + 10) / 2 = 20, both sides pooled; nerve cord has no factor
    W = build_connectome(tmp_path / "n.csv", tmp_path / "c.csv", region_scale=factors).W.toarray()
    assert W[0, 1] == 2 * 20.0 + 1  # MB rows scaled, the unmatched nerve-cord row kept at 1x
    assert build_connectome(tmp_path / "n.csv", tmp_path / "c.csv").W.nnz == 0  # 3 synapses < 5 unscaled


def test_il3ln6_override_applies_on_build_and_cached_reload(tmp_path):
    from noesis.connectome import load_cached

    raw = tmp_path / 'raw'
    raw.mkdir()
    pd.DataFrame({'Root ID': [1, 2, 3], 'Predicted NT type': ['ACH', 'GABA', 'ACH'],
                  'Primary Cell Type': ['il3LN6', 'il3LN6', 'lLN1_bc']}).to_csv(raw / 'neurons.csv', index=False)
    pd.DataFrame({'pre_root_id': [1, 2, 3], 'post_root_id': [3, 3, 1],
                  'syn_count': [9, 8, 7]}).to_csv(raw / 'connections_princeton.csv', index=False)
    cache = tmp_path / 'cache'
    built = load_cached(raw, cache)
    loaded = load_cached(raw, cache)
    assert built.W.toarray().tolist() == loaded.W.toarray().tolist()
    assert built.W[0, 2] == -9 and built.W[1, 2] == -8
    assert built.W[2, 0] == 7  # Established cholinergic LN is untouched.
