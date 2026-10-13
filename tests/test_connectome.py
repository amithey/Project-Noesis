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
