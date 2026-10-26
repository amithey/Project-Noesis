import pandas as pd

from noesis.fafb_tables import neuron_table


def test_classes_from_types_and_alpn_from_connectivity_and_side_from_inputs():
    types = pd.DataFrame({"root_id": [1, 2, 3, 4, 5, 6, 7],
                          "primary_type": ["KCab", "APL", "MBON01", "PAM05", "DA1_lPN", "ORN_DA1", None]})
    stats = pd.DataFrame({
        "in_total": [100, 100, 100, 100, 100, 5, 100], "in_al": [0, 0, 0, 0, 90, 0, 90],
        "out_total": [100, 100, 100, 100, 100, 100, 100], "out_al": [0, 0, 0, 0, 10, 95, 95],
        "in_left": [90, 10, 0, 100, 80, 0, 0], "in_right": [10, 90, 0, 0, 20, 0, 0]},
        index=pd.Index([1, 2, 3, 4, 5, 6, 7], name="root_id"))
    t = neuron_table(types, stats).set_index("root_id")
    assert t.loc[1, "class"] == "Kenyon_Cell" and t.loc[2, "class"] == "APL"
    assert t.loc[3, "class"] == "MBON" and t.loc[4, "class"] == "DAN"
    assert t.loc[5, "class"] == "ALPN"                       # AL inputs, outputs elsewhere
    assert pd.isna(t.loc[6, "class"])                        # receptor neuron: outputs in the AL
    assert pd.isna(t.loc[7, "class"])                        # local neuron: inputs and outputs in the AL
    assert t.loc[1, "side"] == "left" and t.loc[2, "side"] == "right" and t.loc[5, "side"] == "left"
    assert t.loc[3, "side"] is None or pd.isna(t.loc[3, "side"])
