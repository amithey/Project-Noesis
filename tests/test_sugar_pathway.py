import numpy as np

from noesis.experiments.sugar_pathway import type_table


def test_type_table_flags_only_types_that_beat_every_control():
    base = np.zeros(6)
    sugar = np.array([10.0, 10.0, 8.0, 8.0, 1.0, 1.0])
    bitter = np.array([1.0, 1.0, 9.0, 9.0, 1.0, 1.0])     # type B also responds to bitter
    rand = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    groups = {"A": np.array([0, 1]), "B": np.array([2, 3]), "C": np.array([4, 5])}
    rows = type_table({"sugar": sugar, "bitter": bitter, "random_afferent_0": rand}, base, groups, 2.0)
    by = {r["type"]: r for r in rows}
    assert by["A"]["sugar_selective"] and by["A"]["excess_over_best_control_hz"] == 9.0
    assert not by["B"]["sugar_selective"] and not by["C"]["sugar_selective"]
    assert rows[0]["type"] == "A"
