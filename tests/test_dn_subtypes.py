import numpy as np

from noesis.experiments.dn_subtypes import GROUPS, group_stats


def test_groups_are_prespecified_and_match_expected_names():
    import re
    assert re.match(GROUPS["DNa01-04"], "DNa03") and not re.match(GROUPS["DNa01-04"], "DNa05")
    assert re.match(GROUPS["DNa family"], "DNa11") and not re.match(GROUPS["DNa family"], "DNad004")


def test_group_stats_flags_ipsilateral_group_and_reports_missing_ones():
    n = 60
    dn = np.arange(n)
    soma = np.array(["left", "right"] * (n // 2))
    ctype = np.array([""] * n, dtype=object)
    ctype[0], ctype[1] = "DNa01", "DNa01"   # one left, one right soma
    ctype[2] = "DNa02"                       # left only: unavailable
    rng = np.random.default_rng(0)
    trials = []
    for sid in range(4):
        for s in ("left", "right"):
            for _ in range(3):
                rate = rng.normal(0, 0.1, n)
                rate[0 if s == "left" else 1] += 5.0   # DNa01 responds on the ipsilateral side
                trials.append({"set": sid, "side": s, "rate": rate})
    stats = group_stats(trials, dn, soma, ctype, 400)
    assert stats["DNa01"]["S_hz"] > 8 and stats["DNa01"]["p_two_sided"] < 0.05
    assert stats["DNa02"]["status"] == "unavailable"
