import numpy as np
import pandas as pd

from noesis.experiments.mb_assumed_reward import SCHEDULES, assumed_reward_cells


def test_schedules_have_equal_odour_exposure_and_equal_reward_and_paired_overlap():
    odour = {k: sum(o for o, _ in v) for k, v in SCHEDULES.items()}
    reward = {k: sum(r for _, r in v) for k, v in SCHEDULES.items()}
    for family in (("paired", "unpaired", "none"), ("paired_gap", "unpaired_gap", "none_gap")):
        assert len({odour[k] for k in family}) == 1                   # same odour A exposure within a family
    assert reward["paired"] == reward["unpaired"] > reward["none"] == 0
    assert reward["paired_gap"] == reward["unpaired_gap"] > reward["none_gap"] == 0
    assert all(o == r for o, r in SCHEDULES["paired"])                # reward only with the odour
    assert all(o != r for o, r in SCHEDULES["unpaired"])              # reward never with the odour


def test_assumed_reward_cells_selects_pam01_to_pam08_inside_dan_population():
    neurons = pd.DataFrame({"Primary Cell Type": ["PAM01", "PAM09", "PPL101", "PAM08", None, "PAM04"]})
    dan = np.array([0, 1, 2, 3, 5])
    assert assumed_reward_cells(neurons, dan).tolist() == [0, 3, 5]


def test_gap_control_puts_the_reward_beyond_the_kenyon_cell_trace():
    blocks = SCHEDULES["unpaired_gap"]
    last_odour = max(i for i, (o, _) in enumerate(blocks) if o)
    first_reward = min(i for i, (_, r) in enumerate(blocks) if r)
    assert (first_reward - last_odour - 1) * 250 >= 1000
    assert all(o == r for o, r in SCHEDULES["paired_gap"][:2])
