import pytest
from noesis_gpu.train import episode_scorer
from test_cycle_scoring import V8, V9, episode


def test_native_weight_reaches_reward_and_keeps_credit_schedule():
    low, high = episode_scorer(V9), episode_scorer(V9, .5)
    a, b = episode(low, 900), episode(high, 900)
    assert [x['cycle_credit'] for x in a] == [x['cycle_credit'] for x in b]
    assert all(x['cycle_weight'] == .5 for x in b)
    assert high.summary()['intended_bonus_sum_before_clipping'] == pytest.approx(
        5 * low.summary()['intended_bonus_sum_before_clipping'])


@pytest.mark.parametrize('objective,weight', [(V8,.5),(V8,.1),(V9,0),(V9,float('nan')),(V9,1.1)])
def test_invalid_native_weight_rejected_before_simulation(objective, weight):
    with pytest.raises(ValueError):
        episode_scorer(objective, weight)
