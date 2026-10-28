"""The v9 tracker must count contact cycles exactly like the evaluation's debounce, at the same sample."""
import numpy as np
import pytest

from noesis.embodied_metrics import LegCycleTracker, debounce_contact

# Expected strings were produced by noesis_gpu.gait_metrics.debounce_contact (gemini/flygym2-gait-validation
# @ 3bf6594, persistence filter) with dt = 5 ms and min_phase = 15 ms; T = contact, . = no contact.
GEMINI_VECTORS = [
    ("TTTT.T..TTTTT", "TTTTTTTTTTTTT"),
    ("TTTTT.T.T.TTTTT", "TTTTTTTTTTTTTTT"),
    ("..T.TT...TTTT..", "...........TTTT"),
    ("TTTT..TTTT", "TTTTTTTTTT"),
    ("T.T.T.T.T.T.", "TTTTTTTTTTTT"),
    ("TTTTTTTT...TTT", "TTTTTTTTTT...T"),
    ("..TT..TTT..T..", "........TTTTTT"),
    ("TTT.TT.TTTT.T", "TTTTTTTTTTTTT"),
]


def bits(text):
    return np.array([c == "T" for c in text])


def text(arr):
    return "".join("T" if v else "." for v in arr)


@pytest.mark.parametrize("raw, expected", GEMINI_VECTORS)
def test_debounce_matches_the_evaluation_filter(raw, expected):
    assert text(debounce_contact(bits(raw), 5., 15.)) == expected


def test_debounce_is_causal_each_sample_depends_only_on_the_past():
    rng = np.random.default_rng(3)
    for _ in range(50):
        signal = rng.random(80) < .5
        full = debounce_contact(signal, 5., 15.)
        for cut in (10, 33, 61):
            assert np.array_equal(debounce_contact(signal[:cut], 5., 15.), full[:cut])


def offline_events(raw):
    debounced = debounce_contact(raw, 5., 15.)
    diff = np.diff(debounced.astype(int))
    touchdowns = np.flatnonzero(diff == 1) + 1
    liftoffs = np.flatnonzero(diff == -1) + 1
    return [int(t) for t in touchdowns if (liftoffs < t).any()]


def tracker_events(signal):
    tracker = LegCycleTracker()
    events = [[] for _ in range(6)]
    for i, row in enumerate(signal):
        tracker.update(row.astype(float), 1., True)
        for leg in np.flatnonzero(tracker.cycles[-1]):
            events[leg].append(i)  # same sample as the evaluation's touchdown: no read-out delay
    return events


@pytest.mark.parametrize("seed", range(8))
def test_tracker_events_equal_the_offline_events_exactly(seed):
    rng = np.random.default_rng(seed)
    n = 600
    signal = np.zeros((n, 6), bool)
    for leg in range(6):
        state, i = bool(rng.integers(2)), 0
        while i < n:
            length = int(rng.integers(3, 40))
            signal[i:i + length, leg] = state
            if rng.random() < .35:  # flutter burst of 1-2 sample flips
                for j in range(i, min(i + length, n), int(rng.integers(2, 4))):
                    signal[j, leg] = not state
            state, i = not state, i + length
    events = tracker_events(signal)
    for leg in range(6):
        assert events[leg] == offline_events(signal[:, leg])


def test_flutter_is_stance_not_a_lift_and_a_first_landing_is_not_a_cycle():
    flutter = bits("T" * 12 + "T.T.T.T.T.T." + "T" * 12)  # alternating one-sample dropouts: still contact
    assert tracker_events(np.tile(flutter[:, None], (1, 6)))[0] == []
    landing = bits("." * 12 + "T" * 12)  # first landing with no earlier lift
    assert tracker_events(np.tile(landing[:, None], (1, 6)))[0] == []
    real = bits("T" * 8 + "." * 12 + "T" * 10)
    assert tracker_events(np.tile(real[:, None], (1, 6)))[0] == offline_events(real) and len(offline_events(real)) == 1


def test_reset_restarts_the_stream():
    signal = np.tile(bits("T" * 8 + "." * 12 + "T" * 10)[:, None], (1, 6))
    tracker = LegCycleTracker()
    for row in signal:
        tracker.update(row.astype(float), 1., True)
    tracker.reset()
    assert not tracker.started and not tracker.lifted.any() and not tracker.cycles
