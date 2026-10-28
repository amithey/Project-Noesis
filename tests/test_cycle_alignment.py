"""The v9 tracker must count contact cycles exactly like the evaluation's offline debounce."""
import numpy as np
import pytest

from noesis.embodied_metrics import LegCycleTracker, debounce_contact

# Expected strings were produced by noesis_gpu.gait_metrics.debounce_contact (gemini/flygym2-gait-validation
# @ 950e87f) with dt = 5 ms and min_phase = 15 ms; T = contact, . = no contact.
GEMINI_VECTORS = [
    ("TTTT.T..TTTTT", "TTTTTTTTTTTTT"),
    ("TTTTT.T.T.TTTTT", "TTTTTTTTTTTTTTT"),
    ("..T.TT...TTTT..", "..TTTT...TTTT.."),
    ("TTTT..TTTT", "TTTTTTTTTT"),
    ("T.T.T.T.T.T.", "TTTTTTTTTTT."),
    ("TTTTTTTT...TTT", "TTTTTTTT...TTT"),
    ("..TT..TTT..T..", "..TTTTTTTTTT.."),
    ("TTT.TT.TTTT.T", "TTTTTTTTTTTTT"),
]


def bits(text):
    return np.array([c == "T" for c in text])


def text(arr):
    return "".join("T" if v else "." for v in arr)


@pytest.mark.parametrize("raw, expected", GEMINI_VECTORS)
def test_debounce_matches_the_evaluation_filter(raw, expected):
    assert text(debounce_contact(bits(raw), 5., 15.)) == expected


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
            events[leg].append(i - tracker.delay)
    return events, tracker.delay


@pytest.mark.parametrize("seed", range(8))
def test_tracker_events_equal_the_offline_events_exactly(seed):
    rng = np.random.default_rng(seed)
    # contact with long phases plus flutter bursts, the case that used to differ by ~30%
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
    events, delay = tracker_events(signal)
    for leg in range(6):
        visible = [t for t in offline_events(signal[:, leg]) if t <= n - 1 - delay]
        assert events[leg] == visible


def test_flutter_inside_a_lift_counts_the_same_way_as_in_the_evaluation():
    # a long lift whose end is a fluttering landing: closing bridges the dropouts into one touchdown
    leg = bits("TTTTTTTT" + "." * 20 + "T.T.TTTTTTTTTT" + "." * 10 + "TTTTTTTT")
    signal = np.tile(leg[:, None], (1, 6))
    signal = np.vstack([signal, np.ones((20, 6), bool)])  # trailing stance lets the last event surface
    events, _ = tracker_events(signal)
    assert events[0] == offline_events(signal[:, 0]) and len(events[0]) == 2


def test_delay_is_validated_and_exposed():
    assert LegCycleTracker().delay == 6  # 30 ms at 5 ms
    assert LegCycleTracker(delay_ms=0.).delay == 0
    for bad in (-1., float("nan"), float("inf")):
        with pytest.raises(ValueError):
            LegCycleTracker(delay_ms=bad)


def test_reset_restarts_the_stream():
    signal = np.tile(bits("TTTTTTTT" + "." * 12 + "TTTTTTTTTT")[:, None], (1, 6))
    tracker = LegCycleTracker()
    for row in signal:
        tracker.update(row.astype(float), 1., True)
    tracker.reset()
    assert tracker.count == 0 and not tracker.lifted.any() and not tracker.raw
