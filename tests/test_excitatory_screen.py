from copy import deepcopy

from noesis.experiments import excitatory_screen as screen


def test_depression_requires_improved_recovery_and_preserves_passing_sensory_tests(monkeypatch):
    base = {c: {'recovery_mean_hz': [10.]} for c in screen.CONDITIONS}
    improved = {c: {'recovery_mean_hz': [5.]} for c in screen.CONDITIONS}
    base['flags'] = [True]*7+[False]
    improved['flags'] = [True]*8
    monkeypatch.setattr(screen, 'judge', lambda b: [(str(i), ok, '') for i, ok in enumerate(b['flags'])])
    batteries = {0.: base, .03: improved, .1: deepcopy(improved)}
    assert all(d['accepted'] for d in screen.screen_decision(batteries))
    batteries[.1]['flags'][4] = False
    assert not screen.screen_decision(batteries)[1]['accepted']
    batteries[.03]['flags'][-1] = False
    assert not screen.screen_decision(batteries)[0]['accepted']
