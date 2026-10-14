import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from noesis.connectome import Connectome
from noesis.experiments.olfactory_specificity import (
    apl_coverage_audit,
    calibrate_gains,
    frozen_signs,
    make_pairs,
    orn_glomerulus,
    overlap_metrics,
    pn_glomerulus,
    rate_metrics,
    resign_weights,
    spread,
)


def test_glomerulus_labels_do_not_confuse_subtypes_or_unknown_pns():
    assert orn_glomerulus('ORN_DA1') == 'DA1'
    assert orn_glomerulus('D_ORN') == 'D'
    assert orn_glomerulus('ORN') is None
    assert orn_glomerulus('') is None
    available = {'DA1', 'DA1d', 'D'}
    assert pn_glomerulus('DA1_adPN', available) == 'DA1'
    assert pn_glomerulus('DA1d_lPN', available) == 'DA1d'
    for label in ('M_lPN1', 'CB345', 'DA1+D_adPN', 'DA1', 'DA2_adPN'):
        assert pn_glomerulus(label, available) is None


def test_pairs_are_reproducible_and_globally_disjoint():
    labels = [f'G{i}' for i in range(51)]
    pairs = make_pairs(labels)
    assert pairs == make_pairs(reversed(labels))
    all_sets = [set(p[k]) for p in pairs for k in ('A', 'B')]
    assert len(pairs) == 5 and all(len(s) == 5 for s in all_sets)
    assert len(set.union(*all_sets)) == 50
    with pytest.raises(ValueError):
        make_pairs(labels[:49])


def test_rates_include_silent_neurons_and_window_duration():
    m = rate_metrics(np.array([0, 1, 5, 0]), 500)
    assert m == {'n': 4, 'mean_hz': 3, 'active_fraction': .5, 'ge5hz_fraction': .25}
    assert rate_metrics([], 500)['mean_hz'] is None
    with pytest.raises(ValueError):
        rate_metrics([1], 0)
    with pytest.raises(ValueError):
        rate_metrics([-1], 500)


def test_overlap_uses_union_and_chance_and_handles_silence_honestly():
    m = overlap_metrics(np.array([2, 1, 0, 0]), np.array([1, 0, 3, 0]))
    assert m['intersection'] == 1 and m['union'] == 3
    assert m['jaccard'] == pytest.approx(1 / 3)
    assert m['chance_jaccard'] == pytest.approx(1 / 3)
    assert m['cosine'] == pytest.approx(2 / np.sqrt(50))
    assert overlap_metrics(np.zeros(4), np.zeros(4))['jaccard'] is None
    assert overlap_metrics(np.zeros(4), np.ones(4))['cosine'] is None
    with pytest.raises(ValueError):
        overlap_metrics([1], [1, 2])


def test_spread_excludes_undefined_values_and_reports_sample_sd():
    result = spread([1, None, 3])
    assert result == {'n': 2, 'mean': 2, 'sd': pytest.approx(np.sqrt(2)), 'min': 1, 'max': 3}
    assert spread([])['mean'] is None


def test_calibration_keeps_held_out_odours_out_of_gain_selection():
    trials = []
    for gain, training, held in [(1, .20, .075), (16, .075, .14), (64, 0, 0)]:
        for pair in range(5):
            for seed in (0, 1, 2):
                stats = {'full': {'kc': {'active_fraction': training if pair < 3 else held}}}
                trials.append({'gain': gain, 'treatment': 'control', 'pair': pair, 'seed': seed,
                               'A': stats, 'B': stats})
    result = calibrate_gains(trials)
    assert result['selected_gain'] == 16
    assert result['curve'][1]['calibration']['n'] == 18
    assert result['curve'][1]['held_out']['n'] == 12
    assert result['curve'][1]['held_out']['mean'] == pytest.approx(.14)
    incomplete = [t for t in trials if not (t['gain'] == 16 and t['pair'] == 4 and t['seed'] == 2)]
    assert calibrate_gains(incomplete)['selected_gain'] is None
    treatment_trials = [{**t, 'treatment': 'il3ln6_gaba'} for t in trials]
    assert calibrate_gains(treatment_trials)['selected_gain'] is None
    assert calibrate_gains(treatment_trials, treatment='il3ln6_gaba')['selected_gain'] == 16


def test_coverage_audit_distinguishes_missing_edges_from_threshold_loss(tmp_path):
    n = pd.DataFrame({'root_id': [10, 11, 20, 21, 22],
                      'Primary Cell Type': ['APL', 'APL', 'KC1', 'KC2', 'KC3'],
                      'Class': ['local'] * 5, 'Sub Class': ['', '', 'KCg', 'KCg', 'KCab']})
    W = sparse.csr_matrix(([-10., -20.], ([0, 1], [2, 3])), shape=(5, 5))
    raw = pd.DataFrame({'pre_root_id': [10, 11, 10], 'post_root_id': [20, 21, 22],
                        'syn_count': [5, 10, 1]})
    raw.to_csv(tmp_path / 'raw.csv', index=False)
    raw.to_csv(tmp_path / 'ref.csv.gz', index=False)
    pd.DataFrame({'root_id': n.root_id, 'primary_type': n['Primary Cell Type']}).to_csv(
        tmp_path / 'types.csv.gz', index=False)
    result = apl_coverage_audit(Connectome(n, W), tmp_path / 'raw.csv',
                                tmp_path / 'ref.csv.gz', tmp_path / 'types.csv.gz')
    assert result['corrected_uncovered_kcs'] == 1
    assert result['raw_uncovered_kcs'] == result['fafb_uncovered_kcs'] == 0
    assert result['uncovered_by_subclass'] == {'KCab': 1}


def test_frozen_signs_restore_control_without_mutation():
    n = pd.DataFrame({'Primary Cell Type': ['APL', 'il3LN6', 'other', 'unknown'],
                      'Predicted NT type': ['ACH', 'ACH', 'GLUT', None]})
    W = sparse.csr_matrix(([-3., -5., -7., 2.], ([0, 1, 2, 3], [1, 2, 3, 0])), shape=(4, 4))
    original = W.copy()
    baseline = resign_weights(W, n, {'APL': 'GABA'})
    corrected = resign_weights(W, n, {'APL': 'GABA', 'il3LN6': 'GABA'})
    assert frozen_signs(n, {'APL': 'GABA'}).tolist() == [-1, 1, -1, 1]
    assert baseline.data.tolist() == [-3, 5, -7, 2]
    assert corrected.data.tolist() == [-3, -5, -7, 2]
    assert np.array_equal(W.data, original.data)
    assert np.array_equal(baseline.indices, W.indices) and baseline.nnz == W.nnz
