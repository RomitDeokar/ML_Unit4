import itertools
import numpy as np
from scipy.special import logsumexp
from patientwatch.inference import forward_backward, viterbi, horizon_risk
from patientwatch.data import generate_cohort, chronological_split, clean_sequences, FEATURES, outcome_labels


def test_forward_backward_matches_exact_enumeration():
    start = np.array([.6, .4]); a = np.array([[.8, .2], [.3, .7]])
    b = np.array([[.5, .2], [.1, .8], [.7, .4]])
    paths = list(itertools.product(range(2), repeat=3)); probs = []
    for p in paths:
        probs.append(start[p[0]] * b[0,p[0]] * a[p[0],p[1]] * b[1,p[1]] * a[p[1],p[2]] * b[2,p[2]])
    alpha, gamma, ll, _ = forward_backward(np.log(b), start, a)
    exact = np.array([[sum(v for p,v in zip(paths, probs) if p[t] == k) / sum(probs) for k in range(2)] for t in range(3)])
    assert np.allclose(gamma, exact)
    assert np.isclose(ll, np.log(sum(probs)))
    assert tuple(viterbi(np.log(b), start, a)) == paths[np.argmax(probs)]
    assert np.allclose(alpha.sum(axis=1), 1)


def test_long_sequence_is_stable():
    alpha, gamma, ll, scales = forward_backward(np.full((10000, 3), -1000.), np.ones(3)/3, np.full((3,3), 1/3))
    assert np.isfinite(ll) and np.isfinite(gamma).all() and (scales > 0).all()
    assert np.isclose(ll, -10000000)


def test_split_disjoint_and_chronological():
    parts = chronological_split(generate_cohort(patients=20))
    ids = [set(x.patient_id) for x in parts]
    assert not ids[0] & ids[1] and not ids[1] & ids[2] and not ids[0] & ids[2]
    assert parts[0].timestamp.max() < parts[1].timestamp.min() < parts[2].timestamp.min()


def test_preprocessing_and_filter_are_causal():
    raw = generate_cohort(patients=5)
    clean, medians = clean_sequences(raw)
    pid = raw.patient_id.iloc[0]
    g = raw[raw.patient_id == pid].copy()
    changed = g.copy(); changed.loc[changed.hour > 20, FEATURES] = 999
    a, _ = clean_sequences(g, medians); b, _ = clean_sequences(changed, medians)
    assert np.allclose(a[FEATURES].iloc[:21], b[FEATURES].iloc[:21])
    logs = np.random.default_rng(1).normal(size=(48,3))
    start = np.ones(3)/3; trans = np.full((3,3), .1) + np.eye(3)*.7
    alpha, _, _, _ = forward_backward(logs, start, trans)
    logs[21:] = 500
    new, _, _, _ = forward_backward(logs, start, trans)
    assert np.allclose(alpha[:21], new[:21])
    assert clean[FEATURES].notna().all().all()


def test_endpoint_and_censoring():
    import pandas as pd
    g = pd.DataFrame({'true_state': [0]*8 + [2,2,2,2]})
    labels, keep, event = outcome_labels(g)
    assert event == 8 and keep.sum() == 8
    assert labels[:8].tolist() == [0,0,1,1,1,1,1,1]
    g.true_state = 0
    labels, keep, event = outcome_labels(g)
    assert event is None and not labels.any() and keep.sum() == 6


def test_horizon_probability():
    assert np.allclose(horizon_risk(np.eye(3), np.eye(3), 2), [0,0,1])
