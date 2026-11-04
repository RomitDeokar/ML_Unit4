"""Validation-selected causal alert heads; preserve the original HMM submission.

The HMM is unchanged. Supervised heads learn artificial future-event labels on
training patients only. No smoothed probabilities, decoded paths, future vitals,
patient IDs, or simulator states enter their features. These are exploratory
extensions on an already published synthetic test set, not clinical validation.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from patientwatch.data import FEATURES, STATE_NAMES
from patientwatch.pipeline import choose_threshold, metrics

ROOT = Path(__file__).resolve().parents[1]
C_GRID = (.001, .01, .1, 1., 10.)


def causal_features(patient, kind='states'):
    """Prefix-invariant features from the stored, rounded, causal HMM outputs."""
    alpha = np.asarray(patient['filtered'], dtype=float)
    if kind == 'states':
        return alpha
    if kind != 'temporal':
        raise ValueError('Unknown feature family')
    x = pd.DataFrame({f: patient['vitals'][f] for f in FEATURES})
    return np.column_stack([x, x.rolling(6, min_periods=1).mean(),
                            x.diff().fillna(0), x - x.shift(3).fillna(x.iloc[0]), alpha])


def feature_names(kind):
    if kind == 'states':
        return [f'filtered_{s.lower()}' for s in STATE_NAMES]
    return ([f'{prefix}{f}' for prefix in ['', 'mean6_', 'delta1_', 'delta3_'] for f in FEATURES]
            + feature_names('states'))


def eligible_arrays(patients, score_map):
    y, scores = [], []
    for p in patients:
        keep = np.asarray(p['eligible'], dtype=bool)
        y.extend(np.asarray(p['labels'])[keep])
        scores.extend(np.asarray(score_map[p['id']])[keep])
    return np.asarray(y), np.asarray(scores)


def event_metrics(patients, score_map, threshold, horizon=6):
    leads, events, non_events, alerted = [], 0, 0, 0
    for p in patients:
        event = p['event_hour']
        scores = score_map[p['id']]
        if event is not None:
            events += 1
            hits = [t for t in range(max(0, event - horizon), event) if scores[t] >= threshold]
            if hits:
                leads.append(event - hits[0])
        else:
            non_events += 1
            alerted += int(any(k and s >= threshold for k, s in zip(p['eligible'], scores)))
    return dict(total_events=events, detected_events=len(leads), lead_hours=leads,
                event_sensitivity=len(leads) / max(events, 1),
                median_lead_hours=float(np.median(leads)) if leads else None,
                non_event_patients=non_events, non_event_patients_alerted=alerted)


def evaluate(patients, score_map, threshold):
    y, score = eligible_arrays(patients, score_map)
    result = metrics(y, score, threshold)
    result.update(event_metrics(patients, score_map, threshold))
    result['average_precision'] = float(average_precision_score(y, score))
    result['precision'] = result['confusion']['tp'] / max(result['confusion']['tp'] + result['confusion']['fp'], 1)
    fpr, tpr, _ = roc_curve(y, score)
    result['roc'] = dict(fpr=fpr.tolist(), tpr=tpr.tolist())
    return result


def paired_bootstrap(patients, selected_scores, original_scores, seed=142, repeats=400):
    """Paired whole-patient resampling, not an independent-hour confidence interval."""
    rng = np.random.default_rng(seed)
    aucs, deltas = [], []
    for _ in range(repeats):
        sample = [patients[i] for i in rng.integers(0, len(patients), len(patients))]
        y, scores = eligible_arrays(sample, selected_scores)
        _, originals = eligible_arrays(sample, original_scores)
        if len(np.unique(y)) == 2:
            auc = roc_auc_score(y, scores)
            aucs.append(auc)
            deltas.append(auc - roc_auc_score(y, originals))
    return dict(replicates=repeats, valid_replicates=len(aucs),
                auroc_ci95=np.quantile(aucs, [.025, .975]).tolist(),
                paired_auroc_delta_ci95=np.quantile(deltas, [.025, .975]).tolist())


def build_enhancement(source=ROOT / 'artifacts/results.json', output=ROOT / 'artifacts/enhancement.json'):
    source, output = Path(source), Path(output)
    original = json.loads(source.read_text())
    groups = {s: [p for p in original['patients'] if p['split'] == s]
              for s in ['train', 'validation', 'test']}
    original_scores = {p['id']: p['risk'] for p in original['patients']}
    # An explicit identity candidate ensures worse validation models are not promoted.
    y_val, old_val = eligible_arrays(groups['validation'], original_scores)
    candidates = [dict(name='Original HMM', kind='original', C=None,
                       validation_auroc=float(roc_auc_score(y_val, old_val)),
                       validation_brier=float(np.mean((y_val - old_val) ** 2)))]
    fitted = {}
    for kind in ['states', 'temporal']:
        train_x = np.concatenate([causal_features(p, kind)[p['eligible']] for p in groups['train']])
        train_y = np.concatenate([np.asarray(p['labels'])[p['eligible']] for p in groups['train']])
        val_x = np.concatenate([causal_features(p, kind)[p['eligible']] for p in groups['validation']])
        for c in C_GRID:
            head = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000, random_state=42))
            head.fit(train_x, train_y)
            score = head.predict_proba(val_x)[:, 1]
            name = f'{kind.title()} HMM head (C={c:g})'
            candidates.append(dict(name=name, kind=kind, C=c,
                                   validation_auroc=float(roc_auc_score(y_val, score)),
                                   validation_brier=float(np.mean((y_val - score) ** 2))))
            fitted[name] = head
    # Freeze model choice before generating any enhanced test predictions.
    winner = max(candidates, key=lambda x: (x['validation_auroc'], -x['validation_brier']))
    head = fitted.get(winner['name'])
    scores = original_scores if head is None else {
        p['id']: head.predict_proba(causal_features(p, winner['kind']))[:, 1].tolist()
        for p in original['patients']}
    y_val, score_val = eligible_arrays(groups['validation'], scores)
    threshold = choose_threshold(y_val, score_val)
    parameter_export = None
    if head is not None:
        scale, linear = head.steps[0][1], head.steps[1][1]
        parameter_export = dict(feature_names=feature_names(winner['kind']),
                                mean=scale.mean_.tolist(), scale=scale.scale_.tolist(),
                                coef=linear.coef_[0].tolist(), intercept=float(linear.intercept_[0]))
    result = dict(schema_version=1, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  description='Exploratory supervised alert head on frozen unsupervised HMM; synthetic only',
                  selection='Maximum validation AUROC, then minimum validation Brier; test never selects candidates',
                  caveat='Existing test cohort was already published. Improvements require a fresh external cohort; not clinical evidence.',
                  candidates=candidates, selected=winner, threshold=threshold, parameters=parameter_export,
                  validation=evaluate(groups['validation'], scores, threshold),
                  test=evaluate(groups['test'], scores, threshold),
                  original_test=evaluate(groups['test'], original_scores, original['threshold']),
                  baseline_test=evaluate(groups['test'], {p['id']: p['baseline'] for p in original['patients']},
                                         original['metrics']['baseline']['threshold']),
                  bootstrap=paired_bootstrap(groups['test'], scores, original_scores), scores=scores)
    output.write_text(json.dumps(result, allow_nan=False))
    return result


if __name__ == '__main__':
    result = build_enhancement()
    print(json.dumps({k: result[k] for k in ['selected', 'test', 'bootstrap']}, indent=2))
