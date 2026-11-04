"""Train unsupervised HMM; evaluate causal alerts on held-out admissions."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, brier_score_loss
from patientwatch.data import FEATURES, STATE_NAMES, generate_cohort, chronological_split, clean_sequences, outcome_labels
from patientwatch.inference import emissions, forward_backward, viterbi, horizon_risk

ROOT = Path(__file__).resolve().parents[1]


def choose_threshold(y, score):
    fpr, tpr, thresholds = roc_curve(y, score)
    valid = np.isfinite(thresholds) & (thresholds <= 1)
    candidates = np.where(valid)[0]
    return float(thresholds[candidates[np.argmax((tpr - fpr)[valid])]])


def metrics(y, score, threshold):
    tn, fp, fn, tp = confusion_matrix(y, score >= threshold, labels=[0, 1]).ravel()
    return dict(auroc=float(roc_auc_score(y, score)), sensitivity=float(tp / max(tp + fn, 1)),
                specificity=float(tn / max(tn + fp, 1)), brier=float(brier_score_loss(y, score)),
                threshold=float(threshold), confusion=dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp)),
                windows=int(len(y)), positives=int(sum(y)))


def run_pipeline(seed=42, output=ROOT):
    output = Path(output)
    raw = generate_cohort(seed)
    raw_splits = chronological_split(raw)
    train, medians = clean_sequences(raw_splits[0])
    splits = [train] + [clean_sequences(part, medians)[0] for part in raw_splits[1:]]
    scaler = StandardScaler().fit(train[FEATURES])
    lengths = train.groupby('patient_id', sort=False).size().tolist()
    x_train = scaler.transform(train[FEATURES])
    # Select random restart solely by training likelihood; no test selection.
    models = []
    for restart in [seed, seed + 1, seed + 2]:
        model = GaussianHMM(n_components=3, covariance_type='diag', n_iter=150,
                            tol=.001, min_covar=.01, random_state=restart,
                            implementation='scaling').fit(x_train, lengths)
        models.append((model.score(x_train, lengths), model))
    train_ll, model = max(models, key=lambda item: item[0])
    # Interpret states from learned vitals, never from simulator labels.
    severity = model.means_ @ np.array([1, -1, 1, -1, 1])
    order = np.argsort(severity)
    rank = np.argsort(order)
    critical = int(order[-1])
    transition = model.transmat_[order][:, order]
    base_x, base_y = [], []
    for _, g in train.groupby('patient_id', sort=False):
        y, keep, _ = outcome_labels(g)
        base_x.extend(scaler.transform(g[FEATURES])[keep])
        base_y.extend(y[keep])
    baseline = LogisticRegression(max_iter=500, random_state=seed).fit(base_x, base_y)
    patients, by_split = [], {}
    max_gamma_error = max_ll_error = 0.
    for split_name, frame, raw_frame in zip(['train', 'validation', 'test'], splits, raw_splits):
        ys, scores, bases, records = [], [], [], []
        for pid, g in frame.groupby('patient_id', sort=False):
            x = scaler.transform(g[FEATURES])
            log_b = emissions(x, model.means_, model._covars_)
            alpha, gamma, ll, _ = forward_backward(log_b, model.startprob_, model.transmat_)
            ref_ll, ref_gamma = model.score_samples(x)
            max_gamma_error = max(max_gamma_error, float(np.max(np.abs(gamma - ref_gamma))))
            max_ll_error = max(max_ll_error, abs(ll - ref_ll))
            score = horizon_risk(alpha, model.transmat_, critical)
            base_score = baseline.predict_proba(x)[:, 1]
            y, keep, onset = outcome_labels(g)
            ys.extend(y[keep]); scores.extend(score[keep]); bases.extend(base_score[keep])
            original = raw_frame[raw_frame.patient_id == pid]
            record = dict(id=pid, split=split_name, admission=str(g.timestamp.iloc[0]),
                          hours=g.hour.tolist(), timestamps=g.timestamp.tolist(),
                          vitals={f: g[f].round(3).tolist() for f in FEATURES},
                          raw_vitals={f: [None if pd.isna(v) else float(v) for v in original[f]] for f in FEATURES},
                          filtered=alpha[:, order].round(8).tolist(),
                          smoothed=gamma[:, order].round(8).tolist(),
                          decoded=rank[viterbi(log_b, model.startprob_, model.transmat_)].tolist(),
                          # Preserve precision: rounding near-tied scores changes AUROC.
                          risk=score.tolist(), baseline=base_score.tolist(),
                          true_state=g.true_state.tolist(), labels=y.tolist(), eligible=keep.tolist(),
                          event_hour=onset, log_likelihood=ll)
            patients.append(record); records.append(record)
        by_split[split_name] = dict(y=np.array(ys), score=np.array(scores), baseline=np.array(bases), records=records)
    val, test = by_split['validation'], by_split['test']
    threshold = choose_threshold(val['y'], val['score'])
    base_threshold = choose_threshold(val['y'], val['baseline'])
    test_metrics = metrics(test['y'], test['score'], threshold)
    test_metrics['baseline'] = metrics(test['y'], test['baseline'], base_threshold)
    leads, event_count, non_event_alerts, non_events = [], 0, 0, 0
    for p in test['records']:
        event = p['event_hour']
        if event is not None:
            event_count += 1
            # Only alerts in the prediction horizon count as event detection.
            hits = [t for t, s in enumerate(p['risk']) if max(0, event - 6) <= t < event and s >= threshold]
            if hits:
                leads.append(event - hits[0])
        else:
            non_events += 1
            non_event_alerts += int(any(s >= threshold and keep for s, keep in zip(p['risk'], p['eligible'])))
    test_metrics.update(median_lead_hours=float(np.median(leads)) if leads else None,
                        detected_events=len(leads), total_events=event_count, lead_hours=leads,
                        event_sensitivity=len(leads) / max(event_count, 1),
                        non_event_patients=non_events, non_event_patients_alerted=non_event_alerts)
    # Patient-level bootstrap respects dependence among hourly windows.
    rng = np.random.default_rng(seed + 100)
    aucs = []
    for _ in range(400):
        sample = rng.choice(len(test['records']), len(test['records']), replace=True)
        y, s = [], []
        for i in sample:
            p = test['records'][i]
            k = np.array(p['eligible'])
            y.extend(np.array(p['labels'])[k]); s.extend(np.array(p['risk'])[k])
        if len(np.unique(y)) == 2:
            aucs.append(roc_auc_score(y, s))
    test_metrics['auroc_ci95'] = np.quantile(aucs, [.025, .975]).tolist()
    fpr, tpr, _ = roc_curve(test['y'], test['score'])
    bfpr, btpr, _ = roc_curve(test['y'], test['baseline'])
    summary = dict(seed=seed, source='Synthetic longitudinal records; no real patient data',
                   horizon_hours=6, features=FEATURES, state_names=STATE_NAMES,
                   cohort_size=len(patients), observation_count=len(raw),
                   split_sizes={k: len(v['records']) for k, v in by_split.items()},
                   split_dates={k: [v['records'][0]['admission'], v['records'][-1]['timestamps'][-1]] for k, v in by_split.items()},
                   threshold=threshold, metrics=test_metrics,
                   validation=metrics(val['y'], val['score'], threshold),
                   transition=transition.tolist(), means=scaler.inverse_transform(model.means_)[order].tolist(),
                   medians=medians.to_dict(), scaler_mean=scaler.mean_.tolist(), scaler_scale=scaler.scale_.tolist(),
                   training=dict(log_likelihood=train_ll, iterations=model.monitor_.iter,
                                 converged=bool(model.monitor_.converged), restarts=3),
                   reference_check=dict(max_posterior_error=max_gamma_error, max_log_likelihood_error=max_ll_error),
                   roc=dict(fpr=fpr.tolist(), tpr=tpr.tolist(), baseline_fpr=bfpr.tolist(), baseline_tpr=btpr.tolist()),
                   patients=patients)
    for folder in ['data', 'artifacts']:
        (output / folder).mkdir(parents=True, exist_ok=True)
    raw.to_csv(output / 'data/synthetic_vitals.csv', index=False)
    (output / 'artifacts/results.json').write_text(json.dumps(summary, allow_nan=False))
    # Portable learned parameters, not an unsafe pickle.
    np.savez(output / 'artifacts/model.npz', start=model.startprob_, transition=model.transmat_,
             means=model.means_, variances=model._covars_, state_order=order,
             medians=medians.to_numpy(), scaler_mean=scaler.mean_, scaler_scale=scaler.scale_)
    return summary


if __name__ == '__main__':
    result = run_pipeline()
    print(json.dumps({k: result[k] for k in ['split_sizes', 'metrics', 'reference_check']}, indent=2))
