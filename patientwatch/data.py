"""Reproducible artificial trajectories. No real people or clinical records."""
import numpy as np
import pandas as pd

FEATURES = ['heart_rate', 'spo2', 'resp_rate', 'systolic_bp', 'temperature']
BOUNDS = [(30, 220), (65, 100), (4, 65), (50, 240), (33, 43)]
STATE_NAMES = ['Stable', 'Watch', 'Critical']


def generate_cohort(seed=42, patients=180, hours=48):
    rng = np.random.default_rng(seed)
    means = np.array([[75, 97.8, 16, 121, 36.8],
                      [95, 94.4, 23, 107, 37.6],
                      [119, 89.5, 31, 88, 38.5]])
    sd = np.array([7, 1.6, 2.8, 8, .32])
    transition = np.array([[.965, .035, 0], [.06, .90, .04], [.015, .045, .94]])
    rows = []
    for p in range(patients):
        state = int(rng.choice(2, p=[.84, .16]))
        offset = rng.normal(0, sd * .3)
        admission = pd.Timestamp('2025-01-01') + pd.Timedelta(days=3 * p)
        for hour in range(hours):
            if hour:
                state = int(rng.choice(3, p=transition[state]))
            values = means[state] + offset + rng.normal(0, sd)
            values[1] = min(values[1], 100)
            values[rng.random(5) < .075] = np.nan
            if rng.random() < .025:
                values[int(rng.integers(5))] = 999  # Known sensor corruption.
            rows.append([f'SYN-{p+1:03d}', str(admission + pd.Timedelta(hours=hour)),
                         hour, state, *values])
    return pd.DataFrame(rows, columns=['patient_id', 'timestamp', 'hour', 'true_state', *FEATURES])


def clean_sequences(df, medians=None):
    """Range checks + within-patient causal forward-fill; train-only fallback.

    No backwards filling/interpolation across future observations or patients.
    All raw sequences are hourly; missing cells do not invent timestamps.
    """
    df = df.sort_values(['patient_id', 'timestamp']).copy()
    for col, (lo, hi) in zip(FEATURES, BOUNDS):
        df.loc[~df[col].between(lo, hi), col] = np.nan
    if medians is None:
        medians = df[FEATURES].median()
    df[FEATURES] = df.groupby('patient_id')[FEATURES].ffill().fillna(medians)
    return df, medians


def chronological_split(df):
    admissions = df.groupby('patient_id').timestamp.min().sort_values()
    ids = admissions.index
    a, b = int(len(ids) * .6), int(len(ids) * .8)
    return [df[df.patient_id.isin(sub)].copy() for sub in (ids[:a], ids[a:b], ids[b:])]


def outcome_labels(group, horizon=6):
    """First two consecutive synthetic critical hours = artificial event.

    Only pre-event, observable-horizon windows are evaluated. Missing future
    follow-up is censored rather than incorrectly labelled a negative.
    """
    states = group.true_state.to_numpy()
    events = np.where((states[:-1] == 2) & (states[1:] == 2))[0]
    onset = int(events[0]) if len(events) else None
    times = np.arange(len(group))
    eligible = (times < onset) if onset is not None else (times + horizon < len(group))
    if onset is not None:
        eligible &= ((times + horizon < len(group)) | (times + horizon >= onset))
    labels = np.zeros(len(group), dtype=int) if onset is None else ((onset - times > 0) & (onset - times <= horizon)).astype(int)
    return labels, eligible, onset
