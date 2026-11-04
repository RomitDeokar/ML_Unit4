"""PatientWatch Streamlit research dashboard. Run: streamlit run streamlit_app.py."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve

from patientwatch.data import BOUNDS, FEATURES, STATE_NAMES
from patientwatch.enhancement import eligible_arrays

ROOT = Path(__file__).resolve().parent
COLORS = ['#14b8a6', '#f59e0b', '#f43f5e']
LABELS = {'heart_rate': ('Heart rate', 'bpm'), 'spo2': ('Oxygen saturation', '%'),
          'resp_rate': ('Respiratory rate', '/min'), 'systolic_bp': ('Systolic pressure', 'mmHg'),
          'temperature': ('Temperature', '°C')}
st.set_page_config(page_title='PatientWatch | Research Dashboard', layout='wide', initial_sidebar_state='auto')
st.markdown('''<style>
.stApp {background: #f5f7fb;}
[data-testid="stSidebar"] {background: #101e35;}
[data-testid="stSidebar"] * {color: #edf4ff;}
[data-testid="stSidebar"] [data-baseweb="select"] * {color: #152238;}
[data-testid="stSidebar"] input {color: #152238;}
[data-testid="stSidebar"] [data-testid="stLinkButton"] a,
[data-testid="stSidebar"] [data-testid="stLinkButton"] a * {color: #152238;}
[data-testid="stMetric"] {background:white;border:1px solid #e1e7f0;border-radius:12px;padding:16px;}
h1,h2,h3 {letter-spacing:-.035em; color:#142640;}
.block-container {max-width:1500px;padding-top:4.5rem;}
.eyebrow {color:#0f8b82;font-size:12px;font-weight:700;letter-spacing:2px;}
</style>''', unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_artifacts(source_bytes, enhancement_bytes):
    data = json.loads(source_bytes)
    extra = json.loads(enhancement_bytes)
    if extra['source_sha256'] != hashlib.sha256(source_bytes).hexdigest():
        raise ValueError('Enhancement belongs to a different experiment. Run python -m patientwatch.enhancement.')
    return data, extra


def figure(title='', y_title=''):
    fig = go.Figure()
    fig.update_layout(title=dict(text=title, font=dict(size=14)), template='plotly_white', height=365,
                      margin=dict(l=20, r=20, t=50, b=90), paper_bgcolor='white',
                      font=dict(family='Arial', color='#243751'),
                      xaxis_title='Observation hour', yaxis_title=y_title,
                      legend=dict(orientation='h', y=-.25, x=0, yanchor='top', font=dict(size=11)),
                      hovermode='x unified')
    return fig


def plot(fig):
    st.plotly_chart(fig, width='stretch', config={'displaylogo': False})


def export_patient(p, scores):
    frame = pd.DataFrame({'hour': p['hours'], 'timestamp': p['timestamps'], **p['vitals'],
                          'original_score': p['risk'], 'enhanced_score': scores,
                          'evaluation_eligible': p['eligible']})
    for i, name in enumerate(STATE_NAMES):
        frame[f'filtered_{name.lower()}'] = np.asarray(p['filtered'])[:, i]
        frame[f'retrospective_smoothed_{name.lower()}'] = np.asarray(p['smoothed'])[:, i]
    frame['retrospective_viterbi_state'] = [STATE_NAMES[i] for i in p['decoded']]
    return frame.to_csv(index=False).encode()


try:
    data, extra = load_artifacts((ROOT / 'artifacts/results.json').read_bytes(),
                                 (ROOT / 'artifacts/enhancement.json').read_bytes())
except (OSError, ValueError, KeyError) as exc:
    st.error(f'Experiment artifacts are unavailable or inconsistent: {exc}')
    st.info('From the repository root run: python -m patientwatch.pipeline, then python -m patientwatch.enhancement.')
    st.stop()

with st.sidebar:
    st.title('PatientWatch')
    st.caption('SEQUENTIAL HEALTH INTELLIGENCE')
    st.divider()
    page = st.radio('Workspace', ['Patient monitor', 'Model comparison', 'Methods & data', 'Submission files'])
    st.divider()
    st.caption('UNIT 4 / PROJECT 03')
    st.write('Gaussian HMM · 3 latent states')
    st.caption('180 artificial admissions · 48 hours each')
    st.link_button('View GitHub repository', 'https://github.com/RomitDeokar/ML_Unit4')
    st.caption('Temporary research demonstration. No patient uploads or real clinical data.')

st.markdown('<div class="eyebrow">PATIENTWATCH / RESEARCH WORKSPACE</div>', unsafe_allow_html=True)
st.title(page)
st.warning('Synthetic data only. Not a medical device. Do not use for diagnosis, triage, treatment or patient care.')

if page == 'Patient monitor':
    st.caption('Explore historical artificial observations. Scores use only information available up to each hour.')
    a, b, c = st.columns([1, 1, 2])
    split = a.selectbox('Cohort', ['test', 'validation', 'train'])
    state_filter = b.selectbox('Latent-state filter', ['All states'] + STATE_NAMES)
    search = c.text_input('Search patient ID', placeholder='For example, SYN-145').strip().lower()
    hour = st.slider('Observation hour', 0, 47, 24, key='observation_hour')
    cohort = [p for p in data['patients'] if p['split'] == split]
    counts = np.bincount([np.argmax(p['filtered'][hour]) for p in cohort], minlength=3)
    for col, name, n in zip(st.columns(3), STATE_NAMES, counts):
        col.metric(f'{name} · current latent regime', int(n))
    matches = [p for p in cohort if search in p['id'].lower() and
               (state_filter == 'All states' or STATE_NAMES[int(np.argmax(p['filtered'][hour]))] == state_filter)]
    if not matches:
        st.info('No patients match these filters. Clear the search or choose All states.')
    else:
        patient_id = st.selectbox('Patient', [p['id'] for p in matches])
        p = next(p for p in matches if p['id'] == patient_id)
        model_name = st.radio('Alert scoring model', ['Enhanced HMM head', 'Original HMM'], horizontal=True)
        enhanced = model_name == 'Enhanced HMM head'
        scores = extra['scores'][p['id']] if enhanced else p['risk']
        threshold = extra['threshold'] if enhanced else data['threshold']
        alpha = np.asarray(p['filtered'])
        state = int(alpha[hour].argmax())
        cols = st.columns(4)
        cols[0].metric('Selected patient', p['id'])
        cols[1].metric('Filtered latent state', STATE_NAMES[state])
        cols[2].metric('Six-hour alert score', f'{scores[hour]:.3f}')
        cols[3].metric('Frozen validation threshold', f'{threshold:.4f}')
        if not p['eligible'][hour]:
            st.info('Not an evaluated early-warning window: artificial event already started or future follow-up is censored. Score is descriptive only.')
        elif scores[hour] >= threshold:
            st.warning('Research alert: score exceeds the validation-selected threshold. This is not a clinical recommendation.')
        else:
            st.success('Below research alert threshold. This does not establish clinical safety.')
        st.caption('Latent states are learned statistical regimes, not diagnoses. Enhanced scores are supervised estimates of the artificial endpoint, not calibrated clinical risk.')
        selected_vitals = st.multiselect('Vital-sign charts', FEATURES, default=['heart_rate', 'spo2'],
                                         format_func=lambda f: LABELS[f][0])
        show_raw = st.checkbox('Overlay valid raw observations', value=True)
        chart_cols = st.columns(2)
        for i, vital in enumerate(selected_vitals):
            with chart_cols[i % 2]:
                label, unit = LABELS[vital]
                fig = figure(label, unit)
                fig.add_trace(go.Scatter(x=p['hours'][:hour+1], y=p['vitals'][vital][:hour+1],
                                         name='Causally cleaned', line=dict(color='#0f8b82', width=3)))
                if show_raw:
                    lo, hi = BOUNDS[FEATURES.index(vital)]
                    raw = [v if v is not None and lo <= v <= hi else None for v in p['raw_vitals'][vital][:hour+1]]
                    fig.add_trace(go.Scatter(x=p['hours'][:hour+1], y=raw, mode='markers',
                                             name='Valid raw', marker=dict(color='#7487a2', size=5)))
                plot(fig)
        left, right = st.columns(2)
        with left:
            fig = figure('Causal filtered state probabilities', 'Probability')
            for i, name in enumerate(STATE_NAMES):
                fig.add_trace(go.Scatter(x=p['hours'][:hour+1], y=alpha[:hour+1, i], name=name,
                                         stackgroup='one', line=dict(color=COLORS[i], width=1)))
            fig.update_yaxes(range=[0, 1])
            plot(fig)
        with right:
            fig = figure('Six-hour early-warning score', 'Score (not clinical risk)')
            fig.add_trace(go.Scatter(x=p['hours'][:hour+1], y=scores[:hour+1], name=model_name,
                                     line=dict(color='#4f46e5', width=3)))
            fig.add_hline(y=threshold, line_dash='dash', line_color='#f43f5e', annotation_text='Frozen threshold')
            fig.update_yaxes(range=[0, 1])
            plot(fig)
        with st.expander('Retrospective analysis · uses future observations'):
            st.warning('Smoothing and Viterbi below use the complete 48-hour sequence and must not drive early-warning scores.')
            fig = figure('Full-sequence Forward–Backward smoothing', 'Probability')
            for i, name in enumerate(STATE_NAMES):
                fig.add_trace(go.Scatter(x=p['hours'], y=np.asarray(p['smoothed'])[:, i], name=name,
                                         line=dict(color=COLORS[i])))
            if p['event_hour'] is not None:
                fig.add_vline(x=p['event_hour'], line_dash='dash', annotation_text='Artificial event onset')
            plot(fig)
            fig = figure('Viterbi / simulator · retrospective', 'State')
            fig.add_trace(go.Scatter(x=p['hours'], y=p['decoded'], name='Viterbi', line=dict(shape='hv', color='#4f46e5')))
            fig.add_trace(go.Scatter(x=p['hours'], y=p['true_state'], name='Simulator truth (evaluation only)',
                                     line=dict(shape='hv', dash='dot', color='#7487a2')))
            fig.update_yaxes(tickvals=[0, 1, 2], ticktext=STATE_NAMES)
            plot(fig)
        st.download_button('Download patient research CSV', export_patient(p, extra['scores'][p['id']]),
                           file_name=f'{p["id"]}_research.csv', mime='text/csv')

elif page == 'Model comparison':
    st.caption('Published synthetic test cohort · 36 patients · 1,293 eligible hourly windows · 72 positive windows')
    m, old = extra['test'], extra['original_test']
    cols = st.columns(4)
    cols[0].metric('Enhanced AUROC', f'{m["auroc"]:.4f}', f'{m["auroc"] - old["auroc"]:+.4f}')
    cols[1].metric('Window sensitivity', f'{m["sensitivity"]:.1%}', f'{100*(m["sensitivity"]-old["sensitivity"]):+.2f} pp')
    cols[2].metric('Window specificity', f'{m["specificity"]:.1%}', f'{100*(m["specificity"]-old["specificity"]):+.2f} pp')
    cols[3].metric('Median event lead time', f'{m["median_lead_hours"]:g} h')
    st.warning('Trade-off: the selected head has higher AUROC and sensitivity, but lower specificity and more non-event patients alerted. It is not better on every metric.')
    ci = extra['bootstrap']['paired_auroc_delta_ci95']
    st.info(f'Paired patient-bootstrap AUROC difference 95% interval: [{ci[0]:+.4f}, {ci[1]:+.4f}]. Includes zero: no established superiority. The published test set is exploratory, not new external validation.')
    models = {'Original HMM': old, 'Enhanced HMM head': m, 'Current-vitals logistic': extra['baseline_test']}
    table = []
    for name, metric in models.items():
        table.append({'Model': name, 'AUROC': metric['auroc'], 'Average precision': metric['average_precision'],
                      'Sensitivity': metric['sensitivity'], 'Specificity': metric['specificity'],
                      'Brier (lower is better)': metric['brier'], 'Threshold': metric['threshold'],
                      'Events detected': f'{metric["detected_events"]}/{metric["total_events"]}',
                      'Non-event patients alerted': f'{metric["non_event_patients_alerted"]}/{metric["non_event_patients"]}'})
    st.dataframe(pd.DataFrame(table), hide_index=True, width='stretch')
    patients = [p for p in data['patients'] if p['split'] == 'test']
    maps = {'Original HMM': {p['id']: p['risk'] for p in patients}, 'Enhanced HMM head': extra['scores'],
            'Current-vitals logistic': {p['id']: p['baseline'] for p in patients}}
    left, right = st.columns(2)
    with left:
        fig = figure('Receiver operating characteristic')
        for name, metric in models.items():
            fig.add_trace(go.Scatter(x=metric['roc']['fpr'], y=metric['roc']['tpr'], name=name))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name='Chance', line=dict(dash='dash', color='#cbd5e1')))
        fig.update_layout(xaxis_title='False positive rate', yaxis_title='True positive rate')
        plot(fig)
    with right:
        fig = figure('Precision–recall · rare-event evaluation')
        for name, score_map in maps.items():
            y, score = eligible_arrays(patients, score_map)
            precision, recall, _ = precision_recall_curve(y, score)
            fig.add_trace(go.Scatter(x=recall, y=precision, name=name))
        fig.add_hline(y=m['positives']/m['windows'], line_dash='dash', annotation_text='Event-window prevalence')
        fig.update_layout(xaxis_title='Recall', yaxis_title='Precision')
        plot(fig)
    left, right = st.columns(2)
    with left:
        c = m['confusion']
        fig = figure('Enhanced head · frozen-threshold confusion matrix')
        fig.add_trace(go.Heatmap(z=[[c['tn'], c['fp']], [c['fn'], c['tp']]],
                                 x=['No alert', 'Alert'], y=['Negative window', 'Positive window'],
                                 texttemplate='%{z}', colorscale='Teal', showscale=False))
        fig.update_layout(xaxis_title='Prediction', yaxis_title='Artificial outcome')
        plot(fig)
    with right:
        fig = figure('Score reliability · descriptive only')
        for name, score_map in maps.items():
            y, score = eligible_arrays(patients, score_map)
            observed, predicted = calibration_curve(y, score, n_bins=6, strategy='quantile')
            fig.add_trace(go.Scatter(x=predicted, y=observed, name=name, mode='lines+markers'))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name='Ideal', line=dict(dash='dash', color='#cbd5e1')))
        fig.update_layout(xaxis_title='Mean predicted score', yaxis_title='Observed positive fraction')
        plot(fig)
    with st.expander('Validation-only model selection and reproducible parameters'):
        st.write(extra['selection'])
        st.write('Selected:', extra['selected']['name'])
        st.dataframe(pd.DataFrame(extra['candidates']), hide_index=True, width='stretch')
        st.json(extra['parameters'])

elif page == 'Methods & data':
    st.subheader('Sequential learning, without future leakage')
    st.markdown('''1. **Artificial cohort:** 180 admissions, five vital signs and 48 hourly observations each.
2. **Chronological patient split:** 108 training / 36 validation / 36 test; no admission overlap.
3. **Causal cleaning:** range invalidation → within-patient forward fill → training-median fallback. Training-only standardization.
4. **Unsupervised HMM:** three diagonal Gaussian regimes, maximum-likelihood Baum–Welch, three training-likelihood restarts.
5. **Custom inference:** scaled Forward–Backward (chain sum-product) and log-domain Viterbi, checked against hmmlearn.
6. **Original score:** six-hour critical-state reachability. **Enhanced head:** supervised logistic regression on causal filtered probabilities. Temporal features were also tested, but did not win validation selection.
7. **Artificial endpoint:** first pair of consecutive simulator-critical hours. Only pre-onset, observable windows are evaluated; outcomes near sequence end are censored.
8. **Evaluation:** validation-only model selection and Youden-J thresholds; frozen test evaluation with 400 paired patient-bootstrap replicates.''')
    st.caption('Enhanced features use stored rounded vitals/posteriors for reproducibility. Simulator labels define training/evaluation outcomes only; they are never predictor inputs. Original notebook, report and slides describe the original HMM, not this extension.')
    a, b = st.columns(2)
    a.metric('Maximum posterior reference error', f'{data["reference_check"]["max_posterior_error"]:.2e}')
    b.metric('Maximum log-likelihood reference error', f'{data["reference_check"]["max_log_likelihood_error"]:.2e}')
    st.subheader('Learned state-transition matrix')
    fig = figure()
    fig.add_trace(go.Heatmap(z=data['transition'], x=STATE_NAMES, y=STATE_NAMES,
                             texttemplate='%{z:.3f}', colorscale='Teal', zmin=0, zmax=1))
    fig.update_layout(xaxis_title='Next state', yaxis_title='Current state')
    plot(fig)
    st.subheader('Learned emission means · original units')
    st.dataframe(pd.DataFrame(data['means'], index=STATE_NAMES, columns=FEATURES), width='stretch')
    st.subheader('Raw-data quality audit')
    quality = []
    for f, (lo, hi) in zip(FEATURES, BOUNDS):
        values = np.array([np.nan if v is None else v for p in data['patients'] for v in p['raw_vitals'][f]])
        quality.append({'Vital sign': LABELS[f][0], 'Missing cells': int(np.isnan(values).sum()),
                        'Invalid non-missing cells': int(((values < lo) | (values > hi)).sum()),
                        'Valid range': f'{lo}–{hi}', 'Observations': len(values)})
    st.dataframe(pd.DataFrame(quality), hide_index=True, width='stretch')
    st.subheader('Limitations to explain in your presentation')
    st.write('Synthetic generation resembles the HMM assumptions. Small cohort, one simulation seed, no external validation, no real clinical endpoint or treatment effects. High event recall coexists with substantial false-alert burden. AUROC improvement is small and uncertain. Real MIMIC-IV work requires credentialing and approved data handling.')
    st.code('OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m patientwatch.enhancement\npython -m pytest -q\nstreamlit run streamlit_app.py', language='bash')

else:
    st.caption('Original academic submission plus the separately versioned model extension. Downloads contain artificial data only.')
    downloads = [
        ('Executed project notebook', 'notebooks/PatientWatch_Project.ipynb', 'Original end-to-end HMM experiment'),
        ('Technical report', 'reports/technical_report.pdf', 'Original HMM results · PDF'),
        ('Presentation', 'reports/PatientWatch_Presentation.pptx', 'Original HMM results · 14 slides'),
        ('Synthetic dataset', 'data/synthetic_vitals.csv', 'Raw observations with missing and corrupt readings'),
        ('HMM parameters', 'artifacts/model.npz', 'Portable NumPy arrays; no pickle'),
        ('Original experiment results', 'artifacts/results.json', 'Full-precision predictions and reference checks'),
        ('Model extension results', 'artifacts/enhancement.json', 'Validation candidates, head parameters, predictions and trade-offs'),
    ]
    for title, relative, description in downloads:
        with st.container(border=True):
            left, right = st.columns([4, 1])
            left.markdown(f'**{title}**')
            left.caption(description)
            path = ROOT / relative
            if path.is_file():
                right.download_button('Download', path.read_bytes(), file_name=path.name, key=relative)
            else:
                right.error('File missing')
    st.info('To publish on Streamlit Community Cloud, select this GitHub branch (or main after merging), entrypoint streamlit_app.py, and Python 3.13. No API keys or credentials are needed. The current sandbox link is temporary, not a permanent streamlit.app deployment.')

st.divider()
st.caption('PatientWatch · Unit 4 / Project 3 · Synthetic-data research demo · Not for clinical use')
