"""Build and execute the notebook; generate report/slides from actual results."""
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for key, folder in [('MPLCONFIGDIR', '.cache/matplotlib'), ('IPYTHONDIR', '.cache/ipython'), ('JUPYTER_RUNTIME_DIR', '.cache/jupyter')]:
    os.environ.setdefault(key, str(ROOT / folder))
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import json
import inspect
import numpy as np
import nbformat as nbf
from nbclient import NotebookClient
from jupyter_client import KernelManager
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image
from reportlab.lib.pagesizes import A4
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE


def build_notebook():
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    cells = [md('''# PatientWatch: Smart Healthcare Patient Deterioration Monitoring
## Unit 4 · Project 3 · Reproducible research notebook
**Synthetic data only. Educational prototype, not a medical device or clinical decision tool.**

Workflow: dataset → preprocessing → sequence construction → model formulation → algorithms → inference → evaluation → visualization → interpretation.

No MIMIC-IV credentials, patient records or external downloads are required. Run all cells from a cloned repository with `requirements.txt` installed. Ground truth is a simulator regime, not a real clinical outcome.'''),
code('''from pathlib import Path
import os, sys, json, inspect
ROOT = Path.cwd() if (Path.cwd() / 'patientwatch').exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from patientwatch.data import generate_cohort, chronological_split, clean_sequences, outcome_labels, FEATURES
from patientwatch.inference import forward_backward, viterbi, emissions, horizon_risk
from patientwatch.pipeline import run_pipeline
SEED = 42
plt.rcParams.update({'figure.figsize': (10, 3.5), 'axes.spines.top': False, 'axes.spines.right': False})
print('Python:', sys.version.split()[0], '| seed:', SEED)'''),
md('''## 1. Synthetic dataset and quality assessment
180 patient-disjoint artificial admissions × 48 hourly observations = 8,640 rows. A stationary three-regime Markov simulator produces correlated sequences of five vital signs with patient-level offsets. It injects 7.5% cell missingness and occasional 999 sensor corruption. Admission spacing is three days, preventing temporal overlap across splits. All identities, dates and outcomes are artificial.'''),
code('''raw = generate_cohort(SEED)
display(raw.head())
print('Shape:', raw.shape, '| patients:', raw.patient_id.nunique())
display(pd.DataFrame({'missing_percent': raw[FEATURES].isna().mean()*100,
                      'sentinel_count': (raw[FEATURES] == 999).sum()}))'''),
md('''## 2. Chronological splitting, causal cleaning and sequence construction
First 60% of admissions train; next 20% validate; final 20% test. A patient belongs to one split only. We invalidate out-of-range values, forward-fill **within each sequence**, and fill leading gaps from training medians. Standardization is fitted on training only. No backward fill, future interpolation or cross-patient carry-over is allowed. HMM sequence lengths reset inference at admission boundaries.'''),
code('''train_raw, val_raw, test_raw = chronological_split(raw)
train, medians = clean_sequences(train_raw)
val, _ = clean_sequences(val_raw, medians)
test, _ = clean_sequences(test_raw, medians)
for name, frame in [('train',train),('validation',val),('test',test)]:
    print(name, frame.patient_id.nunique(), 'admissions;', frame.timestamp.min(), 'to', frame.timestamp.max())
assert train.timestamp.max() < val.timestamp.min() < test.timestamp.min()
assert not set(train.patient_id) & set(test.patient_id)
assert train[FEATURES].notna().all().all()
display(medians.to_frame('train_only_fallback'))'''),
md(r'''## 3. Model formulation and maximum likelihood
Let $z_t\in\{0,1,2\}$ be an unobserved regime and $x_t\in\mathbb{R}^5$ standardized vital signs.

$$P(z_{0:T},x_{0:T})=\pi_{z_0}\prod_{t=1}^T A_{z_{t-1},z_t}\prod_{t=0}^T\mathcal{N}(x_t;\mu_{z_t},\mathrm{diag}(\sigma^2_{z_t})).$$

Baum–Welch EM in `hmmlearn` maximizes training likelihood. Three seed-fixed restarts are compared using **training likelihood only**. State count is fixed at three before evaluation. Learned means are ranked by standardized HR − SpO₂ + RR − SBP + temperature. The simulator's true state is **not** used for training or state alignment.

## 4. Core algorithms implemented from scratch
The HMM is a chain factor graph; Forward–Backward is sum-product message passing. Shift log emission rows by their maxima before exponentiation. Normalize each forward row with scale $c_t$. Scale backward messages by $c_{t+1}$; recover likelihood by adding log scales and removed emission shifts.

- Filter: $\alpha_t \propto (\alpha_{t-1} A)\odot b(x_t)$, using only observations up to $t$.
- Backward: $\beta_t=A[b(x_{t+1})\odot\beta_{t+1}]/c_{t+1}$.
- Smooth: $\gamma_t\propto\alpha_t\odot\beta_t$, using the entire sequence.
- Viterbi: log-domain max-product with backpointers; retrospective only.

The following displayed source is the actual from-scratch implementation used in the experiment (no reference-inference calls inside these functions).'''),
code('''print(inspect.getsource(forward_backward))
print(inspect.getsource(viterbi))'''),
md('''## 5. Train and run the full experiment
This cell regenerates every record, fits the scaler and HMM, fits a logistic-regression baseline on current vitals, freezes thresholds on validation, and evaluates test data. It also compares our smoothed posterior and log likelihood to `hmmlearn` on every patient.'''),
code('''result = run_pipeline(seed=SEED, output=ROOT)
print(json.dumps(result['training'], indent=2))
print(json.dumps(result['reference_check'], indent=2))
assert result['reference_check']['max_posterior_error'] < 1e-10
assert result['reference_check']['max_log_likelihood_error'] < 1e-9
display(pd.DataFrame(result['transition'], index=result['state_names'], columns=result['state_names']))
display(pd.DataFrame(result['means'], index=result['state_names'], columns=FEATURES))'''),
md('''## 6. Retrospective early-warning protocol (causal scores)
A synthetic event is the first occurrence of **two consecutive simulator-critical hours**, timestamped at the first hour. Eligible windows are strictly before that event and have an observable six-hour outcome. Negative windows with fewer than six remaining hours are censored. After the first event, all windows are excluded.

The model score is the probability of currently occupying or reaching the learned critical regime within six transitions, computed from **filtered** alpha by making critical absorbing. This is a surrogate for the sustained synthetic event, not its exact probability and not calibrated clinical risk.

Threshold maximizes Youden J on validation only. AUROC, sensitivity and specificity are window-level; lead time is event-level. An event is detected only by an alert in the preceding six hours. Report the median only among detected events. Compare against current-vitals logistic regression with the same data and its own validation threshold. Bootstrap whole test patients (400 replicates), not individual hours.'''),
code('''m = result['metrics']
comparison = pd.DataFrame({
    'Gaussian HMM': {k:m[k] for k in ['auroc','sensitivity','specificity','brier','threshold']},
    'Logistic regression': {k:m['baseline'][k] for k in ['auroc','sensitivity','specificity','brier','threshold']}})
display(comparison)
print('Confusion:', m['confusion'])
print('AUROC 95% patient-bootstrap CI:', m['auroc_ci95'])
print('Detected events:', m['detected_events'], '/', m['total_events'])
print('Median lead hours (detected only):', m['median_lead_hours'])
print('Non-event patients alerted:', m['non_event_patients_alerted'], '/', m['non_event_patients'])'''),
md('''## 7. Visualization: filtering vs. smoothing, vital trajectory and ROC
Retrospective smoothing may appear to anticipate changes because it uses future evidence. It must never be substituted for filtered scores in an early-warning evaluation. Event markers are simulator ground truth shown for retrospective interpretation.'''),
code('''p = next(p for p in result['patients'] if p['split']=='test' and p['event_hour'] is not None and p['event_hour'] > 20)
fig, axes = plt.subplots(3,1,figsize=(11,8),sharex=True)
axes[0].plot(p['hours'],p['vitals']['heart_rate'],color='#138477')
axes[0].set_ylabel('HR (bpm)')
for k,name in enumerate(result['state_names']):
    axes[1].plot(p['hours'],np.array(p['filtered'])[:,k],label=name)
    axes[2].plot(p['hours'],np.array(p['smoothed'])[:,k],label=name)
axes[1].set_ylabel('Filtered probability');axes[2].set_ylabel('Smoothed probability')
for ax in axes:
    ax.axvline(p['event_hour'],color='#db8278',linestyle='--',label='Synthetic event')
    ax.grid(alpha=.15)
axes[1].legend(ncol=4,fontsize=8)
axes[0].set_title(p['id']+' · entirely synthetic')
axes[-1].set_xlabel('Hours since admission');plt.tight_layout();plt.show()
fig,ax=plt.subplots()
r=result['roc']
ax.plot(r['fpr'],r['tpr'],label=f"HMM AUROC {m['auroc']:.3f}",color='#138477')
ax.plot(r['baseline_fpr'],r['baseline_tpr'],label=f"Baseline AUROC {m['baseline']['auroc']:.3f}",color='#d7aa52')
ax.plot([0,1],[0,1],'--',color='lightgray');ax.set(xlabel='False-positive rate',ylabel='Sensitivity',title='Held-out synthetic benchmark')
ax.legend();plt.tight_layout();plt.show()'''),
md('''## 8. Causality check and independent reference comparison
Change future observations: filtered posteriors in the prefix must not change. Compare full-sequence Viterbi with the reference library using the saved learned parameters.'''),
code('''from hmmlearn.hmm import GaussianHMM
params=np.load(ROOT/'artifacts/model.npz')
model=GaussianHMM(n_components=3,covariance_type='diag',implementation='scaling',init_params='')
model.startprob_=params['start'];model.transmat_=params['transition']
model.means_=params['means'];model.covars_=params['variances']
x=(test[test.patient_id==p['id']][FEATURES].to_numpy()-params['scaler_mean'])/params['scaler_scale']
log_b=emissions(x,params['means'],params['variances'])
a,g,ll,_=forward_backward(log_b,params['start'],params['transition'])
changed=log_b.copy();changed[25:]=0
new_a,_,_,_=forward_backward(changed,params['start'],params['transition'])
assert np.allclose(a[:25],new_a[:25])
assert np.array_equal(viterbi(log_b,params['start'],params['transition']),model.predict(x))
print('PASS: causal prefix is invariant; Viterbi path equals reference.')'''),
md('''## 9. Interpretation, limitations and next steps
The synthetic HMM gives moderate discrimination, not a perfect early-warning system. Its small AUROC advantage over the non-sequential baseline does not establish statistical superiority. High event recall coexists with substantial false-alert burden. Event sensitivity differs from hourly sensitivity, and lead-time summaries exclude missed events.

The generator is stationary and close to a Gaussian HMM; this favors the model. Missingness is mostly random, not reflective of clinical measurement practice. There are no treatments, demographic subgroups, irregular samples or independent event adjudication. A small test cohort and one seed limit conclusions. Absolute scores are not clinically calibrated.

A real extension requires credentialed MIMIC-IV access, approved data handling, clinically grounded endpoints, temporal/external validation, calibration, subgroup audits, missingness modeling and clinician review. Never deploy this prototype for diagnosis or triage.

## References
- Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition. *Proceedings of the IEEE*, 77(2), 257–286. doi:10.1109/5.18626.
- `hmmlearn` documentation: https://hmmlearn.readthedocs.io/
- `scikit-learn` model evaluation: https://scikit-learn.org/stable/modules/model_evaluation.html
- MIMIC-IV access requirements (not used): https://physionet.org/content/mimiciv/

## Demonstration
Run `python app.py`, visit `http://localhost:3000`, explore patient timelines, switch filtering/smoothing, inspect performance and download files. Full experiment source is in `patientwatch/`.''')]
    nb = nbf.v4.new_notebook(cells=cells, metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
    km = KernelManager(kernel_name='python3')
    km.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
    NotebookClient(nb, timeout=240, km=km, resources={'metadata':{'path':str(ROOT)}}).execute()
    nbf.write(nb, ROOT/'notebooks/PatientWatch_Project.ipynb')
    print('Executed notebook:', len(cells), 'cells')


def figures(r):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    m=r['metrics']; roc=r['roc']
    fig,ax=plt.subplots(figsize=(7,3.8))
    ax.plot(roc['fpr'],roc['tpr'],color='#138477',lw=2,label=f"Gaussian HMM · AUROC {m['auroc']:.3f}")
    ax.plot(roc['baseline_fpr'],roc['baseline_tpr'],color='#d4a44e',lw=1.8,label=f"Logistic baseline · AUROC {m['baseline']['auroc']:.3f}")
    ax.plot([0,1],[0,1],'--',color='#d5dcd9');ax.set(xlabel='False-positive rate',ylabel='Sensitivity',title='Held-out synthetic test windows')
    ax.legend(frameon=False,fontsize=9);ax.grid(alpha=.12);fig.tight_layout();fig.savefig(ROOT/'reports/roc.png',dpi=180);plt.close(fig)
    p=next(p for p in r['patients'] if p['split']=='test' and p['event_hour'] is not None and p['event_hour']>20)
    fig,axes=plt.subplots(2,1,figsize=(9,4.3),sharex=True)
    axes[0].plot(p['hours'],p['vitals']['heart_rate'],color='#138477',lw=1.7);axes[0].set_ylabel('Heart rate (bpm)')
    for k,c in enumerate(['#138477','#d4a44e','#de817b']):axes[1].plot(p['hours'],np.array(p['filtered'])[:,k],label=r['state_names'][k],color=c)
    for ax in axes:ax.axvline(p['event_hour'],ls='--',color='#de817b');ax.grid(alpha=.15)
    axes[0].set_title(p['id']+' · synthetic trajectory; dashed line = artificial event')
    axes[1].set(xlabel='Hours since admission',ylabel='Filtered probability');axes[1].legend(ncol=3,fontsize=8,frameon=False)
    fig.tight_layout();fig.savefig(ROOT/'reports/trajectory.png',dpi=180);plt.close(fig)


def build_report(r):
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='BodyCustom',fontName='Helvetica',fontSize=10,leading=15,textColor=colors.HexColor('#53655f'),spaceAfter=9))
    styles['Title'].textColor=colors.HexColor('#156f60');styles['Title'].fontSize=27;styles['Title'].leading=33
    styles['Heading1'].textColor=colors.HexColor('#156f60');styles['Heading1'].fontSize=16;styles['Heading1'].spaceAfter=14
    body=styles['BodyCustom']; story=[]
    def p(text,style=body):story.append(Paragraph(text,style))
    def heading(text):p(text,styles['Heading1'])
    def table(rows,widths=None):
        t=Table(rows,colWidths=widths,hAlign='LEFT');t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#eaf3ee')),('TEXTCOLOR',(0,0),(-1,-1),colors.HexColor('#416252')),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),9),('TOPPADDING',(0,0),(-1,-1),9),('BOTTOMPADDING',(0,0),(-1,-1),9),('LINEBELOW',(0,0),(-1,-1),.4,colors.HexColor('#e4ebe6'))]));story.append(t);story.append(Spacer(1,12))
    m=r['metrics'];b=m['baseline'];ref=r['reference_check']
    p('UNIT 4 · PROJECT 03 · TECHNICAL REPORT',styles['Heading2'])
    p('PatientWatch',styles['Title'])
    p('Smart Healthcare Patient Deterioration Monitoring and Early Warning System',styles['Heading2'])
    p('<b>Research and education only. All records are synthetic. Not for diagnosis, triage, treatment or clinical monitoring.</b>')
    heading('1. Abstract and objective')
    p('Repeated vital signs can reveal evolving latent health conditions that are difficult to interpret from isolated readings. We implement an end-to-end prototype using a three-state Gaussian hidden Markov model (HMM), causal filtering, retrospective Forward–Backward smoothing, and Viterbi decoding. A read-only dashboard visualizes trajectories and research alert scores. This submission uses the synthetic-data option explicitly permitted by the assignment; no MIMIC-IV data or real patient records are used.')
    p(f"On 36 chronologically held-out synthetic patients, window-level AUROC is {m['auroc']:.3f}, sensitivity {m['sensitivity']:.1%}, specificity {m['specificity']:.1%}, and median lead time {m['median_lead_hours']:.1f} hours among detected events. These numbers do not demonstrate clinical usefulness. All {m['total_events']} test events were detected within the chosen horizon, while {m['non_event_patients_alerted']}/{m['non_event_patients']} non-event patients received an alert.")
    heading('2. Dataset and sequence construction')
    p('The seed-42 simulator creates 180 patient admissions with 48 hourly measurements each (8,640 rows). Features are heart rate (bpm), oxygen saturation (%), respiratory rate (/min), systolic blood pressure (mmHg), and temperature (C). A three-state stationary Markov process generates stable, watch and critical regimes. Gaussian observation noise and patient-specific offsets vary measurements. Cell missingness is 7.5%; occasional 999 values emulate sensor corruption. Each patient has a new ID and admission date, spaced three days apart. The generator and raw CSV are included.')
    table([['Split','Admissions','Purpose'],['Training','108 (60%)','Preprocessing, HMM, baseline'],['Validation','36 (20%)','Freeze alert thresholds'],['Test','36 (20%)','Final metrics and interpretation']],[110,110,270])
    story.append(PageBreak())
    heading('3. Cleaning and leakage prevention')
    p('Range checks invalidate HR outside 30–220, SpO2 outside 65–100, respiratory rate outside 4–65, SBP outside 50–240, and temperature outside 33–43. Missing or invalid cells are forward-filled within patient only. Leading gaps use training medians. Training-fitted standardization is applied unchanged to validation and test. There is no backward fill or future interpolation. Lengths passed to the HMM enforce independent sequences; messages never cross patient boundaries. Admissions and their observations are chronologically ordered with no split overlap.')
    heading('4. Model formulation and estimation')
    p('For latent regime z_t and standardized vital vector x_t, the joint density is pi(z_0) times the product of transition factors A(z_(t-1),z_t) and emission factors N(x_t; mu_(z_t), diag(var_(z_t))). There are three states with diagonal Gaussian emissions. Parameters are learned by Baum–Welch expectation-maximization (maximum likelihood), using hmmlearn with 150 maximum iterations, tolerance 0.001, and three deterministic restarts. The restart with greatest training likelihood is retained. State count and severity interpretation are fixed before test evaluation.')
    p('To address label permutation, states are ranked by learned standardized emission means: HR - SpO2 + respiratory rate - SBP + temperature. The resulting stable/watch/critical names are interpretive, not clinical diagnoses. Simulator labels are used solely to define evaluation outcomes, never to train the HMM or map state names.')
    heading('5. Core algorithm: scaled Forward–Backward')
    p('Implemented from scratch in patientwatch/inference.py. Let log_b(t,k) be the log Gaussian density and s_t its row maximum. Set b_t = exp(log_b_t - s_t). Initialize alpha_0 = pi * b_0 and normalize by c_0. Recur alpha_t = (alpha_(t-1) A) * b_t / c_t, where c_t is the unnormalized row sum. This filter uses only past and present observations. Initialize beta_(T-1) = 1 and recurse beta_t = A[b_(t+1) * beta_(t+1)] / c_(t+1). Normalize gamma_t = alpha_t * beta_t to obtain a retrospective posterior. Restore log likelihood as sum_t(log c_t + s_t). Complexity is O(T K squared), with O(T K) storage. This is chain-factor-graph sum-product.')
    p('Viterbi is also implemented from scratch: maximize log transition plus previous best score for each state, add log emissions, save backpointers, then backtrack. It uses future sequence evidence in the final decoded path and is strictly retrospective. Neither smoothed gamma nor Viterbi paths enter early-warning scores.')
    p(f"Reference checks over all 180 sequences: max posterior error {ref['max_posterior_error']:.3e}; max log-likelihood error {ref['max_log_likelihood_error']:.3e}, relative to hmmlearn. Unit tests also compare against exhaustive enumeration on a tiny chain, check a 10,000-step sequence for numerical stability, and perturb future observations to verify causal prefix invariance.")
    story.append(PageBreak())
    heading('6. Alert protocol and outcome definition')
    p('A synthetic deterioration event is the first pair of consecutive simulator-critical hours, with onset at the first hour. Only windows strictly before the first event are scored. A window is positive if onset occurs within its next six hours. Negative windows without six hours of follow-up are censored; all post-event windows are excluded. This is simulator-ground-truth regime detection, not a validated clinical endpoint.')
    p('The research score is the probability of currently occupying or reaching the learned critical regime within six transitions. Make the critical row absorbing in transition matrix B and compute alpha_t B^6 e_critical. This is a surrogate reachability score; it is not the exact probability of the two-hour event, and it is not a calibrated clinical risk. A threshold is selected by maximum Youden J on validation, then frozen before evaluating test. Threshold comparisons are inclusive.')
    p('Hourly sensitivity = TP/(TP+FN); specificity = TN/(TN+FP). AUROC assesses score ranking across eligible windows. Event detection requires an above-threshold alert in the six hours strictly before onset. Lead time is onset minus earliest alert in that horizon; the median is over detected events only. Event recall and alert burden among non-event patients are reported separately. Whole-patient bootstrap resampling (400 replicates) accounts for within-patient dependence in the AUROC interval.')
    heading('7. Results and baseline comparison')
    table([['Metric','Gaussian HMM','Logistic baseline'],['AUROC',f"{m['auroc']:.4f}",f"{b['auroc']:.4f}"],['Sensitivity',f"{m['sensitivity']:.2%}",f"{b['sensitivity']:.2%}"],['Specificity',f"{m['specificity']:.2%}",f"{b['specificity']:.2%}"],['Brier score',f"{m['brier']:.5f}",f"{b['brier']:.5f}"],['Validation threshold',f"{m['threshold']:.5f}",f"{b['threshold']:.5f}"]],[200,145,145])
    p(f"Test evaluation includes {m['windows']} eligible windows, {m['positives']} positive. Confusion counts: TP={m['confusion']['tp']}, FP={m['confusion']['fp']}, TN={m['confusion']['tn']}, FN={m['confusion']['fn']}. The 95% patient-bootstrap AUROC interval is [{m['auroc_ci95'][0]:.3f}, {m['auroc_ci95'][1]:.3f}]. Event recall is {m['detected_events']}/{m['total_events']}, median detected-event lead time {m['median_lead_hours']:.1f} hours. Alerted non-event patients: {m['non_event_patients_alerted']}/{m['non_event_patients']}.")
    p('The supervised logistic baseline uses current causally cleaned and scaled vitals with training labels only, with a separate validation-selected threshold. The small AUROC difference is not a demonstrated significant advantage. This experiment shows implementation correctness and the trade-off between recall and false alerts, not clinical effectiveness.')
    story.append(PageBreak())
    heading('8. Visualization and working demonstration')
    story.append(Image(str(ROOT/'reports/roc.png'),width=385,height=209))
    story.append(Spacer(1, 16))
    story.append(Image(str(ROOT/'reports/trajectory.png'),width=490,height=234))
    p('The Flask + vanilla JavaScript/SVG dashboard provides cohort state counts, patient search, all five vital charts, raw/cleaned comparison, hour selection, sequence replay, filtered probabilities, optional retrospective smoothing, Viterbi paths, synthetic event markers, a performance view, model transition matrix and submission downloads. Replay is a retrospective animation, not a live patient feed. State names and scores are labeled as research outputs. The application deliberately accepts no patient uploads.')
    story.append(PageBreak())
    heading('9. Limitations and future work')
    p('The synthetic generator closely resembles HMM assumptions, potentially favoring the model. The cohort is small and stationary, with mostly random missingness. It omits treatments, demographic variation, confounding, irregular sampling, clinical adjudication and dataset shift. Artificial event labels depend on simulator regimes, not patient outcomes. Small baseline differences do not establish superiority. Scores are not calibrated to real-world prevalence. High event recall coexists with frequent alerts in non-event patients. No clinical or fairness claims are warranted.')
    p('A real-data extension would require independent credentialed MIMIC-IV access and approved handling, a clinically justified endpoint, outcome-blinded preprocessing, temporal and external validation, calibration, subgroup audits, alarm fatigue analysis and clinician review. This prototype is not intended for patient care. The project-specific objectives do not require a linear dynamical system, so no unnecessary Kalman component is added.')
    heading('10. Reproducibility and submission map')
    p('Python 3.13. Install requirements.txt in a virtual environment, run python -m patientwatch.pipeline, then python -m pytest -q and python app.py. Open http://localhost:3000. Run python scripts/build_submission.py to execute the notebook and regenerate this report and the presentation from actual results. The README includes full commands, data schema, safety notes, API routes and a demonstration script. Seed 42 is fixed. Exact numerical last digits can vary with BLAS/platform; artifacts preserve the submitted run.')
    heading('References')
    p('Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition. Proceedings of the IEEE, 77(2), 257–286. doi:10.1109/5.18626.<br/>hmmlearn documentation: https://hmmlearn.readthedocs.io/<br/>scikit-learn model evaluation: https://scikit-learn.org/stable/modules/model_evaluation.html<br/>MIMIC-IV access information (not used): https://physionet.org/content/mimiciv/')
    def footer(canvas,doc):
        canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#8c9e92'));canvas.drawString(45,28,'PatientWatch | Unit 4 Project 3 | Synthetic data · research only');canvas.drawRightString(A4[0]-45,28,str(doc.page))
    SimpleDocTemplate(str(ROOT/'reports/technical_report.pdf'),pagesize=A4,rightMargin=45,leftMargin=45,topMargin=42,bottomMargin=45,title='PatientWatch Technical Report',author='PatientWatch Research Team').build(story,onFirstPage=footer,onLaterPages=footer)


def build_slides(r):
    prs=Presentation();prs.slide_width=Inches(13.333);prs.slide_height=Inches(7.5)
    teal='#147d6b'; ink='#234339'; mute='#769084'; light='#f3f8f5'
    def rgb(h):return RGBColor.from_string(h.lstrip('#'))
    def text(slide,x,y,w,h,content,size=20,color=ink,bold=False):
        shape=slide.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h));tf=shape.text_frame;tf.word_wrap=True
        for i,line in enumerate(content.split('\n')):
            p=tf.paragraphs[0] if i==0 else tf.add_paragraph();p.text=line;p.font.name='Aptos';p.font.size=Pt(size);p.font.bold=bold;p.font.color.rgb=rgb(color);p.space_after=Pt(14)
        return shape
    def slide(title,kicker='PATIENTWATCH · UNIT 4 / PROJECT 03'):
        s=prs.slides.add_slide(prs.slide_layouts[6]);s.background.fill.solid();s.background.fill.fore_color.rgb=rgb(light)
        sh=s.shapes.add_shape(MSO_SHAPE.RECTANGLE,0,0,prs.slide_width,Inches(.09));sh.fill.solid();sh.fill.fore_color.rgb=rgb(teal);sh.line.fill.background()
        text(s,.65,.35,12,.3,kicker,11,teal,True);text(s,.65,1,12,1,title,34,ink,True)
        text(s,.65,7.08,11,.25,'SYNTHETIC DATA ONLY  /  EDUCATIONAL RESEARCH  /  NOT FOR CLINICAL USE',9,mute)
        text(s,12.1,7.05,.6,.3,f'{len(prs.slides):02d}',10,mute)
        return s
    def bullets(title,items):
        s=slide(title);text(s,.85,2.1,11.6,4.7,'\n'.join('• '+item for item in items),21);return s
    s=slide('PatientWatch','SEQUENTIAL HEALTH INTELLIGENCE')
    text(s,.8,2.25,11.6,1.5,'Smart Healthcare Patient Deterioration\nMonitoring & Early Warning System',29,teal,True)
    text(s,.8,4.35,10,1.4,'A reproducible three-state Gaussian HMM prototype\n180 artificial admissions · 8,640 observations · seed 42',20,mute)
    bullets('The problem & our objective',['Single vital signs do not describe a patient’s entire trajectory.','Estimate evolving latent health states from repeated measurements.','Expose uncertainty using filtering and retrospective smoothing.','Evaluate research alerts honestly before discussing real-world use.'])
    bullets('A disclosed synthetic longitudinal cohort',['180 independent admissions × 48 hourly readings; no real patient records.','Five features: HR, SpO2, respiratory rate, systolic BP and temperature.','Three-regime Markov simulator with Gaussian noise and patient offsets.','7.5% missing cells and occasional sensor corruption; generator included.'])
    bullets('The complete research workflow',['Dataset → range checks → per-patient forward fill → training median fallback.','Chronological admission split: 108 train / 36 validation / 36 test.','Training-fitted scaling → HMM EM → custom sequential inference.','Frozen validation threshold → held-out metrics → interactive dashboard.'])
    bullets('Model: three-state Gaussian HMM',['Hidden states: Stable / Watch / Critical (interpretive labels, not diagnoses).','Transition matrix captures persistence and change between hourly regimes.','Diagonal Gaussian emissions connect hidden state to five observed vital signs.','Baum–Welch maximum likelihood; best of three restarts by train likelihood.'])
    bullets('Core algorithms implemented ourselves',['Scaled Forward–Backward = chain sum-product with explicit scaling factors.','Filter alpha uses only present/past observations; smoother gamma uses future.','Log-domain Viterbi with backpointers gives a retrospective decoded path.','Only causal filtered probabilities enter the early-warning score.'])
    bullets('How we validate implementation',['Exact enumeration on a tiny HMM verifies posteriors and likelihood.','A 10,000-step test checks numerical stability.','Future-perturbation tests verify causal prefix invariance.','Reference posterior max error: '+f"{r['reference_check']['max_posterior_error']:.2e}"+'; Viterbi matches hmmlearn.'])
    bullets('Define the alert before evaluating it',['Synthetic onset = first pair of consecutive critical-regime hours.','Score = six-hour learned-critical reachability from filtered state probabilities.','Censor unknown negative outcomes; exclude all post-onset windows.','Choose threshold on validation only; event lead time requires an alert within 6h.'])
    m=r['metrics'];s=slide('Held-out results: useful, but not clinical evidence')
    stats=[(f"{m['auroc']:.3f}",'AUROC'),(f"{m['sensitivity']:.1%}",'Window sensitivity'),(f"{m['specificity']:.1%}",'Window specificity'),(f"{m['median_lead_hours']:.1f}h",'Median event lead time')]
    for i,(v,label) in enumerate(stats):
        text(s,.8+i*3.1,2.15,3,1,v,39,teal,True);text(s,.8+i*3.1,3.15,3,.5,label,15,mute)
    text(s,.8,4.2,11.8,2.2,f"95% patient-bootstrap AUROC CI: {m['auroc_ci95'][0]:.3f}–{m['auroc_ci95'][1]:.3f}\nDetected {m['detected_events']}/{m['total_events']} events, but alerted {m['non_event_patients_alerted']}/{m['non_event_patients']} non-event patients.\nCurrent-vitals logistic baseline AUROC: {m['baseline']['auroc']:.3f}; no significant advantage claimed.",19)
    s=slide('Compare against a non-sequential baseline');s.shapes.add_picture(str(ROOT/'reports/roc.png'),Inches(.9),Inches(2.05),width=Inches(7.5))
    text(s,8.65,2.45,3.7,3.8,'Same cohort, preprocessing and endpoint.\nSeparate validation thresholds.\n400 whole-patient bootstrap replicates.\nSmall test cohort; synthetic benchmark only.',18,mute)
    s=slide('Explore the patient trajectory');s.shapes.add_picture(str(ROOT/'reports/trajectory.png'),Inches(.8),Inches(2),width=Inches(11.7))
    bullets('Demonstration walkthrough',['Open the dashboard; select a held-out artificial patient.','Switch vital signs; drag the observation-hour slider or replay the sequence.','Compare causal filtering with retrospective smoothing and Viterbi.','Inspect performance, false-alert burden, methodology and submission downloads.'])
    bullets('Limits, responsibility & next steps',['The simulator resembles HMM assumptions; results cannot transfer to patients.','No treatments, demographics, irregular samples or adjudicated clinical endpoints.','High event recall coexists with many non-event patients receiving alerts.','Real extension: credentialed MIMIC-IV, approved handling, external validation and clinical review.'])
    bullets('Submission & reproducibility',['Executed Jupyter notebook, technical report PDF and this presentation.','Dataset CSV, saved parameters, computed results and source code on GitHub.','README: setup, regeneration, testing, data schema and demo script.','github.com/RomitDeokar/ML_Unit4 · Rabiner (1989) · hmmlearn docs'])
    prs.save(ROOT/'reports/PatientWatch_Presentation.pptx')


if __name__=='__main__':
    if '--skip-notebook' not in sys.argv:build_notebook()
    r=json.loads((ROOT/'artifacts/results.json').read_text())
    figures(r);build_report(r);build_slides(r)
    print('Built report PDF, 14-slide presentation and research figures.')
