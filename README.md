# PatientWatch
### Smart Healthcare Patient Deterioration Monitoring and Early Warning System
**Unit 4: Sequential Data · Project 3**

A complete final-year UG research submission: synthetic longitudinal records, Gaussian HMM estimation, custom scaled Forward–Backward and Viterbi, causal retrospective alert evaluation, interactive dashboard, executed notebook, technical report and presentation.

> **Synthetic data only. Not a medical device. Do not use for diagnosis, triage, treatment or real patient monitoring.** “Stable”, “Watch”, “Critical” are interpretive latent regime names, not diagnoses. The app accepts no patient uploads. Sequence replay is an animation of historical artificial observations, not a live clinical feed.

## Quick start
Tested with Python 3.13. From the repository root:

```bash
git clone https://github.com/RomitDeokar/ML_Unit4.git
cd ML_Unit4
# Until the PR is merged, switch to the complete implementation branch:
git checkout genspark_ai_developer
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Visit **http://localhost:3000**. Included artifacts let the demo start without retraining or downloading data. `PORT=8080 python app.py` changes the port (Unix). The Flask development server is suitable for this temporary demo only; use an appropriate WSGI server and hosting hardening for persistent deployments. No external API keys are needed.

## Required submission files

| Requirement | File |
|---|---|
| Reproducible, executed notebook | [`notebooks/PatientWatch_Project.ipynb`](notebooks/PatientWatch_Project.ipynb) |
| Concise technical report | [`reports/technical_report.pdf`](reports/technical_report.pdf) |
| Presentation (14 slides) | [`reports/PatientWatch_Presentation.pptx`](reports/PatientWatch_Presentation.pptx) |
| Source code & README | This repository |
| Working demonstration | `app.py`, `templates/`, `static/` |
| Raw synthetic dataset | [`data/synthetic_vitals.csv`](data/synthetic_vitals.csv) |
| Saved model parameters | [`artifacts/model.npz`](artifacts/model.npz) |
| Metrics & patient inference | [`artifacts/results.json`](artifacts/results.json) |

The dashboard's **Submission files** page offers the six primary downloads. The technical report and presentation are generated from actual experiment artifacts, not manually entered benchmark claims.

## Reproduce the experiment

```bash
# Linux/macOS: limit native thread pools for predictable resource usage
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m patientwatch.pipeline
python -m pytest -q
# Re-executes the notebook in the current Python interpreter and rebuilds PDF/PPTX:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/build_submission.py
# Only regenerate report, slides and figures from existing results:
python scripts/build_submission.py --skip-notebook
```

Open the notebook in Jupyter/VS Code using this virtual environment, then Run All. Its setup detects repository root or the `notebooks/` directory. The build script uses the current interpreter for its execution kernel and requires no global kernel installation. Full rebuild takes around a minute on a typical machine. Fixed seed **42**, with three HMM initialization seeds (42, 43, 44). Small numerical differences across hardware/BLAS are normal. The checked-in artifacts preserve the submitted run.

## Data provenance and schema

**No MIMIC-IV data were accessed.** The PDF permits synthetic longitudinal records; we use that option because MIMIC-IV requires credentialed access. Do not upload or commit identifiable records. The dataset is generated entirely by `patientwatch/data.py` and needs no third-party data license or credentials.

180 admissions × 48 hourly measurements = **8,640 observations**. Admission dates are artificial and spaced three days apart. A stationary three-regime Markov generator emits five vital signs with Gaussian noise and patient offsets. Missingness is about 7.5% of cells; occasional values of `999` deliberately represent sensor corruption.

| Field | Meaning |
|---|---|
| `patient_id` | Artificial ID, `SYN-001`…`SYN-180` |
| `timestamp` | Artificial hourly timestamp (no real encounter) |
| `hour` | Relative hour 0…47 |
| `true_state` | Simulator latent state 0/1/2; evaluation only |
| `heart_rate` | beats per minute |
| `spo2` | oxygen saturation, % |
| `resp_rate` | breaths per minute |
| `systolic_bp` | systolic blood pressure, mmHg |
| `temperature` | temperature, °C |

The raw CSV retains missing cells and corruption to make cleaning observable. The NPZ includes start distribution, learned transition matrix, Gaussian means/variances, state order, training medians and scaler mean/scale. Baseline metrics are included in JSON; baseline coefficients are not required by the demonstration.

## Methodology

1. **Dataset → sequences:** sort admissions chronologically and build 48-step, per-patient sequences. Split 108 training / 36 validation / 36 test admissions. No patient or timestamp overlap between splits.
2. **Preprocessing:** invalidate out-of-range measurements; forward-fill within each patient; training-median fallback for leading missing values. No future interpolation or backward fill. Fit scaler using training data only.
3. **Model:** three-state diagonal Gaussian HMM via maximum-likelihood Baum–Welch EM (`hmmlearn`). Fit three restarts, retain highest training likelihood. Sequence lengths reset at patient boundaries.
4. **Interpret states:** rank learned standardized vital means by HR − SpO₂ + RR − SBP + temperature. Simulator labels never train the HMM or align state names.
5. **Custom algorithms:** scaled Forward–Backward/chain sum-product and log-domain Viterbi are implemented from scratch in `patientwatch/inference.py`. The notebook displays their actual source. Filtering is causal; smoothing and Viterbi are retrospective only.
6. **Score:** six-hour learned-critical reachability from filtered alpha using an absorbing critical transition row. It includes current critical occupancy. This surrogate is not the exact sustained-event probability and is **not calibrated clinical risk**.
7. **Endpoint:** first two consecutive simulator-critical hours, with onset at the first hour. Eligible windows are strictly pre-onset. Positive = onset in the next six hours. Unknown negative outcomes near sequence end are censored. Post-event windows are excluded.
8. **Threshold:** maximize Youden J on validation only, freeze before test evaluation. Baseline logistic regression uses the same current-vitals preprocessing and its own validation threshold.
9. **Evaluation:** window-level AUROC, sensitivity, specificity and Brier; event recall and median lead time among detected events. Count only alerts within six hours before onset. Report non-event-patient alert burden. AUROC intervals use 400 whole-patient bootstrap replicates.
10. **Visualization & interpretation:** dashboard and notebook show vital evolution, posterior probabilities, reference comparisons, transition matrix, ROC and caveats.

The project-specific requirements do not require a linear dynamical system/Kalman filter; the submission focuses on HMMs, maximum likelihood, Forward–Backward, scaling, sum-product and Viterbi.

## Submitted results (held-out synthetic test cohort)

| Metric | Gaussian HMM | Current-vitals logistic baseline |
|---|---:|---:|
| AUROC | 0.7925 | 0.7858 |
| Window sensitivity | 72.22% | 72.22% |
| Window specificity | 84.28% | 83.37% |
| Brier score | 0.04789 | 0.04882 |
| Validation threshold | 0.05945 | 0.06646 |

- Test: **1,293 eligible windows**, 72 positive. HMM TP 52, FP 192, TN 1,029, FN 20.
- AUROC patient-bootstrap 95% interval: **0.708–0.875**.
- Events detected within the horizon: **12/12**. Median lead time among those detected: **5.5 hours**.
- **14/24 non-event patients received at least one alert.** High event recall does not imply low alarm burden.
- Custom/reference maximum smoothed-posterior error: **5.55e-16**; log-likelihood error **1.71e-13**.

Event sensitivity and hourly sensitivity have different denominators. Lead time alone ignores missed events, so event recall is always reported alongside it. The small AUROC difference does not prove statistical superiority. Synthetic results do not establish clinical performance.

## Validation performed

- **12 automated tests pass:** exact HMM enumeration, long-sequence scaling, causal preprocessing/filtering, chronological separation, artificial event censoring, saved-parameter reference inference, HTTP/download routes, artifact validity and recomputation of published test metrics.
- Notebook: **all eight code cells executed**, no errors.
- Submission formats: **five-page PDF**, valid **14-slide PPTX**.
- Real Chromium interaction checks: patient selection, five-vital controls, hour slider, smoothing switch, search/no-match state, replay, all navigation pages and six downloads; no JavaScript page errors.
- Desktop and 390px mobile layouts checked; no document-level horizontal overflow at mobile width.
- Full-precision scores are retained in the JSON so rounding does not introduce rank ties or alter recomputed AUROC.

## Demo walkthrough (3–5 minutes)

1. Monitoring overview: read the research-only banner and synthetic-data tag.
2. Select a test patient and switch between five vital-sign charts.
3. Drag the hour slider: state probabilities and cohort counts update. Replay animates the sequence.
4. Switch to retrospective smoothing; explain why future information improves the retrospective view but cannot be used for early-warning scoring. Show the Viterbi strip.
5. Search patient IDs or filter health states. Export a patient's CSV.
6. Model performance: discuss ROC, baseline, false alerts, confidence interval and frozen threshold.
7. Methodology: inspect transition probabilities and numerical reference checks.
8. Submission files: download notebook, report, slides, dataset, parameters and complete results.

## Structure

```text
app.py                         Read-only Flask app, API and safe download allowlist
patientwatch/data.py           Simulator, cleaning, splitting, artificial outcome
patientwatch/inference.py      From-scratch scaled sum-product and Viterbi
patientwatch/pipeline.py       Training, causal evaluation, reference checks, bootstrap
static/                       Local CSS and JavaScript/SVG dashboard (no CDN)
templates/index.html          Dashboard shell
tests/                        Mathematical, causal, split and HTTP tests
notebooks/                    Executed end-to-end notebook
scripts/build_submission.py   Execute notebook and build PDF/PPTX from results
scripts/ci_workflow.yml        Optional GitHub Actions template (see below)
data/                         Raw synthetic records
artifacts/                    Learned parameters and complete computed results
reports/                      Technical report, presentation and research figures
```

### Read-only endpoints
`GET /health`, `/api/summary`, `/api/patients`, `/api/patients/<id>`, `/download/patient/<id>` and `/download/{notebook,report,slides,dataset,model,results}`. Unknown file names/IDs return 404; writes are not accepted. Downloads use a fixed allowlist, not arbitrary filesystem paths. No credentials or real clinical data are stored.

### CI permission note
The connected GitHub App cannot create `.github/workflows` files without `workflows` permission. The ready-to-use workflow is therefore provided at `scripts/ci_workflow.yml`. To enable automated GitHub CI, the repository owner can copy it to `.github/workflows/tests.yml` using an appropriately authorized account. Tests are executed locally as part of this delivery; no claim is made that GitHub Actions ran.

## Limitations and responsible use
The simulator resembles the HMM assumptions and may favor it. One seed and a small test cohort are not sufficient for robust real-world claims. There is no treatment, demographic representation, adjudicated clinical endpoint, irregular sampling, non-random clinical missingness or domain shift. Posterior confidence can be misleading under misspecification. Scores require calibration and validation before any real-world interpretation. **Do not use this demo for patient care.** A MIMIC-IV extension requires independent credentialing and approved handling plus a redesigned clinically grounded evaluation.

## References
- Rabiner, L. R. (1989). *A tutorial on hidden Markov models and selected applications in speech recognition*. Proceedings of the IEEE 77(2), 257–286. doi:10.1109/5.18626.
- https://hmmlearn.readthedocs.io/
- https://scikit-learn.org/stable/modules/model_evaluation.html
- https://physionet.org/content/mimiciv/ (access information only; not used)
