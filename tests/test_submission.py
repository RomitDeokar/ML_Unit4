"""Verify that checked-in academic artifacts and published metrics agree."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import nbformat
from hmmlearn.hmm import GaussianHMM
from pptx import Presentation
from sklearn.metrics import roc_auc_score, confusion_matrix
from app import app, DOWNLOADS
from patientwatch.data import FEATURES, clean_sequences
from patientwatch.inference import emissions, forward_backward, viterbi

ROOT = Path(__file__).resolve().parents[1]
RESULTS = json.loads((ROOT / 'artifacts/results.json').read_text())


def test_all_submission_downloads_are_available():
    for name in DOWNLOADS:
        response = app.test_client().get('/download/' + name)
        assert response.status_code == 200
        assert len(response.data) > 100
        assert 'attachment' in response.headers['Content-Disposition']


def test_notebook_is_fully_executed_and_slides_valid():
    nb = nbformat.read(ROOT / 'notebooks/PatientWatch_Project.ipynb', as_version=4)
    code_cells = [c for c in nb.cells if c.cell_type == 'code']
    assert len(code_cells) >= 8
    assert all(c.execution_count is not None for c in code_cells)
    assert not [o for c in code_cells for o in c.outputs if o.output_type == 'error']
    assert len(Presentation(ROOT / 'reports/PatientWatch_Presentation.pptx').slides) == 14
    assert (ROOT / 'reports/technical_report.pdf').read_bytes().startswith(b'%PDF')


def test_published_test_metrics_and_posteriors_are_consistent():
    y, scores = [], []
    for p in RESULTS['patients']:
        assert np.allclose(np.array(p['filtered']).sum(axis=1), 1, atol=2e-8)
        assert np.allclose(np.array(p['smoothed']).sum(axis=1), 1, atol=2e-8)
        assert np.isfinite(list(p['vitals'].values())).all()
        if p['split'] == 'test':
            keep = np.array(p['eligible'])
            y.extend(np.array(p['labels'])[keep]); scores.extend(np.array(p['risk'])[keep])
            if p['event_hour'] is not None:
                assert not keep[p['event_hour']:].any()
    assert np.isclose(roc_auc_score(y, scores), RESULTS['metrics']['auroc'], atol=1e-7)
    tn, fp, fn, tp = confusion_matrix(y, np.array(scores) >= RESULTS['threshold']).ravel()
    assert dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp)) == RESULTS['metrics']['confusion']


def test_saved_parameters_reference_inference():
    params = np.load(ROOT / 'artifacts/model.npz')
    raw = pd.read_csv(ROOT / 'data/synthetic_vitals.csv')
    frame, _ = clean_sequences(raw, pd.Series(params['medians'], index=FEATURES))
    model = GaussianHMM(n_components=3, covariance_type='diag', init_params='', implementation='scaling')
    model.startprob_ = params['start']; model.transmat_ = params['transition']
    model.means_ = params['means']; model.covars_ = params['variances']
    for pid in ['SYN-001', 'SYN-109', 'SYN-145', 'SYN-180']:
        x = (frame[frame.patient_id == pid][FEATURES].to_numpy() - params['scaler_mean']) / params['scaler_scale']
        log_b = emissions(x, params['means'], params['variances'])
        _, gamma, ll, _ = forward_backward(log_b, params['start'], params['transition'])
        ref_ll, ref_gamma = model.score_samples(x)
        assert np.allclose(gamma, ref_gamma, atol=1e-12)
        assert np.isclose(ll, ref_ll, atol=1e-10)
        assert np.array_equal(viterbi(log_b, params['start'], params['transition']), model.predict(x))
