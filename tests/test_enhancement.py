"""Regression checks for causal alert heads and the native Streamlit dashboard."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.special import expit
from streamlit.testing.v1 import AppTest

from patientwatch.enhancement import (build_enhancement, causal_features, choose_threshold,
                                      eligible_arrays, evaluate)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'artifacts/results.json'
DATA = json.loads(SOURCE.read_text())
EXTRA = json.loads((ROOT / 'artifacts/enhancement.json').read_text())


@pytest.mark.parametrize('kind', ['states', 'temporal'])
def test_features_are_causal_and_ignore_outcome_and_retrospective_fields(kind):
    patient = deepcopy(DATA['patients'][0])
    expected = causal_features(patient, kind)
    for length in [1, 3, 10, 31, 48]:
        prefix = deepcopy(patient)
        prefix['filtered'] = prefix['filtered'][:length]
        prefix['vitals'] = {key: values[:length] for key, values in prefix['vitals'].items()}
        assert np.allclose(causal_features(prefix, kind), expected[:length])
    for key in ['smoothed', 'true_state', 'decoded', 'labels', 'eligible', 'event_hour', 'id']:
        patient[key] = None
    assert np.allclose(causal_features(patient, kind), expected)


def test_selected_model_and_threshold_are_validation_only():
    winner = max(EXTRA['candidates'], key=lambda x: (x['validation_auroc'], -x['validation_brier']))
    assert winner == EXTRA['selected']
    assert len(EXTRA['candidates']) == 11
    val = [p for p in DATA['patients'] if p['split'] == 'validation']
    y, scores = eligible_arrays(val, EXTRA['scores'])
    assert choose_threshold(y, scores) == EXTRA['threshold']
    assert EXTRA['source_sha256'] == hashlib.sha256(SOURCE.read_bytes()).hexdigest()


def test_exported_parameters_reproduce_every_prediction():
    params = EXTRA['parameters']
    assert params is not None
    for p in DATA['patients']:
        features = causal_features(p, EXTRA['selected']['kind'])
        scaled = (features - params['mean']) / params['scale']
        scores = expit(scaled @ params['coef'] + params['intercept'])
        assert np.allclose(scores, EXTRA['scores'][p['id']], atol=1e-14)
        assert np.isfinite(scores).all()
        assert ((scores >= 0) & (scores <= 1)).all()


def test_test_metrics_event_burden_and_roc_are_recomputable():
    patients = [p for p in DATA['patients'] if p['split'] == 'test']
    assert evaluate(patients, EXTRA['scores'], EXTRA['threshold']) == EXTRA['test']
    assert EXTRA['test']['windows'] == DATA['metrics']['windows']
    assert EXTRA['test']['non_event_patients_alerted'] <= EXTRA['test']['non_event_patients']
    assert EXTRA['bootstrap']['valid_replicates'] == 400
    assert EXTRA['bootstrap']['paired_auroc_delta_ci95'][0] < 0


def test_experiment_rebuild_is_deterministic_and_preserves_original(tmp_path):
    before = SOURCE.read_bytes()
    result = build_enhancement(SOURCE, tmp_path / 'enhancement.json')
    assert result == EXTRA
    assert SOURCE.read_bytes() == before


def dashboard():
    return AppTest.from_file(str(ROOT / 'streamlit_app.py'), default_timeout=30).run()


def test_streamlit_all_pages_render_without_exceptions():
    app = dashboard()
    assert not app.exception
    for page in ['Model comparison', 'Methods & data', 'Submission files', 'Patient monitor']:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception, page
        assert any(title.value == page for title in app.title)
        if page == 'Submission files':
            assert len(app.get('download_button')) == 7


def test_streamlit_filters_hour_and_model_controls():
    app = dashboard()
    app.slider[0].set_value(0).run()
    assert not app.exception
    next(w for w in app.radio if w.label == 'Alert scoring model').set_value('Original HMM').run()
    assert not app.exception
    app.multiselect[0].set_value([]).run()
    assert not app.exception
    app.text_input[0].set_value('does-not-exist').run()
    assert not app.exception
    assert any('No patients match' in item.value for item in app.info)
    app.text_input[0].set_value('').run()
    next(w for w in app.selectbox if w.label == 'Cohort').set_value('train').run()
    assert not app.exception
    assert next(w for w in app.selectbox if w.label == 'Patient').value.startswith('SYN-')
    app.slider[0].set_value(47).run()
    assert not app.exception
