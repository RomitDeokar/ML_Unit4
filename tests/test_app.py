from app import app


def test_dashboard_and_api():
    client = app.test_client()
    assert client.get('/').status_code == 200
    summary = client.get('/api/summary').get_json()
    assert summary['cohort_size'] == 180 and 'patients' not in summary
    patients = client.get('/api/patients').get_json()
    assert len(patients) == 180
    assert len([p for p in patients if p['split'] == 'test']) == 36
    assert client.get('/api/patients/SYN-180').status_code == 200
    assert client.get('/api/patients/not-real').status_code == 404
    assert client.get('/health').get_json()['status'] == 'ok'


def test_exports_and_read_only():
    client = app.test_client()
    response = client.get('/download/patient/SYN-180')
    assert response.status_code == 200 and 'text/csv' in response.content_type
    assert len(response.text.splitlines()) == 49
    assert client.get('/download/dataset').status_code == 200
    assert client.get('/download/not-a-file').status_code == 404
    assert client.post('/api/patients').status_code == 405
    assert 'nosniff' == client.get('/').headers['X-Content-Type-Options']
