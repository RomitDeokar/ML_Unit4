"""Read-only demonstration server. Deliberately accepts no clinical uploads."""
import csv
import io
import json
import os
from pathlib import Path
from flask import Flask, jsonify, render_template, send_file, abort, Response

ROOT = Path(__file__).resolve().parent
app = Flask(__name__)
RESULTS = json.loads((ROOT / 'artifacts/results.json').read_text())
PATIENTS = {p['id']: p for p in RESULTS['patients']}
DOWNLOADS = {
    'dataset': ('data/synthetic_vitals.csv', 'text/csv'),
    'notebook': ('notebooks/PatientWatch_Project.ipynb', 'application/x-ipynb+json'),
    'report': ('reports/technical_report.pdf', 'application/pdf'),
    'slides': ('reports/PatientWatch_Presentation.pptx', 'application/vnd.openxmlformats-officedocument.presentationml.presentation'),
    'model': ('artifacts/model.npz', 'application/octet-stream'),
    'results': ('artifacts/results.json', 'application/json'),
}


@app.after_request
def security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'"
    return response


@app.get('/')
def index():
    return render_template('index.html')


@app.get('/health')
def health():
    return jsonify(status='ok', source='synthetic', patients=len(PATIENTS))


@app.get('/api/summary')
def summary():
    return jsonify({k: v for k, v in RESULTS.items() if k != 'patients'})


@app.get('/api/patients')
def patients():
    return jsonify(RESULTS['patients'])


@app.get('/api/patients/<pid>')
def patient(pid):
    if pid not in PATIENTS:
        abort(404)
    return jsonify(PATIENTS[pid])


@app.get('/download/patient/<pid>')
def patient_csv(pid):
    if pid not in PATIENTS:
        abort(404)
    p = PATIENTS[pid]
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(['patient_id', 'timestamp', *RESULTS['features'], 'p_stable_filtered', 'p_watch_filtered', 'p_critical_filtered', 'six_hour_score', 'synthetic_true_state', 'eligible_for_evaluation'])
    for t in p['hours']:
        writer.writerow([pid, p['timestamps'][t], *[p['vitals'][f][t] for f in RESULTS['features']], *p['filtered'][t], p['risk'][t], p['true_state'][t], p['eligible'][t]])
    return Response(out.getvalue(), mimetype='text/csv', headers={'Content-Disposition': f'attachment; filename={pid}_research.csv'})


@app.get('/download/<name>')
def download(name):
    if name not in DOWNLOADS:
        abort(404)
    path, mime = DOWNLOADS[name]
    if not (ROOT / path).is_file():
        abort(404)
    return send_file(ROOT / path, mimetype=mime, as_attachment=True)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 3000)), debug=False)
