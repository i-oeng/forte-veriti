"""Read-only health, jury contract and privacy checks; key stays in process memory."""
import json
import sqlite3
import sys
import urllib.error
import urllib.request
from pathlib import Path

base = sys.argv[1].rstrip('/') if len(sys.argv) > 1 else 'http://127.0.0.1:8080'
home = Path.home() / 'forte'
key = next(line.split('=', 1)[1].strip().strip('\"\'') for line in
           (home / 'forte.env').read_text().splitlines() if line.startswith('FORTE_API_KEY='))


def request(path, auth=False):
    req = urllib.request.Request(base + path, headers={'X-API-Key': key} if auth else {})
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


code, body = request('/healthz')
assert code == 200 and json.loads(body)['vlm'] is True
with sqlite3.connect(f'file:{home / "var/jobs.sqlite3"}?mode=ro', uri=True) as con:
    row = con.execute("SELECT id FROM jobs WHERE source='api' AND status='done' ORDER BY created_at DESC LIMIT 1").fetchone()
assert row, 'No completed jury job available'
job_id = row[0]
assert request(f'/api/v1/jobs/{job_id}')[0] == 401
code, body = request(f'/api/v1/jobs/{job_id}', auth=True)
result = json.loads(body)
assert code == 200 and result['status'] == 'done'
assert set(result) == {'job_id', 'status', 'verdict', 'violations'}
for violation in result['violations']:
    assert set(violation) == {'type', 'timecode_sec', 'description', 'evidence_frames'}
    assert 1 <= len(violation['evidence_frames']) <= 3
assert request(f'/ui/jobs/{job_id}')[0] == 404
assert request(f'/media/{job_id}/preview.mp4')[0] == 404
assert request(f'/media/{job_id}/frames/v0_0.jpg')[0] == 404
code, body = request('/ui/history?limit=500')
assert code == 200
assert job_id not in {job['job_id'] for job in json.loads(body)['jobs']}
assert not list((home / 'var/jobs' / job_id).glob('source.*'))
print('PASS: health, authenticated jury result, exact JSON fields, unauthenticated 401, UI/media isolation, history privacy, source deletion.')
