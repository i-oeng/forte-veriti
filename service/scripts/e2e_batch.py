"""Batch + history + Russian output check through the public origin.

usage: python e2e_batch.py BASE KEY video1 video2 ...
"""
import json
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

BASE, KEY, VIDEOS = sys.argv[1].rstrip("/"), sys.argv[2], [Path(p) for p in sys.argv[3:]]
sys.stdout.reconfigure(encoding="utf-8")


def req(method, path, data=None, headers=None):
    r = urllib.request.Request(BASE + path, data=data, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(r, timeout=300) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def multipart(field, paths):
    b = uuid.uuid4().hex
    body = b""
    for p in paths:
        body += (f"--{b}\r\nContent-Disposition: form-data; name=\"{field}\"; filename=\"{p.name}\"\r\n"
                 f"Content-Type: video/quicktime\r\n\r\n").encode() + p.read_bytes() + b"\r\n"
    return body + f"--{b}--\r\n".encode(), {"Content-Type": f"multipart/form-data; boundary={b}"}


t0 = time.time()
body, h = multipart("files", VIDEOS + [Path(__file__)])   # the last file is not a video: must be rejected alone
code, raw = req("POST", "/ui/analyze/batch", body, h)
b = json.loads(raw)
print("batch upload", code, [(j["filename"], j["status"], j.get("error")) for j in b["jobs"]])
while True:
    code, raw = req("GET", f"/ui/batches/{b['batch_id']}")
    st = json.loads(raw)
    if st["status"] == "done":
        break
    print(f"  {time.time() - t0:4.0f}s {st['finished']}/{st['total']}",
          [(j["filename"], j["stage"], j["progress"]) for j in st["jobs"]])
    time.sleep(10)
print("batch done in", round(time.time() - t0), "s")
for j in st["jobs"]:
    code, raw = req("GET", f"/ui/jobs/{j['job_id']}")
    d = json.loads(raw)
    print(f"\n== {j['filename']}: {d['status']} {d.get('verdict')} ({j['violations_count']} violations)")
    for v in d.get("violations", []):
        print(f"- {v['timecode_sec']} | type: {v['type']} | title: {v['title']}")
        print(f"  description: {v['description']}")
        print(f"  law: {v['law_short']} | {v['rule_section']} | about: {(v.get('law_about') or '')[:80]}...")
    c1, _ = req("GET", f"/media/{j['job_id']}/preview.mp4")
    print("  preview:", c1)
code, raw = req("GET", "/ui/history?limit=5")
print("\nhistory", code, [(x["filename"], x["status"], x["verdict"], x["violations_count"]) for x in json.loads(raw)["jobs"]])

# jury side: batch via API key; its jobs must not be readable through the key-less UI, nor its media
body, h = multipart("files", VIDEOS[:1])
code, raw = req("POST", "/api/v1/analyze/batch", body, {**h, "X-API-Key": KEY})
jb = json.loads(raw)
jid = jb["jobs"][0]["job_id"]
print("\njury batch", code, jb["batch_id"], jid)
print("jury batch without key:", req("GET", f"/api/v1/batches/{jb['batch_id']}")[0])
print("jury job via /ui:", req("GET", f"/ui/jobs/{jid}")[0], "| jury batch via /ui:", req("GET", f"/ui/batches/{jb['batch_id']}")[0])
while json.loads(req("GET", f"/api/v1/jobs/{jid}", headers={"X-API-Key": KEY})[1])["status"] == "processing":
    time.sleep(5)
code, raw = req("GET", f"/api/v1/jobs/{jid}", headers={"X-API-Key": KEY})
d = json.loads(raw)
print("jury result", code, list(d), [list(v) for v in d["violations"]][:1])
print("jury media:", req("GET", f"/media/{jid}/frames/v0_0.jpg")[0], req("GET", f"/media/{jid}/preview.mp4")[0])
print("in history?", jid in [x["job_id"] for x in json.loads(req("GET", "/ui/history?limit=50")[1])["jobs"]])
