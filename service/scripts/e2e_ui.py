"""End-to-end check through the public frontend origin: upload like the browser, poll, fetch media, check the jury API."""
import json
import sys
import time
import urllib.request
import uuid
from pathlib import Path

BASE = sys.argv[1].rstrip("/")
VIDEO = Path(sys.argv[2])
KEY = sys.argv[3] if len(sys.argv) > 3 else ""
sys.stdout.reconfigure(encoding="utf-8")


def req(method, path, data=None, headers=None):
    r = urllib.request.Request(BASE + path, data=data, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(r, timeout=120) as resp:
            return resp.status, resp.read(), resp.headers.get("content-type", "")
    except urllib.error.HTTPError as e:
        return e.code, e.read(), ""


boundary = uuid.uuid4().hex
body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{VIDEO.name}\"\r\n"
        f"Content-Type: video/quicktime\r\n\r\n").encode() + VIDEO.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
code, raw, _ = req("GET", "/")
print("index", code, b"<div id=\"root\">" in raw)
t0 = time.time()
code, raw, _ = req("POST", "/ui/analyze", body, {"Content-Type": f"multipart/form-data; boundary={boundary}"})
print("upload", code, raw[:120])
job_id = json.loads(raw)["job_id"]
while True:
    code, raw, _ = req("GET", f"/ui/jobs/{job_id}")
    d = json.loads(raw)
    if d["status"] != "processing":
        break
    print(f"  {time.time() - t0:5.0f}s", d.get("job", {}).get("stage"), d.get("job", {}).get("progress"))
    time.sleep(10)
print("done in", round(time.time() - t0), "s:", d["status"], d.get("verdict"), d.get("error", ""))
for v in d.get("violations", []):
    print(" -", v["timecode_sec"], v["type"], "|", v.get("title"), "| tier", v.get("tier"), "| law:", (v.get("law_short") or "-"),
          v.get("rule_section") or "", "| frames", len(v["evidence_frames"]))
    for u in (v.get("frame_urls") or [])[:1] + ([v["clip_url"]] if v.get("clip_url") else []):
        c, b, ct = req("GET", u)
        print("    media", u, c, ct, len(b))
c, b, _ = req("GET", f"/media/{job_id}/source.mov")
print("source hidden:", c)
c, b, _ = req("GET", f"/api/v1/jobs/{job_id}")
print("jury api without key:", c)
if KEY:
    c, b, _ = req("GET", f"/api/v1/jobs/{job_id}", headers={"X-API-Key": KEY})
    j = json.loads(b)
    print("jury api with key:", c, list(j), [list(v) for v in j.get("violations", [])][:1])
