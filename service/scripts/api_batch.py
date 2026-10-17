"""Submit every video in a folder to the API at once (like the jury) and print verdict, type, timecode per file.

usage: FORTE_API_KEY=... python -m scripts.api_batch https://host:8443 /path/to/videos
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import httpx


def main() -> None:
    base, folder = sys.argv[1], Path(sys.argv[2])
    c = httpx.Client(base_url=base, headers={"X-API-Key": os.environ["FORTE_API_KEY"]}, timeout=120)
    jobs: dict[str, tuple[str, float]] = {}
    for p in sorted(folder.glob("*.mov")):
        with p.open("rb") as fh:
            r = c.post("/api/v1/analyze", files={"file": (p.name, fh, "video/quicktime")})
        if r.status_code == 200:
            jobs[p.name] = (r.json()["job_id"], time.time())
        else:
            print(f"{p.name}: upload failed {r.status_code} {r.text[:200]}", flush=True)
    pending = set(jobs)
    while pending:
        time.sleep(4)
        for name in sorted(pending):
            job_id, t0 = jobs[name]
            d = c.get(f"/api/v1/jobs/{job_id}", params={"extended": "true"}).json()
            if d["status"] not in ("done", "failed"):
                continue
            pending.discard(name)
            ext = d.get("extended") or {}
            vid = ext.get("video", {})
            found = [(v["type_id"], v["timecode_sec"]) for v in ext.get("violations", [])]
            print(f"{name:22s} {d['status']:6s} {time.time() - t0:5.0f}s  {vid.get('width')}x{vid.get('height')} "
                  f"{vid.get('codec')}  verdict={d.get('verdict')}  {found}  {d.get('error', '')[:120]}", flush=True)


if __name__ == "__main__":
    main()
