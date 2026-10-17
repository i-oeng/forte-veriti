"""Score the service against the labelled dataset, mirroring the jury: video-level verdict + timecode error.

Direct mode (runs the pipeline in-process, N videos in parallel):
    python -m scripts.eval --videos ../data/videos --labels ../data/open_labels.xlsx --run base
API mode (end-to-end through the public contract, all videos submitted at once):
    python -m scripts.eval --api https://host:8443 --key KEY --run api1
Re-score saved results without running anything:
    python -m scripts.eval --run base --score-only
"""
from __future__ import annotations

import argparse
import os
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import openpyxl

from app import config


def load_labels(path: Path) -> dict[int, list[float | None]]:
    """video number -> list of labelled violation times (None = no violation). Identical files are merged."""
    ws = openpyxl.load_workbook(path).active
    labels: dict[int, list] = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        if not row[0]:
            continue
        num = int(re.findall(r"\d+", str(row[0]))[0])
        nums = [int(x) for x in re.findall(r"\d+", str(row[1] or ""))]
        t = None if not nums else (nums[0] * 60 + nums[1] if len(nums) == 2 else nums[0])
        labels.setdefault(num, []).append(t)
    return labels


def load_types(path: Path) -> dict[int, set[str]]:
    """video number -> catalogue IDs from the team's labels sheet (any cell that names a known type or alias)."""
    from app.catalog import Catalog

    if not path.exists():
        return {}
    cat = Catalog.load()
    ws = openpyxl.load_workbook(path).active
    out: dict[int, set[str]] = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        if not row[0]:
            continue
        num = int(re.findall(r"\d+", str(row[0]))[0])
        ids = set()
        for cell in row[3:]:
            for name in re.split(r"[,;/\s]+", str(cell or "")):
                cid = cat.canonical(name) if name else None
                if cid:
                    ids.add(cid)
        if ids:
            out[num] = ids
    return out


def vid_num(p: Path) -> int:
    return int(re.findall(r"\d+", p.stem)[0])


def run_direct(videos: list[Path], out: Path, parallel: int) -> None:
    from app.pipeline import analyze

    def one(p: Path):
        dst = out / f"{vid_num(p)}.json"
        t0 = time.time()
        try:
            res = analyze.analyze(p, config.VAR_DIR / "eval_media" / p.stem, time.time() + config.JOB_DEADLINE_SEC,
                                  lambda s, v: None)
            res["wall_sec"] = round(time.time() - t0, 1)
        except Exception as e:  # noqa: BLE001
            res = {"error": f"{type(e).__name__}: {e}", "wall_sec": round(time.time() - t0, 1)}
        for v in res.get("violations", []):
            for f in v.get("evidence_frames", []):
                f["image_base64"] = f["image_base64"][:16]
        dst.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
        print(f"  {p.name}: {res.get('verdict', res.get('error'))} in {res['wall_sec']}s", flush=True)

    with ThreadPoolExecutor(max_workers=parallel) as pool:
        list(pool.map(one, videos))


def run_api(videos: list[Path], out: Path, base: str, key: str, parallel: int = 4) -> None:
    """Through the running service, at most `parallel` jobs in flight (a job's time budget starts at upload)."""
    import httpx

    c = httpx.Client(base_url=base, headers={"X-API-Key": key}, timeout=120, verify=True)
    jobs = {}
    t_submit = {}
    todo = list(videos)
    pending: set[Path] = set()
    while todo or pending:
        while todo and len(pending) < parallel:
            p = todo.pop(0)
            with p.open("rb") as fh:
                r = c.post("/api/v1/analyze", files={"file": (p.name, fh, "video/quicktime")})
            r.raise_for_status()
            jobs[p] = r.json()["job_id"]
            t_submit[p] = time.time()
            pending.add(p)
        time.sleep(3)
        for p in list(pending):
            r = c.get(f"/api/v1/jobs/{jobs[p]}", params={"extended": "true"})
            d = r.json()
            if d["status"] in ("done", "failed"):
                d["wall_sec"] = round(time.time() - t_submit[p], 1)
                for v in d.get("violations", []):
                    for f in v.get("evidence_frames", []):
                        f["image_base64"] = f["image_base64"][:16]
                (out / f"{vid_num(p)}.json").write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
                print(f"  {p.name}: {d.get('verdict', d.get('error'))} in {d['wall_sec']}s", flush=True)
                pending.discard(p)


def score(out: Path, labels: dict[int, list], types: dict[int, set[str]] | None = None) -> None:
    rows, tp, fp, tn, fn, errs, errs_vlm, slow = [], 0, 0, 0, 0, [], [], 0
    types = types or {}
    type_hits, type_n = 0, 0
    for num in sorted(labels):
        f = out / f"{num}.json"
        if not f.exists():
            continue
        res = json.loads(f.read_text(encoding="utf-8"))
        truth = [t for t in labels[num] if t is not None]
        pred = res.get("violations", []) if res.get("verdict") else []
        ext = {v["timecode_sec"]: v for v in (res.get("extended") or {}).get("violations", [])}
        is_v = res.get("verdict") == "violation"
        if res.get("wall_sec", 0) > 300:
            slow += 1
        if truth and is_v:
            tp += 1
            if num in types:
                type_n += 1
                type_hits += bool(types[num] & {v.get("type_id") for v in ext.values()})
            for t in truth:
                errs.append(min(abs(v["timecode_sec"] - t) for v in pred))
                if ext:
                    errs_vlm.append(min(abs(v.get("t_vlm", v["timecode_sec"]) - t) for v in ext.values()))
        elif truth:
            fn += 1
        elif is_v:
            fp += 1
        else:
            tn += 1
        preds = "; ".join(f"{ext.get(v['timecode_sec'], {}).get('type_id', v['type'])}@{v['timecode_sec']:.1f}"
                          f"({ext.get(v['timecode_sec'], {}).get('confidence', '')})" for v in pred)
        want = ",".join(sorted(types.get(num, [])))
        rows.append(f"{num:>3} | {','.join(str(t) for t in truth) or '-':>6} | {res.get('verdict', 'ERR'):>12} | "
                    f"{res.get('wall_sec', 0):>5}s | {preds or res.get('error', '')}"
                    + (f"   [labels: {want}]" if want else ""))
    print("\nvid | label  |      verdict | time   | predictions")
    print("\n".join(rows))
    n = tp + fp + tn + fn
    if not n:
        return
    print(f"\naccuracy {(tp + tn) / n:.3f}  (TP {tp}, FP {fp}, TN {tn}, FN {fn}, n={n}); over 300s: {slow}")
    if type_n:
        print(f"type accuracy on detected violations: {type_hits}/{type_n} = {type_hits / type_n:.2f} "
              f"(a predicted type matches one of the team's labels for the clip)")
    import numpy as np
    for name, vals in (("final", errs), ("vlm-only", errs_vlm)):
        if vals:
            e = np.array(vals)
            print(f"timecode |err| ({name}) median {np.median(e):.2f}s, mean {e.mean():.2f}s, <=1s {np.mean(e <= 1):.2f}, "
                  f"<=2s {np.mean(e <= 2):.2f}, <=3s {np.mean(e <= 3):.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="../data/videos")
    ap.add_argument("--labels", default="../data/open_labels.xlsx")
    ap.add_argument("--types", default="../data/Copy of sheets.xlsx", help="team labels sheet with violation types")
    ap.add_argument("--run", required=True)
    ap.add_argument("--only", default="", help="comma-separated video numbers")
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--api", default="")
    ap.add_argument("--key", default=os.environ.get("FORTE_API_KEY", ""), help="default: $FORTE_API_KEY")
    ap.add_argument("--score-only", action="store_true")
    a = ap.parse_args()

    labels = load_labels(Path(a.labels))
    out = config.VAR_DIR / "eval" / a.run
    out.mkdir(parents=True, exist_ok=True)
    if not a.score_only:
        vids = sorted(Path(a.videos).glob("*.mov"), key=vid_num)
        if a.only:
            keep = {int(x) for x in a.only.split(",")}
            vids = [v for v in vids if vid_num(v) in keep]
        t0 = time.time()
        if a.api:
            run_api(vids, out, a.api, a.key, a.parallel)
        else:
            run_direct(vids, out, a.parallel)
        print(f"ran {len(vids)} videos in {time.time() - t0:.0f}s", file=sys.stderr)
    score(out, labels, load_types(Path(a.types)))


if __name__ == "__main__":
    main()
