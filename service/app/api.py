"""HTTP API (jury contract, X-API-Key) and the web UI endpoints."""
from __future__ import annotations

import hmac
import json
import shutil
import subprocess
import time
import uuid
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import config, store
from .catalog import Catalog
from .pipeline import vlm

app = FastAPI(title="Cash Desk Compliance Review", version="1.0", docs_url="/api/docs", redoc_url=None)
# every upload is normalized to MP4 by the worker (video.normalize) before analysis
SUPPORTED_VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".ts",
    ".mts",
    ".m2ts",
    ".mpg",
    ".mpeg",
    ".m4v",
    ".webm",
}


@app.on_event("startup")
def _startup() -> None:
    store.init()
    config.JOBS_DIR.mkdir(parents=True, exist_ok=True)


def require_key(x_api_key: str | None = Header(default=None)) -> None:
    if not config.API_KEY or not x_api_key or not hmac.compare_digest(x_api_key, config.API_KEY):
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


async def _accept(file: UploadFile, source: str, batch: str | None = None) -> dict:
    ext = Path(file.filename or "video.mov").suffix.lower() or ".mov"
    if ext not in SUPPORTED_VIDEO_EXTENSIONS:
        raise HTTPException(status_code=415, detail=f"unsupported file type {ext}; expected one of "
                                                    f"{', '.join(sorted(SUPPORTED_VIDEO_EXTENSIONS))}")
    job_id = uuid.uuid4().hex[:12]
    job_dir = config.JOBS_DIR / job_id
    job_dir.mkdir(parents=True)
    dst = job_dir / f"source{ext}"
    size = 0
    limit = config.MAX_UPLOAD_MB * 1024 * 1024
    with dst.open("wb") as out:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > limit:
                out.close()
                shutil.rmtree(job_dir, ignore_errors=True)
                raise HTTPException(status_code=413, detail=f"file larger than {config.MAX_UPLOAD_MB} MB")
            out.write(chunk)
    probe = await run_in_threadpool(
        subprocess.run, ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name",
                         "-of", "csv=p=0", str(dst)], capture_output=True, text=True, timeout=30)
    if probe.returncode != 0 or not probe.stdout.strip():
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=422, detail="file is not a readable video")
    store.create(job_id, file.filename or dst.name, source, batch)
    return {"job_id": job_id, "status": "processing"}


async def _accept_batch(files: list[UploadFile], source: str) -> dict:
    """Several videos in one request: one job per file, analysed in parallel; a bad file does not sink the rest."""
    if not files:
        raise HTTPException(status_code=422, detail="no files")
    if len(files) > config.MAX_BATCH_FILES:
        raise HTTPException(status_code=413, detail=f"at most {config.MAX_BATCH_FILES} files per batch")
    batch_id = uuid.uuid4().hex[:12]
    jobs = []
    for f in files:
        try:
            jobs.append({**await _accept(f, source, batch_id), "filename": f.filename})
        except HTTPException as e:
            jobs.append({"job_id": None, "status": "rejected", "filename": f.filename, "error": e.detail})
    return {"batch_id": batch_id, "jobs": jobs}


def _summary(row: dict) -> dict:
    status = {"queued": "processing", "running": "processing"}.get(row["status"], row["status"])
    return {"job_id": row["id"], "filename": row["filename"], "status": status, "stage": row["stage"],
            "progress": row["progress"], "verdict": row["verdict"], "violations_count": row["n_violations"],
            "error": row["error"], "batch_id": row["batch"], "created_at": row["created_at"],
            "finished_at": row["finished_at"]}


def _batch(batch_id: str, source: str) -> dict:
    rows = store.batch(batch_id, source)
    if not rows:
        raise HTTPException(status_code=404, detail="batch not found")
    jobs = [_summary(r) for r in rows]
    done = all(j["status"] != "processing" for j in jobs)
    return {"batch_id": batch_id, "status": "done" if done else "processing",
            "finished": sum(j["status"] != "processing" for j in jobs), "total": len(jobs), "jobs": jobs}


@lru_cache(maxsize=1)
def _catalog() -> Catalog:
    return Catalog.load()


@lru_cache(maxsize=1)
def _calibration() -> dict:
    """Per-type share of correct verdicts on the labelled clips (scripts/calibrate.py); {} if not computed."""
    f = Path(__file__).resolve().parent / "calibration.json"
    return json.loads(f.read_text(encoding="utf-8")).get("types", {}) if f.exists() else {}


def _for_ui(job_id: str, api_vs: list[dict], ext_vs: list[dict], media: bool = True) -> list[dict]:
    """The spec fields plus what a UI needs: display names, rule (act, paragraph, link, about), media URLs."""
    base = f"/media/{job_id}/"
    out = []
    for i, v in enumerate(api_vs):
        x = ext_vs[i] if i < len(ext_vs) else {}
        vt = _catalog().by_id.get(_catalog().canonical(x.get("type_id") or "") or "")
        rule = asdict(vt.rule) if vt and vt.rule.filled else (x.get("rule") or {})
        title = vt.name_ru if vt else v["type"]
        if vt and vt.id == "other" and v["type"] not in (vt.label, vt.label_ru):   # new type named by the model
            title = f"{vt.name_ru}: {v['type'].removesuffix(' (other)')}"
        # confidence = how often this type was right on the labelled clips, not the rule's internal score
        cal = _calibration().get(vt.id if vt else "", {})
        conf = {"confidence": cal["confidence"],
                "confidence_basis": f"нарушение подтвердилось на {cal['violation_clips']} из {cal['clips']} "
                                    f"размеченных роликов, где сервис находил этот тип"} if cal else {}
        out.append({
            **v,
            "type_id": x.get("type_id"), "title": title, "title_en": vt.name if vt else v["type"],
            "tier": vt.tier if vt else x.get("tier"), **conf, "rule_score": x.get("confidence"),
            "law": rule.get("document") or None, "law_short": rule.get("short") or None,
            "law_url": rule.get("url") or None, "law_reg": rule.get("reg") or None,
            "law_about": rule.get("about") or None, "rule_section": rule.get("section") or None,
            "rule": rule.get("summary") or None,
            "observation": x.get("observation"), "why": x.get("why"),
            "frame_urls": [base + f["url"] for f in x.get("frames") or []] if media else [],
            "clip_url": base + x["clip"]["url"] if media and x.get("clip") else None,
        })
    return out


def _public(job: dict, extended: bool) -> dict:
    status = {"queued": "processing", "running": "processing"}.get(job["status"], job["status"])
    body: dict = {"job_id": job["id"], "status": status}
    if job["status"] == "done":
        res = job["result"]
        body["verdict"] = res["verdict"]
        body["violations"] = res["violations"]
        if extended:
            body["violations"] = _for_ui(job["id"], res["violations"], res["extended"]["violations"],
                                         media=job["source"] == "ui")
            body["extended"] = res["extended"]
    elif job["status"] == "failed":
        body["error"] = job["error"]
    if extended:
        body["job"] = {
            "filename": job["filename"], "state": job["status"], "stage": job["stage"], "progress": job["progress"],
            "created_at": job["created_at"], "started_at": job["started_at"], "finished_at": job["finished_at"],
            "queue_position": store.queue_position(job["id"]) if job["status"] == "queued" else 0,
            "media_base": f"/media/{job['id']}/", "batch_id": job.get("batch"),
        }
    return body


# ---- jury API --------------------------------------------------------------------------------------------
@app.post("/api/v1/analyze", dependencies=[Depends(require_key)])
async def analyze(file: UploadFile = File(...)):
    return await _accept(file, "api")


@app.post("/api/v1/analyze/batch", dependencies=[Depends(require_key)])
async def analyze_batch(files: list[UploadFile] = File(...)):
    """Extension: several .mov files (field `files`) -> one job per file; poll each with /api/v1/jobs/{job_id}."""
    return await _accept_batch(files, "api")


@app.get("/api/v1/batches/{batch_id}", dependencies=[Depends(require_key)])
def batch_status(batch_id: str):
    return _batch(batch_id, "api")


@app.get("/api/v1/jobs/{job_id}", dependencies=[Depends(require_key)])
def job_status(job_id: str, extended: bool = Query(False)):
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return JSONResponse(_public(job, extended))


# ---- web UI ----------------------------------------------------------------------------------------------
@app.post("/ui/analyze")
async def ui_analyze(file: UploadFile = File(...)):
    return await _accept(file, "ui")


@app.post("/ui/analyze/batch")
async def ui_analyze_batch(files: list[UploadFile] = File(...)):
    return await _accept_batch(files, "ui")


@app.get("/ui/batches/{batch_id}")
def ui_batch(batch_id: str):
    return _batch(batch_id, "ui")


@app.get("/ui/history")
def ui_history(limit: int = Query(50, ge=1, le=500)):
    """Past checks made through the web UI (jury API jobs are never listed: test videos stay private)."""
    return {"jobs": [_summary(r) for r in store.history("ui", limit)]}


@app.get("/ui/jobs/{job_id}")
def ui_job(job_id: str):
    job = store.get(job_id)
    if job is None or job["source"] != "ui":   # jury jobs are readable only with the API key
        raise HTTPException(status_code=404, detail="job not found")
    return JSONResponse(_public(job, True))


@app.get("/media/{job_id}/{path:path}")
def media(job_id: str, path: str):
    job = store.get(job_id)
    if job is None or job["source"] != "ui":   # no frames, clips or previews of jury test videos
        raise HTTPException(status_code=404)
    base = (config.JOBS_DIR / job_id).resolve()
    target = (base / path).resolve()
    if base not in target.parents or not target.is_file() or target.name.startswith("source"):
        raise HTTPException(status_code=404)
    return FileResponse(target)


@app.get("/healthz")
def healthz():
    ok_vlm = vlm.healthy()
    return JSONResponse({"ok": ok_vlm, "vlm": ok_vlm, "time": time.time()}, status_code=200 if ok_vlm else 503)


app.mount("/", StaticFiles(directory=config.WEB_DIR, html=True), name="web")
