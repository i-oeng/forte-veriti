"""Job worker: claims queued jobs from SQLite and runs up to MAX_PARALLEL_JOBS analyses at once.

GPU detector calls are serialised inside the process; VLM requests from all jobs run concurrently and are
batched by the VLM server. Several worker processes (or hosts sharing the DB) can run side by side.
"""
from __future__ import annotations

import logging
import os
import signal
import socket
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

from . import config, store
from .pipeline import analyze, perception, video

log = logging.getLogger("forte.worker")
_stop = threading.Event()


def _run(job: dict) -> None:
    job_id = job["id"]
    job_dir = config.JOBS_DIR / job_id
    deadline = job["created_at"] + config.JOB_DEADLINE_SEC
    try:
        src = next(f for f in job_dir.glob("source.*") if f.name != video.NORMALIZED_NAME)
        store.progress(job_id, "normalizing video", 0.01)
        src = video.normalize(src, job_dir / video.NORMALIZED_NAME,
                              timeout=max(30.0, deadline - time.time() - 30))
        # a browser preview only for web-UI uploads: jury test videos are not copied or kept
        result = analyze.analyze(src, job_dir, deadline, lambda stage, v: store.progress(job_id, stage, v),
                                 preview=job["source"] == "ui")
        store.finish(job_id, result)
        log.info("job %s done in %.1fs: %s (%d violations)", job_id, time.time() - job["created_at"],
                 result["verdict"], len(result["violations"]))
    except Exception as e:  # noqa: BLE001 - every failure must end up in the job record
        log.error("job %s failed: %s\n%s", job_id, e, traceback.format_exc())
        store.fail(job_id, f"{type(e).__name__}: {e}")
    finally:
        if not config.KEEP_SOURCE:
            for f in job_dir.glob("source.*"):
                f.unlink(missing_ok=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    store.init()
    name = os.environ.get("FORTE_WORKER_NAME", socket.gethostname())
    n = store.requeue_orphans(name)
    if n:
        log.info("requeued %d jobs interrupted by a previous run", n)
    perception.Detector.get()  # load weights before taking jobs
    log.info("worker %s ready, %d parallel jobs", name, config.MAX_PARALLEL_JOBS)

    signal.signal(signal.SIGTERM, lambda *_: _stop.set())
    running: set = set()
    with ThreadPoolExecutor(max_workers=config.MAX_PARALLEL_JOBS) as pool:
        while not _stop.is_set():
            running = {f for f in running if not f.done()}
            job = store.claim(name) if len(running) < config.MAX_PARALLEL_JOBS else None
            if job:
                log.info("claimed job %s (%s, %s)", job["id"], job["source"], job["filename"])
                running.add(pool.submit(_run, job))
                continue
            time.sleep(0.25)
        log.info("stopping; waiting for %d running jobs", len(running))


if __name__ == "__main__":
    main()
