"""SQLite-backed job store shared by the API process and the worker process(es)."""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from typing import Any, Iterator

from . import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id          TEXT PRIMARY KEY,
    status      TEXT NOT NULL,               -- queued | running | done | failed
    source      TEXT NOT NULL DEFAULT 'api', -- api jobs are served before ui jobs
    filename    TEXT,
    created_at  REAL NOT NULL,
    started_at  REAL,
    finished_at REAL,
    stage       TEXT,
    progress    REAL DEFAULT 0,
    worker      TEXT,
    result      TEXT,
    error       TEXT
);
CREATE INDEX IF NOT EXISTS jobs_queue ON jobs(status, source, created_at);
"""

# list views read only small fields; `result` holds base64 frames, so verdict and count come from SQLite JSON
_SUMMARY = (
    "SELECT id, status, source, filename, batch, created_at, started_at, finished_at, stage, progress, error, "
    "json_extract(result, '$.verdict') AS verdict, json_array_length(result, '$.violations') AS n_violations "
    "FROM jobs"
)


@contextmanager
def _conn() -> Iterator[sqlite3.Connection]:
    config.VAR_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(config.DB_PATH, timeout=30, isolation_level=None)
    con.row_factory = sqlite3.Row
    try:
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")
        yield con
    finally:
        con.close()


def init() -> None:
    with _conn() as con:
        con.executescript(_SCHEMA)
        if "batch" not in {r["name"] for r in con.execute("PRAGMA table_info(jobs)")}:
            con.execute("ALTER TABLE jobs ADD COLUMN batch TEXT")
        con.execute("CREATE INDEX IF NOT EXISTS jobs_batch ON jobs(batch)")


def create(job_id: str, filename: str, source: str, batch: str | None = None) -> None:
    with _conn() as con:
        con.execute(
            "INSERT INTO jobs(id, status, source, filename, created_at, stage, batch) "
            "VALUES (?, 'queued', ?, ?, ?, 'queued', ?)",
            (job_id, source, filename, time.time(), batch),
        )


def history(source: str, limit: int = 50) -> list[dict[str, Any]]:
    """Newest jobs of one source (the web UI never lists jury jobs)."""
    with _conn() as con:
        rows = con.execute(_SUMMARY + " WHERE source=? ORDER BY created_at DESC LIMIT ?", (source, limit)).fetchall()
    return [dict(r) for r in rows]


def batch(batch_id: str, source: str) -> list[dict[str, Any]]:
    with _conn() as con:
        rows = con.execute(_SUMMARY + " WHERE batch=? AND source=? ORDER BY created_at, rowid",
                           (batch_id, source)).fetchall()
    return [dict(r) for r in rows]


def claim(worker: str) -> dict[str, Any] | None:
    """Atomically take the oldest queued job, API jobs first."""
    with _conn() as con:
        con.execute("BEGIN IMMEDIATE")
        row = con.execute(
            "SELECT * FROM jobs WHERE status='queued' "
            "ORDER BY CASE source WHEN 'api' THEN 0 ELSE 1 END, created_at LIMIT 1"
        ).fetchone()
        if row is None:
            con.execute("COMMIT")
            return None
        con.execute(
            "UPDATE jobs SET status='running', started_at=?, worker=?, stage='starting' WHERE id=?",
            (time.time(), worker, row["id"]),
        )
        con.execute("COMMIT")
        return dict(row)


def progress(job_id: str, stage: str, value: float) -> None:
    with _conn() as con:
        con.execute("UPDATE jobs SET stage=?, progress=? WHERE id=?", (stage, round(value, 3), job_id))


def finish(job_id: str, result: dict[str, Any]) -> None:
    with _conn() as con:
        con.execute(
            "UPDATE jobs SET status='done', stage='done', progress=1, finished_at=?, result=? WHERE id=?",
            (time.time(), json.dumps(result, ensure_ascii=False), job_id),
        )


def fail(job_id: str, error: str) -> None:
    with _conn() as con:
        con.execute(
            "UPDATE jobs SET status='failed', stage='failed', finished_at=?, error=? WHERE id=?",
            (time.time(), error[:2000], job_id),
        )


def get(job_id: str) -> dict[str, Any] | None:
    with _conn() as con:
        row = con.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    if row is None:
        return None
    job = dict(row)
    job["result"] = json.loads(job["result"]) if job["result"] else None
    return job


def queue_position(job_id: str) -> int:
    with _conn() as con:
        row = con.execute("SELECT source, created_at FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            return 0
        return con.execute(
            "SELECT COUNT(*) FROM jobs WHERE status='queued' AND id!=? AND "
            "(CASE source WHEN 'api' THEN 0 ELSE 1 END < CASE ? WHEN 'api' THEN 0 ELSE 1 END "
            " OR (source=? AND created_at < ?))",
            (job_id, row["source"], row["source"], row["created_at"]),
        ).fetchone()[0]


def requeue_orphans(worker: str) -> int:
    """Jobs left 'running' by a crashed worker with the same name go back to the queue."""
    with _conn() as con:
        cur = con.execute(
            "UPDATE jobs SET status='queued', stage='queued', progress=0 WHERE status='running' AND worker=?",
            (worker,),
        )
        return cur.rowcount
