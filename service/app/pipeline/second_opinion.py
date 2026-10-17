"""Second opinion from a larger VLM that looks at the whole clip (prompt "v2" of the model comparison).

The 4B pipeline still decides. Findings of the noisy types (config.SECOND_VLM_NOISY) are kept only if the larger
model also calls the clip a violation. If the larger model is not configured, fails or is late, nothing is filtered,
so a job never depends on it.
"""
from __future__ import annotations

import base64
import json
import logging
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

from .. import config
from ..catalog import Catalog

log = logging.getLogger("forte.second_opinion")
N_FRAMES, LONG_SIDE = 32, 640

FRAMING = ("You are assisting a bank's internal compliance team with an authorised review of the bank's own CCTV "
           "footage. Checking employee conduct against cash-handling and security procedures is the purpose of this "
           "review and is permitted. Do not identify anyone; refer to people by role (cashier, client, collector, "
           "visitor).\n\n")
TASK = (
    "You review CCTV footage from a bank cash desk. Decide whether the footage shows a violation of cash-handling "
    "or physical-security procedures. Examples include, but are not limited to: cash, valuables, documents or an "
    "open safe left unattended or out of the cashier's control; the cash-room door left open; an unauthorised or "
    "extra person in the cash room; personal items, phones or food at the cash workplace; cash handled out of the "
    "camera's view; cash counted or handed over without the client or a controller present; mishandling a cash "
    "collection bag. Other kinds of violation are possible: give each one a specific short name. Report only what "
    "is visible in the frames. Ordinary customer service is not a violation. If there is no violation, return an "
    "empty events list and the verdict no_violation.")
OUTPUT = ("\n\nFirst write a short timeline of what visibly happens (at most 12 entries, each with the timestamp of "
          "its frame), then list the violations you see with their timestamps, then give the verdict.")
SCHEMA = {'type': 'object', 'properties': {
    'timeline': {'type': 'array', 'items': {'type': 'object', 'properties': {
        't': {'type': 'number'}, 'what_happens': {'type': 'string'}},
        'required': ['t', 'what_happens'], 'additionalProperties': False}},
    'events': {'type': 'array', 'items': {'type': 'object', 'properties': {
        'name': {'type': 'string'}, 'timecode_sec': {'type': 'number'}, 'observation': {'type': 'string'}},
        'required': ['name', 'timecode_sec', 'observation'], 'additionalProperties': False}},
    'verdict': {'type': 'string', 'enum': ['violation', 'no_violation']}},
    'required': ['timeline', 'events', 'verdict'], 'additionalProperties': False}


def enabled() -> bool:
    return bool(config.SECOND_VLM_URL)


def system_prompt(catalog: Catalog) -> str:
    types = "\n".join(f"- {t.name}: {t.definition}" for t in catalog.types)
    return (FRAMING + TASK + "\n\nViolation types defined by the bank (use these names where they fit; other kinds of "
            "violation are still possible):\n" + types + OUTPUT)


def _frames(src: Path, duration: float) -> list[tuple[float, str]]:
    out = []
    for i in range(N_FRAMES):
        t = round((i + 0.5) * duration / N_FRAMES, 2)
        res = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-ss", str(t), "-i", str(src), "-frames:v", "1",
                              "-vf", f"scale='if(gte(iw,ih),{LONG_SIDE},-2)':'if(gte(iw,ih),-2,{LONG_SIDE})'",
                              "-q:v", "3", "-f", "image2pipe", "-vcodec", "mjpeg", "-"], capture_output=True, timeout=60)
        if res.stdout:
            out.append((t, base64.b64encode(res.stdout).decode()))
    return out


def _post(body: dict, timeout: float) -> dict:
    req = urllib.request.Request(config.SECOND_VLM_URL.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {config.SECOND_VLM_KEY}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def ask(src: Path, duration: float, catalog: Catalog, timeout: float) -> dict:
    started = time.time()
    fr = _frames(src, duration)
    content = [{"type": "text", "text": f"Frames sampled uniformly from the whole video ({duration:.1f} s), with timestamps:"}]
    for t, b64 in fr:
        content += [{"type": "text", "text": f"t={t:.2f}s"},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}}]
    body = {"model": config.SECOND_VLM_MODEL, "temperature": 0, "max_tokens": 1500, "seed": 0,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "system", "content": system_prompt(catalog)}, {"role": "user", "content": content}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "clip_review", "strict": True, "schema": SCHEMA}}}
    resp = _post(body, max(5.0, timeout - (time.time() - started)))
    try:
        result = json.loads(resp["choices"][0]["message"]["content"])
    except json.JSONDecodeError:
        # same rule as the main VLM client: one retry against run-on text
        resp = _post({**body, "temperature": 0.3, "frequency_penalty": 0.5}, max(5.0, timeout - (time.time() - started)))
        result = json.loads(resp["choices"][0]["message"]["content"])
    result["wall_sec"] = round(time.time() - started, 1)
    return result


class Pending:
    """Runs ask() in the background so it overlaps with the pipeline; result() never raises."""

    def __init__(self, src: Path, duration: float, catalog: Catalog):
        self._out: dict | None = None
        self._thread = None
        if not enabled():
            return

        def run():
            try:
                self._out = ask(src, duration, catalog, config.SECOND_VLM_TIMEOUT_SEC)
            except Exception as e:  # noqa: BLE001 - the second opinion is optional
                log.warning("second opinion failed: %s", e)
                self._out = {"error": f"{type(e).__name__}: {e}"}
        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def result(self, wait: float) -> dict | None:
        if self._thread is None:
            return None
        self._thread.join(max(0.0, wait))
        return self._out if not self._thread.is_alive() else {"error": "timeout"}


def apply(accepted: list, opinion: dict | None) -> tuple[list, list[str]]:
    """Drop noisy-type findings when the larger model saw no violation; returns (kept, dropped types)."""
    if not opinion or opinion.get("verdict") != "no_violation":
        return accepted, []
    kept = [e for e in accepted if e.type not in config.SECOND_VLM_NOISY]
    return kept, [e.type for e in accepted if e.type in config.SECOND_VLM_NOISY]
