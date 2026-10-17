"""Vision-language verification through any OpenAI-compatible server (vLLM by default).

Output is constrained with a JSON schema (vLLM uses xgrammar), so `type` is always one of the catalogue ids.
"""
from __future__ import annotations

import base64
import json
import logging
import threading
import time
from dataclasses import dataclass

import cv2
import httpx
import numpy as np

from .. import config
from ..catalog import Catalog

log = logging.getLogger(__name__)

CONFIDENCE = {"high": 0.9, "medium": 0.65, "low": 0.35}

_client: httpx.Client | None = None
_client_lock = threading.Lock()
_slots = threading.BoundedSemaphore(config.VLM_MAX_CONCURRENCY)


def _http() -> httpx.Client:
    global _client
    with _client_lock:
        if _client is None:
            _client = httpx.Client(base_url=config.VLM_BASE_URL, timeout=config.VLM_TIMEOUT_SEC,
                                   headers={"Authorization": f"Bearer {config.VLM_API_KEY}"})
        return _client


@dataclass
class WindowFrame:
    t: float
    image: np.ndarray                 # BGR analysis frame
    crop: np.ndarray | None = None    # optional close-up of the region of interest


@dataclass
class Finding:
    type: str
    t: float
    confidence: float
    evidence: str
    window: tuple[float, float]
    source: str
    other_name: str = ""


def _b64(img: np.ndarray, long_side: int, label: str | None = None) -> str:
    """JPEG for the VLM. Full frames use a pixel budget (same detail for portrait and landscape cameras);
    close-up crops are scaled to `long_side`."""
    h, w = img.shape[:2]
    s = long_side / max(h, w)
    if long_side == config.VLM_FRAME_LONG_SIDE:
        s = min((config.VLM_FRAME_PIXELS / (h * w)) ** 0.5, 1280 / max(h, w))
    if abs(s - 1.0) > 0.02:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    else:
        img = img.copy()
    if label:
        cv2.rectangle(img, (0, 0), (12 + 11 * len(label), 26), (0, 0, 0), -1)
        cv2.putText(img, label, (5, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return base64.b64encode(buf.tobytes()).decode()


NORMAL = (
    "These are NORMAL and are not violations: a client walking in through the door, sitting down across the desk, "
    "being served and leaving; the door opening and closing as people pass through it; the cashier serving one "
    "client; counting cash by hand or with the counting machine; handing cash or documents to the client being "
    "served after counting; opening the safe, taking or putting something and closing it again; the cashier "
    "walking to the safe; the cashier leaving the room when the safe is closed and nothing valuable is left on the "
    "desk; people seen outside the glass walls or reflected in the glass."
)


def _system_prompt(catalog: Catalog) -> str:
    types = "\n".join(f"- {t.id}: {t.name}. {t.definition}" for t in catalog.types)
    return (
        "You are a bank compliance analyst reviewing CCTV footage of a bank cash room (a cashier's desk with a "
        "cash counting machine, a computer and a safe). The camera is mounted high in a corner. You receive "
        "frames from a short window of the video in time order; each frame is labelled with its number and "
        "timestamp.\n"
        "Roles: the CASHIER is the employee on the staff side of the desk (office chair next to the safe, the laptop "
        "and the counting machine). A CLIENT sits or stands on the visitor side across the desk. Only the cashier's "
        "actions and the state of the cash room can be violations; a client using his own phone or holding his own "
        "items is normal. Small rectangles in the cashier's hands are usually banknotes or cards, not a phone.\n\n"
        "Report cash-discipline or security violations that are clearly VISIBLE in these frames. Violation types:\n"
        f"{types}\n\n"
        f"{NORMAL}\n"
        "Rules: describe only what is visible (say 'the cashier', 'a man in a white shirt'; never identify people). "
        "Do not infer anything that is not visible, such as system records or document correctness. "
        "Most windows show normal work: report a violation only when the frames clearly show it, otherwise return "
        "an empty violations list. Keep 'scene' to one sentence and each 'evidence' to one sentence. For each violation give "
        "the frame number where the violating action happens (the moment it starts or becomes clearly visible). "
        "Use type 'other' only for a clear violation that fits no listed type, and then put a short specific name "
        "of it (at most 8 words, e.g. 'Keys left in the safe lock') in 'other_name'; for all other types leave "
        "'other_name' empty."
    )


def _schema(catalog: Catalog, n_frames: int) -> dict:
    return {
        "type": "object",
        "properties": {
            "scene": {"type": "string"},
            "violations": {
                "type": "array",
                "maxItems": 3,
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "enum": [t.id for t in catalog.types]},
                        "other_name": {"type": "string"},
                        "frame": {"type": "integer", "enum": list(range(n_frames))},
                        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                        "evidence": {"type": "string"},
                    },
                    "required": ["type", "other_name", "frame", "confidence", "evidence"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["scene", "violations"],
        "additionalProperties": False,
    }


def check_window(catalog: Catalog, frames: list[WindowFrame], duration: float, hints: list[str],
                 source: str, deadline: float) -> tuple[str, list[Finding]]:
    """Ask the VLM about one window. Returns (scene summary, findings)."""
    start, end = frames[0].t, frames[-1].t
    content: list[dict] = [{"type": "text", "text": (
        f"Window {start:.1f}s-{end:.1f}s of a {duration:.1f}s video, {len(frames)} frames.\n"
        + ("Motion detector notes (automatic, may be wrong):\n" + "\n".join(f"- {h}" for h in hints) + "\n" if hints else "")
    )}]
    for i, f in enumerate(frames):
        content.append({"type": "text", "text": f"Frame {i} (t={f.t:.1f}s)"})
        content.append({"type": "image_url", "image_url": {
            "url": "data:image/jpeg;base64," + _b64(f.image, config.VLM_FRAME_LONG_SIDE, f"#{i} {f.t:.1f}s")}})
        if f.crop is not None:
            content.append({"type": "text", "text": f"Close-up of the active area in frame {i}"})
            content.append({"type": "image_url", "image_url": {
                "url": "data:image/jpeg;base64," + _b64(f.crop, 512)}})
    content.append({"type": "text", "text": "Return the JSON answer."})

    body = {
        "model": config.VLM_MODEL,
        "messages": [{"role": "system", "content": _system_prompt(catalog)}, {"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": 600,
        "response_format": {"type": "json_schema", "json_schema": {"name": "window_review", "strict": True,
                                                                  "schema": _schema(catalog, len(frames))}},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    data = _post_json(body, deadline)

    findings = []
    for v in data.get("violations", []):
        idx = min(max(int(v.get("frame", 0)), 0), len(frames) - 1)
        findings.append(Finding(type=v["type"], t=frames[idx].t, confidence=CONFIDENCE.get(v.get("confidence"), 0.35),
                                evidence=str(v.get("evidence", "")).strip(), window=(start, end), source=source,
                                other_name=" ".join(str(v.get("other_name", "")).split()[:8]) if v["type"] == "other" else ""))
    return str(data.get("scene", "")).strip(), findings


class BadAnswer(ValueError):
    """The model answered, but not with parseable JSON (usually cut off by max_tokens)."""


def _post_json(body: dict, deadline: float) -> dict:
    """POST a chat completion and parse its JSON content.

    Transport errors (server restarting) are retried until the deadline. A malformed answer is retried once with
    slightly varied sampling and more tokens (greedy decoding would repeat the same output), then given up.
    """
    attempt, bad = 0, 0
    while True:
        remaining = deadline - time.time()
        if remaining < 5:
            raise TimeoutError("job deadline reached before the VLM answered")
        try:
            with _slots:
                r = _http().post("/chat/completions", json=body, timeout=min(config.VLM_TIMEOUT_SEC, remaining))
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
        except (httpx.HTTPError, KeyError) as e:
            attempt += 1
            log.warning("VLM call failed (attempt %d): %s", attempt, e)
            time.sleep(min(2.0 * attempt, 10.0))
            continue
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            bad += 1
            if bad > 1:
                raise BadAnswer(str(e)) from e
            log.info("malformed VLM JSON (%s); retrying with varied sampling", e)
            body = {**body, "temperature": 0.3, "frequency_penalty": 0.5, "max_tokens": body.get("max_tokens", 600) + 400}


def verify(catalog: Catalog, type_id: str, other_name: str, frames: list[WindowFrame], deadline: float) -> dict:
    """Second, focused look at one candidate: is this exact violation clearly visible? -> {visible, t, evidence}."""
    vt = catalog.by_id[type_id]
    name = other_name or vt.name
    definition = vt.definition if type_id != "other" else f"{other_name}. {vt.definition}"
    system = _system_prompt(catalog).split("Report cash-discipline")[0] + (
        "An automatic screening step flagged a POSSIBLE violation in this part of the video. The screen is often "
        "wrong, so check it carefully against the definition.\n"
        f"Candidate violation: {name}. Definition: {definition}\n\n"
        f"{NORMAL}\n"
        "Answer 'yes' only if the frames clearly show this exact violation and you can point to the frame where it "
        "happens; 'no' if they show normal activity or a different situation; 'unclear' if the view does not allow "
        "a decision. Describe only what is visible; never identify people."
    )
    content: list[dict] = [{"type": "text", "text": f"{len(frames)} frames from {frames[0].t:.1f}s to {frames[-1].t:.1f}s."}]
    for i, f in enumerate(frames):
        content.append({"type": "text", "text": f"Frame {i} (t={f.t:.1f}s)"})
        content.append({"type": "image_url", "image_url": {
            "url": "data:image/jpeg;base64," + _b64(f.image, config.VLM_FRAME_LONG_SIDE, f"#{i} {f.t:.1f}s")}})
        if f.crop is not None:
            content.append({"type": "text", "text": f"Close-up of the active area in frame {i}"})
            content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + _b64(f.crop, 512)}})
    content.append({"type": "text", "text": f"Is '{name}' clearly visible? Return the JSON answer."})
    schema = {
        "type": "object",
        "properties": {
            "evidence": {"type": "string"},
            "visible": {"type": "string", "enum": ["yes", "no", "unclear"]},
            "frame": {"type": "integer", "enum": list(range(len(frames)))},
        },
        "required": ["evidence", "visible", "frame"],
        "additionalProperties": False,
    }
    body = {
        "model": config.VLM_MODEL,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
        "temperature": 0, "max_tokens": 300,
        "response_format": {"type": "json_schema", "json_schema": {"name": "verify", "strict": True, "schema": schema}},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    data = _post_json(body, deadline)
    idx = min(max(int(data.get("frame", 0)), 0), len(frames) - 1)
    return {"visible": data.get("visible", "unclear"), "t": frames[idx].t, "evidence": str(data.get("evidence", "")).strip()}


def to_russian(texts: list[str], deadline: float) -> list[str]:
    """Translate short English review notes into Russian with the same local model; the input on any failure."""
    if not texts:
        return []
    numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
    body = {
        "model": config.VLM_MODEL,
        "messages": [
            {"role": "system", "content": (
                "You translate notes from a CCTV review of a bank cash desk from English into Russian. Translate "
                "faithfully and briefly; keep times such as 00:12.3 unchanged; do not add facts, guesses or "
                "conclusions. Terms: cashier = кассир, client = клиент, cash = наличные, banknotes = банкноты, "
                "safe = сейф, cash room = касса, collector = инкассатор, counting machine = счётная машина.")},
            {"role": "user", "content": f"Translate each item into Russian and return {len(texts)} items in the same "
                                        f"order:\n{numbered}"},
        ],
        "temperature": 0, "max_tokens": 200 + 160 * len(texts),
        "response_format": {"type": "json_schema", "json_schema": {"name": "ru", "strict": True, "schema": {
            "type": "object", "properties": {"ru": {"type": "array", "items": {"type": "string"}}},
            "required": ["ru"], "additionalProperties": False}}},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        ru = [" ".join(str(x).split()) for x in _post_json(body, deadline).get("ru", [])]
        if len(ru) == len(texts) and all(ru):
            return ru
        log.warning("translation returned %d of %d items; keeping English", len(ru), len(texts))
    except (BadAnswer, TimeoutError) as e:
        log.warning("translation failed (%s); keeping English", e)
    return list(texts)


def healthy() -> bool:
    try:
        return _http().get("/models", timeout=5).status_code == 200
    except httpx.HTTPError:
        return False
