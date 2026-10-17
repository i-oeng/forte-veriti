"""Probe: single-object grounding of the safe on several frames per clip; draws the boxes for a visual check.

usage: python -m scripts.safe_probe
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import cv2
import httpx
import numpy as np

from app import config

VIDEOS = Path(__file__).resolve().parents[2] / "data" / "videos"
OUT = config.VAR_DIR / "probe"
CLIPS = ["Х-22", "Х-23", "Х-11", "Х-20", "Х-21", "Х-25", "Х-30", "Х-33", "Х-44", "X-1"]
BOX = {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4}
SCHEMA = {"type": "object", "properties": {"safe": BOX}, "required": ["safe"], "additionalProperties": False}


def ask(c, img) -> list[int] | None:
    s = 768 / max(img.shape[:2])
    small = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)))
    b64 = base64.b64encode(cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 90])[1]).decode()
    body = {"model": config.VLM_MODEL, "temperature": 0, "max_tokens": 60,
            "messages": [{"role": "system", "content": "You locate one object in a CCTV frame of a bank cash room. "
                                                       "Answer with its bounding box [x1, y1, x2, y2] in relative "
                                                       "coordinates from 0 to 1000."},
                         {"role": "user", "content": [
                             {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}},
                             {"type": "text", "text": "The safe: a small grey metal cabinet with a hinged door and a "
                                                      "key lock, standing on a white cabinet or on the floor next to "
                                                      "the cashier's desk. Not the desk, not the laptop, not the "
                                                      "counting machine. Box of the safe only."}]}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "s", "strict": True, "schema": SCHEMA}},
            "chat_template_kwargs": {"enable_thinking": False}}
    for temp in (0, 0.3):
        body["temperature"] = temp
        try:
            r = json.loads(c.post("/chat/completions", json=body).json()["choices"][0]["message"]["content"])
            return r["safe"]
        except (json.JSONDecodeError, KeyError):
            continue
    return None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    c = httpx.Client(base_url=config.VLM_BASE_URL, timeout=120)
    for v in CLIPS:
        p = VIDEOS / f"{v}.mov"
        cap = cv2.VideoCapture(str(p))
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        boxes, vis = [], None
        for frac in (0.1, 0.4, 0.7):
            cap.set(cv2.CAP_PROP_POS_MSEC, n * frac * 1000)
            ok, img = cap.read()
            h, w = img.shape[:2]
            r = ask(c, img)
            if r is None:
                continue
            x1, y1, x2, y2 = (int(r[0] / 1000 * w), int(r[1] / 1000 * h), int(r[2] / 1000 * w), int(r[3] / 1000 * h))
            bw, bh = x2 - x1, y2 - y1
            ok_box = bw > 0 and bh > 0 and 0.005 <= bw * bh / (w * h) <= 0.08 and 0.8 <= bh / bw <= 3.0
            boxes.append(((x1, y1, x2, y2), ok_box))
            vis = img.copy() if vis is None else vis
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0) if ok_box else (0, 0, 255), 2)
        good = [b for b, ok in boxes if ok]
        med = tuple(int(x) for x in np.median(np.array(good), axis=0)) if good else None
        if med and vis is not None:
            cv2.rectangle(vis, med[:2], med[2:], (255, 0, 255), 3)
        if vis is not None:
            cv2.imwrite(str(OUT / f"safe_{v}.jpg"), vis)
        print(v, [(b, ok) for b, ok in boxes], "median of valid:", med, flush=True)


if __name__ == "__main__":
    main()
