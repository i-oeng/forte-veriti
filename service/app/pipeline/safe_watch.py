"""What changed at the safe after someone used it? (key left in the lock, door left open)

Silhouettes (person boxes) tell when someone is at THIS video's safe; the safe close-up just before the visit and just
after it are compared. A key left in the lock is a small, compact new edge blob on an otherwise unchanged safe door,
so the blob score is compared with the change over the whole crop (lighting, door swinging open). Candidates are then
confirmed by the VLM on the before / after pair. Nothing depends on where the safe is in the frame.
"""
from __future__ import annotations

import cv2
import numpy as np

from .. import config
from . import vlm
from .perception import FrameDets

PATCH = (96, 128)   # w, h of the normalised safe crop


def _near(d: FrameDets, box, share: float = 0.12) -> bool:
    x1, y1, x2, y2 = box
    area = max(1.0, (x2 - x1) * (y2 - y1))
    for p in d.persons:
        ix = max(0.0, min(p.box[2], x2) - max(p.box[0], x1))
        iy = max(0.0, min(p.box[3], y2) - max(p.box[1], y1))
        if ix * iy > share * area:
            return True
    return False


def _edges(img: np.ndarray, box, pad: float = 0.12) -> np.ndarray:
    x1, y1, x2, y2 = box
    pw, ph = int(pad * (x2 - x1)), int(pad * (y2 - y1))
    crop = img[max(0, y1 - ph):y2 + ph, max(0, x1 - pw):x2 + pw]
    g = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), PATCH, interpolation=cv2.INTER_AREA).astype(np.float32)
    g = (g - g.mean()) / (g.std() + 1e-3)
    return np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))


def visits(frames, dets: list[FrameDets], box) -> list[dict]:
    """Each time someone stands at the safe: the clear frame before, the clear frame after, and how the crop changed."""
    if box is None or not frames:
        return []
    near = [_near(d, box) for d in dets]
    t = [f.t for f in frames]
    out, i, n = [], 0, len(near)
    while i < n:
        if not near[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and (near[j + 1] or (j + 2 < n and near[j + 2])):   # bridge one-frame dropouts
            j += 1
        if t[j] - t[i] >= 0.6:
            before = next((k for k in range(i - 1, -1, -1) if not near[k] and t[i] - t[k] <= 15.0), None)
            after = next((k for k in range(j + 1, n - 2) if not any(near[k:k + 3]) and t[k] - t[j] <= 8.0), None)
            if before is not None and after is not None:
                diff = np.abs(_edges(frames[after].image, box) - _edges(frames[before].image, box))
                blob = float(cv2.blur(diff, (7, 7)).max())
                out.append({"from": round(t[i], 2), "to": round(t[j], 2), "t_before": round(t[before], 2),
                            "t_after": round(t[after], 2), "global": round(float(diff.mean()), 3),
                            "blob": round(blob, 3), "before": before, "after": after})
        i = j + 1
    return out


_SCHEMA = {"type": "object", "properties": {
    "evidence": {"type": "string"},
    "key_left_in_lock": {"type": "string", "enum": ["yes", "no", "unclear"]},
    "door_open_after": {"type": "boolean"}},
    "required": ["evidence", "key_left_in_lock", "door_open_after"], "additionalProperties": False}


def confirm(frames, visit: dict, box, deadline: float) -> dict | None:
    """VLM look at the safe before and after the visit (zoomed crops)."""
    x1, y1, x2, y2 = box
    pw, ph = int(0.25 * (x2 - x1)), int(0.15 * (y2 - y1))

    def crop(i):
        img = frames[i].image
        return img[max(0, y1 - ph):y2 + ph, max(0, x1 - pw):x2 + pw]

    content = [{"type": "text", "text": "Image 1 — the safe BEFORE the cashier used it:"},
               {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + vlm._b64(crop(visit["before"]), 448)}},
               {"type": "text", "text": "Image 2 — the same safe AFTER the cashier walked away:"},
               {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + vlm._b64(crop(visit["after"]), 448)}},
               {"type": "text", "text": "key_left_in_lock: in image 2, is a key sticking out of the safe's lock that was "
                                        "not there in image 1? door_open_after: is the safe door open in image 2? One "
                                        "sentence of evidence."}]
    body = {"model": config.VLM_MODEL, "temperature": 0, "max_tokens": 120,
            "messages": [{"role": "system", "content": "You compare two zoomed CCTV crops of a bank safe."},
                         {"role": "user", "content": content}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "safe", "strict": True, "schema": _SCHEMA}},
            "chat_template_kwargs": {"enable_thinking": False}}
    try:
        return vlm._post_json(body, deadline)
    except (vlm.BadAnswer, TimeoutError):
        return None
