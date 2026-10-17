"""Probe: can the VLM localise the safe / room door / desk (grounding), and read the safe door from a zoomed crop?

Writes annotated frames to var/probe/ for a visual check.
usage: python -m scripts.layout_probe
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import cv2
import httpx

from app import config

VIDEOS = Path(__file__).resolve().parents[2] / "data" / "videos"
OUT = config.VAR_DIR / "probe"

# one frame per room layout, plus frames with a known safe state
LAYOUT = [("X-1", 2.0), ("Х-11", 2.0), ("Х-20", 9.0), ("Х-30", 2.0), ("Х-33", 1.0), ("Х-44", 2.0)]
SAFE_STATE = [("Х-20", 2.0, "open"), ("Х-20", 6.0, "open"), ("Х-21", 3.2, "open"), ("Х-21", 11.0, "closed"),
              ("Х-25", 22.4, "open"), ("Х-25", 31.0, "closed")]

BOX = {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4}
LAYOUT_SCHEMA = {"type": "object", "properties": {"safe": BOX, "room_door": BOX, "desk": BOX},
                 "required": ["safe", "room_door", "desk"], "additionalProperties": False}


def read(video: str, t: float):
    p = VIDEOS / f"{video}.mov"
    cap = cv2.VideoCapture(str(p))
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    return cap.read()[1]


def b64(img, long_side=768) -> str:
    s = long_side / max(img.shape[:2])
    img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)), interpolation=cv2.INTER_CUBIC)
    return base64.b64encode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1]).decode()


def ask(c, content, schema, system):
    body = {"model": config.VLM_MODEL, "temperature": 0, "max_tokens": 200,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "a", "strict": True, "schema": schema}},
            "chat_template_kwargs": {"enable_thinking": False}}
    for temp in (0, 0.3, 0.6):
        body["temperature"] = temp
        try:
            return json.loads(c.post("/chat/completions", json=body).json()["choices"][0]["message"]["content"])
        except json.JSONDecodeError:
            continue
    raise RuntimeError("no valid JSON after retries")


def to_px(box, img):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    return int(x1 / 1000 * w), int(y1 / 1000 * h), int(x2 / 1000 * w), int(y2 / 1000 * h)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    c = httpx.Client(base_url=config.VLM_BASE_URL, timeout=120)
    sys_ground = ("Locate objects in this CCTV frame of a bank cash room. Give each bounding box as "
                  "[x1, y1, x2, y2] in relative coordinates from 0 to 1000.")
    layouts = {}
    for v, t in LAYOUT:
        img = read(v, t)
        r = ask(c, [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64(img)}},
                    {"type": "text", "text": "Boxes for: the grey metal safe (a small cabinet with a door), the glass door "
                                             "of the room, the cashier's desk."}], LAYOUT_SCHEMA, sys_ground)
        layouts[v] = r
        vis = img.copy()
        for name, color in (("safe", (0, 0, 255)), ("room_door", (255, 0, 0)), ("desk", (0, 200, 0))):
            x1, y1, x2, y2 = to_px(r[name], img)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)
            cv2.putText(vis, name, (x1 + 4, y1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imwrite(str(OUT / f"layout_{v}.jpg"), vis)
        print(v, r, flush=True)

    # safe state from a zoomed crop (safe box from the layout of the same video, found on its own frame)
    sys_state = "You look at a zoomed crop of a grey metal safe in a bank cash room."
    schema = {"type": "object", "properties": {"safe_door": {"type": "string", "enum": ["open", "closed"]}},
              "required": ["safe_door"], "additionalProperties": False}
    right = 0
    for v, t, want in SAFE_STATE:
        ref = read(v, 0.5)
        r = ask(c, [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64(ref)}},
                    {"type": "text", "text": "Box for the grey metal safe (a small cabinet with a door)."}],
                {"type": "object", "properties": {"safe": BOX}, "required": ["safe"], "additionalProperties": False},
                sys_ground)
        img = read(v, t)
        x1, y1, x2, y2 = to_px(r["safe"], img)
        pw, ph = int(0.5 * (x2 - x1)), int(0.5 * (y2 - y1))
        crop = img[max(0, y1 - ph):y2 + ph, max(0, x1 - pw):x2 + pw]
        cv2.imwrite(str(OUT / f"safe_{v}_{t}.jpg"), crop)
        got = ask(c, [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64(crop, 512)}},
                      {"type": "text", "text": "Is the safe door open or closed?"}], schema, sys_state)["safe_door"]
        right += got == want
        print(f"{v}@{t}: safe {got} (want {want}) {'ok' if got == want else 'WRONG'}", flush=True)
    print(f"safe state from crop: {right}/{len(SAFE_STATE)}")


if __name__ == "__main__":
    main()
