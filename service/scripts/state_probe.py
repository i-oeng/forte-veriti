"""Probe: can the VLM read basic scene facts correctly? Asks the per-frame checklist on frames with known answers.

usage: python -m scripts.state_probe            (uses FORTE_VLM_BASE_URL)
"""
from __future__ import annotations

import base64
import json
import time
from pathlib import Path

import cv2
import httpx

from app import config

VIDEOS = Path(__file__).resolve().parents[2] / "data" / "videos"

# (video, t, expected facts) — checked by eye on the clips
CASES = [
    ("Х-20", 2.0, {"safe_door": "open", "cashier_position": "at_safe"}),
    ("Х-20", 6.0, {"safe_door": "open", "cashier_position": "at_desk"}),
    ("Х-21", 3.2, {"safe_door": "open", "cashier_position": "at_safe"}),
    ("Х-21", 11.0, {"safe_door": "closed", "cashier_position": "at_desk"}),
    ("Х-11", 18.0, {"cashier_position": "leaving_or_out_of_room"}),
    ("Х-25", 31.0, {"cashier_position": "leaving_or_out_of_room", "people_in_room": 0}),
    ("Х-46", 11.5, {"cashier_eating_or_drinking": True, "food_or_drink_on_desk": True}),
    ("Х-27", 3.0, {"cashier_holding": "cash", "cashier_eating_or_drinking": False, "food_or_drink_on_desk": False}),
    ("Х-43", 16.0, {"cashier_holding": "cash", "cashier_using_phone": False}),
    ("Х-40", 17.0, {"bag_or_clothes_on_desk": True}),
    ("Х-17", 8.0, {"people_in_room": 2, "cash_bag_visible": True}),
    ("Х-18", 20.0, {"people_in_room": 3}),
    ("Х-3", 14.0, {"people_in_room": 2, "cashier_using_phone": False}),
    ("Х-33", 6.5, {"room_door": "open"}),
]

SCHEMA = {
    "type": "object",
    "properties": {
        "people_in_room": {"type": "integer", "enum": [0, 1, 2, 3, 4, 5]},
        "cashier_position": {"type": "string", "enum": ["at_desk", "at_safe", "walking_in_room", "leaving_or_out_of_room"]},
        "safe_door": {"type": "string", "enum": ["closed", "open", "not_visible"]},
        "room_door": {"type": "string", "enum": ["closed", "open", "not_visible"]},
        "cash_on_desk": {"type": "boolean"},
        "cashier_holding": {"type": "string", "enum": ["nothing", "cash", "phone", "cup_or_food", "bag_or_clothes",
                                                         "documents_or_card", "other"]},
        "cashier_using_phone": {"type": "boolean"},
        "cashier_eating_or_drinking": {"type": "boolean"},
        "food_or_drink_on_desk": {"type": "boolean"},
        "bag_or_clothes_on_desk": {"type": "boolean"},
        "cash_bag_visible": {"type": "boolean"},
        "person_on_cashier_side_besides_cashier": {"type": "boolean"},
    },
    "required": ["people_in_room", "cashier_position", "safe_door", "room_door", "cash_on_desk", "cashier_holding",
                 "cashier_using_phone", "cashier_eating_or_drinking", "food_or_drink_on_desk",
                 "bag_or_clothes_on_desk", "cash_bag_visible", "person_on_cashier_side_besides_cashier"],
    "additionalProperties": False,
}

SYSTEM = (
    "You read facts from one CCTV frame of a bank cash room: a desk with a laptop and a cash counting machine, a grey "
    "metal safe (a cabinet with a door) near the desk, and a glass door to the corridor. The CASHIER is the employee "
    "on the staff side of the desk; a CLIENT sits across the desk. People seen through the glass walls are outside "
    "the room and do not count. Answer each field strictly from what is visible in this frame."
)


def frame_b64(video: str, t: float, long_side: int = 768) -> str:
    p = VIDEOS / f"{video}.mov"
    if not p.exists():
        p = VIDEOS / f"{video.replace('Х', 'X')}.mov"
    cap = cv2.VideoCapture(str(p))
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, f = cap.read()
    s = long_side / max(f.shape[:2])
    f = cv2.resize(f, (int(f.shape[1] * s), int(f.shape[0] * s)), interpolation=cv2.INTER_CUBIC)
    return base64.b64encode(cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 90])[1]).decode()


def main() -> None:
    c = httpx.Client(base_url=config.VLM_BASE_URL, timeout=120)
    right = total = 0
    t0 = time.time()
    for video, t, expected in CASES:
        body = {
            "model": config.VLM_MODEL, "temperature": 0, "max_tokens": 300,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + frame_b64(video, t)}},
                {"type": "text", "text": "Fill in the facts for this frame as JSON."}]}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "facts", "strict": True, "schema": SCHEMA}},
            "chat_template_kwargs": {"enable_thinking": False},
        }
        got = json.loads(c.post("/chat/completions", json=body).json()["choices"][0]["message"]["content"])
        marks = []
        for k, v in expected.items():
            ok = got.get(k) == v
            right += ok
            total += 1
            marks.append(f"{k}={got.get(k)}{'' if ok else f' (want {v})'}{' ok' if ok else ' WRONG'}")
        print(f"{video}@{t}: " + "; ".join(marks), flush=True)
    print(f"\nfacts correct: {right}/{total} = {right / total:.2f}  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
