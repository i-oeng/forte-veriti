"""Fact timeline: the VLM reads simple, checkable facts from frames; rules (rules.py) decide.

Small VLMs are far more reliable at "is the safe door open?" than at "is this a violation?", so the model is
only asked for facts. Positions of people come from the detector, not from the VLM.

Two kinds of reads:
* per-sample facts (~1 fps, denser around perception events): full frame + zoomed safe crop, compact JSON
* targeted checks at specific moments: door crop after someone passes, desk crop when the room empties
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from .. import config
from . import vlm

log = logging.getLogger(__name__)

FACTS_FPS = config.fenv("FORTE_FACTS_FPS", 0.5)              # base sampling rate
FACTS_DENSE_FPS = config.fenv("FORTE_FACTS_DENSE_FPS", 2.0)  # around perception events
MAX_SAMPLES = int(config.fenv("FORTE_FACTS_MAX_SAMPLES", 90))

_BOX = {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4}

FACT_SCHEMA = {
    "type": "object",
    "properties": {
        "people": {"type": "integer", "enum": [0, 1, 2, 3, 4, 5]},
        "extra_on_staff_side": {"type": "boolean"},
        "safe_open": {"type": "boolean"},
        "someone_at_safe": {"type": "boolean"},
        "holding": {"type": "string", "enum": ["nothing", "cash", "phone", "cup", "food", "bag_or_clothes",
                                                 "documents_or_card", "other", "no_cashier"]},
        "phone": {"type": "boolean"},
        "eating": {"type": "boolean"},
        "drinking": {"type": "boolean"},
        "food_on_desk": {"type": "boolean"},
        "bag_on_desk": {"type": "boolean"},
        "cash_bag": {"type": "boolean"},
        "counting_by_hand": {"type": "boolean"},
        "machine_in_use": {"type": "boolean"},
    },
    "required": ["people", "extra_on_staff_side", "safe_open", "someone_at_safe", "holding", "phone", "eating",
                 "drinking", "food_on_desk", "bag_on_desk", "cash_bag", "counting_by_hand", "machine_in_use"],
    "additionalProperties": False,
}

FACT_SYSTEM = (
    "You read facts from one CCTV frame of a bank cash room: a desk with a laptop and a cash counting machine, a grey "
    "metal safe (a cabinet with a door) near the desk, and a glass door to the corridor. The CASHIER is the employee "
    "on the staff side of the desk (office chair next to the safe and the counting machine); a CLIENT sits or stands "
    "across the desk. People seen through the glass walls or reflected in them are outside the room and do not count. "
    "A zoomed crop of the safe follows the frame.\n"
    "Fields: people = people inside the room; extra_on_staff_side = someone other than the cashier is on the "
    "cashier's side of the desk or at the safe; safe_open = the safe door is open; someone_at_safe = a person stands "
    "right at the safe using it; holding = what the cashier holds; phone = the cashier uses a mobile phone; eating / "
    "drinking = the cashier eats / drinks; food_on_desk = food on the desk; bag_on_desk = a handbag, backpack, coat or "
    "jacket lies on the desk; cash_bag = a cash-in-transit bag (sealed sack) is visible; counting_by_hand = the "
    "cashier counts or inspects banknotes in her hands; machine_in_use = banknotes are being put into or taken out of "
    "the cash counting machine. Small rectangles in hands are usually banknotes or cards; say phone only for a "
    "clearly visible phone."
)


@dataclass
class Layout:
    safe: tuple[int, int, int, int] | None = None
    room_door: tuple[int, int, int, int] | None = None
    desk: tuple[int, int, int, int] | None = None


@dataclass
class FactFrame:
    t: float
    facts: dict = field(default_factory=dict)


def _to_px(box, w: int, h: int):
    x1, y1, x2, y2 = (max(0, min(1000, int(v))) for v in box)
    if x2 <= x1 or y2 <= y1:
        return None
    return int(x1 / 1000 * w), int(y1 / 1000 * h), int(x2 / 1000 * w), int(y2 / 1000 * h)


def _crop(img: np.ndarray, box, pad: float) -> np.ndarray | None:
    if box is None:
        return None
    x1, y1, x2, y2 = box
    pw, ph = int(pad * (x2 - x1)), int(pad * (y2 - y1))
    c = img[max(0, y1 - ph):min(img.shape[0], y2 + ph), max(0, x1 - pw):min(img.shape[1], x2 + pw)]
    return c if c.size and min(c.shape[:2]) >= 16 else None


def _body(system: str, content: list, schema: dict, max_tokens: int) -> dict:
    return {
        "model": config.VLM_MODEL, "temperature": 0, "max_tokens": max_tokens,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "facts", "strict": True, "schema": schema}},
        "chat_template_kwargs": {"enable_thinking": False},
    }


def _img(img: np.ndarray, long_side: int) -> dict:
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + vlm._b64(img, long_side)}}


# Plausibility checks are relative (share of the frame, shape) — never positions, so any camera angle works.
_AREA = {"safe": (0.003, 0.10), "room_door": (0.02, 0.75), "desk": (0.02, 0.65)}
_ASPECT = {"safe": (0.25, 2.5), "room_door": (0.12, 8.0), "desk": (0.3, 8.0)}   # width / height
_ASK = {
    "safe": "The safe: a small grey metal cabinet with a hinged door and a key lock, standing on a cabinet or on the "
            "floor near the cashier's desk. Not the desk, not the laptop, not the counting machine. Box of the safe only.",
    "room_door": "The door of the room through which people enter (often a glass door in a glass wall). Box of the "
                 "door only.",
    "desk": "The cashier's desk: the table with the laptop and the cash counting machine. Box of the desk only.",
}


def _valid(box, kind: str, w: int, h: int) -> bool:
    if box is None:
        return False
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    lo, hi = _AREA[kind]
    alo, ahi = _ASPECT[kind]
    return bw > 4 and bh > 4 and lo <= bw * bh / (w * h) <= hi and alo <= bw / bh <= ahi


def _iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u else 0.0


def _ground_one(img: np.ndarray, kind: str, deadline: float):
    """One object per question: small VLMs are much steadier this way than with several objects at once."""
    h, w = img.shape[:2]
    schema = {"type": "object", "properties": {"box": _BOX}, "required": ["box"], "additionalProperties": False}
    body = _body("You locate one object in a CCTV frame of a bank cash room. Answer with its bounding box "
                 "[x1, y1, x2, y2] in relative coordinates from 0 to 1000.",
                 [_img(img, 768), {"type": "text", "text": _ASK[kind]}], schema, 60)
    for temp in (0.0, 0.3):
        try:
            r = vlm._post_json({**body, "temperature": temp}, deadline)
        except (vlm.BadAnswer, TimeoutError):
            continue
        box = _to_px(r["box"], w, h)
        if _valid(box, kind, w, h):
            return box
    return None


def locate_layout(frames, counts: np.ndarray, deadline: float) -> Layout:
    """Find the safe, the room door and the desk in THIS video (no fixed positions, so any camera angle works):
    each object is asked separately on three of the emptiest frames, implausible boxes are dropped, and the median of
    the rest is kept."""
    n = len(frames)
    picks = set()
    for a, b in ((0.0, 0.34), (0.33, 0.67), (0.66, 1.0)):
        lo, hi = int(a * n), max(int(a * n) + 1, int(b * n))
        seg = counts[lo:hi] if len(counts) >= hi else np.zeros(hi - lo)
        picks.add(lo + int(np.argmin(seg)))
    found: dict[str, list] = {k: [] for k in _ASK}
    with ThreadPoolExecutor(max_workers=6) as pool:
        futs = [(k, pool.submit(_ground_one, frames[i].image, k, deadline)) for i in sorted(picks) for k in _ASK]
        for k, f in futs:
            box = f.result()
            if box is not None:
                found[k].append(box)
    med = {k: tuple(int(v) for v in np.median(np.array(v), axis=0)) if v else None for k, v in found.items()}
    if med["safe"] and med["desk"] and _iou(med["safe"], med["desk"]) > 0.5:
        med["safe"] = None   # the model boxed the desk again: better no safe crop than a wrong one
    return Layout(safe=med["safe"], room_door=med["room_door"], desk=med["desk"])


def sample_times(duration: float, props, frames) -> list[int]:
    """Frame indices to read facts from: base rate everywhere, denser around perception events."""
    times = set(np.arange(0.0, duration, 1.0 / FACTS_FPS).round(2))
    step = 1.0 / FACTS_DENSE_FPS
    for p in props:   # dense samples on a fixed grid so overlapping events share them
        if p.kind not in ("person_enter", "person_exit", "person_moving") and not p.kind.startswith("object_"):
            continue
        a, b = np.floor(max(0.0, p.t - 2.0) / step) * step, min(duration, p.t + 2.5)
        times.update(np.arange(a, b, step).round(2))
    ft = np.array([f.t for f in frames])
    idx = sorted({int(np.argmin(np.abs(ft - t))) for t in times})
    if len(idx) > MAX_SAMPLES:  # keep coverage: thin uniformly
        idx = [idx[int(i)] for i in np.linspace(0, len(idx) - 1, MAX_SAMPLES)]
    return idx


def _read_one(frame, layout: Layout, deadline: float) -> dict | None:
    content: list[dict] = [_img(frame.image, config.VLM_FRAME_LONG_SIDE)]
    safe = _crop(frame.image, layout.safe, 0.6)
    if safe is not None:
        content += [{"type": "text", "text": "Zoomed crop of the safe:"}, _img(safe, 288)]
    content.append({"type": "text", "text": "Facts for this moment as JSON."})
    try:
        return vlm._post_json(_body(FACT_SYSTEM, content, FACT_SCHEMA, 120), deadline)
    except vlm.BadAnswer:
        return None


def read_facts(frames, idx: list[int], layout: Layout, deadline: float, progress=None) -> list[FactFrame]:
    out: list[FactFrame] = []
    with ThreadPoolExecutor(max_workers=config.VLM_MAX_CONCURRENCY) as pool:
        futs = [(i, pool.submit(_read_one, frames[i], layout, deadline)) for i in idx]
        for k, (i, fut) in enumerate(futs):
            try:
                facts = fut.result()
            except TimeoutError:
                break
            if facts is not None:
                out.append(FactFrame(frames[i].t, facts))
            if progress and k % 10 == 0:
                progress(k / max(1, len(futs)))
    return out


# ---- moment checks: close-ups at the moments that decide a violation ----------------------------------------
_DEPARTURE_SCHEMA = {
    "type": "object",
    "properties": {
        "objects_on_desk": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
        "cash_on_desk": {"type": "boolean"},
        "bag_bundle_case_or_folder": {"type": "boolean"},
        "screen": {"type": "string", "enum": ["work_window", "lock_screen", "black_or_off", "not_visible"]},
        "safe_door": {"type": "string", "enum": ["open", "closed", "not_visible"]},
        "key_in_safe_lock": {"type": "boolean"},
    },
    "required": ["objects_on_desk", "cash_on_desk", "bag_bundle_case_or_folder", "screen", "safe_door",
                 "key_in_safe_lock"],
    "additionalProperties": False,
}


def departure_check(frame, layout: Layout, laptop_box, deadline: float) -> dict | None:
    """The cashier has just left: what is left behind? Desk, laptop screen and safe close-ups."""
    content: list[dict] = [_img(frame.image, config.VLM_FRAME_LONG_SIDE)]
    for label, box, pad, side in (("the desk", layout.desk, 0.18, 640), ("the laptop screen", laptop_box, 0.25, 384),
                                  ("the safe", layout.safe, 0.5, 320)):
        crop = _crop(frame.image, box, pad)
        if crop is not None:
            content += [{"type": "text", "text": f"Zoomed crop of {label}:"}, _img(crop, side)]
    content.append({"type": "text", "text": (
        "The cashier has just left the workplace. objects_on_desk: name every object lying on the desk (a few words "
        "each). cash_on_desk: are banknotes lying on the desk? bag_bundle_case_or_folder: is a bag, sack, money bundle, "
        "case, box, folder or tablet lying on the desk (the laptop, the cash counting machine, the desk phone, cables "
        "and cups do not count)? screen: "
        "what the laptop screen shows (an open program/document/table = work_window; a picture with a clock or a "
        "login box = lock_screen). safe_door: is the safe door open. key_in_safe_lock: is a key sticking out of the "
        "safe's lock.")})
    body = _body("You inspect a bank cash room right after the cashier left her workplace.", content,
                 _DEPARTURE_SCHEMA, 80)
    try:
        return vlm._post_json(body, deadline)
    except (vlm.BadAnswer, TimeoutError):
        return None


_ENTRY_SCHEMA = {"type": "object", "properties": {"cash_on_desk": {"type": "boolean"},
                                                   "cashier_handling_cash": {"type": "boolean"}},
                 "required": ["cash_on_desk", "cashier_handling_cash"], "additionalProperties": False}


def entry_check(frame, layout: Layout, deadline: float) -> dict | None:
    """Someone is walking in: is cash exposed at this moment?"""
    content: list[dict] = [_img(frame.image, config.VLM_FRAME_LONG_SIDE)]
    crop = _crop(frame.image, layout.desk, 0.08)
    if crop is not None:
        content += [{"type": "text", "text": "Zoomed crop of the desk:"}, _img(crop, 640)]
    content.append({"type": "text", "text": "cash_on_desk: are banknotes lying on the desk? cashier_handling_cash: is "
                                            "the cashier holding, counting or loading banknotes (also into the "
                                            "counting machine) right now?"})
    body = _body("You inspect a bank cash room at the moment a person walks in.", content, _ENTRY_SCHEMA, 40)
    try:
        return vlm._post_json(body, deadline)
    except (vlm.BadAnswer, TimeoutError):
        return None


_CIT_SCHEMA = {"type": "object", "properties": {"documents_checked": {"type": "string", "enum": ["yes", "no", "unclear"]},
                                                 "evidence": {"type": "string"}},
               "required": ["evidence", "documents_checked"], "additionalProperties": False}


def cit_check(frames: list, deadline: float) -> dict | None:
    """A cash bag was handed over: did the cashier check the courier's ID / documents before taking it?"""
    content: list[dict] = []
    for i, f in enumerate(frames):
        content += [{"type": "text", "text": f"Frame {i} (t={f.t:.1f}s)"}, _img(f.image, 512)]
    content.append({"type": "text", "text": (
        "A courier / cash collector brings a cash bag to the cashier. Before the cashier accepts the bag, is there a "
        "visible check — the courier shows an ID card or papers and the cashier looks at them, or the cashier examines "
        "the bag and its seal (documents_checked = yes) — or is the bag handed over and taken directly (no)? One "
        "sentence of evidence.")})
    body = _body("You review a cash-in-transit bag handover at a bank cash desk.", content, _CIT_SCHEMA, 120)
    try:
        return vlm._post_json(body, deadline)
    except (vlm.BadAnswer, TimeoutError):
        return None


_ID_SCHEMA = {"type": "object", "properties": {
    "evidence": {"type": "string"},
    "cash_exchanged": {"type": "string", "enum": ["yes", "no"]},
    "id_shown": {"type": "string", "enum": ["yes", "no", "unclear"]},
    "handover_frame": {"type": "integer"}},
    "required": ["evidence", "cash_exchanged", "id_shown", "handover_frame"], "additionalProperties": False}


def id_check(frames: list, deadline: float) -> dict | None:
    """A client's visit: does cash change hands, and is an identity document shown before that?"""
    content: list[dict] = []
    for i, f in enumerate(frames):
        content += [{"type": "text", "text": f"Frame {i} (t={f.t:.1f}s)"}, _img(f.image, 512)]
    content.append({"type": "text", "text": (
        "A client has come to the cashier's desk. cash_exchanged: do banknotes pass between the client and the cashier "
        "in these frames? handover_frame: the frame where cash first changes hands (0 if none). id_shown: before that, "
        "does the client show or hand over an identity document (ID card, passport) or does the cashier look at one? "
        "One sentence of evidence.")})
    body = _body("You review a client's visit at a bank cash desk.", content, _ID_SCHEMA, 120)
    try:
        r = vlm._post_json(body, deadline)
    except (vlm.BadAnswer, TimeoutError):
        return None
    k = min(max(int(r.get("handover_frame", 0)), 0), len(frames) - 1)
    r["t_handover"] = round(frames[k].t, 2)
    return r
