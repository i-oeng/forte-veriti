"""Turn per-frame detections into time-precise event proposals (who entered, who left, what appeared)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .perception import FrameDets, group_of


@dataclass
class Proposal:
    t: float                    # seconds, precise to the perception sampling step
    kind: str                   # person_enter | person_exit | person_moving | interaction | object_<group>
    strength: float             # 0..1, used to order VLM checks
    boxes: list[tuple[float, float, float, float]] = field(default_factory=list)
    note: str = ""


def _median_filter(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or len(x) == 0:
        return x
    pad = k // 2
    xp = np.pad(x, pad, mode="edge")
    return np.array([np.median(xp[i:i + k]) for i in range(len(x))])


def _rising_edges(signal: np.ndarray, min_len: int) -> list[int]:
    """Indices where a boolean signal turns on and stays on for at least min_len samples."""
    edges = []
    i = 0
    n = len(signal)
    while i < n:
        if signal[i] and (i == 0 or not signal[i - 1]):
            j = i
            while j < n and signal[j]:
                j += 1
            if j - i >= min_len:
                edges.append(i)
            i = j
        else:
            i += 1
    return edges


def _center(b):
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def _gap(a, b) -> float:
    """Distance between two boxes (0 if overlapping), normalised by the smaller box height."""
    dx = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    h = max(1.0, min(a[3] - a[1], b[3] - b[1]))
    return float(np.hypot(dx, dy) / h)


def propose(dets: list[FrameDets], fps: float) -> list[Proposal]:
    if not dets:
        return []
    times = np.array([d.t for d in dets])
    k = max(3, int(round(fps * 0.8)) | 1)  # ~0.8 s median window, odd
    props: list[Proposal] = []

    # 1) person count changes -> entries / exits
    count = _median_filter(np.array([len(d.persons) for d in dets], dtype=float), k)
    for i in range(1, len(count)):
        if count[i] > count[i - 1]:
            props.append(Proposal(float(times[i]), "person_enter", 0.8, [p.box for p in dets[i].persons],
                                  f"people in view {int(count[i - 1])} -> {int(count[i])}"))
        elif count[i] < count[i - 1]:
            props.append(Proposal(float(times[i]), "person_exit", 0.8, [p.box for p in dets[i - 1].persons],
                                  f"people in view {int(count[i - 1])} -> {int(count[i])}"))

    # 2) track births/deaths and large displacements (standing up, walking away, approaching)
    tracks: dict[int, list[tuple[float, tuple]]] = {}
    for d in dets:
        for p in d.persons:
            if p.track is not None:
                tracks.setdefault(p.track, []).append((d.t, p.box))
    window = max(1.0, 1.5)
    for tid, seq in tracks.items():
        if len(seq) < max(3, fps * 0.6):
            continue
        ts = np.array([s[0] for s in seq])
        cs = np.array([_center(s[1]) for s in seq])
        hs = np.array([s[1][3] - s[1][1] for s in seq])
        last_flag = -1e9
        for i in range(len(seq)):
            j = np.searchsorted(ts, ts[i] + window)
            if j >= len(seq):
                break
            disp = float(np.hypot(*(cs[j] - cs[i])) / max(1.0, np.median(hs)))
            if disp > 0.45 and ts[i] - last_flag > 3.0:
                props.append(Proposal(float(ts[i]), "person_moving", min(1.0, 0.4 + disp / 2), [seq[i][1]],
                                      f"track {tid} moved {disp:.2f} body-heights in {window:.1f}s"))
                last_flag = ts[i]

    # 3) two people close together (handover / reaching across the desk)
    close = np.zeros(len(dets), dtype=bool)
    pair_boxes: list[list] = [[] for _ in dets]
    for i, d in enumerate(dets):
        ps = d.persons
        for a in range(len(ps)):
            for b in range(a + 1, len(ps)):
                if _gap(ps[a].box, ps[b].box) < 0.15:
                    close[i] = True
                    pair_boxes[i] = [ps[a].box, ps[b].box]
    for i in _rising_edges(_median_filter(close.astype(float), k) > 0.5, max(2, int(fps * 0.6))):
        props.append(Proposal(float(times[i]), "interaction", 0.6, pair_boxes[i], "two people in contact range"))

    # 4) object groups appearing (phone, food/drink, personal bags)
    for group in ("phone", "food", "personal"):
        present = np.array([any(group_of(o.label) == group for o in d.objects) for d in dets], dtype=float)
        for i in _rising_edges(_median_filter(present, k) > 0.5, max(2, int(fps * 0.6))):
            boxes = [o.box for o in dets[i].objects if group_of(o.label) == group]
            labels = sorted({o.label for o in dets[i].objects if group_of(o.label) == group})
            conf = max((o.conf for o in dets[i].objects if group_of(o.label) == group), default=0.4)
            props.append(Proposal(float(times[i]), f"object_{group}", float(conf), boxes, ", ".join(labels)))

    props.sort(key=lambda p: p.t)
    return props


def person_count_series(dets: list[FrameDets]) -> list[tuple[float, int]]:
    return [(round(d.t, 2), len(d.persons)) for d in dets]
