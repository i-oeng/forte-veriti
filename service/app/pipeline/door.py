"""Door state from pixels: compare the door region with its closed appearance, ignoring people.

A glass door barely changes colour when it opens, but its frame edges, handle and frosted band move. Edge maps
(not raw pixels) of a contrast-normalised patch are compared, so a lighting change is not taken for an open door.
People (inside the room or behind the glass) are masked out.
"""
from __future__ import annotations

import cv2
import numpy as np

from .perception import FrameDets

SIZE = (64, 96)  # w, h of the normalised door patch


def _edges(img: np.ndarray, box) -> np.ndarray:
    x1, y1, x2, y2 = box
    patch = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    patch = cv2.resize(patch, SIZE, interpolation=cv2.INTER_AREA).astype(np.float32)
    patch = (patch - patch.mean()) / (patch.std() + 1e-3)
    gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
    return np.hypot(gx, gy)


def _person_mask(d: FrameDets, box) -> np.ndarray:
    """True where a person covers the door patch."""
    x1, y1, x2, y2 = box
    m = np.zeros((SIZE[1], SIZE[0]), dtype=bool)
    sx, sy = SIZE[0] / max(1, x2 - x1), SIZE[1] / max(1, y2 - y1)
    for p in d.persons:
        px1, py1, px2, py2 = p.box
        a, b = int((px1 - x1) * sx) - 3, int((py1 - y1) * sy) - 3
        c, e = int((px2 - x1) * sx) + 3, int((py2 - y1) * sy) + 3
        if c > 0 and e > 0 and a < SIZE[0] and b < SIZE[1]:
            m[max(0, b):min(SIZE[1], e), max(0, a):min(SIZE[0], c)] = True
    return m


def open_series(frames, dets: list[FrameDets], box) -> tuple[list[tuple[float, float | None]], float | None]:
    """([(t, change score or None if mostly hidden)], threshold). Score = mean edge difference from the closed look."""
    if box is None or not frames:
        return [], None
    x1, y1, x2, y2 = box
    if x2 - x1 < 16 or y2 - y1 < 16:
        return [], None
    edges = [_edges(f.image, box) for f in frames]
    masks = [_person_mask(d, box) for d in dets]
    # reference = the door as seen in the first seconds, people masked out (the door starts closed in practice)
    early = max(5, len(frames) // 5)
    ref_idx = [i for i in range(early) if masks[i].mean() < 0.5][:12]
    if len(ref_idx) < 3:
        return [], None
    stack = np.stack([np.where(masks[i], np.nan, edges[i]) for i in ref_idx])
    with np.errstate(all="ignore"):
        ref = np.nanmedian(stack, axis=0)
    ref_ok = ~np.isnan(ref)
    out = []
    for f, e, m in zip(frames, edges, masks):
        visible = ~m & ref_ok
        if visible.mean() < 0.4:
            out.append((f.t, None))
            continue
        out.append((f.t, float(np.abs(e - ref)[visible].mean())))
    base_vals = [v for (_, v), i in zip(out, range(len(out))) if v is not None and i in set(ref_idx)]
    if len(base_vals) < 3:
        return out, None
    base = float(np.median(base_vals))
    spread = float(np.median(np.abs(np.array(base_vals) - base))) + 1e-3
    return out, base + max(0.25, 8.0 * spread)


def left_open_events(series, thr: float | None, min_open: float = 3.0) -> list[tuple[float, float]]:
    """(open_from, open_to) runs where the door looks open for >= min_open s while the doorway is clear."""
    if thr is None:
        return []
    out, start, last, gaps = [], None, None, 0
    for t, v in series:
        if v is None:            # hidden by a person: neither confirms nor breaks the run
            continue
        if v > thr:
            start = t if start is None else start
            last = t
        elif start is not None:
            if last - start >= min_open:
                out.append((start, last))
            start = None
    if start is not None and last - start >= min_open:
        out.append((start, last))
    return out
