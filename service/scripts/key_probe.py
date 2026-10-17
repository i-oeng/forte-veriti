"""Probe: safe visits, change scores and the VLM before/after verdict on clips with and without a key left in the lock.

usage: python -m scripts.key_probe
"""
from __future__ import annotations

import time
from pathlib import Path

from app.pipeline import facts, perception, safe_watch, video

VIDEOS = Path(__file__).resolve().parents[2] / "data" / "videos"
# key left: 22, 11 — clean safe visits: 23, 21, 25, 29 — safe left open: 20 — other departures: 15, 24
CLIPS = ["Х-22", "Х-11", "Х-23", "Х-21", "Х-25", "Х-29", "Х-20", "Х-15", "Х-24"]


def main() -> None:
    for name in CLIPS:
        src = VIDEOS / f"{name}.mov"
        info = video.probe(src)
        frames = list(video.iter_frames(src, info, 5, 960))
        dets = perception.run(frames, 5)
        counts = [len(d.persons) for d in dets]
        import numpy as np
        layout = facts.locate_layout(frames, np.array(counts, dtype=float), time.time() + 120)
        vs = safe_watch.visits(frames, dets, layout.safe)
        out = []
        for v in vs:
            c = safe_watch.confirm(frames, v, layout.safe, time.time() + 60) or {}
            out.append((v["from"], v["to"], v["t_before"], v["t_after"], v["global"], v["blob"],
                        c.get("key_left_in_lock"), c.get("door_open_after")))
        print(name, "safe", layout.safe, "visits (from,to,before,after,global,blob,key,open):", out, flush=True)


if __name__ == "__main__":
    main()
