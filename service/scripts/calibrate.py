"""Per-type confidence measured on the labelled clips -> app/calibration.json.

For every violation type the service reported in an evaluation run: on how many clips it was reported, on how
many of those the clip really contains a violation (the verdict was right), and on how many the type is among
the team's labels. The UI's confidence is the share of correct verdicts, smoothed as (hits + 1) / (n + 2) so a
type seen twice does not show 100%.

usage (on the server): ~/forte/venv-app/bin/python -m scripts.calibrate --run v8
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app import config
from scripts import eval as ev


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--labels", default="../data/open_labels.xlsx")
    ap.add_argument("--types", default="../data/Copy of sheets.xlsx")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "app" / "calibration.json"))
    a = ap.parse_args()
    labels = ev.load_labels(Path(a.labels))
    types = ev.load_types(Path(a.types))
    stats: dict[str, dict] = {}
    for f in sorted((config.VAR_DIR / "eval" / a.run).glob("*.json")):
        num = int(f.stem)
        d = json.loads(f.read_text(encoding="utf-8"))
        ext = (d.get("extended") or {}).get("violations") or []
        is_violation = bool(labels.get(num)) and labels[num][0] is not None
        for tid in {v.get("type_id") for v in ext if v.get("type_id")}:
            s = stats.setdefault(tid, {"clips": 0, "violation_clips": 0, "type_in_labels": 0})
            s["clips"] += 1
            s["violation_clips"] += is_violation
            s["type_in_labels"] += tid in types.get(num, set())
    for s in stats.values():
        s["confidence"] = round((s["violation_clips"] + 1) / (s["clips"] + 2), 2)
    out = {"run": a.run, "clips": len(list((config.VAR_DIR / "eval" / a.run).glob("*.json"))), "types": stats}
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for tid, s in sorted(stats.items(), key=lambda kv: -kv[1]["clips"]):
        print(f"{tid:50} reported on {s['clips']:2} clips, violation {s['violation_clips']:2}, "
              f"type in labels {s['type_in_labels']:2} -> confidence {s['confidence']:.2f}")


if __name__ == "__main__":
    main()
