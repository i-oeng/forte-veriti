"""Door events on all clips from the v7 debug files: which ones the rule kept, which it dropped and why, vs the labels."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / "forte/service"))
sys.stdout.reconfigure(encoding="utf-8")
from scripts import eval as ev  # noqa: E402

labels = ev.load_labels(Path.home() / "forte/data/open_labels.xlsx")
types = ev.load_types(Path.home() / "forte/data/Copy of sheets.xlsx")
media = Path.home() / "forte/var/eval_media"
for d in sorted(media.iterdir(), key=lambda p: int("".join(c for c in p.name if c.isdigit()) or 0)):
    f = d / "debug.json"
    if not f.exists():
        continue
    num = int("".join(c for c in d.name if c.isdigit()))
    dbg = json.loads(f.read_text(encoding="utf-8"))
    passes = sorted({e["t"] for e in dbg["entries"]} | {x["t_empty"] for x in dbg["departures"] if "t_empty" in x}
                    | {p["t"] for p in dbg["proposals"] if p["kind"] == "person_enter"})
    thr = dbg.get("door_threshold") or 0
    series = {round(t, 2): v for t, v in dbg.get("door_series", []) if v is not None}
    rows = []
    for a, b in dbg.get("door_events", []):
        peak = max((v for t, v in series.items() if a <= t <= b), default=0)
        near = [x for x in passes if a - 3.0 <= x <= a + 3.0]
        rows.append(f"[{a:.1f}-{b:.1f} dur {b - a:.1f} peak/thr {peak / thr if thr else 0:.1f} pass {'Y' if near else '-'}]")
    lab = labels.get(num)
    door_lab = "DOOR" if "cash_desk_access_door_left_unsecured" in types.get(num, set()) else ""
    print(f"{d.name:6} label {str(lab):10} {door_lab:4} thr {thr:.2f} events {' '.join(rows)}")
