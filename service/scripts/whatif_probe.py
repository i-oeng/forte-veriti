"""Signals the v7 rules did not use, per clip, against the labels (from debug.json of the last direct evaluation)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / "forte/service"))
sys.stdout.reconfigure(encoding="utf-8")
from scripts import eval as ev  # noqa: E402

labels = ev.load_labels(Path.home() / "forte/data/open_labels.xlsx")
media = Path.home() / "forte/var/eval_media"
outs = Path.home() / "forte/var/eval/v7"
for d in sorted(media.iterdir(), key=lambda p: int("".join(c for c in p.name if c.isdigit()) or 0)):
    f = d / "debug.json"
    if not f.exists():
        continue
    num = int("".join(c for c in d.name if c.isdigit()))
    dbg = json.loads(f.read_text(encoding="utf-8"))
    res = json.loads((outs / f"{num}.json").read_text(encoding="utf-8")) if (outs / f"{num}.json").exists() else {}
    verdict = res.get("verdict", "?")
    facts = dbg["facts"]
    yes = [(c["type"], c["t"], c["score"], c["support"]) for c in dbg["candidates"] if c["verified"] == "yes"]
    bag_raw = [f_["t"] for f_ in facts if f_.get("cash_bag")]
    hand = [f_["t"] for f_ in facts if f_.get("counting_by_hand") and not f_.get("machine_in_use")]
    safe_open_alone = [f_["t"] for f_ in facts if f_.get("safe_open") and not f_.get("someone_at_safe")]
    lab = labels.get(num)
    tag = ("TP" if verdict == "violation" else "FN") if lab and lab[0] is not None else ("FP" if verdict == "violation" else "TN")
    print(f"{d.name:6} {tag} lab {str(lab):8} yes={yes} bag={bag_raw[:6]} hand={len(hand)} safe_open_alone={safe_open_alone[:8]}")
