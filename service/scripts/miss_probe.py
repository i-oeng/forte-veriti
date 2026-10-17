"""What did v7 see on the missed clips? (debug.json from the last direct evaluation)"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
media = Path.home() / "forte/var/eval_media"
for stem in sys.argv[1:]:
    d = json.loads((media / stem / "debug.json").read_text(encoding="utf-8"))
    print(f"===== {stem}  dur={d['video'].get('duration')}  layout={ {k: v for k, v in d['layout'].items() if v} }")
    print("  door_events:", d.get("door_events"), "thr", d.get("door_threshold"))
    ds = [v for _, v in d.get("door_series", []) if v is not None]
    if ds:
        print(f"  door series max {max(ds):.3f} at", max(d["door_series"], key=lambda x: x[1] or -1)[0])
    print("  departures:", [(x.get("t_depart"), {k: v for k, v in (x.get("check") or {}).items() if k != "evidence"}) for x in d["departures"]])
    print("  entries:", [(x["t"], {k: v for k, v in (x.get("check") or {}).items() if k != "evidence"}) for x in d["entries"]])
    print("  cits:", [(x["t"], x.get("check")) for x in d["cits"]])
    print("  id_checks:", [(x["t"], {k: v for k, v in (x.get("check") or {}).items() if k != "evidence"}) for x in d["id_checks"]])
    print("  rule_events:", [(r["type"], round(r["t"], 1)) for r in d["rule_events"]])
    print("  candidates:", [(c["type"], c["t"], c["score"], c["verified"]) for c in d["candidates"]])
    keys = ["people", "safe_open", "someone_at_safe", "holding", "counting_by_hand", "machine_in_use", "cash_bag", "bag_on_desk", "phone", "eating", "extra_on_staff_side"]
    for f in d["facts"]:
        print("   ", f["t"], " ".join(f"{k}={f.get(k)}" for k in keys if f.get(k) not in (None, False, 0, "none", "nothing")))
