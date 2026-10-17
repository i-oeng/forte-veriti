"""Dev helper: run the pipeline directly on files (no API), print a compact summary per video.

usage: python -m scripts.run_local data/videos/Х-11.mov [...]
"""
import json
import sys
import time
from pathlib import Path

from app import config
from app.pipeline import analyze

for p in sys.argv[1:]:
    src = Path(p)
    out = config.VAR_DIR / "dev" / src.stem
    t0 = time.time()
    res = analyze.analyze(src, out, time.time() + 280, lambda s, v: None)
    ext = res["extended"]
    print(json.dumps({"video": src.name, "verdict": res["verdict"], "sec": round(time.time() - t0, 1),
                      "windows": ext["windows_reviewed"],
                      "violations": [(v["type_id"], v["timecode_sec"], v["confidence"], v["snapped_to"], v["observation"][:140])
                                     for v in ext["violations"]]}, ensure_ascii=False), flush=True)
