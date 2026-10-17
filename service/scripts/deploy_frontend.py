"""Publish a built frontend using versioned assets and an atomic index replacement.

Run on the serving host: python3 deploy_frontend.py BUILD_DIR RELEASE_NAME
The build must use Vite's --base=/releases/RELEASE_NAME/ option.
"""
import os
import re
import shutil
import sys
import time
from pathlib import Path


def main():
    build = Path(sys.argv[1]).resolve(strict=True)
    release = sys.argv[2]
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", release):
        raise SystemExit("Invalid release name")
    root = Path("/var/www/forte").resolve(strict=True)
    index = (build / "index.html").read_bytes()
    assert f"/releases/{release}/".encode() in index, "Build has the wrong asset base"
    bundles = b"\n".join(p.read_bytes() for p in (build / "assets").glob("*.js"))
    for path in (b"/ui/analyze", b"/ui/history", b"/ui/analyze/batch"):
        assert path in bundles, f"Missing endpoint: {path!r}"
    assert b"X-API-Key" not in bundles and b"/api/v1/analyze" not in bundles
    for line in (Path.home() / "forte/forte.env").read_text().splitlines():
        if line.startswith("FORTE_API_KEY="):
            key = line.split("=", 1)[1].strip().strip("\"'").encode()
            assert not key or key not in bundles, "Jury key found in build"
    target = root / "releases" / release
    assert not target.exists(), "Release already exists; use a new name"
    backup = Path.home() / "forte/var/frontend-backups" / time.strftime("%Y%m%d-%H%M%S")
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(root, backup)
    shutil.copytree(build, target)
    index_tmp = root / f".index-{release}.tmp"
    index_tmp.write_bytes(index)
    index_tmp.chmod(0o644)
    os.replace(index_tmp, root / "index.html")
    assert (root / "index.html").read_bytes() == index
    print(f"Deployed {release}; previous deployment: {backup}")


if __name__ == "__main__":
    main()
