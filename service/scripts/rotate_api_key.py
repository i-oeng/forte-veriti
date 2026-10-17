"""Rotate the jury key without printing it or passing it through process arguments."""
import os
import re
import secrets
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


def write_env(path, contents):
    fd, name = tempfile.mkstemp(prefix=".forte-env-", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w") as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def status(key):
    req = urllib.request.Request("http://127.0.0.1:8000/api/v1/jobs/key-rotation-check",
                                 headers={"X-API-Key": key})
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            return response.status
    except urllib.error.HTTPError as error:
        return error.code


def main():
    path = Path.home() / "forte/forte.env"
    original = path.read_text()
    matches = re.findall(r"^FORTE_API_KEY=(.*)$", original, flags=re.MULTILINE)
    assert len(matches) == 1, "Expected exactly one FORTE_API_KEY entry"
    old = matches[0].strip().strip("\"'")
    new = secrets.token_urlsafe(32)
    updated = re.sub(r"^FORTE_API_KEY=.*$", "FORTE_API_KEY=" + new, original, flags=re.MULTILINE)
    write_env(path, updated)
    try:
        subprocess.run(["systemctl", "--user", "restart", "forte-api"], check=True)
        deadline = time.monotonic() + 40
        while True:
            try:
                if status(new) == 404:
                    break
            except urllib.error.URLError:
                pass
            if time.monotonic() >= deadline:
                raise RuntimeError("API did not accept the replacement key")
            time.sleep(1)
        assert status(old) == 401, "Previous key still accepted"
    except Exception:
        write_env(path, original)
        subprocess.run(["systemctl", "--user", "restart", "forte-api"], check=False)
        raise
    print("Jury key rotated; new key accepted, previous key rejected (401).")
    print("Key stored only in ~/forte/forte.env (mode 600).")


if __name__ == "__main__":
    main()
