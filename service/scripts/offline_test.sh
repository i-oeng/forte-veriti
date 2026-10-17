#!/usr/bin/env bash
# Run as xixi in an interactive terminal; sudo prompts normally, never via an env password.
# Usage: bash ~/forte/service/scripts/offline_test.sh ~/forte/data/videos/Х-11.mov
# Blocks IPv4/IPv6 internet traffic for this user, preserving loopback and tailnet access.
set -euo pipefail
VIDEO=${1:?Usage: offline_test.sh VIDEO}
[[ -f "$VIDEO" ]] || { echo 'Video not found'; exit 1; }
[[ $(id -un) == xixi ]] || { echo 'Run as xixi on the Forte server'; exit 1; }
for cmd in iptables ip6tables systemd-run curl python3; do command -v "$cmd" >/dev/null; done
python3 - <<'PY'
import sqlite3
from pathlib import Path
db = Path.home() / 'forte/var/jobs.sqlite3'
with sqlite3.connect(f'file:{db}?mode=ro', uri=True) as con:
    count = con.execute("SELECT count(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0]
if count:
    raise SystemExit(f'Wait for {count} active jobs before the offline restart.')
PY
sudo -v
TAG="forte-offline-$(id -u)-$$"
R4=(OUTPUT -m owner --uid-owner "$(id -u)" ! -o lo ! -d 100.64.0.0/10 -m comment --comment "$TAG" -j REJECT)
R6=(OUTPUT -m owner --uid-owner "$(id -u)" ! -o lo ! -d fd7a:115c:a1e0::/48 -m comment --comment "$TAG" -j REJECT)

cleanup() {
  local status=$? failed=0
  trap - EXIT
  if ! sudo -n true; then
    echo 'Sudo expired; automatic unblock remains scheduled.' >&2
    exit 1
  fi
  if sudo -n iptables -w -C "${R4[@]}" 2>/dev/null; then
    sudo -n iptables -w -D "${R4[@]}" || failed=1
  fi
  if sudo -n ip6tables -w -C "${R6[@]}" 2>/dev/null; then
    sudo -n ip6tables -w -D "${R6[@]}" || failed=1
  fi
  if (( failed == 0 )); then
    sudo -n systemctl stop "$TAG.timer" 2>/dev/null || true
    echo 'Internet block removed.'
  else
    echo 'Cleanup failed; automatic unblock remains scheduled.' >&2
    status=1
  fi
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Schedule cleanup before installing either rule. It survives disconnection or SIGKILL.
printf -v CLEAN4 '%q ' iptables -w -D "${R4[@]}"
printf -v CLEAN6 '%q ' ip6tables -w -D "${R6[@]}"
sudo -n systemd-run --quiet --unit="$TAG" --on-active=600s \
  /bin/bash -c "$CLEAN4 2>/dev/null || true; $CLEAN6 2>/dev/null || true"
sudo -n iptables -w -I "${R4[@]}"
sudo -n ip6tables -w -I "${R6[@]}"
T0=$(date +%s)
DEADLINE=$((T0 + 540))

assert_blocked() {
  sudo -n iptables -w -C "${R4[@]}"
  sudo -n ip6tables -w -C "${R6[@]}"
  if curl --noproxy '*' -4 -s --connect-timeout 3 --max-time 5 https://huggingface.co/ -o /dev/null; then
    echo 'FAIL: internet is reachable while blocked' >&2
    exit 1
  fi
  # A fixed public IP also checks that failure is not merely DNS resolution.
  if curl --noproxy '*' -4 -s --connect-timeout 3 --max-time 5 https://1.1.1.1/ -o /dev/null; then
    echo 'FAIL: direct internet IP is reachable while blocked' >&2
    exit 1
  fi
  echo 'IPv4 and IPv6 block rules installed; external requests fail as expected.'
}

assert_blocked
systemctl --user restart forte-vllm forte-worker forte-api
until curl -fsS --max-time 3 http://127.0.0.1:8000/healthz >/dev/null 2>&1; do
  (( $(date +%s) < T0 + 300 )) || { echo 'FAIL: service did not start offline'; exit 1; }
  sleep 3
done
echo "Service started offline after $(( $(date +%s) - T0 ))s."
JOB=$(curl -fsS --max-time 40 -F "file=@$VIDEO" http://127.0.0.1:8000/ui/analyze |
  python3 -c 'import json,sys; print(json.load(sys.stdin)["job_id"])')
while :; do
  (( $(date +%s) < DEADLINE )) || { echo 'FAIL: offline analysis exceeded deadline'; exit 1; }
  STATUS=$(curl -fsS --max-time 10 "http://127.0.0.1:8000/ui/jobs/$JOB" |
    python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["status"], d.get("verdict"), len(d.get("violations") or []), d.get("error", ""))')
  case "$STATUS" in
    processing*) sleep 3 ;;
    done\ *) break ;;
    *) echo "FAIL: $STATUS"; exit 1 ;;
  esac
done
assert_blocked
echo "PASS: offline analysis $STATUS; total $(( $(date +%s) - T0 ))s; job $JOB."
