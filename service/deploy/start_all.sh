#!/usr/bin/env bash
# Start vLLM, the API and the worker with automatic restarts — for hosts without systemd (Runpod pods).
# On hosts with systemd use the forte-*.service units instead.
#   bash deploy/start_all.sh          # start (idempotent)
#   bash deploy/start_all.sh stop     # stop everything
set -u
FORTE_HOME=${FORTE_HOME:-$(cd "$(dirname "$0")/../.." && pwd)}
cd "$FORTE_HOME/service"
set -a; . "$FORTE_HOME/forte.env"; set +a
LOGS=$FORTE_HOME/logs
mkdir -p "$LOGS" "$FORTE_HOME/var"

supervise() {   # name, command...: restart the command whenever it exits
  local name=$1; shift
  ( while true; do
      echo "=== $(date -Is) start $name" >> "$LOGS/$name.log"
      "$@" >> "$LOGS/$name.log" 2>&1
      echo "=== $(date -Is) $name exited ($?), restarting in 5 s" >> "$LOGS/$name.log"
      sleep 5
    done ) &
  echo $! > "$FORTE_HOME/var/$name.supervisor.pid"
}

stop_all() {
  for n in vllm api worker; do
    f=$FORTE_HOME/var/$n.supervisor.pid
    [ -f "$f" ] && kill "$(cat "$f")" 2>/dev/null; rm -f "$f"
  done
  pkill -f "vllm serv[e]" ; pkill -f "uvicorn app.ap[i]" ; pkill -f "app.worke[r]"
  sleep 3; pkill -9 -f "VLLM::EngineCor[e]" 2>/dev/null || true
}

if [ "${1:-start}" = "stop" ]; then stop_all; echo stopped; exit 0; fi
stop_all >/dev/null 2>&1

supervise vllm deploy/run_vllm.sh
echo -n "waiting for vLLM"
for _ in $(seq 1 120); do curl -sf localhost:8100/v1/models >/dev/null && break; echo -n .; sleep 3; done; echo
supervise api "$FORTE_HOME/venv-app/bin/uvicorn" app.api:app --host 0.0.0.0 --port 8000 --proxy-headers \
  --forwarded-allow-ips='*' --timeout-keep-alive 30
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True supervise worker "$FORTE_HOME/venv-app/bin/python" -m app.worker
sleep 5
curl -s localhost:8000/healthz; echo
echo "logs: $LOGS/{vllm,api,worker}.log"
