#!/usr/bin/env bash
# VLM bake-off for hosts run with deploy/start_all.sh (no systemd), e.g. a Vast.ai / Runpod box.
# For each model: write its settings into forte.env, restart the stack, wait for vLLM, score all clips.
# usage: FORTE_HOME=/workspace/forte bash scripts/bakeoff_hosts.sh "name|weights_dir|mm_kwargs|extra_args" ...
set -u
FH=${FORTE_HOME:-$HOME/forte}
cd "$FH/service"
OUT=$FH/var/bakeoff
mkdir -p "$OUT"
cp "$FH/forte.env" "$OUT/forte.env.orig"

setenv() {   # key value: replace or append a line in forte.env (values quoted for bash)
  local k=$1 v=$2
  grep -v "^$k=" "$FH/forte.env" > "$FH/forte.env.tmp"; mv "$FH/forte.env.tmp" "$FH/forte.env"
  printf "%s='%s'\n" "$k" "$v" >> "$FH/forte.env"
}

for spec in "$@"; do
  IFS='|' read -r name weights mm extra <<< "$spec"
  echo "=== $name ($weights) $(date +%T)" | tee -a "$OUT/summary.txt"
  setenv FORTE_VLM_WEIGHTS "$weights"
  setenv VLLM_MM_KWARGS "$mm"
  setenv VLLM_EXTRA_ARGS "$extra"
  FORTE_HOME=$FH bash deploy/start_all.sh stop > /dev/null 2>&1
  : > "$FH/logs/vllm.log"
  FORTE_HOME=$FH nohup bash deploy/start_all.sh > "$OUT/start_$name.log" 2>&1 < /dev/null &
  ready=0
  for _ in $(seq 1 180); do
    curl -sf localhost:8100/v1/models > /dev/null && { ready=1; break; }
    sleep 5
  done
  if [ $ready -eq 0 ]; then
    echo "$name: vLLM did not start: $(grep -m2 -E 'Error|error' "$FH/logs/vllm.log" | cut -c1-200)" | tee -a "$OUT/summary.txt"
    cp "$FH/logs/vllm.log" "$OUT/vllm_$name.log"
    continue
  fi
  sleep 20
  vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader)
  start=$(date +%s)
  FORTE_HOME=$FH PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$FH/venv-app/bin/python" -m scripts.eval \
    --run "bake_$name" > "$OUT/eval_$name.txt" 2>&1
  { echo "$name: VRAM $vram (vLLM + worker detector), $(( $(date +%s) - start ))s for 42 clips"
    grep -E "^accuracy|^timecode \|err\| \(final\)" "$OUT/eval_$name.txt"; } | tee -a "$OUT/summary.txt"
done
cp "$OUT/forte.env.orig" "$FH/forte.env"
echo "BAKEOFF_DONE $(date +%T)" | tee -a "$OUT/summary.txt"
