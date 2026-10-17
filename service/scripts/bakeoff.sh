#!/usr/bin/env bash
# VLM bake-off: same pipeline, same clips, one model at a time.
# For each model: start vLLM, wait until ready, run the scorer with 4 videos in parallel, sample peak VRAM.
# usage: scripts/bakeoff.sh [model_dir ...]      (dirs under ~/forte/models)
set -u
cd "$(dirname "$0")/.."
OUT=$HOME/forte/var/bakeoff
mkdir -p "$OUT"
MODELS=("$@")
[ ${#MODELS[@]} -eq 0 ] && MODELS=(Qwen3.5-4B-AWQ-4bit Qwen3.5-4B-NVFP4 Qwen3-VL-4B-Instruct-AWQ-4bit
                                   Qwen3-VL-2B-Instruct-FP8 Qwen3.5-2B Ovis2.5-2B InternVL3_5-2B InternVL3_5-8B-AWQ-4bit)
QWEN_MM='{"size": {"longest_edge": 589824, "shortest_edge": 3136}}'

stop_vllm() {
  pkill -f "[v]llm serve" ; for _ in $(seq 1 30); do pgrep -f "[v]llm serve" >/dev/null || break; sleep 1; done
  pkill -9 -f "VLLM::EngineCore" 2>/dev/null; sleep 2
}

for m in "${MODELS[@]}"; do
  echo "=== $m $(date +%T)"
  stop_vllm
  case "$m" in
    Ovis*)     mm='{}'; extra="--trust-remote-code" ;;
    InternVL*) mm='{"max_dynamic_patch": 2}'; extra="--trust-remote-code" ;;
    *)         mm="$QWEN_MM"; extra="" ;;
  esac
  FORTE_VLM_WEIGHTS=$m VLLM_MM_KWARGS=$mm VLLM_EXTRA_ARGS=$extra \
    nohup deploy/run_vllm.sh > "$OUT/vllm_$m.log" 2>&1 < /dev/null &
  ready=0
  for _ in $(seq 1 120); do
    curl -sf localhost:8100/v1/models >/dev/null && { ready=1; break; }
    pgrep -f "[v]llm serve" >/dev/null || break
    sleep 3
  done
  if [ $ready -eq 0 ]; then
    echo "$m: vLLM failed to start ($(grep -m1 -E 'Error|error' "$OUT/vllm_$m.log" | cut -c1-200))" | tee -a "$OUT/summary.txt"
    continue
  fi
  idle=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  ( peak=0; while true; do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); \
      [ "$u" -gt "$peak" ] && peak=$u && echo $peak > "$OUT/peak_$m"; sleep 1; done ) &
  sampler=$!
  start=$(date +%s)
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FORTE_HOME=$HOME/forte FORTE_VLM_MODEL=vlm \
    timeout 2400 "$HOME/forte/venv-app/bin/python" -m scripts.eval --run "bake_$m" > "$OUT/eval_$m.txt" 2>&1
  kill $sampler 2>/dev/null
  {
    echo "--- $m: idle VRAM ${idle} MiB, peak $(cat "$OUT/peak_$m" 2>/dev/null) MiB, wall $(( $(date +%s) - start ))s"
    grep -E "^accuracy|timecode" "$OUT/eval_$m.txt"
  } | tee -a "$OUT/summary.txt"
done
stop_vllm
echo BAKEOFF_DONE | tee -a "$OUT/summary.txt"
