#!/usr/bin/env bash
# OpenAI-compatible VLM server for the worker. Sized for an 8 GB GPU shared with the RF-DETR detector:
# images are capped at ~590k pixels so profiling and KV cache fit, and vLLM leaves ~30% of VRAM free.
# Per-model overrides: FORTE_VLM_WEIGHTS (dir under ~/forte/models), VLLM_MM_KWARGS, VLLM_EXTRA_ARGS.
MODEL=${FORTE_VLM_WEIGHTS:-Qwen3.5-4B-AWQ-4bit}
MM_KWARGS=${VLLM_MM_KWARGS:-'{"size": {"longest_edge": 589824, "shortest_edge": 3136}}'}
export VLLM_USE_FLASHINFER_SAMPLER=0   # no nvcc on the host; greedy decoding does not need it
FH=${FORTE_HOME:-$HOME/forte}
# CUDA libraries shipped as wheels (e.g. libnvrtc-builtins, used by vLLM's JIT kernels for FP8 on Ampere GPUs)
for d in "$FH"/venv-vllm/lib/python3*/site-packages/nvidia/*/lib; do
  [ -d "$d" ] && LD_LIBRARY_PATH="$d${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
done
export LD_LIBRARY_PATH
# shellcheck disable=SC2086
exec "$FH/venv-vllm/bin/vllm" serve "$FH/models/$MODEL" \
  --served-model-name vlm --host 127.0.0.1 --port 8100 \
  --gpu-memory-utilization "${VLLM_GPU_UTIL:-0.70}" --max-model-len 16384 --max-num-seqs "${VLLM_MAX_SEQS:-12}" \
  --max-num-batched-tokens "${VLLM_BATCHED_TOKENS:-2048}" \
  --limit-mm-per-prompt '{"image": 24, "video": 0}' \
  --mm-processor-kwargs "$MM_KWARGS" \
  --enable-prefix-caching --structured-outputs-config.backend xgrammar \
  --structured-outputs-config.disable_any_whitespace true ${VLLM_EXTRA_ARGS:-}
