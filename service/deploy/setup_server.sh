#!/usr/bin/env bash
# One-shot setup of the Cash Desk Review service on a fresh GPU machine (Runpod pod or bare Linux).
# Idempotent: safe to re-run. Needs internet once; afterwards everything runs offline.
#
#   FORTE_HOME=/workspace/forte VLM_REPO=RedHatAI/Qwen3.5-9B-FP8-dynamic bash deploy/setup_server.sh
#
# Expects the code at $FORTE_HOME/service and the data (optional, for evaluation) at $FORTE_HOME/data.
set -euo pipefail

FORTE_HOME=${FORTE_HOME:-$HOME/forte}
VLM_REPO=${VLM_REPO:-cyankiwi/Qwen3.5-4B-AWQ-4bit}
VLM_DIR=$(basename "$VLM_REPO")
cd "$FORTE_HOME"
mkdir -p models logs var

SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
if ! command -v ffmpeg >/dev/null; then
  $SUDO apt-get update -qq && $SUDO apt-get install -y -qq ffmpeg curl ca-certificates >/dev/null
fi

export PATH=$HOME/.local/bin:$PATH
unset UV_TORCH_BACKEND   # some images pin it (e.g. cu128); we pick the wheel index from the driver below
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=$HOME/.local/bin:$PATH

# CUDA wheel flavour from the driver version
DRIVER_MAJOR=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
TORCH_INDEX=https://download.pytorch.org/whl/cu128
[ "${DRIVER_MAJOR:-0}" -ge 580 ] && TORCH_INDEX=https://download.pytorch.org/whl/cu130
echo "driver $DRIVER_MAJOR -> $TORCH_INDEX"

# VLM server environment
[ -d venv-vllm ] || uv venv -q -p 3.12 venv-vllm
uv pip install -q -p venv-vllm/bin/python "vllm==0.31.0" "huggingface_hub[cli]"

# worker / API environment
[ -d venv-app ] || uv venv -q -p 3.12 venv-app
uv pip install -q -p venv-app/bin/python torch torchvision --index-url "$TORCH_INDEX"
uv pip install -q -p venv-app/bin/python -r service/deploy/requirements-app.txt
# `trackers` pulls the GUI build of OpenCV; keep only the headless one. Removing either deletes the shared cv2
# package, so the headless build is reinstalled afterwards.
uv pip uninstall -q -p venv-app/bin/python opencv-python 2>/dev/null || true
uv pip install -q -p venv-app/bin/python --reinstall "opencv-python-headless==5.0.0.93"

# weights (downloaded once; the service runs offline afterwards)
[ -f "models/$VLM_DIR/config.json" ] || venv-vllm/bin/hf download "$VLM_REPO" --local-dir "models/$VLM_DIR"
venv-app/bin/python -c "from rfdetr import RFDETRSmall; RFDETRSmall(device='cpu')" >/dev/null 2>&1 || true

# settings + API key (kept if it already exists)
if [ ! -f forte.env ]; then
  sed -e "s|^FORTE_HOME=.*|FORTE_HOME=$FORTE_HOME|" \
      -e "s|^FORTE_VLM_WEIGHTS=.*|FORTE_VLM_WEIGHTS=$VLM_DIR|" \
      -e "s|^FORTE_API_KEY=.*|FORTE_API_KEY=$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')|" \
      service/deploy/forte.env.example > forte.env
  chmod 600 forte.env
fi
sed -i "s/\r$//" service/deploy/*.sh service/scripts/*.sh 2>/dev/null || true
chmod +x service/deploy/*.sh service/scripts/*.sh

venv-app/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
echo "SETUP_DONE  model=$VLM_DIR  home=$FORTE_HOME  (start with: bash service/deploy/start_all.sh)"
