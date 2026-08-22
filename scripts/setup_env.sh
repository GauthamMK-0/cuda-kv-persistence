#!/usr/bin/env bash
# Recreate the Python environment for motion-gated-kv-cache from scratch.
# Mirrors the known-good venv this project was developed against
# (cloned originally from /root/projects/edgeRunner/.venv).
#
# Usage:  bash scripts/setup_env.sh
# Safe to re-run: skips creation when an existing .venv already imports torch.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.venv"
PY="$VENV/bin/python"

TORCH_VERSION="2.5.1+cu121"
TV_VERSION="0.20.1+cu121"
TA_VERSION="2.5.1+cu121"
CUDA_INDEX="https://download.pytorch.org/whl/cu121"

echo "==> Project root: $ROOT"

if [ -x "$PY" ] && "$PY" -c "import torch" >/dev/null 2>&1; then
    echo "==> Existing .venv with torch found; nothing to do."
    "$PY" -c "import torch; print('    torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"
    exit 0
fi

echo "==> Creating virtualenv at $VENV (python3 required)"
python3 -m venv "$VENV"

echo "==> Upgrading pip tooling"
"$VENV/bin/pip" install --upgrade pip setuptools wheel

echo "==> Installing PyTorch $TORCH_VERSION (CUDA 12.1 wheels) from $CUDA_INDEX"
"$VENV/bin/pip" install \
    "torch==$TORCH_VERSION" \
    "torchvision==$TV_VERSION" \
    "torchaudio==$TA_VERSION" \
    --index-url "$CUDA_INDEX"

echo "==> Installing supporting packages"
"$VENV/bin/pip" install \
    "numpy==2.4.4" \
    "opencv-python-headless==5.0.0.93" \
    "einops==0.8.2" \
    "matplotlib==3.11.1" \
    "scipy==1.18.0" \
    "pytest"

echo "==> Verifying installation"
"$PY" - <<'EOF'
import torch, cv2, numpy, einops
assert torch.backends.cuda.is_built(), "torch built without CUDA"
print(f"    torch {torch.__version__} | cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"    GPU: {torch.cuda.get_device_name(0)} | cc: {torch.cuda.get_device_capability(0)}")
print(f"    opencv {cv2.__version__} | numpy {numpy.__version__}")
EOF

echo "==> Environment ready."
