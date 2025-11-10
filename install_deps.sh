#!/bin/bash
# Installation script for TokAlign medical pipeline dependencies.
# - Supports both virtualenv installs (if .venv exists) and global installs.
# - Installs PyTorch 2.8.0 (CUDA 12.8 wheels) first.
# - Installs remaining requirements, excluding torch/flash-attn to avoid conflicts.
# - Installs flash-attn with robust fallbacks (prebuilt wheels first, then source with --no-build-isolation).
# - Verifies flash-attn import; exits non-zero if unavailable.

set -euo pipefail

echo "[install] Python: $(python -V 2>/dev/null || true)"
echo "[install] PIP: $(pip -V 2>/dev/null || true)"

# Activate venv if present; otherwise proceed with global install
if [ -d ".venv" ]; then
  echo "[install] Activating virtual environment at .venv"
  # shellcheck disable=SC1091
  . .venv/bin/activate
else
  echo "[install] No .venv found; proceeding with global installation (sudo not used)."
fi

echo "[install] Upgrading pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel

echo "[install] Installing torch==2.8.0 (CUDA 12.8 wheels)..."
pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128

echo "[install] Installing remaining Python dependencies (excluding torch/flash-attn)..."
# Filter out torch and flash-attn from bulk install to avoid build/ABI issues
TMP_REQ="/tmp/requirements_no_torch_flash.txt"
grep -Ev '^(torch|flash-attn)[[:space:]]*([=<>!]|$)' requirements.txt > "${TMP_REQ}" || true
if [ -s "${TMP_REQ}" ]; then
  pip install --no-cache-dir -r "${TMP_REQ}"
else
  echo "[install] No additional requirements to install from requirements.txt"
fi
rm -f "${TMP_REQ}"

echo "[install] Installing flash-attn (prefer prebuilt wheels)..."
# Try common wheel indexes first (versions vary by CUDA/Torch minor subdir published upstream).
set +e
pip install --no-cache-dir \
  --extra-index-url https://flashattn.github.io/whl/cu128/torch2.8/ \
  --extra-index-url https://flashattn.github.io/whl/cu121/torch2.8/ \
  "flash-attn==2.5.*"
FA_STATUS=$?
if [ $FA_STATUS -ne 0 ]; then
  echo "[install] Prebuilt flash-attn wheels not available; falling back to source build with --no-build-isolation."
  pip install --no-cache-dir --no-build-isolation "flash-attn==2.5.*"
  FA_STATUS=$?
fi
set -e

echo "[install] Verifying flash-attn import..."
python - <<'PY'
import sys
try:
    import flash_attn  # noqa: F401
except Exception as e:
    sys.stderr.write(f"[install] flash-attn import failed: {e}\n")
    sys.exit(1)
print("[install] flash-attn is installed and importable.")
PY

echo "[install] All dependencies installed successfully."

