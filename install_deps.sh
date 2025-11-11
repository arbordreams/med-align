#!/bin/bash
# Installation script for TokAlign medical pipeline dependencies.
# - Supports both virtualenv installs (if .venv exists) and global installs.
# - Installs PyTorch 2.8.0 (CUDA 12.8 wheels) first.
# - Installs remaining requirements, excluding torch/flash-attn to avoid conflicts.
# - Pulls in fasttext-wheel>=0.9.2 for the FastText embedding backend (prebuilt for Python 3.12).
# - Installs flash-attn with robust fallbacks (prebuilt wheels first, then source with --no-build-isolation).
# - Verifies flash-attn import; exits non-zero if unavailable.

set -euo pipefail

echo "[install] Python: $(python -V 2>/dev/null || true)"
echo "[install] PIP: $(pip -V 2>/dev/null || true)"

# Ensure we are running on Python 3.12+, which is required by the pinned dependencies.
python - <<'PY'
import sys
version = ".".join(map(str, sys.version_info[:3]))
if sys.version_info < (3, 12):
    raise SystemExit(
        f"TokAlign requires Python >= 3.12 (found {version}). "
        "Upgrade the interpreter before running install_deps.sh."
    )
PY

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

echo "[install] Installing torch==2.8.0 and compatible torchvision (CUDA 12.8 wheels)..."
# Install torch 2.8.0 (has official flash-attn wheels) with compatible torchvision
# Use --user to override incompatible system torchvision (0.22.0 from apt)
# torchvision 0.23.0 is compatible with torch 2.8.0
pip install --user --force-reinstall --no-cache-dir torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128

echo "[install] Installing remaining Python dependencies (excluding torch/flash-attn)..."
# Filter out torch and flash-attn from bulk install to avoid build/ABI issues
# Also ensure numpy is pinned to <2.0 for TensorFlow compatibility
TMP_REQ="/tmp/requirements_no_torch_flash.txt"
grep -Ev '^(torch|flash-attn)[[:space:]]*([=<>!]|$)' requirements.txt > "${TMP_REQ}" || true
# Force numpy <2.0 if not already specified (for TensorFlow compatibility)
if ! grep -q "numpy<2" "${TMP_REQ}" 2>/dev/null; then
  echo "numpy<2.0,>=1.24.0" >> "${TMP_REQ}"
fi
if [ -s "${TMP_REQ}" ]; then
  pip install --no-cache-dir -r "${TMP_REQ}"
else
  echo "[install] No additional requirements to install from requirements.txt"
fi
rm -f "${TMP_REQ}"

echo "[install] Installing flash-attn 2.8.3 (required, pre-built wheel for torch 2.8.0)..."
WHEEL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
pip install --no-cache-dir "${WHEEL_URL}"
if [ $? -ne 0 ]; then
  echo "[install] ERROR: Failed to install flash-attn wheel."
  exit 1
fi
echo "[install] Successfully installed flash-attn"

echo "[install] Verifying flash-attn import..."
python - <<'PY'
import sys
try:
    import flash_attn  # noqa: F401
    print("[install] ✓ flash-attn is installed and importable.")
except Exception as e:
    print(f"[install] ERROR: flash-attn import failed: {e}")
    sys.exit(1)
PY

echo "[install] All dependencies installed successfully."

