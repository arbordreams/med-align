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

echo "[install] Installing torch and compatible torchvision (CUDA 12.8 wheels)..."
# Install torch and torchvision together to ensure compatibility
# Use --user to override incompatible system torchvision (0.22.0 from apt)
# Note: torchvision 0.24.0 requires torch 2.9.0, so we install compatible versions
pip install --user --force-reinstall --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu128

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

echo "[install] Installing flash-attn (required)..."
# Try multiple wheel sources for torch 2.9.0 compatibility
set +e
FA_STATUS=1

# Check torch version to determine which wheel to use
TORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null | cut -d'+' -f1)
echo "[install] Detected PyTorch version: ${TORCH_VERSION}"

# Try torch 2.9 wheel from community source (windreamer)
if [[ "${TORCH_VERSION}" == "2.9.0" ]]; then
  echo "[install] Attempting to install flash-attn from community wheels (torch 2.9.0)..."
  # Try using pip with find-links to community wheel index
  pip install --no-cache-dir flash-attn --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch290 || \
  pip install --no-cache-dir flash-attn --index-url https://windreamer.github.io/flash-attention3-wheels/cu128_torch290
  FA_STATUS=$?
fi

# If community wheels failed, try official torch 2.8 wheel (may work with 2.9)
if [ $FA_STATUS -ne 0 ]; then
  echo "[install] Trying official torch 2.8 wheel (may be compatible with torch 2.9)..."
  WHEEL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
  pip install --no-cache-dir "${WHEEL_URL}"
  FA_STATUS=$?
fi

# Verify import works (even if install succeeded, symbol mismatch may occur)
if [ $FA_STATUS -eq 0 ]; then
  echo "[install] Verifying flash-attn import..."
  python -c "import flash_attn" 2>/dev/null
  if [ $? -ne 0 ]; then
    echo "[install] WARNING: flash-attn installed but import failed (symbol mismatch)."
    echo "[install] This may be due to PyTorch version incompatibility."
    FA_STATUS=1
  fi
fi

if [ $FA_STATUS -ne 0 ]; then
  echo "[install] ERROR: Failed to install working flash-attn wheel for torch ${TORCH_VERSION}."
  echo "[install] No compatible pre-built wheel found. Please check for available wheels at:"
  echo "[install]   - https://github.com/Dao-AILab/flash-attention/releases"
  echo "[install]   - https://windreamer.github.io/flash-attention3-wheels/"
  exit 1
else
  echo "[install] Successfully installed flash-attn"
fi
set -e

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

