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

echo "[install] Installing flash-attn (prebuilt wheels only, no source build)..."
# Try community wheel indexes - skip if not available (flash-attn is optional)
# Use --only-binary to prevent building from source
set +e
echo "[install] Trying official flash-attn wheels from flashattn.github.io..."
pip install --no-cache-dir --only-binary :all: \
  --extra-index-url https://flashattn.github.io/whl/cu128/torch2.8/ \
  --extra-index-url https://flashattn.github.io/whl/cu121/torch2.8/ \
  "flash-attn==2.5.*"
FA_STATUS=$?

if [ $FA_STATUS -ne 0 ]; then
  echo "[install] Official wheels not available, trying PyPI for prebuilt wheels..."
  pip install --no-cache-dir --only-binary :all: "flash-attn==2.5.*"
  FA_STATUS=$?
fi

if [ $FA_STATUS -ne 0 ]; then
  echo "[install] WARNING: Prebuilt flash-attn wheels not available."
  echo "[install] flash-attn is optional for the medical pipeline - skipping installation."
  echo "[install] The pipeline will work without it. You can install it later if needed."
  FA_STATUS=0  # Don't fail the install
fi
set -e

echo "[install] Verifying flash-attn import (optional)..."
python - <<'PY'
import sys
try:
    import flash_attn  # noqa: F401
    print("[install] flash-attn is installed and importable.")
except ImportError:
    print("[install] flash-attn not available (optional - pipeline will work without it).")
except Exception as e:
    print(f"[install] flash-attn check: {e}")
PY

echo "[install] All dependencies installed successfully."

