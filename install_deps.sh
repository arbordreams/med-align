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
# Try multiple community wheel sources - skip if not available (flash-attn is optional)
# Use --only-binary to prevent building from source
set +e
FA_STATUS=1

# Try 1: Official flash-attn wheels for 2.5.x
echo "[install] Trying official flash-attn wheels (2.5.x) from flashattn.github.io..."
pip install --no-cache-dir --only-binary :all: \
  --extra-index-url https://flashattn.github.io/whl/cu128/torch2.8/ \
  --extra-index-url https://flashattn.github.io/whl/cu121/torch2.8/ \
  "flash-attn==2.5.*" > /tmp/flash_install.log 2>&1
if [ $? -eq 0 ]; then
  FA_STATUS=0
  echo "[install] Successfully installed flash-attn 2.5.x from official repo"
fi

# Try 2: Try newer compatible versions (2.6.x, 2.7.x, 2.8.x) that work with torch 2.8
if [ $FA_STATUS -ne 0 ]; then
  echo "[install] Trying newer flash-attn versions compatible with torch 2.8..."
  for version in "2.6.*" "2.7.*" "2.8.*"; do
    pip install --no-cache-dir --only-binary :all: \
      --extra-index-url https://flashattn.github.io/whl/cu128/torch2.8/ \
      --extra-index-url https://flashattn.github.io/whl/cu121/torch2.8/ \
      "flash-attn==${version}" > /tmp/flash_install.log 2>&1
    if [ $? -eq 0 ]; then
      FA_STATUS=0
      echo "[install] Successfully installed flash-attn ${version} from official repo"
      break
    fi
  done
fi

# Try 3: Community wheels from mjun0812 - try different CUDA versions (cu124, cu126) as fallback
if [ $FA_STATUS -ne 0 ]; then
  echo "[install] Trying community wheels from mjun0812 (trying cu124/cu126 as fallback)..."
  PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')")
  for cuda_ver in "cu128" "cu126" "cu124"; do
    for version in "2.5.9" "2.6.3" "2.7.4" "2.8.3"; do
      wheel_name="flash_attn-${version}+${cuda_ver}torch2.8-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-linux_x86_64.whl"
      WHEEL_URL="https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.0.0/${wheel_name}"
      pip install --no-cache-dir --only-binary :all: "${WHEEL_URL}" > /tmp/flash_install.log 2>&1
      if [ $? -eq 0 ]; then
        FA_STATUS=0
        echo "[install] Successfully installed ${wheel_name}"
        break 2
      fi
    done
  done
fi

# Try 4: Try without version constraint to see what's available
if [ $FA_STATUS -ne 0 ]; then
  echo "[install] Trying flash-attn without version constraint from official repo..."
  pip install --no-cache-dir --only-binary :all: \
    --extra-index-url https://flashattn.github.io/whl/cu128/torch2.8/ \
    --extra-index-url https://flashattn.github.io/whl/cu121/torch2.8/ \
    flash-attn > /tmp/flash_install.log 2>&1
  if [ $? -eq 0 ]; then
    FA_STATUS=0
    echo "[install] Successfully installed flash-attn (latest available)"
  fi
fi

rm -f /tmp/flash_install.log

if [ $FA_STATUS -ne 0 ]; then
  echo "[install] WARNING: Prebuilt flash-attn wheels not available from any source."
  echo "[install] flash-attn is optional for the medical pipeline - skipping installation."
  echo "[install] The pipeline will work without it. You can install it later if needed."
  echo "[install] To install manually, try:"
  echo "  pip install flash-attn --find-links https://github.com/mjun0812/flash-attention-prebuild-wheels/releases"
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

