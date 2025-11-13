#!/bin/bash
# Installation script for TokAlign medical pipeline dependencies.
# - Supports both virtualenv installs (if .venv exists) and global installs.
# - Works on Lambda Labs 24.04, RunPod, and other CUDA 12.8 environments.
# - Installs PyTorch 2.9.1 (CUDA 12.8 wheels) first (torch 2.8.0 not available).
# - Installs remaining requirements, excluding torch/flash-attn to avoid conflicts.
# - Pulls in fasttext-wheel>=0.9.2 for the FastText embedding backend (prebuilt for Python 3.12).
# - Installs flash-attn with robust fallbacks (prebuilt wheels first, then source with --no-build-isolation).
# - Verifies flash-attn import; continues without it if unavailable (non-fatal).

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

echo "[install] Installing torch==2.9.1 and compatible torchvision (CUDA 12.8 wheels)..."
# Install torch 2.9.1 (torch 2.8.0 not available in CUDA 12.8 wheels) with compatible torchvision
# Use --user flag only if not in venv (venv detection happens above)
# torchvision 0.24.1 is compatible with torch 2.9.1
# Lambda Labs 24.04 images may have torch pre-installed, so we use --force-reinstall
PIP_USER_FLAG=""
if [ -z "${VIRTUAL_ENV:-}" ]; then
  PIP_USER_FLAG="--user"
fi

# Check if torch is already installed and what version
if python -c "import torch; print(torch.__version__)" 2>/dev/null; then
  CURRENT_TORCH=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "")
  echo "[install] Detected existing PyTorch: ${CURRENT_TORCH}"
  echo "[install] Reinstalling with CUDA 12.8 wheels to ensure compatibility..."
fi

pip install ${PIP_USER_FLAG} --force-reinstall --no-cache-dir torch==2.9.1+cu128 torchvision==0.24.1+cu128 --index-url https://download.pytorch.org/whl/cu128

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

echo "[install] Installing tf-keras for transformers TensorFlow compatibility..."
pip install --no-cache-dir tf-keras

echo "[install] Installing flash-attn (optional, for torch 2.9.1 compatibility)..."
# Detect architecture for flash-attn wheel
ARCH=$(uname -m)
PY_VER=$(python -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
TORCH_VER=$(python -c "import torch; print(torch.__version__.split('+')[0])" 2>/dev/null || echo "2.9.1")
echo "[install] Detected architecture: ${ARCH}, Python: ${PY_VER}, PyTorch: ${TORCH_VER}"

if [ "${ARCH}" = "aarch64" ] || [ "${ARCH}" = "arm64" ]; then
  echo "[install] ARM64 detected - attempting to install flash-attn from source..."
  echo "[install] NOTE: Flash-attn compilation on ARM64 can be memory-intensive and time-consuming."
  echo "[install] Limiting parallel jobs to avoid memory exhaustion (MAX_JOBS=2)..."
  echo "[install] This may take 10-30 minutes depending on system resources."
  # Try to install from source for ARM64 with limited parallel jobs
  # Use MAX_JOBS=2 to reduce memory pressure on ARM64 systems
  # Flash-attn 2.8.3 should work with torch 2.9.1 when built from source
  if ! MAX_JOBS=2 pip install --no-cache-dir flash-attn==2.8.3 --no-build-isolation 2>&1 | tee /tmp/flash_attn_build.log; then
    echo "[install] WARNING: flash-attn source build failed."
    echo "[install] Common causes:"
    echo "[install]   - Insufficient RAM (compilation can use 8GB+)"
    echo "[install]   - Missing CUDA toolkit 12.8+"
    echo "[install]   - Missing build tools (gcc, make, ninja)"
    echo "[install]   - CUDA/compiler version mismatch"
    echo "[install] Continuing without flash-attn (pipeline will run slower but still functional)."
    echo "[install] Build log saved to /tmp/flash_attn_build.log"
    echo "[install] You can retry flash-attn installation later if needed."
    # Don't fail the entire installation - flash-attn is optional for basic functionality
  else
    echo "[install] flash-attn installation completed successfully"
  fi
else
  # x86_64: try pre-built wheel first, then fall back to source
  # Note: Pre-built wheels for torch 2.9.1 may not exist, so we'll likely build from source
  echo "[install] Attempting to install flash-attn for torch ${TORCH_VER}..."
  if ! pip install --no-cache-dir flash-attn==2.8.3 --no-build-isolation 2>&1 | tee /tmp/flash_attn_build.log; then
    echo "[install] WARNING: flash-attn installation failed."
    echo "[install] Continuing without flash-attn (pipeline will run slower but still functional)."
    echo "[install] Build log saved to /tmp/flash_attn_build.log"
  else
    echo "[install] flash-attn installation completed successfully"
  fi
fi

echo "[install] Verifying flash-attn import..."
python - <<'PY'
import sys
try:
    import flash_attn  # noqa: F401
    print("[install] ✓ flash-attn is installed and importable.")
except Exception as e:
    print(f"[install] WARNING: flash-attn import failed: {e}")
    print("[install] Pipeline will run without flash-attn (slower but functional).")
    print("[install] To fix: ensure CUDA toolkit and build tools are installed, then rebuild.")
    # Don't exit - flash-attn is optional for basic functionality
PY

echo "[install] Verifying PyTorch CUDA availability..."
python - <<'PY'
import sys
try:
    import torch
    print(f"[install] PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"[install] ✓ CUDA available in PyTorch")
        print(f"[install] ✓ PyTorch CUDA version: {torch.version.cuda}")
        print(f"[install] ✓ GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"[install]   - GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("[install] WARNING: CUDA not available in PyTorch")
        print("[install] This may indicate:")
        print("[install]   - PyTorch was installed without CUDA support")
        print("[install]   - CUDA drivers are not properly installed")
        print("[install]   - Architecture mismatch (ARM64 vs x86_64)")
        sys.exit(1)
except ImportError:
    print("[install] ERROR: PyTorch not installed")
    sys.exit(1)
except Exception as e:
    print(f"[install] ERROR: PyTorch verification failed: {e}")
    sys.exit(1)
PY

echo "[install] All dependencies installed successfully."

