#!/bin/bash
# Installation script for med-align dependencies
# Handles torch installation with CUDA 12.8 support first

set -e

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Upgrading pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel

echo "Installing torch 2.8.0 with CUDA 12.8 support..."
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128

echo "Installing remaining dependencies..."
# Remove torch line temporarily to avoid reinstall
grep -v "^torch==" requirements.txt > /tmp/requirements_no_torch.txt || true
pip install -r /tmp/requirements_no_torch.txt
rm /tmp/requirements_no_torch.txt

echo "Attempting optional flash-attn install from prebuilt wheel..."
if ! pip install flash-attn==2.5.0 --find-links https://github.com/Dao-AILab/flash-attention/releases/expanded_assets/v2.5.0 2>/dev/null; then
  echo "Prebuilt wheel not found, trying source build..."
  if ! pip install --no-build-isolation flash-attn==2.5.0 2>/dev/null; then
    echo "flash-attn installation failed; continuing without it (optional for medical pipeline)."
  fi
fi

echo "Installation complete!"

