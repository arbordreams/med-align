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

echo "Attempting optional flash-attn install from community wheels..."
# Use community wheel repo for flash-attn 2.x (py312, cu121/cu118)
pip install flash-attn==2.5.0 \
  --extra-index-url https://flashattn.github.io/whl/cu121/torch2.4/ \
  || echo "flash-attn skipped (optional for medical pipeline)"

echo "Installation complete!"

