<!-- f11e1783-e84f-48f6-8df8-7df3ad9c1ba5 a2001844-abde-402f-8144-8e29dfc202e6 -->
# Fix Keras Compatibility for Transformers

## Problem

Transformers library requires `tf-keras` (backwards-compatible Keras) but Keras 3 is installed, causing import failures when transformers tries to load TensorFlow-related modules.

## Solution

Install `tf-keras` package in the install script to provide the backwards-compatible Keras that transformers expects.

## Changes

### 1. Update `install_deps.sh`

Add installation of `tf-keras` after installing other dependencies but before flash-attn verification:

```bash
# After line 58 (after installing requirements), add:
echo "[install] Installing tf-keras for transformers TensorFlow compatibility..."
pip install --no-cache-dir tf-keras
```

This ensures transformers can import TensorFlow-related modules without errors.

2.  Add to `requirements.txt`

add`tf-keras` to requirements.txt for documentation.



Testing

After installation, transformers should be able to import TensorFlow modules without the Keras 3 compatibility error.

### To-dos

- [ ] Add tf-keras installation to install_deps.sh after requirements installation
- [ ] Verify transformers can import TensorFlow modules after tf-keras installation