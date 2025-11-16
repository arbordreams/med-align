# GH200 ARM64 Support

This document outlines the adaptations made to support NVIDIA GH200 (ARM64 + H100) systems.

## Architecture Overview

### GH200 (1x GH200 Superchip)

**GH200 Specifications:**
- CPU: ARM64 (aarch64) Grace CPU
- GPU: H100 (96GB - more than standard 80GB H100)
- RAM: 432 GiB
- vCPUs: 64
- Storage: 4 TiB SSD
- **Cost: $1.49/hr** (most cost-effective option)

### H100 Configurations

**1x H100 (80GB PCIe):**
- CPU: x86_64
- GPU: H100 80GB PCIe
- RAM: 200 GiB
- vCPUs: 26
- Storage: 1 TiB SSD
- **Cost: $2.49/hr**

**8x H100 (80GB SXM5):**
- CPU: x86_64
- GPU: 8x H100 80GB SXM5
- RAM: 1800 GiB
- vCPUs: 208
- Storage: 22 TiB SSD
- **Cost: $23.92/hr** ($2.99/GPU/hr)

**Note:** The TokAlign medical pipeline is optimized for single-GPU runs. Multi-GPU configurations (8x H100) are primarily useful for large-scale training beyond the scope of the medical alignment pipeline.

## Changes Made

### 1. Installation Script (`install_deps.sh`)

**Architecture Detection:**
- Auto-detects ARM64 vs x86_64
- Uses appropriate installation method for each

**Flash-Attention:**
- **x86_64**: Uses pre-built wheel (fast)
- **ARM64**: Builds from source (no pre-built wheels available)
  - Limits parallel jobs (`MAX_JOBS=4`) to prevent memory exhaustion
  - Requires CUDA toolkit 12.8+ and build tools
  - Installation failure is non-fatal (pipeline runs slower without it)

**PyTorch:**
- Uses PyTorch's official wheel repository
- Auto-detects architecture and installs appropriate wheel
- CUDA 12.8 support available for both architectures

**Other Dependencies:**
- Most Python packages work automatically (architecture-agnostic)
- `deepspeed` and `bitsandbytes` are Linux-only (work on ARM64 Linux)
- `fasttext-wheel` supports ARM64

### 2. Research Script (`run_medical_pipeline_research.sh`)

**Auto-Scaling:**
- Auto-detects vCPU count using `nproc`
- Scales thread counts automatically:
  - **GH200 (64 vCPUs)**: 64 threads for OMP/MKL, 32 threads per FastText worker
  - **H100 PCIe (26 vCPUs)**: 24 threads for OMP/MKL, 12 threads per FastText worker
  - **8x H100 SXM5 (208 vCPUs)**: Capped at 64 threads (optimal for single-GPU pipeline)

**Resource Utilization:**
- Takes advantage of GH200's additional resources compared to 1x H100 PCIe:
  - More GPU memory (96GB vs 80GB) → larger batch sizes possible
  - More RAM (432GB vs 200GB) → more data in memory
  - More vCPUs (64 vs 26) → faster FastText training (2.5x more CPUs)

## Known Issues and Limitations

### Flash-Attention on ARM64

**Status:** Challenging but possible

**Issues:**
- No pre-built wheels available
- Compilation is memory-intensive (can use 8GB+)
- Build time: 1-2 hours
- May require additional build tools

**Solutions:**
1. **Limit parallel jobs**: `MAX_JOBS=4` (already implemented)
2. **Ensure sufficient RAM**: At least 8GB free during compilation
3. **Install build tools**: `gcc`, `make`, `ninja`, CUDA toolkit
4. **Fallback**: Pipeline works without flash-attn (slower but functional)

### PyTorch on ARM64

**Status:** Supported

- Official PyTorch wheels available for ARM64 with CUDA
- Auto-detected by pip when using `--index-url https://download.pytorch.org/whl/cu128`
- No manual intervention needed

### Other Dependencies

**Status:** Generally compatible

- Most Python packages are architecture-agnostic
- `deepspeed` and `bitsandbytes` work on ARM64 Linux
- `fasttext-wheel` has ARM64 support

## Installation Instructions

### Prerequisites

1. **CUDA Toolkit 12.8+**
   ```bash
   # Verify CUDA is installed
   nvcc --version
   ```

2. **Build Tools** (for flash-attn compilation)
   ```bash
   # Ubuntu/Debian
   sudo apt-get install build-essential gcc g++ make ninja-build
   ```

3. **Python 3.12+**
   ```bash
   python --version  # Should be 3.12.x
   ```

### Installation

```bash
# Run the installation script
bash install_deps.sh
```

The script will:
1. Detect ARM64 architecture
2. Install PyTorch (auto-detects ARM64 wheel)
3. Install other dependencies
4. Attempt to build flash-attn from source (with MAX_JOBS=4)
5. Continue even if flash-attn fails (non-fatal)

### Verification

```bash
# Check PyTorch
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

# Check flash-attn (optional)
python -c "import flash_attn; print('Flash-attn available')" || echo "Flash-attn not available (pipeline will run slower)"
```

## Performance Expectations

### With Flash-Attn
- **Training speed**: Similar to x86_64 H100
- **Memory usage**: Similar to x86_64 H100
- **Compilation time**: 1-2 hours (one-time cost)

### Without Flash-Attn
- **Training speed**: 10-20% slower (uses standard attention)
- **Memory usage**: Similar
- **Functionality**: Fully functional, just slower

## Troubleshooting

### Flash-Attn Compilation Fails

**Symptoms:**
- Build fails with memory errors
- Build fails with CUDA errors
- Build takes too long

**Solutions:**
1. **Reduce parallel jobs**: Already set to MAX_JOBS=4
2. **Increase available RAM**: Close other applications
3. **Check CUDA toolkit**: Ensure CUDA 12.8+ is installed
4. **Check build tools**: Ensure gcc, make, ninja are installed
5. **Skip flash-attn**: Pipeline works without it

### PyTorch Installation Issues

**Symptoms:**
- PyTorch wheel not found
- CUDA not detected

**Solutions:**
1. **Check architecture**: `uname -m` should show `aarch64`
2. **Check CUDA**: `nvcc --version` should show 12.8+
3. **Manual install**: 
   ```bash
   pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
   ```

### Performance Issues

**Symptoms:**
- Slower than expected
- Not using all vCPUs

**Solutions:**
1. **Check thread counts**: Script auto-detects, but you can override:
   ```bash
   export OMP_NUM_THREADS=64
   export MKL_NUM_THREADS=64
   export FASTTEXT_THREAD=32
   ```
2. **Check flash-attn**: Ensure it's installed if you want optimal performance
3. **Check GPU**: `nvidia-smi` should show H100

## Verification

The install script automatically verifies PyTorch and CUDA availability after installation. This ensures the pipeline is ready to run.

## References

- [Flash-Attention ARM64 Issue](https://github.com/Dao-AILab/flash-attention/issues/879)
- [PyTorch ARM64 Support](https://pytorch.org/get-started/locally/)
- [GH200 Documentation](https://www.nvidia.com/en-us/data-center/grace-hopper-superchip/)

## Runtime Performance Comparison

### Estimated Runtimes (5GB corpus, research mode)

**1x H100 PCIe (x86_64, 26 vCPUs, 80GB GPU, 200GB RAM):**
- Data prep: 30-45 min
- Term mining: 10-15 min
- Tokenization: 2-3 hours
- FastText (30 epochs): 3-4 hours
- Alignment: 30-45 min
- Model adaptation: 15-30 min
- Evaluation: 20-30 min
- **TOTAL: ~6-10 hours**
- **Cost: $2.49/hr → $14.94-24.90 per run**

**1x GH200 (ARM64, 64 vCPUs, 96GB GPU, 432GB RAM):**
- Data prep: 15-25 min (2.5x faster)
- Term mining: 5-8 min (2x faster)
- Tokenization: 1-1.5 hours (2x faster)
- FastText (30 epochs): 1.5-2 hours (2.5x faster)
- Alignment: 15-20 min (2x faster)
- Model adaptation: 10-20 min (slightly faster)
- Evaluation: 15-25 min (slightly faster)
- **TOTAL: ~3-5 hours**
- **Cost: $1.49/hr → $4.47-7.45 per run**

**8x H100 SXM5 (x86_64, 208 vCPUs, 8x 80GB GPU, 1800GB RAM):**
- **Note:** The medical pipeline is single-GPU optimized. Multi-GPU configurations provide minimal benefit for this workload.
- Runtime similar to 1x H100 PCIe (CPU-bound tasks benefit from more vCPUs, but GPU tasks are single-GPU)
- **Cost: $23.92/hr → $143.52-239.20 per run** (not cost-effective for single-GPU pipeline)

### Performance Factors

**CPU-Bound Tasks (GH200 advantage over 1x H100 PCIe):**
- 2.5x more vCPUs (64 vs 26) → 2-2.5x faster
- FastText training: 2.5x faster (32 threads × 2 workers vs 12 × 2)
- Tokenization: 2x faster (more workers)
- Data prep/alignment: 2x faster

**GPU-Bound Tasks (GH200 slight advantage):**
- 20% more GPU memory (96GB vs 80GB)
- Better CPU-GPU bandwidth (NVLink-C2C)
- Up to 17% faster inference (MLPerf benchmarks)
- Can use larger batch sizes

**Memory-Bound Tasks (GH200 advantage):**
- 2.2x more RAM (432GB vs 200GB)
- More data fits in memory
- Less swapping/disk I/O

### Overall Speedup

**1.5-2x faster on GH200** compared to 1x H100 PCIe for the full pipeline

**Key Bottlenecks:**
- **1x H100 PCIe**: FastText (3-4h) and Tokenization (2-3h) are CPU-bound
- **1x GH200**: FastText (1.5-2h) is still longest but much faster
- **8x H100 SXM5**: Not recommended for this pipeline (single-GPU optimized, minimal benefit from multi-GPU)

### Cost Comparison

**1x H100 PCIe (80GB):**
- Runtime: 6-10 hours
- Cost: $2.49/hour = **$14.94-24.90 per run**

**1x GH200 (96GB):**
- Runtime: 3-5 hours
- Cost: $1.49/hour = **$4.47-7.45 per run**
- **3-5x cheaper per run!**

**8x H100 SXM5 (80GB):**
- Runtime: 6-10 hours (similar to 1x H100 PCIe for single-GPU pipeline)
- Cost: $23.92/hour = **$143.52-239.20 per run**
- **Not recommended** for this single-GPU optimized pipeline

### First Run Considerations

**1x GH200 First Run:**
- Includes flash-attn compilation: **4-6 hours total**
- Without flash-attn: **3.5-5.5 hours** (still faster than 1x H100 PCIe)

**Subsequent Runs:**
- 1x GH200: **3-5 hours** (faster!)
- 1x H100 PCIe: **6-10 hours** (same)

## Summary

**Difficulty:** Easy to Medium

**Main Challenge:** Flash-attn compilation (optional)

**Time Estimate:**
- Quick install (without flash-attn): 30 minutes
- Full install (with flash-attn): 2-3 hours

**Performance:**
- **1.5-2x faster** than 1x H100 PCIe for full pipeline
- **3-5x cheaper** per run ($4.47-7.45 vs $14.94-24.90)

**Recommendation:** 
- Start without flash-attn to verify pipeline works
- Build flash-attn later if needed for optimal performance
- Pipeline is fully functional without flash-attn (just slower)
- GH200 offers significant speedup and cost savings
