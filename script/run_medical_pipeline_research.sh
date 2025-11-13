#!/usr/bin/env bash
# Research-grade overnight run for TokAlign medical pipeline on H100 (80GB) or GH200 (96GB).
# Quality-first configuration targeting ~5GB corpus and stable outputs.
#
# Requirements:
# - Python 3.12.x
# - CUDA 12.8, Torch 2.8.0 + flash-attn 2.8.3
# - Recommended: 1x GH200 (96GB, ARM64) or 1x H100 PCIe (80GB, x86_64)
#   - 1x GH200: 64 vCPUs, 432 GiB RAM, $1.49/hr (recommended for cost efficiency)
#   - 1x H100 PCIe: 26 vCPUs, 200 GiB RAM, $2.49/hr
#   - 8x H100 SXM5: Not recommended (single-GPU optimized pipeline)
# - Supports both x86_64 and ARM64 architectures
#
# You may override the defaults with environment variables before invoking.
#
# Inputs (env overrides):
#   SRC_MODEL, SRC_TOK, TGT_TOK          - identifiers for source model/tokenizers
#   RUN_ROOT                              - run root (default: runs/tokenizer_adapt)
#   CORPUS_SIZE_GB                        - target corpus size in GB (default: 5.0)
#   BYTE_BUDGET                           - corpus cap in bytes (overrides CORPUS_SIZE_GB)
#   TERM_TOP_K, MIN_TERM_FREQ             - term mining knobs
#   PIVOT_COUNT                           - alignment pivots
#   FASTTEXT_EPOCHS, FASTTEXT_MINCOUNT    - FastText quality knobs
#   FASTTEXT_LR, FASTTEXT_THREAD          - FastText LR and CPU threads
#   EVAL_DATASET                          - HF dataset for perplexity
#   MAX_EVAL                              - evaluation sample cap
#   HF_TOKEN                              - token for gated datasets (optional)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# Hardware/perf environment
# Auto-detect vCPU count for optimal thread settings (GH200 has 64 vCPUs, H100 typically 24)
NUM_VCPUS=$(nproc 2>/dev/null || echo "24")
# Use up to 64 threads on GH200, but cap at available vCPUs
OPTIMAL_THREADS=$((NUM_VCPUS > 64 ? 64 : NUM_VCPUS))
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${OPTIMAL_THREADS}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${OPTIMAL_THREADS}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export NVIDIA_TF32_OVERRIDE="${NVIDIA_TF32_OVERRIDE:-1}"   # enable TF32 on Hopper
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256,expandable_segments:True}"

# HF cache (persist under mounted volume if available)
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${ROOT_DIR}/.cache/huggingface}"
mkdir -p "${HF_DATASETS_CACHE}" || true

echo "[research] Updating repository (git pull)..."
git pull --rebase --autostash origin main || true

SRC_MODEL="${SRC_MODEL:-BioMistral/BioMistral-7B}"
SRC_TOK="${SRC_TOK:-BioMistral/BioMistral-7B}"
TGT_TOK="${TGT_TOK:-mistralai/Mistral-7B-v0.3}"
RUN_ROOT="${RUN_ROOT:-runs/tokenizer_adapt}"

# Corpus size (default 5GB for better coverage)
CORPUS_SIZE_GB="${CORPUS_SIZE_GB:-5.0}"
# Derive byte budget from CORPUS_SIZE_GB unless explicitly provided
if [[ -z "${BYTE_BUDGET:-}" ]]; then
  BYTE_BUDGET="$(python - <<'PY'
import os
size_gb=float(os.environ.get('CORPUS_SIZE_GB','5.0'))
print(int(size_gb*1073741824))
PY
)"
fi

# Research-grade defaults (keep performance knobs; logical knobs now come from config unless explicitly overridden)
USE_TFIDF="--use-tfidf"
FASTTEXT_EPOCHS="${FASTTEXT_EPOCHS:-30}"
FASTTEXT_MINCOUNT="${FASTTEXT_MINCOUNT:-1}"
FASTTEXT_LR="${FASTTEXT_LR:-0.05}"
# FastText threading: 2 workers × N threads = 2N threads total
# Auto-scale based on available vCPUs (GH200: 64 vCPUs → 32 threads per worker = 64 total)
# H100: 24 vCPUs → 12 threads per worker = 24 total (default)
OPTIMAL_FASTTEXT_THREAD=$((OPTIMAL_THREADS / 2))
FASTTEXT_THREAD="${FASTTEXT_THREAD:-${OPTIMAL_FASTTEXT_THREAD}}"
# Tokenizer workers: auto-scale based on vCPUs
# For systems with many vCPUs (GH200): use 75% (optimal for I/O, avoids contention)
# For smaller systems (H100): use 100% (use all available vCPUs)
# GH200: 64 vCPUs → 48 workers (75% of 64, optimal for I/O)
# H100: 24 vCPUs → 24 workers (100% of 24, use all available)
if [ "${OPTIMAL_THREADS}" -ge 48 ]; then
  # Large systems: use 75% (optimal for I/O bound workloads)
  OPTIMAL_TOKENIZER_WORKERS=$((OPTIMAL_THREADS * 3 / 4))
else
  # Smaller systems: use 100% (use all available)
  OPTIMAL_TOKENIZER_WORKERS="${OPTIMAL_THREADS}"
fi
# Cap at 64 workers (diminishing returns beyond this due to I/O limits)
if [ "${OPTIMAL_TOKENIZER_WORKERS}" -gt 64 ]; then
  OPTIMAL_TOKENIZER_WORKERS=64
fi
TOKENIZER_WORKERS="${TOKENIZER_WORKERS:-${OPTIMAL_TOKENIZER_WORKERS}}"
# Similarity threshold (used by alignment step)
SIMILARITY_THRESHOLD_USER_OVERRIDE=false
if [[ -n "${SIMILARITY_THRESHOLD+x}" ]]; then
  SIMILARITY_THRESHOLD_USER_OVERRIDE=true
fi
SIMILARITY_THRESHOLD="${SIMILARITY_THRESHOLD:-0.3}"

# Log detected hardware and auto-scaled settings
echo "[research] Hardware detection:"
echo "[research]   - Detected vCPUs: ${NUM_VCPUS}"
echo "[research]   - Optimal threads: ${OPTIMAL_THREADS}"
echo "[research]   - FastText threads per worker: ${FASTTEXT_THREAD} (2 workers = $((FASTTEXT_THREAD * 2)) total)"
echo "[research]   - Tokenizer workers: ${TOKENIZER_WORKERS}"
echo "[research]   - OMP_NUM_THREADS: ${OMP_NUM_THREADS}"
echo "[research]   - MKL_NUM_THREADS: ${MKL_NUM_THREADS}"

# Evaluation defaults
EVAL_DATASET="${EVAL_DATASET:-uiyunkim-hub/pubmed-abstract:train}"
MAX_EVAL="${MAX_EVAL:-1000}"

echo "[research] Installing dependencies..."
chmod +x "${ROOT_DIR}/install_deps.sh"
bash "${ROOT_DIR}/install_deps.sh"

echo "[research] Building medical corpus (up to ${BYTE_BUDGET} bytes)..."
export MAIN_DIR="${ROOT_DIR}"
CORPUS_DIR="${ROOT_DIR}/runs/corpora/research_${CORPUS_SIZE_GB}gb"
export MEDICAL_CORPUS_DIR="${CORPUS_DIR}"
export MEDICAL_DATASETS="${MEDICAL_DATASETS:-pubmed_abstract}"
export MEDICAL_MAX_SAMPLES="${MEDICAL_MAX_SAMPLES:-0}"
export MEDICAL_BYTE_BUDGET="${BYTE_BUDGET}"
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN="${HF_TOKEN}"
fi
bash "${ROOT_DIR}/script/build_medical_corpus.sh"

AGGREGATED_CORPUS="${CORPUS_DIR}/aggregated/medical_corpus.jsonl"
if [[ ! -f "${AGGREGATED_CORPUS}" ]]; then
  echo "[research] ERROR: Failed to build medical corpus. Check logs above."
  exit 1
fi
echo "[research] Medical corpus ready: ${AGGREGATED_CORPUS}"

echo "[research] Launching research-grade pipeline..."
PY_ARGS=(
  "--input" "${AGGREGATED_CORPUS}"
  "--source-tokenizer" "${SRC_TOK}"
  "--target-tokenizer" "${TGT_TOK}"
  "--source-model" "${SRC_MODEL}"
  "--run-root" "${RUN_ROOT}"
  "--evaluate"
  "--qa"
  "--research-mode"
)
# If a CONFIG_FILE is provided, prefer it and avoid passing knobs unless explicitly overridden
if [[ -n "${CONFIG_FILE:-}" ]]; then
  PY_ARGS+=( "--config" "${CONFIG_FILE}" )
  # Allow explicit env overrides to win over config
  if [[ -n "${EVAL_DATASET:-}" ]]; then PY_ARGS+=( "--evaluation-dataset" "${EVAL_DATASET}" ); fi
  if [[ -n "${MAX_EVAL:-}" ]]; then PY_ARGS+=( "--max-eval-samples" "${MAX_EVAL}" ); fi
  if [[ -n "${TOKENIZER_WORKERS:-}" ]]; then PY_ARGS+=( "--tokenizer-workers" "${TOKENIZER_WORKERS}" ); fi
  if [[ -n "${FASTTEXT_EPOCHS:-}" ]]; then PY_ARGS+=( "--fasttext-epochs" "${FASTTEXT_EPOCHS}" ); fi
  if [[ -n "${FASTTEXT_MINCOUNT:-}" ]]; then PY_ARGS+=( "--fasttext-mincount" "${FASTTEXT_MINCOUNT}" ); fi
  if [[ -n "${FASTTEXT_LR:-}" ]]; then PY_ARGS+=( "--fasttext-lr" "${FASTTEXT_LR}" ); fi
  if [[ -n "${FASTTEXT_THREAD:-}" ]]; then PY_ARGS+=( "--fasttext-thread" "${FASTTEXT_THREAD}" ); fi
  if [[ "${SIMILARITY_THRESHOLD_USER_OVERRIDE}" == "true" ]]; then PY_ARGS+=( "--similarity-threshold" "${SIMILARITY_THRESHOLD}" ); fi
  if [[ -n "${CORPUS_SIZE_GB:-}" ]]; then PY_ARGS+=( "--corpus-size-gb" "${CORPUS_SIZE_GB}" ); fi
else
  # No config: retain explicit CLI to match historical behavior
  PY_ARGS+=( "--embedding-backend" "fasttext" )
  PY_ARGS+=( "--tokenizer-workers" "${TOKENIZER_WORKERS}" )
  PY_ARGS+=( "--evaluation-dataset" "${EVAL_DATASET}" )
  PY_ARGS+=( "--max-eval-samples" "${MAX_EVAL}" )
  PY_ARGS+=( "--corpus-size-gb" "${CORPUS_SIZE_GB}" )
  PY_ARGS+=( "--fasttext-epochs" "${FASTTEXT_EPOCHS}" )
  PY_ARGS+=( "--fasttext-mincount" "${FASTTEXT_MINCOUNT}" )
  PY_ARGS+=( "--fasttext-lr" "${FASTTEXT_LR}" )
  PY_ARGS+=( "--fasttext-thread" "${FASTTEXT_THREAD}" )
  PY_ARGS+=( "--similarity-threshold" "${SIMILARITY_THRESHOLD:-0.3}" )
fi

python "${ROOT_DIR}/script/run_medical_pipeline.py" "${PY_ARGS[@]}"

echo "[research] Completed. See ${RUN_ROOT} for artifacts."


