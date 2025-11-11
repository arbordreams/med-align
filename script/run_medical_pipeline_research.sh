#!/usr/bin/env bash
# Research-grade overnight run for TokAlign medical pipeline on H100 (80GB).
# Quality-first configuration targeting ~5GB corpus and stable outputs.
#
# Requirements:
# - Python 3.12.x
# - CUDA 12.8, Torch 2.8.0 + flash-attn 2.8.3
# - H100 80GB (or comparable) with 200GB RAM and fast NFS/SSD
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
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-24}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-24}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export NVIDIA_TF32_OVERRIDE="${NVIDIA_TF32_OVERRIDE:-1}"   # enable TF32 on Hopper
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256,expandable_segments:True}"

# HF cache (persist under mounted volume if available)
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${ROOT_DIR}/.cache/huggingface}"
mkdir -p "${HF_DATASETS_CACHE}" || true

echo "[research] Updating repository (git pull)..."
git pull --rebase --autostash origin main || true

SRC_MODEL="${SRC_MODEL:-mistralai/Mistral-7B-v0.3}"
SRC_TOK="${SRC_TOK:-mistralai/Mistral-7B-v0.3}"
TGT_TOK="${TGT_TOK:-BioMistral/BioMistral-7B}"
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

# Research-grade defaults
TERM_TOP_K="${TERM_TOP_K:-2000}"
MIN_TERM_FREQ="${MIN_TERM_FREQ:-3}"
USE_TFIDF="--use-tfidf"  # always on in research mode
PIVOT_COUNT="${PIVOT_COUNT:-2000}"
FASTTEXT_EPOCHS="${FASTTEXT_EPOCHS:-30}"
FASTTEXT_MINCOUNT="${FASTTEXT_MINCOUNT:-1}"
FASTTEXT_LR="${FASTTEXT_LR:-0.05}"
# FastText threading: 2 workers × 12 threads = 24 threads total (matches 24 vCPUs)
# Each worker trains one embedding (source or target) in parallel
FASTTEXT_THREAD="${FASTTEXT_THREAD:-12}"

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
python "${ROOT_DIR}/script/run_medical_pipeline.py" \
  --input "${AGGREGATED_CORPUS}" \
  --source-tokenizer "${SRC_TOK}" \
  --target-tokenizer "${TGT_TOK}" \
  --source-model "${SRC_MODEL}" \
  --embedding-backend fasttext \
  --tokenizer-workers 24 \
  --evaluation-dataset "${EVAL_DATASET}" \
  --max-eval-samples "${MAX_EVAL}" \
  --research-mode \
  --corpus-size-gb "${CORPUS_SIZE_GB}" \
  --fasttext-epochs "${FASTTEXT_EPOCHS}" \
  --fasttext-mincount "${FASTTEXT_MINCOUNT}" \
  --fasttext-lr "${FASTTEXT_LR}" \
  --fasttext-thread "${FASTTEXT_THREAD}" \
  --evaluate \
  --qa

echo "[research] Completed. See ${RUN_ROOT} for artifacts."


