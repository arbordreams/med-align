#!/usr/bin/env bash
# Quick start for running the TokAlign medical pipeline on a fresh RunPod GPU pod.
# Assumes the container image provides Python 3.12 and CUDA 12.8 with Torch 2.8 wheels available.
#
# Steps:
# 1) Install dependencies via ./install_deps.sh (Torch first, robust flash-attn).
# 2) Build a demo medical corpus from HuggingFace datasets.
# 3) Run the end-to-end medical pipeline using FastText embeddings.
# 4) Optionally evaluate on a small streaming dataset.
#
# You can override defaults via environment variables:
#   SRC_MODEL:   source model identifier (default: EleutherAI/pythia-1b)
#   SRC_TOK:     source tokenizer (default: EleutherAI/pythia-1b)
#   TGT_TOK:     target tokenizer (default: google/gemma-2b)
#   RUN_ROOT:    run root directory (default: runs/tokenizer_adapt)
#   BYTE_BUDGET: corpus byte budget for quick run (default: 5000000 ~ 5MB)
#   MAX_SAMPLES: max samples per dataset for demo (default: 2000)
#   EVAL_DATASET: HF dataset for evaluation (default: uiyunkim-hub/pubmed-abstract:test)
#   MAX_EVAL:    maximum samples for evaluation (default: 50)
#   HF_TOKEN:    HuggingFace token for gated datasets (required for some models)

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

SRC_MODEL="${SRC_MODEL:-EleutherAI/pythia-1b}"
SRC_TOK="${SRC_TOK:-EleutherAI/pythia-1b}"
TGT_TOK="${TGT_TOK:-google/gemma-2b}"
RUN_ROOT="${RUN_ROOT:-runs/tokenizer_adapt}"
BYTE_BUDGET="${BYTE_BUDGET:-5000000}"
MAX_SAMPLES="${MAX_SAMPLES:-2000}"
EVAL_DATASET="${EVAL_DATASET:-uiyunkim-hub/pubmed-abstract:test}"
MAX_EVAL="${MAX_EVAL:-50}"

echo "[quickstart] Repository root: ${ROOT_DIR}"
echo "[quickstart] Installing dependencies..."
chmod +x "${ROOT_DIR}/install_deps.sh"
bash "${ROOT_DIR}/install_deps.sh"

echo "[quickstart] Building demo medical corpus from HuggingFace datasets..."
export MAIN_DIR="${ROOT_DIR}"
CORPUS_DIR="${ROOT_DIR}/runs/corpora/quickstart_demo"
export MEDICAL_CORPUS_DIR="${CORPUS_DIR}"
export MEDICAL_DATASETS="${MEDICAL_DATASETS:-pubmed_abstract}"
export MEDICAL_MAX_SAMPLES="${MAX_SAMPLES}"
export MEDICAL_BYTE_BUDGET="${BYTE_BUDGET}"

# Use HF_TOKEN if available
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN="${HF_TOKEN}"
fi

bash "${ROOT_DIR}/script/build_medical_corpus.sh"

AGGREGATED_CORPUS="${CORPUS_DIR}/aggregated/medical_corpus.jsonl"
if [[ ! -f "${AGGREGATED_CORPUS}" ]]; then
  echo "[quickstart] ERROR: Failed to build medical corpus. Check logs above."
  exit 1
fi

echo "[quickstart] Medical corpus built: ${AGGREGATED_CORPUS}"
echo "[quickstart] Launching medical pipeline (FastText backend)..."
python "${ROOT_DIR}/script/run_medical_pipeline.py" \
  --input "${AGGREGATED_CORPUS}" \
  --source-tokenizer "${SRC_TOK}" \
  --target-tokenizer "${TGT_TOK}" \
  --source-model "${SRC_MODEL}" \
  --embedding-backend fasttext \
  --byte-budget "${BYTE_BUDGET}" \
  --evaluation-dataset "${EVAL_DATASET}" \
  --max-eval-samples "${MAX_EVAL}" \
  --evaluate

echo "[quickstart] Completed. See runs directory under ${RUN_ROOT} for artifacts."


