#!/usr/bin/env bash
# Quick start for running the TokAlign medical pipeline on a fresh RunPod GPU pod.
# Assumes the container image provides Python 3.12 and CUDA 12.8 with Torch 2.8 wheels available.
#
# Steps:
# 1) Install dependencies via ./install_deps.sh (Torch first, robust flash-attn).
# 2) Run the end-to-end medical pipeline on a tiny sample using FastText embeddings.
# 3) Optionally evaluate on a small streaming dataset.
#
# You can override defaults via environment variables:
#   INPUT_JSONL: path to a JSONL file or directory (default: data/pretrain-corpus/lang-code-math-mix.sample.json)
#   SRC_MODEL:   source model identifier (default: EleutherAI/pythia-1b)
#   SRC_TOK:     source tokenizer (default: EleutherAI/pythia-1b)
#   TGT_TOK:     target tokenizer (default: google/gemma-2b)
#   RUN_ROOT:    run root directory (default: runs/tokenizer_adapt)
#   BYTE_BUDGET: corpus byte budget for quick run (default: 3000000 ~ few MB)
#   EVAL_DATASET: HF dataset for evaluation (default: uiyunkim-hub/pubmed-abstract:test)
#   MAX_EVAL:    maximum samples for evaluation (default: 50)

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

INPUT_JSONL="${INPUT_JSONL:-${ROOT_DIR}/data/pretrain-corpus/lang-code-math-mix.sample.json}"
SRC_MODEL="${SRC_MODEL:-EleutherAI/pythia-1b}"
SRC_TOK="${SRC_TOK:-EleutherAI/pythia-1b}"
TGT_TOK="${TGT_TOK:-google/gemma-2b}"
RUN_ROOT="${RUN_ROOT:-runs/tokenizer_adapt}"
BYTE_BUDGET="${BYTE_BUDGET:-3000000}"
EVAL_DATASET="${EVAL_DATASET:-uiyunkim-hub/pubmed-abstract:test}"
MAX_EVAL="${MAX_EVAL:-50}"

echo "[quickstart] Repository root: ${ROOT_DIR}"
echo "[quickstart] Installing dependencies..."
chmod +x "${ROOT_DIR}/install_deps.sh"
bash "${ROOT_DIR}/install_deps.sh"

echo "[quickstart] Launching medical pipeline (FastText backend)..."
python "${ROOT_DIR}/script/run_medical_pipeline.py" \
  --input "${INPUT_JSONL}" \
  --source-tokenizer "${SRC_TOK}" \
  --target-tokenizer "${TGT_TOK}" \
  --source-model "${SRC_MODEL}" \
  --embedding-backend fasttext \
  --byte-budget "${BYTE_BUDGET}" \
  --evaluation-dataset "${EVAL_DATASET}" \
  --max-eval-samples "${MAX_EVAL}" \
  --evaluate

echo "[quickstart] Completed. See runs directory under ${RUN_ROOT} for artifacts."


