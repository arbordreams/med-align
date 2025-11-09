#!/usr/bin/env bash
set -e

# Build a curated medical corpus from vetted Hugging Face datasets.
# Results are stored under ${MEDICAL_CORPUS_DIR:-$MAIN_DIR/runs/corpora/default_medical}.

export MAIN_DIR="${MAIN_DIR:-/path/2/TokAlign/}"
cd "${MAIN_DIR}"

OUTPUT_DIR="${MEDICAL_CORPUS_DIR:-${MAIN_DIR}/runs/corpora/default_medical}"
mkdir -p "${OUTPUT_DIR}"

ARGS=(--output-dir "${OUTPUT_DIR}")

if [[ -n "${MEDICAL_DATASETS:-}" ]]; then
  for DATASET in ${MEDICAL_DATASETS}; do
    ARGS+=(--include "${DATASET}")
  done
fi

if [[ -n "${MEDICAL_EXCLUDE:-}" ]]; then
  for DATASET in ${MEDICAL_EXCLUDE}; do
    ARGS+=(--exclude "${DATASET}")
  done
fi

if [[ -n "${MEDICAL_MAX_SAMPLES:-}" ]]; then
  ARGS+=(--max-samples "${MEDICAL_MAX_SAMPLES}")
fi

if [[ -n "${MEDICAL_BYTE_BUDGET:-}" ]]; then
  ARGS+=(--byte-budget "${MEDICAL_BYTE_BUDGET}")
fi

if [[ "${MEDICAL_DEDUP:-1}" == "0" ]]; then
  ARGS+=(--no-dedup)
fi

if [[ -n "${MEDICAL_HASH_NAME:-}" ]]; then
  ARGS+=(--hash-name "${MEDICAL_HASH_NAME}")
fi

if [[ -n "${HF_TOKEN:-}" ]]; then
  ARGS+=(--hf-token "${HF_TOKEN}")
fi

python -m src.medical_corpus_builder "${ARGS[@]}"

MANIFEST_PATH="${OUTPUT_DIR}/manifest.json"
AGGREGATED_PATH="${OUTPUT_DIR}/aggregated/medical_corpus.jsonl"

echo ""
echo "[TokAlign] Medical corpus manifest: ${MANIFEST_PATH}"
if [[ -f "${AGGREGATED_PATH}" ]]; then
  echo "[TokAlign] Aggregated corpus available at ${AGGREGATED_PATH}"
  echo "[TokAlign] To feed TokAlign medical mode:"
  echo "           export MEDICAL_INPUTS=\"${AGGREGATED_PATH}\""
fi

