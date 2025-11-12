#!/usr/bin/env bash
set -euo pipefail

# Lightweight evaluation wrapper for adapted model perplexity on medical datasets
# Requires:
#   MODEL_DIR   - path to adapted model directory
#   TOKENIZER   - tokenizer path/id (defaults to MODEL_DIR)
#   DATASET     - HF dataset name or load_from_disk path with :split (e.g., uiyunkim-hub/pubmed-abstract:test)
# Optional:
#   MAX_SAMPLES - limit number of samples (default: 1000)
#   BATCH_SIZE  - batch size for evaluation (default: 32)
#   MAX_LEN     - max length (default: 1024)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

MODEL_DIR="${MODEL_DIR:?MODEL_DIR must point to the model directory}"
TOKENIZER="${TOKENIZER:-$MODEL_DIR}"
DATASET="${DATASET:?DATASET must specify dataset (hf name) optionally with :split}"
MAX_SAMPLES="${MAX_SAMPLES:-1000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_LEN="${MAX_LEN:-1024}"

DS_NAME="${DATASET%:*}"
SPLIT="${DATASET#*:}"
if [[ "${DS_NAME}" == "${SPLIT}" ]]; then
  SPLIT="test"
fi

python - <<'PY'
import os, json, sys
from src import eval_medical

model_path = os.environ["MODEL_DIR"]
tokenizer_path = os.environ.get("TOKENIZER", model_path)
dataset_name = os.environ["DS_NAME"]
split = os.environ["SPLIT"]
max_samples = int(os.environ.get("MAX_SAMPLES","1000"))
batch_size = int(os.environ.get("BATCH_SIZE","32"))
max_len = int(os.environ.get("MAX_LEN","1024"))

ppl = eval_medical.evaluate_perplexity(
    model_path=model_path,
    tokenizer_path=tokenizer_path,
    dataset_name=dataset_name,
    split=split,
    max_samples=max_samples,
    batch_size=batch_size,
    max_length=max_len,
)
print(json.dumps({"dataset": f"{dataset_name}:{split}", "perplexity": ppl}, indent=2))
PY

