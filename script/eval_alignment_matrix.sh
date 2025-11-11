#!/bin/bash
# Script to evaluate alignment matrix quality
# Usage: ./script/eval_alignment_matrix.sh

set -euo pipefail

MAIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${MAIN_DIR}"

# Configuration
RUN_DIR="${RUN_DIR:-runs/tokenizer_adapt/20251111-112649}"
ALIGN_MATRIX="${ALIGN_MATRIX:-${RUN_DIR}/alignment/align_matrix.json}"
EVAL_METHOD="${EVAL_METHOD:-bleu}"  # or "bert-score"

# For BLEU evaluation
REFERENCE_PAIRS="${REFERENCE_PAIRS:-}"  # TSV file: src_ids\ttgt_ids per line
BLEU_NGRAM="${BLEU_NGRAM:-1}"  # 1, 2, 3, or 4

# For BERT-Score evaluation
TOKENIZER_PATH="${TOKENIZER_PATH:-mistralai/Mistral-7B-v0.3}"
BERT_MODEL="${BERT_MODEL:-all-mpnet-base-v2}"

echo "=========================================="
echo "Alignment Matrix Evaluation"
echo "=========================================="
echo "Matrix: ${ALIGN_MATRIX}"
echo "Method: ${EVAL_METHOD}"
echo ""

if [ ! -f "${ALIGN_MATRIX}" ]; then
    echo "ERROR: Alignment matrix not found: ${ALIGN_MATRIX}"
    exit 1
fi

if [ "${EVAL_METHOD}" = "bleu" ]; then
    if [ -z "${REFERENCE_PAIRS}" ] || [ ! -f "${REFERENCE_PAIRS}" ]; then
        echo "ERROR: Reference pairs file required for BLEU evaluation"
        echo "  Set REFERENCE_PAIRS=/path/to/pairs.tsv"
        echo "  Format: Each line should be: <src_token_ids_space_separated>\\t<tgt_token_ids_space_separated>"
        exit 1
    fi
    
    # Convert BLEU ngram to weights
    case "${BLEU_NGRAM}" in
        1) BLEU_WEIGHTS="1,0,0,0" ;;
        2) BLEU_WEIGHTS="0.5,0.5,0,0" ;;
        3) BLEU_WEIGHTS="0.333333,0.333333,0.333333,0" ;;
        4) BLEU_WEIGHTS="0.25,0.25,0.25,0.25" ;;
        *) echo "ERROR: BLEU_NGRAM must be 1-4"; exit 1 ;;
    esac
    
    echo "Running BLEU-${BLEU_NGRAM} evaluation..."
    python src/eval_matrix.py \
        -e bleu \
        -m "${ALIGN_MATRIX}" \
        -f "${REFERENCE_PAIRS}" \
        -w "${BLEU_WEIGHTS}"
        
elif [ "${EVAL_METHOD}" = "bert-score" ]; then
    if [ -z "${REFERENCE_PAIRS}" ] || [ ! -f "${REFERENCE_PAIRS}" ]; then
        echo "ERROR: Reference pairs file required for BERT-Score evaluation"
        echo "  Set REFERENCE_PAIRS=/path/to/pairs.tsv"
        exit 1
    fi
    
    echo "Running BERT-Score evaluation..."
    python src/eval_matrix.py \
        -e bert-score \
        -m "${ALIGN_MATRIX}" \
        -f "${REFERENCE_PAIRS}" \
        -t "${TOKENIZER_PATH}" \
        -b "${BERT_MODEL}"
else
    echo "ERROR: Unknown evaluation method: ${EVAL_METHOD}"
    echo "  Use 'bleu' or 'bert-score'"
    exit 1
fi

echo ""
echo "Evaluation complete!"

