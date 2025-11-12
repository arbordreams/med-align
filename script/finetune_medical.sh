#!/usr/bin/env bash
set -euo pipefail

# Stage-wise CLM fine-tuning for BioMistral → Mistral adapted model
# Defaults target a single H100 80GB with bf16 and gradient checkpointing.
#
# Required env:
#   MODEL_DIR_STAGE0   - path to adapted model (Mistral tokenizer) from TokAlign apply stage
#   DATASET_PATH       - Hugging Face dataset saved to disk (load_from_disk), containing train/validation
#
# Optional env:
#   GENERAL_DATASET_PATH - optional general-domain dataset (load_from_disk) for mixing
#   GENERAL_MIX_RATIO    - e.g., 0.1 for 10% general-domain mix (default: 0.0)
#   OUTPUT_ROOT          - output directory root (default: runs/finetune)
#   SEQ_LEN              - sequence length (default: 2048)
#   STAGE1_STEPS         - max steps for Stage-1 (default: 2000)
#   STAGE2_STEPS         - max steps for Stage-2 (default: 5000)
#   TRAIN_BS             - per-device train batch size (default: 2)
#   GRAD_ACC             - gradient accumulation steps (default: 16)
#   SAVE_STEPS           - save every N steps (default: 500)
#   EVAL_STEPS           - eval every N steps (default: 500)
#   SEED                 - random seed (default: 0)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

MODEL_DIR_STAGE0="${MODEL_DIR_STAGE0:?MODEL_DIR_STAGE0 must point to the adapted model dir}"
DATASET_PATH="${DATASET_PATH:?DATASET_PATH must point to a load_from_disk dataset dir}"
GENERAL_DATASET_PATH="${GENERAL_DATASET_PATH:-}"
GENERAL_MIX_RATIO="${GENERAL_MIX_RATIO:-0.0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/finetune}"
SEQ_LEN="${SEQ_LEN:-2048}"
STAGE1_STEPS="${STAGE1_STEPS:-2000}"
STAGE2_STEPS="${STAGE2_STEPS:-5000}"
TRAIN_BS="${TRAIN_BS:-2}"
GRAD_ACC="${GRAD_ACC:-16}"
SAVE_STEPS="${SAVE_STEPS:-500}"
EVAL_STEPS="${EVAL_STEPS:-500}"
SEED="${SEED:-0}"

mkdir -p "${OUTPUT_ROOT}"

echo "[TokAlign] Stage-1: embeddings-only warmup"
STAGE1_OUT="${OUTPUT_ROOT}/stage1_embed_only"
mkdir -p "${STAGE1_OUT}"
python -m src.clm_train \
  --model_name "${MODEL_DIR_STAGE0}" \
  --tokenizer_path "${MODEL_DIR_STAGE0}" \
  --dataset_name "${DATASET_PATH}" \
  --max_seq_length "${SEQ_LEN}" \
  --per_device_train_batch_size "${TRAIN_BS}" \
  --gradient_accumulation_steps "${GRAD_ACC}" \
  --use_gradient_checkpointing \
  --bf16 True \
  --use_flash_attn True \
  --output_dir "${STAGE1_OUT}" \
  --max_steps "${STAGE1_STEPS}" \
  --save_steps "${SAVE_STEPS}" \
  --eval_steps "${EVAL_STEPS}" \
  --logging_steps 50 \
  --learning_rate 6.4e-4 \
  --lr_scheduler_type "cosine" \
  --weight_decay 0.01 \
  --evaluation_strategy "steps" \
  --early_stopping_patience 300 \
  --early_stopping_threshold 0.0 \
  --finetune_embed_only True \
  --seed "${SEED}" \
  $( [ -n "${GENERAL_DATASET_PATH}" ] && echo --general_dataset_name "${GENERAL_DATASET_PATH}" ) \
  $( [ -n "${GENERAL_DATASET_PATH}" ] && echo --general_mix_ratio "${GENERAL_MIX_RATIO}" )

echo "[TokAlign] Stage-2: full-model continued pretraining"
STAGE2_IN="${STAGE1_OUT}"
STAGE2_OUT="${OUTPUT_ROOT}/stage2_full"
mkdir -p "${STAGE2_OUT}"
python -m src.clm_train \
  --model_name "${STAGE2_IN}" \
  --tokenizer_path "${STAGE2_IN}" \
  --dataset_name "${DATASET_PATH}" \
  --max_seq_length "${SEQ_LEN}" \
  --per_device_train_batch_size "${TRAIN_BS}" \
  --gradient_accumulation_steps "${GRAD_ACC}" \
  --use_gradient_checkpointing \
  --bf16 True \
  --use_flash_attn True \
  --output_dir "${STAGE2_OUT}" \
  --max_steps "${STAGE2_STEPS}" \
  --save_steps "${SAVE_STEPS}" \
  --eval_steps "${EVAL_STEPS}" \
  --logging_steps 50 \
  --learning_rate 5e-5 \
  --lr_scheduler_type "cosine" \
  --weight_decay 0.01 \
  --evaluation_strategy "steps" \
  --early_stopping_patience 300 \
  --early_stopping_threshold 0.0 \
  --seed "${SEED}" \
  $( [ -n "${GENERAL_DATASET_PATH}" ] && echo --general_dataset_name "${GENERAL_DATASET_PATH}" ) \
  $( [ -n "${GENERAL_DATASET_PATH}" ] && echo --general_mix_ratio "${GENERAL_MIX_RATIO}" )

echo "[TokAlign] Fine-tuning completed."

