#!/bin/sh

export MAIN_DIR="/path/2/TokAlign/"
cd ${MAIN_DIR}

export MODLE_PATH1=${MODLE_PATH1:-"mistralai/Mistral-7B-v0.3"}
export TOKENIZER_PATH2=${TOKENIZER_PATH2:-"BioMistral/BioMistral-7B"}

TOKALIGN_MODE=${TOKALIGN_MODE:-parallel}

if [ "${TOKALIGN_MODE}" = "medical" ]; then
  if [ -z "${TOKALIGN_RUN_DIR}" ]; then
    echo "[TokAlign] TOKALIGN_RUN_DIR must be exported before running init_model.sh in medical mode."
    exit 1
  fi

  ALIGN_MATRIX_PATH=${TGT_ID_2_SRC_ID_RES_PATH:-${TOKALIGN_RUN_DIR}/alignment/align_matrix.json}
  TARGET_TOKENIZER=${TOKENIZER_PATH2:-${TOKALIGN_RUN_DIR}/tokenizers/target}
  OUTPUT_PATH=${TOKALIGN_RUN_DIR}/adapted_model

  python -m src.medical_pipeline apply \
    --run-dir "${TOKALIGN_RUN_DIR}" \
    --align-matrix "${ALIGN_MATRIX_PATH}" \
    --source-model "${MODLE_PATH1}" \
    --target-tokenizer "${TARGET_TOKENIZER}"

  echo "[TokAlign] Medical model initialised at ${OUTPUT_PATH}"
  exit 0
fi

# Parallel fallback
# export TGT_ID_2_SRC_ID_RES_PATH="${MAIN_DIR}/data/mistral2biomistral/align_matrix.json"
export TGT_ID_2_SRC_ID_RES_PATH="${MAIN_DIR}/data/mistral2biomistral/align_matrix_demo.json"

export OUTPUT_PATH="${MAIN_DIR}/data/mistral2biomistral/TokAlign-Init-7B"

python src/convert.py \
    -m ${TGT_ID_2_SRC_ID_RES_PATH} \
    -s ${MODLE_PATH1} \
    -t ${TOKENIZER_PATH2} \
    -o ${OUTPUT_PATH}
