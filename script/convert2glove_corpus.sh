#!/bin/sh

# The legacy parallel pipeline remains the default. Setting TOKALIGN_MODE=medical
# reroutes the script through the JSONL medical corpus flow provided by
# src/medical_pipeline.py without breaking existing behaviour.

export MAIN_DIR="/path/2/TokAlign/"
cd ${MAIN_DIR}
export CACHE_DIR="${MAIN_DIR}/data/cache"

TOKALIGN_MODE=${TOKALIGN_MODE:-parallel}

if [ "${TOKALIGN_MODE}" = "medical" ]; then
  if [ -z "${MEDICAL_INPUTS}" ]; then
    echo "[TokAlign] MEDICAL_INPUTS must list JSONL files or directories when TOKALIGN_MODE=medical."
    exit 1
  fi

  RUN_DIR=${TOKALIGN_RUN_DIR:-${MAIN_DIR}/runs/tokenizer_adapt/$(date +"%Y%m%d-%H%M%S")}
  mkdir -p "${RUN_DIR}"

  echo "[TokAlign] Medical data prep run directory: ${RUN_DIR}"

  for INPUT_PATH in ${MEDICAL_INPUTS}; do
    MEDICAL_ARGS="${MEDICAL_ARGS} --input ${INPUT_PATH}"
  done

  python -m src.medical_pipeline data-prep \
    --run-dir "${RUN_DIR}" \
    ${MEDICAL_ARGS} \
    --byte-budget ${MEDICAL_BYTE_BUDGET:-0} \
    $( [ "${MEDICAL_DEDUP:-1}" = "0" ] && echo "--no-dedup" ) \
    --hash-name ${MEDICAL_HASH_NAME:-sha256}

  TERMS_FILE="${RUN_DIR}/corpus/medical_terms.txt"
  python -m src.medical_terms mine \
    --corpus "${RUN_DIR}/corpus/medical_corpus.jsonl" \
    --output "${TERMS_FILE}" \
    --top-k ${MEDICAL_TERM_TOP_K:-500} \
    --min-count ${MEDICAL_MIN_TERM_FREQ:-5} \
    $( [ "${MEDICAL_USE_TFIDF:-0}" = "1" ] && echo "--use-tfidf" )

  AUG_SOURCE="${RUN_DIR}/tokenizers/source"
  AUG_TARGET="${RUN_DIR}/tokenizers/target"

  python -m src.medical_terms augment \
    --tokenizer "${TOKENIZER_PATH1}" \
    --terms "${TERMS_FILE}" \
    --output "${AUG_SOURCE}"

  python -m src.medical_terms augment \
    --tokenizer "${TOKENIZER_PATH2}" \
    --terms "${TERMS_FILE}" \
    --output "${AUG_TARGET}"

  python -m src.medical_pipeline tokenize \
    --run-dir "${RUN_DIR}" \
    --aggregated-jsonl "${RUN_DIR}/corpus/medical_corpus.jsonl" \
    --tokenizer-source "${AUG_SOURCE}" \
    --tokenizer-target "${AUG_TARGET}" \
    --tokenizer-workers ${NUM_WORKERS:-48} \
    --tokenizer-cache "${CACHE_DIR}"

  export TOKENIZER_PATH1="${AUG_SOURCE}"
  export TOKENIZER_PATH2="${AUG_TARGET}"
  export DATASET_PATH1="${RUN_DIR}/datasets/source"
  export DATASET_PATH2="${RUN_DIR}/datasets/target"
  export GLOVE_TRAIN_PATH1="${RUN_DIR}/glove_corpus/source.txt"
  export GLOVE_TRAIN_PATH2="${RUN_DIR}/glove_corpus/target.txt"
  export TOKALIGN_RUN_DIR="${RUN_DIR}"

  echo "[TokAlign] Medical corpora prepared:"
  echo "  Source dataset: ${DATASET_PATH1}"
  echo "  Target dataset: ${DATASET_PATH2}"
  echo "  Source embedding corpus: ${GLOVE_TRAIN_PATH1}"
  echo "  Target embedding corpus: ${GLOVE_TRAIN_PATH2}"
  exit 0
fi

# Parallel corpus fallback configuration (unchanged from upstream).

# export TRAIN_FILE="${MAIN_DIR}/data/pretrain-corpus/lang-code-math-mix.json"
# sample corpus for demonstration
export TRAIN_FILE="${MAIN_DIR}/data/pretrain-corpus/lang-code-math-mix.sample.json"

# Source Tokenizer
export MODLE_PATH1="EleutherAI/pythia-1b"
export TOKENIZER_PATH1="EleutherAI/pythia-1b"

export DATASET_PATH1="${MAIN_DIR}/data/pretrain-dataset/mix-pythia-tok"
export GLOVE_TRAIN_PATH1="${MAIN_DIR}/data/pretrain-dataset/mix-pythia-glove"

# Target Tokenizer
export MODLE_PATH2="google/gemma-2b"
export TOKENIZER_PATH2="google/gemma-2b"

export DATASET_PATH2="${MAIN_DIR}/data/pretrain-dataset/mix-gemma-tok"
export GLOVE_TRAIN_PATH2="${MAIN_DIR}/data/pretrain-dataset/mix-gemma-glove"

export MATRIX_EVAL_PATH="${MAIN_DIR}/data/pretrain-dataset/pythia-2-gemma-glove-eval-mix"

export NUM_WORKERS=48

tokenize () {
  HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  python -u src/process_dataset.py \
    --model_name_or_path ${MODLE_PATH} \
    --tokenizer_name ${TOKENIZER_PATH} \
    --train_file ${TRAIN_FILE} \
    --only_tokenize \
    --cache_dir ${CACHE_DIR} \
    --dataset_path_in_disk ${DATASET_PATH} \
    --preprocessing_num_workers ${NUM_WORKERS} \
    --output_dir ./log 2>&1
}

# Stage-1: tokenize the text corpus into token-ID corpus
MODLE_PATH=$MODLE_PATH1
TOKENIZER_PATH=$TOKENIZER_PATH1
DATASET_PATH=$DATASET_PATH1

printf "\n### Tokenize ${TRAIN_FILE} into the token ID corpus ${DATASET_PATH1} with tokenizer ${TOKENIZER_PATH1} ... ###\n\n"
tokenize

MODLE_PATH=$MODLE_PATH2
TOKENIZER_PATH=$TOKENIZER_PATH2
DATASET_PATH=$DATASET_PATH2

printf "\n### Tokenize ${TRAIN_FILE} into the token ID corpus ${DATASET_PATH2} with tokenizer ${TOKENIZER_PATH2} ... ###\n\n"
tokenize

MIN_LEN=0
MAX_LINE_TRAIN=1000000000
MAX_LINE_EVAL=1000

# Stage-2: extract token-ID corpus to train GloVe vector and evaluate the one-to-one mapping matrix learned.

printf "\n### Extract token IDs from ${DATASET_PATH1} for GloVe Training. ###\n\n"
python src/convert2glove_train.py \
  -s $DATASET_PATH1 \
  -k train \
  -m ${MIN_LEN} \
  -l ${MAX_LINE_TRAIN} \
  -o ${GLOVE_TRAIN_PATH1}

printf "\n### Extract token IDs from ${DATASET_PATH2} for GloVe Training. ###\n\n"
python src/convert2glove_train.py \
  -s $DATASET_PATH2 \
  -k train \
  -m ${MIN_LEN} \
  -l ${MAX_LINE_TRAIN} \
  -o ${GLOVE_TRAIN_PATH2}

MIN_LEN=10

printf "\n### Extract aligned token IDs from source token IDs (${DATASET_PATH1}) and target token IDs (${DATASET_PATH2}) for matrix evaluation. ###\n\n"
python src/convert2glove_train.py \
  -s $DATASET_PATH1 \
  -t $DATASET_PATH2 \
  -k validation \
  -m ${MIN_LEN} \
  -l ${MAX_LINE_EVAL} \
  -o ${MATRIX_EVAL_PATH}
