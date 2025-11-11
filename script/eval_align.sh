#!/bin/sh

export MAIN_DIR="/path/2/TokAlign/"
cd ${MAIN_DIR}

# The path of token alignment matrix
# export TGT_ID_2_SRC_ID_RES_PATH="${MAIN_DIR}/data/mistral2biomistral/align_matrix.json"
export TGT_ID_2_SRC_ID_RES_PATH="${MAIN_DIR}/data/mistral2biomistral/align_matrix_demo.json"

export MATRIX_EVAL_DATA_PATH="${MAIN_DIR}/data/pretrain-dataset/mistral-2-biomistral-eval"

# BLEU-1 evaluation
export EVAL_METHOD=bleu
export BLEU_WEIGHT="1,0,0,0"

# Bert-score evaluation
# export EVAL_METHOD=bert-score
export BERT_SOCRE_EVAL_MODEL="all-mpnet-base-v2"
export TOKENIZER_PATH="mistralai/Mistral-7B-v0.3"

python src/eval_matrix.py \
    -e ${EVAL_METHOD} \
    -m ${TGT_ID_2_SRC_ID_RES_PATH} \
    -f ${MATRIX_EVAL_DATA_PATH} \
    -t ${TOKENIZER_PATH} \
    -b ${BERT_SOCRE_EVAL_MODEL} \
    -w ${BLEU_WEIGHT}