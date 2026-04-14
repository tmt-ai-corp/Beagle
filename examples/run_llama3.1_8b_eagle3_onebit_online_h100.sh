#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname "$SCRIPT_DIR")

export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$ROOT_DIR/cache/compiled_kernels}
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

NUM_GPUS=${1:-4}
TP_SIZE=${2:-1}
BUILD_DATASET_NUM_PROC=${BUILD_DATASET_NUM_PROC:-64}
TARGET_MODEL_BACKEND=${TARGET_MODEL_BACKEND:-custom}
TARGET_MODEL_PATH=${TARGET_MODEL_PATH:-meta-llama/Llama-3.1-8B-Instruct}
BASE_DRAFT_CKPT=${BASE_DRAFT_CKPT:-/home/tmtaicorp/SpecForge/models/Eagle3-Llama-3.1-8B-Intruct}
TRAIN_DATA_PATH=${TRAIN_DATA_PATH:-$ROOT_DIR/cache/dataset/sharegpt_train.jsonl}
EVAL_DATA_PATH=${EVAL_DATA_PATH:-$ROOT_DIR/cache/dataset/sharegpt_test.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT_DIR/outputs/llama3.1-8b-eagle3-onebit}
MAX_LENGTH=${MAX_LENGTH:-4096}
BATCH_SIZE=${BATCH_SIZE:-1}
LEARNING_RATE=${LEARNING_RATE:-5e-5}
NUM_EPOCHS=${NUM_EPOCHS:-3}
LOG_INTERVAL=${LOG_INTERVAL:-10}
SAVE_INTERVAL=${SAVE_INTERVAL:-1000}
EVAL_INTERVAL=${EVAL_INTERVAL:-1000}
ONEBIT_QUANT_FUNC=${ONEBIT_QUANT_FUNC:-STEBinary}
ONEBIT_IS_PO2=${ONEBIT_IS_PO2:-0}
ONEBIT_INCLUDE_LM_HEAD=${ONEBIT_INCLUDE_LM_HEAD:-0}
ONEBIT_ADD_LAYERNORM=${ONEBIT_ADD_LAYERNORM:-1}

EXTRA_ARGS=()
if [[ -n "${EVAL_DATA_PATH}" && -f "${EVAL_DATA_PATH}" ]]; then
    EXTRA_ARGS+=(--eval-data-path "${EVAL_DATA_PATH}")
fi
if [[ "${ONEBIT_IS_PO2}" == "1" ]]; then
    EXTRA_ARGS+=(--onebit-is-po2)
fi
if [[ "${ONEBIT_INCLUDE_LM_HEAD}" == "1" ]]; then
    EXTRA_ARGS+=(--onebit-include-lm-head)
fi
if [[ "${ONEBIT_ADD_LAYERNORM}" == "0" ]]; then
    EXTRA_ARGS+=(--no-onebit-add-layernorm)
fi

torchrun \
    --standalone \
    --nproc_per_node "${NUM_GPUS}" \
    "$ROOT_DIR/scripts/train_eagle3.py" \
    --target-model-path "${TARGET_MODEL_PATH}" \
    --draft-model-config "$ROOT_DIR/configs/llama3-8B-eagle3.json" \
    --ckpt-dir "${BASE_DRAFT_CKPT}" \
    --train-data-path "${TRAIN_DATA_PATH}" \
    --build-dataset-num-proc "${BUILD_DATASET_NUM_PROC}" \
    --output-dir "${OUTPUT_DIR}" \
    --num-epochs "${NUM_EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --tp-size "${TP_SIZE}" \
    --learning-rate "${LEARNING_RATE}" \
    --max-length "${MAX_LENGTH}" \
    --chat-template llama3 \
    --cache-dir "$ROOT_DIR/cache" \
    --attention-backend sdpa \
    --target-model-backend "${TARGET_MODEL_BACKEND}" \
    --log-interval "${LOG_INTERVAL}" \
    --save-interval "${SAVE_INTERVAL}" \
    --eval-interval "${EVAL_INTERVAL}" \
    --use-onebit \
    --onebit-quant-func "${ONEBIT_QUANT_FUNC}" \
    "${EXTRA_ARGS[@]}"
