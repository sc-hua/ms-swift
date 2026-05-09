#!/usr/bin/env bash
set -euo pipefail

# Run from /Users/hua/work/OneASR/ms-swift:
#   bash examples/oneasr/ts_asr_sft.sh

ONEASR_ROOT="${ONEASR_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}"

MODEL_PATH=${ONEASR_ROOT}/ckpts/qwen3-asr-0_6b
DATA_ROOT=${ONEASR_ROOT}/data/ts_asr_v6
EXP_NAME=swift_ts_asr_v6

# add qwen3-asr and ms-swift modules to PYTHONPATH
export PYTHONPATH="${ONEASR_ROOT}/Qwen3-ASR:${ONEASR_ROOT}/ms-swift:${PYTHONPATH:-}"

swift sft \
  --external_plugins examples/oneasr/ts_asr_plugin.py \
  --callbacks oneasr_ts_asr_eval \
  --model "${MODEL_PATH}" \
  --model_type qwen3_asr \
  --template qwen3_asr \
  --dataset "${DATA_ROOT}/train.swift.jsonl" \
  --val_dataset "${DATA_ROOT}/val.swift.jsonl" \
  --tuner_type lora \
  --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --lora_rank "${LORA_RANK:-64}" \
  --lora_alpha "${LORA_ALPHA:-128}" \
  --lora_dropout "${LORA_DROPOUT:-0.05}" \
  --torch_dtype "${TORCH_DTYPE:-bfloat16}" \
  --attn_impl "${ATTN_IMPL:-eager}" \
  --per_device_train_batch_size "${BATCH_SIZE:-4}" \
  --per_device_eval_batch_size "${EVAL_BATCH_SIZE:-4}" \
  --gradient_accumulation_steps "${GRAD_ACC:-8}" \
  --learning_rate "${LR:-2e-5}" \
  --num_train_epochs "${EPOCHS:-1}" \
  --max_steps "${MAX_STEPS:--1}" \
  --save_strategy steps \
  --save_steps "${SAVE_STEPS:-200}" \
  --eval_steps "${EVAL_STEPS:-${SAVE_STEPS:-200}}" \
  --logging_steps "${LOGGING_STEPS:-10}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT:-3}" \
  --dataloader_num_workers "${NUM_WORKERS:-4}" \
  --report_to "swanlab" \
  --swanlab_project "Swift-OneASR" \
  --swanlab_exp_name "${EXP_NAME}" \
  --output_dir "${ONEASR_ROOT}/runs/${EXP_NAME}"
