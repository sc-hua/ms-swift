#!/usr/bin/env bash
set -euo pipefail

# Run from OneASR/ms-swift:
#   bash examples/oneasr/ts_asr_sft.sh

ONEASR_ROOT="${ONEASR_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}"

MODEL_PATH=${ONEASR_ROOT}/ckpts/qwen3-asr-0_6b
DATA_ROOT=${ONEASR_ROOT}/data/ts_asr_v6
EXP_NAME=swift_ts_asr_v6
VAL_DATASET="${VAL_DATASET:-${DATA_ROOT}/val.swift.jsonl}"

cd "${ONEASR_ROOT}"

CUDA_VISIBLE_DEVICES=0 SWIFT_SINGLE_DEVICE_MODE=1 \
python ms-swift/examples/oneasr/ts_asr_plugin.py eval \
  --model "${MODEL_PATH}" \
  --adapters runs/swift_ts_asr_v6/... \
  --val_dataset "${VAL_DATASET}" \
  --output_dir runs/swift_ts_asr_v6/.../manual_eval \
  --batch_size 4 \
  --max_new_tokens 128
