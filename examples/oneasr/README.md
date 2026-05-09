# OneASR TS-ASR Training in ms-swift

This folder contains OneASR project extensions for running Qwen3-ASR TS-ASR SFT through ms-swift.

## Files

- `ts_asr_sft.sh`: ms-swift SFT entrypoint with Qwen3-ASR, LoRA, validation, logging, and checkpoint eval callback.
- `ts_asr_plugin.py`: external plugin that registers `oneasr_ts_asr_eval`.

## Required Data

`swift sft` expects converted ms-swift JSONL files:

- `TRAIN_DATASET`: default `data/ts_asr_v6/train.swift.jsonl`
- `VAL_DATASET`: default `data/ts_asr_v6/val.swift.jsonl`
- `OUTPUT_DIR`: default `runs/swift_ts_asr_v6`
- `REPORT_TO`: default `swanlab`

The callback reads the first `--val_dataset` ms-swift JSONL and computes lightweight generation WER/CER. It uses the same ms-swift arguments for output directory, eval batch size, max generated tokens, and SwanLab logging.

## Run

From `ms-swift/`:

```bash
bash examples/oneasr/ts_asr_sft.sh
```

Common overrides:

```bash
TRAIN_DATASET=/abs/train.swift.jsonl \
VAL_DATASET=/abs/val.swift.jsonl \
OUTPUT_DIR=/abs/runs/swift_ts_asr \
REPORT_TO=none \
bash examples/oneasr/ts_asr_sft.sh
```

This callback is intentionally minimal: it removes the final assistant message, generates from the remaining prompt, and reports WER/CER/exact against that assistant target. It does not implement OneASR's full `with_ref` / `external_ref` / `clean_speaker` manifest expansion.
