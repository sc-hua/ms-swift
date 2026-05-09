# OneASR TS-ASR Training in ms-swift

This folder contains OneASR project extensions for running Qwen3-ASR TS-ASR SFT through ms-swift.

## Files

- `ts_asr_sft.sh`: ms-swift SFT entrypoint with Qwen3-ASR, LoRA, validation, logging, and checkpoint eval callback.
- `ts_asr_plugin.py`: external plugin that registers `oneasr_ts_asr_eval`.

## Required Data

`swift sft` expects converted ms-swift JSONL files:

- `TRAIN_DATASET`: default `data/ts_asr_v6/train.swift.jsonl`
- `VAL_DATASET`: default `data/ts_asr_v6/val.swift.jsonl`
- `SOURCE_MANIFEST`: default `data/ts_asr_v6/val.jsonl`
- `OUTPUT_DIR`: default `runs/swift_ts_asr_v6`
- `REPORT_TO`: default `swanlab`

The callback reads the first `--val_dataset` ms-swift JSONL and computes lightweight generation metrics. The ms-swift JSONL only carries `oneasr_id` for traceability; task, SNR, power, and other analysis fields are joined from `SOURCE_MANIFEST`. Chinese samples use CER as the primary metric; English samples report WER. `normalized_exact_same` ignores punctuation and whitespace differences. The report includes an overall summary and a task summary; `task=asr` rows from ASR replay are shown as `asr_replay`. SNR buckets only include source rows with `target_active_snr_db`; plain `task=asr` replay rows are not included in SNR buckets. If component power fields are present, the report also splits each target SNR bucket by dominant interference: `non_target_speech` or `noise`. It uses the same ms-swift arguments for output directory, eval batch size, max generated tokens, and SwanLab logging.

## Run

From `ms-swift/`:

```bash
bash examples/oneasr/ts_asr_sft.sh
```

Common overrides:

```bash
TRAIN_DATASET=/abs/train.swift.jsonl \
VAL_DATASET=/abs/val.swift.jsonl \
SOURCE_MANIFEST=/abs/val.jsonl \
OUTPUT_DIR=/abs/runs/swift_ts_asr \
REPORT_TO=none \
bash examples/oneasr/ts_asr_sft.sh
```

This callback is intentionally minimal: it removes the final assistant message, generates from the remaining prompt, and reports language-aware WER/CER/exact_same metrics against that assistant target. It does not implement OneASR's full `with_ref` / `external_ref` / `clean_speaker` manifest expansion.

## Manual Eval

Run eval directly without launching SFT from the OneASR repo root:

```bash
cd /abs/OneASR
CUDA_VISIBLE_DEVICES=0 SWIFT_SINGLE_DEVICE_MODE=1 \
python ms-swift/examples/oneasr/ts_asr_plugin.py eval \
  --model /abs/ckpts/qwen3-asr-0_6b \
  --adapters /abs/runs/swift_ts_asr_v6/checkpoint-180 \
  --val_dataset /abs/data/ts_asr_v6/val.swift.jsonl \
  --source_manifest /abs/data/ts_asr_v6/val.jsonl \
  --output_dir /abs/runs/swift_ts_asr_v6/manual_eval \
  --batch_size 32 \
  --max_new_tokens 128
```

When reading results, split `task=ts_asr` from `task=asr` replay if both exist in the source manifest. Overall metrics mix both tasks, while SNR buckets reflect TS-ASR rows only. Older manifests that only have `target_active_snr_db` should be patched before using the dominant interference table. Reconvert the ms-swift JSONL only when `oneasr_id` needs to be aligned with patched source ids.
