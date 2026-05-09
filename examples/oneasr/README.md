# OneASR TS-ASR ms-swift 训练

本目录包含通过 ms-swift 运行 Qwen3-ASR TS-ASR SFT 的 OneASR 扩展。

## 文件

- `ts_asr_sft.sh`：ms-swift SFT 入口，包含 Qwen3-ASR、LoRA、验证、日志和 checkpoint eval callback。
- `ts_asr_plugin.py`：外部插件，注册 `oneasr_ts_asr_eval`。

## 数据要求

`swift sft` 需要转换后的 ms-swift JSONL 文件：

- `TRAIN_DATASET`：默认 `data/ts_asr_v6/train.swift.jsonl`
- `VAL_DATASET`：默认 `data/ts_asr_v6/val.swift.jsonl`
- `SOURCE_MANIFEST`：默认 `data/ts_asr_v6/val.jsonl`
- `OUTPUT_DIR`：默认 `runs/swift_ts_asr_v6`
- `REPORT_TO`：默认 `swanlab`

Callback 读取第一个 `--val_dataset` ms-swift JSONL 并计算轻量生成式指标。ms-swift JSONL 只保留 `oneasr_id` 用于溯源；task、SNR、power 等分析字段从 `SOURCE_MANIFEST` 指向的原始 OneASR manifest 读取。中文样本主看 CER，英文样本报 WER。`normalized_exact_same` 忽略标点和空白差异。报告包含 overall、task 和 eval mode 三级汇总。`with_ref` 是原始 ref+mix 请求；`no_ref_mix` 是仅给 mix 音频、使用普通 ASR prompt 的 control。`No-ref Mix Control` 表显示 ref 是否真正起作用：`with_ref_only_correct` 表示 ref conditioning 修正了该样本，`no_ref_also_correct` 表示不带 ref 也能转写正确。`task=asr` 且 `source=asr_replay` 的普通 ASR replay 单独显示为 `asr_replay`。SNR 分桶只统计源 manifest 里带 `target_active_snr_db` 的样本，普通 `task=asr` replay 不进入 SNR 分桶。若源 manifest 带 component power 字段，报告还会在每个 target SNR 桶内按 dominant interference 拆分为 `non_target_speech` 和 `noise`。Callback 复用 `swift sft` 的 output_dir、eval batch size、max_new_tokens 和 SwanLab 配置。

## 运行

在 `ms-swift/` 目录下：

```bash
bash examples/oneasr/ts_asr_sft.sh
```

常用覆盖参数：

```bash
TRAIN_DATASET=/abs/train.swift.jsonl \
VAL_DATASET=/abs/val.swift.jsonl \
SOURCE_MANIFEST=/abs/val.jsonl \
OUTPUT_DIR=/abs/runs/swift_ts_asr \
REPORT_TO=none \
bash examples/oneasr/ts_asr_sft.sh
```

此 callback 设计上保持最小化：移除最后一个 assistant 消息作为 prompt，用当前训练模型生成，再对 assistant target 计算按语言区分的 WER/CER/exact_same。不展开 OneASR 原 manifest 的 `with_ref` / `external_ref` / `clean_speaker` 完整评估模式。

## 手动评估

在 OneASR 根目录下直接跑 eval，不启动 SFT：

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

读报告时先拆分 `task=ts_asr` 和 `task=asr` replay，再对比 `with_ref` 与 `no_ref_mix`。`no_ref_mix` 不是 base model 指标，而是同一个 checkpoint 在没有 ref 时的 control；要评估微调相对 base 的收益，需要用同一套 eval 分别跑 base model 和 adapter checkpoint。旧 manifest 如果只有 `target_active_snr_db`，需要先补齐 component power/SNR 再使用 dominant interference 表。仅当 `oneasr_id` 需要与补丁后的 source id 对齐时才需重新转换 ms-swift JSONL。
