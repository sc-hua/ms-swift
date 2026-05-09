"""OneASR lightweight ASR evaluation for ms-swift training.

This plugin intentionally stays self-contained. It evaluates a ms-swift JSONL
validation file by removing the last assistant message, generating with the
current trainer model, and computing WER/CER against that assistant target.
"""

import argparse
import json
import os
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

SWIFT_ROOT = Path(__file__).resolve().parents[2]
if str(SWIFT_ROOT) not in sys.path:
    sys.path.insert(0, str(SWIFT_ROOT))
QWEN3_ASR_ROOT = SWIFT_ROOT.parent / "Qwen3-ASR"
if QWEN3_ASR_ROOT.exists() and str(QWEN3_ASR_ROOT) not in sys.path:
    sys.path.insert(0, str(QWEN3_ASR_ROOT))

try:
    from swift.callbacks.base import TrainerCallback
    from swift.callbacks.mapping import callbacks_map
except ModuleNotFoundError:
    if __name__ != "__main__":
        raise

    class TrainerCallback:
        pass

    callbacks_map = {}


CALLBACK_NAME = "oneasr_ts_asr_eval"
ASR_TEXT_TAG = "<asr_text>"
WITH_REF_MODE = "with_ref"
NO_REF_MIX_MODE = "no_ref_mix"
NO_REF_MIX_PROMPT = "Transcribe the speech in the input audio."
SOURCE_ANALYSIS_FIELDS = [
    "task",
    "source",
    "target_present",
    "mix_audio",
    "target_active_snr_db",
    "target_active_target_power",
    "target_active_interference_power",
    "target_active_non_target_speech_power",
    "target_active_noise_power",
    "target_active_non_target_speech_snr_db",
    "target_active_noise_snr_db",
]


def resolve_eval_dataset(args) -> str:
    val_dataset = getattr(args, "val_dataset", None)
    if isinstance(val_dataset, str):
        return val_dataset
    if isinstance(val_dataset, (list, tuple)) and val_dataset:
        return str(val_dataset[0])
    args_path = Path(getattr(args, "output_dir", ".")) / "args.json"
    if args_path.exists():
        with open(args_path, "r", encoding="utf-8") as f:
            saved_args = json.load(f)
        val_dataset = saved_args.get("val_dataset")
        if isinstance(val_dataset, str):
            return val_dataset
        if isinstance(val_dataset, (list, tuple)) and val_dataset:
            return str(val_dataset[0])
    return ""


def infer_source_manifest(dataset_path: str) -> str:
    path = Path(dataset_path)
    if path.name.endswith(".swift.jsonl"):
        candidate = path.with_name(path.name.replace(".swift.jsonl", ".jsonl"))
        if candidate.exists():
            return str(candidate)
    return ""


def resolve_source_manifest(args, dataset_path: str) -> str:
    source_manifest = getattr(args, "source_manifest", None)
    if source_manifest:
        return str(source_manifest)
    source_manifest = os.environ.get("ONEASR_SOURCE_MANIFEST")
    if source_manifest:
        return source_manifest
    args_path = Path(getattr(args, "output_dir", ".")) / "args.json"
    if args_path.exists():
        with open(args_path, "r", encoding="utf-8") as f:
            saved_args = json.load(f)
        source_manifest = saved_args.get("source_manifest")
        if source_manifest:
            return str(source_manifest)
    return infer_source_manifest(dataset_path)


def parse_asr_text(raw: str) -> str:
    raw = str(raw or "").strip()
    if ASR_TEXT_TAG not in raw:
        return raw
    return raw.split(ASR_TEXT_TAG, 1)[1].strip()


def parse_language(raw: str) -> str:
    match = re.match(r"^\s*language\s+([^<\s]+)\s*<asr_text>", str(raw or ""))
    return match.group(1) if match else "unknown"


def normalize_for_exact(text: str) -> str:
    text = parse_asr_text(text).lower()
    return "".join(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+", text))


def normalize_words(text: str) -> list[str]:
    text = parse_asr_text(text).lower()
    text = re.sub(r"[^a-z0-9']+", " ", text)
    return text.split()


def normalize_chars(text: str) -> list[str]:
    text = parse_asr_text(text).lower()
    text = re.sub(r"\s+", "", text)
    return list(text)


def edit_distance(pred: list[str], ref: list[str]) -> int:
    if not ref:
        return len(pred)
    prev = list(range(len(pred) + 1))
    for i, ref_item in enumerate(ref, start=1):
        cur = [i]
        for j, pred_item in enumerate(pred, start=1):
            cost = 0 if pred_item == ref_item else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def error_rate(pred: list[str], ref: list[str]) -> float:
    if not ref:
        return 0.0 if not pred else 1.0
    return edit_distance(pred, ref) / len(ref)


def word_error_rate(prediction: str, reference: str) -> float:
    return error_rate(normalize_words(prediction), normalize_words(reference))


def char_error_rate(prediction: str, reference: str) -> float:
    return error_rate(normalize_chars(prediction), normalize_chars(reference))


def uses_word_error_rate(language: str) -> bool:
    return language.lower() == "english"


def metric_language(prediction: str, label: str) -> str:
    label_language = parse_language(label)
    if label_language != "unknown":
        return label_language
    return parse_language(prediction)


def build_metric_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metric_rows = []
    for row in rows:
        prediction = row["prediction"]
        label = row["label"]
        language = metric_language(prediction, label)
        pred_text = parse_asr_text(prediction)
        label_text = parse_asr_text(label)
        metric_rows.append(
            {
                **row,
                "language": language,
                "prediction_text": pred_text,
                "label_text": label_text,
                "wer": word_error_rate(prediction, label) if uses_word_error_rate(language) else None,
                "cer": char_error_rate(prediction, label),
                "exact_same": pred_text.strip() == label_text.strip(),
                "normalized_exact_same": normalize_for_exact(prediction) == normalize_for_exact(label),
            }
        )
    return metric_rows


def _copy_request_fields(
    row: dict[str, Any],
    messages: list[dict[str, Any]],
    label: str,
    dataset_path: str,
    line_no: int,
    eval_index: int,
) -> dict[str, Any]:
    request = {
        "eval_index": eval_index,
        "eval_source": {"dataset_path": dataset_path, "line_no": line_no},
        "messages": messages,
        "label": label,
    }
    passthrough_keys = [
        "images",
        "audios",
        "videos",
        "tools",
        "objects",
        "chat_template_kwargs",
        "oneasr_id",
    ]
    for key in passthrough_keys:
        if key in row:
            request[key] = row[key]
    return request


def load_eval_records(dataset_path: str) -> list[dict[str, Any]]:
    records = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            messages = deepcopy(row.get("messages") or [])
            if not messages or messages[-1].get("role") != "assistant":
                continue
            label = str(messages.pop().get("content") or "")
            records.append(_copy_request_fields(row, messages, label, dataset_path, line_no, len(records) + 1))
    return records


def source_manifest_id(row: dict[str, Any]) -> str:
    value = row.get("id") or row.get("pair_id") or row.get("mix_id")
    if not value:
        raise ValueError("source manifest row missing stable id: expected id, pair_id, or mix_id")
    return str(value)


def load_source_manifest(source_manifest: str) -> dict[str, dict[str, Any]]:
    records = {}
    with open(source_manifest, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            source_id = source_manifest_id(row)
            if source_id in records:
                raise ValueError(f"{source_manifest}:{line_no}: duplicate source id: {source_id}")
            records[source_id] = row
    return records


def attach_source_manifest(rows: list[dict[str, Any]], source_manifest: str) -> list[dict[str, Any]]:
    source_rows = load_source_manifest(source_manifest)
    enriched = []
    missing_ids = []
    for row in rows:
        oneasr_id = row.get("oneasr_id")
        if not oneasr_id:
            enriched.append(row)
            continue
        source_row = source_rows.get(str(oneasr_id))
        if source_row is None:
            missing_ids.append(oneasr_id)
            enriched.append(row)
            continue
        joined = dict(row)
        for key in SOURCE_ANALYSIS_FIELDS:
            if key in source_row:
                joined[key] = source_row[key]
        enriched.append(joined)
    if missing_ids:
        print(f"[oneasr eval] WARNING: {len(missing_ids)} rows not found in source manifest, first 5: {missing_ids[:5]}")
    return enriched


def resolve_audio_path(path: Any) -> str:
    audio_path = Path(str(path)).expanduser()
    if not audio_path.is_absolute():
        audio_path = SWIFT_ROOT.parent / audio_path
    return str(audio_path.resolve())


def eval_pair_id(row: dict[str, Any]) -> str:
    return f"{row.get('oneasr_id')}#{row.get('eval_index')}"


def no_ref_mix_record(row: dict[str, Any]) -> dict[str, Any] | None:
    if task_group(row) != "ts_asr" or row.get("target_present") is False or not row.get("mix_audio"):
        return None
    return {
        **row,
        "eval_mode": NO_REF_MIX_MODE,
        "eval_pair_id": eval_pair_id(row),
        "messages": [
            {"role": "system", "content": NO_REF_MIX_PROMPT},
            {"role": "user", "content": "Input audio:\n<audio>"},
        ],
        "audios": [resolve_audio_path(row["mix_audio"])],
    }


def build_eval_requests(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    requests = []
    for row in rows:
        with_ref = {**row, "eval_mode": WITH_REF_MODE, "eval_pair_id": eval_pair_id(row)}
        requests.append(with_ref)
        no_ref = no_ref_mix_record(row)
        if no_ref is not None:
            requests.append(no_ref)
    return requests


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _rate(values: list[bool]) -> float:
    return sum(values) / len(values) if values else 0.0


def _mean_present(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return _mean(values) if values else None


def task_group(row: dict[str, Any]) -> str:
    task = str(row.get("task") or "")
    source = str(row.get("source") or "")
    oneasr_id = str(row.get("oneasr_id") or "")
    if source == "asr_replay" or task == "asr" or oneasr_id.startswith("asr_replay_"):
        return "asr_replay"
    if task:
        return task
    if row.get("target_active_snr_db") is not None or oneasr_id.startswith("val_"):
        return "ts_asr"
    return "unknown"


SNR_BUCKET_EDGES = [-15, -10, -5, -2, 0, 2, 5, 10, 15, 20, 30]


def _snr_bucket_label(low: float | None, high: float | None) -> str:
    if low is None:
        return f"(-inf, {high}]"
    if high is None:
        return f"({low}, inf)"
    return f"({low}, {high}]"


def _snr_buckets() -> list[tuple[str, float | None, float | None]]:
    buckets: list[tuple[str, float | None, float | None]] = []
    buckets.append((_snr_bucket_label(None, SNR_BUCKET_EDGES[0]), None, SNR_BUCKET_EDGES[0]))
    for i in range(len(SNR_BUCKET_EDGES) - 1):
        buckets.append((_snr_bucket_label(SNR_BUCKET_EDGES[i], SNR_BUCKET_EDGES[i + 1]), SNR_BUCKET_EDGES[i], SNR_BUCKET_EDGES[i + 1]))
    buckets.append((_snr_bucket_label(SNR_BUCKET_EDGES[-1], None), SNR_BUCKET_EDGES[-1], None))
    return buckets


def _in_bucket(snr: float, low: float | None, high: float | None) -> bool:
    if low is not None and snr <= low:
        return False
    if high is not None and snr > high:
        return False
    return True


def summarize_by_snr(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows_with_snr = [r for r in metric_rows if r.get("target_active_snr_db") is not None]
    if not rows_with_snr:
        return []
    result = []
    for label, low, high in _snr_buckets():
        bucket_rows = [r for r in rows_with_snr if _in_bucket(r["target_active_snr_db"], low, high)]
        if not bucket_rows:
            result.append({"snr_bucket": label, "records": 0})
            continue
        wer_rows = [r for r in bucket_rows if r.get("wer") is not None]
        result.append({
            "snr_bucket": label,
            "records": len(bucket_rows),
            "cer": _mean([r["cer"] for r in bucket_rows]),
            "wer": _mean([r["wer"] for r in wer_rows]) if wer_rows else None,
            "normalized_exact_same": _rate([r["normalized_exact_same"] for r in bucket_rows]),
        })
    return result


def dominant_interference_source(row: dict[str, Any]) -> str:
    speech = row.get("target_active_non_target_speech_power")
    noise = row.get("target_active_noise_power")
    if speech is None and noise is None:
        return "unknown"
    speech = float(speech or 0.0)
    noise = float(noise or 0.0)
    if speech <= 0 and noise <= 0:
        return "none"
    return "non_target_speech" if speech >= noise else "noise"


def summarize_by_snr_and_interference(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows_with_snr = [r for r in metric_rows if r.get("target_active_snr_db") is not None]
    if not rows_with_snr:
        return []
    result = []
    for label, low, high in _snr_buckets():
        bucket_rows = [r for r in rows_with_snr if _in_bucket(r["target_active_snr_db"], low, high)]
        for source in ["non_target_speech", "noise", "none", "unknown"]:
            source_rows = [r for r in bucket_rows if dominant_interference_source(r) == source]
            if not source_rows:
                continue
            wer_rows = [r for r in source_rows if r.get("wer") is not None]
            result.append({
                "snr_bucket": label,
                "interference_source": source,
                "records": len(source_rows),
                "cer": _mean([r["cer"] for r in source_rows]),
                "wer": _mean([r["wer"] for r in wer_rows]) if wer_rows else None,
                "normalized_exact_same": _rate([r["normalized_exact_same"] for r in source_rows]),
                "target_active_non_target_speech_snr_db": _mean_present(source_rows, "target_active_non_target_speech_snr_db"),
                "target_active_noise_snr_db": _mean_present(source_rows, "target_active_noise_snr_db"),
            })
    return result


def summarize_predictions(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {
            "records": 0,
            "wer": 0.0,
            "wer_records": 0,
            "cer": 0.0,
            "exact_same": 0.0,
            "normalized_exact_same": 0.0,
            "english_records": 0,
            "english_wer": 0.0,
            "english_cer": 0.0,
            "english_exact_same": 0.0,
            "english_normalized_exact_same": 0.0,
            "chinese_records": 0,
            "chinese_cer": 0.0,
            "chinese_exact_same": 0.0,
            "chinese_normalized_exact_same": 0.0,
        }

    metric_rows = rows if "cer" in rows[0] else build_metric_rows(rows)
    wer_rows = [row for row in metric_rows if row.get("wer") is not None]
    english_rows = [row for row in metric_rows if row["language"].lower() == "english"]
    chinese_rows = [row for row in metric_rows if row["language"].lower() == "chinese"]
    return {
        "records": len(metric_rows),
        "wer": _mean([row["wer"] for row in wer_rows]),
        "wer_records": len(wer_rows),
        "cer": _mean([row["cer"] for row in metric_rows]),
        "exact_same": _rate([row["exact_same"] for row in metric_rows]),
        "normalized_exact_same": _rate([row["normalized_exact_same"] for row in metric_rows]),
        "english_records": len(english_rows),
        "english_wer": _mean([row["wer"] for row in english_rows if row.get("wer") is not None]),
        "english_cer": _mean([row["cer"] for row in english_rows]),
        "english_exact_same": _rate([row["exact_same"] for row in english_rows]),
        "english_normalized_exact_same": _rate([row["normalized_exact_same"] for row in english_rows]),
        "chinese_records": len(chinese_rows),
        "chinese_cer": _mean([row["cer"] for row in chinese_rows]),
        "chinese_exact_same": _rate([row["exact_same"] for row in chinese_rows]),
        "chinese_normalized_exact_same": _rate([row["normalized_exact_same"] for row in chinese_rows]),
    }


def summarize_by_task(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    metric_rows = rows if rows and "cer" in rows[0] else build_metric_rows(rows)
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in metric_rows:
        groups.setdefault(task_group(row), []).append(row)

    result = {"overall": summarize_predictions(metric_rows)}
    for name in ["ts_asr", "asr_replay", *sorted(k for k in groups if k not in {"ts_asr", "asr_replay"})]:
        if name in groups:
            result[name] = summarize_predictions(groups[name])
    return result


def summarize_by_eval_mode(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    metric_rows = rows if rows and "cer" in rows[0] else build_metric_rows(rows)
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in metric_rows:
        groups.setdefault(str(row.get("eval_mode") or WITH_REF_MODE), []).append(row)
    result = {}
    for name in [WITH_REF_MODE, NO_REF_MIX_MODE, *sorted(k for k in groups if k not in {WITH_REF_MODE, NO_REF_MIX_MODE})]:
        if name in groups:
            result[name] = summarize_predictions(groups[name])
    return result


def summarize_no_ref_control(rows: list[dict[str, Any]]) -> dict[str, float]:
    metric_rows = rows if rows and "cer" in rows[0] else build_metric_rows(rows)
    groups: dict[str, dict[str, dict[str, Any]]] = {}
    for row in metric_rows:
        pair_id = row.get("eval_pair_id")
        mode = row.get("eval_mode")
        if pair_id and mode in {WITH_REF_MODE, NO_REF_MIX_MODE}:
            groups.setdefault(str(pair_id), {})[str(mode)] = row

    pairs = [pair for pair in groups.values() if WITH_REF_MODE in pair and NO_REF_MIX_MODE in pair]
    with_ref_cers = [pair[WITH_REF_MODE]["cer"] for pair in pairs]
    no_ref_cers = [pair[NO_REF_MIX_MODE]["cer"] for pair in pairs]
    result = {
        "pairs": len(pairs),
        "with_ref_cer": _mean(with_ref_cers),
        "no_ref_mix_cer": _mean(no_ref_cers),
        "cer_delta_no_ref_minus_with_ref": _mean(no_ref_cers) - _mean(with_ref_cers) if pairs else 0.0,
        "with_ref_accuracy": 0.0,
        "no_ref_mix_accuracy": 0.0,
        "accuracy_delta_with_ref_minus_no_ref": 0.0,
        "with_ref_only_correct": 0,
        "no_ref_also_correct": 0,
        "no_ref_only_correct": 0,
        "both_wrong": 0,
    }
    if not pairs:
        return result

    with_ref_correct = 0
    no_ref_correct = 0
    for pair in pairs:
        wr = bool(pair[WITH_REF_MODE]["normalized_exact_same"])
        nr = bool(pair[NO_REF_MIX_MODE]["normalized_exact_same"])
        with_ref_correct += int(wr)
        no_ref_correct += int(nr)
        if wr and nr:
            result["no_ref_also_correct"] += 1
        elif wr and not nr:
            result["with_ref_only_correct"] += 1
        elif nr and not wr:
            result["no_ref_only_correct"] += 1
        else:
            result["both_wrong"] += 1
    result["with_ref_accuracy"] = with_ref_correct / len(pairs)
    result["no_ref_mix_accuracy"] = no_ref_correct / len(pairs)
    result["accuracy_delta_with_ref_minus_no_ref"] = result["with_ref_accuracy"] - result["no_ref_mix_accuracy"]
    return result


def _fmt_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6f}"


def _append_metric_table(lines: list[str], title: str, rows: list[tuple[str, dict[str, float]]]) -> None:
    lines.extend([
        "",
        title,
        "",
        "| task | records | CER | exact_same | normalized_exact_same | WER records | English WER | Chinese records | Chinese CER |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ])
    for name, metrics in rows:
        lines.append(
            f"| {name} | {metrics['records']} | {metrics['cer']:.6f} | {metrics['exact_same']:.6f} | "
            f"{metrics['normalized_exact_same']:.6f} | {metrics['wer_records']} | "
            f"{metrics['english_wer']:.6f} | {metrics['chinese_records']} | {metrics['chinese_cer']:.6f} |"
        )


def write_eval_outputs(output_dir: str, step: int, rows: list[dict[str, Any]], metrics: dict[str, float]) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    jsonl_path = output / f"checkpoint-{step}.jsonl"
    md_path = output / f"checkpoint-{step}.md"

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    lines = [
        f"# OneASR ms-swift Eval checkpoint-{step}",
        "",
        "| records | CER | exact_same | normalized_exact_same | WER records | English WER | Chinese records | Chinese CER |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
        (
            f"| {metrics['records']} | {metrics['cer']:.6f} | {metrics['exact_same']:.6f} | "
            f"{metrics['normalized_exact_same']:.6f} | {metrics['wer_records']} | "
            f"{metrics['english_wer']:.6f} | {metrics['chinese_records']} | {metrics['chinese_cer']:.6f} |"
        ),
    ]

    task_metrics = summarize_by_task(rows)
    _append_metric_table(lines, "## Task Summary", list(task_metrics.items()))

    mode_metrics = summarize_by_eval_mode(rows)
    _append_metric_table(lines, "## Eval Mode Summary", list(mode_metrics.items()))

    no_ref_control = summarize_no_ref_control(rows)
    if no_ref_control["pairs"]:
        lines.extend([
            "",
            "## No-ref Mix Control",
            "",
            "| pairs | with_ref CER | no_ref_mix CER | CER delta(no_ref-with_ref) | with_ref acc | no_ref_mix acc | acc delta(with_ref-no_ref) | with_ref_only_correct | no_ref_also_correct | no_ref_only_correct | both_wrong |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            (
                f"| {no_ref_control['pairs']} | {no_ref_control['with_ref_cer']:.6f} | "
                f"{no_ref_control['no_ref_mix_cer']:.6f} | {no_ref_control['cer_delta_no_ref_minus_with_ref']:.6f} | "
                f"{no_ref_control['with_ref_accuracy']:.6f} | {no_ref_control['no_ref_mix_accuracy']:.6f} | "
                f"{no_ref_control['accuracy_delta_with_ref_minus_no_ref']:.6f} | "
                f"{no_ref_control['with_ref_only_correct']} | {no_ref_control['no_ref_also_correct']} | "
                f"{no_ref_control['no_ref_only_correct']} | {no_ref_control['both_wrong']} |"
            ),
        ])

    snr_buckets = summarize_by_snr(rows)
    if snr_buckets:
        lines.append("")
        lines.append("## SNR Buckets")
        lines.append("")
        lines.append("| SNR bucket | records | CER | WER | normalized_exact_same |")
        lines.append("| --- | --- | --- | --- | --- |")
        for b in snr_buckets:
            if b["records"] == 0:
                lines.append(f"| {b['snr_bucket']} | 0 | - | - | - |")
            else:
                lines.append(
                    f"| {b['snr_bucket']} | {b['records']} | {b['cer']:.6f} | "
                    f"{_fmt_metric(b.get('wer'))} | {b['normalized_exact_same']:.6f} |"
                )

    interference_buckets = summarize_by_snr_and_interference(rows)
    if interference_buckets:
        lines.append("")
        lines.append("## SNR Buckets By Dominant Interference")
        lines.append("")
        lines.append(
            "| SNR bucket | dominant interference | records | CER | WER | normalized_exact_same | "
            "target/non-target speech SNR | target/noise SNR |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for b in interference_buckets:
            lines.append(
                f"| {b['snr_bucket']} | {b['interference_source']} | {b['records']} | {b['cer']:.6f} | "
                f"{_fmt_metric(b.get('wer'))} | {b['normalized_exact_same']:.6f} | "
                f"{_fmt_metric(b.get('target_active_non_target_speech_snr_db'))} | "
                f"{_fmt_metric(b.get('target_active_noise_snr_db'))} |"
            )

    lines.append("")
    lines.append("| idx | eval_mode | language | WER | CER | normalized_exact_same | prediction | label |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for idx, row in enumerate(rows[:50], start=1):
        pred = str(row["prediction"]).replace("|", "\\|")
        label = str(row["label"]).replace("|", "\\|")
        lines.append(
            f"| {idx} | {row.get('eval_mode', WITH_REF_MODE)} | {row['language']} | {_fmt_metric(row.get('wer'))} | {row['cer']:.6f} | "
            f"{row['normalized_exact_same']} | {pred} | {label} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[oneasr eval output] {jsonl_path}")
    print(f"[oneasr eval output] {md_path}")


def report_to_swanlab(args) -> bool:
    report_to = getattr(args, "report_to", [])
    if isinstance(report_to, str):
        report_to = [report_to]
    return "swanlab" in report_to


def log_metrics_to_swanlab(args, metrics: dict[str, float], step: int) -> None:
    if not report_to_swanlab(args):
        return
    try:
        import swanlab
    except ImportError:
        return
    if swanlab.get_run() is None:
        return
    payload = {f"oneasr_eval/{key}": value for key, value in metrics.items()}
    payload["oneasr_eval/global_step"] = step
    try:
        swanlab.log(payload, step=step)
    except TypeError:
        swanlab.log(payload)


def print_eval_summary(metrics: dict[str, float], rows: list[dict[str, Any]] | None = None) -> None:
    print(
        "[oneasr eval summary] "
        f"records={metrics['records']} cer={metrics['cer']:.6f} "
        f"exact_same={metrics['exact_same']:.6f} "
        f"normalized_exact_same={metrics['normalized_exact_same']:.6f} "
        f"english_wer={metrics['english_wer']:.6f}(n={metrics['english_records']}) "
        f"chinese_cer={metrics['chinese_cer']:.6f}(n={metrics['chinese_records']})"
    )
    if rows:
        for name, item in summarize_by_eval_mode(rows).items():
            print(
                "[oneasr eval mode] "
                f"mode={name} records={item['records']} cer={item['cer']:.6f} "
                f"normalized_exact_same={item['normalized_exact_same']:.6f} "
                f"english_wer={item['english_wer']:.6f}(n={item['english_records']}) "
                f"chinese_cer={item['chinese_cer']:.6f}(n={item['chinese_records']})"
            )
        no_ref_control = summarize_no_ref_control(rows)
        if no_ref_control["pairs"]:
            print(
                "[oneasr eval no_ref_control] "
                f"pairs={no_ref_control['pairs']} "
                f"with_ref_cer={no_ref_control['with_ref_cer']:.6f} "
                f"no_ref_mix_cer={no_ref_control['no_ref_mix_cer']:.6f} "
                f"cer_delta={no_ref_control['cer_delta_no_ref_minus_with_ref']:.6f} "
                f"with_ref_only_correct={no_ref_control['with_ref_only_correct']} "
                f"no_ref_also_correct={no_ref_control['no_ref_also_correct']} "
                f"no_ref_only_correct={no_ref_control['no_ref_only_correct']} "
                f"both_wrong={no_ref_control['both_wrong']}"
            )
        for name, item in summarize_by_task(rows).items():
            print(
                "[oneasr eval task] "
                f"task={name} records={item['records']} cer={item['cer']:.6f} "
                f"normalized_exact_same={item['normalized_exact_same']:.6f} "
                f"english_wer={item['english_wer']:.6f}(n={item['english_records']}) "
                f"chinese_cer={item['chinese_cer']:.6f}(n={item['chinese_records']})"
            )


def run_internal_eval(trainer, args, dataset_path: str, source_manifest: str, output_dir: str, step: int) -> dict[str, float]:
    from swift.infer_engine import RequestConfig, TransformersEngine

    records = attach_source_manifest(load_eval_records(dataset_path), source_manifest)
    records = build_eval_requests(records)
    if not records:
        raise ValueError(f"No eval records with a final assistant message found: {dataset_path}")

    model = trainer.model
    was_training = model.training
    template = deepcopy(trainer.template)
    template.packing = False
    template.padding_free = False
    engine = TransformersEngine(
        model,
        template=template,
        max_batch_size=getattr(args, "per_device_eval_batch_size", 1),
    )
    request_config = RequestConfig(max_tokens=getattr(args, "max_new_tokens", 128), temperature=0.0)

    model.eval()
    try:
        responses = engine.infer(records, request_config=request_config, use_tqdm=False)
    finally:
        if was_training:
            model.train()

    rows = []
    for record, response in zip(records, responses):
        prediction = response.choices[0].message.content
        rows.append({**record, "prediction": prediction})

    rows = build_metric_rows(rows)
    metrics = summarize_predictions([row for row in rows if row.get("eval_mode") == WITH_REF_MODE])
    write_eval_outputs(output_dir, step, rows, metrics)
    log_metrics_to_swanlab(args, metrics, step)
    print_eval_summary(metrics, rows)
    return metrics


def infer_step_from_path(path: str) -> int:
    match = re.search(r"checkpoint-(\d+)", str(path or ""))
    return int(match.group(1)) if match else 0


def run_standalone_eval(args) -> dict[str, float]:
    from swift.arguments import InferArguments
    from swift.infer_engine import RequestConfig, TransformersEngine
    from swift.pipelines.utils import prepare_model_template

    infer_args = InferArguments(
        model=args.model or None,
        adapters=args.adapters,
        model_type=args.model_type,
        template=args.template,
        infer_backend="transformers",
        torch_dtype=args.torch_dtype,
        attn_impl=args.attn_impl,
        device_map=args.device_map,
        max_batch_size=args.batch_size,
        load_args=False,
        external_plugins=[],
    )
    model, template = prepare_model_template(infer_args)
    template.packing = False
    template.padding_free = False
    engine = TransformersEngine(model, template=template, max_batch_size=args.batch_size)
    request_config = RequestConfig(max_tokens=args.max_new_tokens, temperature=0.0)

    records = attach_source_manifest(load_eval_records(args.val_dataset), args.source_manifest)
    records = build_eval_requests(records)
    if not records:
        raise ValueError(f"No eval records with a final assistant message found: {args.val_dataset}")
    model.eval()
    responses = engine.infer(records, request_config=request_config, use_tqdm=True)

    rows = []
    for record, response in zip(records, responses):
        rows.append({**record, "prediction": response.choices[0].message.content})
    rows = build_metric_rows(rows)
    metrics = summarize_predictions([row for row in rows if row.get("eval_mode") == WITH_REF_MODE])
    step = args.step if args.step is not None else infer_step_from_path(args.adapters[0] if args.adapters else args.model)
    write_eval_outputs(args.output_dir, step, rows, metrics)
    print_eval_summary(metrics, rows)
    return metrics


def parse_standalone_eval_args(argv=None):
    parser = argparse.ArgumentParser(description="OneASR ms-swift generation eval")
    subparsers = parser.add_subparsers(dest="command", required=True)
    eval_parser = subparsers.add_parser("eval", help="Run generation eval without SFT")
    eval_parser.add_argument("--model", default="", help="Base/full model path. For LoRA, pass the base model here.")
    eval_parser.add_argument("--adapters", nargs="*", default=[], help="Optional LoRA checkpoint path(s).")
    eval_parser.add_argument("--val_dataset", required=True, help="ms-swift validation JSONL.")
    eval_parser.add_argument("--source_manifest", required=True, help="Original OneASR manifest JSONL for task/SNR analysis.")
    eval_parser.add_argument("--output_dir", required=True, help="Directory for checkpoint-*.jsonl/md outputs.")
    eval_parser.add_argument("--model_type", default="qwen3_asr")
    eval_parser.add_argument("--template", default="qwen3_asr")
    eval_parser.add_argument("--torch_dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    eval_parser.add_argument("--attn_impl", default="eager")
    eval_parser.add_argument("--device_map", default=None)
    eval_parser.add_argument("--batch_size", type=int, default=4)
    eval_parser.add_argument("--max_new_tokens", type=int, default=128)
    eval_parser.add_argument("--step", type=int, default=None)
    args = parser.parse_args(argv)
    if args.command == "eval" and not args.model and not args.adapters:
        parser.error("eval requires --model or --adapters")
    return args


def main(argv=None):
    args = parse_standalone_eval_args(argv)
    if args.command == "eval":
        return run_standalone_eval(args)
    raise ValueError(f"Unsupported command: {args.command}")


class OneASRTSASREvalCallback(TrainerCallback):
    """Run a lightweight in-process WER/CER eval after checkpoint save."""

    def on_save(self, args, state, control, **kwargs):
        if getattr(args, "process_index", 0) != 0:
            return control

        dataset_path = resolve_eval_dataset(args)
        if not dataset_path:
            print("[oneasr eval] skip: no val_dataset")
            return control
        source_manifest = resolve_source_manifest(args, dataset_path)
        if not source_manifest:
            print("[oneasr eval] skip: no source_manifest")
            return control

        step = int(getattr(state, "global_step", 0))
        default_output = str(Path(getattr(args, "output_dir", ".")) / "oneasr_eval")
        run_internal_eval(self.trainer, args, dataset_path, source_manifest, default_output, step)
        return control


callbacks_map[CALLBACK_NAME] = OneASRTSASREvalCallback


if __name__ == "__main__":
    main()
