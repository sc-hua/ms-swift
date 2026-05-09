"""OneASR lightweight ASR evaluation for ms-swift training.

This plugin intentionally stays self-contained. It evaluates a ms-swift JSONL
validation file by removing the last assistant message, generating with the
current trainer model, and computing WER/CER against that assistant target.
"""

import argparse
import json
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
    for key in ["images", "audios", "videos", "tools", "objects", "chat_template_kwargs"]:
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


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _rate(values: list[bool]) -> float:
    return sum(values) / len(values) if values else 0.0


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


def _fmt_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6f}"


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
        "",
        "| idx | language | WER | CER | normalized_exact_same | prediction | label |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for idx, row in enumerate(rows[:50], start=1):
        pred = str(row["prediction"]).replace("|", "\\|")
        label = str(row["label"]).replace("|", "\\|")
        lines.append(
            f"| {idx} | {row['language']} | {_fmt_metric(row.get('wer'))} | {row['cer']:.6f} | "
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


def print_eval_summary(metrics: dict[str, float]) -> None:
    print(
        "[oneasr eval summary] "
        f"records={metrics['records']} cer={metrics['cer']:.6f} "
        f"exact_same={metrics['exact_same']:.6f} "
        f"normalized_exact_same={metrics['normalized_exact_same']:.6f} "
        f"english_wer={metrics['english_wer']:.6f}(n={metrics['english_records']}) "
        f"chinese_cer={metrics['chinese_cer']:.6f}(n={metrics['chinese_records']})"
    )


def run_internal_eval(trainer, args, dataset_path: str, output_dir: str, step: int) -> dict[str, float]:
    from swift.infer_engine import RequestConfig, TransformersEngine

    records = load_eval_records(dataset_path)
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
    metrics = summarize_predictions(rows)
    write_eval_outputs(output_dir, step, rows, metrics)
    log_metrics_to_swanlab(args, metrics, step)
    print_eval_summary(metrics)
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
    )
    model, template = prepare_model_template(infer_args)
    template.packing = False
    template.padding_free = False
    engine = TransformersEngine(model, template=template, max_batch_size=args.batch_size)
    request_config = RequestConfig(max_tokens=args.max_new_tokens, temperature=0.0)

    records = load_eval_records(args.val_dataset)
    if not records:
        raise ValueError(f"No eval records with a final assistant message found: {args.val_dataset}")
    model.eval()
    responses = engine.infer(records, request_config=request_config, use_tqdm=True)

    rows = []
    for record, response in zip(records, responses):
        rows.append({**record, "prediction": response.choices[0].message.content})
    rows = build_metric_rows(rows)
    metrics = summarize_predictions(rows)
    step = args.step if args.step is not None else infer_step_from_path(args.adapters[0] if args.adapters else args.model)
    write_eval_outputs(args.output_dir, step, rows, metrics)
    print_eval_summary(metrics)
    return metrics


def parse_standalone_eval_args(argv=None):
    parser = argparse.ArgumentParser(description="OneASR ms-swift generation eval")
    subparsers = parser.add_subparsers(dest="command", required=True)
    eval_parser = subparsers.add_parser("eval", help="Run generation eval without SFT")
    eval_parser.add_argument("--model", default="", help="Base/full model path. For LoRA, pass the base model here.")
    eval_parser.add_argument("--adapters", nargs="*", default=[], help="Optional LoRA checkpoint path(s).")
    eval_parser.add_argument("--val_dataset", required=True, help="ms-swift validation JSONL.")
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

        step = int(getattr(state, "global_step", 0))
        default_output = str(Path(getattr(args, "output_dir", ".")) / "oneasr_eval")
        run_internal_eval(self.trainer, args, dataset_path, default_output, step)
        return control


callbacks_map[CALLBACK_NAME] = OneASRTSASREvalCallback


if __name__ == "__main__":
    main()
