"""OneASR lightweight ASR evaluation for ms-swift training.

This plugin intentionally stays self-contained. It evaluates a ms-swift JSONL
validation file by removing the last assistant message, generating with the
current trainer model, and computing WER/CER against that assistant target.
"""

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from swift.callbacks.base import TrainerCallback
from swift.callbacks.mapping import callbacks_map


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


def _copy_request_fields(row: dict[str, Any], messages: list[dict[str, Any]], label: str) -> dict[str, Any]:
    request = {"messages": messages, "label": label}
    for key in ["images", "audios", "videos", "tools", "objects", "chat_template_kwargs"]:
        if key in row:
            request[key] = row[key]
    return request


def load_eval_records(dataset_path: str) -> list[dict[str, Any]]:
    records = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            messages = deepcopy(row.get("messages") or [])
            if not messages or messages[-1].get("role") != "assistant":
                continue
            label = str(messages.pop().get("content") or "")
            records.append(_copy_request_fields(row, messages, label))
    return records


def summarize_predictions(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {"records": 0, "wer": 0.0, "cer": 0.0, "exact": 0.0}
    wers = [word_error_rate(row["prediction"], row["label"]) for row in rows]
    cers = [char_error_rate(row["prediction"], row["label"]) for row in rows]
    exact = [
        parse_asr_text(row["prediction"]).strip() == parse_asr_text(row["label"]).strip()
        for row in rows
    ]
    return {
        "records": len(rows),
        "wer": sum(wers) / len(wers),
        "cer": sum(cers) / len(cers),
        "exact": sum(exact) / len(exact),
    }


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
        "| records | WER | CER | exact |",
        "| --- | --- | --- | --- |",
        f"| {metrics['records']} | {metrics['wer']:.6f} | {metrics['cer']:.6f} | {metrics['exact']:.6f} |",
        "",
        "| idx | prediction | label |",
        "| --- | --- | --- |",
    ]
    for idx, row in enumerate(rows[:50], start=1):
        pred = str(row["prediction"]).replace("|", "\\|")
        label = str(row["label"]).replace("|", "\\|")
        lines.append(f"| {idx} | {pred} | {label} |")
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
        row = {
            "prediction": prediction,
            "label": record["label"],
            "prediction_text": parse_asr_text(prediction),
            "label_text": parse_asr_text(record["label"]),
            "wer": word_error_rate(prediction, record["label"]),
            "cer": char_error_rate(prediction, record["label"]),
        }
        rows.append(row)

    metrics = summarize_predictions(rows)
    write_eval_outputs(output_dir, step, rows, metrics)
    log_metrics_to_swanlab(args, metrics, step)
    print(
        "[oneasr eval summary] "
        f"records={metrics['records']} wer={metrics['wer']:.6f} "
        f"cer={metrics['cer']:.6f} exact={metrics['exact']:.6f}"
    )
    return metrics


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
