"""Per-field recall for the training set, its readable documents, and the test set."""

import argparse
import csv
import gc
import logging
from pathlib import Path
from time import perf_counter

from .dataset import ROOT, files_by_stem, read_labels, write_predictions
from .fields import CSV_COLUMNS, FIELDS, FIELD_TO_COLUMN
from .predict import make_extractor, predict_file, prepare_texts
from .llm import DEFAULT_MODEL

UNREADABLE = {"44737744-Maddireddy-Bhargava-Reddy-Rental-Agreement"}


def readable_labels(labels: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in labels if row["File Name"] not in UNREADABLE]


def lenient(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def recall_rows(
    labels: list[dict[str, str]], predictions: list[dict[str, str]], missing: set[str] | None = None,
) -> tuple[list[dict], list[dict]]:
    if not labels:
        raise ValueError("Cannot evaluate an empty label set")
    predicted = {row["File Name"]: row for row in predictions}
    if len(predicted) != len(predictions):
        raise ValueError("Duplicate prediction identifiers")
    missing = missing or set()
    strict_counts = dict.fromkeys(FIELDS, 0)
    lenient_counts = dict.fromkeys(FIELDS, 0)
    mistakes = []
    for gold in labels:
        stem = gold["File Name"]
        available = stem in predicted and stem not in missing
        for field, column in FIELD_TO_COLUMN.items():
            expected = gold[column].strip()
            actual = predicted.get(stem, {}).get(column, "").strip()
            exact = available and actual == expected
            relaxed = available and lenient(actual) == lenient(expected)
            strict_counts[field] += int(exact)
            lenient_counts[field] += int(relaxed)
            if not exact:
                mistakes.append({"file": stem, "field": field, "expected": expected, "predicted": actual})
    total = len(labels)
    scores = [{
        "Field": field, "Exact Matches": strict_counts[field], "Lenient Matches": lenient_counts[field],
        "Total Documents": total, "Recall": strict_counts[field] / total,
        "Lenient Recall": lenient_counts[field] / total,
    } for field in FIELDS]
    return scores, mistakes


def evaluate_split(split: str, method: str, extractor, data_dir: Path, output_dir: Path, texts: dict[Path, str]) -> tuple[dict, list[float]]:
    files = files_by_stem(data_dir / split)
    labels = read_labels(data_dir / "train.csv") if split == "train" else None
    if labels is not None and method == "qa":
        labels = readable_labels(labels)
    stems = [row["File Name"] for row in labels] if labels is not None else list(files)
    predictions, timings, missing = [], [], set()
    for stem in stems:
        if stem not in files:
            logging.warning("Missing %s document: %s", split, stem)
            missing.add(stem)
            predictions.append(dict.fromkeys(CSV_COLUMNS, "") | {"File Name": stem})
            continue
        row, seconds = predict_file(files[stem], extractor, leave_one_out=split == "train", text=texts[files[stem]])
        predictions.append(row)
        timings.append(seconds)
    if labels is None:
        labels = read_labels(data_dir / f"{split}.csv")
    scores, mistakes = recall_rows(labels, predictions, missing)
    views = {}
    if split == "train":
        readable_scores, _ = recall_rows(readable_labels(labels), predictions, missing)
        views[f"{method.upper()} train readable"] = readable_scores
        if method == "llm":
            views["LLM train"] = scores
    else:
        views[f"{method.upper()} test"] = scores
    if method == "llm":
        write_predictions(output_dir / f"{split}_predictions.csv", predictions)
        with (output_dir / f"errors_{split}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("file", "field", "expected", "predicted"))
            writer.writeheader()
            writer.writerows(mistakes)
    return views, timings


def write_results(output_dir: Path, results: dict[str, list[dict]], timings: list[float], model: str, device: str, quantization: str, text_seconds: float = 0, ocr_engine: str = "glm") -> None:
    lines = []
    order = ("LLM test", "LLM train", "LLM train readable", "QA test", "QA train readable")
    for title, metric in (("Strict recall", "Recall"), ("Lenient recall", "Lenient Recall")):
        lines.extend([title, "", "| Method / split | Value | Start | End | Notice | Party One | Party Two | Mean |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|"])
        for name in order:
            if name in results:
                values = [row[metric] for row in results[name]]
                numbers = values + [sum(values) / len(values)]
                lines.append(f"| {name} | " + " | ".join(f"{value:.1%}" for value in numbers) + " |")
        lines.append("")
    mean_seconds = f"{sum(timings) / len(timings):.2f}" if timings else "n/a"
    lines.extend([
        "- Strict recall strips leading/trailing spaces; lenient recall also ignores case, spaces and punctuation. Empty answers match empty labels.",
        "- Train has 9 rows (unlabelled 46239065 is not scored); train readable leaves out 44737744, whose Word file is unreadable OCR text. Test has all 4.",
        "- Labels were corrected where they contradict the document; README.md lists every change. Train: swapped rows 54770958/54945838 put back, 24158401 removed (its file is in test/), impossible end dates fixed, names copied as the documents write them.",
        "- test.csv: 228094620 Party Two (.B.Kishore -> B.Kishore) and 95980236 end date (31.03.2011 -> 28.02.2011).",
        "- The prompt holds two training documents as examples; a training document is never its own example.",
        "- QA answers are spans copied from the text, so it cannot produce computed end dates or notice in days.",
        f"- LLM {model} ({quantization}, {'GPU' if device == 'cuda' else 'CPU'}): mean {mean_seconds} seconds/document, excluding model loading.",
        f"- Text preparation ({ocr_engine} OCR and DOCX, including cache reads): {text_seconds:.2f} seconds, before LLM loading.",
        "",
    ])
    (output_dir / "results.md").write_text("\n".join(lines), encoding="utf-8")


def release(extractor) -> None:
    import torch
    if torch.cuda.is_available():
        logging.info("Peak allocated VRAM: %.2f GiB", torch.cuda.max_memory_allocated() / 2**30)
    del extractor.model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "test", "both"), default="both")
    parser.add_argument("--method", choices=("llm", "qa", "all"), default="all")
    parser.add_argument("--model")
    parser.add_argument("--ocr", choices=("glm", "tesseract"), default="glm")
    parser.add_argument("--quantization", choices=("auto", "nf4", "fp16"), default="auto")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results, llm_timings = {}, []
    splits = ("train", "test") if args.split == "both" else (args.split,)
    start = perf_counter()
    texts = prepare_texts([path for split in splits for path in files_by_stem(args.data_dir / split).values()], args.ocr)
    text_seconds = perf_counter() - start
    logging.info("Text preparation (%s): %.2fs", args.ocr, text_seconds)
    llm_model, llm_device, llm_quantization = args.model or DEFAULT_MODEL, args.device, args.quantization
    for method in (("llm", "qa") if args.method == "all" else (args.method,)):
        model = args.model if method == "llm" or args.method == "qa" else None
        extractor = make_extractor(method, model, args.quantization, args.device, data_dir=args.data_dir)
        if method == "llm":
            llm_model, llm_device = extractor.model_id, extractor.device
            llm_quantization = extractor.quantization
        for split in splits:
            views, timings = evaluate_split(split, method, extractor, args.data_dir, args.output_dir, texts)
            results.update(views)
            if method == "llm":
                llm_timings.extend(timings)
        release(extractor)
    write_results(args.output_dir, results, llm_timings, llm_model, llm_device, llm_quantization, text_seconds, args.ocr)


if __name__ == "__main__":
    main()

