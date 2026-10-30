"""Predict metadata for a directory without reading its labels."""

import argparse
import logging
from pathlib import Path
from time import perf_counter

from .dataset import ROOT, files_by_stem, prediction_row, write_predictions
from .text import extract_text


def make_extractor(
    method: str, model: str | None = None, quantization: str = "auto",
    device: str = "auto", data_dir: Path = ROOT / "data",
):
    if method == "qa":
        from .qa_baseline import QAExtractor
        return QAExtractor(model_id=model or "deepset/roberta-base-squad2", device=device)
    from .llm import DEFAULT_MODEL, LLMExtractor
    return LLMExtractor(model_id=model or DEFAULT_MODEL, quantization=quantization, device=device, data_dir=data_dir)


def prepare_texts(paths, engine: str = "glm") -> dict[Path, str]:
    from .ocr import release

    try:
        return {path: extract_text(path, engine=engine) for path in paths}
    finally:
        release()


def predict_file(path: Path, extractor, leave_one_out: bool = False, text: str | None = None) -> tuple[dict[str, str], float]:
    from .text import document_stem

    start = perf_counter()
    if text is None:
        text = extract_text(path)
    stem = document_stem(path)
    answers = extractor.extract(text, exclude_stem=stem if leave_one_out else None)
    seconds = perf_counter() - start
    logging.info("%s: %.2fs", path.name, seconds)
    return prediction_row(stem, answers), seconds


def predict_directory(directory: Path, extractor, out: Path, texts: dict[Path, str]) -> list[dict[str, str]]:
    rows = []
    for path in files_by_stem(directory).values():
        row, _ = predict_file(path, extractor, text=texts[path])
        rows.append(row)
    if not rows:
        raise ValueError(f"No supported documents in {directory}")
    write_predictions(out, rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--out", type=Path, default=Path("outputs/test_predictions.csv"))
    parser.add_argument("--method", choices=("llm", "qa"), default="llm")
    parser.add_argument("--model")
    parser.add_argument("--ocr", choices=("glm", "tesseract"), default="glm")
    parser.add_argument("--quantization", choices=("auto", "nf4", "fp16"), default="auto")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    start = perf_counter()
    texts = prepare_texts(files_by_stem(args.directory).values(), args.ocr)
    logging.info("Text preparation (%s): %.2fs", args.ocr, perf_counter() - start)
    extractor = make_extractor(args.method, args.model, args.quantization, args.device)
    predict_directory(args.directory, extractor, args.out, texts)


if __name__ == "__main__":
    main()
