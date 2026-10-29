"""Dataset I/O; labels remain strings, including invalid annotated dates."""

import csv
from pathlib import Path

from .fields import CSV_COLUMNS, FIELDS, FIELD_TO_COLUMN
from .text import SUPPORTED_SUFFIXES, document_stem

ROOT = Path(__file__).resolve().parent.parent


def read_labels(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(CSV_COLUMNS):
            raise ValueError(f"Unexpected CSV columns in {path}: {reader.fieldnames}")
        return [{key: (value or "").strip() for key, value in row.items()} for row in reader]


def files_by_stem(directory: Path) -> dict[str, Path]:
    files = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            stem = document_stem(path)
            if stem in files:
                raise ValueError(f"Duplicate document stem: {stem}")
            files[stem] = path
    return files


def prediction_row(stem: str, answers: dict[str, str]) -> dict[str, str]:
    return {"File Name": stem, **{FIELD_TO_COLUMN[field]: answers.get(field, "") for field in FIELDS}}


def write_predictions(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
