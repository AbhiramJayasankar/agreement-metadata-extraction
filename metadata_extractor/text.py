"""Document text and OCR, with a reusable plain-text cache."""

import os
from collections.abc import Iterator
from pathlib import Path

from docx import Document
from docx.document import Document as DocumentObject
from docx.table import Table, _Cell
from docx.text.paragraph import Paragraph
from PIL import Image, ImageOps
import pytesseract

SUPPORTED_SUFFIXES = {".docx", ".png", ".jpg", ".jpeg"}
TESSERACT_WINDOWS = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")


def document_stem(path: Path) -> str:
    """Treat .pdf.docx as a compound extension in the supplied dataset."""
    stem = path.stem
    return stem[:-4] if stem.lower().endswith(".pdf") else stem


def _blocks(parent: DocumentObject | _Cell) -> Iterator[str]:
    element = parent.element.body if isinstance(parent, DocumentObject) else parent._tc
    for child in element.iterchildren():
        if child.tag.endswith("}p"):
            yield Paragraph(child, parent).text
        elif child.tag.endswith("}tbl"):
            table = Table(child, parent)
            seen = set()
            for row in table.rows:
                for cell in row.cells:
                    # Merged cells share XML and must appear only once.
                    if cell._tc not in seen:
                        seen.add(cell._tc)
                        yield from _blocks(cell)


def docx_text(path: Path) -> str:
    return "\n".join(_blocks(Document(path)))


def image_text(path: Path) -> str:
    executable = os.environ.get("TESSERACT_CMD")
    if executable:
        pytesseract.pytesseract.tesseract_cmd = executable
    elif TESSERACT_WINDOWS.exists():
        pytesseract.pytesseract.tesseract_cmd = str(TESSERACT_WINDOWS)
    with Image.open(path) as source:
        image = ImageOps.grayscale(ImageOps.exif_transpose(source))
        if image.width < 1200:
            scale = 1200 / image.width
            image = image.resize((1200, round(image.height * scale)), Image.Resampling.LANCZOS)
        return pytesseract.image_to_string(image)


def extract_text(path: str | Path, cache_dir: str | Path | None = "outputs/text") -> str:
    """Read DOCX or OCR an image, normalizing whitespace only."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported file extension: {suffix}")
    cache = Path(cache_dir) / f"{document_stem(path)}.txt" if cache_dir is not None else None
    if cache and cache.exists() and cache.stat().st_mtime_ns >= path.stat().st_mtime_ns:
        return cache.read_text(encoding="utf-8")
    raw = docx_text(path) if suffix == ".docx" else image_text(path)
    text = "\n".join(" ".join(line.split()) for line in raw.splitlines() if line.strip())
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(text, encoding="utf-8")
    return text
