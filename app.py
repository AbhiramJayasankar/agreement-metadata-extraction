"""HTTP service for rental agreement metadata."""

from contextlib import asynccontextmanager
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock

from fastapi import FastAPI, File, HTTPException, UploadFile

from metadata_extractor.dataset import ROOT
from metadata_extractor.predict import make_extractor
from metadata_extractor.text import SUPPORTED_SUFFIXES, extract_text


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.extractor = None
    app.state.model_lock = Lock()
    yield
    from metadata_extractor.ocr import release as release_ocr
    from metadata_extractor.evaluate import release
    release_ocr()
    if app.state.extractor is not None:
        release(app.state.extractor)


def get_extractor():
    if app.state.extractor is None:
        app.state.extractor = make_extractor(
            "llm", model=os.environ.get("METADATA_MODEL"),
            quantization=os.environ.get("METADATA_QUANTIZATION", "auto"),
            device=os.environ.get("METADATA_DEVICE", "auto"),
        )
    return app.state.extractor


app = FastAPI(title="Rental Agreement Metadata", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    extractor = app.state.extractor
    if extractor is None:
        return {"status": "ok", "model": os.environ.get("METADATA_MODEL", "Qwen/Qwen3.5-4B"), "loaded": False}
    return {"status": "ok", "model": extractor.model_id, "quantization": extractor.quantization, "device": extractor.device}


@app.post("/extract")
def extract(file: UploadFile = File(...)) -> dict[str, str]:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(status_code=415, detail="Upload a .docx, .png, .jpg, or .jpeg file")
    with TemporaryDirectory(prefix="rental-upload-", dir=ROOT) as directory:
        path = Path(directory) / f"document{suffix}"
        with path.open("wb") as destination:
            while chunk := file.file.read(1024 * 1024):
                destination.write(chunk)
        with app.state.model_lock:
            try:
                text = extract_text(path, cache_dir=None, engine=os.environ.get("OCR_ENGINE", "glm"))
            except Exception as error:
                raise HTTPException(status_code=422, detail="Could not read the uploaded document") from error
            if not text:
                raise HTTPException(status_code=422, detail="No readable text in document")
            try:
                return get_extractor().extract(text)
            except ValueError as error:
                raise HTTPException(status_code=422, detail=str(error)) from error
