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
    app.state.extractor = make_extractor(
        "llm", model=os.environ.get("METADATA_MODEL"),
        quantization=os.environ.get("METADATA_QUANTIZATION", "auto"),
        device=os.environ.get("METADATA_DEVICE", "auto"),
    )
    app.state.model_lock = Lock()
    yield
    from metadata_extractor.evaluate import release
    release(app.state.extractor)


app = FastAPI(title="Rental Agreement Metadata", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    extractor = app.state.extractor
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
        try:
            text = extract_text(path, cache_dir=None)
        except Exception as error:
            raise HTTPException(status_code=422, detail="Could not read the uploaded document") from error
        if not text:
            raise HTTPException(status_code=422, detail="No readable text in document")
        try:
            with app.state.model_lock:
                return app.state.extractor.extract(text)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
