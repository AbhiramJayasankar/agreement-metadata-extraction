"""Page splitting and lazy GLM-OCR inference."""

import gc
from pathlib import Path

import numpy as np
from PIL import Image

MODEL_ID = "zai-org/GLM-OCR"
MAX_PAGE_WIDTH = 1275
_model = None
_processor = None


def split_pages(image: Image.Image) -> list[Image.Image]:
    width, height = image.size
    count = max(1, round(height / (width * 1.33)))
    grey = np.asarray(image.convert("L"), dtype=np.float32)
    cuts = [0]
    for index in range(1, count):
        boundary = int(index * height / count)
        low, high = max(0, boundary - 80), min(height, boundary + 80)
        cuts.append(low + int(np.argmax(grey[low:high].mean(axis=1))))
    cuts.append(height)
    return [image.crop((0, cuts[index], width, cuts[index + 1])) for index in range(count)]


def _load() -> None:
    global _model, _processor
    if _model is not None:
        return
    import torch
    from transformers import AutoProcessor, GlmOcrForConditionalGeneration

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _processor = AutoProcessor.from_pretrained(MODEL_ID)
    _model = GlmOcrForConditionalGeneration.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map={"": device},
    ).eval()


def image_text(path: Path, max_width: int | None = MAX_PAGE_WIDTH) -> str:
    import torch

    _load()
    with Image.open(path) as source:
        pages = split_pages(source.convert("RGB"))
    texts = []
    for page in pages:
        if max_width and page.width > max_width:
            page = page.resize((max_width, round(page.height * max_width / page.width)), Image.Resampling.LANCZOS)
        messages = [{"role": "user", "content": [
            {"type": "image", "image": page}, {"type": "text", "text": "Text Recognition:"},
        ]}]
        inputs = _processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
        ).to(_model.device)
        with torch.inference_mode():
            output = _model.generate(**inputs, max_new_tokens=4096, do_sample=False)
        texts.append(_processor.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
    return "\n\n".join(texts)


def release() -> None:
    global _model, _processor
    _model = _processor = None
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
