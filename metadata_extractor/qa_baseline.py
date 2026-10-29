"""Extractive QA comparison, using overlapping token windows."""

from .fields import format_fields

QUESTIONS = {
    "Agreement Value": "What is the monthly rent amount?",
    "Agreement Start Date": "On what date does the tenancy commence?",
    "Agreement End Date": "On what date does the tenancy end?",
    "Renewal Notice (Days)": "How much advance notice must either party give to terminate or renew the tenancy?",
    "Party One": "What is the name of the landlord or lessor?",
    "Party Two": "What is the name of the tenant or lessee?",
}


class QAExtractor:
    def __init__(self, model_id: str = "deepset/roberta-base-squad2", device: str = "auto") -> None:
        import torch
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer

        self.model_id = model_id
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else device
        if self.device == "auto":
            self.device = "cpu"
        self.quantization = "fp32"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_id).to(self.device).eval()

    def extract(self, text: str, exclude_stem: str | None = None) -> dict[str, str]:
        import torch

        answers = {}
        for field, question in QUESTIONS.items():
            tokens = self.tokenizer(
                question, text, truncation="only_second", max_length=384,
                stride=128, return_overflowing_tokens=True, return_offsets_mapping=True,
                padding="max_length", return_tensors="pt",
            )
            offsets = tokens.pop("offset_mapping")
            tokens.pop("overflow_to_sample_mapping")
            best_score, best_answer = float("-inf"), ""
            for start in range(0, len(tokens.input_ids), 8):
                batch = {key: value[start:start + 8].to(self.device) for key, value in tokens.items()}
                with torch.inference_mode():
                    output = self.model(**batch)
                for local in range(len(batch["input_ids"])):
                    window = start + local
                    valid = torch.tensor([seq == 1 and end > begin for seq, (begin, end) in zip(tokens.sequence_ids(window), offsets[window].tolist())], device=self.device)
                    # Include the SQuAD2 null span in each window's probability space.
                    valid[0] = True
                    starts = output.start_logits[local].masked_fill(~valid, float("-inf")).softmax(-1)
                    ends = output.end_logits[local].masked_fill(~valid, float("-inf")).softmax(-1)
                    null_score = float((starts[0] * ends[0]).item())
                    scores = starts[:, None] * ends[None, :]
                    positions = torch.arange(len(starts), device=self.device)
                    allowed = (positions[None, :] >= positions[:, None]) & (positions[None, :] - positions[:, None] < 64)
                    allowed[0, :] = False
                    allowed[:, 0] = False
                    scores.masked_fill_(~allowed, -1)
                    index = int(scores.argmax().item())
                    first, last = divmod(index, len(starts))
                    score = float(scores[first, last].item())
                    if score > best_score:
                        begin, end = int(offsets[window, first, 0]), int(offsets[window, last, 1])
                        best_score = score
                        best_answer = text[begin:end] if score > null_score else ""
            answers[field] = best_answer
        return format_fields(answers)
