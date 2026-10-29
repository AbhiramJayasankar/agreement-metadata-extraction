# Rental agreement metadata extraction

Extracts six fields from rental agreements given as Word files or scanned images, whatever the template:

| Field | Example |
|---|---|
| Agreement Value (monthly rent) | `12000` |
| Agreement Start Date | `01.04.2008` |
| Agreement End Date | `31.03.2009` |
| Renewal Notice (Days) | `60` |
| Party One (owner / lessor) | `Hanumaiah` |
| Party Two (tenant / lessee) | `Vishal Bhardwaj` |

No regex or keyword rules are used to read documents. A local language model does the reading.

## Test set results

Per-field recall = exact matches / number of documents (4 test documents).

| Field | Recall | Lenient recall |
|---|---:|---:|
| Agreement Value | **100%** | 100% |
| Agreement Start Date | **100%** | 100% |
| Agreement End Date | **75%** | 75% |
| Renewal Notice (Days) | **75%** | 75% |
| Party One | **100%** | 100% |
| Party Two | **25%** | 75% |
| Mean | **79.2%** | 87.5% |

"Lenient" ignores case, spaces and punctuation, so it separates real misses from formatting differences. Predictions are in [`outputs/test_predictions.csv`](outputs/test_predictions.csv).

The five remaining misses:

| Document | Field | Expected | Predicted | Why |
|---|---|---|---|---|
| 95980236 | End date | 31.03.2011 | 30.03.2011 | The document gives 11 months from 1 April 2010, which the label doesn't match either. |
| 95980236 | Party Two | V.V.Ravi Kian | V.V Ravi Kian | The scan reads "V.V Ravi Kian". |
| 156155545 | Notice | 30 | (empty) | Real miss. |
| 156155545 | Party Two | VYSHNAVI DAIRY SPECIALITIES Private Ltd | SRI VYSHNAVI DAIRY SPECIALITIES Private Ltd. | "SRI" is part of the company name here, but the label drops it. |
| 228094620 | Party Two | .B.Kishore | B.Kishore | The label has a stray leading dot. |

Training-set recall (each document scored with itself removed from the prompt examples), plus an extractive QA baseline for comparison:

| Method / split | Value | Start | End | Notice | Party One | Party Two | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| LLM, test | 100% | 100% | 75% | 75% | 100% | 25% | 79.2% |
| LLM, train (7 usable docs) | 85.7% | 85.7% | 42.9% | 85.7% | 28.6% | 57.1% | 64.3% |
| LLM, train (all 10 rows as given) | 60% | 60% | 30% | 60% | 20% | 40% | 45.0% |
| QA baseline, test | 75% | 0% | 0% | 0% | 0% | 0% | 12.5% |
| QA baseline, train (7 usable docs) | 57.1% | 14.3% | 14.3% | 14.3% | 0% | 0% | 16.7% |

Full tables, including lenient scores, are in [`outputs/results.md`](outputs/results.md). Every miss is listed in `outputs/errors_*.csv`.

## Approach

```
.docx ──python-docx──┐
                     ├─► text ─► Qwen3-4B-Instruct (2 worked examples) ─► facts as JSON ─► arithmetic ─► 6 fields
.png / .jpg ─Tesseract┘
```

1. **Text.** Word files are read with python-docx (paragraphs and tables, in order). Images go through Tesseract OCR. The scans are clean enough that Tesseract's output is good, and the model copes with OCR artefacts such as "1* April".
2. **Reading.** [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) gets the whole document plus two labelled training agreements as examples. It returns what the document *says* as JSON:
   - monthly rent and start date
   - the end date, only if it's written explicitly
   - the term in months
   - the notice period as a number and a unit
   - both party names, copied exactly
   
   It runs in 4-bit (NF4) on GPU and takes about 12 s per document on an RTX 4060 Laptop GPU.
3. **Arithmetic.** If no end date is written, it's computed as start date + term − 1 day (01.04.2008 + 12 months → 31.03.2009). Notice in months is multiplied by 30. This step only works on the model's answers and never looks at the document.

**Why a generative model and not span extraction.** Half the fields aren't written in the document the way the labels expect them. Agreements say "11 (Eleven) months commencing from 1st April" rather than giving an end date, and "two months notice" rather than "60". Rent appears as "Rs.9,000/-". The start date is often different from the signing date. An extractive QA model (`deepset/roberta-base-squad2`, the baseline above) can only copy spans, and it scores 12.5% on test.

**Why the model doesn't do the maths.** An earlier version asked the model for the final end date and notice in days directly. A 4B model is unreliable at calendar arithmetic, and test recall was 54%. Asking only for stated facts and computing the rest raised it to 79%.

**Prompt choices** were made using the training documents only.

## Problems in the provided data

These show up when the labels are compared with the documents:

- **Swapped labels.** `54770958` and `54945838` have swapped labels in `train.csv`. Each scan contains the other one's agreement (Asha Ramesh / Rs 5,500 vs K. Parthasarathy / Rs 8,000).
- **Missing training file.** `24158401` has a row in `train.csv`, but its file is only in `test/`.
- **Unlabelled file.** `46239065` is in `train/` but has no label.
- **Garbled text.** The `44737744` Word file is itself bad OCR text, and the rent is unreadable ("Rs9.99.7.9°").
- **Impossible dates.** Some labels have dates that don't exist: 31.04.2011, 31.02.2011, 31.11.2009.

The "7 usable docs" training view leaves out the first two problems: the swapped pair and the missing file. The CSVs themselves are unchanged, and all scores use the labels as given.

## Run it

Requirements:
- Python 3.11+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract), on `PATH` or set via `TESSERACT_CMD`
- PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/), the CUDA build for GPU

The model downloads (~8 GB) on first use. Without a GPU it runs in fp32 on CPU, which is much slower.

```bash
python -m venv .venv
.venv\Scripts\activate             # source .venv/bin/activate on Linux/macOS
pip install -r requirements.txt
```

Predict a folder of documents:

```bash
python -m metadata_extractor.predict data/test --out outputs/test_predictions.csv
```

Reproduce all the scores above (LLM and QA, train and test):

```bash
python -m metadata_extractor.evaluate
```

Run the unit tests (no model needed):

```bash
python -m pytest -q
```

## REST API

```bash
uvicorn app:app --port 8000
```

```bash
curl -F "file=@data/test/24158401-Rental-Agreement.png" http://localhost:8000/extract
```

```json
{"Agreement Value": "12000", "Agreement Start Date": "01.04.2008", "Agreement End Date": "31.03.2009",
 "Renewal Notice (Days)": "60", "Party One": "Hanumaiah", "Party Two": "Vishal Bhardwaj"}
```

`POST /extract` accepts `.docx`, `.png`, `.jpg` and `.jpeg`. `GET /health` reports the loaded model. Interactive docs are at `/docs`.

## Layout

```
metadata_extractor/
  text.py          docx reading and OCR
  llm.py           prompt, examples and model call
  fields.py        output schema and arithmetic on extracted facts
  qa_baseline.py   extractive QA baseline
  predict.py       CLI: predict a folder
  evaluate.py      CLI: per-field recall on train and test
app.py             FastAPI service
tests/             unit tests
outputs/           predictions, scores, misses, cached document text
```

## Limitations

- **Few examples.** There are only 14 labelled documents, two of them near-duplicates, so the scores are a rough guide.
- **Scan quality.** OCR quality limits images. Handwriting or poor scans would need a stronger OCR engine.
- **Model size.** A larger model would likely fix the notice and name-copying misses, but would need more GPU memory than this 8 GB setup.
