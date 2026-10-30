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

No regex or keyword rules are used to read documents. Two small local models do the reading: an OCR model for scans and a language model for the fields.

## Test set results

Per-field recall = exact matches / number of documents (4 test documents).

| Field | Recall |
|---|---:|
| Agreement Value | **100%** |
| Agreement Start Date | **100%** |
| Agreement End Date | **100%** |
| Renewal Notice (Days) | **100%** |
| Party One | **100%** |
| Party Two | **100%** |
| Mean | **100%** |

Predictions are in [`outputs/test_predictions.csv`](outputs/test_predictions.csv).

**The labels had errors, and I fixed them.** The dataset looks deliberately messy, so I checked every miss against its document. Where a label contradicts the document, I corrected it in the CSV. Where the document simply doesn't contain the answer, I left the label alone.

Test labels (`test.csv`):

| Document | Field | Original label | Corrected to | What the document says |
|---|---|---|---|---|
| 228094620 | Party Two | .B.Kishore | B.Kishore | "Mr.B.Kishore". Whoever labelled it removed "Mr" and left the dot behind. |
| 95980236 | End date | 31.03.2011 | 28.02.2011 | "11 (Eleven) months commencing from 1st April 2010", which ends on 28 February 2011. |

Training labels (`train.csv`):

| Document | Problem | Fix |
|---|---|---|
| 54770958, 54945838 | Swapped: each scan contains the other one's agreement | Swapped the rows back |
| 24158401 | Its file is in `test/`, and the row duplicates its test label | Removed the row |
| 36199312 | End date 31.04.2011 doesn't exist (11 months from 1 May 2010) | 31.03.2011 |
| 47854715 | End date 31.02.2011 doesn't exist (11 months from 1 April 2010) | 28.02.2011 |
| 18325926 | End date 31.11.2009 doesn't exist ("eleven months from" 5 December 2008) | 04.11.2009 |
| 18325926 | Party One "MR.K.Kuttan" keeps the title every other label drops | K.Kuttan |
| 6683127, 6683129 | "GERALDINE Q." and "Ingles, Jr." only appear in the signature lines of the full contract; the page-1 file never contains them. Both contracts name the parties as "GERALDINE O. GALINATO" and "Ingles. Jr." | Copied the names as the contracts write them |
| 54770958 | "Deepthi" and "Ramesh K.N"; the scan writes "Deepti" and "Ramesh.K.N" everywhere, signatures included | Copied as written |

With the original labels, the scores were 91.7% on test and 71.4% on the 7 training documents I could use (50% on all 10 rows as given).

Recall with the corrected labels (each training document is scored with itself removed from the prompt examples), plus an extractive QA baseline:

| Method / split | Value | Start | End | Notice | Party One | Party Two | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| LLM, test | 100% | 100% | 100% | 100% | 100% | 100% | **100%** |
| LLM, train (8 readable docs) | 100% | 87.5% | 75% | 100% | 87.5% | 100% | **91.7%** |
| LLM, train (all 9 labelled docs) | 88.9% | 77.8% | 66.7% | 100% | 77.8% | 100% | **85.2%** |
| QA baseline, test | 75% | 0% | 0% | 0% | 0% | 0% | 12.5% |
| QA baseline, train (8 readable docs) | 75% | 12.5% | 12.5% | 0% | 0% | 0% | 16.7% |

The 8 remaining training misses:
- **44737744 (4 misses).** The Word file is itself bad OCR text: the rent reads "Rs 9.99.7.9°" and the dates aren't there. The labels are probably right, so I left them, and "readable" leaves this file out.
- **54770958 (2 misses).** The label says 21.04.2011 to 19.02.2012, but the scan never states a tenancy start date, so there is nothing to check it against. I left it as given.
- **18325926 (2 misses), my model's errors.** It keeps the father's name ("K.Kuttan S/o Kelu Aehari (Late)") and gives 30.11.2009 as the end date instead of 04.11.2009.

Full tables are in [`outputs/results.md`](outputs/results.md), and every miss is listed in `outputs/errors_*.csv`.

## Approach

```
.docx ──python-docx──┐
                     ├─► text ─► Qwen3.5-4B (2 worked examples) ─► facts as JSON ─► arithmetic ─► 6 fields
.png / .jpg ─GLM-OCR─┘
```

1. **Text.** Word files are read with python-docx (paragraphs and tables, in order). Scans are split into pages at the whitest row near each page boundary, scaled to at most 1275 px wide, and read by [GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) (0.9B).
2. **Reading.** [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) gets the document text plus two labelled training agreements as examples. It returns what the document *says* as JSON:
   - monthly rent and start date
   - the end date, only if it's written explicitly
   - the term in months
   - the notice period as a number and a unit
   - both party names, copied exactly, without titles; a company rather than the person signing for it
3. **Arithmetic and formatting.** If no end date is written, it's start date + term − 1 day (01.04.2008 + 12 months → 31.03.2009). Notice in months is multiplied by 30. Each name in a party ("Asha Ramesh & Mr.Ramesh.K.N") loses one leading title (Sri, Mr, Mrs, ...), and trailing punctuation is dropped. This step only works on the model's answers and never looks at the document.

**Why a generative model and not span extraction.** Half the fields aren't written the way the labels expect them. Agreements say "11 (Eleven) months commencing from 1st April" rather than giving an end date, and "two months notice" rather than "60". An extractive QA model (`deepset/roberta-base-squad2`, the baseline above) can only copy spans and scores 12.5% on test.

**Why the model doesn't do the maths.** A 4B model is unreliable at calendar arithmetic. Asking only for stated facts and computing the rest took test recall from 54% to 79% in an earlier version.

**How the models were chosen.** I compared current small OCR and vision-language models on the five scans and the training documents (the test set was only scored, never used to choose):

| OCR + language model | Test | Train (7 usable docs) |
|---|---:|---:|
| OvisOCR3 + Qwen3-4B-Instruct | 87.5% | 64.3% |
| GLM-OCR + Qwen3-4B-Instruct | 87.5% | 66.7% |
| Tesseract + Qwen3.5-4B | 83.3% | 71.4% |
| GLM-OCR + Qwen3.5-4B | 87.5% | 71.4% |
| **GLM-OCR + Qwen3.5-4B, final prompt and name formatting** | **91.7%** | **71.4%** |

I made this comparison before correcting any labels, so the best possible test score here was 91.7%. The first four rows use the same prompt. My first version (Tesseract + Qwen3-4B-Instruct) scored 79.2% on test and 64.3% on train.

- **OCR.** Tesseract drops punctuation inside names ("V.V.Ravi Kian" → "V.V Ravi Kian"), which exact matching punishes. OvisOCR3 fixed that but misread "Kartheek" as "Karthreek". GLM-OCR read every name, date and amount correctly. On the 2550 px scan it gives the same names at 1275 px width in a third of the time and memory.
- **Language model.** Qwen3.5-4B scored higher than Qwen3-4B-Instruct on the training documents, mostly on names. With the first prompt it named a company's signatory as the tenant; the final prompt asks for the company itself.
- **Reading page images directly.** I also tried giving Qwen3.5-4B the page images instead of OCR text. Three-page scans ran out of memory on an 8 GB GPU, so OCR text it is.
- **LFM3-VL-2B.** Liquid's new 2B vision model is small enough to read the scans itself, which would remove the OCR model. It reads the text fine ("V.V.Ravi Kian", the rent), but it's much weaker at the extraction: it copies dates from the example agreements, swaps owner and tenant, and adds parentage to names. Before I corrected the training labels, its best setup (page images capped at 1024 tokens per page, no examples) scored 79.2% on test and 47.6% on train. Used as a text model in place of Qwen, it scored 45.8% and 50.0%. On the same labels, this pipeline scored 100% and 71.4%.

## Problems in the provided data

- **Wrong labels.** The two tables above: swapped rows, a test document labelled in train, dates that don't exist (31.04, 31.02, 31.11), and names that don't match the document. All corrected in the CSVs.
- **Unlabelled file.** `46239065` is in `train/` but has no label, so it isn't scored.
- **Garbled text.** The `44737744` Word file is itself bad OCR text, and the rent is unreadable ("Rs9.99.7.9°").

## Run it

Requirements:
- Python 3.11+
- PyTorch and torchvision from [pytorch.org](https://pytorch.org/get-started/locally/), the CUDA build for GPU
- An NVIDIA GPU with 8 GB is enough. Without one, everything runs on CPU, much more slowly.

```bash
python -m venv .venv
.venv\Scripts\activate             # source .venv/bin/activate on Linux/macOS
pip install -r requirements.txt
```

The two models (about 11 GB) download on first use. On an RTX 4060 Laptop GPU, OCR takes about 25-45 s per scan and the language model about 9 s per document. OCR text is cached in `outputs/text`.

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

Options: `--ocr tesseract` uses Tesseract instead of GLM-OCR (install the [Tesseract](https://github.com/tesseract-ocr/tesseract) executable; set `TESSERACT_CMD` if it isn't on `PATH`). `--model` takes another Hugging Face model, for example `Qwen/Qwen3-4B-Instruct-2507`.

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

`POST /extract` accepts `.docx`, `.png`, `.jpg` and `.jpeg`. Models load on the first request, and both fit on an 8 GB GPU together. `GET /health` reports the model. Set `OCR_ENGINE=tesseract` or `METADATA_MODEL=<model id>` to change the defaults. Interactive docs are at `/docs`.

## Layout

```
metadata_extractor/
  text.py          docx reading, OCR dispatch and text cache
  ocr.py           page splitting and GLM-OCR
  llm.py           prompt, examples and model call
  fields.py        output schema, arithmetic and name formatting
  qa_baseline.py   extractive QA baseline
  predict.py       CLI: predict a folder
  evaluate.py      CLI: per-field recall on train and test
app.py             FastAPI service
tests/             unit tests
outputs/           predictions, scores, misses, cached document text
```

## Limitations

- **Few examples.** There are only 14 labelled documents, two of them near-duplicates, so the scores are a rough guide.
- **Typed scans only.** All the scans are clean typed pages. Handwriting or photos of documents haven't been tested.
- **Speed.** OCR dominates on scans. A llama.cpp or vLLM server would be faster than plain transformers but is harder to set up on Windows.
