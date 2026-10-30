# Rental agreement metadata extraction

Extracts six fields from rental agreements, given as Word files or scanned images, whatever the template:

| Field | Example |
|---|---|
| Agreement Value (monthly rent) | `12000` |
| Agreement Start Date | `01.04.2008` |
| Agreement End Date | `31.03.2009` |
| Renewal Notice (Days) | `60` |
| Party One (owner / lessor) | `Hanumaiah` |
| Party Two (tenant / lessee) | `Vishal Bhardwaj` |

No regex or keyword rules read the documents. Two small local models do the reading: an OCR model for scans and a language model for the fields. Everything runs on one 8 GB GPU, and there is a REST API.

## Result

Per-field recall = exact matches / number of documents.

| Split | Value | Start | End | Notice | Party One | Party Two | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Test (4 docs)** | 100% | 100% | 100% | 100% | 100% | 100% | **100%** |
| **Train (8 readable docs)** | 100% | 100% | 100% | 100% | 100% | 100% | **100%** |
| Train (all 9 labelled docs) | 88.9% | 88.9% | 88.9% | 100% | 88.9% | 100% | 92.6% |

Training documents are scored with themselves removed from the prompt examples. The only misses are on one Word file that is itself unreadable (see below). Test predictions are in [`outputs/test_predictions.csv`](outputs/test_predictions.csv), full tables in [`outputs/results.md`](outputs/results.md), and every miss in `outputs/errors_*.csv`.

## How I got there

### 1. Looking at the data

There are 10 labelled training files and 4 test files, a mix of Word documents and tall multi-page PNG scans. Before modelling I read every document against its label. The data turned out to be messy in ways that matter for an exact-match metric:

- **Swapped labels.** `54770958` and `54945838` each contain the other one's agreement.
- **A test document in train.** `24158401` has a row in `train.csv`, but its file is only in `test/`, and the row duplicates its test label.
- **Dates that don't exist.** End dates like 31.04.2011, 31.02.2011 and 31.11.2009.
- **Names that don't match the document.** For example "GERALDINE Q." where the contract names "GERALDINE O.", and ".B.Kishore" where the document says "Mr.B.Kishore" (the title was removed but its dot left behind).
- **Facts the document doesn't contain.** One scan's label gives a start and end date that appear nowhere in it.
- **A garbled Word file.** `44737744` is itself bad OCR text: the rent reads "Rs 9.99.7.9°", the dates are blank and the owner is "M. V, V« VI JAY A SHANKAR". I checked the file for any other source (embedded images, hidden text, fields) and there is none.
- **An unlabelled file.** `46239065` is in `train/` but has no label.

Where a label contradicts its document, I corrected it in the CSV. Where the answer isn't in the document at all, I left it empty. The garbled file keeps its labels, which are probably right, and is reported separately as unreadable.

Test labels (`test.csv`):

| Document | Field | Label as given | Corrected to | What the document says |
|---|---|---|---|---|
| 228094620 | Party Two | .B.Kishore | B.Kishore | "Mr.B.Kishore" |
| 95980236 | End date | 31.03.2011 | 28.02.2011 | "11 (Eleven) months commencing from 1st April 2010" |

Training labels (`train.csv`):

| Document | Problem | Fix |
|---|---|---|
| 54770958, 54945838 | Each scan contains the other one's agreement | Swapped the rows back |
| 24158401 | Its file is in `test/`, and the row duplicates its test label | Removed the row |
| 36199312 | End date 31.04.2011 doesn't exist (11 months from 1 May 2010) | 31.03.2011 |
| 47854715 | End date 31.02.2011 doesn't exist (11 months from 1 April 2010) | 28.02.2011 |
| 18325926 | End date 31.11.2009 doesn't exist ("eleven months from" 5 December 2008) | 04.11.2009 |
| 18325926 | Party One "MR.K.Kuttan" keeps the title every other label drops | K.Kuttan |
| 6683127, 6683129 | "GERALDINE Q." and "Ingles, Jr." only appear in the signature lines of the full contract, and the page-1 file never contains them. Both contracts name the parties "GERALDINE O. GALINATO" and "Ingles. Jr." | Copied the names as the contracts write them |
| 54770958 | "Deepthi" and "Ramesh K.N"; the scan writes "Deepti" and "Ramesh.K.N" everywhere, signatures included | Copied as written |
| 54770958 | Start 21.04.2011 and end 19.02.2012 aren't in the scan. Its only date is "executed on 20 April 2011", and the term runs "from the date of Tenancy", which isn't given | Left empty |

On the labels as given, the final pipeline scores 91.7% on test (the two cells above are its only misses) and 71.4% on the 7 training documents that are usable without fixes.

### 2. Why not rules or span extraction

Half the fields aren't written the way the labels expect them. Agreements say "11 (Eleven) months commencing from 1st April" instead of an end date, and "two months notice" instead of "60". The brief rules out regex, and an extractive QA model (`deepset/roberta-base-squad2`) can only copy spans out of the text: it scores 12.5% on test. The task needs a model that reads.

### 3. A small language model that reports facts, and code that does the maths

A 4B model reads well but is unreliable at calendar arithmetic. So I ask it only for what the document *says*, as JSON:

- monthly rent and start date;
- the end date, only if the document writes it;
- the term in months;
- the notice period as a number and a unit;
- both party names, copied exactly, without titles or parentage, and the company rather than the person signing for it.

Two labelled training agreements go in the prompt as worked examples. The code then does the rest: no written end date means start + term − 1 day (01.04.2008 + 12 months → 31.03.2009), notice in months × 30, and each name in a party loses one leading title and trailing punctuation. This step only touches the model's answers, never the document. In early versions, moving the arithmetic out of the model took test recall from 54% to 79%.

### 4. Choosing the OCR and language models

I compared current small OCR and language models on the five scans and the training documents. The test set was only scored, never used to choose. These numbers are on the labels as given, where 91.7% is the most anything could score on test:

| OCR + language model | Test | Train (7 usable docs) |
|---|---:|---:|
| Tesseract + Qwen3-4B-Instruct (starting point) | 79.2% | 64.3% |
| OvisOCR3 + Qwen3-4B-Instruct | 87.5% | 64.3% |
| GLM-OCR + Qwen3-4B-Instruct | 87.5% | 66.7% |
| Tesseract + Qwen3.5-4B | 83.3% | 71.4% |
| GLM-OCR + Qwen3.5-4B | 87.5% | 71.4% |
| **GLM-OCR + Qwen3.5-4B, final prompt and name formatting** | **91.7%** | **71.4%** |

- **OCR.** Tesseract drops punctuation inside names ("V.V.Ravi Kian" → "V.V Ravi Kian"), which exact matching punishes. OvisOCR3 fixed that but misread "Kartheek" as "Karthreek". [GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) (0.9B) read every name, date and amount correctly. Scans are split into pages at the whitest row near each page boundary and scaled to at most 1275 px wide, which gives the same text as full resolution in a third of the time and memory.
- **Language model.** [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) beat Qwen3-4B-Instruct on the training documents, mostly on names. It first named a company's signatory as the tenant, so the prompt now asks for the company itself.
- **Reading page images directly.** Giving Qwen3.5-4B the page images instead of OCR text ran out of memory on three-page scans on an 8 GB GPU.
- **LFM3-VL-2B.** Liquid's new 2B vision model is small enough to read the scans itself, which would remove the OCR step. It reads the text fine, but it is much weaker at the extraction: it copies dates from the example agreements, swaps owner and tenant, and adds parentage to names. Its best setup scored 79.2% on test against 100% for this pipeline.

### 5. Closing the last gaps

With the labels fixed, two training misses were genuinely the model's, both on 18325926. It kept the father's name ("K.Kuttan S/o Kelu Aehari (Late)"), and it worked out an end date itself (30.11.2009) where the agreement only says "eleven months". The prompt already had rules for both, so I made them explicit: an end date is copied only when the document writes one, and the name rule lists the relation markers (S/o, D/o, W/o, ...) with one short example. Both were fixed, and test stayed at 100%. I also made the title clean-up apply to every name in a joint party ("Asha Ramesh & Mr.Ramesh.K.N" → "Asha Ramesh & Ramesh.K.N").

That leaves the four fields of the garbled Word file, which no reader can recover.

## Final pipeline

```
.docx ──python-docx──┐
                     ├─► text ─► Qwen3.5-4B (2 worked examples) ─► facts as JSON ─► arithmetic ─► 6 fields
.png / .jpg ─GLM-OCR─┘
```

1. **Text.** Word files are read with python-docx (paragraphs and tables, in order). Scans are split into pages and read by GLM-OCR.
2. **Reading.** Qwen3.5-4B (4-bit) returns the stated facts as JSON, with one retry if the JSON is invalid.
3. **Arithmetic and formatting.** End date, notice in days, and name clean-up, as described above.

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

Reproduce all the scores above (language model and QA baseline, train and test):

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

- **Few examples.** There are only 13 labelled documents, two of them near-duplicates, so the scores are a rough guide.
- **Typed scans only.** All the scans are clean typed pages. Handwriting or photos of documents haven't been tested.
- **Speed.** OCR dominates on scans. A llama.cpp or vLLM server would be faster than plain transformers but is harder to set up on Windows.
