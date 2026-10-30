Strict recall

| Method / split | Value | Start | End | Notice | Party One | Party Two | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| LLM test | 100.0% | 100.0% | 75.0% | 100.0% | 100.0% | 75.0% | 91.7% |
| LLM test, labels corrected | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| LLM train usable | 85.7% | 85.7% | 42.9% | 100.0% | 42.9% | 71.4% | 71.4% |
| LLM train as provided | 60.0% | 60.0% | 30.0% | 70.0% | 30.0% | 50.0% | 50.0% |
| QA test | 75.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 12.5% |
| QA train usable | 57.1% | 14.3% | 14.3% | 14.3% | 0.0% | 0.0% | 16.7% |

Lenient recall

| Method / split | Value | Start | End | Notice | Party One | Party Two | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| LLM test | 100.0% | 100.0% | 75.0% | 100.0% | 100.0% | 100.0% | 95.8% |
| LLM test, labels corrected | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| LLM train usable | 85.7% | 85.7% | 42.9% | 100.0% | 85.7% | 71.4% | 78.6% |
| LLM train as provided | 60.0% | 60.0% | 30.0% | 70.0% | 60.0% | 50.0% | 55.0% |
| QA test | 75.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 12.5% |
| QA train usable | 57.1% | 14.3% | 14.3% | 14.3% | 0.0% | 0.0% | 16.7% |

- Strict recall strips leading/trailing spaces; lenient recall also ignores case, spaces and punctuation. Empty answers match empty labels.
- Train usable has 7 rows: 24158401 is missing; 54770958 and 54945838 have swapped labels. Unlabeled 46239065 is excluded from both views. Train as provided has all 10 CSV rows; test has all 4.
- Except for the corrected test view, labels are scored as given, including impossible dates such as 31.02.2011.
- Corrected test labels: 228094620-Rental-Agreement, Party Two: B.Kishore. Document reads "Mr.B.Kishore"; the label kept the dot after removing "Mr". 95980236-Rental-Agreement, Agreement End Date: 28.02.2011. Document states 11 months commencing 1st April 2010.
- The prompt holds two training documents as examples; a training document is never its own example.
- QA answers are spans copied from the text, so it cannot produce computed end dates or notice in days.
- LLM Qwen/Qwen3.5-4B (nf4, GPU): mean 9.07 seconds/document, excluding model loading.
- Text preparation (glm OCR and DOCX, including cache reads): 4.14 seconds, before LLM loading.
