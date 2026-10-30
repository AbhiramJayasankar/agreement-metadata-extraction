Strict recall

| Method / split | Value | Start | End | Notice | Party One | Party Two | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| LLM test | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| LLM train | 88.9% | 77.8% | 77.8% | 100.0% | 88.9% | 100.0% | 88.9% |
| LLM train readable | 100.0% | 87.5% | 87.5% | 100.0% | 100.0% | 100.0% | 95.8% |
| QA test | 75.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 12.5% |
| QA train readable | 75.0% | 12.5% | 12.5% | 0.0% | 0.0% | 0.0% | 16.7% |

Lenient recall

| Method / split | Value | Start | End | Notice | Party One | Party Two | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| LLM test | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| LLM train | 88.9% | 77.8% | 77.8% | 100.0% | 100.0% | 100.0% | 90.7% |
| LLM train readable | 100.0% | 87.5% | 87.5% | 100.0% | 100.0% | 100.0% | 95.8% |
| QA test | 75.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 12.5% |
| QA train readable | 75.0% | 12.5% | 12.5% | 0.0% | 0.0% | 0.0% | 16.7% |

- Strict recall strips leading/trailing spaces; lenient recall also ignores case, spaces and punctuation. Empty answers match empty labels.
- Train has 9 rows (unlabelled 46239065 is not scored); train readable leaves out 44737744, whose Word file is unreadable OCR text. Test has all 4.
- Labels were corrected where they contradict the document; README.md lists every change. Train: swapped rows 54770958/54945838 put back, 24158401 removed (its file is in test/), impossible end dates fixed, names copied as the documents write them.
- test.csv: 228094620 Party Two (.B.Kishore -> B.Kishore) and 95980236 end date (31.03.2011 -> 28.02.2011).
- The prompt holds two training documents as examples; a training document is never its own example.
- QA answers are spans copied from the text, so it cannot produce computed end dates or notice in days.
- LLM Qwen/Qwen3.5-4B (nf4, GPU): mean 9.51 seconds/document, excluding model loading.
- Text preparation (glm OCR and DOCX, including cache reads): 4.39 seconds, before LLM loading.
