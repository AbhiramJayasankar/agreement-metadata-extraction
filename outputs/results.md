Strict recall

| Method / split | Value | Start | End | Notice | Party One | Party Two | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| LLM test | 100.0% | 100.0% | 75.0% | 75.0% | 100.0% | 25.0% | 79.2% |
| LLM train usable | 85.7% | 85.7% | 42.9% | 85.7% | 28.6% | 57.1% | 64.3% |
| LLM train as provided | 60.0% | 60.0% | 30.0% | 60.0% | 20.0% | 40.0% | 45.0% |
| QA test | 75.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 12.5% |
| QA train usable | 57.1% | 14.3% | 14.3% | 14.3% | 0.0% | 0.0% | 16.7% |

Lenient recall

| Method / split | Value | Start | End | Notice | Party One | Party Two | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| LLM test | 100.0% | 100.0% | 75.0% | 75.0% | 100.0% | 75.0% | 87.5% |
| LLM train usable | 85.7% | 85.7% | 42.9% | 85.7% | 71.4% | 57.1% | 71.4% |
| LLM train as provided | 60.0% | 60.0% | 30.0% | 60.0% | 50.0% | 40.0% | 50.0% |
| QA test | 75.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 12.5% |
| QA train usable | 57.1% | 14.3% | 14.3% | 14.3% | 0.0% | 0.0% | 16.7% |

- Strict recall strips leading/trailing spaces; lenient recall also ignores case, spaces and punctuation. Empty answers match empty labels.
- Train usable has 7 rows: 24158401 is missing; 54770958 and 54945838 have swapped labels. Unlabeled 46239065 is excluded from both views. Train as provided has all 10 CSV rows; test has all 4.
- Labels are scored as given, including impossible dates such as 31.02.2011.
- The prompt holds two training documents as examples; a training document is never its own example.
- QA answers are spans copied from the text, so it cannot produce computed end dates or notice in days.
- LLM Qwen/Qwen3-4B-Instruct-2507 (nf4, GPU): mean 11.81 seconds/document, excluding model loading.
