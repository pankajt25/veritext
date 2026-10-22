# Veritext Benchmark Results

- **Evaluated Documents**: 104 (from synthetic dataset)
- **Total Runtime**: 162.21 s (~1.5597 s/doc)
- **Peak Memory**: 253.33 MB
- **Date**: 2026-10-22 14:58:18 UTC

## 1. Plagiarism Detection Engine

| Method | Accuracy | Precision | Recall | F1 Score | False Positive Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Exact Winnowing (Veritext)** | 66.67% | 100.00% | 50.00% | 66.67% | 0.00% |
| Baseline (difflib SequenceMatcher) | 66.67% | 100.00% | 50.00% | 66.67% | 0.00% |

**Mean Similarity Score by Cohort**:
- Group A (Verbatim / Lightly Edited Copies): **98.28%**
- Group B (Paraphrased Passages): **0.85%**
- Group C (Unrelated Human Originals): **0.00%**

## 2. AI-Written Text Detector

| Model / Configuration | Accuracy | Precision | Recall | F1 Score | Overall Human FPR | Non-Native (ESL) FPR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stylometry Only (Lite Mode)** | 77.50% | 100.00% | 40.00% | 57.14% | 0.00% | 0.00% |

## 3. Integrity Checks, Collusion & Cryptographic Seals

| Module | Metric Tested | Result | Target / Standard |
| :--- | :--- | :---: | :---: |
| **Spinner / Tortured Phrases** | Precision / Recall | 100.00% / 100.00% | Cabanac et al. (2021) |
| **Batch Collusion Matrix** | Expected Clusters Discovered | YES | Pairwise Winnowing |
| **MinHash / LSH Screening** | True Pairs Candidate Recall | 50.00% | Carter-Wegman 128-perm |
| **Cryptographic Report Seals** | Tamper Detection Rate | 100.00% | HMAC-SHA256 (1-char edit) |

## 4. Notable Failure Cases & Honest Limitations

- **Plagiarism Missed Paraphrases**: 15 documents. (Exact winnowing with k=5 relies on identical token n-grams; heavily rephrased text with 0 shared 5-grams requires semantic vector matching).
- **AI False Positives on Human Text**: 0 documents.
- **AI False Positives on Non-Native (ESL) Text**: 0 documents. (Stylometric variance can mistake uniform ESL syntax for machine generation; conservative thresholds are essential).
