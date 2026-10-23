# Veritext Benchmark Results

- **Evaluated Documents**: 114 (from synthetic dataset)
- **Total Runtime**: 803.08 s (~7.0446 s/doc)
- **Peak Memory**: 377.64 MB
- **Date**: 2026-10-23 17:54:22 UTC

> **Corpus & Methodology Note**: The evaluation corpus is synthetic, compact, and created by the system author.
> In accordance with the Stage 7 audit guidelines, results are evaluated across distinct easy and hard paraphrase subsets,
> and false positive rates are transparently measured across unrelated human originals, AI-generated text, and ESL writing.

## 1. Plagiarism Detection Engine (Headline Threshold: 15%)

| Method | Accuracy | Precision | Recall | F1 Score | False Positive Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Exact Winnowing (Veritext)** | 77.27% | 100.00% | 37.50% | 54.55% | 0.00% |
| Baseline (difflib SequenceMatcher) | 77.27% | 100.00% | 37.50% | 54.55% | 0.00% |
| **Winnowing + Semantic Paraphrase** | 79.09% | 65.45% | 90.00% | 75.79% | 27.14% |

### Decision Threshold Sensitivity Analysis

| Threshold | Method | Accuracy | Precision | Recall | F1 Score | Overall FPR |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 5% | Exact Winnowing | 77.27% | 94.12% | 40.00% | 56.14% | 1.43% |
| 5% | **Winnowing + Semantic** | 79.09% | 65.45% | 90.00% | 75.79% | 27.14% |
| 10% | Exact Winnowing | 78.18% | 100.00% | 40.00% | 57.14% | 0.00% |
| 10% | **Winnowing + Semantic** | 79.09% | 65.45% | 90.00% | 75.79% | 27.14% |
| 15% | Exact Winnowing | 77.27% | 100.00% | 37.50% | 54.55% | 0.00% |
| 15% | **Winnowing + Semantic** | 79.09% | 65.45% | 90.00% | 75.79% | 27.14% |
| 25% | Exact Winnowing | 77.27% | 100.00% | 37.50% | 54.55% | 0.00% |
| 25% | **Winnowing + Semantic** | 79.09% | 65.45% | 90.00% | 75.79% | 27.14% |

### Paraphrase Recall: Easy vs. Hard Subsets

| Threshold | Group A (Copies) | Group B Easy Paraphrases | Group B Hard Paraphrases (Complex) |
| :---: | :---: | :---: | :---: |
| 5% | 100.00% | 93.33% | 70.00% |
| 10% | 100.00% | 93.33% | 70.00% |
| 15% | 100.00% | 93.33% | 70.00% |
| 25% | 100.00% | 93.33% | 70.00% |

### False Positive Rates across Negative Cohorts (Semantic Engine ON)

| Threshold | Group C (Originals) | Group D (AI Text) | Group E (ESL Non-Native) | Group I (Same-Topic Originals) | Group J (Common Knowledge) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 5% | 0.00% | 0.00% | 0.00% | 95.00% | 0.00% |
| 10% | 0.00% | 0.00% | 0.00% | 95.00% | 0.00% |
| 15% | 0.00% | 0.00% | 0.00% | 95.00% | 0.00% |
| 25% | 0.00% | 0.00% | 0.00% | 95.00% | 0.00% |

**Mean Similarity Score by Cohort (Exact vs. Semantic)**:
- Group A (Verbatim / Lightly Edited Copies): Exact **98.28%** | Semantic **98.28%**
- Group B (Easy Paraphrased Passages): Exact **0.85%** | Semantic **68.44%**
- Group B (Hard Complex Paraphrases): Exact **0.00%** | Semantic **43.74%**
- Group C (Unrelated Human Originals): Exact **0.00%** | Semantic **0.00%**
- Group D (AI-Generated Text): Exact **0.00%** | Semantic **0.00%**
- Group E (Non-Native English): Exact **0.00%** | Semantic **0.00%**
- Group I (Same-Topic Originals): Exact **0.51%** | Semantic **65.85%**
- Group J (Common Knowledge Definitions): Exact **0.00%** | Semantic **0.00%**

## 2. AI-Written Text Detector

| Model / Configuration | Accuracy | Precision | Recall | F1 Score | Overall Human FPR | Non-Native (ESL) FPR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stylometry Only (Lite Mode)** | 77.50% | 100.00% | 40.00% | 57.14% | 0.00% | 0.00% |
| **Full Ensemble (Stylometry + DistilGPT2)** | 95.00% | 100.00% | 86.67% | 92.86% | 0.00% | 0.00% |

## 3. Integrity Checks, Collusion & Cryptographic Seals

| Module | Metric Tested | Result | Target / Standard |
| :--- | :--- | :---: | :---: |
| **Spinner / Tortured Phrases** | Precision / Recall | 100.00% / 100.00% | Cabanac et al. (2021) |
| **Batch Collusion Matrix** | Expected Clusters Discovered | YES | Pairwise Winnowing |
| **Fingerprint Index Screening** | True Pairs Candidate Recall | 100.00% | Exact Inverted Index (Default) |
| **MinHash / LSH Screening** | True Pairs Candidate Recall | 100.00% | Adaptive (64x2) 128-perm |
| **Cryptographic Report Seals** | Tamper Detection Rate | 100.00% | HMAC-SHA256 (1-char edit) |

## 4. Notable Failure Cases & Honest Limitations

- **Plagiarism Missed Paraphrases (Easy)**: 1 documents.
- **Plagiarism Missed Paraphrases (Hard)**: 3 documents.
- **AI False Positives on Human Text**: 0 documents.
- **AI False Positives on Non-Native (ESL) Text**: 0 documents. (Stylometric variance can mistake uniform ESL syntax for machine generation; conservative thresholds are essential).
