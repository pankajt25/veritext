# Veritext Benchmark Results

- **Evaluated Documents**: 104 (from synthetic dataset)
- **Total Runtime**: 641.61 s (~6.1694 s/doc)
- **Peak Memory**: 377.52 MB
- **Date**: 2026-10-23 11:06:33 UTC

## 1. Plagiarism Detection Engine

| Method | Accuracy | Precision | Recall | F1 Score | False Positive Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Exact Winnowing (Veritext)** | 66.67% | 100.00% | 50.00% | 66.67% | 0.00% |
| Baseline (difflib SequenceMatcher) | 66.67% | 100.00% | 50.00% | 66.67% | 0.00% |
| Winnowing + Semantic Paraphrase | 66.67% | 100.00% | 50.00% | 66.67% | 0.00% |

**Mean Similarity Score by Cohort**:
- Group A (Verbatim / Lightly Edited Copies): **98.28%**
- Group B (Paraphrased Passages): **0.85%**
- Group C (Unrelated Human Originals): **0.00%**

## 2. AI-Written Text Detector

| Model / Configuration | Accuracy | Precision | Recall | F1 Score | Overall Human FPR | Non-Native (ESL) FPR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stylometry Only (Lite Mode)** | 77.50% | 100.00% | 40.00% | 57.14% | 0.00% | 0.00% |
| DistilGPT2 Features Only | 62.50% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| **Full Ensemble (Stylometry + DistilGPT2)** | 95.00% | 100.00% | 86.67% | 92.86% | 0.00% | 0.00% |

## 3. Integrity Checks, Collusion & Cryptographic Seals

| Module | Metric Tested | Result | Target / Standard |
| :--- | :--- | :---: | :---: |
| **Spinner / Tortured Phrases** | Precision / Recall | 100.00% / 100.00% | Cabanac et al. (2021) |
| **Batch Collusion Matrix** | Expected Clusters Discovered | YES | Pairwise Winnowing |
| **MinHash / LSH Screening** | True Pairs Candidate Recall | 50.00% | Carter-Wegman 128-perm |
| **Cryptographic Report Seals** | Tamper Detection Rate | 100.00% | HMAC-SHA256 (1-char edit) |

## 4. Notable Failure Cases & Honest Limitations

- **Dataset Characteristics**: The evaluation benchmark is small and synthetic (110 manifest items, 104 evaluated documents), engineered for reproducible offline execution on dual-core hardware without network access.
- **Plagiarism Paraphrase Recall**: Exact winnowing ($k=5$) detects 100% of verbatim/obfuscated copies (Group A, 98.28% mean similarity), but achieves 0.00% recall on synthetic paraphrases (Group B, 0.85% mean similarity). While dense neural embeddings capture paraphrasing on individual sentence windows (e.g. 33.1% on `paraphrased_doc.txt`), heavily reworded synthetic documents in Group B did not surpass the 15% document-level character overlap threshold (15 missed paraphrases).
- **AI Detection Ablation**: Stylometry alone achieves 40.00% recall (6/15). DistilGPT2 features alone fail to trigger on short texts (0.00% recall). The full ensemble combines both signals to achieve 86.67% recall (13/15) with 95.00% overall accuracy. Two AI texts fell below the classification threshold.
- **ESL / Non-Native Protection**: Across all configurations, the False Positive Rate on non-native English essays (Group E) is **0.00%** (0/10), validating conservative calibration safeguards.
