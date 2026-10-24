# Veritext Benchmark Results

- **Evaluated Documents**: 114 (from synthetic dataset)
- **Total Runtime**: 761.05 s (~6.6759 s/doc)
- **Peak Memory**: 377.68 MB
- **Date**: 2026-10-24 21:36:45 UTC

> **Corpus & Methodology Note**: The evaluation corpus is synthetic, compact, and created by the system author.
> In accordance with the Stage 7 audit guidelines, results are evaluated across distinct easy and hard paraphrase subsets,
> and false positive rates are transparently measured across unrelated human originals, AI-generated text, and ESL writing.
> Tuning and evaluation use the same small synthetic corpus, so reported numbers are inherently optimistic.

## 1. Plagiarism Detection Engine (Headline Threshold: 15%)

> **Headline Scoring Rule (v1.0.3)**: Headline similarity percentage is computed ONLY from exact and near-copy matches.
> Semantic vector matches are strictly excluded by default and presented separately as unverified paraphrase review leads.

| Method | Accuracy | Precision | Recall | F1 Score | False Positive Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Headline Score / Exact Winnowing (Veritext Default)** | 77.27% | 100.00% | 37.50% | 54.55% | 0.00% |
| Baseline (difflib SequenceMatcher) | 77.27% | 100.00% | 37.50% | 54.55% | 0.00% |
| *Winnowing + Semantic (Opt-In: `--include-semantic-in-score`)* | 79.09% | 65.45% | 90.00% | 75.79% | 27.14% |

### Decision Threshold Sensitivity Analysis

| Threshold | Method | Accuracy | Precision | Recall | F1 Score | Overall FPR |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 5% | **Headline Score (Default)** | 77.27% | 94.12% | 40.00% | 56.14% | 1.43% |
| 5% | *Winnowing + Semantic (Opt-In)* | 79.09% | 65.45% | 90.00% | 75.79% | 27.14% |
| 10% | **Headline Score (Default)** | 78.18% | 100.00% | 40.00% | 57.14% | 0.00% |
| 10% | *Winnowing + Semantic (Opt-In)* | 79.09% | 65.45% | 90.00% | 75.79% | 27.14% |
| 15% | **Headline Score (Default)** | 77.27% | 100.00% | 37.50% | 54.55% | 0.00% |
| 15% | *Winnowing + Semantic (Opt-In)* | 79.09% | 65.45% | 90.00% | 75.79% | 27.14% |
| 25% | **Headline Score (Default)** | 77.27% | 100.00% | 37.50% | 54.55% | 0.00% |
| 25% | *Winnowing + Semantic (Opt-In)* | 79.09% | 65.45% | 90.00% | 75.79% | 27.14% |

### False Positive Rates across Negative Cohorts (Headline Engine Default)

| Threshold | Group C (Originals) | Group D (AI Text) | Group E (ESL Non-Native) | Group I (Same-Topic Originals) | Group J (Common Knowledge) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 5% | 0.00% | 0.00% | 0.00% | 5.00% | 0.00% |
| 10% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| 15% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| 25% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |

### False Positive Rates across Negative Cohorts (Opt-In Engine: Semantic in Score)

| Threshold | Group C (Originals) | Group D (AI Text) | Group E (ESL Non-Native) | Group I (Same-Topic Originals) | Group J (Common Knowledge) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 5% | 0.00% | 0.00% | 0.00% | 95.00% | 0.00% |
| 10% | 0.00% | 0.00% | 0.00% | 95.00% | 0.00% |
| 15% | 0.00% | 0.00% | 0.00% | 95.00% | 0.00% |
| 25% | 0.00% | 0.00% | 0.00% | 95.00% | 0.00% |

## 1b. Paraphrase Leads & Lexical Anchors Evaluation

> **Explanatory Note**: Similar meaning is not proof of copying; passages on the same topic often match. Check the linked source.
> **Interpretation Guide**: Strong lead: several matching sentences and shared specific terms; weak lead: similar topic only.

### Lead Statistics Across Cohorts

| Cohort | Mean Leads / Doc | Mean Strong Leads / Doc | Mean Weak Leads / Doc | Flagged Rate |
| :--- | :---: | :---: | :---: | :---: |
| Group A (Copies) | 0.00 | 0.00 | 0.00 | 0.00% |
| Group B (Easy Paraphrases) | 3.07 | 3.07 | 0.00 | 93.33% |
| Group B (Hard Complex Paraphrases) | 2.20 | 2.20 | 0.00 | 80.00% |
| Group C (Unrelated Originals) | 0.00 | 0.00 | 0.00 | 0.00% |
| Group D (AI Generated) | 0.00 | 0.00 | 0.00 | 0.00% |
| Group E (ESL Non-Native) | 0.00 | 0.00 | 0.00 | 0.00% |
| Group I (Same-Topic Originals) | 2.70 | 2.70 | 0.00 | 95.00% |
| Group J (Common Knowledge) | 0.00 | 0.00 | 0.00 | 0.00% |

- **Strong-Lead Precision (Group B vs Group I)**: **55.74%**

### Ordering Metric (ROC AUC) on Paraphrase vs. Same-Topic Spans

| Ordering Metric | ROC AUC | Finding |
| :--- | :---: | :--- |
| **Cosine Alone** | **0.5340** | Near-random (~0.50); cosine overlaps between paraphrases and same-topic text |
| **Anchor Count Alone** | **0.3622** | Inverted discrimination; technical vocabulary is dense in originals |
| **Cosine + Anchors Composite** | **0.4939** | Negligible improvement; anchors do not cleanly separate domains |

### Lexical Anchor Rule Grid Search (N in {1,2,3,4}, ratio in {0.1, 0.2, 0.3})

| Min Anchors (N) | Min Content Ratio | Group B Recall (All) | Group B Recall (Easy) | Group B Recall (Hard) | Group I FPR | Group J FPR | Rule Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | 0.1 | 88.00% | 93.33% | 80.00% | 95.00% | 0.00% | Does not cut FPR by half |
| 1 | 0.2 | 88.00% | 93.33% | 80.00% | 95.00% | 0.00% | Does not cut FPR by half |
| 1 | 0.3 | 72.00% | 93.33% | 40.00% | 90.00% | 0.00% | Does not cut FPR by half |
| 2 | 0.1 | 88.00% | 93.33% | 80.00% | 95.00% | 0.00% | Does not cut FPR by half |
| 2 | 0.2 | 88.00% | 93.33% | 80.00% | 95.00% | 0.00% | Does not cut FPR by half |
| 2 | 0.3 | 72.00% | 93.33% | 40.00% | 90.00% | 0.00% | Does not cut FPR by half |
| 3 | 0.1 | 88.00% | 93.33% | 80.00% | 95.00% | 0.00% | Does not cut FPR by half |
| 3 | 0.2 | 88.00% | 93.33% | 80.00% | 95.00% | 0.00% | Does not cut FPR by half |
| 3 | 0.3 | 72.00% | 93.33% | 40.00% | 90.00% | 0.00% | Does not cut FPR by half |
| 4 | 0.1 | 88.00% | 93.33% | 80.00% | 95.00% | 0.00% | Does not cut FPR by half |
| 4 | 0.2 | 88.00% | 93.33% | 80.00% | 95.00% | 0.00% | Does not cut FPR by half |
| 4 | 0.3 | 72.00% | 93.33% | 40.00% | 90.00% | 0.00% | Does not cut FPR by half |

> **Honest Technical Conclusion**: Because no parameter combination achieved Group B recall >= 70% while reducing Group I FPR by half (to <= 47.5%), lexical anchors cannot reliably separate same-topic technical writing from true paraphrases using this model (`all-MiniLM-L6-v2`) on this synthetic corpus. Semantic matching is kept strictly out of the default headline score. Matches meeting anchor criteria are tagged as 'strong lead' and others as 'weak lead' for human review only.

**Mean Similarity Score by Cohort (Exact/Headline vs. Opt-In Semantic)**:
- Group A (Verbatim / Lightly Edited Copies): Exact/Headline **98.28%** | Opt-In Semantic **98.28%**
- Group B (Easy Paraphrased Passages): Exact/Headline **0.85%** | Opt-In Semantic **68.44%**
- Group B (Hard Complex Paraphrases): Exact/Headline **0.00%** | Opt-In Semantic **43.74%**
- Group C (Unrelated Human Originals): Exact/Headline **0.00%** | Opt-In Semantic **0.00%**
- Group D (AI-Generated Text): Exact/Headline **0.00%** | Opt-In Semantic **0.00%**
- Group E (Non-Native English): Exact/Headline **0.00%** | Opt-In Semantic **0.00%**
- Group I (Same-Topic Originals): Exact/Headline **0.51%** | Opt-In Semantic **65.85%**
- Group J (Common Knowledge Definitions): Exact/Headline **0.00%** | Opt-In Semantic **0.00%**

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

- **Plagiarism Missed Paraphrases (Easy)**: 15 documents.
- **Plagiarism Missed Paraphrases (Hard)**: 10 documents.
- **AI False Positives on Human Text**: 0 documents.
- **AI False Positives on Non-Native (ESL) Text**: 0 documents. (Stylometric variance can mistake uniform ESL syntax for machine generation; conservative thresholds are essential).
