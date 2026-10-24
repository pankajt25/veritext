# Veritext: Final Engineering & Project Report

## Executive Summary
**Veritext** is a production-grade, open-source academic integrity platform designed to provide educators, researchers, and students with transparent, verifiable textual analysis. Unlike closed commercial alternatives that require ongoing subscriptions and harvest student intellectual property to train proprietary models, Veritext executes entirely on local infrastructure, respects user privacy, and guarantees algorithmic inspectability.

Over seven engineering stages, Veritext evolved from a foundational parser into a full-featured integrity suite encompassing winnowing fingerprinting, locality-sensitive hashing (LSH), calibrated stylometric and neural AI detection, text-spinning forensics, institutional class repositories, cryptographic report seals, and a real-time REST API.

---

## 1. Architectural Stages & Implementation Summary

### Stage 1: Document Ingestion, Parsing & Text Cleaning
- **Multi-Format Loaders**: Seamless ingestion of PDF (`pdfplumber` with fallback to `pypdfium2` and `rapidocr`), Microsoft Word (`python-docx`), plain text, LaTeX (`pylatexenc`), and Markdown.
- **Structural Decomposition**: Dual-level segmentation into paragraphs and sentences with offset tracking (`Sentence.start`, `Sentence.end`) cached via NLTK tokenizers.
- **Academic Content Filtering**: Automated detection and stripping of reference bibliographies, block quotations, mathematical formulas, and computer code blocks to prevent spurious match flags on standardized apparatus.

### Stage 2: Normalization & Adversarial Obfuscation Detection
- **Adversarial Resiliency**: Identifies deliberate evasion tactics including Unicode Cyrillic/Greek homoglyphic character substitutions and invisible zero-width spaces (`\u200b`, `\u200c`, `\u200d`, `\ufeff`).
- **Canonical Representation**: Preserves an exact character offset map (`offset_map`) mapping canonical clean tokens back to raw source character coordinates.

### Stage 3: Fingerprinting, Winnowing & Locality-Sensitive Hashing
- **Winnowing Algorithm**: Implements the algorithm developed by **Schleimer, Wilkerson & Aiken (2003)**. Uses rolling 5-gram hashes filtered through sliding windows ($w=4$) to guarantee detection of shared substrings of length $\ge 8$ tokens while bounding index storage.
- **Fingerprint Inverted Index & Adaptive MinHash/LSH**: Batch and class screening default to an exact inverted index of shared winnowing fingerprints, guaranteeing 100% candidate recall. For large batches, 128-permutation Carter-Wegman MinHash ($(a \cdot x + b) \pmod{4294967311}$) is available, with bands $b$ and rows $r$ derived automatically from the user threshold $t$ ($s^* = (1/b)^{1/r} \approx t/2$, subject to $b \times r \le 128$) to favour recall. Small cohorts ($N \le 150$) automatically bypass screening to evaluate exact pairwise winnowing.

### Stage 4: Candidate Source Retrieval & Reference Storage
- **Local Reference Store**: High-throughput SQLite library indexing local textbook and journal collections.
- **Open-Access Academic APIs**: Extracts salient TF-IDF keywords and queries OpenAlex, arXiv, and CrossRef for open-access candidate comparison texts.

### Stage 5: Hybrid Similarity Scoring, Semantic Leads & Cross-Language Matching
- **Match Span Clustering**: Contiguous match interval chaining with gap tolerance.
- **Headline Scoring Rule (v1.0.3)**: Headline similarity percentage is computed **strictly** from exact and near-copy matches. Dense semantic vector matches are excluded by default to eliminate domain false positives.
- **Paraphrase Review Leads & Lexical Anchors**: Dense embedding search via `all-MiniLM-L6-v2` identifies candidate paraphrase spans, categorized as "strong lead" or "weak lead" based on shared rare content lemmas, numbers, and named entities. Displayed separately as exploratory leads for instructor review.
- **Legacy Opt-In Mode**: `--include-semantic-in-score` flag allows users to restore the legacy additive behavior, with an explicit score label (`"score includes unverified semantic leads"`).
- **Multilingual Support**: Optional cross-language semantic matching across Spanish, French, and German translations.

### Stage 6: Explainable AI-Written Text Detection
- **Multi-Feature Ensemble**: Weighted logistic regression combining stylometric features (Type-Token Ratio, sentence length variance, discourse marker density, repeated n-gram rate) with DistilGPT2 perplexity and mean log-rank.
- **Confidence Calibration**: Outputs transparent confidence bands and plain-language driver notes (e.g., *"Elevated frequency of formulaic transition phrases"*) to eliminate black-box score interpretation.
- **Protection for Non-Native (ESL) Writers**: Explicit tuning to maintain low false-positive rates on uniform human syntax.

### Stage 6B-6D: Forensic Integrity Checks, Batch Collusion & Sealed Reports
- **Tortured-Phrase Forensics**: Detects odd synonym substitutions (e.g., *"counterfeit consciousness"* for artificial intelligence) based on the seminal research of **Cabanac, Labbe & Magazinov (2021)**.
- **Authorship Timeline**: Analyzes sentence-level stylometry shifts across long documents to identify potential multi-author compilation.
- **Batch Collusion Matrix**: Pairwise similarity heatmaps, connected-component clustering, and shared passage extraction across entire student cohorts.
- **Cryptographic Report Seals**: Anchors exported reports with canonical JSON serialization and HMAC-SHA256 signatures, ensuring 100% detection of unauthorized post-generation tampering.
- **Institutional Class Repository**: Privacy-preserving inverted index for detecting cross-cohort collusion across multiple academic terms.
- **REST API with Server-Sent Events (SSE)**: Asynchronous FastAPI server streaming real-time stage progress to client applications.

### Stage 7: Performance Tuning, Packaging, Deployment & Evaluation
- Profiled and resolved top bottlenecks, achieving a 4.5x fast test suite speedup and 3x CLI speedup on 2-core machines.
- Created complete benchmark suite on a 110-document synthetic evaluation dataset.
- Packaged production distribution with console script `veritext`, Dockerfile, DEPLOY.md, CITATION.cff, and Ruff linting.

---

## 2. Evaluation & Benchmark Results

Veritext was benchmarked on a synthetic dataset of 114 evaluated documents executed entirely offline on dual-core hardware (total runtime: 761.05 s, peak memory: 377.68 MB).

> **Methodology and Dataset Note**: The evaluation corpus is small, synthetic, and written by the same author as the detector. It serves to verify algorithmic guarantees, component wiring, and relative ranking rather than open-domain generalization. Results are evaluated across distinct easy and hard paraphrase subsets, and false positive rates are transparently measured across unrelated human originals, AI-generated text, ESL writing, same-topic originals, and common-knowledge standard definitions. Tuning and evaluation use the same small synthetic corpus, so reported numbers are inherently optimistic. Veritext makes no claim of parity with commercial services with proprietary multi-billion document databases.

### Plagiarism Detection Engine Ablation (Headline Threshold: 15%)

> **Headline Scoring Rule (v1.0.3)**: Headline similarity percentage is computed **ONLY** from exact and near-copy matches. Semantic vector matches are strictly excluded by default to eliminate false positives on technical terminology and are presented separately as unverified paraphrase review leads.

| Method / Configuration | Accuracy | Precision | Recall | F1 Score | False Positive Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Headline Score / Exact Winnowing (Veritext Default)** | **77.27%** | **100.00%** | **37.50%** | **54.55%** | **0.00%** |
| Baseline (difflib SequenceMatcher) | 77.27% | 100.00% | 37.50% | 54.55% | 0.00% |
| *Winnowing + Semantic (Opt-In: `--include-semantic-in-score`)* | 79.09% | 65.45% | 90.00% | 75.79% | 27.14% |

#### Decision Threshold Sensitivity Analysis

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

#### False Positive Rates across Negative Cohorts (Headline Engine Default)

| Threshold | Group C (Originals) | Group D (AI Text) | Group E (ESL Non-Native) | Group I (Same-Topic Originals) | Group J (Common Knowledge) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 5% | 0.00% | 0.00% | 0.00% | 5.00% | 0.00% |
| 10% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| 15% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| 25% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |

#### False Positive Rates across Negative Cohorts (Opt-In Engine: Semantic in Score)

| Threshold | Group C (Originals) | Group D (AI Text) | Group E (ESL Non-Native) | Group I (Same-Topic Originals) | Group J (Common Knowledge) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 5% | 0.00% | 0.00% | 0.00% | 95.00% | 0.00% |
| 10% | 0.00% | 0.00% | 0.00% | 95.00% | 0.00% |
| 15% | 0.00% | 0.00% | 0.00% | 95.00% | 0.00% |
| 25% | 0.00% | 0.00% | 0.00% | 95.00% | 0.00% |

### Paraphrase Leads & Lexical Anchors Evaluation

> **Explanatory Note**: *Similar meaning is not proof of copying; passages on the same topic often match. Check the linked source.*  
> **Interpretation Guide**: *Strong lead: several matching sentences and shared specific terms; weak lead: similar topic only.*

#### Lead Statistics Across Cohorts

| Cohort | Mean Leads / Doc | Mean Strong Leads / Doc | Mean Weak Leads / Doc | Flagged Rate |
| :--- | :---: | :---: | :---: | :---: |
| Group A (Verbatim Copies) | 0.00 | 0.00 | 0.00 | 0.00% |
| Group B (Easy Paraphrases) | 3.07 | 3.07 | 0.00 | 93.33% |
| Group B (Hard Complex Paraphrases) | 2.20 | 2.20 | 0.00 | 80.00% |
| Group C (Unrelated Originals) | 0.00 | 0.00 | 0.00 | 0.00% |
| Group D (AI Generated Text) | 0.00 | 0.00 | 0.00 | 0.00% |
| Group E (ESL Non-Native) | 0.00 | 0.00 | 0.00 | 0.00% |
| Group I (Same-Topic Originals) | 2.70 | 2.70 | 0.00 | 95.00% |
| Group J (Common Knowledge Definitions) | 0.00 | 0.00 | 0.00 | 0.00% |

- **Strong-Lead Precision (Group B vs Group I)**: **55.74%**

#### Ordering Metric (ROC AUC) on Paraphrase vs. Same-Topic Spans

| Ordering Metric | ROC AUC | Finding |
| :--- | :---: | :--- |
| **Cosine Alone** | **0.5340** | Near-random (~0.50); cosine overlaps heavily between paraphrases and same-topic text |
| **Anchor Count Alone** | **0.3622** | Inverted discrimination; technical vocabulary is dense in same-topic original writing |
| **Cosine + Anchors Composite** | **0.4939** | No separation; anchors do not cleanly disambiguate same-topic domains |

#### Lexical Anchor Rule Grid Search ($N \in \{1,2,3,4\}$, ratio $\in \{0.1, 0.2, 0.3\}$)

| Min Anchors ($N$) | Min Content Ratio | Group B Recall (All) | Group B Recall (Easy) | Group B Recall (Hard) | Group I FPR | Group J FPR | Rule Status |
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

> **Honest Technical Conclusion**: Because no parameter combination achieved Group B recall $\ge 70\%$ while reducing Group I FPR by half (to $\le 47.5\%$), lexical anchors cannot reliably separate same-topic technical writing from true paraphrases using this model (`all-MiniLM-L6-v2`) on this synthetic corpus. Semantic matching is kept strictly out of the default headline score. Matches meeting anchor criteria are tagged as 'strong lead' and others as 'weak lead' for human review only.

- **Mean Similarity Detected by Cohort**:
  - Group A (Verbatim / Lightly Edited Copies): Exact/Headline **98.28%** | Opt-In Semantic **98.28%**
  - Group B (Easy Paraphrased Passages): Exact/Headline **0.85%** | Opt-In Semantic **68.44%**
  - Group B (Hard Complex Paraphrases): Exact/Headline **0.00%** | Opt-In Semantic **43.74%**
  - Group C (Unrelated Human Originals): Exact/Headline **0.00%** | Opt-In Semantic **0.00%**
  - Group D (AI-Generated Text): Exact/Headline **0.00%** | Opt-In Semantic **0.00%**
  - Group E (Non-Native English): Exact/Headline **0.00%** | Opt-In Semantic **0.00%**
  - Group I (Same-Topic Originals): Exact/Headline **0.51%** | Opt-In Semantic **65.85%**
  - Group J (Common Knowledge Definitions): Exact/Headline **0.00%** | Opt-In Semantic **0.00%**

### How to Read a Veritext Report

1. **Headline Similarity Percentage**: Computed strictly from exact and near-copy matches using winnowing token fingerprints. Labeled `"exact & near-copy matches only"`. Exhibits 0.00% false-positive rate across negative cohorts at standard thresholds ($\ge 10\%$).
2. **Paraphrase Leads (Review)**: Shown separately as `"Paraphrase leads (review): N passages, M% of text"`. These highlight semantic vector proximity without contaminating the objective headline metric.
3. **Explanatory & Interpretation Guidance**:
   - Explanatory note: *"Similar meaning is not proof of copying; passages on the same topic often match. Check the linked source."*
   - Lead categories: Strong lead (several matching sentences and shared specific terms/anchors) vs. Weak lead (similar topic only).
4. **Grouped by Source**: Leads are aggregated per candidate document with lead counts. Multiple concentrated leads against a single source suggest substantive derivation, whereas scattered single leads indicate normal disciplinary topic overlap.
5. **Opt-in Legacy Mode**: When invoked with `--include-semantic-in-score`, the score includes semantic leads and is explicitly labeled `"score includes unverified semantic leads"`.

### AI Detection Ablation

| Model / Configuration | Accuracy | Precision | Recall | F1 Score | Overall Human FPR | Non-Native (ESL) FPR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stylometry Only (Lite Mode)** | 77.50% | 100.00% | 40.00% | 57.14% | 0.00% | 0.00% |
| DistilGPT2 Features Only | 62.50% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| **Full Ensemble (Stylometry + DistilGPT2)** | **95.00%** | **100.00%** | **86.67%** | **92.86%** | **0.00%** | **0.00%** |

- **ESL Protection**: False positive rate on non-native English writing (Group E) is strictly **0.00%** across all ablation tiers.
- **Ensemble Synergy**: DistilGPT2 alone fails to classify short text (0.00% recall), but when combined with stylometry, recall increases from 40.00% to 86.67% and accuracy reaches 95.00%.

### Forensic Integrity Checks & Seals

| Module | Metric Tested | Result | Standard / Baseline |
| :--- | :--- | :---: | :---: |
| **Spinner Forensics** | Precision / Recall | **100.00%** / **100.00%** | Cabanac et al. (2021) |
| **Batch Collusion Matrix** | Expected Clusters Discovered | **YES** | Pairwise Winnowing |
| **Fingerprint Index Screening** | True Pairs Candidate Recall | **100.00%** | Exact Inverted Index (Default) |
| **MinHash / LSH Screening** | True Pairs Candidate Recall | **100.00%** | Adaptive (64x2) Carter-Wegman 128-perm |
| **Cryptographic Report Seals** | Tamper Detection Rate | **100.00%** | HMAC-SHA256 (1-char edit) |

---

## 3. References Cited

1. **Schleimer, S., Wilkerson, D. S., & Aiken, A. (2003)**. *Winnowing: local algorithms for document fingerprinting*. In Proceedings of the 2003 ACM SIGMOD International Conference on Management of Data (pp. 76–85).
2. **Cabanac, G., Labbe, C., & Magazinov, A. (2021)**. *Tortured phrases: A dubious writing style emerging in science*. arXiv preprint arXiv:2107.06751.
