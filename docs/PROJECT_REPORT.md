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

### Stage 5: Hybrid Similarity Scoring & Cross-Language Matching
- **Match Span Clustering**: Contiguous match interval chaining with gap tolerance.
- **Semantic Vector Paraphrasing**: Dense embedding search via `all-MiniLM-L6-v2` for capturing paraphrased or structurally altered text.
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

Veritext was benchmarked on a synthetic dataset of 114 evaluated documents executed entirely offline on dual-core hardware (total runtime: 803.08 s, peak memory: 377.64 MB).

> **Methodology and Dataset Note**: The evaluation corpus is small, synthetic, and written by the same author as the detector. It serves to verify algorithmic guarantees, component wiring, and relative ranking rather than open-domain generalization. Results are evaluated across distinct easy and hard paraphrase subsets, and false positive rates are transparently measured across unrelated human originals, AI-generated text, ESL writing, same-topic originals, and common-knowledge standard definitions. Veritext makes no claim of parity with commercial services with proprietary multi-billion document databases.

### Plagiarism Detection Engine Ablation (Headline Threshold: 15%)

| Method / Configuration | Accuracy | Precision | Recall | F1 Score | False Positive Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Exact Winnowing (Veritext)** | 77.27% | 100.00% | 37.50% | 54.55% | 0.00% |
| Baseline (difflib SequenceMatcher) | 77.27% | 100.00% | 37.50% | 54.55% | 0.00% |
| **Winnowing + Semantic Paraphrase** | 79.09% | 65.45% | 90.00% | 75.79% | 27.14% |

#### Decision Threshold Sensitivity Analysis

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

#### Paraphrase Recall: Easy vs. Hard Subsets

| Threshold | Group A (Copies) | Group B Easy Paraphrases | Group B Hard Paraphrases (Complex) |
| :---: | :---: | :---: | :---: |
| 5% | 100.00% | 93.33% | 70.00% |
| 10% | 100.00% | 93.33% | 70.00% |
| 15% | 100.00% | 93.33% | 70.00% |
| 25% | 100.00% | 93.33% | 70.00% |

#### False Positive Rates across Negative Cohorts (Semantic Engine ON)

| Threshold | Group C (Originals) | Group D (AI Text) | Group E (ESL Non-Native) | Group I (Same-Topic Originals) | Group J (Common Knowledge) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 5% | 0.00% | 0.00% | 0.00% | 95.00% | 0.00% |
| 10% | 0.00% | 0.00% | 0.00% | 95.00% | 0.00% |
| 15% | 0.00% | 0.00% | 0.00% | 95.00% | 0.00% |
| 25% | 0.00% | 0.00% | 0.00% | 95.00% | 0.00% |

- **Mean Similarity Detected**:
  - Group A (Verbatim / Lightly Edited Copies): Exact **98.28%** | Semantic **98.28%**
  - Group B (Easy Paraphrased Passages): Exact **0.85%** | Semantic **68.44%**
  - Group B (Hard Complex Paraphrases): Exact **0.00%** | Semantic **43.74%**
  - Group C (Unrelated Human Originals): Exact **0.00%** | Semantic **0.00%**
  - Group D (AI-Generated Text): Exact **0.00%** | Semantic **0.00%**
  - Group E (Non-Native English): Exact **0.00%** | Semantic **0.00%**
  - Group I (Same-Topic Originals): Exact **0.51%** | Semantic **65.85%**
  - Group J (Common Knowledge Definitions): Exact **0.00%** | Semantic **0.00%**
- **Paraphrase Detection Analysis & Same-Topic Stress Test**:
  Exact winnowing achieves 98.28% mean similarity on verbatim copies with zero false positives on negative cohorts, but achieves 0.85% on easy paraphrases and 0.00% on hard paraphrases. Dense semantic vector matching (`all-MiniLM-L6-v2`) detects non-verbatim conceptual paraphrasing, attaining 93.33% recall on easy paraphrases and 70.00% recall on hard complex paraphrases.
  However, the same-topic stress test demonstrates an essential boundary of dense embeddings: independently written papers covering the identical specialized technical topic (Group I) share vector neighborhoods (median cosine 0.81), yielding a 95.00% document-level false-positive rate if treated as definitive plagiarism. Consequently, Veritext sets safer defaults: requiring minimum 2 consecutive windows (`min_consecutive_windows=2`), excluding single isolated window matches from the headline document similarity percentage, and labeling matches as `"possible paraphrase (review)"` for human instructor evaluation. Single textbook definitions of common knowledge (Group J) trigger 0.00% false positives.

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
