# Veritext Final Verification & Audit Report

**Date of Audit**: October 23, 2026  
**Auditor**: Antigravity Automated Verification Agent  
**Environment**: Intel Core i3-7020U CPU @ 2.30GHz (2 physical cores), 8 GB RAM, no GPU, running under WSL (Ubuntu Linux)  
**Python Runtime**: Python 3.14.4 virtual environment (`veritext/.venv`)  
**Commit Audited**: `2668460` (Tagged `v1.0.0`, up to date with `origin/main`); Updated for `v1.0.1`  

---

## 1. Executive Summary

A comprehensive, stage-by-stage audit and benchmark verification was conducted on **Veritext**. Every stage deliverable from Stage 1 through Stage 7 (including the Adaptive Analysis pack and change packs 6B, 6C, and 6D) was tested directly against demo files and verified with automated test suites.

- **Checklist Results**: **52 / 52 Items PASS (100%)**, 0 FAIL, 0 PARTIAL.
- **Automated Test Suite**: **244 / 244 Tests Passing** (237 fast tests in 135.71s; 7 slow integration tests in 475.2s).
- **Code Linter**: `ruff check .` reported **All checks passed!** with 0 errors.
- **Lazy Import Verification**: Confirmed via `sys.modules` that importing `veritext` and executing document integrity checks does **not** load `torch` or `transformers`.
- **Repo & Security Hygiene**: Zero secrets or credentials detected across tracked files and the last 30 commits. `.gitignore` strictly covers all temporary, cache, database, and credential artifacts.
- **Documentation Parity**: Synchronized API default port (8600), fixed CLI subcommands (`api serve`), updated benchmark ablation tables, and refined `DEMO_SCRIPT.md` to a 5-minute viva-ready script.

---

## 2. Stage-by-Stage Verification Checklist

| Stage | Deliverable / Capability | Status | Evidence & Command Output |
| :--- | :--- | :---: | :--- |
| **Stage 1** | Loaders for `.docx`, `.pdf`, `.txt`, `.md`, `.tex` | **PASS** | `tests/test_loaders.py` (6 passed). All 5 sample files ingested successfully via `python -m veritext.ingest`. |
| **Stage 1** | Academic text cleaner removes references & quotes | **PASS** | `tests/test_cleaner.py` (6 passed). Stripped 218 chars of references and 92 chars of quotes from `sample_academic.docx`. |
| **Stage 1** | Sentence character offsets exact | **PASS** | `tests/test_splitter.py`, `tests/test_pipeline.py` (13 passed). Character slices in original text match `Sentence.start:end`. |
| **Stage 1** | Ingestion CLI functions on all 5 sample files | **PASS** | `python -m veritext.ingest data/samples/sample_academic.{docx,pdf,txt,md,tex}` all returned exit status 0 with parsed structure summaries. |
| **Stage 2** | Source clients parse API fixtures | **PASS** | `tests/test_sources_clients.py` (8 passed). arXiv, CrossRef, Europe PMC, and OpenAlex fixture payloads parsed accurately. |
| **Stage 2** | SQLite store and FTS5 full-text indexing | **PASS** | `tests/test_store_and_library.py` (3 passed). Reference library in `data/veritext.sqlite` indexes 8 documents and 36 passages. |
| **Stage 2** | `sources` CLI (add, search, stats) offline execution | **PASS** | `veritext sources stats` reported 8 indexed sources; `veritext sources search --offline data/samples/sample_academic.txt` returned 6 ranked candidates using BM25. |
| **Stage 2** | HTTP caching and API rate limiting | **PASS** | `tests/test_http_and_cache.py` (5 passed). `data/cache/http_cache.sqlite` holds 78 cached requests; sliding window rate limiter prevents 429 errors. |
| **Stage 3** | Text normalisation (homoglyphs, zero-width, offset map) | **PASS** | `tests/test_normalize.py` (7 passed). Cyrillic/Greek homoglyphs canonicalized, zero-width spaces stripped, bidirectional tokens removed. |
| **Stage 3** | Mathematical winnowing guarantee | **PASS** | `tests/test_fingerprint.py` (5 passed). Parameters $k=5, w=4$; guarantee holds for any shared token run $\ge 8$ words. |
| **Stage 3** | Exact, near-copy and dense semantic matching | **PASS** | `tests/test_similarity.py`, `tests/test_semantic.py` (11 passed). Hybrid scoring accurately unions overlapping match intervals. *(Note: The initial benchmark ablation discrepancy was confirmed as a benchmark-harness bug—empty sentence lists—not a product bug; see Section 4.1 & 6).* |
| **Stage 3** | Similarity CLI differential validation on demo corpus | **PASS** | `veritext check --offline`: `copied_doc.txt` = **85.5%** (HIGH); `paraphrased_doc.txt` = **33.1%** (MEDIUM, 3 semantic spans); `original_doc.txt` = **0.0%** (NEAR ZERO). |
| **Stage 4** | AI detection (DistilGPT2 + stylometry ensemble) | **PASS** | `tests/test_ai_detect.py` (14 passed). Validated on full neural ensemble. |
| **Stage 4** | Stylometry-only Lite Mode fallback | **PASS** | `tests/test_ai_detect.py`. Fallback operates CPU-only without neural models. |
| **Stage 4** | Sentence ranking and character offsets | **PASS** | Highlighted sentence offsets and plain-language driver notes generated with ranked confidence scores. |
| **Stage 4** | Insufficient-text rule (< 150 words) | **PASS** | `veritext ai data/samples/plagiarism_demo/copied_doc.txt` (104 words) produced: *"Result: INSUFFICIENT TEXT (104 words). Statistical AI detection requires at least 150 words"*. |
| **Stage 4** | Calibration CLI | **PASS** | `veritext ai --calibrate data/calibration/starter.csv` completed with 1.000 accuracy and updated `data/config/ai_weights.json`. |
| **Stage 4** | Benchmark against all six `ai_demo` files | **PASS** | `ai_sample_1.txt`: **85.0%**; `ai_sample_2.txt`: **88.8%**; `ai_sample_3.txt`: **78.0%**; `human_sample_1.txt`: **6.3%**; `human_sample_2.txt`: **1.8%**; `human_sample_3_esl.txt`: **2.0%** (0% False Positives on ESL). |
| **Stage 5** | Hidden content & adversarial obfuscation detection | **PASS** | `veritext checks check data/samples/checks_demo/hidden_content_demo.docx` detected white font and <2pt font; `homoglyphs_zero_width_demo.txt` flagged mixed-script homoglyphs & zero-width characters. |
| **Stage 5** | Citation & bibliography consistency validation | **PASS** | `veritext checks check data/samples/checks_demo/citations_mismatch_demo.txt` flagged uncited empirical assertions, mixed numeric/author-year styles, and missing references. |
| **Stage 5** | Self-plagiarism verification against prior submissions | **PASS** | `veritext checks add-previous` + `veritext checks check data/samples/checks_demo/self_plag_new.txt` detected **52.85%** overlap with registered prior submission `self_plag_previous.txt`. |
| **Stage 5** | Source code similarity with boilerplate template exclusion | **PASS** | `veritext checks code data/samples/checks_demo/code_demo --template ...` identified **100.0%** similarity between `original.py` and `renamed_copy.py` while ignoring boilerplate lines. |
| **Stage 5** | Checks CLI on every demo in `checks_demo` | **PASS** | `tests/test_checks.py` (14 passed). All demo documents triggered appropriate warning/severity statuses. |
| **Stage 6** | JSON, HTML, and PDF consolidated reports | **PASS** | `tests/test_report.py` (13 passed). Self-contained HTML exports, robust UTF-8/Latin-1 PDF formatting, and JSON schemas generated. |
| **Stage 6** | Streamlit app headless startup & HTTP 200 response | **PASS** | `test_report.py::test_streamlit_app_smoke` passed. Headless server launched on port 8502; responded HTTP 200 on health and main routes within 21.07s. |
| **Stage 6** | Hostile HTML script injection escaping | **PASS** | `test_report.py::test_html_escaping_hostile_document` passed. All raw `<script>` tags and angular characters sanitized. |
| **Adaptive** | Document profiler and size categorization | **PASS** | `src/veritext/profile.py`, `tests/test_adaptive_analysis.py`. Word counts, token estimates, and language heuristics computed. |
| **Adaptive** | Per-engine execution planning & time budgeting | **PASS** | `src/veritext/planner.py`. Dynamically scales word budgets based on available hardware and user-specified limits. |
| **Adaptive** | Hardware timing calibration (`timing.json`) | **PASS** | `data/config/timing.json` calibrated to dual-core execution speeds; used by planner for accurate time forecasting. |
| **Adaptive** | `--plan-only` CLI execution mode | **PASS** | `veritext check --plan-only` and `veritext report --plan-only` print profile and plan table, exiting cleanly without engine execution. |
| **Adaptive** | Chunked processing for AI detection | **PASS** | Documents segmented into bounded token windows, preventing OOM spikes on long texts. |
| **Adaptive** | Block sampling including first and last sections | **PASS** | Verified in `tests/test_adaptive_analysis.py`. Sampling always preserves opening and concluding paragraphs. |
| **Adaptive** | No hardcoded 5,000-word caps | **PASS** | `git grep` confirmed no hardcoded 5,000-word limit; word budgets are determined dynamically via document profiling. |
| **Adaptive** | Hidden check false-positive reduction | **PASS** | `tests/test_hidden_false_positives.py` (8 passed). Soft hyphens, legal typography, and standard font boundaries exempted. |
| **Stage 6B** | Full-text retrieval security (HTTPS only, private IP refusal, size cap) | **PASS** | `tests/test_fulltext.py` (14 passed). Private RFC-1918 IPs, loopback addresses, non-HTTPS protocols, and files >20 MB strictly blocked. |
| **Stage 6B** | Spinner & tortured-phrase forensics | **PASS** | `tests/test_spinner.py` (5 passed). 100% precision and recall on predatory journal tortured-phrase catalog. |
| **Stage 6B** | Institutional class repository (fingerprints-only & anonymized IDs) | **PASS** | `tests/test_classrepo.py` (4 passed). `veritext classrepo add --fingerprints-only` discarded raw text, storing only winnowing hashes under anonymized ID `sub_f79973a9`. |
| **Stage 6B** | Pairwise batch collusion heatmap & clustering | **PASS** | `tests/test_batch.py` (1 passed). Batch run detected 2 distinct collusion clusters across 8 submissions and generated self-contained HTML heatmap. |
| **Stage 6B** | Live interactive filter recalculation without engine re-runs | **PASS** | `tests/test_filters.py` (3 passed). Real-time score recomputation upon excluding quotes, bibliographies, small matches, or allowlisted sources. |
| **Stage 6C** | Review workflow, rubric grading, and feedback exports | **PASS** | `tests/test_review.py` (4 passed). Rubrics loaded from `data/config/rubric_default.json`; student view sanitizes adversarial terminology. |
| **Stage 6C** | Academic style and grammar linter | **PASS** | `tests/test_style.py` (8 passed). Passive voice, readability indices (Flesch-Kincaid), and filler phrases flagged with line offsets. |
| **Stage 6C** | Dashboard heat strip and submission portfolio table | **PASS** | `tests/test_dashboard.py` (3 passed). Longitudinal risk heatmaps and cohort portfolio views verified. |
| **Stage 6C** | Local REST API (key auth, size/type limits, single background worker, SSE) | **PASS** | `tests/test_api.py` (13 passed). 401 on missing/invalid key; 413 on files >15 MB; single worker queue prevents CPU thrashing; SSE streams stage progress. |
| **Stage 6C** | Honest scope boundaries in `FEATURES.md` | **PASS** | `docs/FEATURES.md`. Explicitly states no commercial parity claims, details closed-source database boundaries, and documents compute trade-offs. |
| **Stage 6D** | Side-by-side excerpt diff visualizer | **PASS** | `tests/test_diff.py` (5 passed). Token-level diff generation with contextual matching. |
| **Stage 6D** | Cryptographic report sealing & verification | **PASS** | `tests/test_seal.py` (3 passed). Canonical JSON + HMAC-SHA256. Valid report returned `VALID`; single-character modification returned `TAMPERED`. |
| **Stage 6D** | Cross-language matching graceful degradation | **PASS** | `tests/test_crosslang.py` (7 passed). Gracefully reports disabled state without crashing when multilingual model weights are not downloaded. |
| **Stage 6D** | Vectorized Carter-Wegman MinHash & LSH pre-screening | **PASS** | `tests/test_minhash.py` (8 passed). Exact fingerprint inverted index default (100% recall); adaptive 128-perm Carter-Wegman LSH option with dynamically derived $(b, r)$ ($s^* \approx t/2$). |
| **Stage 6D** | Non-accusatory authorship timeline terminology | **PASS** | `tests/test_authorship.py` (4 passed). Findings use neutral pedagogical terminology (*"timeline milestone"*, *"revisions"*) with explicit disclaimers. |
| **Stage 6D** | CORE API optional configuration | **PASS** | `tests/test_core.py` (3 passed). Safely bypasses CORE API when API key is unconfigured. |
| **Stage 6D** | Live Server-Sent Events (SSE) progress streaming | **PASS** | `tests/test_api.py`. Progress percentages and stage events stream to connected clients via `/progress/{job_id}`. |
| **Stage 7** | Lazy module imports (no PyTorch imported during checks) | **PASS** | Verified via `sys.modules`: importing `veritext` packages and running `run_all_checks()` executes with `torch` and `transformers` completely unloaded. |
| **Stage 7** | Evaluation benchmark dataset & automated runner | **PASS** | `tests/test_eval.py` (6 passed). 114-document synthetic dataset evaluated via `veritext eval run`. Runner verified to populate sentence structures, index reference store, evaluate easy vs. hard paraphrases, and track negative cohort FPR. |
| **Stage 7** | Report polish phrasing & single disclaimer rule | **PASS** | `tests/test_report.py`. Displays *"Insufficient text (under 150 words, not scored)"*; enforces exactly one consolidated disclaimer block across exports. |
| **Stage 7** | Packaging, console script & optional extras | **PASS** | `pyproject.toml` defines console entrypoint `veritext`, extras (`lite`, `full`, `dev`). `veritext --help` operational. |
| **Stage 7** | Dockerfile, deployment guide & citation specifications | **PASS** | `Dockerfile`, `DEPLOY.md`, `CITATION.cff` verified and aligned with production defaults. |
| **Stage 7** | CI workflow & Ruff linting | **PASS** | `.github/workflows/tests.yml` configured; `ruff check .` passes with zero violations. |

---

## 3. End-to-End Execution Measurements

All operations were executed on the test machine (Intel Core i3-7020U, 2 cores, 8 GB RAM, WSL / Ubuntu Linux) inside a temporary directory (`veritext/audit_tmp/`, gitignored and deleted upon completion):

| Pipeline / Task | Input Target | Elapsed Time (Wall Clock) | Peak Memory (RSS) | Primary Output / Result |
| :--- | :--- | :---: | :---: | :--- |
| **Full Report Pipeline** (`--ai --semantic --offline`) | `data/samples/ai_demo/ai_sample_1.txt` (173 words) | 5m 39.20s | 1,222.8 MB | Generated JSON, HTML, and PDF reports; similarity: 0.0%, AI probability: 85.3%. |
| **Batch Collusion Matrix** | 8 submissions in `collusion_demo/` | 36.88s | 396.4 MB | Detected 2 collusion pairs (73.6% and 65.2%); generated CSV matrix & HTML heatmap. |
| **Class Repository Workflow** | `sample_academic.txt` (fingerprints-only mode) | 43.45s (add/list)<br>3.44s (delete) | 396.8 MB (add)<br>34.4 MB (delete) | Anonymized ID `sub_f79973a9` registered with 62 hashes; raw text discarded; deleted cleanly. |
| **REST API Lifecycle** | `127.0.0.1:8600`, key auth, job submit & poll | 54.61s | 456.3 MB | Key generated, rejected 401 on bad key, rejected 413 on 16 MB file, completed job, fetched report, shut down cleanly. |
| **Report Seal & Tamper Flow** | JSON report (`verify` subcommand) | 21.97s | 83.5 MB | Untouched report returned `VALID`; single-character modification returned `TAMPERED`. |
| **Headless Streamlit Application** | `src/veritext/ui/app.py` on port 8502 | 22.19s | 60.8 MB | Health check and main page returned HTTP 200 after 21.07s; process stopped cleanly. |

---

## 4. Full Benchmark Evaluation Results

The complete evaluation benchmark was executed with:
```bash
veritext eval run --semantic --ai
```
*(Note: `--crosslang` was omitted because multilingual weights are not locally installed, per instructions).*

- **Evaluated Documents**: 114 documents (from synthetic corpus)
- **Total Runtime**: 1911.1 s (~16.76 s per document)
- **Peak Memory**: 389.19 MB

> **Corpus & Methodology Note**: The evaluation corpus is synthetic, compact, and created by the system author. In accordance with audit guidelines, results are evaluated across distinct easy and hard paraphrase subsets, and false positive rates are transparently measured across unrelated human originals, AI-generated text, and ESL writing.

### 4.1 Plagiarism Detection Engine Ablation

| Method / Configuration | Accuracy | Precision | Recall | F1 Score | False Positive Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Exact Winnowing (Veritext)** | 68.75% | 100.00% | 37.50% | 54.55% | 0.00% |
| Baseline (difflib SequenceMatcher) | 68.75% | 100.00% | 37.50% | 54.55% | 0.00% |
| **Winnowing + Semantic Paraphrase** | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **0.00%** |

#### Decision Threshold Sensitivity Analysis

| Threshold | Method | Accuracy | Precision | Recall | F1 Score | Overall FPR |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 5% | Exact Winnowing | 70.00% | 100.00% | 40.00% | 57.14% | 0.00% |
| 5% | **Winnowing + Semantic** | 100.00% | 100.00% | 100.00% | 100.00% | 0.00% |
| 10% | Exact Winnowing | 70.00% | 100.00% | 40.00% | 57.14% | 0.00% |
| 10% | **Winnowing + Semantic** | 100.00% | 100.00% | 100.00% | 100.00% | 0.00% |
| 15% | Exact Winnowing | 68.75% | 100.00% | 37.50% | 54.55% | 0.00% |
| 15% | **Winnowing + Semantic** | 100.00% | 100.00% | 100.00% | 100.00% | 0.00% |
| 25% | Exact Winnowing | 68.75% | 100.00% | 37.50% | 54.55% | 0.00% |
| 25% | **Winnowing + Semantic** | 100.00% | 100.00% | 100.00% | 100.00% | 0.00% |

#### Paraphrase Recall: Easy vs. Hard Subsets

| Threshold | Group A (Copies) | Group B Easy Paraphrases | Group B Hard Paraphrases (Complex) |
| :---: | :---: | :---: | :---: |
| 5% | 100.00% | 100.00% | 100.00% |
| 10% | 100.00% | 100.00% | 100.00% |
| 15% | 100.00% | 100.00% | 100.00% |
| 25% | 100.00% | 100.00% | 100.00% |

#### False Positive Rates across Negative Cohorts (Semantic Engine ON)

| Threshold | Group C (Originals) | Group D (AI Text) | Group E (ESL Non-Native) |
| :---: | :---: | :---: | :---: |
| 5% | 0.00% | 0.00% | 0.00% |
| 10% | 0.00% | 0.00% | 0.00% |
| 15% | 0.00% | 0.00% | 0.00% |
| 25% | 0.00% | 0.00% | 0.00% |

- **Mean Similarity Detected**:
  - Group A (Verbatim / Lightly Edited Copies): Exact **98.28%** | Semantic **98.28%**
  - Group B (Easy Paraphrased Passages): Exact **0.85%** | Semantic **90.96%** (4.27 spans/doc, mean cosine 0.81)
  - Group B (Hard Complex Paraphrases): Exact **0.00%** | Semantic **83.94%** (3.90 spans/doc, mean cosine 0.77)
  - Group C (Unrelated Human Originals): Exact **0.00%** | Semantic **0.00%** (0 false positive spans)
  - Group D (AI-Generated Text): Exact **0.00%** | Semantic **0.00%** (0 false positive spans)
  - Group E (Non-Native English): Exact **0.00%** | Semantic **0.00%** (0 false positive spans)

- **Benchmark-Harness Bug Root Cause & Findings**:
  The initial 0.85% semantic benchmark result was investigated and confirmed to be a **benchmark-harness bug** (`src/veritext/eval/runner.py`), **not a product bug**. Specifically:
  1. **Empty Sentence Structures**: The benchmark runner's `_create_document()` constructed in-memory `Document` instances with empty `sentences=[]` and `paragraphs=[]` lists. Because semantic sliding windows require populated `Sentence` structures, no candidate windows were extracted during benchmark evaluation.
  2. **Transient Library Indexing**: The runner queried the SQLite store without persistently indexing reference passages from `data/eval/references/`.
  3. **Score Reporting Key**: The runner's ablation aggregator recorded the winnowing score rather than the semantic union score in `scores_by_group`.
  
  Once `_create_document()` was fixed to invoke `split_paragraphs()` and `split_sentences()` and reference indexing was ensured, semantic similarity jumped to **90.96%** on easy paraphrases and **83.94%** on the newly created hard paraphrase corpus (`para_hard_01` to `para_hard_10`), with **100% recall** across all evaluated thresholds (5% to 25%) and **0.00% false positive rates** across all negative cohorts.

### 4.2 AI Detection Ablation

| Model / Configuration | Accuracy | Precision | Recall | F1 Score | Overall Human FPR | Non-Native (ESL) FPR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stylometry Only (Lite Mode)** | 77.50% | 100.00% | 40.00% | 57.14% | 0.00% | 0.00% |
| DistilGPT2 Features Only | 62.50% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| **Full Ensemble (Stylometry + DistilGPT2)** | **95.00%** | **100.00%** | **86.67%** | **92.86%** | **0.00%** | **0.00%** |

- **Non-Native (ESL) Safeguards**: The false positive rate on non-native English essays (Group E) is **0.00%** across all ablation tiers, confirming conservative calibration.
- **Ensemble Synergy**: DistilGPT2 perplexity features alone fail to classify short documents (0.00% recall). Combining DistilGPT2 perplexity with stylometric markers increases recall from 40.00% to 86.67% and overall accuracy to 95.00% (2 false negatives out of 15 AI documents).

### 4.3 Integrity Checks, Collusion & Cryptographic Seals

| Module | Metric Tested | Result | Standard / Baseline |
| :--- | :--- | :---: | :---: |
| **Spinner / Tortured Phrases** | Precision / Recall | **100.00%** / **100.00%** | Cabanac et al. (2021) |
| **Batch Collusion Matrix** | Expected Clusters Discovered | **YES** | Pairwise Winnowing |
| **Fingerprint Index Screening** | True Pairs Candidate Recall | **100.00%** | Exact Inverted Index (Default) |
| **MinHash / LSH Screening** | True Pairs Candidate Recall | **100.00%** | Adaptive (64x2) Carter-Wegman 128-perm |
| **Cryptographic Report Seals** | Tamper Detection Rate | **100.00%** | HMAC-SHA256 (1-char edit) |

---

## 5. Repository & Security Hygiene

1. **Tracked Files Cleanliness**:
   `git ls-files` was scanned for uploads, reports, reviews, class repository records, API keys, signing keys, caches, SQLite databases, and neural model binaries. **Zero sensitive or runtime files are tracked in git.**
2. **Ten Largest Tracked Files**:
   - `src/veritext/eval/dataset_builder.py` (106 KB)
   - `data/eval/results/confusion_matrices.png` (51 KB)
   - `src/veritext/ui/app.py` (45 KB)
   - `data/samples/sample_academic.docx` (37 KB)
   - `data/samples/checks_demo/hidden_content_demo.docx` (37 KB)
   - `src/veritext/eval/runner.py` (34 KB)
   - `src/veritext/checks/hidden.py` (31 KB)
   - `src/veritext/checks/citations.py` (25 KB)
   - `src/veritext/report/html_export.py` (23 KB)
   - `src/veritext/report/pdf_export.py` (20 KB)
3. **`.gitignore` Completeness**: Verified coverage for `data/uploads`, `data/reports`, `data/reviews`, `data/class_repository`, `data/cache`, `data/config/api_keys.json`, `data/config/report_key*`, `data/api_jobs.sqlite`, `models`, and `audit_tmp/`.
4. **Credential & Secret Scan**:
   Tracked files and the last 30 commits were scanned for patterns matching GitHub tokens (`ghp_`), OpenAI keys (`sk-`), AWS access keys (`AKIA`), private keys (`BEGIN PRIVATE KEY`), and hex secrets. **Zero secrets or tokens were found.**

---

## 6. Documentation Fixes Applied

During the audit, the following discrepancies were identified and fixed directly:

1. **API Port Consistency (`README.md`, `DEPLOY.md`)**:
   - *Issue*: `README.md` and `DEPLOY.md` instructed users to run `veritext api --port 8000`, which failed because `veritext api` requires the `serve` subcommand and the code defaults to port 8600.
   - *Fix*: Updated documentation across all files to `veritext api serve --host 127.0.0.1 --port 8600`.
2. **Class Repository Path Discrepancy (`DEPLOY.md`)**:
   - *Issue*: Section 6 referenced `data/class_repos/` instead of `data/class_repository/`.
   - *Fix*: Corrected path to `data/class_repository/`.
3. **Reference Citation Constraints (`docs/FEATURES.md`)**:
   - *Issue*: A passing mention of Kirchenbauer et al. (2023) was present in a discussion of watermarking limitations.
   - *Fix*: Removed the external citation to adhere strictly to the project rule that only Schleimer et al. (2003) and Cabanac et al. (2021) are cited.
4. **Missing Benchmark Tables (`docs/PROJECT_REPORT.md`, `README.md`)**:
   - *Issue*: `PROJECT_REPORT.md` lacked the benchmark tables; `README.md` only had exact-only numbers.
   - *Fix*: Embedded the complete benchmark ablation tables (Exact Winnowing, SequenceMatcher, Semantic Paraphrase, Stylometry, DistilGPT2, and Full Ensemble) in both files.
5. **5-Minute Viva Demonstration Script (`docs/DEMO_SCRIPT.md`)**:
   - *Issue*: `DEMO_SCRIPT.md` was a 10-minute outline lacking explicit viva technical defense prompts.
   - *Fix*: Shortened to a crisp 5-minute timed walkthrough with exact CLI commands and 5 technical viva questions/answers.
6. **Temporary Audit Directory in `.gitignore`**:
   - *Issue*: `audit_tmp/` was not explicitly listed in `.gitignore`.
   - *Fix*: Added `audit_tmp/` to `.gitignore`.
7. **Benchmark Harness Bug Fix & LSH Shortlist Upgrade (v1.0.1)**:
   - *Issue 1 (Benchmark Harness)*: The evaluation runner constructed synthetic `Document` objects with empty `sentences` lists and failed to persistently index reference documents into the SQLite candidate library before querying, resulting in artificially suppressed semantic paraphrase metrics (0.85%).
   - *Fix 1*: Updated `src/veritext/eval/runner.py` to segment text using `split_paragraphs()` and `split_sentences()`, persistently index reference texts, record semantic scores, evaluate against 10 newly created hard complex paraphrases (`para_hard_01` to `para_hard_10`), and track false-positive rates on negative cohorts across thresholds 5%, 10%, 15%, 25%.
   - *Issue 2 (LSH Recall)*: Fixed $(b=32, r=4)$ LSH configuration had low recall (50%) on moderate overlap pairs.
   - *Fix 2*: Defaulted batch and class screening to an exact inverted index of shared winnowing fingerprints (100% recall, <60ms for 500 documents). Added `--method lsh` option with dynamically derived $(b, r)$ satisfying $b \times r \le 128$ and $(1/b)^{1/r} \approx t/2$, with an explicit user warning and small-batch ($N \le 150$) automatic bypass to exact pairwise winnowing.

---

## 7. Known Limitations & Recommended Next Steps

1. **Semantic Matching vs. Exact Matching Trade-Offs**:
   While dense neural vector search (`all-MiniLM-L6-v2`) achieves 100% recall on both easy (90.96% mean similarity) and hard (83.94% mean similarity) synthetic paraphrases, neural semantic similarity carries higher compute cost than exact winnowing. On negative cohorts (unrelated originals, AI text, ESL writing), the false positive rate is 0.00% under the current thresholding ($t \ge 5\%$).
2. **MinHash/LSH Probabilistic Screening**:
   MinHash/LSH screening is probabilistic; while the new adaptive $(b, r)$ derivation targets $s^* \approx t/2$ to maximize candidate recall, exact fingerprint inverted indexing remains the recommended and default screening mechanism for batches under 10,000 documents.
3. **Dual-Core Neural Inference Latency**:
   On dual-core CPUs without GPU acceleration, loading DistilGPT2 and running causal language modeling requires ~5-6 seconds per 200 words. For larger batches, Lite Mode (stylometry-only) remains the recommended default for responsive interactive workflows.
4. **Open-Access Scholar API Coverage**:
   Veritext queries public APIs (arXiv, CrossRef, OpenAlex, Europe PMC). It intentionally does not index paywalled publisher databases or proprietary institutional repositories.

---

## 8. Defect Summary & Release of v1.0.1

- **PASS Count**: 52 items (100%)
- **FAIL Count**: 0 items
- **PARTIAL Count**: 0 items
- **All Defects Fixed**:
  1. Semantic benchmark wiring fixed (benchmark-harness bug resolved; 100% recall on easy and hard paraphrase benchmarks).
  2. Batch/class screening defaulted to exact fingerprint inverted index (100% recall), with adaptive MinHash/LSH retained as an option.
  3. 10 hard complex paraphrases added to evaluation corpus.
  4. Multi-threshold sensitivity (5%, 10%, 15%, 25%) and negative cohort false-positive rates measured and documented.
- **Tag Status**: Tag `v1.0.0` remains preserved at commit `2668460`. Tag `v1.0.1` is released for the benchmark harness fix, hard paraphrase dataset addition, and LSH shortlist recall upgrade.
